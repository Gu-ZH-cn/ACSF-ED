import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import build_backbone_2d
from ..backbone import build_backbone_3d
from .encoder import build_channel_encoder
# from .head import build_head
from ..decoder.decoder import build_decoder
# from models.backbone.backbone_2d.cnn_2d.PResNet_HybridEncoder.HybridEncoder import build_HybridEncoder

from utils.nms import multiclass_nms


# Adapative Cross-Scale Fusion Encoder-Decoder 
class acsfed(nn.Module):
    def __init__(self, 
                 cfg,
                 device,
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 topk = 50,
                 trainable = False,
                 multi_hot = False):
        super(acsfed, self).__init__()
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.multi_hot = multi_hot

        # ------------------ Network ---------------------
        ## 2D backbone
        self.backbone_2d, bk_dim_2d = build_backbone_2d(
            cfg, pretrained=cfg['pretrained_2d'] and trainable)
            
        ## 3D backbone
        self.backbone_3d, bk_dim_3d = build_backbone_3d(
            cfg, pretrained=cfg['pretrained_3d'] and trainable)

        ## channel encoder
        self.channel_encoders = nn.ModuleList(
            [build_channel_encoder(cfg, bk_dim_2d[i]+bk_dim_3d, cfg['head_dim'])
                for i in range(len(cfg['stride']))])

        ## pred
        head_dim = cfg['head_dim']

        ## decoder
        self.decoder = build_decoder(cfg, self.num_classes)



    def init_acsfed(self): 
        # Init yolo
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
                
        # Init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # obj pred
        for conf_pred in self.conf_preds:
            b = conf_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            conf_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # cls pred
        for cls_pred in self.cls_preds:
            b = cls_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def generate_anchors(self, fmp_size, stride):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        anchor_xy *= stride
        anchors = anchor_xy.to(self.device)

        return anchors
        

    def decode_boxes(self, anchors, pred_reg, stride):
        """
            anchors:  (List[Tensor]) [1, M, 2] or [M, 2]
            pred_reg: (List[Tensor]) [B, M, 4] or [B, M, 4]
        """
        # center of bbox
        pred_ctr_xy = anchors + pred_reg[..., :2] * stride
        # size of bbox
        pred_box_wh = pred_reg[..., 2:].exp() * stride

        pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box


    def post_process_one_hot(self, conf_preds, cls_preds, reg_preds, anchors):
        """
        Input:
            conf_preds: (Tensor) [H x W, 1]
            cls_preds: (Tensor) [H x W, C]
            reg_preds: (Tensor) [H x W, 4]
        """
        
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for level, (conf_pred_i, cls_pred_i, reg_pred_i, anchors_i) in enumerate(zip(conf_preds, cls_preds, reg_preds, anchors)):
            # (H x W x C,)
            scores_i = (torch.sqrt(conf_pred_i.sigmoid() * cls_pred_i.sigmoid())).flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk, reg_pred_i.size(0))

            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = scores_i.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            labels = topk_idxs % self.num_classes

            reg_pred_i = reg_pred_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]

            # decode box: [M, 4]
            bboxes = self.decode_boxes(anchors_i, reg_pred_i, self.stride[level])

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, False)

        return scores, labels, bboxes
    

    def post_process_multi_hot(self, conf_preds, cls_preds, reg_preds, anchors):
        """
        Input:
            cls_pred: (Tensor) [H x W, C]
            reg_pred: (Tensor) [H x W, 4]
        """        
        all_conf_preds = []
        all_cls_preds = []
        all_box_preds = []
        for level, (conf_pred_i, cls_pred_i, reg_pred_i, anchors_i) in enumerate(zip(conf_preds, cls_preds, reg_preds, anchors)):
            # decode box
            box_pred_i = self.decode_boxes(anchors_i, reg_pred_i, self.stride[level])
            
            # conf pred 
            conf_pred_i = torch.sigmoid(conf_pred_i.squeeze(-1))   # [M,]

            # cls_pred
            cls_pred_i = torch.sigmoid(cls_pred_i)                 # [M, C]

            # topk
            topk_conf_pred_i, topk_inds = torch.topk(conf_pred_i, self.topk)
            topk_cls_pred_i = cls_pred_i[topk_inds]
            topk_box_pred_i = box_pred_i[topk_inds]

            # threshold
            keep = topk_conf_pred_i.gt(self.conf_thresh)
            topk_conf_pred_i = topk_conf_pred_i[keep]
            topk_cls_pred_i = topk_cls_pred_i[keep]
            topk_box_pred_i = topk_box_pred_i[keep]

            all_conf_preds.append(topk_conf_pred_i)
            all_cls_preds.append(topk_cls_pred_i)
            all_box_preds.append(topk_box_pred_i)

        # concatenate
        conf_preds = torch.cat(all_conf_preds, dim=0)  # [M,]
        cls_preds = torch.cat(all_cls_preds, dim=0)    # [M, C]
        box_preds = torch.cat(all_box_preds, dim=0)    # [M, 4]

        # to cpu
        scores = conf_preds.cpu().numpy()
        labels = cls_preds.cpu().numpy()
        bboxes = box_preds.cpu().numpy()

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, True)

        # [M, 5 + C]
        out_boxes = np.concatenate([bboxes, scores[..., None], labels], axis=-1)

        return out_boxes
    

    @torch.no_grad()
    def inference(self, video_clips):
        """
        Input:
            video_clips: (Tensor) -> [B, 3, T, H, W].
        return:
        """
        B, _, _, img_h, img_w = video_clips.shape

        # key frame
        key_frame = video_clips[:, :, -1, :, :]
        # 3D backbone
        feat_3d = self.backbone_3d(video_clips)

        # 2D backbone
        feats_2d = self.backbone_2d(key_frame)

        # channel encoder output
        channel_encoder_feats = []

        for level, feat_2d in enumerate(feats_2d):
            # upsample
            feat_3d_up = F.interpolate(feat_3d, scale_factor=2 ** (2 - level))

            # channel encoder
            channel_encoder_feat = self.channel_encoders[level](feat_2d, feat_3d_up)
            channel_encoder_feats.append(channel_encoder_feat)



        # decoder
        pred = self.decoder(channel_encoder_feats)

        return pred


    def forward(self, video_clips, targets=None):                     
        if not self.trainable:
            return self.inference(video_clips)
        else:
            # key frame
            key_frame = video_clips[:, :, -1, :, :]
            # 3D backbone
            feat_3d = self.backbone_3d(video_clips)

            # 2D backbone
            feats_2d = self.backbone_2d(key_frame)

            # channel encoder output
            channel_encoder_feats = []

            for level, feat_2d in enumerate(feats_2d):
                # upsample
                feat_3d_up = F.interpolate(feat_3d, scale_factor=2 ** (2 - level))

                # channel encoder
                channel_encoder_feat = self.channel_encoders[level](feat_2d, feat_3d_up)
                channel_encoder_feats.append(channel_encoder_feat)

            # decoder
            pred = self.decoder(channel_encoder_feats, targets)

            return pred
