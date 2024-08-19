# Model configuration


acsf_ed_config =
    'acsf_ed':{
        # backbone
        ## 2D
        'backbone_2d': 'PResNet',
        'pretrained_2d': False,
        ### PResNet
        'depth': 50,
        'variant': 'd',
        'freeze_at': 0,
        'return_idx': [1, 2, 3],
        'num_stages': 4,
        'freeze_norm': True,
        'pretrained_2d': False,
        'stride': [8, 16, 32],
        ### HybridEncoder
        'in_channels': [512, 1024, 2048],
        'feat_strides': [8, 16, 32],
        #### intra
        'hidden_dim': 256,
        'use_encoder_idx': [2],
        'num_encoder_layers': 1,
        'nhead': 8,
        'dim_feedforward': 1024,
        'dropout': 0.,
        'enc_act': 'gelu',
        'pe_temperature': 10000,
        #### cross
        'expansion': 1.0,
        'depth_mult': 1,
        'act': 'silu',
        #### eval
        'eval_spatial_size': [224, 224],
        ## 3D
        'backbone_3d': 'resnext101',
        'pretrained_3d': True,
        'memory_momentum': 0.9,

        # decoder
        'feat_channels': [256, 256, 256],
        'feat_strides': [8, 16, 32],
        'hidden_dim': 256,
        'num_levels': 3,
        ## query selection
        'num_queries': 300,
        ## contrastive denoising
        'num_decoder_layers': 6, # TODO: 6
        'num_denoising': 100,# 100
        'eval_idx': -1,
        'eval_spatial_size': [224, 224],

        # matcher
        'matcher_weight_dict': {'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
        # use_focal_loss: True
        'matcher_alpha': 0.25,
        'matcher_gamma': 2.0,

        # criterion
        'weight_dict': {'loss_vfl': 2.5, 'loss_bbox': 5, 'loss_giou': 2, }, #TODO: loss_vfl:1
        'losses': ['vfl', 'boxes'],
        'alpha': 0.75,
        'gamma': 2.0,

        # head
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False
    }
}
