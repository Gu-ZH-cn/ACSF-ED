import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from config.yowo_v2_config import yowo_v2_config

try:
    from .PResNet import build_PResNet
    from .HybridEncoder import build_HybridEncoder
except:
    from PResNet import build_PResNet
    from HybridEncoder import build_HybridEncoder

class PResNet_HybridEncoder(nn.Module):
    def __init__(self, cfg):
        super(PResNet_HybridEncoder, self).__init__()
        self.cfg = cfg
        self.backbone = build_PResNet(self.cfg, pretrained=cfg['pretrained_2d'])
        self.encoder = build_HybridEncoder(self.cfg)

    def forward(self, x):
        # PResNet
        x = self.backbone(x)
        # HybridEncoder
        x = self.encoder(x)
        return x


# build PResNet + HybridEncoder
def build_PResNet_HybridEncoder(cfg, pretrained=False):

    # PResNet_HybridEncoder
    model = PResNet_HybridEncoder(cfg)
    feat_dims = [cfg['head_dim']] * 3

    return model, feat_dims



if __name__ == '__main__':
    model, fpn_dim = build_PResNet_HybridEncoder(yowo_v2_config['yowo_v2_detr'])
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    x = model(x)

    for n in x:
        print(n.shape)
