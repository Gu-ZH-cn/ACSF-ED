# import 2D backbone
from .PResNet_HybridEncoder.PResNet_HybridEncoder import build_PResNet_HybridEncoder


def build_2d_cnn(cfg, pretrained=False):
    print('==============================')
    print('2D Backbone: {}'.format(cfg['backbone_2d'].upper()))
    print('--pretrained: {}'.format(pretrained))

    if cfg['backbone_2d'] in ['PResNet']:
        model, feat_dims = build_PResNet_HybridEncoder(cfg, pretrained)

    else:
        print('Unknown 2D Backbone ...')
        exit()

    return model, feat_dims
