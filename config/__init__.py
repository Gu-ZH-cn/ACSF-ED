from .dataset_config import dataset_config
from .acsf_ed_config import acsf_ed_config


def build_model_config(args):
    print('==============================')
    print('Model Config: {} '.format(args.version.upper()))
    
    if 'ACSF-ED' in args.version:
        m_cfg = acsf_ed2_config[args.version]

    return m_cfg


def build_dataset_config(args):
    print('==============================')
    print('Dataset Config: {} '.format(args.dataset.upper()))
    
    d_cfg = dataset_config[args.dataset]

    return d_cfg
