from .acsfed.build import build_ACSFED


def build_model(args,
                d_cfg,
                m_cfg, 
                device, 
                num_classes=80, 
                trainable=False,
                resume=None):
    # build action detector
    if 'ACSF-ED' in args.version:
        model, criterion = build_acsfed(
            args=args,
            d_cfg=d_cfg,
            m_cfg=m_cfg,
            device=device,
            num_classes=num_classes,
            trainable=trainable,
            resume=resume
            )

    return model, criterion

