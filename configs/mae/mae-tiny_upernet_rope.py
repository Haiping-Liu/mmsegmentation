_base_ = [
    '../_base_/models/upernet_mae.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py',
]


crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

custom_imports = dict(
    imports=['mmseg.models.backbones.VisionTransformerRoPE'],  
    allow_failed_imports=False
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='VisionTransformerRoPE',
        config_path=(
                    '/workspace/Learning-RoPE/ckpt/householder/config.yaml'
        ),
        ckpt_path='/workspace/Learning-RoPE/ckpt/householder/householder-epoch-399.ckpt',
        encoder_type='mae',
        return_cls_token=False,
        return_feat_only=False,
        norm_eval=True,
        img_size=512,
        out_indices=[3, 5, 7, 11],
    ),
    neck=None,

    decode_head=dict(
        type='UPerHead',
        in_channels=[384, 384, 384, 384],
        channels=384,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        loss_decode=dict(type='CrossEntropyLoss',
                         use_sigmoid=False,
                         loss_weight=1.0)
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        channels=384,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        loss_decode=dict(type='CrossEntropyLoss',
                         use_sigmoid=False,
                         loss_weight=0.4)
    ),
    test_cfg=dict(
        mode='slide',
        crop_size=(512, 512),
        stride=(341, 341)
    )
)
    
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),        weight_decay=0.05,
    ),
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.65,      
    ),
    constructor='LayerDecayOptimizerConstructor'
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=160000,
    val_interval=2000
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

log_processor = dict(by_epoch=False)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    visualization=dict(
        type='SegVisualizationHook',
        draw=True,
        interval=200
    ),
    checkpoint=dict(
        type='CheckpointHook',
        interval=2000,
        save_best='mIoU',
        rule='greater'
    )
)


# mixed precision
fp16 = dict(loss_scale='dynamic')

# By default, models are trained on 8 GPUs with 2 images per GPU
dataset_type = 'ADE20KDataset'
data_root = '/workspace/Learning-RoPE/data/ade/ADEChallengeData2016'  # ✅ use absolute path

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training',
            seg_map_path='annotations/training'),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
    )
)

test_dataloader = val_dataloader

