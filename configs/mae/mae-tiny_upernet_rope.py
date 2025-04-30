_base_ = [
    '../_base_/models/upernet_mae.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

custom_imports = dict(
    imports=['mmseg.models.backbones.VisionTransformerRoPE'],  # 注意是 Python 模块路径
    allow_failed_imports=False
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='VisionTransformerRoPE',
        config_path=(
            '/workspace/Learning-RoPE/outputs/'
            'householder-mae-rope-lr0.00015-bs256-0422_1540/'
            'configs/config.yaml'
        ),
        encoder_type='mae',
        return_cls_token=False,
        return_feat_only=False,
        norm_eval=True,
        img_size=512,
        out_indices=[3, 5, 7, 11],
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/workspace/Learning-RoPE/mae_encoder_only.pth'
        )
    ),
    # 明确告诉 mmseg：没有 neck，删掉基线里的 FeaturePyramid
    neck=None,

    decode_head=dict(
        type='UPerHead',
        in_channels=[192, 192, 192, 192],
        channels=192,
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
        in_channels=192,
        channels=192,
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



# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
#     paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65),
#     constructor='LearningRateDecayOptimizerConstructor')
    
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
        layer_decay_rate=0.65,      # 你的旧写法
    ),
    constructor='LayerDecayOptimizerConstructor'  # 注意这里
)

# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW',
#         lr=1e-4,
#         betas=(0.9, 0.999),
#         weight_decay=0.05
#     ),
#     constructor='LearningRateDecayOptimizerConstructor',
#     paramwise_cfg=dict(
#         num_layers=12,
#         decay_type='layer_wise_vit',
#         decay_rate=0.65
#     )
# )

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

# mixed precision
fp16 = dict(loss_scale='dynamic')

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
