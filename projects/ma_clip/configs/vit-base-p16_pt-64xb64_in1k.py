_base_ = [
    'mmpretrain::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'projects.clip.models', 'projects.clip.datasets',
        'projects.ma_clip.models', 'projects.ma_clip.datasets',
        'mmdet.datasets.objects365'], 
    allow_failed_imports=False)

img_size = 224
patch_size = 16
checkpoint = 'data/pretrained/clip/CLIP-ViT-B-16-laion2B-s34B-b88K/pretrain.pth'
# model setting
model = dict(
    type='CLIPClassifier',
    backbone=dict(
        type='MACLIP',
        visual=dict(
            type='VisionTransformer',
            arch='b',
            img_size=img_size,
            patch_size=patch_size,
            drop_rate=0.1,
            pre_norm=True),
        visual_proj=False,
        output_dims=512,
        out_type='raw',
        frozen_visual_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    neck=dict(
        type='MACLIPNeck',
        encoder=dict(
            type='MACLIPMaskEncoder',
            embed_dims=768,
            hidden_dims=128),
        decoder=dict(
            type='MACLIPMaskDecoder',
            img_size=img_size,
            patch_size=patch_size,
            embed_dims=768 * 2,
            hidden_dims=1024,
            output_dims=512,
            num_layers=2,
            num_heads=8,
            mlp_ratio=4)), 
    head=dict(
        type='ZeroShotClsHead',
        num_classes=365,
        zs_weight_path='work_dirs/cache/o365_clip_openai.npy',
        zs_weight_dims=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        cal_acc=True,
        topk=(1, 5)))

# data settings
data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[125.307, 122.961, 113.8575, 0.0],
    std=[51.5865, 50.847, 51.255, 1.0],
    # loaded images are already RGB format
    to_rgb=False)

train_pipeline = [
    dict(
        type='LoadInstanceImage',
        with_mask=True,
        exp_factor=1.2,
        channel_order='rgb'),
    dict(
        type='ResizeEdge', 
        scale=256, 
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomCrop', crop_size=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(
        type='LoadInstanceImage',
        with_mask=True,
        exp_factor=1.2,
        channel_order='rgb'),
    dict(
        type='ResizeEdge',
        scale=224,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataset = dict(
    type='InstanceDataset',
    dataset=dict(
        type='mmdet.Objects365V2Dataset',
        data_root='data/Objects365/Obj365_v2/',
        # ann_file='debug/train.json',
        ann_file='annotations/pseudo_mask/zhiyuan_objv2_train_pmask_mini_1.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True)),
    filter_cfg=dict(min_size=32),
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=train_dataset,
    sampler=dict(type='DefaultSampler', shuffle=True))

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=train_dataset,
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# schedule setting
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.003, weight_decay=0.3),
    # specific to vit pretrain
    paramwise_cfg=dict(custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    }),
    clip_grad=dict(max_norm=1.0)
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=25,
        by_epoch=True,
        begin=5,
        end=30,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=30)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=512)
