_base_ = [
    'mmpretrain::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'projects.clip.models', 'projects.clip.datasets', 'projects.ma_clip.models'], 
    allow_failed_imports=False)

img_size = 224
patch_size = 16
context_length = 77
checkpoint = 'data/pretrained/clip/CLIP-ViT-B-16-laion2B-s34B-b88K/pretrain.pth'
# model setting
model = dict(
    type='MACLIP',
    vision_backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=img_size,
        patch_size=patch_size,
        drop_rate=0.1,
        out_type='raw',
        pre_norm=True,
        final_norm=True),
    text_backbone=dict(
        type='CLIPTextTransformer',
        context_length=context_length,
        vocab_size=49408,
        embed_dims=512,
        num_heads=8,
        num_layers=12),
    neck=dict(
        type='MACLIPNeck',
        encoder=dict(
            type='MACLIPMaskEncoder',
            embed_dims=768,
            hidden_dims=256),
        decoder=dict(
            type='MACLIPMaskDecoder',
            embed_dims=768 * 2,
            hidden_dims=1024,
            output_dims=512,
            img_size=img_size,
            patch_size=patch_size,
            num_layers=2,
            num_heads=8,
            mlp_ratio=4,
            with_cls_token=True)), 
    head=dict(
        type='CLIPClsHead',
        pred_mode='classifier',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
        cal_acc=True),
    proj_dims=512,
    tokenizer=dict(type='SimpleTokenizer'),
    context_length=context_length,
    text_prototype='objects365v2',
    text_prompt='openai',
    # text_prototype_path='work_dirs/cache/o365_clip_openai.npy',
    frozen_vision_stages=2,
    frozen_text_stages=2,
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint))

# data settings
data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53, 0.0],
    std=[58.395, 57.12, 57.375, 1.0],
    # loaded images are already RGB format
    to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropInstanceFromImage', exp_factor=1.2),
    dict(type='InstanceMaskPacker'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(0.5, 1.0),
        backend='cv2',  # image resize backend type must be cv2
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropInstanceFromImage', exp_factor=1.),
    dict(type='InstanceMaskPacker'),
    dict(
        type='Resize',
        scale=224,
        backend='cv2',
        interpolation='bicubic'),
    dict(type='PackInputs'),
]

train_dataset = dict(
    type='InstanceDataset',
    dataset=dict(
        type='mmdet.Objects365V2Dataset',
        data_root='data/Objects365/Obj365_v2/',
        ann_file='annotations/pseudo_mask/zhiyuan_objv2_minitrain_pmask.json',
        # ann_file='annotations/pseudo_mask/zhiyuan_objv2_train_pmask_mini_1.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True)),
    filter_cfg=dict(min_size=32),
    pipeline=train_pipeline)

val_dataset = dict(
    type='InstanceDataset',
    dataset=dict(
        type='mmdet.Objects365V2Dataset',
        data_root='data/Objects365/Obj365_v2/',
        ann_file='annotations/pseudo_mask/zhiyuan_objv2_val_pmask.json',
        data_prefix=dict(img='val/')),
    filter_cfg=dict(min_size=32),
    pipeline=test_pipeline)

train_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=train_dataset,
    sampler=dict(type='DefaultSampler', shuffle=True))

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=val_dataset,
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# schedule setting
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.000375, weight_decay=0.3),
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
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=512)
