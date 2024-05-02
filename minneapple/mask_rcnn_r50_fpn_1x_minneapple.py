# the new config inherits the base configs to highlight the necessary modification
MMDET_CONFIG_DIR='/home/m351534/mmdetection/configs/'
_base_ = MMDET_CONFIG_DIR + 'mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'

# 1. data pipeline

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(720, 1280), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(720, 1280),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# 2. dataset settings
dataset_type = 'CocoDataset'
classes = ("apple",)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/m351534/data/minneapple/annotations/instances_train.json',
        img_prefix='/home/m351534/data/minneapple/images/',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/m351534/data/minneapple/annotations/instances_val.json',
        img_prefix='/home/m351534/data/minneapple/images/',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/m351534/data/minneapple/annotations/instances_test.json',
        img_prefix='/home/m351534/data/minneapple/images/',
        pipeline=test_pipeline
    )
)

# 3. model
n_classes = 1
model = dict(
    roi_head =dict(
        bbox_head=dict(
            num_classes=n_classes
        ),
        mask_head=dict(
            num_classes=n_classes
        )
    )
)

# 4. runtime
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TensorboardLoggerHook')
    ]
)

runner = dict(type='EpochBasedRunner', max_epochs=150)
