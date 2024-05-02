# the new config inherits the base configs to highlight the necessary modification
_base_ = '/home/m351534/mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'

# 1. data pipeline

img_size=(1333, 800)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=img_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=img_size, keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')
    )
]

# 2. dataset settings
dataset_type = 'CocoDataset'
classes = (
    "Chardonnay",
    "Cabernet Franc",
    "Cabernet Sauvignon",
    "Sauvignon Blanc",
    "Syrah"
)
data_root='/home/m351534/data/wgisd/'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        metainfo=dict(classes=classes),
        type=dataset_type,
        data_root=data_root,
        ann_file='coco_annotations/train_polygons_instances.json',
        data_prefix=dict(img='data/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=None
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo=dict(classes=classes),
        type=dataset_type,
        data_root=data_root,
        ann_file='coco_annotations/test_polygons_instances.json',
        data_prefix=dict(img='data/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/coco_annotations/test_polygons_instances.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=None
)

test_evaluator = val_evaluator


# 3. model
n_classes = 5
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,  
        pad_size_divisor=32
    ),  
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
resume=True
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=800, val_interval=10)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=80,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=720,
        by_epoch=True,
        begin=80,
        end=800,
        convert_to_iter_based=True)
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')


default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=5
    ),
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto',
        by_epoch=True,
        interval=10,
        max_keep_ckpts=3
    )
)
