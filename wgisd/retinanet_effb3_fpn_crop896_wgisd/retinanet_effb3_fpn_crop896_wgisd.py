model = dict(
    type='RetinaNet',
    backbone=dict(
        type='EfficientNet',
        arch='b3',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        norm_cfg=dict(type='BN', requires_grad=True, eps=0.001, momentum=0.01),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'
        )),
    neck=dict(
        type='FPN',
        in_channels=[48, 136, 384],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=5,
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=dict(type='BN', requires_grad=True)),
    bbox_head=dict(
        type='RetinaSepBNHead',
        num_classes=5,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        num_ins=5,
        norm_cfg=dict(type='BN', requires_grad=True)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(896, 896),
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(896, 896)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(896, 896)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(896, 896),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(896, 896)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='CocoDataset',
        ann_file=
        '/home/m351534/data/wgisd/coco_annotations/train_polygons_instances.json',
        img_prefix='/home/m351534/data/wgisd/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=(896, 896),
                ratio_range=(0.8, 1.2),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(896, 896)),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(896, 896)),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=('Chardonnay', 'Cabernet Franc', 'Cabernet Sauvignon',
                 'Sauvignon Blanc', 'Syrah')),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/home/m351534/data/wgisd/coco_annotations/test_polygons_instances.json',
        img_prefix='/home/m351534/data/wgisd/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(896, 896),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size=(896, 896)),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('Chardonnay', 'Cabernet Franc', 'Cabernet Sauvignon',
                 'Sauvignon Blanc', 'Syrah')),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/home/m351534/data/wgisd/coco_annotations/test_polygons_instances.json',
        img_prefix='/home/m351534/data/wgisd/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(896, 896),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size=(896, 896)),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('Chardonnay', 'Cabernet Franc', 'Cabernet Sauvignon',
                 'Sauvignon Blanc', 'Syrah')))
evaluation = dict(interval=1, metric='bbox')
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=32)
cudnn_benchmark = True
norm_cfg = dict(type='BN', requires_grad=True)
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'
img_size = (896, 896)
optimizer_config = dict(grad_clip=None)
optimizer = dict(
    type='SGD',
    lr=0.04,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(norm_decay_mult=0, bypass_duplicate=True))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=30)
MMDET_CONFIG_DIR = '/home/m351534/mmdetection/configs/'
classes = ('Chardonnay', 'Cabernet Franc', 'Cabernet Sauvignon',
           'Sauvignon Blanc', 'Syrah')
n_classes = 5
work_dir = '/home/m351534/tutfrutdet/wgisd/retinanet_effb3_fpn_crop896_wgisd/'
auto_resume = False
gpu_ids = [0]
