# the new config inherits the base configs to highlight the necessary modification
MMDET_CONFIG_DIR='/home/m351534/mmdetection/configs/'
_base_ = MMDET_CONFIG_DIR + 'efficientnet/retinanet_effb3_fpn_crop896_8x4_1x_coco.py'

# 1. data pipeline: same as base config

# 2. dataset settings
dataset_type = 'CocoDataset'
classes = (
    "Chardonnay",
    "Cabernet Franc",
    "Cabernet Sauvignon",
    "Sauvignon Blanc",
    "Syrah"
)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/m351534/data/wgisd/coco_annotations/train_polygons_instances.json',
        img_prefix='/home/m351534/data/wgisd/data'
    ),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/m351534/data/wgisd/coco_annotations/test_polygons_instances.json',
        img_prefix='/home/m351534/data/wgisd/data'
    ),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/m351534/data/wgisd/coco_annotations/test_polygons_instances.json',
        img_prefix='/home/m351534/data/wgisd/data'
    )
)

# 3. model
n_classes = 5
model = dict(
    backbone=dict(
        norm_cfg=dict(
            type='BN', requires_grad=True, eps=1e-3, momentum=0.01)),
    bbox_head=dict(
        num_classes=n_classes
    )
)

# 4. runtime
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TensorboardLoggerHook')
    ]
)

runner = dict(type='EpochBasedRunner', max_epochs=30)
