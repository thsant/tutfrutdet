# the new config inherits the base configs to highlight the necessary modification
MMDET_CONFIG_DIR='/home/m351534/mmdetection/configs/'
_base_ = MMDET_CONFIG_DIR + 'efficientnet/retinanet_effb3_fpn_crop896_8x4_1x_coco.py'

# 1. data pipeline: same as base config

# 2. dataset settings
dataset_type = 'CocoDataset'
classes = ("apple",)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/m351534/data/minneapple/annotations/instances_train.json',
        img_prefix='/home/m351534/data/minneapple/images/'
    ),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/m351534/data/minneapple/annotations/instances_val.json',
        img_prefix='/home/m351534/data/minneapple/images/'
    ),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/m351534/data/minneapple/annotations/instances_test.json',
        img_prefix='/home/m351534/data/minneapple/images/'
    )
)

# 3. model
n_classes = 1
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

runner = dict(type='EpochBasedRunner', max_epochs=50)
