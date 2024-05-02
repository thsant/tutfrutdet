# Configurações do dataset: WGISD

# Tipo: Formato Microsoft COCO (arquivos JSON)
dataset_type = 'CocoDataset'
classes = (
    "Chardonnay",
    "Cabernet Franc",
    "Cabernet Sauvignon",
    "Sauvignon Blanc",
    "Syrah"
)

# Localização do diretório do dataset. Abaixo, assumimos que o WGISD
# está disponível no diretório 'wgisd', no diretório local 'data'. Para
# realizar o download do WGISD, utilize:
#
# $ cd data
# $ git clone https://github.com/thsant/wgisd.git
#
data_root = 'data/wgisd/'           

# Média e desvio padrão para os pixeis, computado para todas as imagens
# do conjunto de treinamento. São utilizados no processo de 'standardization'
# (subtração da média e divisão pelo desvio padrão)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
        img_scale=(1333, 800),
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

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'coco_annotations/train_polygons_instances.json', # Localização do JSON com anotações
        img_prefix=data_root + 'data',                                         # Localização das imagens
        pipeline=train_pipeline                                                # Executa o pipeline para dados de TREINAMENTO 
    ),                                              
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'coco_annotations/test_polygons_instances.json', # Localização do JSON com anotações
        img_prefix=data_root + 'data',                                        # Localização das imagens
        pipeline=test_pipeline                                                # Executa o pipeline para dados de TESTE 
    ),                                              
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'coco_annotations/test_polygons_instances.json', # Localização do JSON com anotações
        img_prefix=data_root + 'data',                                        # Localização das imagens
        pipeline=test_pipeline                                                # Executa o pipeline para dados de TESTE 
    )
)
