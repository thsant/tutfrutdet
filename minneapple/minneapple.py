# Configurações do dataset: MinneApple

# Tipo: Formato Microsoft COCO (arquivos JSON)
dataset_type = 'CocoDataset'

# Tupla contendo o nome das classes. Note a vírgula ao final, que força a criação
# de uma tupla em Python, mesmo havendo somente um valor ("maçã").
classes = (
    "maçã",
)

TUTFRUTDET_ROOT='/home/m351534/tutfrutdet/'      # Diretório onde o tutorial se encontra
IMG_DIR='/home/m351534/data/minneapple/images/'  # Diretório contendo as imagens da Minneapple

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

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=TUTFRUTDET_ROOT + 'minneapple/annotations/instances_train.json', # Localização do JSON com anotações
        img_prefix=IMG_DIR,                                                       # Localização das imagens
        pipeline=train_pipeline                                                   # Executa o pipeline para dados de TREINAMENTO 
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=TUTFRUTDET_ROOT + 'minneapple/annotations/instances_val.json',  # Localização do JSON com anotações
        img_prefix=IMG_DIR,                                                      # Localização das imagens
        pipeline=test_pipeline                                                   # Executa o pipeline para dados de TESTE 
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=TUTFRUTDET_ROOT + 'minneapple/annotations/instances_test.json', # Localização do JSON com anotações
        img_prefix=IMG_DIR,                                                      # Localização das imagens
        pipeline=test_pipeline                                                   # Executa o pipeline para dados de TESTE
    )
)
