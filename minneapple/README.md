Base de dados anotada
---------------------

**MinneApple** é uma base de dados para **detecção e segmentação de instâncias para maçãs**, criada por Nicolai Haeni, Pravakar Roy e Volkan Isler no *Robotic Sensor Network Laboratory* da Universidade de Minnesota, EUA. O download da base deve ser feito diretamente do [Repositório de Dados da Universidade de Minnesota](https://doi.org/10.13020/8ecp-3r13). Aqui, nós oferecemos uma versão das anotações originais de Haeni et al. no [formato COCO em JSON](https://cocodataset.org/#format-data). Com essas anotações, é possível usar diretamente o [CocoDataset](https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.datasets.CocoDataset) da MMDetection para realizar a leitura dos dados e entrada no *pipeline* (outras bibliotecas e frameworks de *deep learning* também oferecem utilitários para carga de dados no formato COCO).

Os autores da MinneApple oferecem anotações com máscaras *apenas para as 670 imagens no conjunto de treinamento original* da base (não há anotações para o conjunto de testes). Para termos conjuntos de treinamento, validação e teste *com anotações* para o treinamento e avaliação em segmentação de maçãs, dividimos o conjunto original de 670 imagens anotadas em um conjunto de treinamento (536 imagens), validação (67 imagens) e teste (67 imagens), separados aleatoriamente.

Um exemplo de configuração de base de dados pode ser visto em [minneapple.py](minneapple.py) (veja *[Customize Datasets](https://mmdetection.readthedocs.io/en/latest/tutorials/customize_dataset.html#tutorial-2-customize-datasets)* para detalhes). As imagens são os arquivos PNG encontrados em `detection/train/images/` na MinneApple original.

A MMDetection oferece um utilitário para visualização das bases de dados: `browse_dataset.py`. Ele se encontra no diretório `tools/misc` na distribuição da MMDetection. A base pode ser visualizada a partir do script de configuração:

```
cd ~/tutfrutdet/minneapple/
conda activate openmmlab
export MMDET_ROOT=~/mmdetection/
python $MMDET_ROOT/tools/misc/browse_dataset.py minneapple.py
```

