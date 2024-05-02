Base de dados anotada
---------------------

**WGISD** (*Embrapa Wine Grape Instance Segmentation Dataset*) é uma base de dados para **detecção e segmentação de instâncias de cachos de uva**, criada por Thiago Santos, Andreza dos Santos, Leonardo de Souza e Sandra Avila. O download da base deve ser feito diretamente do GitHub:

```
cd ~/tutfrutdet/wgisd/
mkdir data
cd data
git clone https://github.com/thsant/wgisd.git
```

O repositório oficial da WGISD no GitHub possui uma descrição detalhada da base, seguindo as recomendações de *Datasheets for Datasets* ([Gebru *et al.*, 2021](https://doi.org/10.1145/3458723)).

Um exemplo de configuração de base de dados pode ser visto em [wgisd.py](wgisd.py) (veja *[Customize Datasets](https://mmdetection.readthedocs.io/en/latest/tutorials/customize_dataset.html#tutorial-2-customize-datasets)* para detalhes). As imagens são os arquivos JPEG encontrados em `wgisd/data`.

A MMDetection oferece um utilitário para visualização das bases de dados: `browse_dataset.py`. Ele se encontra no diretório `tools/misc` na distribuição da MMDetection. A base pode ser visualizada a partir do script de configuração:

```
cd ~/tutfrutdet/wgisd/
conda activate openmmlab
export MMDET_ROOT=~/mmdetection/
python $MMDET_ROOT/tools/misc/browse_dataset.py wgisd.py
```

