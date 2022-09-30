# TutFrutDet

Código que acompanha o tutorial **Detecção de frutos por imagem com redes neurais convolutivas: um tutorial**, parte do 3º volume de livros da Rede de Agricultura de Precisão da Embrapa.

![Imagem com cachos de uva anotados](figures/CSV_1865.jpg)

## Instalação

Este tutorial utiliza a biblioteca [MMDetection](https://mmdetection.readthedocs.io) e PyTorch para pipeline de desenvolvimento de um detector de objetos (frutos). Também utilizamos o TensorBoard para monitoramento do processo de treinamento das redes neurais. Os seguintes comandos realizam a instalação de todos essas dependências.

```
miniconda3/condabin/conda create --name openmmlab python=3.9 -y
miniconda3/condabin/conda init bash
conda activate openmmlab
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install tensorboard
pip install -U openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection/
pip install -v -e .
```

## Treinamento

Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

```
python ./tools/train.py ~/tutfrutdet/minneapple/retinanet_effb3_fpn_crop896_minneapple.py \
--work-dir ~/tutfrutdet/minneapple/retinanet_effb3_fpn_crop896_minneapple
```

## Avaliação e inferência

```
python ./tools/test.py ~/tutfrutdet/minneapple/retinanet_effb3_fpn_crop896_minneapple/retinanet_effb3_fpn_crop896_minneapple.py \
~/tutfrutdet/minneapple/retinanet_effb3_fpn_crop896_minneapple/latest.pth \
--out ~/tutfrutdet/minneapple/retinanet_effb3_fpn_crop896_minneapple/results.pkl \
--show-dir ~/tutfrutdet/minneapple/retinanet_effb3_fpn_crop896_minneapple/results-output/  \
--eval bbox --show

python ./tools/analysis_tools/analyze_results.py \
~/tutfrutdet/minneapple/retinanet_effb3_fpn_crop896_minneapple/retinanet_effb3_fpn_crop896_minneapple.py \
~/tutfrutdet/minneapple/retinanet_effb3_fpn_crop896_minneapple/results.pkl \
~/tutfrutdet/minneapple/retinanet_effb3_fpn_crop896_minneapple/results-output/ \
--show --topk 7
```

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
