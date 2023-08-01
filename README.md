# AdaFocal: Calibration-aware Adaptive Focal Loss (NeurIPS 2022)
This is the official code for the paper **AdaFocal: Calibration-aware Adaptive Focal Loss** <br />
**Authors**: Arindam Ghosh, Thomas Schaaf, and Matt Gormley <br />
**URL**: https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a692a24dbc744fca340b9ba33bc6522-Abstract-Conference.html <br />
**Arxiv**: https://arxiv.org/abs/2211.11838 <br />

The code provides the bare minimum to reproduce the calibration related results obtained from training a ResNet-50 model on CIFAR-10 dataset using Adafocal loss. <br />

Most of the starter code for training, evaluation and calculating calibration related metrics is borrowed from https://github.com/torrvision/focal_calibration.

## Training
To train Resnet-50 on CIFAR-10 with default settings for Adafocal, run:
```train
python main.py --dataset cifar10 --model resnet50 --loss adafocal -e 350 --save-path exp/cifar10_resnet50_adafocal
```

## Citation
If the code or the paper has been useful in your research, please add the citation:
```citation
@inproceedings{NEURIPS2022_0a692a24,
 author = {Ghosh, Arindam and Schaaf, Thomas and Gormley, Matthew},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {1583--1595},
 title = {AdaFocal: Calibration-aware Adaptive Focal Loss},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/0a692a24dbc744fca340b9ba33bc6522-Paper-Conference.pdf},
 volume = {35},
 year = {2022}
}
```
