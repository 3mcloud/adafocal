# AdaFocal: Calibration-aware Adaptive Focal Loss (NeurIPS 2022)
This is the official implementation of the paper titled **"AdaFocal: Calibration-aware Adaptive Focal Loss"** published in neurIPS 2022. <br />
**Authors: Arindam Ghosh, Thomas Schaaf, and Matt Gormley** <br />
**Url**: https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a692a24dbc744fca340b9ba33bc6522-Abstract-Conference.html <br />


The code provides the bare minimum required to reproduce the calibration related results obtained from training a ResNet-50 model on the CIFAR-10 dataset using the Adafocal loss function.
Most of the starter code is borrowed from the repository https://github.com/torrvision/focal_calibration.

## Training
To train Resnet-50 on CIFAR-10 with default settings, run:
```train
python main.py --dataset cifar10 --model resnet50 --loss adafocal -e 350 --save-path exp/cifar10_resnet50_adafocal
```

## Citation
If the code or the paper has been useful in your research, please add the citation:
```citation
@inproceedings{NEURIPS2022_0a692a24,
 author = {Ghosh, Arindam and Schaaf, Thomas and Gormley, Matthew},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {1583--1595},
 publisher = {Curran Associates, Inc.},
 title = {AdaFocal: Calibration-aware Adaptive Focal Loss},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/0a692a24dbc744fca340b9ba33bc6522-Paper-Conference.pdf},
 volume = {35},
 year = {2022}
}
```
