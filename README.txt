This repository contains the reference code for the NeurIPS-2022 accepted paper 
"AdaFocal: Calibration-aware Adaptive Focal Loss".
Authors: Arindam Ghosh, Thomas Schaaf, and Matt Gormley.
Url: https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a692a24dbc744fca340b9ba33bc6522-Abstract-Conference.html 

The code provides the bare minimum required to reproduce the calibration related results obtained from training 
a ResNet-50 model on the CIFAR-10 dataset using the Adafocal loss function.
CIFAR-10 is an image dataset publicly available at https://www.cs.toronto.edu/~kriz/cifar.html
ResNet-50 is a deep residual network whose implementation is publicly available at https://github.com/KaimingHe/deep-residual-networks

If the code or the paper has been useful in your research, please add the citation:
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