Files containing code:
1) asudata.py = for creating augmented data
2) augutil.py = as a utility to support asudata.py
3) fooling.py = for fooling the network, called via train.py
4) guided.py = for guided back propm called via train.py
5) model.py = contains model created in Pytorch, called via train.py
6) train.py = main file where the programs start
7) utils.py = contains support functions

Report PDF document:
    Report.pdf

Running Requirements:
    1) pytorch
    2) torchvision
    3) opencv
    4) python 3.6
    5) PIL
    6) skimage.io
    7) shutil
    8) multiprocessing

Running Instructions:
    Run main file as: python train.py
    (give parameters as required, parameter help available using -h standard help flag)