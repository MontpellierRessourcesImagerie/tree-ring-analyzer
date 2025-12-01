# Tree Ring Analyzer

[![License MIT](https://img.shields.io/pypi/l/tree-ring-analyzer.svg?color=green)](https://github.com/MontpellierRessourcesImagerie/tree-ring-analyzer/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tree-ring-analyzer.svg?color=green)](https://pypi.org/project/tree-ring-analyzer)
[![Python Version](https://img.shields.io/pypi/pyversions/tree-ring-analyzer.svg?color=green)](https://python.org)
[![tests](https://github.com/MontpellierRessourcesImagerie/tree-ring-analyzer/workflows/tests/badge.svg)](https://github.com/MontpellierRessourcesImagerie/tree-ring-analyzer/actions)

Tree Ring Analyzer is an automatic framework that allows the segmentation and detection of tree rings on 2D stained microscopy images.

## How to install/upgrade it?

| Action      | Command                                                                                               |
|-------------|-------------------------------------------------------------------------------------------------------|
| **Install** | `pip install tree-ring-analyzer`                                                                      |
| **Upgrade** | `pip install --upgrade tree-ring-analyzer`                                                            |


## How to use it?
- The input image should be stained microscopy cross-sectional tree-ring image with the format of TIF file.
- The two input models, pith prediction and ring segmentation, can be downloaded via the links in: [model_urls.json](https://raw.githubusercontent.com/MontpellierRessourcesImagerie/napari-tree-rings/refs/heads/main/src/napari_tree_rings/config/model_urls.json)
- Run `TreeRingSegmentation(resize=5, pithWhole=False, rotate=True, lossType='H0', removeRing=True, thickness=1).segmentImage(modelRing, modelPith, image)`, in which:
    - resize: Controls how much your image is scaled down before post-processing (default is 5).
    - pithWhole: True/False. If True, the pith image will not be cropped (default is False).
    - rotate: True/False. If True, FDRS algorithm will be used (default is True).
    - lossType: The type of heuristic function, including 'H0', 'H01', and 'H02' (default is 'H0').
    - removeRing: True/False. If True, IRR algorithm will be used (default is True).
    - thickness: the thickness of output ring (default is 1).
    - modelRing: Choose loaded Keras model for segmenting tree ring boundaries.
    - modelPith: Choose loaded Keras model for segmenting pith.
    - image: The loaded tree cross-sectional image, should be in the shape of (Y, X, 1).
- The output will consist of:
    - predictedRings: list of detected ring boundaries
    - maskRings: binary prediction of ring boundaries

- For more details, check the [detailed documentation](https://montpellierressourcesimagerie.github.io/napari-tree-rings).


## Pre-processing
Please run the file `preprocessing.py` to generate pre-processed data before training, in which:
- input_path: directory of original images
- mask_path: directory of ground truths
- pith_path: directory to save pre-processed images for training pith-prediction model. If None, the pith dataset will not be generated.
- tile_path: directory to save pre-processed images for training ring-segmentation model. If None, the ring dataset will not be generated.
- pithWhole: True/False. If True, the pith image will not be cropped (default is False).
- whiteHoles: True/False. If True, the white holes will be added into ring dataset for augmentation (default is True).
- gaussianHoles: True/False. If True, the gaussian holes will be added into ring dataset for augmentation (default is False).
- changeColor: True/False. If True, the order of image channels will be changed for augmentation (default is False).
- dilate: an integer. If not None, the tree rings in ground truth will be dilated with the given number of iterations before calculating the distance map (default is 10).
- distance: True/False. If True, distance map will be calculated.
- skeleton: True/False. If True, the tree rings in ground truth will be skeletonized.

## Training
Please run the file `training.py` to train the pith-prediction or ring-segmentation models, in which:
- train_input_path: directory of training input path
- train_mask_path: directory of training mask path
- val_input_path: directory of validation input path
- val_mask_path: directory of validation mask path
- filter_num: the number of filters in UNet architecture (default is [16, 24, 40, 80, 960])
- attention: True/False. If True, the model will be Attention UNet.
- output_activation: output activation. In pith prediction, the recommended output activation is 'sigmoid', while in the ring segmentation, the recommended output activation is 'linear'.
- loss: loss function. In pith prediction, the recommended loss function is bce_dice_loss(bce_coef=0.5), while in the ring segmentation, the recommended loss function is 'mse'. 
- name: name of the saved model
- numEpochs: number of epochs. In pith prediction, the recommended number is 100, while in the ring segmentation, the recommended number is 30. 
- input_size: size of input. Default is (256, 256, 1).

The outputs will include:
- models in keras and H5 formats (saved in models folder).
- history in JSON format (saved in history folder)

## Testing
Please run the file `test_segmentation.py` to test, in which:
- input_folder: directory of original input path (no pre-processing)
- mask_folder: directory of original mask path (no pre-processing)
- output_folder: directory of output path
- checkpoint_ring_path: directory of trained ring-segmentation model
- checkpoint_pith_path: directory of trained pith-prediction model
- csv_file: directory of output csv file
- pithWhole: True/False. If True, the pith image will not be cropped (default is False).
- rotate: True/False. If True, FDRS algorithm will be used (default is True).
- removeRing: True/False. If True, IRR algorithm will be used (default is True).
- lossType: the type of heuristic function, including 'H0', 'H01', and 'H02' (default is 'H0').
- thickness: the thickness of output ring (default is 1).

The outputs will include the predicted rings in output folder and csv file (containing the results of the evaluation metrics, including Hausdorff distance, mAR, ARAND, recall, precision, accuracy).

[🐛 Found a bug?]: https://github.com/MontpellierRessourcesImagerie/tree-ring-analyzer/issues
[🔍 Need some help?]: mri-cia@mri.cnrs.fr
