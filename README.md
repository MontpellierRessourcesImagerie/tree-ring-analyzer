# Tree Ring Analyzer

Tree Ring Analyzer is an automatic framework that allows the segmentation and detection of tree rings on 2D stained microscopy images.

## Preparations
### Dataset
The input image should be stained microscopy cross-sectional tree-ring image with the format of TIF file.

### Downloading dependencies
Run the following command to install the necessary libraries
```
pip install -e .
```

### Pre-processing
Please run the file `preprocessing.py` to generate pre-processed data before training, in which:
- input_path: directory of original images
- mask_path: directory of ground truths
- pith_path: directory of pre-processed images for training pith-prediction model. If None, the pith dataset would not be generated.
- tile_path: directory of pre-processed images for training ring-regression model. If None, the ring dataset would not be generated.
- pithWhole: True/False. If True, the pith image would not be cropped (default is False).
- whiteHoles: True/False. If True, the white holes would be added into ring dataset for augmentation (default is True).
- gaussianHoles: True/False. If True, the gaussian holes would be added into ring dataset for augmentation (default is False).
- changeColor: True/False. If True, the order of image channels would be changed for augmentation (default is False).
- dilate: an interger. If not None, the tree rings in ground truth would be dilated with the given number of iterations before calculating the distance map (default is True).
- distance: True/False. If True, distance map would be calculated.
- skeleton: True/False. If True, the tree rings in ground truth would be skeletonized.

## Training
Please run the file `training.py` to train the pith-prediction or ring-regression models, in which:
- train_input_path: directory of training input path
- train_mask_path: directory of training mask path
- val_input_path: directory of validation input path
- val_mask_path: directory of validation mask path
- filter_num: the number of filters in UNet architecture (default is [16, 24, 40, 80, 960])
- attention: True/False. If True, the model would be Attention UNet.
- output_activation: output activation. In pith prediction, the recommended output activation is 'sigmoid', while in the ring regression, the recommended output activation is 'linear'.
- loss: loss function. In pith prediction, the recommended loss function is bce_dice_loss(bce_coef=0.5), while in the ring regression, the recommended loss function is 'mse'. 
- name: name of the saved model
- numEpochs: number of epochs. In pith prediction, the recommended number is 100, while in the ring regression, the recommended number is 30. 
- input_size: size of input. Default is (256, 256, 1).

The outputs would include:
- models in keras and H5 formats (saved in models folder).
- history in JSON format (saved in history folder)

## Testing
Please run the file `test_segmentation.py` to test, in which:
- input_folder: directory of original input path (no pre-processing)
- mask_folder: directory of original mask path (no pre-processing)
- output_folder: directory of output path
- checkpoint_ring_path: directory of trained ring-regression model
- checkpoint_pith_path: directory of trained pith-prediction model
- csv_file: directory of output csv file
- pithWhole: True/False. If True, the pith image would not be cropped (default is False).
- rotate: True/False. If True, FDRS algorithm would be used (default is True).
- removeRing: True/False. If True, IRR algorithm would be used (default is True).
- lossType: the type of heuristic function, including 'H0', 'H01', and 'H02' (default is 'H0').
- thickness: the thickness of output ring

The outputs would include the predicted rings in output folder and csv file (containing the results of the evaluation metrics, including Hausdorff distance, mAR, ARAND, recall, precision, accuracy).

[🐛 Found a bug?]: https://github.com/MontpellierRessourcesImagerie/tree-ring-analyzer/issues
[🔍 Need some help?]: mri-cia@mri.cnrs.fr
