from tree_ring_analyzer.segmentation import TreeRingSegmentation, Evaluation
from tree_ring_analyzer.dl.train import bce_dice_loss
import tifffile
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import cv2



if __name__ == '__main__':
    input_folder = '/home/khietdang/Documents/khiet/treeRing/Luidmila/50 tilias'
    mask_folder = '/home/khietdang/Documents/khiet/treeRing/masks'
    output_folder = '/home/khietdang/Documents/khiet/treeRing/Luidmila/50 tilias_output'
    checkpoint_ring_path = '/home/khietdang/Documents/khiet/tree-ring-analyzer/models/bigDisRingAugGrayNormal.keras'
    checkpoint_pith_path = '/home/khietdang/Documents/khiet/tree-ring-analyzer/models/pithGrayNormal.keras'

    # image_list = glob.glob(os.path.join(input_folder, '*.tif')) + glob.glob(os.path.join(input_folder, '*.jpg'))
    image_list = [os.path.join(input_folder, '48(3)_x50_8 µm.tif')]
    hausdorff = []
    for image_path in image_list:
        print(image_path)
        if image_path.endswith('.tif'):
            image = tifffile.imread(image_path)
        elif image_path.endswith('.jpg'):
            image = cv2.imread(image_path)
        # mask = tifffile.imread(os.path.join(mask_folder, os.path.basename(image_path)))

        image = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2])[:, :, None]

        modelRing = tf.keras.models.load_model(checkpoint_ring_path)

        modelPith = tf.keras.models.load_model(checkpoint_pith_path, custom_objects={'bcl': bce_dice_loss(bce_coef=0.5)})

        treeRingSegment = TreeRingSegmentation(resize=5)
        treeRingSegment.segmentImage(modelRing, modelPith, image)
        
        result = treeRingSegment.maskRings
        image[result == 255] = 0

        tifffile.imwrite(os.path.join(output_folder, os.path.basename(image_path)), image.astype(np.uint8))

        # hausdorff.append(Evaluation(mask, treeRingSegment.maskRings).evaluateHausdorff())
        # print('\tHausdorff Distance:', hausdorff[-1])

    # print('Mean:', np.mean(hausdorff))

