from tree_ring_analyzer.segmentation import TreeRingSegmentation
from tree_ring_analyzer.dl.train import bce_dice_loss
import tifffile
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os
import numpy as np



if __name__ == '__main__':
    input_folder = '/home/khietdang/Documents/khiet/treeRing/input'
    # mask_folder = '/home/khietdang/Documents/khiet/treeRing/masks'
    output_folder = '/home/khietdang/Documents/khiet/treeRing/output'
    checkpoint_ring_path = '/home/khietdang/Documents/khiet/tree-ring-analyzer/models/bigDisRingAugGray.keras'
    checkpoint_pith_path = '/home/khietdang/Documents/khiet/tree-ring-analyzer/models/pithGray.keras'

    image_list = glob.glob(os.path.join(input_folder, '*.tif'))
    # image_list = [os.path.join(input_folder, '12 t_8µm_x50.tif')]
    for image_path in image_list:
        print(image_path)
        image = tifffile.imread(image_path)
        # mask = tifffile.imread(os.path.join(mask_folder, os.path.basename(image_path)))

        image = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2])[:, :, None]

        modelRing = tf.keras.models.load_model(checkpoint_ring_path)

        modelPith = tf.keras.models.load_model(checkpoint_pith_path, custom_objects={'bcl': bce_dice_loss(bce_coef=0.3)})

        treeRingSegment = TreeRingSegmentation(resize=5)
        treeRingSegment.segmentImage(modelRing, modelPith, image)
        
        result = treeRingSegment.maskRings
        image[result == 255] = 0

        tifffile.imwrite(os.path.join(output_folder, os.path.basename(image_path)), image.astype(np.uint8))

        # hausdorff, mse = treeRingSegment.evaluate(mask)
        # print('\tHausdorff Distance:', hausdorff)
        # print('\tMSE:', mse)

