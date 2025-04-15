from tree_ring_analyzer.segmentation import TreeRingSegmentation
import tifffile
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os
import cv2



if __name__ == '__main__':
    input_folder = '/home/khietdang/Documents/khiet/treeRing/transfer/input_transfer'
    # mask_folder = '/home/khietdang/Documents/khiet/treeRing/masks'
    output_folder = '/home/khietdang/Documents/khiet/treeRing/transfer/output_transfer'
    checkpoint_ring_path = '/home/khietdang/Documents/khiet/tree-ring-analyzer/models/bigDisRingAugGray.keras'
    checkpoint_pith_path = '/home/khietdang/Documents/khiet/tree-ring-analyzer/models/pith.keras'

    # image_list = glob.glob(os.path.join(input_folder, '*.tif'))
    image_list = [os.path.join(input_folder, '39(6)_x50_8 µm.tif')]
    for image_path in image_list:
        print(image_path)
        image = tifffile.imread(image_path)
        # mask = tifffile.imread(os.path.join(mask_folder, os.path.basename(image_path)))

        modelRing = tf.keras.models.load_model(checkpoint_ring_path)

        modelPith = tf.keras.models.load_model(checkpoint_pith_path)

        treeRingSegment = TreeRingSegmentation(resize=5)
        treeRingSegment.segmentImage(modelRing, modelPith, image)
        
        result = treeRingSegment.maskRings
        image[result == 255] = 0

        tifffile.imwrite(os.path.join(output_folder, os.path.basename(image_path)), image)

        # hausdorff, mse = treeRingSegment.evaluate(mask)
        # print('\tHausdorff Distance:', hausdorff)
        # print('\tMSE:', mse)

