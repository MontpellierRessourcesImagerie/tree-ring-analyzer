from tree_ring_analyzer.segmentation import TreeRingSegmentation
import tifffile
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os
import cv2



if __name__ == '__main__':
    input_folder = '/home/khietdang/Documents/khiet/treeRing/input'
    output_folder = '/home/khietdang/Documents/khiet/treeRing/output'
    checkpoint_ring_path = '/home/khietdang/Documents/khiet/tree-ring-analyzer/models/bigDistance.keras'
    checkpoint_pith_path = '/home/khietdang/Documents/khiet/tree-ring-analyzer/models/pith.keras'

    image_list = glob.glob(os.path.join(input_folder, '*.tif'))
    # image_list = [os.path.join(input_folder, '39(6)_x50_8 µm.tif')]
    for image_path in image_list:
        print(image_path)
        image = tifffile.imread(image_path)

        model_ring = tf.keras.models.load_model(checkpoint_ring_path)

        model_pith = tf.keras.models.load_model(checkpoint_pith_path)

        treeRingSegment = TreeRingSegmentation(model_ring, model_pith)
        treeRingSegment.segmentImage(image)
        
        result = treeRingSegment.maskRings

        tifffile.imwrite(os.path.join(output_folder, os.path.basename(image_path)), result)

