from tree_ring_analyzer.segmentation import TreeRingSegmentation, Evaluation
from tree_ring_analyzer.dl.train import bce_dice_loss
import tifffile
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import csv



if __name__ == '__main__':
    input_folder = '/home/khietdang/Documents/khiet/treeRing/input'
    mask_folder = '/home/khietdang/Documents/khiet/treeRing/masks'
    output_folder = '/home/khietdang/Documents/khiet/treeRing/output_H012'
    checkpoint_ring_path = '/home/khietdang/Documents/khiet/tree-ring-analyzer/models/bigDisRingAugGrayNormal16.keras'
    checkpoint_pith_path = '/home/khietdang/Documents/khiet/tree-ring-analyzer/models/pithGrayNormal16.keras'
    csv_file = '/home/khietdang/Documents/khiet/treeRing/doc/result_our.csv'
    pithWhole = False

    modelRing = tf.keras.models.load_model(checkpoint_ring_path)

    modelPith = tf.keras.models.load_model(checkpoint_pith_path, custom_objects={'bcl': bce_dice_loss(bce_coef=0.5)})
    channel = modelPith.get_config()['layers'][0]['config']['batch_shape'][-1]

    # image_list = glob.glob(os.path.join(input_folder, '*.tif')) + glob.glob(os.path.join(input_folder, '*.jpg'))
    # _, image_list = train_test_split(image_list, test_size=0.2, shuffle=True, random_state=42)
    # image_list = sorted(image_list)
    image_list = [os.path.join(input_folder, '3Tmilieu8microns_x40.tif')]

    hausdorff = []
    mAR = []
    arand = []
    recall = []
    precision = []
    f1 = []
    acc = []
    for image_path in image_list:
        print(image_path)
        if image_path.endswith('.tif'):
            image = tifffile.imread(image_path)
        elif image_path.endswith('.jpg'):
            image = cv2.imread(image_path)
        mask = tifffile.imread(os.path.join(mask_folder, os.path.basename(image_path)))

        plt.figure(figsize=(15, 10))
        plt.subplot(231)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Input', fontsize=20)

        if channel == 1:
            image = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2])[:, :, None]

        treeRingSegment = TreeRingSegmentation(resize=5, pithWhole=pithWhole)
        treeRingSegment.segmentImage(modelRing, modelPith, image)
        
        result = treeRingSegment.maskRings

        tifffile.imwrite(os.path.join(output_folder, os.path.basename(image_path)), result.astype(np.uint8))

        plt.subplot(232)
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.title('Ground truth', fontsize=20)
        plt.subplot(233)
        plt.imshow(result, cmap='gray')
        plt.axis('off')
        plt.title('Prediction', fontsize=20)

        evaluation = Evaluation(mask, treeRingSegment.maskRings)
        hausdorff = evaluation.evaluateHausdorff()
        mAR = evaluation.evaluatemAR()
        arand = evaluation.evaluateARAND()
        _recall, _precision, _f1, _acc = evaluation.evaluateRPFA()

        plt.subplot(235)
        plt.imshow(evaluation.gtSeg)
        plt.axis('off')
        plt.subplot(236)
        plt.imshow(evaluation.predictedSeg)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('/home/khietdang/Documents/khiet/treeRing/doc/fig_metrics_example.png', dpi=300)


