from tree_ring_analyzer.segmentation import TreeRingSegmentation, Evaluation
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
    output_folder = '/home/khietdang/Documents/khiet/treeRing/output_H0RR'
    checkpoint_ring_path = '/home/khietdang/Documents/khiet/tree-ring-analyzer/models/bigDisRingAugGrayWH16.keras'
    checkpoint_pith_path = '/home/khietdang/Documents/khiet/tree-ring-analyzer/models/pithGrayNormal16.keras'
    csv_file = '/home/khietdang/Documents/khiet/treeRing/doc/result_our.csv'
    pithWhole = False
    rotate = True
    removeRing = True
    lossType = 'H0' # H0, H01, H02
    thickness = 1

    with open(csv_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Hausdorff Distance', 'mean Average Recall', 'ARAND', 'Recall', 'Precision', 'F1', 'Accuracy'])

    modelRing = tf.keras.models.load_model(checkpoint_ring_path, compile=False)

    modelPith = tf.keras.models.load_model(checkpoint_pith_path, compile=False)

    channel = modelPith.get_config()['layers'][0]['config']['batch_shape'][-1]

    image_list = glob.glob(os.path.join(input_folder, '*.tif')) + glob.glob(os.path.join(input_folder, '*.jpg'))
    _, image_list = train_test_split(image_list, test_size=0.2, shuffle=True, random_state=42)
    image_list = sorted(image_list)

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

        if channel == 1:
            image = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2])[:, :, None]

        treeRingSegment = TreeRingSegmentation(resize=5, pithWhole=pithWhole, rotate=rotate, lossType=lossType, removeRing=removeRing,
                                               thickness=thickness)
        treeRingSegment.segmentImage(modelRing, modelPith, image)
        
        result = treeRingSegment.maskRings
        image[result == 255] = 0

        tifffile.imwrite(os.path.join(output_folder, os.path.basename(image_path)), result.astype(np.uint8))
        evaluation = Evaluation(treeRingSegment.maskRings, treeRingSegment.maskRings)

        evaluation = Evaluation(mask, treeRingSegment.maskRings)
        hausdorff.append(evaluation.evaluateHausdorff())
        mAR.append(evaluation.evaluatemAR())
        arand.append(evaluation.evaluateARAND())
        _recall, _precision, _f1, _acc = evaluation.evaluateRPFA()
        recall.append(_recall)
        precision.append(_precision)
        f1.append(_f1)
        acc.append(_acc)

        with open(csv_file, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([os.path.basename(image_path).split('.')[0], hausdorff[-1], mAR[-1], arand[-1],
                             recall[-1], precision[-1], f1[-1], acc[-1]])

    with open(csv_file, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(['Average', np.mean(hausdorff), np.mean(mAR), np.mean(arand), np.mean(recall), np.mean(precision), np.mean(f1),
                         np.mean(acc)])

