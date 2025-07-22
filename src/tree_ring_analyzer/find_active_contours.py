from tree_ring_analyzer.dl.postprocessing import activeContour
import glob
import os
from sklearn.model_selection import train_test_split
import tifffile
from tree_ring_analyzer.segmentation import Evaluation
import csv
import numpy as np



ring_path = '/home/khietdang/Documents/khiet/treeRing/predictions_bigDisRingAugGrayWH16'
pith_path = '/home/khietdang/Documents/khiet/treeRing/predictions_pithGrayNormal16'
output_path = '/home/khietdang/Documents/khiet/treeRing/output_A'
mask_path = '/home/khietdang/Documents/khiet/treeRing/masks'
csv_file = '/home/khietdang/Documents/khiet/treeRing/doc/result_A.csv'
thickness = 1

# with open(csv_file, 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Image', 'Hausdorff Distance', 'mean Average Recall', 'ARAND', 'Recall', 'Precision', 'F1', 'Accuracy'])

# image_list = glob.glob(os.path.join(ring_path, '*.tif'))
# _, image_list = train_test_split(image_list, test_size=0.2, random_state=42, shuffle=True)
# image_list = sorted(image_list)
image_list = [os.path.join(ring_path, '12 E 3 t_8µm_x50.tif')]

hausdorff = []
mAR = []
arand = []
recall = []
precision = []
f1 = []
acc = []

for image_path in image_list:
    print(image_path)
    prediction, predictedRings = activeContour(image_path, pith_path, output_path, thickness)
    mask = tifffile.imread(os.path.join(mask_path, os.path.basename(image_path)))

    # evaluation = Evaluation(mask, prediction, predictedRings=predictedRings)
    # tifffile.imwrite(os.path.join('/home/khietdang/Documents/khiet/treeRing/predictedSeg_A', 
    #     os.path.basename(image_path)), evaluation.predictedSeg)
#     hausdorff.append(evaluation.evaluateHausdorff())
#     mAR.append(evaluation.evaluatemAR())
#     arand.append(evaluation.evaluateARAND())
#     _recall, _precision, _f1, _acc = evaluation.evaluateRPFA()
#     recall.append(_recall)
#     precision.append(_precision)
#     f1.append(_f1)
#     acc.append(_acc)

#     with open(csv_file, 'a') as file:
#         writer = csv.writer(file)
#         writer.writerow([os.path.basename(image_path).split('.')[0], hausdorff[-1], mAR[-1], arand[-1],
#                             recall[-1], precision[-1], f1[-1], acc[-1]])

# with open(csv_file, 'a') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Average', np.mean(hausdorff), np.mean(mAR), np.mean(arand), np.mean(recall), np.mean(precision), np.mean(f1),
#                         np.mean(acc)])



