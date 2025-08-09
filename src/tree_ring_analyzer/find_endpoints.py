from tree_ring_analyzer.dl.postprocessing import endpoints
import glob
import os
from sklearn.model_selection import train_test_split
import tifffile
from tree_ring_analyzer.segmentation import Evaluation
import csv
import numpy as np



ring_path = '/home/khietdang/Documents/khiet/treeRing/MO_bigDisRingAugGrayWH16'
pith_path = '/home/khietdang/Documents/khiet/treeRing/predictions_pithGrayNormal16'
output_path = '/home/khietdang/Documents/khiet/treeRing/output_MOE'
mask_path = '/home/khietdang/Documents/khiet/treeRing/masks'
csv_file = '/home/khietdang/Documents/khiet/treeRing/doc/result_MOE.csv'

# with open(csv_file, 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Image', 'Hausdorff Distance', 'mean Average Recall', 'ARAND', 'Recall', 'Precision', 'F1', 'Accuracy'])

# image_list = glob.glob(os.path.join(ring_path, '*.tif'))
# _, image_list = train_test_split(image_list, test_size=0.2, random_state=42, shuffle=True)
# image_list = sorted(image_list)
image_list = [os.path.join(ring_path, '4 E 4 m_8µm_x50.tif')]

hausdorff = []
mAR = []
arand = []
recall = []
precision = []
f1 = []
acc = []

for image_path in image_list:
    print(image_path)
    prediction = endpoints(image_path, pith_path, output_path)
    mask = tifffile.imread(os.path.join(mask_path, os.path.basename(image_path)))

    evaluation = Evaluation(mask, prediction)
    tifffile.imwrite(os.path.join('/home/khietdang/Documents/khiet/treeRing/predictedSeg_MOE', os.path.basename(image_path)), 
                     evaluation.predictedSeg.astype(np.uint8))
    tifffile.imwrite(os.path.join('/home/khietdang/Documents/khiet/treeRing/maskSeg', os.path.basename(image_path)), 
                     evaluation.gtSeg.astype(np.uint8))

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



