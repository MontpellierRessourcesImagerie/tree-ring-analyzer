from tree_ring_analyzer.dl.postprocessing import endpoints
import glob
import os



ring_path = '/home/khietdang/Documents/khiet/treeRing/MO_bigDisRingAugGrayNormal16'
pith_path = '/home/khietdang/Documents/khiet/treeRing/predictions_pithGrayNormal16'
output_path = '/home/khietdang/Documents/khiet/treeRing/output9'
input_path = '/home/khietdang/Documents/khiet/treeRing/input'

image_list = glob.glob(os.path.join(ring_path, '*.tif'))
for image_path in image_list:
    endpoints(image_path, pith_path, output_path)
