from scipy.ndimage import distance_transform_edt, binary_dilation
import tifffile
import numpy as np
import os


def createDistanceMap(list_mask, output_folder, iterations=10):
    for mask_path in list_mask:
        mask = tifffile.imread(mask_path)
        mask = mask / 255
        mask = binary_dilation(mask, iterations=iterations)
        dis = distance_transform_edt(mask).astype(np.float16)
        
        tifffile.imwrite(os.path.join(output_folder, os.path.basename(mask_path)), dis)