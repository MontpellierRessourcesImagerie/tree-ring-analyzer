import tifffile
import cv2
from scipy.ndimage import binary_dilation
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import copy
from skimage import measure
import os
import glob
import matplotlib.pyplot as plt

if __name__ == '__main__':
    image_list = glob.glob('/home/khietdang/Documents/khiet/treeRing/predictions_segmentation_from_bigDistance/*.tif')
    # image_list = [image_list[0]]
    # image_list = ['/home/khietdang/Documents/khiet/treeRing/predictions_segmentation_from_bigDistance/4 E 1 b_8µm_x50.tif']
    for image_path in image_list:
        image = tifffile.imread(image_path)
        image = binary_dilation(image, iterations=30).astype(np.uint8)
        height, width = image.shape[0], image.shape[1]
        image = cv2.resize(image, (512, 512))

        one_indice = np.where(image == 1)
        center = int(np.average(one_indice[1])), int(np.average(one_indice[0]))
        
        solid_circle = np.zeros_like(image)
        for ksize in [8, 16, 32, 64, 128]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize))  # Adjust size as needed

            mask1 = np.zeros_like(image)
            cv2.circle(mask1, center, ksize * 2, 1, -1)
            mask2 = np.zeros_like(image)
            cv2.circle(mask2, center, ksize * 4, 1, -1)
            mask = mask1 + (1 - mask2)

            solid_circle += cv2.morphologyEx(image * mask, cv2.MORPH_CLOSE, kernel)

        solid_circle = cv2.resize(solid_circle, (height, width))
        
        plt.imshow(solid_circle)
        plt.show()
        plt.close()
        
        tifffile.imwrite('/home/khietdang/Documents/khiet/treeRing/morphoOval_bigDistance/' + os.path.basename(image_path), solid_circle)
