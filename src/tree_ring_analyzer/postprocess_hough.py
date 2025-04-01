from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.draw import circle_perimeter, ellipse_perimeter
import tifffile
import cv2
from scipy.ndimage import binary_dilation
import numpy as np
import matplotlib.pyplot as plt
import copy
import glob
import os
if __name__ == '__main__':
    image_list = glob.glob('/home/khietdang/Documents/khiet/treeRing/predictions_segmentation_from_bigDistance/*.tif')
    # image_list = [image_list[0]]
    # image_list = ['/home/khietdang/Documents/khiet/treeRing/predictions_segmentation_from_bigDistance/4 E 1 b_8µm_x50.tif']
    for image_path in image_list:
        image = tifffile.imread(image_path)
        image = binary_dilation(image, iterations=30).astype(np.uint8)
        height, width = image.shape[0], image.shape[1]
        image = cv2.resize(image, (int(width / 10), int(height / 10)))

        hough_radii = np.arange(int(width / 300), int(width / 30), 5)
        hough_res = hough_circle(image, hough_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=100, num_peaks=100)
        hough_image_circle = np.zeros_like(image)
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
            hough_image_circle[circy, circx] = 255

        
        # result = hough_ellipse(image, accuracy=20, threshold=250, min_size=100, max_size=120)
        # result.sort(order='accumulator')

        # # Estimated parameters for the ellipse
        # best = list(result[-1])
        # yc, xc, a, b = (int(round(x)) for x in best[1:5])
        # orientation = best[5]
        # cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        # hough_image_ellipse = np.zeros_like(image)
        # hough_image_ellipse[cy, cx] = 1

        # plt.subplot(1, 2, 1)
        # plt.imshow(image)
        # plt.subplot(1, 2, 2)
        # plt.imshow(hough_image_circle)
        # plt.show()
        # plt.close()
        hough_image_circle = cv2.resize(hough_image_circle, (width, height))

        tifffile.imwrite('/home/khietdang/Documents/khiet/treeRing/hough_bigDistance/' + os.path.basename(image_path), hough_image_circle)