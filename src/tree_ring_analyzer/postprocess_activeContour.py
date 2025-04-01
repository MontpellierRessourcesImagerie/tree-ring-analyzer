
from skimage.segmentation import active_contour, morphological_geodesic_active_contour
from skimage.morphology import skeletonize
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import binary_dilation
import copy
import glob

if __name__ == '__main__':
    image_list = glob.glob('/home/khietdang/Documents/khiet/treeRing/predictions_segmentation_from_bigDistance/*.tif')
    for image_path in image_list:
        image = tifffile.imread(image_path)
        image = binary_dilation(image, iterations=30).astype(np.uint8)

        image = cv2.resize(image, (256, 256))
        image[image==1] = 255
        one_indice = np.where(image == 255)

        center = np.mean(one_indice[0]), np.mean(one_indice[1])
        radius = 120
        i = 1
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        while True:
            s = np.linspace(0, 2 * np.pi, int(np.pi * radius * 2))
            r = center[1] + radius * np.sin(s)
            c = center[0] + radius * np.cos(s)
            init = np.array([r, c]).T
            snake = active_contour(
                image,
                init,
                alpha=radius / 100,
                beta=0.5,
                gamma=0.001,
                max_num_iter=1000,
                boundary_condition='periodic'
            )

            radius = np.mean(np.sqrt((center[0] - snake[:, 0]) ** 2 + (center[1] - snake[:, 1]) ** 2))

            s = np.linspace(0, 2 * np.pi, int(np.pi * radius * 2))
            r = center[1] + radius * np.sin(s)
            c = center[0] + radius * np.cos(s)
            init = np.array([r, c]).T
            snake = active_contour(
                image,
                init,
                alpha=radius / 100,
                beta=10,
                gamma=0.001,
                max_num_iter=100,
                boundary_condition='periodic'
            )

            plt.plot(snake[:, 1], snake[:, 0])
            radius = np.min(np.sqrt((center[0] - snake[:, 0]) ** 2 + (center[1] - snake[:, 1]) ** 2)) - 10
            
            if radius <= 5 or i >= 5:
                break

            i += 1
        plt.savefig(image_path.replace('predictions_segmentation_from_bigDistance', 'active_contour_bigDistance'), format='tiff')