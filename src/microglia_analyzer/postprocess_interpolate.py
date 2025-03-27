import tifffile
import cv2
from scipy.ndimage import binary_dilation
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

if __name__ == '__main__':
    pred_seg = tifffile.imread('/home/khietdang/Documents/khiet/treeRing/predictions_segmentation_from_bigDistance/4 E 1 m_8µm_x50.tif')
    # pred = tifffile.imread('/home/khietdang/Documents/khiet/treeRing/predictions_bigDistance/4 E 1 m_8µm_x50.tif')

    image = binary_dilation(pred_seg, iterations=30).astype(np.uint8)
    height, width = image.shape[0], image.shape[1]
    image = cv2.resize(image, (int(height / 10), int(width / 10)))
    # pred = cv2.resize(pred, (int(height / 10), int(width / 10)))

    one_indice = np.where(image == 1)
    one_choose = np.random.randint(0, len(one_indice[0]), int(len(one_indice[0])))
    one_indice = (one_indice[0][one_choose], one_indice[1][one_choose])

    zero_indice = np.where(image == 0)
    zero_choose = np.random.randint(0, len(zero_indice[0]), len(one_indice[0]))
    zero_indice = (zero_indice[0][zero_choose], zero_indice[1][zero_choose])

    px = np.concatenate((one_indice[0], zero_indice[0]))
    py = np.concatenate((one_indice[1], zero_indice[1]))

    x = np.arange(0, image.shape[0])
    y =  np.arange(0, image.shape[1])
    X, Y = np.meshgrid(x,y)

    T = griddata((px, py), image[px, py], (X, Y), method='linear').T

    plt.subplot(1, 2, 1)
    plt.imshow(image, 'gray')
    plt.subplot(1, 2, 2)
    plt.imshow(T, 'gray')
    plt.show()
