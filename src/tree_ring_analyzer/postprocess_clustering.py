import tifffile
import cv2
from scipy.ndimage import binary_dilation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

if __name__ == '__main__':
    image = tifffile.imread('/home/khietdang/Documents/khiet/treeRing/predictions_segmentation_from_bigDistance/4 E 1 b_8µm_x50.tif')
    image = binary_dilation(image, iterations=30).astype(np.uint8)
    height, width = image.shape[0], image.shape[1]
    image = cv2.resize(image, (int(height / 10), int(height / 10)))

    one_indice = np.where(image == 1)
    center = np.mean(one_indice[0]), np.mean(one_indice[1])
    dis = (center[0] - one_indice[0]) ** 2 + (center[1] - one_indice[1]) ** 2

    # X = np.array(one_indice).transpose(1, 0)
    # X = np.concatenate([X, dis[:, None]], axis=1)
    X = dis[:, None]
    X = StandardScaler().fit_transform(X)
    
    model = DBSCAN(eps=0.2, min_samples=2).fit(X)

    label = model.labels_
    
    re = np.zeros_like(image)
    for i in range(0, 5):
        re[one_indice[0][label == i], one_indice[1][label == i]] = i + 1
        _center = np.mean(one_indice[0][label == i]), np.mean(one_indice[1][label == i])


    plt.imshow(re)
    plt.show()
