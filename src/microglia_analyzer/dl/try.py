import matplotlib.pyplot as plt
import tifffile
import numpy as np

img_name = '4 E 1 b_8µm_x50.tif'
val_images = tifffile.imread('/home/khietdang/Documents/khiet/treeRing/input/' + img_name)
val_masks = tifffile.imread('/home/khietdang/Documents/khiet/treeRing/masks/' + img_name)
predictions = tifffile.imread('/home/khietdang/Documents/khiet/treeRing/predictions/' + img_name)
print(val_images.shape, val_masks.shape, predictions.shape)
minv = np.min(val_images, axis=(0, 1))[None, None, :]
maxv = np.max(val_images, axis=(0, 1))[None, None, :]
print(minv.shape, maxv.shape)
val_images = (val_images - minv) / (maxv - minv)
plt.subplot(1, 3, 1)
plt.imshow(val_images)
plt.subplot(1, 3, 2)
plt.imshow(val_masks)
plt.subplot(1, 3, 3)
plt.imshow(predictions)
plt.show()