import tifffile
import glob
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_dilation, rotate
import os
from sklearn.model_selection import train_test_split
import shutil
import random
import multiprocessing
from multiprocessing import Pool

def measure_distance(points):
    x1 = points[:, 0][:, None]
    x2 = points[:, 0][None, :]
    y1 = points[:, 1][:, None]
    y2 = points[:, 1][None, :]
    return (x1 - x2) ** 2 + (y1 - y2) ** 2

def augmentation(mask_path, pith, image, i, crop_size):
    pith_aug = copy.deepcopy(pith)
    img_aug = copy.deepcopy(image)
    if random.random() > 0.5:
        angle = np.random.randint(-20, 20)
        pith_aug = rotate(pith_aug, angle, order=0, reshape=False)
        img_aug = rotate(img_aug, angle, order=0, reshape=False)
    if random.random() > 0.5:
        k = np.random.randint(0, 4)
        axis = np.random.randint(0, 1)
        pith_aug = np.rot90(pith_aug, k=k)
        pith_aug = np.flip(pith_aug, axis=axis)
        img_aug = np.rot90(img_aug, k=k)
        img_aug = np.flip(img_aug, axis=axis)

    one_indices = np.where(pith_aug == 1)
    center = np.mean(one_indices[0]), np.mean(one_indices[1])

    xStart = int(center[0] - crop_size / 2) + np.random.randint(-int(0.375 * crop_size), int(0.375 * crop_size))
    yStart = int(center[1] - crop_size / 2) + np.random.randint(-int(0.375 * crop_size), int(0.375 * crop_size))

    pith_crop = pith_aug[xStart:xStart + crop_size,
                    yStart:yStart + crop_size]
    
    if np.sum(pith_crop) != 0:
        img_crop = img_aug[xStart:xStart + crop_size,
                        yStart:yStart + crop_size]
        
        pith_crop = cv2.resize(pith_crop.astype(np.uint8), (256, 256))[:, :, None]
        img_crop = cv2.resize(img_crop.astype(np.uint8), (256, 256))
        pith_crop[pith_crop != 0] = 255

        tifffile.imwrite(os.path.join(f'/home/khietdang/Documents/khiet/treeRing/pith/train/x', 
                                        os.path.basename(mask_path)[:-4] + f'_aug{i}.tif'),
                                        img_crop.astype(np.uint8))
        tifffile.imwrite(os.path.join(f'/home/khietdang/Documents/khiet/treeRing/pith/train/y', 
                                    os.path.basename(mask_path)[:-4] + f'_aug{i}.tif'),
                                    pith_crop.astype(np.uint8))

if __name__ == '__main__':
    masks_list = glob.glob('/home/khietdang/Documents/khiet/treeRing/masks/*.tif')
    # masks_list = ['/home/khietdang/Documents/khiet/treeRing/masks/T 5 b_8µm_x50.tif']
    train, test = train_test_split(masks_list, test_size=0.2, shuffle=True, random_state=42)
    train, val = train_test_split(train, test_size=0.2, shuffle=True, random_state=42)

    crop_size = 1024

    if os.path.exists('/home/khietdang/Documents/khiet/treeRing/pith'):
        shutil.rmtree('/home/khietdang/Documents/khiet/treeRing/pith')
    os.makedirs('/home/khietdang/Documents/khiet/treeRing/pith/train/x', exist_ok=True)
    os.makedirs('/home/khietdang/Documents/khiet/treeRing/pith/train/y', exist_ok=True)
    os.makedirs('/home/khietdang/Documents/khiet/treeRing/pith/val/x', exist_ok=True)
    os.makedirs('/home/khietdang/Documents/khiet/treeRing/pith/val/y', exist_ok=True)
    os.makedirs('/home/khietdang/Documents/khiet/treeRing/pith/test/x', exist_ok=True)
    os.makedirs('/home/khietdang/Documents/khiet/treeRing/pith/test/y', exist_ok=True)

    for mask_path in masks_list:
        mask = tifffile.imread(mask_path)
        image = tifffile.imread(mask_path.replace('masks', 'input'))

        num_labels, labels = cv2.connectedComponents(mask)

        area = []
        for i in range(1, np.max(labels) + 1):
            _labels = np.zeros_like(labels)
            _labels[labels == i] = 1

            _area = np.sum(_labels)
            if _area < 10:
                area.append(image.shape[0] * image.shape[1])
            else:
                area.append(_area)

        area = np.array(area)
        sort_label = np.argsort(area)
        pith_label = sort_label[0] + 1

        pith = np.zeros_like(labels)
        pith[labels == pith_label] = 1
        contours, _ = cv2.findContours(pith.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(pith, contours, 0, color=1, thickness=-1)

        other_rings = copy.deepcopy(mask)
        other_rings[pith == 1] = 0
        other_rings[other_rings == 255] = 1
        other_rings = binary_dilation(other_rings, iterations=10)
        other_rings_dis = distance_transform_edt(other_rings).astype(np.float16)

        # plt.subplot(1, 3, 1)
        # plt.imshow(mask)
        # plt.subplot(1, 3, 2)
        # plt.imshow(other_rings)
        # plt.subplot(1, 3, 3)
        # plt.imshow(pith)
        # plt.show()
        # raise ValueError

        tifffile.imwrite(os.path.join('/home/khietdang/Documents/khiet/treeRing/big_dis_otherrings', os.path.basename(mask_path)),
                         other_rings_dis)

        if mask_path in train:
            save_type = 'train'
            num = 300
        elif mask_path in test:
            save_type = 'test'
            num = 1
        else:
            save_type = 'val'
            num = 1

        if mask_path in test or mask_path in val:
            one_indices = np.where(pith == 1)
            center = np.mean(one_indices[0]), np.mean(one_indices[1])

            xStart = int(center[0] - crop_size / 2)
            yStart = int(center[1] - crop_size / 2)

            pith_crop = pith[xStart:xStart + crop_size,
                            yStart:yStart + crop_size]
            
            if np.sum(pith_crop) != 0:
                img_crop = image[xStart:xStart + crop_size,
                                yStart:yStart + crop_size]
                
                pith_crop = cv2.resize(pith_crop.astype(np.uint8), (256, 256))[:, :, None]
                img_crop = cv2.resize(img_crop.astype(np.uint8), (256, 256))
                pith_crop[pith_crop != 0] = 255

                tifffile.imwrite(os.path.join(f'/home/khietdang/Documents/khiet/treeRing/pith/{save_type}/x', 
                                                os.path.basename(mask_path)),
                                                img_crop.astype(np.uint8))
                tifffile.imwrite(os.path.join(f'/home/khietdang/Documents/khiet/treeRing/pith/{save_type}/y', 
                                            os.path.basename(mask_path)),
                                            pith_crop.astype(np.uint8))
        else:
            data = []
            for i in range(0, num):
                data.append((mask_path, pith, image, i, crop_size))

            with Pool(multiprocessing.cpu_count()) as pool:
                pool.starmap(augmentation, data)



        