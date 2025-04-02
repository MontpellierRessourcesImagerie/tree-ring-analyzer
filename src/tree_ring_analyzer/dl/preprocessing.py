from scipy.ndimage import distance_transform_edt, binary_dilation
import tifffile
import numpy as np
import os
import copy
import random
from scipy.ndimage import distance_transform_edt, binary_dilation, rotate
import cv2
from tree_ring_analyzer.tiles.tiler import ImageTiler2D



def augmentImages(img, mask):
    if random.random() > 0.5:
        angle = np.random.randint(-20, 20)
        img = rotate(img, angle, order=0, reshape=False)
        mask = rotate(mask, angle, order=0, reshape=False)
    if random.random() > 0.5:
        k = np.random.randint(0, 4)
        axis = np.random.randint(0, 1)
        img = np.rot90(img, k=k)
        img = np.flip(img, axis=axis)
        mask = np.rot90(mask, k=k)
        mask = np.flip(mask, axis=axis)

    return img, mask



def savePith(mask_path, pith, image, i, output_path, save_type, augment=True):
    crop_size = int(0.1 * max(image.shape)) * 2
    pith_aug = copy.deepcopy(pith)
    img_aug = copy.deepcopy(image)
    if augment:
        img_aug, pith_aug = augmentImages(img_aug, pith_aug)

        one_indices = np.where(pith_aug == 1)
        center = np.mean(one_indices[0]), np.mean(one_indices[1])

        xStart = int(center[0] - crop_size / 2) + np.random.randint(-int(0.375 * crop_size), int(0.375 * crop_size))
        yStart = int(center[1] - crop_size / 2) + np.random.randint(-int(0.375 * crop_size), int(0.375 * crop_size))
    else:
        one_indices = np.where(pith_aug == 1)
        center = np.mean(one_indices[0]), np.mean(one_indices[1])

        xStart = int(center[0] - crop_size / 2)
        yStart = int(center[1] - crop_size / 2)

    pith_crop = pith_aug[xStart:xStart + crop_size, yStart:yStart + crop_size]
    
    if np.sum(pith_crop) != 0:
        img_crop = img_aug[xStart:xStart + crop_size, yStart:yStart + crop_size]
        
        pith_crop = cv2.resize(pith_crop.astype(np.uint8), (256, 256))[:, :, None]
        img_crop = cv2.resize(img_crop.astype(np.uint8), (256, 256))
        pith_crop[pith_crop != 0] = 255

        tifffile.imwrite(os.path.join(output_path, save_type, 'x', os.path.basename(mask_path)[:-4] + f'_aug{i}.tif'),
                         img_crop.astype(np.uint8))
        tifffile.imwrite(os.path.join(output_path, save_type, 'y', os.path.basename(mask_path)[:-4] + f'_aug{i}.tif'),
                         pith_crop.astype(np.uint8))
        


def saveTile(mask_path, mask, image, i, output_path, save_type, augment=True):
    mask_aug = copy.deepcopy(mask)
    img_aug = copy.deepcopy(image)
    if augment:
        img_aug, mask_aug = augmentImages(img_aug, mask_aug)

    tiles_manager = ImageTiler2D(256, 60, mask_aug.shape)
    img_tiles = np.array(tiles_manager.image_to_tiles(img_aug, use_normalize=False))
    mask_tiles = np.array(tiles_manager.image_to_tiles(mask_aug, use_normalize=False))
    
    for j in range(0, len(img_tiles)):
        mask_tile = mask_tiles[j]
        if np.max(mask_tile) >= 12:
            img_tile = img_tiles[j]

            tifffile.imwrite(os.path.join(output_path, save_type, 'x', os.path.basename(mask_path)[:-4] + f'_aug{i}_{j}.tif'),
                            img_tile.astype(np.uint8))
            tifffile.imwrite(os.path.join(output_path, save_type, 'y', os.path.basename(mask_path)[:-4] + f'_aug{i}_{j}.tif'),
                            mask_tile)



def createFolder(path):
    os.makedirs(os.path.join(path, 'train/x'), exist_ok=True)
    os.makedirs(os.path.join(path, 'train/y'), exist_ok=True)
    os.makedirs(os.path.join(path, 'val/x'), exist_ok=True)
    os.makedirs(os.path.join(path, 'val/y'), exist_ok=True)
    os.makedirs(os.path.join(path, 'test/x'), exist_ok=True)
    os.makedirs(os.path.join(path, 'test/y'), exist_ok=True)



def splitRingsAndPith(mask, iterations=10):
    _, labels = cv2.connectedComponents(mask)

    area = []
    for i in range(1, np.max(labels) + 1):
        _labels = np.zeros_like(labels)
        _labels[labels == i] = 1

        _area = np.sum(_labels)
        if _area < 10:
            area.append(mask.shape[0] * mask.shape[1])
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
    other_rings = binary_dilation(other_rings, iterations=iterations)
    other_rings_dis = distance_transform_edt(other_rings).astype(np.float16)

    return pith, other_rings_dis.astype(np.uint8)

