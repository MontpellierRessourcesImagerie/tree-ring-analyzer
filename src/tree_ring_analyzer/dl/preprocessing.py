import tifffile
import numpy as np
import os
import copy
import random
from scipy.ndimage import distance_transform_edt, binary_dilation, rotate, gaussian_filter
import cv2
from tree_ring_analyzer.tiles.tiler import ImageTiler2D
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize



def augmentImagesRotate(img, mask):
    angle = np.random.randint(-20, 20)
    img = rotate(img, angle, order=0, reshape=False)
    mask = rotate(mask, angle, order=0, reshape=False)

    return img, mask


def augmentImagesFlip(img, mask):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 1)
    img = np.rot90(img, k=k)
    img = np.flip(img, axis=axis)
    mask = np.rot90(mask, k=k)
    mask = np.flip(mask, axis=axis)

    return img, mask


def augmentImagesHolesWhite(img, mask):
    maskHoles = copy.deepcopy(mask)
    num = np.random.randint(1, 11)
    one_indices = np.where(mask >= 1)
    img = np.ascontiguousarray(img)
    maskHoles = np.ascontiguousarray(maskHoles)
    for i in range(num):
        radius = np.random.randint(1, 128)
        chose_center = np.random.randint(len(one_indices[0]))
        center = one_indices[0][chose_center], one_indices[1][chose_center]
        cv2.circle(img, (center[1], center[0]), radius, (255, 255, 255), -1)
        cv2.circle(maskHoles, (center[1], center[0]), radius, 0, -1)
    
    return img, mask, maskHoles


def augmentImagesHolesGaussian(img, mask):
    maskHoles = copy.deepcopy(mask)
    new_image = np.zeros_like(mask, dtype=np.float32)
    num = np.random.randint(1, 11)
    one_indices = np.where(mask >= 1)
    maskHoles = np.ascontiguousarray(maskHoles)
    for i in range(num):
        radius = np.random.randint(1, 128)
        chose_center = np.random.randint(len(one_indices[0]))
        center = one_indices[0][chose_center], one_indices[1][chose_center]

        _new_image = np.ascontiguousarray(np.zeros_like(mask, dtype=np.float32))
        cv2.circle(_new_image, (center[1], center[0]), radius, -0.5, -1)
        cv2.circle(maskHoles, (center[1], center[0]), radius, 0, -1)

        _new_image = gaussian_filter(_new_image, sigma=radius*random.random())
        new_image += _new_image
        
        new_image += 1
        img = img * new_image[:, :, None]
    
    return img.astype(np.uint8), mask, maskHoles
        

def savePith(mask_path, pith, image, i, output_path, save_type, augment=True, pithWhole=False):
    crop_size = int(0.1 * image.shape[0]) * 2
    pith_aug = copy.deepcopy(pith)
    img_aug = copy.deepcopy(image)
 
    if augment:
        if random.random() > 0.5:
            img_aug, pith_aug = augmentImagesRotate(img_aug, pith_aug)
        if random.random() > 0.5:
            img_aug, pith_aug = augmentImagesFlip(img_aug, pith_aug)
        one_indices = np.where(pith_aug == 1)
        center = np.mean(one_indices[0]), np.mean(one_indices[1])
        xStart = int(center[0] - crop_size / 2) + np.random.randint(-int(0.375 * crop_size), int(0.375 * crop_size))
        yStart = int(center[1] - crop_size / 2) + np.random.randint(-int(0.375 * crop_size), int(0.375 * crop_size))

    else:
        one_indices = np.where(pith_aug == 1)
        center = np.mean(one_indices[0]), np.mean(one_indices[1])
        xStart = int(center[0] - crop_size / 2)
        yStart = int(center[1] - crop_size / 2)
    
    if pithWhole:
        pith_aug = cv2.resize(pith_aug.astype(np.uint8), (256, 256))[:, :, None]
        img_aug = cv2.resize(img_aug.astype(np.uint8), (256, 256))
    
        tifffile.imwrite(os.path.join(output_path, save_type, 'x', os.path.basename(mask_path)[:-4] + f'_aug{i}.tif'),
                            img_aug / 255)
    
        tifffile.imwrite(os.path.join(output_path, save_type, 'y', os.path.basename(mask_path)[:-4] + f'_aug{i}.tif'),
                        pith_aug.astype(np.uint8))
    else:
        pith_crop = pith_aug[xStart:xStart + crop_size, yStart:yStart + crop_size]
    
        if np.sum(pith_crop) != 0:
            img_crop = img_aug[xStart:xStart + crop_size, yStart:yStart + crop_size]
            pith_crop = cv2.resize(pith_crop.astype(np.uint8), (256, 256))[:, :, None]
            img_crop = cv2.resize(img_crop.astype(np.uint8), (256, 256))
    
            tifffile.imwrite(os.path.join(output_path, save_type, 'x', os.path.basename(mask_path)[:-4] + f'_aug{i}.tif'),
                            img_crop / 255)
    
            tifffile.imwrite(os.path.join(output_path, save_type, 'y', os.path.basename(mask_path)[:-4] + f'_aug{i}.tif'),
                            pith_crop.astype(np.uint8))
        

def saveTile(mask_path, mask, image, i, output_path, save_type, augment=True, whiteHole=True, gauHole=False, changeColor=False, thres=10):
    mask_aug = copy.deepcopy(mask)
    img_aug = copy.deepcopy(image)
    if augment:
        if random.random() > 0.5:
            img_aug, mask_aug = augmentImagesRotate(img_aug, mask_aug)
        if random.random() > 0.5:
            img_aug, mask_aug = augmentImagesFlip(img_aug, mask_aug)
        if whiteHole and random.random() > 0.5:
            img_aug, mask_aug, maskHolesWhite = augmentImagesHolesWhite(img_aug, mask_aug)
        else:
            maskHolesWhite = copy.deepcopy(mask_aug)
        if gauHole and random.random() > 0.5:
            img_aug, mask_aug, maskHolesGaussian = augmentImagesHolesGaussian(img_aug, mask_aug)
        else:
            maskHolesGaussian = copy.deepcopy(mask_aug)
        maskHoles = maskHolesWhite * maskHolesGaussian
        if changeColor and random.random() > 0.5:
            channelOrder = np.arange(0, 3)
            np.random.shuffle(channelOrder)
            img_aug = img_aug[:, :, channelOrder]
    else:
        maskHoles = copy.deepcopy(mask_aug)

    tiles_manager = ImageTiler2D(256, 60, mask_aug.shape)
    img_tiles = np.array(tiles_manager.image_to_tiles(img_aug, use_normalize=True))
    mask_tiles = np.array(tiles_manager.image_to_tiles(mask_aug, use_normalize=False))
    maskHole_tiles = np.array(tiles_manager.image_to_tiles(maskHoles, use_normalize=False))
    
    for j in range(0, len(img_tiles)):
        maskHole_tile = maskHole_tiles[j]
        if np.max(maskHole_tile) >= thres and np.sum(maskHole_tile) >= 10:
            img_tile = img_tiles[j]
            mask_tile = mask_tiles[j]

            tifffile.imwrite(os.path.join(output_path, save_type, 'x', os.path.basename(mask_path)[:-4] + f'_aug{i}_{j}.tif'),
                            img_tile)
            tifffile.imwrite(os.path.join(output_path, save_type, 'y', os.path.basename(mask_path)[:-4] + f'_aug{i}_{j}.tif'),
                            mask_tile.astype(np.uint8))


def createFolder(path):
    os.makedirs(os.path.join(path, 'train/x'), exist_ok=True)
    os.makedirs(os.path.join(path, 'train/y'), exist_ok=True)
    os.makedirs(os.path.join(path, 'val/x'), exist_ok=True)
    os.makedirs(os.path.join(path, 'val/y'), exist_ok=True)
    os.makedirs(os.path.join(path, 'test/x'), exist_ok=True)
    os.makedirs(os.path.join(path, 'test/y'), exist_ok=True)


def splitRingsAndPith(mask, iterations=10, distance=True, skeleton=False):
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
    if skeleton:
        other_rings = skeletonize(other_rings)
    if iterations:
        other_rings = binary_dilation(other_rings, iterations=iterations)
    if distance:
        other_rings = distance_transform_edt(other_rings).astype(np.float16)

    return pith, other_rings.astype(np.uint8)

