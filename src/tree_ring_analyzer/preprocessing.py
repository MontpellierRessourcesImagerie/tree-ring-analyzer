import tifffile
import glob
import os
from sklearn.model_selection import train_test_split
import shutil
import multiprocessing
from multiprocessing import Pool
from tree_ring_analyzer.dl.preprocessing import savePith, createFolder, splitRingsAndPith, saveTile
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random


if __name__ == '__main__':
    input_path = '/home/khietdang/Documents/khiet/treeRing/input'
    masks_path = '/home/khietdang/Documents/khiet/treeRing/masks'
    pith_path = None
    tile_path = '/home/khietdang/Documents/khiet/treeRing/tile_big_dis_otherrings'
    seed = 42
    pithWhole = False
    whiteHoles = True
    gaussianHoles = False
    changeColor = False
    dilate = 10
    distance = True
    skeleton = False

    random.seed(seed)
    np.random.seed(seed)

    masks_list = glob.glob(os.path.join(masks_path, '*.tif')) + glob.glob(os.path.join(masks_path, '*.jpg'))

    train, test = train_test_split(masks_list, test_size=0.2, shuffle=True, random_state=seed)
    train, val = train_test_split(train, test_size=0.2, shuffle=True, random_state=seed)

    if pith_path is not None:
        if os.path.exists(pith_path):
            shutil.rmtree(pith_path)
        createFolder(pith_path)

    if tile_path is not None:
        if os.path.exists(tile_path):
            shutil.rmtree(tile_path)
        createFolder(tile_path)

    for mask_path in masks_list:
        print(mask_path)
        if mask_path.endswith('.tif'):
            mask = tifffile.imread(mask_path)
            image = tifffile.imread(os.path.join(input_path, os.path.basename(mask_path)))
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(os.path.join(input_path, os.path.basename(mask_path)))

        mask[mask == 255] = 1

        pith, other_rings_dis = splitRingsAndPith(mask, dilate, distance, skeleton)
        thres = np.max(other_rings_dis)

        if mask_path in train:
            save_type = 'train'
            num = 100
        elif mask_path in test:
            save_type = 'test'
            num = 1
        else:
            save_type = 'val'
            num = 1

        if mask_path in test or mask_path in val:
            if pith_path is not None:
                savePith(mask_path, pith, image, 0, pith_path, save_type, False, pithWhole)
            if tile_path is not None:
                saveTile(mask_path, other_rings_dis, image, 0, tile_path, save_type, False, whiteHoles, gaussianHoles, changeColor, thres)
        else:
            if pith_path is not None:
                data = []
                for i in range(0, num):
                    data.append((mask_path, pith, image, i, pith_path, save_type, True, pithWhole))

                with Pool(int(multiprocessing.cpu_count())) as pool:
                    pool.starmap(savePith, data)
            
            if tile_path is not None:
                data = []
                for i in range(0, num):
                    data.append((mask_path, other_rings_dis, image, i, tile_path, save_type, True, whiteHoles, gaussianHoles, changeColor, thres))

                with Pool(int(multiprocessing.cpu_count() * 0.5)) as pool:
                    pool.starmap(saveTile, data)

