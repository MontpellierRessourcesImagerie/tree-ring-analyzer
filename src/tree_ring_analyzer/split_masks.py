import tifffile
import glob
import os
from sklearn.model_selection import train_test_split
import shutil
import multiprocessing
from multiprocessing import Pool
from tree_ring_analyzer.dl.preprocessing import savePith, createFolder, splitRingsAndPith, saveTile, saveTileHoles, saveTileHolesGaussian
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random


if __name__ == '__main__':
    input_path = '/home/khietdang/Documents/khiet/treeRing/input'
    masks_path = '/home/khietdang/Documents/khiet/treeRing/masks'
    big_dis_path = '/home/khietdang/Documents/khiet/treeRing/big_dis_otherrings'
    pith_path = '/home/khietdang/Documents/khiet/treeRing/pithWhole'
    tile_path = '/home/khietdang/Documents/khiet/treeRing/tile_big_dis_otherrings'
    seed = 42
    pithWhole = True

    random.seed(seed)
    np.random.seed(seed)

    masks_list = glob.glob(os.path.join(masks_path, '*.tif')) + glob.glob(os.path.join(masks_path, '*.jpg'))

    train, test = train_test_split(masks_list, test_size=0.2, shuffle=True, random_state=seed)
    train, val = train_test_split(train, test_size=0.2, shuffle=True, random_state=seed)

    # if not os.path.exists(big_dis_path):
    #     os.makedirs(big_dis_path)

    # if os.path.exists(pith_path):
    #     shutil.rmtree(pith_path)
    # createFolder(pith_path)

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

        pith, other_rings_dis = splitRingsAndPith(mask)
        thres = np.max(other_rings_dis)

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
            # savePith(mask_path, pith, image, 0, pith_path, save_type, False, pithWhole)
            # saveTile(mask_path, other_rings_dis, image, 0, tile_path, save_type, False, thres)
            saveTileHoles(mask_path, other_rings_dis, image, 0, tile_path, save_type, False, thres)
            # saveTileHolesGaussian(mask_path, other_rings_dis, image, 0, tile_path, save_type, False, thres)
        else:
            # data = []
            # for i in range(0, num):
            #     data.append((mask_path, pith, image, i, pith_path, save_type, True, pithWhole))

            # with Pool(int(multiprocessing.cpu_count())) as pool:
            #     pool.starmap(savePith, data)

            data = []
            for i in range(0, num):
                data.append((mask_path, other_rings_dis, image, i, tile_path, save_type, True, thres))

            with Pool(int(os.cpu_count() * 0.5)) as pool:
                # pool.starmap(saveTile, data)
                pool.starmap(saveTileHoles, data)
                # pool.starmap(saveTileHolesGaussian, data)

