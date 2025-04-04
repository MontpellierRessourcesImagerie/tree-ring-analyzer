import tifffile
import glob
import os
from sklearn.model_selection import train_test_split
import shutil
import multiprocessing
from multiprocessing import Pool
from tree_ring_analyzer.dl.preprocessing import savePith, createFolder, splitRingsAndPith, saveTile
import cv2


if __name__ == '__main__':
    input_path = '/home/khietdang/Documents/khiet/treeRing/input'
    masks_path = '/home/khietdang/Documents/khiet/treeRing/masks'
    big_dis_path = '/home/khietdang/Documents/khiet/treeRing/big_dis_otherrings'
    pith_path = '/home/khietdang/Documents/khiet/treeRing/pith'
    tile_path = '/home/khietdang/Documents/khiet/treeRing/tile_big_dis_otherrings'

    masks_list = glob.glob(os.path.join(masks_path, '*.tif'))

    train, test = train_test_split(masks_list, test_size=0.2, shuffle=True, random_state=42)
    train, val = train_test_split(train, test_size=0.2, shuffle=True, random_state=42)

    if not os.path.exists(big_dis_path):
        os.makedirs(big_dis_path)

    if os.path.exists(pith_path):
        shutil.rmtree(pith_path)
    createFolder(pith_path)

    if os.path.exists(tile_path):
        shutil.rmtree(tile_path)
    createFolder(tile_path)

    for mask_path in masks_list:
        mask = tifffile.imread(mask_path)
        image = tifffile.imread(os.path.join(input_path, os.path.basename(mask_path)))

        mask[mask == 255] = 1
        mask = cv2.resize(mask, (2560, int(2560 * mask.shape[0] / mask.shape[1])))
        image = cv2.resize(image, (2560, int(2560 * image.shape[0] / image.shape[1])))

        pith, other_rings_dis = splitRingsAndPith(mask)

        tifffile.imwrite(os.path.join(big_dis_path, os.path.basename(mask_path)), other_rings_dis)

        if mask_path in train:
            save_type = 'train'
            num = 500
        elif mask_path in test:
            save_type = 'test'
            num = 1
        else:
            save_type = 'val'
            num = 1

        if mask_path in test or mask_path in val:
            savePith(mask_path, pith, image, 0, pith_path, save_type, False)
            saveTile(mask_path, other_rings_dis, image, 0, tile_path, save_type, False)
        else:
            data = []
            for i in range(0, num):
                data.append((mask_path, pith, image, i, pith_path, save_type, True))

            with Pool(multiprocessing.cpu_count()) as pool:
                pool.starmap(savePith, data)

            data = []
            for i in range(0, num):
                data.append((mask_path, other_rings_dis, image, i, tile_path, save_type, True))

            with Pool(multiprocessing.cpu_count()) as pool:
                pool.starmap(saveTile, data)

