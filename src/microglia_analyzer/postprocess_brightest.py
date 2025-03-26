from brightest_path_lib.algorithm import AStarSearch
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.signal import find_peaks
import cv2
import multiprocessing
from multiprocessing import Pool
import os
import glob
from scipy.ndimage import binary_dilation, median_filter
from skimage.filters import threshold_otsu

def plot_half_ring(image, peak1, peak2, dark_point, light_point):
    start_point = np.array([light_point, peak1])
    goal_point = np.array([light_point, peak2])
    search_algorithm = AStarSearch(image, start_point=start_point, goal_point=goal_point)
    brightest_path =search_algorithm.search()

    try:
        result = np.array(search_algorithm.result)[:, 1]
    except:
        print('abc')
    brightest_path = np.array(search_algorithm.result)[:, 0]

    result = np.append(start_point[1], np.append(result, goal_point[1]))
    brightest_path = np.append(dark_point, np.append(brightest_path, dark_point))

    cor = np.stack([result[:, None], brightest_path[:, None]], axis=-1)
    return cor


if __name__ == '__main__':
    folder_name = 'predictions_bigDistance'
    image_list = glob.glob(f'/home/khietdang/Documents/khiet/treeRing/{folder_name}/*.tif')
    # image_list = [f'/home/khietdang/Documents/khiet/treeRing/{folder_name}/1E_4milieu8microns_x40.tif']
    for image_path in image_list:
        print(image_path)
        image = tifffile.imread(image_path)
        image = cv2.resize(image, (int(image.shape[1] / 10), int(image.shape[0] / 10)))
        height, width = image.shape
        # image = median_filter(image, (3, 3))

        pith_whole = tifffile.imread(image_path.replace(folder_name, 'predictions_pith'))
        pith_whole[pith_whole >= 0.5] = 1
        pith_whole[pith_whole < 0.5] = 0
        pith_whole = cv2.resize(pith_whole, (image.shape[1], image.shape[0]))
        contours, _ = cv2.findContours(pith_whole.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        longest_contour = np.argmax(np.array([len(contour) for contour in contours]))
        one_indice = np.where(pith_whole == 1)
        center = int(np.mean(one_indice[0])), int(np.mean(one_indice[1]))

        pith_whole = binary_dilation(pith_whole, iterations=int(0.005 * width))
        image = image * (1 - pith_whole)

        min_value = np.min(image)
        max_value = int(np.max(image))
        dark_point = center[0]
        image[dark_point, :] = copy.deepcopy(min_value)

        light_part = np.mean(image[dark_point - 10:dark_point + 10, :], axis=0)
        ret = threshold_otsu(light_part)
        peaks1, _ = find_peaks(light_part[:center[1]], height=ret, distance=0.05 * width)
        peaks1 = peaks1[::-1]
        peaks2, _ = find_peaks(light_part[center[1]:], height=ret, distance=0.05 * width)
        peaks2 = peaks2 + center[1]

        if len(peaks1) < len(peaks2):
            a = copy.deepcopy(peaks1)
            peaks1 = copy.deepcopy(peaks2)
            peaks2 = copy.deepcopy(a)

        peaks1_center = np.abs(peaks1 - center[1])
        peaks2_center = np.abs(peaks2 - center[1])
        num_pair = min(len(peaks1), len(peaks2))
        data = []
        remains = list(np.arange(0, len(peaks1)))
        for i in range(0, num_pair, 1):
            difference = np.abs(peaks2_center[i] - peaks1_center)
            sort = np.argsort(difference)
            j = sort[np.isin(sort, remains)]
            if len(j):
                j = j[0]
    
                data.append((image, peaks1[j], peaks2[i], dark_point, dark_point - 1))
                data.append((image, peaks1[j], peaks2[i], dark_point, dark_point + 1))
                remains.remove(j)

        for j in remains:
            if 0 <= 2 * center[1] - peaks1[j] < width:
                data.append((image, 2 * center[1] - peaks1[j], peaks1[j], dark_point, dark_point - 1))
                data.append((image, 2 * center[1] - peaks1[j], peaks1[j], dark_point, dark_point + 1))

        with Pool(multiprocessing.cpu_count()) as pool:
            results = pool.starmap(plot_half_ring, data)

        image_white = np.zeros_like(image, dtype=np.uint8)
        add = False
        i = 0
        while i < len(results):
            image_i = np.zeros_like(image, dtype=np.uint8)
            cv2.polylines(image_i, results[i], True, 1, 1)
            cv2.polylines(image_i, results[i + 1], True, 1, 1)
            j = i + 2
            while j < len(results):
                image_j = np.zeros_like(image, dtype=np.uint8)
                cv2.polylines(image_j, results[j], True, 1, 1)
                cv2.polylines(image_j, results[j + 1], True, 1, 1)
                image_ij = np.bitwise_and(image_i, image_j)
                sum_ij1 = np.sum(image_ij[:dark_point])
                sum_ij2 = np.sum(image_ij[dark_point + 1:])
                if sum_ij1 >= 1 and sum_ij2 >= 1:
                    add = True
                    if len(results[i]) < len(results[j]):
                        image_white = np.bitwise_or(image_white, image_i)
                    else:
                        image_white = np.bitwise_or(image_white, image_j)
                    results.pop(j)
                    results.pop(j)
                elif sum_ij1 >= 1 and sum_ij2 == 0:
                    add = True
                    if len(results[i]) < len(results[j]):
                        image_white = np.bitwise_or(image_white, image_i)
                        image_white[dark_point + 1:] = np.bitwise_or(image_white[dark_point + 1:], image_j[dark_point + 1:])
                        opposite_ring = np.stack([results[j + 1][:, :, 0][:, :, None], 
                                             2 * dark_point - results[j + 1][:, :, 1][:, :, None]], axis=-1)
                        cv2.polylines(image_white, opposite_ring.astype(np.int32), True, 1, 1)
                    else:
                        image_white = np.bitwise_or(image_white, image_j)
                        image_white[dark_point + 1:] = np.bitwise_or(image_white[dark_point + 1:], image_i[dark_point + 1:])
                        opposite_ring = np.stack([results[i + 1][:, :, 0][:, :, None],
                                            2 * dark_point - results[i + 1][:, :, 1][:, :, None]], axis=-1)
                        cv2.polylines(image_white, opposite_ring.astype(np.int32), True, 1, 1)
                    results.pop(j)
                    results.pop(j)
                elif sum_ij1 == 0 and sum_ij2 >= 1:
                    add = True
                    if len(results[i]) < len(results[j]):
                        image_white = np.bitwise_or(image_white, image_i)
                        image_white[:dark_point] = np.bitwise_or(image_white[:dark_point], image_j[:dark_point])
                        opposite_ring = np.stack([results[j][:, :, 0][:, :, None], 
                                                  2 * dark_point - results[j][:, :, 1][:, :, None]], axis=-1)
                        cv2.polylines(image_white, opposite_ring.astype(np.int32), True, 1, 1)
                    else:
                        image_white = np.bitwise_or(image_white, image_j)
                        image_white[:dark_point] = np.bitwise_or(image_white[:dark_point], image_i[:dark_point])
                        opposite_ring = np.stack([results[i][:, :, 0][:, :, None], 
                                             2 * dark_point - results[i][:, :, 1][:, :, None]], axis=-1)
                        cv2.polylines(image_white, opposite_ring.astype(np.int32), True, 1, 1)
                    
                    results.pop(j)
                    results.pop(j)
                j += 2
            if not add:
                image_white = np.bitwise_or(image_white, image_i)

            add = False
            i += 2
            

        # for i in range(len(results)):
        #     cor = results[i]
        #     cv2.polylines(image, cor, True, int(max_value), 1)

        cv2.drawContours(image_white, contours, longest_contour, 1, 1)

        # plt.figure()
        # plt.imshow(image_white)
        # for i in range(0, len(peaks1), 1):
        #     plt.plot(peaks1[i], dark_point, 'ro')
        # for i in range(0, len(peaks2), 1):
        #     plt.plot(peaks2[i], dark_point, 'bo')
        # plt.show()

        input_image = tifffile.imread(image_path.replace(folder_name, 'input'))
        image_white = cv2.resize(image_white, (input_image.shape[1], input_image.shape[0]))
        image_white[image_white == 1] = 255
        input_image[image_white == 255] = 0
        tifffile.imwrite('/home/khietdang/Documents/khiet/treeRing/brightest_bigDistance_pith/' + os.path.basename(image_path),
                         image_white.astype(np.uint8))
        tifffile.imwrite('/home/khietdang/Documents/khiet/treeRing/final_brightest_bigDistance_pith/' + os.path.basename(image_path),
                         input_image.astype(np.uint8))