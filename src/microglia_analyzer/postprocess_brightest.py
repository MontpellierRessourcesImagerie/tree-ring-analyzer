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
import time
from brightest_path_lib.heuristic import Heuristic
import math

class CustomHeuristicFunction(Heuristic):
    def __init__(self, scale, center, radius):
        if scale is None:
            raise TypeError
        if len(scale) == 0:
            raise ValueError

        self.scale_x = scale[0]
        self.scale_y = scale[1]
        self.center = center
        self.radius = radius

    def estimate_cost_to_goal(self, current_point, goal_point):
        if current_point is None or goal_point is None:
            raise TypeError
        if (len(current_point) == 0 or len(goal_point) == 0) or (len(current_point) != len(goal_point)):
            raise ValueError

        current_x, current_y = current_point[1], current_point[0]
        goal_x, goal_y = goal_point[1], goal_point[0]
        
        x_diff = (goal_x - current_x) * self.scale_x
        y_diff = (goal_y - current_y) * self.scale_y

        diff = np.abs(np.sqrt(np.sum((self.center - current_point) ** 2)) - self.radius)
        
        return math.sqrt((x_diff * x_diff) + (y_diff * y_diff)) + diff ** 2


def create_cone(image, center, max_height):
    h, w = image.shape
    x0, y0 = center
    
    # Generate coordinate grids
    x = np.arange(w)
    y = np.arange(h)
    Y, X = np.meshgrid(x, y)
    
    # Compute the Euclidean distance from the given point
    D = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    
    # Compute cone height map
    D = max_height - D
    D[D < 0] = 0  # Clip negative values
    
    return D.astype(np.int32)


def create_light_part(image, dark_point1, dark_point2):
    max_height = min(dark_point1, image.shape[0] - dark_point1, dark_point2, image.shape[1] - dark_point2)
    cone_image = create_cone(image, (dark_point1, dark_point2), max_height=max_height)

    image1, cone_image1 = image[:, :dark_point2], cone_image[:, :dark_point2]
    light_part1 = [np.mean(image1[cone_image1 == i]) for i in range(0, max_height)]

    image2, cone_image2 = image[:, dark_point2:], cone_image[:, dark_point2:]
    light_part2 = [np.mean(image2[cone_image2 == i]) for i in range(0, max_height)]

    light_part = np.zeros(image.shape[1])
    light_part[dark_point2 - len(light_part1):dark_point2] = copy.deepcopy(light_part1)
    light_part[dark_point2 + 1:dark_point2 + len(light_part2) + 1] = copy.deepcopy(light_part2[::-1])

    return light_part
    

def plot_half_ring(image, peak1, peak2, light_point, center):
    start_point = np.array([light_point, peak1])
    goal_point = np.array([light_point, peak2])
    radius = (np.sqrt(np.sum((start_point - center) ** 2)) + np.sqrt(np.sum((start_point - center) ** 2))) / 2
    search_algorithm = AStarSearch(image, start_point=start_point, goal_point=goal_point)
    search_algorithm.heuristic_function = CustomHeuristicFunction(scale=(1.0, 1.0), center=center, radius=radius)
    brightest_path =search_algorithm.search()

    result = np.array(search_algorithm.result)[:, 1]
    brightest_path = np.array(search_algorithm.result)[:, 0]

    cor = np.stack([result[:, None], brightest_path[:, None]], axis=-1)
    return cor

if __name__ == '__main__':
    t0 = time.time()
    thickness = 2
    folder_name = 'predictions_bigDistance'
    # image_list = glob.glob(f'/home/khietdang/Documents/khiet/treeRing/{folder_name}/*.tif')
    image_list = [f'/home/khietdang/Documents/khiet/treeRing/{folder_name}/4 E 3 t_8µm_x50 2.tif']
    for image_path in image_list:
        print(image_path)
        image = tifffile.imread(image_path)
        height_ori, width_ori = image.shape
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

        dark_point = center[0]

        # light_part = create_light_part(image, dark_point, center[1])
        light_part = np.mean(image[dark_point - 10:dark_point + 10, :], axis=0)
        ret = threshold_otsu(light_part)
        peaks1, _ = find_peaks(light_part[:center[1]], height=ret, distance=0.05 * width)
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
        image_upper = np.zeros_like(image)
        image_upper[:dark_point] = 1
        image_upper = image_upper * image
        image_lower = np.zeros_like(image)
        image_lower[dark_point:] = 1
        image_lower = image_lower * image
        for i in range(0, num_pair, 1):
            difference = np.abs(peaks2_center[i] - peaks1_center)
            sort = np.argsort(difference)
            j = sort[np.isin(sort, remains)]
            if len(j):
                j = j[0]
    
                data.append((image_upper, peaks1[j], peaks2[i], dark_point - 1, np.array(center)))
                data.append((image_lower, peaks1[j], peaks2[i], dark_point, np.array(center)))
                remains.remove(j)

        for j in remains:
            if 0 <= 2 * center[1] - peaks1[j] < width:
                data.append((image_upper, 2 * center[1] - peaks1[j], peaks1[j], dark_point - 1, np.array(center)))
                data.append((image_lower, 2 * center[1] - peaks1[j], peaks1[j], dark_point, np.array(center)))

        # with Pool(1) as pool:
        with Pool(multiprocessing.cpu_count()) as pool:
            results = pool.starmap(plot_half_ring, data)

        draw_image = []
        for i in range(0, len(results), 2):
            _image = np.zeros((height, width), dtype=np.uint8)
            cv2.polylines(_image, results[i], True, 1, thickness)
            cv2.polylines(_image, results[i + 1], True, 1, thickness)
            draw_image.append(_image)

        image_white = np.zeros((height, width), dtype=np.uint8)
        add = False
        i = 0
        while i < len(results):
            image_i = draw_image[int(i / 2)]
            j = i + 2
            while j < len(results):
                image_j = draw_image[int(j / 2)]
                image_ij = np.bitwise_and(image_i, image_j)
                sum_ij = np.sum(image_ij)

                if sum_ij >= 1:
                    add = True
                    if len(results[i]) < len(results[j]):
                        image_white = np.bitwise_or(image_white, image_i)
                    else:
                        image_white = np.bitwise_or(image_white, image_j)
                    
                    results.pop(j)
                    results.pop(j)
                    draw_image.pop(int(j / 2))
                else:
                    j += 2
            if not add:
                image_white = np.bitwise_or(image_white, image_i)

            add = False
            i += 2

        cv2.drawContours(image_white, contours, longest_contour, 1, thickness)

        input_image = tifffile.imread(image_path.replace(folder_name, 'input'))
        image_white = cv2.resize(image_white, (input_image.shape[1], input_image.shape[0]))
        image_white[image_white == 1] = 255
        input_image[image_white == 255] = 0

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image_white)
        for i in range(0, len(peaks1), 1):
            plt.plot(peaks1[i] * 10, dark_point * 10, 'ro')
        for i in range(0, len(peaks2), 1):
            plt.plot(peaks2[i] * 10, dark_point * 10, 'bo')
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.show()

        tifffile.imwrite('/home/khietdang/Documents/khiet/treeRing/brightest_bigDistance_pith/' + os.path.basename(image_path),
                         image_white.astype(np.uint8))
        tifffile.imwrite('/home/khietdang/Documents/khiet/treeRing/final_brightest_bigDistance_pith/' + os.path.basename(image_path),
                         input_image.astype(np.uint8))
    print(time.time() - t0)