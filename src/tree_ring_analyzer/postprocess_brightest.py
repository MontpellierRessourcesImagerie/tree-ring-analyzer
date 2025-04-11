from brightest_path_lib.algorithm import AStarSearch
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.signal import find_peaks
import cv2
from multiprocessing import Pool
import os
from scipy.ndimage import binary_dilation, median_filter
from skimage.filters import threshold_otsu
import time
from tree_ring_analyzer.segmentation import CircleHeuristicFunction

    
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


def predict_num_peak(image, dark_point1, dark_point2):
    max_height = min(dark_point1, image.shape[0] - dark_point1, dark_point2, image.shape[1] - dark_point2)
    image_new = np.zeros((2 * max_height, 2 * max_height))
    image_new[:max_height, :max_height] = cv2.resize(image[:dark_point1, :dark_point2], (max_height, max_height))
    image_new[max_height:, :max_height] = cv2.resize(image[dark_point1:, :dark_point2], (max_height, max_height))
    image_new[:max_height, max_height:] = cv2.resize(image[:dark_point1, dark_point2:], (max_height, max_height))
    image_new[max_height:, max_height:] = cv2.resize(image[dark_point1:, dark_point2:], (max_height, max_height))
    cone_image = create_cone(image_new, (dark_point1, dark_point2), max_height=max_height)

    light_part = np.array([np.mean(image_new[cone_image == i]) for i in range(0, max_height)])

    peaks, _ = find_peaks(light_part, distance= 0.05 *  image.shape[1])

    return len(peaks)
    

def plot_half_ring(image, peak1, peak2, light_point, center):
    start_point = np.array([light_point, peak1])
    goal_point = np.array([light_point, peak2])
    radius = (np.sqrt(np.sum((start_point - center) ** 2)) + np.sqrt(np.sum((goal_point - center) ** 2))) / 2
    search_algorithm = AStarSearch(image, start_point=start_point, goal_point=goal_point)
    search_algorithm.heuristic_function = CircleHeuristicFunction(center=center, radius=radius, image=image)
    # search_algorithm.cost_function = CustomCostFunction()
    brightest_path =search_algorithm.search()

    result = np.array(search_algorithm.result)[:, 1]
    brightest_path = np.array(search_algorithm.result)[:, 0]

    cor = np.stack([result[:, None], brightest_path[:, None]], axis=-1)
    return cor

if __name__ == '__main__':
    t0 = time.time()
    thickness = 1
    folder_name = '/home/khietdang/Documents/khiet/treeRing/transfer/predictions_bigDisRingAugGray'
    pith_name = '/home/khietdang/Documents/khiet/treeRing/transfer/predictions_pith'
    input_name = '/home/khietdang/Documents/khiet/treeRing/transfer/input_transfer'
    resize = 5
    # mask_name = '/home/khietdang/Documents/khiet/treeRing/masks'
    # image_list = glob.glob(os.path.join(folder_name, '*.tif'))
    image_list = [os.path.join(folder_name, '23(4)_x50_8 µm.tif')]
    for image_path in image_list:
        print(image_path)
        image_ori = tifffile.imread(image_path)
        height_ori, width_ori = image_ori.shape
        image = cv2.resize(image_ori, (int(width_ori / resize), int(height_ori / resize)))
        height, width = image.shape
        # filter_size = (int(0.0075 * height), int(0.0075 * width))
        # max_value = int(np.max(image) / 2)
        # filter_size = (filter_size[0] if filter_size[0] < max_value else max_value, 
        #                filter_size[1] if filter_size[1] < max_value else max_value)
        # image = median_filter(image, filter_size)

        pith_whole = tifffile.imread(os.path.join(pith_name, os.path.basename(image_path)))
        pith_whole[pith_whole >= 0.5] = 1
        pith_whole[pith_whole < 0.5] = 0
        pith_whole = cv2.resize(pith_whole, (image.shape[1], image.shape[0]))
        
        one_indice = np.where(pith_whole == 1)
        center = int(np.mean(one_indice[0])), int(np.mean(one_indice[1]))

        pith_dilated = binary_dilation(pith_whole, iterations=int(0.005 * width))
        image = image * (1 - pith_dilated)

        dark_point = center[0]
        # num_peak = predict_num_peak(image, dark_point, center[1])
        
        light_part = np.mean(image[dark_point - int(0.05 * height):dark_point + int(0.05 * height), :], axis=0)
        ret = threshold_otsu(light_part)
        peaks1, _ = find_peaks(light_part[:center[1]], height=ret, distance=0.05 * width)
        peaks1 = peaks1[peaks1 > 0.05 * width]
        if len(peaks1) <= 1:
            peaks1, _ = find_peaks(light_part[:center[1]], height=threshold_otsu(light_part[:center[1]]), 
                                   distance=0.05 * width)
            peaks1 = peaks1[peaks1 > 0.05 * width]

        peaks2, _ = find_peaks(light_part[center[1]:], height=ret, distance=0.05 * width)
        peaks2 = peaks2 + center[1]
        peaks2 = peaks2[peaks2 < 0.95 * width]
        if len(peaks2) <= 1:
            peaks2, _ = find_peaks(light_part[center[1]:], height=threshold_otsu(light_part[center[1]:]), 
                                   distance=0.05 * width)
            peaks2 = peaks2 + center[1]
            peaks2 = peaks2[peaks2 < 0.95 * width]
            
        # max_height = min(dark_point, image.shape[0] - dark_point, center[1], image.shape[1] - center[1])
        # cone_image = create_cone(image, center, max_height)
        # image[cone_image > max_height - min(center[1] - peaks1[-1], peaks2[0] - center[1]) * 0.9] = 0

        if len(peaks1) < len(peaks2):
            a = copy.deepcopy(peaks1)
            peaks1 = copy.deepcopy(peaks2)
            peaks2 = copy.deepcopy(a)

        peaks1_center = np.abs(peaks1 - center[1])[:, None]
        peaks2_center = np.abs(peaks2 - center[1])[None, :]
        diff_pp = np.abs(peaks1_center - peaks2_center)
        max_value = np.max(diff_pp) + 1

        num_pair = min(len(peaks1), len(peaks2))
        data = []
        remains = list(np.arange(0, len(peaks1)))

        image_upper = np.zeros_like(image)
        image_upper[:dark_point] = 1
        image_upper = image_upper * image
        image_lower = np.zeros_like(image)
        image_lower[dark_point:] = 1
        image_lower = image_lower * image
        for k in range(0, num_pair, 1):
            min_value = np.min(diff_pp)
            if min_value == max_value:
                break
            j, i = np.where(diff_pp == min_value)
            j = j[0]
            i = i[0]
            data.append((image_upper, peaks1[j], peaks2[i], dark_point - 1, np.array(center)))
            data.append((image_lower, peaks1[j], peaks2[i], dark_point, np.array(center)))
            remains.remove(j)
            diff_pp[j, :] = copy.deepcopy(max_value)
            diff_pp[:, i] = copy.deepcopy(max_value)

        for j in remains:
            if 0.05 * width <= 2 * center[1] - peaks1[j] < 0.95 * width:
                data.append((image_upper, 2 * center[1] - peaks1[j], peaks1[j], dark_point - 1, np.array(center)))
                data.append((image_lower, 2 * center[1] - peaks1[j], peaks1[j], dark_point, np.array(center)))

        with Pool(1) as pool:
        # with Pool(multiprocessing.cpu_count()) as pool:
            results = pool.starmap(plot_half_ring, data)
        
        image_white = np.zeros((height_ori, width_ori), dtype=np.uint8)
        contours, _ = cv2.findContours(pith_whole.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        chosen_contour = np.argmax([cv2.pointPolygonTest(contour, [center[1], center[0]], True) for contour in contours])
        cv2.drawContours(image_white, [np.array(contours[chosen_contour]) * resize], 0, 1, thickness)
        predictedRings = [np.array(contours[chosen_contour]) * resize]

        length_results_sorted = np.argsort([len(results[i]) + len(results[i + 1]) for i in range(0, len(results), 2)])
        for i in range(0, len(length_results_sorted)):
            _image = np.zeros((height_ori, width_ori), dtype=np.uint8)
            j = length_results_sorted[i]
            predictedRing = np.append(results[2 * j], results[2 * j + 1][::-1], axis=0) * resize
            cv2.drawContours(_image, [predictedRing], 0, 1, thickness)
            if np.sum(np.bitwise_and(_image, image_white)) == 0:
                image_white = np.bitwise_or(image_white, _image)
                predictedRings.append(predictedRing)

        input_image = tifffile.imread(os.path.join(input_name, os.path.basename(image_path)))
        image_white = cv2.resize(image_white, (input_image.shape[1], input_image.shape[0]))
        image_white[image_white == 1] = 255
        input_image[image_white == 255] = 0

        # mask = tifffile.imread(os.path.join(mask_name, os.path.basename(image_path)))
        # treeRingSeg = TreeRingSegmentation()
        # treeRingSeg.predictedRings = predictedRings
        # treeRingSeg.predictionRing = image_ori
        # hausdorff, mse = treeRingSeg.evaluate(mask)
        # print(hausdorff, mse)
        # input_image[mask == 1] = 0

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(image_white)
        for i in range(0, len(peaks1), 1):
            plt.plot(peaks1[i] * resize, dark_point * resize, 'ro')
        for i in range(0, len(peaks2), 1):
            plt.plot(peaks2[i] * resize, dark_point * resize, 'bo')
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.show()
        plt.close()

    print(time.time() - t0)
