from brightest_path_lib.algorithm import AStarSearch
from brightest_path_lib.heuristic import Heuristic
import copy
import cv2
import math
import multiprocessing
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu
from tree_ring_analyzer.tiles.tiler import ImageTiler2D
import matplotlib.pyplot as plt



class CircleHeuristicFunction(Heuristic):
    def __init__(self, center, radius, height, width):
        self.center = center
        self.radius = radius
        self.height = height
        self.width = width


    def estimate_cost_to_goal(self, current_point, goal_point):
        if current_point is None or goal_point is None:
            raise TypeError
        if (len(current_point) == 0 or len(goal_point) == 0) or (len(current_point) != len(goal_point)):
            raise ValueError

        diff0 = np.sum((current_point - goal_point) ** 2)
        diff1 = np.abs(self.radius - np.sqrt(np.sum((self.center - current_point) ** 2)))
        
        cost = np.sqrt(diff0) + diff1 ** 2
        if 0.05 * self.height < current_point[0] < 0.95 * self.height and 0.05 * self.width < current_point[1] < 0.95 * self.width:
            return cost
        else:
            return cost * 1000
    


class TreeRingSegmentation:
    

    def __init__(self, modelRing, modelPith):
        self.modelRing = modelRing
        self.modelPith = modelPith
        self.patchSize = 256
        self.overlap = self.patchSize - 196
        self.batchSize = 8
        self.thickness = 1

        self.predictionRing = None
        self.pith = None
        self.maskRings = None
        self.center = None
        self.shape = None


    def predictRing(self, image):
        self.shape = image.shape[0], image.shape[1]

        ## Tiling
        tiles_manager = ImageTiler2D(self.patchSize, self.overlap, self.shape)
        tiles = tiles_manager.image_to_tiles(image, use_normalize=False)
        tiles = np.array(tiles) / 255

        ## Prediction
        prediction_ring = np.squeeze(self.modelRing.predict(tiles, batch_size=self.batchSize, verbose=0))

        ## Reconstruction
        tiles_manager = ImageTiler2D(self.patchSize, self.overlap, self.shape[:2])
        self.predictionRing = tiles_manager.tiles_to_image(prediction_ring)


    def predictPith(self, image):
        ## Center identification from ring prediction
        prediction_ring = self.predictionRing
        ret = threshold_otsu(self.predictionRing)
        ring_indices = np.where(prediction_ring > ret)
        chose_indices = (0.2 * image.shape[0] < ring_indices[0]) & (ring_indices[0] < 0.8 * image.shape[0]) \
            & (0.2 * image.shape[1] < ring_indices[1]) & (ring_indices[1] < 0.8 * image.shape[1])
        center = int(np.mean(ring_indices[0][chose_indices])), int(np.mean(ring_indices[1][chose_indices]))

        ## Cropping
        crop_size = int(0.1 * max(self.shape[1], self.shape[0])) * 2 
        crop_img = image[center[0] - int(crop_size / 2):center[0] + int(crop_size / 2),
                         center[1] - int(crop_size / 2):center[1] + int(crop_size / 2)]
        crop_img = cv2.resize(crop_img, (self.patchSize, self.patchSize))[None, :, :, :]
        crop_img = crop_img / 255

        ## Prediction
        prediction_crop_pith = self.modelPith.predict(crop_img, batch_size=1, verbose=0)
        prediction_crop_pith = cv2.resize(prediction_crop_pith[0], (crop_size, crop_size))
        prediction_pith = np.zeros((self.shape[0], self.shape[1]))
        prediction_pith[center[0] - int(crop_size / 2):center[0] + int(crop_size / 2),
                        center[1] - int(crop_size / 2):center[1] + int(crop_size / 2)] = copy.deepcopy(prediction_crop_pith)

        ## Binarization
        pith = np.zeros((self.shape[0], self.shape[1]))
        pith[prediction_pith > 0.5] = 1
        self.pith = pith


    @staticmethod
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

    
    def findEndPoints(self):
        ## Resize
        prediction_ring = cv2.resize(self.predictionRing, (int(self.shape[1] / 10), int(self.shape[0] / 10)))
        height, width = prediction_ring.shape
        pith = cv2.resize(self.pith, (width, height))

        ## Center identification from pith
        one_indice = np.where(pith == 1)
        self.center = int(np.mean(one_indice[0])), int(np.mean(one_indice[1]))
        dark_point = self.center[0]

        ## Delete pith area from ring prediction
        pith_dilated = binary_dilation(pith, iterations=int(0.005 * width))
        prediction_ring = prediction_ring * (1 - pith_dilated)

        ## Find start and goal points
        light_part = np.mean(prediction_ring[dark_point - int(0.05 * width):dark_point + int(0.05 * width), :], axis=0)
        ret = threshold_otsu(light_part)
        peaks1, _ = find_peaks(light_part[:self.center[1]], height=ret, distance=0.05 * width)
        peaks1 = peaks1[peaks1 > 0.05 * width]
        if len(peaks1) == 0:
            peaks1, _ = find_peaks(light_part[:self.center[1]], height=threshold_otsu(light_part[int(0.1 * width):self.center[1]]), 
                                   distance=0.05 * width)
            peaks1 = peaks1[peaks1 > 0.05 * width]

        peaks2, _ = find_peaks(light_part[self.center[1]:], height=ret, distance=0.05 * width)
        peaks2 = peaks2 + self.center[1]
        peaks2 = peaks2[peaks2 < 0.95 * width]
        if len(peaks2) == 0:
            peaks2, _ = find_peaks(light_part[self.center[1]:], height=threshold_otsu(light_part[self.center[1]:int(0.9 * width)]), 
                                   distance=0.05 * width)
            peaks2 = peaks2 + self.center[1]
            peaks2 = peaks2[peaks2 < 0.95 * width]

        max_height = min(dark_point, prediction_ring.shape[0] - dark_point, self.center[1], prediction_ring.shape[1] - self.center[1])
        cone_image = self.create_cone(prediction_ring, self.center, max_height)
        prediction_ring[cone_image > max_height - min(self.center[1] - peaks1[-1], peaks2[0] - self.center[1]) * 0.9] = 0

        if len(peaks1) < len(peaks2):
            a = copy.deepcopy(peaks1)
            peaks1 = copy.deepcopy(peaks2)
            peaks2 = copy.deepcopy(a)

        ## Catch up the pairs of start points and goal points
        peaks1_center = np.abs(peaks1 - self.center[1])[:, None]
        peaks2_center = np.abs(peaks2 - self.center[1])[None, :]
        diff_pp = np.abs(peaks1_center - peaks2_center)
        max_value = np.max(diff_pp)

        num_pair = min(len(peaks1), len(peaks2))
        data = []
        remains = list(np.arange(0, len(peaks1)))

        image_upper = np.zeros_like(prediction_ring)
        image_upper[:dark_point] = 1
        image_upper = image_upper * prediction_ring
        image_lower = np.zeros_like(prediction_ring)
        image_lower[dark_point:] = 1
        image_lower = image_lower * prediction_ring
        for k in range(0, num_pair, 1):
            j, i = np.where(diff_pp == np.min(diff_pp))
            if len(j) >= len(peaks1) * len(peaks2):
                break
            j = j[0]
            i = i[0]
            data.append((image_upper, peaks1[j], peaks2[i], dark_point - 1, np.array(self.center)))
            data.append((image_lower, peaks1[j], peaks2[i], dark_point, np.array(self.center)))
            remains.remove(j)
            diff_pp[j, :] = max_value
            diff_pp[:, i] = max_value

        for j in remains:
            if 0.05 * width <= 2 * self.center[1] - peaks1[j] < 0.95 * width:
                data.append((image_upper, 2 * self.center[1] - peaks1[j], peaks1[j], dark_point - 1, np.array(self.center)))
                data.append((image_lower, 2 * self.center[1] - peaks1[j], peaks1[j], dark_point, np.array(self.center)))
        
        return data
    

    @staticmethod
    def traceHalfRing(image, peak1, peak2, light_point, center):
        start_point = np.array([light_point, peak1])
        goal_point = np.array([light_point, peak2])
        radius = (np.sqrt(np.sum((start_point - center) ** 2)) + np.sqrt(np.sum((goal_point - center) ** 2))) / 2
        search_algorithm = AStarSearch(image, start_point=start_point, goal_point=goal_point)
        search_algorithm.heuristic_function = CircleHeuristicFunction(center=center, radius=radius, height=image.shape[0], width=image.shape[1])
        brightest_path = search_algorithm.search()

        result = np.array(search_algorithm.result)[:, 1]
        brightest_path = np.array(search_algorithm.result)[:, 0]

        cor = np.stack([result[:, None], brightest_path[:, None]], axis=-1)
        return cor


    def createMaskOfRings(self, results):
        image_white = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)
        contours, _ = cv2.findContours(self.pith.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        chosen_contour = np.argmax([cv2.pointPolygonTest(contour, [self.center[1], self.center[0]], True) for contour in contours])
        cv2.drawContours(image_white, [np.array(contours[chosen_contour])], 0, 1, self.thickness)

        length_results_sorted = np.argsort([len(results[i]) + len(results[i + 1]) for i in range(0, len(results), 2)])
        for i in range(0, len(length_results_sorted)):
            _image = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)
            j = length_results_sorted[i]
            cv2.drawContours(_image, [np.append(results[2 * j], results[2 * j + 1][::-1], axis=0) * 10], 0, 1, self.thickness)
            if np.sum(np.bitwise_and(_image, image_white)) == 0:
                image_white = np.bitwise_or(image_white, _image)

        image_white[image_white == 1] = 255

        return image_white


    def segmentImage(self, image):
        self.predictRing(image)
        self.predictPith(image)

        data = self.findEndPoints()

        with Pool(multiprocessing.cpu_count()) as pool:
            results = pool.starmap(self.traceHalfRing, data)

        self.maskRings = self.createMaskOfRings(results)

