from brightest_path_lib.algorithm import AStarSearch
from brightest_path_lib.heuristic import Heuristic
from brightest_path_lib.cost import Cost
import copy
import cv2
import multiprocessing
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import find_peaks, savgol_filter
from skimage.filters import threshold_otsu
from tree_ring_analyzer.tiles.tiler import ImageTiler2D
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion
from skimage.morphology import skeletonize



class CustomCostFunction(Cost):
    def __init__(self, min_intensity, max_intensity):
        super().__init__()
        if min_intensity is None or max_intensity is None:
            raise TypeError
        if min_intensity > max_intensity:
            raise ValueError
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.RECIPROCAL_MIN = float(1E-6)
        self.RECIPROCAL_MAX = 255.0
        self._min_step_cost = 1.0 / self.RECIPROCAL_MAX


    def cost_of_moving_to(self, intensity_at_new_point):
        if intensity_at_new_point > self.max_intensity:
            raise ValueError

        intensity_at_new_point = self.RECIPROCAL_MAX * (intensity_at_new_point - self.min_intensity) / (self.max_intensity - self.min_intensity)

        if intensity_at_new_point < self.RECIPROCAL_MIN:
            intensity_at_new_point = self.RECIPROCAL_MIN
        
        return  1 / (intensity_at_new_point ** 3)
    
    def minimum_step_cost(self):
        return self._min_step_cost
    


class CircleHeuristicFunction(Heuristic):
    def __init__(self, image, center, startPoint, radius):
        self.radius = radius
        self.image = image

        self.center = center
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.startPoint = startPoint


    def estimate_cost_to_goal(self, current_point, goal_point):
        if current_point is None or goal_point is None:
            raise TypeError
        if (len(current_point) == 0 or len(goal_point) == 0) or (len(current_point) != len(goal_point)):
            raise ValueError

        diff0 = np.sqrt(np.sum((current_point - goal_point) ** 2))
        diff1 = np.abs(self.radius - np.sqrt(np.sum((self.center - current_point) ** 2)))
        
        # cost = diff1 ** 2 + diff0 + (diff1 ** 2) * np.sum((current_point - goal_point) * (current_point - self.startPoint))
        cost = diff0 + diff1 ** 2

        return cost
    


class TreeRingSegmentation:
    

    def __init__(self, resize=10):
        self.patchSize = 256
        self.overlap = self.patchSize - 196
        self.batchSize = 8
        self.thickness = 1
        self.iterations = 10
        self.resize = resize

        self.predictionRing = None
        self.pith = None
        self.pithContour = None
        self.predictedRings = []
        self.maskRings = None
        self.center = None
        self.shape = None


    def predictRing(self, modelRing, image):
        self.shape = image.shape[0], image.shape[1]

        ## Tiling
        image = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2])[:, :, None]
        tiles_manager = ImageTiler2D(self.patchSize, self.overlap, self.shape)
        tiles = tiles_manager.image_to_tiles(image, use_normalize=True)
        tiles = np.array(tiles)

        ## Prediction
        predictionRing = np.squeeze(modelRing.predict(tiles, batch_size=self.batchSize, verbose=0))

        ## Reconstruction
        predictionRing = tiles_manager.tiles_to_image(predictionRing)

        return predictionRing


    def predictPith(self, modelPith, image):
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
        prediction_crop_pith = modelPith.predict(crop_img, batch_size=1, verbose=0)
        prediction_crop_pith = cv2.resize(prediction_crop_pith[0], (crop_size, crop_size))
        prediction_pith = np.zeros((self.shape[0], self.shape[1]))
        prediction_pith[center[0] - int(crop_size / 2):center[0] + int(crop_size / 2),
                        center[1] - int(crop_size / 2):center[1] + int(crop_size / 2)] = copy.deepcopy(prediction_crop_pith)

        ## Binarization
        pith = np.zeros((self.shape[0], self.shape[1]))
        pith[prediction_pith > 0.5] = 1
        self.pith = pith


    def postprocessPith(self):
        one_indice = np.where(self.pith == 1)
        center = int(np.mean(one_indice[0])), int(np.mean(one_indice[1]))
        contours, _ = cv2.findContours(self.pith.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        chosen_contour = np.argmax([cv2.pointPolygonTest(contour, [center[1], center[0]], True) for contour in contours])
        
        self.pithContour = contours[chosen_contour]
        self.center = (int(np.mean(contours[chosen_contour][:, :, 1]) / self.resize), 
                       int(np.mean(contours[chosen_contour][:, :, 0]) / self.resize))
        
    
    def createMask(self, image):
        imageBinary = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2])
        ret = threshold_otsu(imageBinary)
        imageBinary[imageBinary < ret] = 0
        imageBinary[imageBinary >= ret] = 1
        imageBinary = 1 - imageBinary

        contours, _ = cv2.findContours(imageBinary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        chose_contour = np.argmax(np.array([len(contour) for contour in contours]))

        mask = np.zeros_like(imageBinary)
        cv2.drawContours(mask, [contours[chose_contour]], 0, 1, -1)
        mask = binary_erosion(mask, iterations=self.iterations)

        return mask

    
    def findEndPoints(self):
        ## Postprocess prediction ring
        prediction_ring = cv2.resize(self.predictionRing, (int(self.shape[1] / self.resize), int(self.shape[0] / self.resize)))
        height, width = prediction_ring.shape
        
        ## Delete pith area from ring prediction
        pith = cv2.resize(self.pith, (width, height))
        pith_dilated = binary_dilation(pith, iterations=int(0.005 * width))
        prediction_ring = prediction_ring * (1 - pith_dilated)

        dark_point = self.center[0]

        ## Find start and goal points
        light_part = np.mean(prediction_ring[dark_point - int(0.05 * height):dark_point + int(0.05 * height), :], axis=0)
        ret = threshold_otsu(light_part)
        peaks1, _ = find_peaks(light_part[:self.center[1]], height=ret, distance=0.05 * width)
        if len(peaks1) <= 1:
            peaks1, _ = find_peaks(light_part[:self.center[1]], height=threshold_otsu(light_part[:self.center[1]]), 
                                   distance=0.05 * width)

        peaks2, _ = find_peaks(light_part[self.center[1]:], height=ret, distance=0.05 * width)
        peaks2 = peaks2 + self.center[1]
        if len(peaks2) <= 1:
            peaks2, _ = find_peaks(light_part[self.center[1]:], height=threshold_otsu(light_part[self.center[1]:]), 
                                   distance=0.05 * width)
            peaks2 = peaks2 + self.center[1]

        if len(peaks1) < len(peaks2):
            a = copy.deepcopy(peaks1)
            peaks1 = copy.deepcopy(peaks2)
            peaks2 = copy.deepcopy(a)

        # plt.figure(figsize=(10, 10))
        # plt.imshow(prediction_ring)
        # for i in range(0, len(peaks1), 1):
        #     plt.plot(peaks1[i], dark_point, 'ro')
        # for i in range(0, len(peaks2), 1):
        #     plt.plot(peaks2[i], dark_point, 'bo')
        # plt.show()
        # plt.close()
        # raise ValueError

        ## Catch up the pairs of start points and goal points
        peaks1_center = np.abs(peaks1 - self.center[1])[:, None]
        peaks2_center = np.abs(peaks2 - self.center[1])[None, :]
        diff_pp = np.abs(peaks1_center - peaks2_center)
        max_value = np.max(diff_pp) + 1

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
            min_value = np.min(diff_pp)
            if min_value == max_value:
                break
            j, i = np.where(diff_pp == min_value)
            j = j[0]
            i = i[0]
            data.append((image_upper, peaks1[j], peaks2[i], dark_point - 1, np.array(self.center), self.resize))
            data.append((image_lower, peaks1[j], peaks2[i], dark_point, np.array(self.center), self.resize))
            remains.remove(j)
            diff_pp[j, :] = max_value
            diff_pp[:, i] = max_value

        for j in remains:
            if 0.05 * width <= 2 * self.center[1] - peaks1[j] < 0.95 * width:
                data.append((image_upper, 2 * self.center[1] - peaks1[j], peaks1[j], dark_point - 1, np.array(self.center), self.resize))
                data.append((image_lower, 2 * self.center[1] - peaks1[j], peaks1[j], dark_point, np.array(self.center), self.resize))
        
        return data
    

    @classmethod
    def smooth(cls, cor, resize):
        y = savgol_filter(cor[:, 0, 1] * resize, 11, 3, mode='wrap')
        return np.round(np.append(cor[:, 0, 0][:, None, None] * resize, y[:, None, None], axis=-1)).astype(np.int32)


    @staticmethod
    def traceHalfRing(image, peak1, peak2, light_point, center, resize):
        start_point = np.array([light_point, peak1])
        goal_point = np.array([light_point, peak2])
        radius = (np.sqrt(np.sum((start_point - center) ** 2)) + np.sqrt(np.sum((goal_point - center) ** 2))) / 2
        search_algorithm = AStarSearch(image, start_point=start_point, goal_point=goal_point)
        search_algorithm.heuristic_function = CircleHeuristicFunction(image=image, center=center, startPoint=start_point, radius=radius)
        search_algorithm.cost_function = CustomCostFunction(min_intensity=np.min(image), max_intensity=np.max(image))
        brightest_path = search_algorithm.search()

        result = np.array(search_algorithm.result)[:, 1]
        brightest_path = np.array(search_algorithm.result)[:, 0]

        cor = np.stack([result[:, None], brightest_path[:, None]], axis=-1)
        cor = TreeRingSegmentation.smooth(cor, resize)

        return cor


    def createMaskOfRings(self, results):
        image_white = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)
        cv2.drawContours(image_white, [np.array(self.pithContour)], 0, 1, self.thickness)
        self.predictedRings.append(np.array(self.pithContour))

        length_results_sorted = np.argsort([len(results[i]) + len(results[i + 1]) for i in range(0, len(results), 2)])
        for i in range(0, len(length_results_sorted)):
            _image = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)
            j = length_results_sorted[i]
            ring = np.append(results[2 * j], results[2 * j + 1][::-1], axis=0)
            cv2.drawContours(_image, [ring], 0, 1, self.thickness)
            if np.sum(np.bitwise_and(_image, image_white)) == 0:
                image_white = np.bitwise_or(image_white, _image)
                self.predictedRings.append(ring)

        image_white[image_white == 1] = 255

        return image_white


    def segmentImage(self, modelRing, modelPith, image):
        mask = self.createMask(image)

        predictionRing = self.predictRing(modelRing, image)
        self.predictionRing = predictionRing * mask

        self.predictPith(modelPith, image)
        self.postprocessPith()

        data = self.findEndPoints()

        with Pool(multiprocessing.cpu_count()) as pool:
            results = pool.starmap(self.traceHalfRing, data)

        self.maskRings = self.createMaskOfRings(results)


    @staticmethod
    def calculateRadius(ring):
        center = np.mean(np.squeeze(ring), axis=0)
        dis = np.sqrt(np.sum((np.squeeze(ring) - center[None, :]) ** 2, axis=-1))
        return np.mean(dis)
    

    @staticmethod
    def calculateDistance(ring1, ring2):
        return np.sqrt(np.sum((ring1[:, None, :] - ring2[None, :, :]) ** 2, axis=-1))


    def calculateHausdorffDistance(self, ring1, ring2):
        dis = self.calculateDistance(ring1, ring2)
        return max(np.max(np.min(dis, axis=0)), np.max(np.min(dis, axis=1)))
    

    def evaluate(self, mask):
        predictedRadius = []
        for i in range(0, len(self.predictedRings)):
            predictedRadius.append(self.calculateRadius(self.predictedRings[i]))

        mask[np.bitwise_not(skeletonize(mask))] = 0
        num_labels, labeled_mask = cv2.connectedComponents(mask)
        gtRings = []
        gtRadius = []
        for i in range(1, num_labels):
            gtRing = np.transpose(np.array(np.where(labeled_mask == i)))
            gtRings.append(gtRing[:, ::-1])
            gtRadius.append(self.calculateRadius(gtRing))

        diffRadius = np.abs(np.array(predictedRadius)[:, None] - np.array(gtRadius)[None, :])
        max_value = np.max(diffRadius)
        num_pairs = np.min(diffRadius.shape)
        hausdorff = []
        for i in range(0, num_pairs):
            j, k = np.where(diffRadius ==  np.min(diffRadius))
            j = j[0]
            k = k[0]
            hausdorff.append(self.calculateHausdorffDistance(np.squeeze(self.predictedRings[j]), np.squeeze(gtRings[k])))
            diffRadius[j, :] = max_value
            diffRadius[:, k] = max_value

        mask[mask == 255] = 1
        dilatedMask = binary_dilation(mask, iterations=self.iterations)
        distanceMap = distance_transform_edt(dilatedMask)
        return np.mean(hausdorff), np.mean((self.predictionRing - distanceMap) ** 2)
    
