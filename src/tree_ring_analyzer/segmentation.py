from brightest_path_lib.algorithm import AStarSearch
from brightest_path_lib.heuristic import Heuristic
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



class CircleHeuristicFunction(Heuristic):
    def __init__(self, image, center, startPoint, radius, thres):
        self.radius = radius
        self.image = image

        self.center = center
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.maxValue = np.max(image)
        self.startPoint = startPoint
        self.thres = thres


    def estimate_cost_to_goal(self, current_point, goal_point):
        if current_point is None or goal_point is None:
            raise TypeError
        if (len(current_point) == 0 or len(goal_point) == 0) or (len(current_point) != len(goal_point)):
            raise ValueError

        diff0 = np.abs(current_point[1] - goal_point[1])

        currentRadius = (diff0 / (self.radius[0] + self.radius[1])) * (self.radius[0] - self.radius[1]) + self.radius[1]
        # currentThres = (diff0 / (self.radius[0] + self.radius[1])) * (self.thres[0] - self.thres[1]) + self.thres[1]

        cosTheta = np.abs(current_point[1] - self.center[1]) / np.sqrt(np.sum((current_point - self.center) ** 2))
        currentRadiusX = currentRadius * cosTheta
        currentRadiusY = currentRadius * np.sqrt(1 - cosTheta ** 2) * (self.height / self.width)
        currentRadius = np.sqrt(currentRadiusX ** 2 + currentRadiusY ** 2)
        
        # currentThresX = currentThres * cosTheta
        # currentThresY = currentThres * np.sqrt(1 - cosTheta ** 2) * (self.height / self.width)
        # currentThres = np.sqrt(currentThresX ** 2 + currentThresY ** 2)

        diff1 = np.abs(currentRadius - np.sqrt(np.sum((self.center - current_point) ** 2)))

        if diff1 > 0.2 * currentRadius and self.image[current_point[0], current_point[1]] < self.maxValue * 0.2:
            diff2 = np.abs(np.sum((current_point - goal_point) * (current_point - self.startPoint)))
            cost = diff0 + diff1 * 2 + diff2 ** 2
        else:
            cost = diff0

        return cost
    


class TreeRingSegmentation:
    

    def __init__(self, resize=10):
        self.patchSize = 256
        self.overlap = self.patchSize - 196
        self.batchSize = 8
        self.thickness = 10
        self.iterations = 10
        self.resize = resize

        self.predictionRing = None
        self.pith = None
        self.pithContour = None
        self.outerMask = None
        self.predictedRings = []
        self.maskRings = None
        self.center = None
        self.shape = None


    def predictRing(self, modelRing, image):
        self.shape = image.shape[0], image.shape[1]

        ## Tiling
        tiles_manager = ImageTiler2D(self.patchSize, self.overlap, self.shape)
        tiles = tiles_manager.image_to_tiles(image, use_normalize=True)
        tiles = np.array(tiles)

        ## Prediction
        predictionRing = np.squeeze(modelRing.predict(tiles, batch_size=self.batchSize, verbose=0))

        ## Reconstruction
        predictionRing = tiles_manager.tiles_to_image(predictionRing)

        return predictionRing
    

    def cropAndPredictPith(self, modelPith, image, center, crop_size, shiftX=0, shiftY=0, n_iters=0):
        ## Cropping
        crop_img = image[center[0] - int(crop_size / 2) + shiftX:center[0] + int(crop_size / 2) + shiftX,
                        center[1] - int(crop_size / 2) + shiftY:center[1] + int(crop_size / 2) + shiftY]
        crop_img = cv2.resize(crop_img, (self.patchSize, self.patchSize))[None, :, :]
        if len(crop_img.shape) == 3:
            crop_img = crop_img[:, :, :, None]
        crop_img = crop_img / 255

        ## Prediction
        prediction_crop_pith = modelPith.predict(crop_img, batch_size=1, verbose=0)
        prediction_crop_pith[prediction_crop_pith >= 0.5] = 1
        prediction_crop_pith[prediction_crop_pith < 0.5] = 0

        thres = 0.01 * crop_size
        c1x = np.sum(prediction_crop_pith[:, 0, :, :]) >= thres
        c2x = np.sum(prediction_crop_pith[:, -1, :, :]) >= thres

        stop1, stop2 = False, False
        if c1x:
            shiftX = shiftX - int(0.1 * crop_size)
        elif c2x:
            shiftX = shiftX + int(0.1 * crop_size)
        else:
            stop1 = True

        c1y = np.sum(prediction_crop_pith[:, :, 0, :]) >= thres
        c2y = np.sum(prediction_crop_pith[:, :, -1, :]) >= thres
        if c1y:
            shiftY = shiftY - int(0.1 * crop_size)
        elif c2y:
            shiftY = shiftY + int(0.1 * crop_size)
        else:
            stop2 = True

        if (not (stop1 and stop2)) and n_iters < 2:
            prediction_crop_pith, shiftX, shiftY, crop_size = self.cropAndPredictPith(modelPith, image, center, crop_size, 
                                                                                      shiftX, shiftY, n_iters + 1)

        return prediction_crop_pith, shiftX, shiftY, crop_size


    def predictPith(self, modelPith, image):
        ## Center identification from ring prediction
        prediction_ring = self.predictionRing
        ret = threshold_otsu(self.predictionRing)
        ring_indices = np.where(prediction_ring > ret)
        chose_indices = (0.2 * image.shape[0] < ring_indices[0]) & (ring_indices[0] < 0.8 * image.shape[0]) \
            & (0.2 * image.shape[1] < ring_indices[1]) & (ring_indices[1] < 0.8 * image.shape[1])
        center = int(np.mean(ring_indices[0][chose_indices])), int(np.mean(ring_indices[1][chose_indices]))

        crop_size = int(0.1 * max(self.shape[1], self.shape[0])) * 2 
        
        prediction_crop_pith, shiftX, shiftY, crop_size = self.cropAndPredictPith(modelPith, image, center, crop_size)
    
        prediction_crop_pith = cv2.resize(prediction_crop_pith[0], (crop_size, crop_size))
        prediction_pith = np.zeros((self.shape[0], self.shape[1]))
        prediction_pith[center[0] - int(crop_size / 2) + shiftX:center[0] + int(crop_size / 2) + shiftX,
                        center[1] - int(crop_size / 2) + shiftY:center[1] + int(crop_size / 2) + shiftY] = copy.deepcopy(prediction_crop_pith)
        self.pith = prediction_pith


    def postprocessPith(self):
        one_indice = np.where(self.pith == 1)
        center = int(np.mean(one_indice[0])), int(np.mean(one_indice[1]))
        contours, _ = cv2.findContours(self.pith.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        chosen_contour = np.argmax([cv2.pointPolygonTest(contour, [center[1], center[0]], True) for contour in contours])
        
        self.pithContour = contours[chosen_contour]
        self.center = (int(np.mean(contours[chosen_contour][:, :, 1]) / self.resize), 
                       int(np.mean(contours[chosen_contour][:, :, 0]) / self.resize))
        
    
    def createMask(self, image):
        imageBinary = cv2.resize(image, (int(self.shape[1] / self.resize), int(self.shape[0] / self.resize)))
        ret = threshold_otsu(imageBinary)
        imageBinary[imageBinary < ret] = 0
        imageBinary[imageBinary >= ret] = 1
        imageBinary = 1 - imageBinary

        one_indices = np.where(imageBinary == 1)
        chose_indices = (0.05 * imageBinary.shape[0] < one_indices[0]) & (one_indices[0] < 0.95 * imageBinary.shape[0]) \
            & (0.05 * imageBinary.shape[1] < one_indices[1]) & (one_indices[1] < 0.95 * imageBinary.shape[1])
        indices = np.append(one_indices[1][chose_indices][:, None, None], one_indices[0][chose_indices][:, None, None], axis=-1)

        mask = np.zeros_like(imageBinary)
        cv2.drawContours(mask, [indices], 0, 1, -1)
        self.outerMask = binary_erosion(mask, iterations=self.iterations * 2)

    
    def findEndPoints(self):
        ## Postprocess prediction ring
        prediction_ring = cv2.resize(self.predictionRing, (int(self.shape[1] / self.resize), int(self.shape[0] / self.resize))) * self.outerMask
        height, width = prediction_ring.shape
        
        ## Delete pith area from ring prediction
        pith = cv2.resize(self.pith, (width, height))
        pith_dilated = binary_dilation(pith, iterations=int(0.005 * width))
        prediction_ring = prediction_ring * (1 - pith_dilated)

        dark_point = self.center[0]

        ## Find start and goal points
        light_part = np.mean(prediction_ring[dark_point - int(0.05 * height):dark_point + int(0.05 * height), :], axis=0)
        ret = threshold_otsu(light_part)

        peaks1, _ = find_peaks(light_part[:self.center[1]], height=ret, distance=0.025 * width)
        if len(peaks1) <= 1:
            peaks1, _ = find_peaks(light_part[:self.center[1]], height=threshold_otsu(light_part[:self.center[1]]), 
                                   distance=0.025 * width)
        length1 = self.center[1] - np.sum(1 - self.outerMask[dark_point, :self.center[1]])

        peaks2, _ = find_peaks(light_part[self.center[1]:], height=ret, distance=0.025 * width)
        peaks2 = peaks2 + self.center[1]
        if len(peaks2) <= 1:
            peaks2, _ = find_peaks(light_part[self.center[1]:], height=threshold_otsu(light_part[self.center[1]:]), 
                                   distance=0.025 * width)
            peaks2 = peaks2 + self.center[1]
        length2 = width - self.center[1] - np.sum(1 - self.outerMask[dark_point, self.center[1]:])

        if len(peaks1) < len(peaks2):
            a = copy.deepcopy(peaks1)
            peaks1 = copy.deepcopy(peaks2)
            peaks2 = copy.deepcopy(a)

            a = copy.deepcopy(length1)
            length1 = copy.deepcopy(length2)
            length2 = copy.deepcopy(a)

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
        num_pair = min(len(peaks1), len(peaks2))
        data = []
        remains = list(np.arange(0, len(peaks1)))

        minLength = min(length1, length2)
        peaks1_center = np.abs(peaks1 - self.center[1])[:, None] * minLength / length1
        peaks2_center = np.abs(peaks2 - self.center[1])[None, :] * minLength / length2
        diff_pp = np.abs(peaks1_center - peaks2_center)
        max_value = np.max(diff_pp) + 1

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

            dis1 = np.abs(peaks1[j] - peaks1)
            dis2 = np.abs(peaks2[i] - peaks2)
            thres = (np.min(dis1[dis1 != 0]), np.min(dis2[dis2 != 0]))
            data.append((image_upper, peaks1[j], peaks2[i], dark_point - 1, np.array(self.center), self.resize, thres))
            data.append((image_lower, peaks1[j], peaks2[i], dark_point, np.array(self.center), self.resize, thres))
            remains.remove(j)
            diff_pp[j, :] = max_value
            diff_pp[:, i] = max_value

        for j in remains:
            newPoint = self.center[1] + (self.center[1] - peaks1[j])
            if 0.05 * width <= newPoint < 0.95 * width:
                dis1 = np.abs(peaks1[j] - peaks1)
                dis2 = np.abs(newPoint - peaks2)
                thres = (np.min(dis1[dis1 != 0]), np.min(dis2[dis2 != 0]))
                data.append((image_upper, peaks1[j], newPoint, dark_point - 1, np.array(self.center), self.resize, thres))
                data.append((image_lower, peaks1[j], newPoint, dark_point, np.array(self.center), self.resize, thres))
        
        return data
    

    @classmethod
    def smooth(cls, cor, resize):
        y = savgol_filter(cor[:, 0, 1] * resize, 11, 3, mode='wrap')
        return np.round(np.append(cor[:, 0, 0][:, None, None] * resize, y[:, None, None], axis=-1)).astype(np.int32)


    @staticmethod
    def traceHalfRing(image, peak1, peak2, light_point, center, resize, thres):
        start_point = np.array([light_point, peak1])
        goal_point = np.array([light_point, peak2])
        radius = (np.abs(start_point[1] - center[1]), np.abs(goal_point[1] - center[1]))
        search_algorithm = AStarSearch(image, start_point=start_point, goal_point=goal_point)
        search_algorithm.heuristic_function = CircleHeuristicFunction(image=image, center=center, startPoint=start_point, radius=radius, thres=thres)

        brightest_path = search_algorithm.search()

        result = np.array(search_algorithm.result)[:, 1]
        brightest_path = np.array(search_algorithm.result)[:, 0]

        cor = np.stack([result[:, None], brightest_path[:, None]], axis=-1)
        cor = TreeRingSegmentation.smooth(cor, resize)

        return cor


    def createMaskOfRings(self, cor):
        image_white = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)
        cv2.drawContours(image_white, [np.array(self.pithContour)], 0, 1, self.thickness)
        self.predictedRings.append(np.array(self.pithContour))

        meanIntensity = np.array([np.mean(self.predictionRing[cor[i][:, 0, 1], cor[i][:, 0, 0]]) + 
                                  np.mean(self.predictionRing[cor[i + 1][:, 0, 1], cor[i + 1][:, 0, 0]]) 
                                  for i in range(0, len(cor), 2)])
        results_sorted = np.argsort(meanIntensity)[::-1]
        results_sorted = results_sorted[meanIntensity > 0.5 * np.max(self.predictionRing)]
        
        for i in range(0, len(results_sorted)):
            _image = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)
            j = results_sorted[i]
            ring = np.append(cor[2 * j], cor[2 * j + 1][::-1], axis=0)
            cv2.drawContours(_image, [ring], 0, 1, self.thickness)
            if np.sum(np.bitwise_and(_image, image_white)) == 0:
                image_white = np.bitwise_or(image_white, _image)
                self.predictedRings.append(ring)

        image_white[image_white == 1] = 255

        return image_white


    def segmentImage(self, modelRing, modelPith, image):
        self.predictionRing = self.predictRing(modelRing, image)

        self.predictPith(modelPith, image)
        self.postprocessPith()

        self.createMask(image)

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
    
