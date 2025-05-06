from brightest_path_lib.algorithm import AStarSearch
from brightest_path_lib.heuristic import Heuristic
import copy
import cv2
import multiprocessing
from multiprocessing import Pool
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import linear_sum_assignment
from skimage.filters import threshold_otsu
from tree_ring_analyzer.tiles.tiler import ImageTiler2D
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from skimage.morphology import skeletonize
import skimage



class CircleHeuristicFunction(Heuristic):
    def __init__(self, image, center, startPoint, radius):
        self.radius = radius
        self.image = image

        self.center = center
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.maxValue = np.max(image)
        self.startPoint = startPoint
        

    def estimate_cost_to_goal(self, current_point, goal_point):
        if current_point is None or goal_point is None:
            raise TypeError
        if (len(current_point) == 0 or len(goal_point) == 0) or (len(current_point) != len(goal_point)):
            raise ValueError

        diff0 = np.sqrt(np.sum((current_point[1] - goal_point[1]) ** 2))

        currentRadius = (diff0 / (self.radius[0] + self.radius[1])) * (self.radius[0] - self.radius[1]) + self.radius[1]

        diff1 = np.abs(currentRadius - np.sqrt(np.sum((self.center - current_point) ** 2)))

        if diff1 > 0.2 * currentRadius and self.image[current_point[0], current_point[1]] < 1:
            cost = diff0 + diff1 ** 2
        else:
            cost = diff0

        return cost
    


class TreeRingSegmentation:
    

    def __init__(self, resize=10):
        self.patchSize = 256
        # self.cropSize = 1024
        self.overlap = self.patchSize - 196
        self.batchSize = 8
        self.thickness = 3
        self.iterations = 10
        self.resize = resize

        self.predictionRing = None
        self.pith = None
        self.pithContour = None
        self.outerMask = None
        self.maskRings = None
        self.center = None
        self.shape = None
        self.shapeOriginal = None
        self.centerOriginal = None


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
    

    def postprocessRing(self, modelPostprocess):
        predictionRing = cv2.resize(self.predictionRing.astype(np.uint8), (256, 256))
        postPredictionRing = modelPostprocess.predict(predictionRing[None, :, :, None], batch_size=1, verbose=0)
        postPredictionRing = cv2.resize(postPredictionRing[0, :, :, 0].astype(np.uint8), (self.shape[1], self.shape[0]))
        # plt.subplot(121)
        # plt.imshow(predictionRing)
        # plt.subplot(122)
        # plt.imshow(postPredictionRing)
        # plt.show()
        # raise ValueError
        return postPredictionRing
    

    def cropAndPredictPith(self, modelPith, image, center, cropSize, n_iters=0):
        ## Cropping
        crop_img = image[center[0] - int(cropSize / 2):center[0] + int(cropSize / 2),
                        center[1] - int(cropSize / 2):center[1] + int(cropSize / 2)]
        crop_img = cv2.resize(crop_img, (self.patchSize, self.patchSize))[:, :, None]
        crop_img = crop_img[None, :, :, :]
        crop_img = crop_img / 255

        ## Prediction
        prediction_crop_pith = modelPith.predict(crop_img, batch_size=1, verbose=0)
        prediction_crop_pith[prediction_crop_pith >= 0.5] = 1
        prediction_crop_pith[prediction_crop_pith < 0.5] = 0

        # plt.subplot(121)
        # plt.imshow(crop_img[0, :, :, 0])
        # plt.subplot(122)
        # plt.imshow(prediction_crop_pith[0, :, :, 0])
        # plt.show()

        thres = 0.01 * cropSize
        one_indices = np.where(prediction_crop_pith[0, :, :, 0] == 1)

        c1x = np.sum(prediction_crop_pith[:, 0, :, :]) >= thres
        c2x = np.sum(prediction_crop_pith[:, -1, :, :]) >= thres

        stop1, stop2 = False, False
        new_center = copy.deepcopy(center)
        if (c1x and not c2x) or (c2x and not c1x):
            new_center[0] = center[0] - int(self.cropSize / 2) + int(np.mean(one_indices[0]))
        else:
            stop1 = True

        c1y = np.sum(prediction_crop_pith[:, :, 0, :]) >= thres
        c2y = np.sum(prediction_crop_pith[:, :, -1, :]) >= thres
        if (c1y and not c2y) or (c2y and not c1y):
            new_center[1] = center[1] - int(self.cropSize / 2) + int(np.mean(one_indices[1]))
        else:
            stop2 = True

        if (not (stop1 and stop2)) and n_iters < 2:
            prediction_crop_pith, center = self.cropAndPredictPith(modelPith, image, new_center, n_iters + 1)

        return prediction_crop_pith, center


    def predictPith(self, modelPith, image):
        ## Cropping
        one_indices = np.where(self.outerMask == 1)
        xStart, xEnd = np.min(one_indices[0]) * self.resize, np.max(one_indices[0]) * self.resize
        yStart, yEnd = np.min(one_indices[1]) * self.resize, np.max(one_indices[1]) * self.resize

        ## Identify center
        ret = threshold_otsu(self.predictionRing[xStart:xEnd, yStart:yEnd])
        one_indices = np.where(self.predictionRing[xStart:xEnd, yStart:yEnd] >= ret)
        if len(one_indices[0]):
            center = [xStart + int(np.mean(one_indices[0])), yStart + int(np.mean(one_indices[1]))]
        else:
            center = [int((xStart + xEnd) / 2), int((yStart + yEnd) / 2)]
        
        ## Prediction
        cropSize = int(0.1 * image.shape[0]) * 2
        prediction_pith, center = self.cropAndPredictPith(modelPith, image, center, cropSize)
        self.center = int(center[0] / self.resize), int(center[1] / self.resize)

        self.pith = np.zeros((image.shape[0], image.shape[1]))
        self.pith[center[0] - int(cropSize / 2):center[0] + int(cropSize / 2),
                        center[1] - int(cropSize / 2):center[1] + int(cropSize / 2)] = cv2.resize(prediction_pith[0, :, :, 0], (cropSize, cropSize))


    def postprocessPith(self):
        one_indice = np.where(self.predictionRing >= 1)
        center = int(np.mean(one_indice[0])), int(np.mean(one_indice[1]))
        contours, _ = cv2.findContours(self.pith.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours):
            chosen_contour = np.argmax([cv2.pointPolygonTest(contour, [center[1], center[0]], True) for contour in contours])
            
            pithContour = self.smooth(contours[chosen_contour], 1)
            self.pithContour = pithContour
            self.center = int(np.mean(pithContour[:, :, 1]) / self.resize), int(np.mean(pithContour[:, :, 0]) / self.resize)
        
    
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
        
        # Delete pith area from ring prediction
        pith = cv2.resize(self.pith, (width, height))
        prediction_ring = prediction_ring * (1 - pith)

        dark_point = self.center[0]

        ## Find start and goal points
        light_part = np.mean(prediction_ring[dark_point - int(0.05 * height):dark_point + int(0.05 * height), :], axis=0)
        if np.max(light_part) == 0:
            self.center = int(self.shape[0] / 2 / self.resize), int(self.shape[1] / 2 / self.resize)
            dark_point = self.center[0]
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
        data = []
        remains = list(np.arange(0, len(peaks1)))

        minLength = min(length1, length2)
        peaks1_center = np.abs(peaks1 - self.center[1])[:, None] * minLength / length1
        peaks2_center = np.abs(peaks2 - self.center[1])[None, :] * minLength / length2
        diff_pp = np.abs(peaks1_center - peaks2_center)
        # max_value = np.max(diff_pp) + 1

        image_upper = np.zeros_like(prediction_ring)
        image_upper[:dark_point] = 1
        image_upper = image_upper * prediction_ring
        image_lower = np.zeros_like(prediction_ring)
        image_lower[dark_point:] = 1
        image_lower = image_lower * prediction_ring

        # for k in range(0, num_pair, 1):
        #     min_value = np.min(diff_pp)
        #     if min_value == max_value:
        #         break
        #     j, i = np.where(diff_pp == min_value)
        #     j = j[0]
        #     i = i[0]
            
        #     data.append((image_upper, peaks1[j], peaks2[i], dark_point - 1, np.array(self.center), self.resize))
        #     data.append((image_lower, peaks1[j], peaks2[i], dark_point, np.array(self.center), self.resize))
        #     remains.remove(j)
        #     diff_pp[j, :] = max_value
        #     diff_pp[:, i] = max_value

        row_ind, col_ind = linear_sum_assignment(diff_pp)
        for k in range(len(row_ind)):
            j = row_ind[k]
            i = col_ind[k]
            data.append((image_upper, peaks1[j], peaks2[i], dark_point - 1, np.array(self.center), self.resize))
            data.append((image_lower, peaks1[j], peaks2[i], dark_point, np.array(self.center), self.resize))
            remains.remove(j)

        for j in remains:
            oppositePoint = 2 * self.center[1] - peaks1[j]
            if 0.05 * width <= oppositePoint < 0.95 * width:
                newPoint = np.argmax(light_part[oppositePoint - int(0.025 * width):oppositePoint + int(0.025 * width)]) + oppositePoint - int(0.025 * width)
                data.append((image_upper, peaks1[j], newPoint, dark_point - 1, np.array(self.center), self.resize))
                data.append((image_lower, peaks1[j], newPoint, dark_point, np.array(self.center), self.resize))
        
        return data
    

    @classmethod
    def smooth(cls, cor, resize):
        y = savgol_filter(cor[:, 0, 1] * resize, 11, 3, mode='wrap')
        return np.round(np.append(cor[:, 0, 0][:, None, None] * resize, y[:, None, None], axis=-1)).astype(np.int32)


    @staticmethod
    def traceHalfRing(image, peak1, peak2, light_point, center, resize):
        start_point = np.array([light_point, peak1])
        goal_point = np.array([light_point, peak2])
        radius = (np.abs(start_point[1] - center[1]), np.abs(goal_point[1] - center[1]))
        search_algorithm = AStarSearch(image, start_point=start_point, goal_point=goal_point)
        search_algorithm.heuristic_function = CircleHeuristicFunction(image=image, center=center, startPoint=start_point, 
                                                                      radius=radius)

        brightest_path = search_algorithm.search()

        result = np.array(search_algorithm.result)[:, 1]
        brightest_path = np.array(search_algorithm.result)[:, 0]

        cor = np.stack([result[:, None], brightest_path[:, None]], axis=-1)
        cor = TreeRingSegmentation.smooth(cor, resize)

        return cor


    def createMaskOfRings(self, cor):
        image_white = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)
        if self.pithContour is not None:
            cv2.drawContours(image_white, [np.array(self.pithContour)], 0, 1, self.thickness)

        meanIntensity = np.array([np.mean(self.predictionRing[cor[i][:, 0, 1], cor[i][:, 0, 0]]) + 
                                  np.mean(self.predictionRing[cor[i + 1][:, 0, 1], cor[i + 1][:, 0, 0]]) 
                                  for i in range(0, len(cor), 2)])
        results_sorted = np.argsort(meanIntensity)[::-1]
        
        for i in range(0, len(results_sorted)):
            _image = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)
            j = results_sorted[i]
            ring = np.append(cor[2 * j], cor[2 * j + 1][::-1], axis=0)
            cv2.drawContours(_image, [ring], 0, 1, self.thickness)
            if np.sum(np.bitwise_and(_image, image_white)) == 0:
                image_white = np.bitwise_or(image_white, _image)

        image_white[image_white == 1] = 255

        return image_white


    def resizeImg(self, image, center, inverse=False):
        if not inverse:
            maxLength = max(image.shape[0] - center[0], center[0], image.shape[1] - center[1], center[1])
            resizedImage = np.zeros((2 * maxLength, 2 * maxLength))
            resizedImage[:maxLength, :maxLength] = cv2.resize(image[:center[0], :center[1]], (maxLength, maxLength))
            resizedImage[maxLength:, :maxLength] = cv2.resize(image[center[0]:, :center[1]], (maxLength, maxLength))
            resizedImage[:maxLength, maxLength:] = cv2.resize(image[:center[0], center[1]:], (maxLength, maxLength))
            resizedImage[maxLength:, maxLength:] = cv2.resize(image[center[0]:, center[1]:], (maxLength, maxLength))
        else:
            maxLength = int(image.shape[0] / 2)
            resizedImage = np.zeros(self.shapeOriginal)
            resizedImage[:center[0], :center[1]] = cv2.resize(image[:maxLength, :maxLength], (center[1], center[0]))
            resizedImage[center[0]:, :center[1]] = cv2.resize(image[maxLength:, :maxLength], (center[1], self.shapeOriginal[0] - center[0]))
            resizedImage[:center[0], center[1]:] = cv2.resize(image[:maxLength, maxLength:], (self.shapeOriginal[1] - center[1], center[0]))
            resizedImage[center[0]:, center[1]:] = cv2.resize(image[maxLength:, maxLength:], (self.shapeOriginal[1] - center[1], self.shapeOriginal[0] - center[0]))
        return resizedImage
    

    def segmentImage(self, modelRing, modelPith, image):
        self.predictionRing = self.predictRing(modelRing, image)

        self.createMask(image)

        self.predictPith(modelPith, image)
        self.postprocessPith()

        data = self.findEndPoints()

        with Pool(multiprocessing.cpu_count()) as pool:
            results = pool.starmap(self.traceHalfRing, data)

        self.maskRings = self.createMaskOfRings(results)
    


class Evaluation:

    
    def __init__(self, mask, prediction):
        self.mask = mask
        self.prediction = prediction
        self.createRings()
        self.createSeg()


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
    

    def createRings(self):
        self.prediction[np.bitwise_not(skeletonize(self.prediction))] = 0
        num_labels, labeled_mask = cv2.connectedComponents(self.prediction)
        self.predictedRings = []
        self.predictedRadius = []
        for i in range(1, num_labels):
            predictedRing = np.transpose(np.array(np.where(labeled_mask == i)))
            self.predictedRings.append(predictedRing[:, ::-1])
            self.predictedRadius.append(self.calculateRadius(predictedRing))

        self.mask[np.bitwise_not(skeletonize(self.mask))] = 0
        num_labels, labeled_mask = cv2.connectedComponents(self.mask)
        self.gtRings = []
        self.gtRadius = []
        for i in range(1, num_labels):
            gtRing = np.transpose(np.array(np.where(labeled_mask == i)))
            self.gtRings.append(gtRing[:, ::-1])
            self.gtRadius.append(self.calculateRadius(gtRing))


    def createSeg(self):
        self.predictedSeg = np.zeros_like(self.prediction)
        sortedRadius = np.argsort(self.predictedRadius)[::-1]
        for i in range(len(sortedRadius)):
            cv2.drawContours(self.predictedSeg, [self.predictedRings[sortedRadius[i]][:, None, :]], 0, len(sortedRadius) - i, -1)

        self.gtSeg = np.zeros_like(self.mask)
        sortedRadius = np.argsort(self.gtRadius)[::-1]
        for i in range(len(sortedRadius)):
            cv2.drawContours(self.gtSeg, [self.gtRings[sortedRadius[i]][:, None, :]], 0, len(sortedRadius) - i, -1)


    def evaluateHausdorff(self):
        diffRadius = np.abs(np.array(self.predictedRadius)[:, None] - np.array(self.gtRadius)[None, :])
        max_value = np.max(diffRadius)
        num_pairs = np.min(diffRadius.shape)
        hausdorff = []
        for i in range(0, num_pairs):
            j, k = np.where(diffRadius ==  np.min(diffRadius))
            j = j[0]
            k = k[0]
            hausdorff.append(self.calculateHausdorffDistance(np.squeeze(self.predictedRings[j]), np.squeeze(self.gtRings[k])))
            diffRadius[j, :] = max_value
            diffRadius[:, k] = max_value

        return np.max(hausdorff)


    @staticmethod
    def calculateIoU(a, b):
        return np.sum(np.bitwise_and(a, b)) / np.sum(np.bitwise_or(a, b))
    

    def evaluatemAR(self, iou_levels=np.arange(0.5, 1, 0.05)):
        predicted_uniques = np.unique(self.predictedSeg[self.predictedSeg > 0])
        gt_uniques = np.unique(self.gtSeg[self.gtSeg > 0])

        iou_matrix = np.zeros((len(predicted_uniques), len(gt_uniques)))
        for i in range(len(predicted_uniques)):
            a = predicted_uniques[i]
            for j in range(len(gt_uniques)):
                b = gt_uniques[j]
                iou_matrix[i, j] = self.calculateIoU(self.predictedSeg == a, self.gtSeg == b)

        row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)
        ious = iou_matrix[row_ind, col_ind]
        AR = []
        for level in iou_levels:
            ious_ok = ious >= level
            row_ind1, col_ind1 = row_ind[ious_ok], col_ind[ious_ok]

            TP = len(row_ind1)
            FN = len(iou_matrix.T) - len(row_ind1)
            AR.append(TP / (TP + FN))

        return np.nanmean(AR)
    

    def evaluateARAND(self):
        arand = skimage.metrics.adapted_rand_error(self.gtSeg, self.predictedSeg, ignore_labels=[0])[0]
        return arand
            
