from brightest_path_lib.algorithm import AStarSearch
from brightest_path_lib.heuristic import Heuristic
import copy
import cv2
from multiprocessing import Pool
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import linear_sum_assignment
from skimage.filters import threshold_otsu
from tree_ring_analyzer.tiles.tiler import ImageTiler2D
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from skimage.morphology import skeletonize
from skimage.metrics import adapted_rand_error, hausdorff_distance
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from polylabel import polylabel



class CircleHeuristicFunction(Heuristic):
    def __init__(self, image, center, startPoint, radius, lossType='H02'):
        self.radius = radius
        self.image = image

        self.center = center
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.maxValue = np.max(image)
        self.startPoint = startPoint
        self.lossType = lossType

    def estimate_cost_to_goal(self, current_point, goal_point):
        if current_point is None or goal_point is None:
            raise TypeError
        if (len(current_point) == 0 or len(goal_point) == 0) or (len(current_point) != len(goal_point)):
            raise ValueError

        h0 = np.sqrt(np.sum((current_point - goal_point) ** 2))
        currentRadius = (h0 / (self.radius[0] + self.radius[1])) * (self.radius[0] - self.radius[1]) + self.radius[1]

        if self.lossType == 'H0':
            cost = h0
        else:
            h1 = np.abs(currentRadius - np.sqrt(np.sum((self.center - current_point) ** 2)))
            h2 = np.abs(np.sum((current_point - goal_point) * (current_point - self.startPoint)))

            if (h1 > 0.2 * currentRadius and self.image[current_point[0], current_point[1]] < 1):
                if self.lossType == 'H01':
                    cost = h0 + (h1 ** 2)
                elif self.lossType == 'H02':
                    cost = h0 + h2
                else:
                    raise ValueError
            else:
                cost = h0

        return cost
    


class TreeRingSegmentation:
    

    def __init__(self, resize=5, pithWhole=False, rotate=True, lossType='H0', removeRing=True, thickness=1):
        self.patchSize = 256
        self.overlap = self.patchSize - 196
        self.batchSize = 8
        self.thickness = thickness
        self.iterations = 10
        self.resize = resize
        self.pithWhole = pithWhole
        self.rotate = rotate
        self.angle = 0
        self.lossType = lossType
        self.removeRing = removeRing

        self.predictionRing = None
        self.pith = None
        self.pithContour = None
        self.outerMask = None
        self.maskRings = None
        self.center = None
        self.shape = None
        self.shapeOriginal = None
        self.centerOriginal = None
        self.predictedRings = []


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
    

    def cropAndPredictPith(self, modelPith, image, center, cropSize, n_iters=0):
        ## Cropping
        crop_img = image[center[0] - int(cropSize / 2):center[0] + int(cropSize / 2),
                        center[1] - int(cropSize / 2):center[1] + int(cropSize / 2)]
        crop_img = cv2.resize(crop_img, (self.patchSize, self.patchSize))
        if len(crop_img.shape) == 2:
            crop_img = crop_img[None, :, :, None]
        elif len(crop_img.shape) == 3:
            crop_img = crop_img[None, :, :, :]
        crop_img = crop_img / 255

        ## Prediction
        prediction_crop_pith = modelPith.predict(crop_img, batch_size=1, verbose=0)
        ret = threshold_otsu(prediction_crop_pith)
        prediction_crop_pith[prediction_crop_pith > ret] = 1
        prediction_crop_pith[prediction_crop_pith <= ret] = 0
        prediction_crop_pith = cv2.resize(prediction_crop_pith[0, :, :, 0], (cropSize, cropSize))

        thres = 0.01 * cropSize
        one_indices = np.where(prediction_crop_pith == 1)
        if not len(one_indices[0]):
            one_indices = (int(prediction_crop_pith.shape[0] / 2), int(prediction_crop_pith.shape[1] / 2))

        c1x = np.sum(prediction_crop_pith[0, :]) >= thres
        c2x = np.sum(prediction_crop_pith[-1, :]) >= thres

        stop1, stop2, stop3 = False, False, False
        new_center = copy.deepcopy(center)
        if (c1x and not c2x) or (c2x and not c1x):
            new_center[0] = center[0] - int(cropSize / 2) + int(np.mean(one_indices[0]))
        else:
            stop1 = True

        c1y = np.sum(prediction_crop_pith[:, 0]) >= thres
        c2y = np.sum(prediction_crop_pith[:, -1]) >= thres
        if (c1y and not c2y) or (c2y and not c1y):
            new_center[1] = center[1] - int(cropSize / 2) + int(np.mean(one_indices[1]))
        else:
            stop2 = True

        if np.sum(prediction_crop_pith) == 0:
            new_cropSize = cropSize * 2
            new_cropSize = int(min(cropSize / 2, center[0], image.shape[0] - center[0], center[1], image.shape[1] - center[1])) * 2
        else:
            new_cropSize = copy.deepcopy(cropSize)
            stop3 = True

        if (not (stop1 and stop2 and stop3)) and n_iters < 2:
            prediction_crop_pith, center, cropSize = self.cropAndPredictPith(modelPith, image, new_center, new_cropSize, n_iters + 1)

        return prediction_crop_pith, center, cropSize


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
        
        if self.pithWhole:
            resizedImage = cv2.resize(image, (self.patchSize, self.patchSize)) / 255
            if len(resizedImage.shape) == 2:
                resizedImage = resizedImage[None, :, :, None]
            elif len(resizedImage.shape) == 3:
                resizedImage = resizedImage[None, :, :, :]
            prediction_pith = modelPith.predict(resizedImage, batch_size=1, verbose=0)
            ret = threshold_otsu(prediction_pith)
            prediction_pith[prediction_pith > ret] = 1
            prediction_pith[prediction_pith <= ret] = 0
            self.pith = cv2.resize(prediction_pith[0], (image.shape[1], image.shape[0]))

        else:
            cropSize = int(0.1 * image.shape[0]) * 2
            prediction_pith, center, cropSize = self.cropAndPredictPith(modelPith, image, center, cropSize)

            self.pith = np.zeros((image.shape[0], image.shape[1]))
            self.pith[center[0] - int(cropSize / 2):center[0] + int(cropSize / 2),
                            center[1] - int(cropSize / 2):center[1] + int(cropSize / 2)] = prediction_pith
            
        self.center = int(center[0] / self.resize), int(center[1] / self.resize)


    def postprocessPith(self):
        ret = threshold_otsu(self.predictionRing)
        one_indice = np.where(self.predictionRing > ret)
        center = int(np.mean(one_indice[0])), int(np.mean(one_indice[1]))
        contours, _ = cv2.findContours(self.pith.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours):
            chosen_contour = np.argmax([cv2.pointPolygonTest(contour, [center[1], center[0]], True) for contour in contours])
            
            pithContour = self.smooth(contours[chosen_contour], 1)
            self.pithContour = pithContour
            polygon = [[point[0].tolist() for point in pithContour]]
            self.center = polylabel(polygon)  # Returns (x, y)
            self.center = int(self.center[1] / self.resize), int(self.center[0] / self.resize)

        
    def createMask(self, image):
        if image.shape[-1] == 3:
            image = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2])[:, :, None]
        imageBinary = cv2.resize(image, (int(self.shape[1] / self.resize), int(self.shape[0] / self.resize)))
        ret = threshold_otsu(imageBinary)
        imageBinary[imageBinary < ret] = 0
        imageBinary[imageBinary >= ret] = 1
        imageBinary = 1 - imageBinary

        one_indices = np.where(imageBinary == 1)
        if not len(one_indices[0]):
            radius = 0.9 * min(imageBinary.shape[0], imageBinary.shape[1])
            s = np.linspace(0, 2 * np.pi, int(np.pi * radius))
            r = imageBinary.shape[0] / 2 + radius * np.sin(s)
            c = imageBinary.shape[1] / 2 + radius * np.cos(s)
            indices = np.array([r, c]).T
            indices = indices[:, None, :].astype(np.int32)
        else:
            chose_indices = (0.05 * imageBinary.shape[0] < one_indices[0]) & (one_indices[0] < 0.95 * imageBinary.shape[0]) \
                & (0.05 * imageBinary.shape[1] < one_indices[1]) & (one_indices[1] < 0.95 * imageBinary.shape[1])
            indices = np.append(one_indices[1][chose_indices][:, None, None], one_indices[0][chose_indices][:, None, None], axis=-1)

        mask = np.zeros_like(imageBinary)
        cv2.drawContours(mask, [indices], 0, 1, -1)
        self.outerMask = binary_erosion(mask, iterations=self.iterations * 2)

    
    def getPeaks(self, prediction_ring, center, height, width):
        light_part = np.mean(prediction_ring[center[0] - int(0.05 * height):center[0] + int(0.05 * height), :], axis=0)

        ret = threshold_otsu(light_part)

        peaks1, _ = find_peaks(light_part[:center[1]], height=ret, distance=0.025 * width)
        if len(peaks1) < 1:
            peaks1, _ = find_peaks(light_part[:center[1]], height=threshold_otsu(light_part[:center[1]]), 
                                   distance=0.025 * width)
        
        peaks2, _ = find_peaks(light_part[center[1]:], height=ret, distance=0.025 * width)
        peaks2 = peaks2 + center[1]
        if len(peaks2) < 1:
            peaks2, _ = find_peaks(light_part[center[1]:], height=threshold_otsu(light_part[center[1]:]), 
                                   distance=0.025 * width)
            peaks2 = peaks2 + center[1]
        
        return peaks1, peaks2, light_part


    def findEndPoints(self):
        ## Postprocess prediction ring
        prediction_ring = cv2.resize(self.predictionRing, (int(self.shape[1] / self.resize), int(self.shape[0] / self.resize))) * self.outerMask
        height, width = prediction_ring.shape
        
        # Delete pith area from ring prediction
        pith = cv2.resize(self.pith, (width, height))
        prediction_ring = prediction_ring * (1 - pith)

        ## Find start and goal points
        length1 = self.center[1] - np.sum(1 - self.outerMask[self.center[0], :self.center[1]])
        length2 = width - self.center[1] - np.sum(1 - self.outerMask[self.center[0], self.center[1]:])

        angle = 0
        _prediction_ring = copy.deepcopy(prediction_ring)
        _outerMask = self.outerMask.astype(np.uint8)
        _center = copy.deepcopy(self.center)
        center = copy.deepcopy(self.center)
        peaks1, peaks2 = [], []
        self.centerRotate = int(height / 2), int(width / 2)
        rotation_matrix = cv2.getRotationMatrix2D((self.centerRotate[1], self.centerRotate[0]), 45, scale=1)
        for i in range(0, 1 if not self.rotate else 4):
            _peaks1, _peaks2, _light_part = self.getPeaks(_prediction_ring, _center, height, width)
            if len(_peaks1) - len(peaks1) >= 1 and len(_peaks2) - len(peaks2) >= 1:
                peaks1 = copy.deepcopy(_peaks1)
                peaks2 = copy.deepcopy(_peaks2)
                center = copy.deepcopy(_center)
                length1 = self.center[1] - np.sum(1 - _outerMask[_center[0], :_center[1]])
                length2 = width - self.center[1] - np.sum(1 - _outerMask[_center[0], _center[1]:])
                self.angle = angle
                prediction_ring = copy.deepcopy(_prediction_ring)
                light_part = copy.deepcopy(_light_part)

            angle += 45
            _prediction_ring = cv2.warpAffine(_prediction_ring, rotation_matrix, (width, height))
            _outerMask = cv2.warpAffine(_outerMask, rotation_matrix, (width, height))
            _center = self.rotatePoint(rotation_matrix, _center[1], _center[0]).astype(np.int32)[::-1]


        if len(peaks1) < len(peaks2):
            a = copy.deepcopy(peaks1)
            peaks1 = copy.deepcopy(peaks2)
            peaks2 = copy.deepcopy(a)

            a = copy.deepcopy(length1)
            length1 = copy.deepcopy(length2)
            length2 = copy.deepcopy(a)

        ## Catch up the pairs of start points and goal points
        remains = list(np.arange(0, len(peaks1)))

        minLength = min(length1, length2)
        peaks1_center = np.abs(peaks1 - center[1])[:, None] * minLength / length1
        peaks2_center = np.abs(peaks2 - center[1])[None, :] * minLength / length2
        diff_pp = np.abs(peaks1_center - peaks2_center)

        image_upper = np.zeros_like(prediction_ring)
        image_upper[:center[0]] = 1
        image_upper = image_upper * prediction_ring
        image_lower = np.zeros_like(prediction_ring)
        image_lower[center[0]:] = 1
        image_lower = image_lower * prediction_ring

        row_ind, col_ind = linear_sum_assignment(diff_pp)
        sort_ind = np.argsort(light_part[np.array(peaks1)[row_ind]] + light_part[np.array(peaks2)[col_ind]])[::-1]
        row_ind = row_ind[sort_ind]
        col_ind = col_ind[sort_ind]

        results = []
        for k in range(len(row_ind)):
            j = row_ind[k]
            i = col_ind[k]
            data = []
            if light_part[peaks1[j]] >= light_part[peaks2[i]]:
                data.append((image_upper, peaks1[j], peaks2[i], center[0] - 1, np.array(center), self.resize, self.lossType))
                data.append((image_lower, peaks1[j], peaks2[i], center[0], np.array(center), self.resize, self.lossType))
            else:
                data.append((image_upper, peaks2[i], peaks1[j], center[0] - 1, np.array(center), self.resize, self.lossType))
                data.append((image_lower, peaks2[i], peaks1[j], center[0], np.array(center), self.resize, self.lossType))
            remains.remove(j)

            with Pool(2) as pool:
                    result = pool.starmap(self.traceHalfRing, data)

            results += result
            
            if self.removeRing:
                mask = self.createRingMask(result)

                image_upper *= mask
                image_lower *= mask

                if np.sum(image_lower) == 0 or np.sum(image_upper) == 0:
                    break

        for j in remains:
            if np.sum(image_lower) == 0 or np.sum(image_upper) == 0:
                break
            oppositePoint = 2 * self.center[1] - peaks1[j]
            if 0.05 * width <= oppositePoint < 0.95 * width:
                newPoint = np.argmax(light_part[oppositePoint - int(0.025 * width):oppositePoint + int(0.025 * width)]) + oppositePoint - int(0.025 * width)
                data = []
                data.append((image_upper, peaks1[j], newPoint, center[0] - 1, np.array(center), self.resize, self.lossType))
                data.append((image_lower, peaks1[j], newPoint, center[0], np.array(center), self.resize, self.lossType))

                with Pool(2) as pool:
                    result = pool.starmap(self.traceHalfRing, data)

                results += result
                
                if self.removeRing:
                    mask = self.createRingMask(result)

                    image_upper *= mask
                    image_lower *= mask
   
        return results
    

    @classmethod
    def smooth(cls, cor, resize):
        y = savgol_filter(cor[:, 0, 1] * resize, 11, 3, mode='wrap')
        return np.round(np.append(cor[:, 0, 0][:, None, None] * resize, y[:, None, None], axis=-1)).astype(np.int32)


    @staticmethod
    def traceHalfRing(image, peak1, peak2, light_point, center, resize, lossType='H02'):
        start_point = np.array([light_point, peak1])
        goal_point = np.array([light_point, peak2])
        radius = (np.abs(start_point[1] - center[1]), np.abs(goal_point[1] - center[1]))

        search_algorithm = AStarSearch(image, start_point=start_point, goal_point=goal_point)
        search_algorithm.heuristic_function = CircleHeuristicFunction(image=image, center=center, startPoint=start_point, 
                                                                      radius=radius, lossType=lossType)

        brightest_path = search_algorithm.search()

        result = np.array(search_algorithm.result)[:, 1]
        brightest_path = np.array(search_algorithm.result)[:, 0]

        cor = np.stack([result[:, None], brightest_path[:, None]], axis=-1)
        cor = TreeRingSegmentation.smooth(cor, resize)

        return cor
    

    def rotateContour(self, rotation_matrix, contours):
        rotated_contours = []
        for contour in contours:
            rotated_contour = []
            for point in contour:
                x, y = point[0]
                new_point = self.rotatePoint(rotation_matrix, x, y)
                rotated_contour.append(new_point)
            rotated_contours.append(np.array(rotated_contour, dtype=np.int32).reshape((-1, 1, 2)))
        return rotated_contours
    

    def rotatePoint(self, rotation_matrix, x, y):
        return np.dot(rotation_matrix, [x, y, 1])


    def createMaskOfRings(self, cor):
        image_white = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)
        if self.pithContour is not None:
            cv2.drawContours(image_white, [np.array(self.pithContour)], 0, 1, self.thickness)
            self.predictedRings.append(np.array(self.pithContour))

        meanIntensity = np.array([np.mean(self.predictionRing[cor[i][:, 0, 1], cor[i][:, 0, 0]]) + 
                                  np.mean(self.predictionRing[cor[i + 1][:, 0, 1], cor[i + 1][:, 0, 0]]) 
                                  for i in range(0, len(cor), 2)])
        results_sorted = np.argsort(meanIntensity)[::-1]
        ret = threshold_otsu(self.predictionRing[self.predictionRing >= np.max(self.predictionRing) / 10])
        if np.max(meanIntensity) > ret:
            results_sorted = results_sorted[meanIntensity[results_sorted] > ret]
        
        for i in range(0, len(results_sorted)):
            _image = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)
            j = results_sorted[i]
            ring = np.append(cor[2 * j], cor[2 * j + 1][::-1], axis=0)
            cv2.drawContours(_image, [ring], 0, 1, self.thickness)
            if np.sum(np.bitwise_and(_image, image_white)) == 0 \
            and cv2.pointPolygonTest(ring, (self.center[1] * self.resize, self.center[0] * self.resize), True) > 0:
                image_white = np.bitwise_or(image_white, _image)
                self.predictedRings.append(ring)

        image_white[image_white == 1] = 255

        return image_white
    
    
    def createRingMask(self, contours):
        contour = np.append(contours[0], contours[1], axis=0)
        hull = cv2.convexHull(contour)

        mask = np.zeros((self.outerMask.shape[0], self.outerMask.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask, [(hull / self.resize).astype(np.int32)], 0, 1, int(0.05 * mask.shape[1]))
        return 1 - mask


    def segmentImage(self, modelRing, modelPith, image):
        self.predictionRing = self.predictRing(modelRing, image)

        self.createMask(image)

        self.predictPith(modelPith, image)
        self.postprocessPith()

        results = self.findEndPoints()

        if self.angle > 0:
            rotation_matrix = cv2.getRotationMatrix2D((self.centerRotate[1] * self.resize, self.centerRotate[0] * self.resize), 
                                                      -self.angle, scale=1)
            results = self.rotateContour(rotation_matrix, results)

        self.maskRings = self.createMaskOfRings(results)
    


class Evaluation:

    
    def __init__(self, mask, prediction, gtRings=None, predictedRings=None):
        self.mask = mask
        self.prediction = prediction
        self.predictedRings = predictedRings
        self.gtRings = gtRings
        
        self.createRings()
        self.createSeg()


    def createRings(self):
        if self.predictedRings is None:
            self.prediction[np.bitwise_not(skeletonize(self.prediction))] = 0
            num_labels, labeled_mask = cv2.connectedComponents(self.prediction)
            self.predictedRings = []
            for i in range(1, num_labels):
                predictedRing = np.transpose(np.array(np.where(labeled_mask == i)))
                self.predictedRings.append(predictedRing[:, ::-1])

        if self.gtRings is None:
            self.mask[np.bitwise_not(skeletonize(self.mask))] = 0
            num_labels, labeled_mask = cv2.connectedComponents(self.mask)
            self.gtRings = []
            for i in range(1, num_labels):
                gtRing = np.transpose(np.array(np.where(labeled_mask == i)))
                self.gtRings.append(gtRing[:, ::-1])


    def createSeg(self):
        self.predictedSeg = np.zeros_like(self.prediction)
        sortedRadius = np.argsort([len(ring) for ring in self.predictedRings])[::-1]
        for i in range(len(sortedRadius)):
            cv2.drawContours(self.predictedSeg, [self.predictedRings[sortedRadius[i]][:, None, :]], 0, len(sortedRadius) - i, -1)

        self.gtSeg = np.zeros_like(self.mask)
        sortedRadius = np.argsort([len(ring) for ring in self.gtRings])[::-1]
        for i in range(len(sortedRadius)):
            cv2.drawContours(self.gtSeg, [self.gtRings[sortedRadius[i]][:, None, :]], 0, len(sortedRadius) - i, -1)


    def evaluateHausdorff(self):
        return hausdorff_distance(self.mask, self.prediction, method='modified')


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
        arand = adapted_rand_error(self.gtSeg, self.predictedSeg, ignore_labels=[0])[0]
        return arand
            

    def evaluateRPFA(self):
        gtRingSeg = skeletonize(self.mask).astype(np.uint8).flatten()

        predictedRingSeg = np.zeros_like(self.prediction)
        predictedRingSeg[self.prediction == 255] = 1
        predictedRingSeg = predictedRingSeg.flatten()

        recall = recall_score(gtRingSeg, predictedRingSeg, zero_division=0)
        precision = precision_score(gtRingSeg, predictedRingSeg, zero_division=0)
        f1 = f1_score(gtRingSeg, predictedRingSeg, zero_division=0)
        acc = accuracy_score(gtRingSeg, predictedRingSeg)
        return recall, precision, f1, acc
