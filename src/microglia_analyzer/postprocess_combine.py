import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import glob
import os
import math
from scipy.ndimage import binary_dilation

def angle_at_point(BA, B, C):
    # ay, ax = A
    by, bx = B
    cy, cx = C
    
    # Compute vectors BA and BC
    BAx, BAy = BA
    BCx, BCy = cx - bx, cy - by
    
    dot_product = BAx * BCx + BAy * BCy

    # Magnitudes
    mag_BA = math.sqrt(BAx**2 + BAy**2)
    mag_BC = math.sqrt(BCx**2 + BCy**2)

    # Avoid division by zero
    if mag_BA == 0 or mag_BC == 0:
        return None  # Undefined angle if a vector has zero length

    # Compute the angle in radians
    angle_rad = math.acos(dot_product / (mag_BA * mag_BC))

    # Convert to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def measure_distance(points):
    x1 = points[:, 0][:, None]
    x2 = points[:, 0][None, :]
    y1 = points[:, 1][:, None]
    y2 = points[:, 1][None, :]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def choose_point(i, dis, dis_center, endpoints, remains, max_value, image, thresh, center):
    if np.all(dis[i, :] == max_value):
        return None, image
    chose_point = np.argmin(dis[i, :])
    # if chose_point == 36 and i == 25:
    #     print('abc')
    img_try = np.zeros_like(image)
    cv2.line(img_try, endpoints[i, 1:3], endpoints[chose_point, 1:3], 1, 1)
    # if np.sum(image * img_try) > 2:
    #     img_try = np.zeros_like(image)
    #     angle_i = angle_at_point((0, 1), center, endpoints[i, 1:3])
    #     angle_chosepoint = angle_at_point((0, 1), center, endpoints[chose_point, 1:3])
    #     cv2.ellipse(img_try, center, np.sort(np.array([int(dis_center[i]), int(dis_center[chose_point])]))[::-1], 
    #                 0, -angle_i, -angle_chosepoint, 1, 1)

    img_new = np.bitwise_or(image, img_try)

    if (chose_point not in remains) \
        or ((not (0.8 < dis_center[i] / dis_center[chose_point] < 1.2)) and (dis_center[i] > thresh or dis_center[chose_point] > thresh)) \
        or (angle_at_point(endpoints[i, 3:], endpoints[i, 1:3], endpoints[chose_point, 1:3]) < 90) \
        or np.sum(image * img_try) > 2:
        dis[i, chose_point] = max_value
        chose_point, img_new = choose_point(i, dis, dis_center, endpoints, remains, max_value, image, thresh, center)

    return chose_point, img_new

if __name__ == '__main__':
    image_list = glob.glob('/home/khietdang/Documents/khiet/treeRing/predictions_segmentation_from_bigDistance/*.tif')
    # image_list = ['/home/khietdang/Documents/khiet/treeRing/predictions_segmentation_from_bigDistance/8 E 2 b_8µm_x50.tif']
    crop_size = 1024
    for image_path in image_list:
        image = tifffile.imread(image_path)

        image[image == 255] = 1
        one_indice = np.where(image == 1)
        center = int(np.mean(one_indice[0])), int(np.mean(one_indice[1]))

        pith = tifffile.imread(image_path.replace('predictions_segmentation_from_bigDistance', 'predictions_pith'))

        pith[pith >= 0.5] = 1
        pith[pith < 0.5] = 0
        pith = cv2.resize(pith.astype(np.uint8), (crop_size, crop_size))
        
        pith_whole = np.zeros_like(image)
        pith_whole[center[0] - int(crop_size / 2):center[0] + int(crop_size / 2),
                   center[1] - int(crop_size / 2):center[1] + int(crop_size / 2)] = copy.deepcopy(pith)
        pith_clear = binary_dilation(pith_whole, iterations=50)

        image_combined = (image * (1 - pith_clear)).astype(np.uint8)
        
        num_labels, labels = cv2.connectedComponents(image_combined)
        endpoints = []
        for i in range(1, num_labels):
            component_indices = np.where(labels == i)
            if len(component_indices[0]) <= 0.01 * image_combined.shape[0]:
                image_combined[labels == i] = 0
                continue
            for j in range(0, len(component_indices[0])):
                chose_index = copy.deepcopy(image_combined[component_indices[0][j] - 1:component_indices[0][j] + 2,
                                    component_indices[1][j] - 1:component_indices[1][j] + 2])
                if np.sum(chose_index) == 2:
                    chose_index[1, 1] = 0
                    one_indices_component = np.where(chose_index == 1)
                    grad = one_indices_component[0][0] - 1, one_indices_component[1][0] - 1
                    endpoints.append([i, component_indices[1][j], component_indices[0][j], grad[0], grad[1]])
        
        image1 = copy.deepcopy(image_combined)
        if len(endpoints):
            endpoints = np.stack(endpoints)

            one_indice = np.where(pith_whole == 1)
            center = int(np.mean(one_indice[0])), int(np.mean(one_indice[1]))
            dis_center = np.sqrt((center[0] - endpoints[:, 2]) ** 2 + (center[1] - endpoints[:, 1]) ** 2)

            dis = measure_distance(endpoints[:, 1:3])
            max_value = np.max(dis) + 1
            dis[dis == 0] = max_value
            
            remains = list(range(0, len(dis)))
            while len(remains) >= 2:
                i = remains[0]
                # if i == 5:
                #     print('abc')
                chose_point, image1 = choose_point(i, dis, dis_center, endpoints, remains, max_value, 
                                                   image1, 0.10 * image.shape[0], [center[1], center[0]])
                
                if chose_point is None:
                    dis[i, :] = copy.deepcopy(max_value)
                    dis[:, i] = copy.deepcopy(max_value)
                    remains.remove(i)
                    continue
                
                dis[i, :] = copy.deepcopy(max_value)
                dis[:, i] = copy.deepcopy(max_value)
                dis[chose_point, :] = copy.deepcopy(max_value)
                dis[:, chose_point] = copy.deepcopy(max_value)
                remains.remove(i)
                remains.remove(chose_point)
                # print(i, chose_point)

            image_final = np.zeros_like(image_combined)
            contours, hierarchy = cv2.findContours(image1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(0, len(contours)):
                if hierarchy[0][i][3] == -1:
                    hull = cv2.convexHull(contours[i])
                    cv2.drawContours(image_final, [hull], 0, 1, 1)

            contours, _ = cv2.findContours(pith_whole, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            longest_contour = np.argmax(np.array([len(contour) for contour in contours]))
            cv2.drawContours(image_final, contours, longest_contour, 1, 1)

            # plt.subplot(1, 3, 1)
            # plt.imshow(image, cmap='gray')
            # plt.subplot(1, 3, 2)
            # plt.imshow(labels, cmap='gray')
            # for i in [5]:
            #     plt.plot(endpoints[i, 1], endpoints[i, 2], 'bo')
            # # plt.subplot(1, 3, 3)
            # plt.imshow(image1)
            # plt.plot(int(center[1]), int(center[0]), 'bo')
            # plt.show()

        image_final[image_final == 1] = 255
        tifffile.imwrite('/home/khietdang/Documents/khiet/treeRing/endpoints_bigDistance_pith/' + os.path.basename(image_path),
                         image_final.astype(np.uint8))