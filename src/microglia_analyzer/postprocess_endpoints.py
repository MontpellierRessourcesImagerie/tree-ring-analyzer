import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import glob
import os
import math

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
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def choose_point(i, dis, dis_center, endpoints, remains, max_value, image, thresh):
    if np.all(dis[i, :] == max_value):
        return None
    chose_point = np.argmin(dis[i, :])
    # if chose_point == 36 and i == 25:
    #     print('abc')
    img_try = np.zeros_like(image)
    cv2.line(img_try, endpoints[i, 1:3], endpoints[chose_point, 1:3], 1, 1)

    if (chose_point not in remains) \
        or ((not (0.8 < dis_center[i] / dis_center[chose_point] < 1.2)) and (dis_center[i] > thresh or dis_center[chose_point] > thresh)) \
        or (angle_at_point(endpoints[i, 3:], endpoints[i, 1:3], endpoints[chose_point, 1:3]) < 90) \
        or np.sum(image * img_try) > 2:
        dis[i, chose_point] = max_value
        chose_point = choose_point(i, dis, dis_center, endpoints, remains, max_value, image, thresh)

    return chose_point

if __name__ == '__main__':
    image_list = glob.glob('/home/khietdang/Documents/khiet/treeRing/predictions_segmentation_from_bigDistance/*.tif')
    # image_list = ['/home/khietdang/Documents/khiet/treeRing/predictions_segmentation_from_bigDistance/4 E 3 t_8µm_x50 2.tif']
    for image_path in image_list:
        image = tifffile.imread(image_path)
        image[image == 255] = 1

        num_labels, labels = cv2.connectedComponents(image)
        endpoints = []
        for i in range(1, num_labels):
            component_indices = np.where(labels == i)
            if len(component_indices[0]) <= 0.01 * image.shape[0]:
                image[labels == i] = 0
                continue
            for j in range(0, len(component_indices[0])):
                chose_index = copy.deepcopy(image[component_indices[0][j] - 1:component_indices[0][j] + 2,
                                    component_indices[1][j] - 1:component_indices[1][j] + 2])
                if np.sum(chose_index) == 2:
                    chose_index[1, 1] = 0
                    one_indices_component = np.where(chose_index == 1)
                    grad = one_indices_component[0][0] - 1, one_indices_component[1][0] - 1
                    endpoints.append([i, component_indices[1][j], component_indices[0][j], grad[0], grad[1]])
        
        image_new = copy.deepcopy(image)
        if len(endpoints):
            endpoints = np.stack(endpoints)

            one_indice = np.where(image == 1)
            center = np.average(one_indice[0]), np.average(one_indice[1])
            dis_center = np.sqrt((center[0] - endpoints[:, 2]) ** 2 + (center[1] - endpoints[:, 1]) ** 2)

            dis = measure_distance(endpoints[:, 1:3])
            max_value = np.max(dis) + 1
            dis[dis == 0] = max_value
            
            remains = list(range(0, len(dis)))
            while len(remains) >= 2:
                i = remains[0]
                # if i == 5:
                #     print('abc')
                chose_point = choose_point(i, dis, dis_center, endpoints, remains, max_value, image_new, 0.10 * image.shape[0])
                
                if chose_point is None:
                    dis[i, :] = copy.deepcopy(max_value)
                    dis[:, i] = copy.deepcopy(max_value)
                    remains.remove(i)
                    continue
                
                cv2.line(image_new, endpoints[i, 1:3], endpoints[chose_point, 1:3], 1, 1)
                dis[i, :] = copy.deepcopy(max_value)
                dis[:, i] = copy.deepcopy(max_value)
                dis[chose_point, :] = copy.deepcopy(max_value)
                dis[:, chose_point] = copy.deepcopy(max_value)
                remains.remove(i)
                remains.remove(chose_point)
                # print(i, chose_point)

            # plt.subplot(1, 3, 1)
            # plt.imshow(image, cmap='gray')
            # plt.subplot(1, 3, 2)
            # plt.imshow(labels, cmap='gray')
            # for i in [5]:
            #     plt.plot(endpoints[i, 1], endpoints[i, 2], 'bo')
            # # plt.subplot(1, 3, 3)
            # plt.imshow(image_new)
            # plt.plot(int(center[1]), int(center[0]), 'bo')
            # plt.show()

        image_new[image_new == 1] = 255
        tifffile.imwrite('/home/khietdang/Documents/khiet/treeRing/endpoints_bigDistance/' + os.path.basename(image_path),
                         image_new.astype(np.uint8))