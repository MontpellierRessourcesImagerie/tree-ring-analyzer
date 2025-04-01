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

def findCircle(x1, y1, x2, y2, x3, y3) :
    x12 = x1 - x2; 
    x13 = x1 - x3; 

    y12 = y1 - y2; 
    y13 = y1 - y3; 

    y31 = y3 - y1; 
    y21 = y2 - y1; 

    x31 = x3 - x1; 
    x21 = x2 - x1; 

    # x1^2 - x3^2 
    sx13 = pow(x1, 2) - pow(x3, 2); 

    # y1^2 - y3^2 
    sy13 = pow(y1, 2) - pow(y3, 2); 

    sx21 = pow(x2, 2) - pow(x1, 2); 
    sy21 = pow(y2, 2) - pow(y1, 2); 

    f = (((sx13) * (x12) + (sy13) * 
          (x12) + (sx21) * (x13) + 
          (sy21) * (x13)) // (2 * 
          ((y31) * (x12) - (y21) * (x13))));
            
    g = (((sx13) * (y12) + (sy13) * (y12) + 
          (sx21) * (y13) + (sy21) * (y13)) // 
          (2 * ((x31) * (y12) - (x21) * (y13)))); 

    c = (-pow(x1, 2) - pow(y1, 2) - 
         2 * g * x1 - 2 * f * y1); 

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0 
    # where centre is (h = -g, k = -f) and 
    # radius r as r^2 = h^2 + k^2 - c 
    h = -g; 
    k = -f; 
    sqr_of_r = h * h + k * k - c; 

    # r is the radius 
    r = round(np.sqrt(sqr_of_r), 5); 

    return (h, k), r

def recta(x1, y1, x2, y2):
    a = (y1 - y2) / (x1 - x2)
    b = y1 - a * x1
    return (a, b)

def curva_b(xa, ya, xb, yb, xc, yc):
    (x1, y1, x2, y2) = (xa, ya, xb, yb)
    (a1, b1) = recta(xa, ya, xb, yb)
    (a2, b2) = recta(xb, yb, xc, yc)
    puntos = []

    for i in range(0, 100000):
        if x1 == x2:
            continue
        else:
            (a, b) = recta(x1, y1, x2, y2)
        x = i*(x2 - x1)/100000 + x1
        y = a*x + b
        puntos.append((int(x),int(y)))
        x1 += (xb - xa)/100000
        y1 = a1*x1 + b1
        x2 += (xc - xb)/100000
        y2 = a2*x2 + b2
    return puntos

def plot_curve(point1, point2, center, num=100000, thresh=100):
    x1, y1 = point1
    x2, y2 = point2
    cx, cy = center
    
    nx1, ny1 = - y1 + cy,  x1 - cx
    nx2, ny2 = - y2 + cy,  x2 - cx

    nx1 = copy.deepcopy(thresh) if nx1 < thresh else nx1
    ny1 = copy.deepcopy(thresh) if ny1 < thresh else ny1
    nx2 = copy.deepcopy(thresh) if nx2 < thresh else nx2
    ny2 = copy.deepcopy(thresh) if ny2 < thresh else ny2

    tanx1 = (nx1 / ny1) / 2
    tany1 = (ny1 / nx1) / 2

    tanx2 = (nx2 / ny2) / 2
    tany2 = (ny2 / nx2) / 2
    
    cor1 = np.zeros((num, 1, 2))
    cor1[0, 0, 0] = x1
    cor1[0, 0, 1] = y1
    
    cor2 = np.zeros((num, 1, 2))
    cor2[0, 0, 0] = x2
    cor2[0, 0, 1] = y2

    a, b = recta(x1, y1, x2, y2)
    direct = - 1 if (a * cx + b) > cy else 1

    for i in range(1, num):
        _x = (cor2[i - 1, 0, 0] - cor1[i - 1, 0, 0]) / (2*(num - i))
        _y = (cor2[i - 1, 0, 1] - cor1[i - 1, 0, 1]) / (2*(num - i))

        if tanx1 > math.tan(math.pi / 4):
            _x1 = _x + direct * _y * tanx1 * (num - i) / num
            _y1 = _y + direct * _x * tany1 * (i) / num
        else:
            _x1 = _x + direct * _y * tanx1 * (i) / num
            _y1 = _y + direct * _x * tany1 * (num - i) / num
        cor1[i, 0, 0] = cor1[i - 1, 0, 0] + _x1
        cor1[i, 0, 1] = cor1[i - 1, 0, 1] + _y1

        if tanx2 > math.tan(math.pi / 4):
            _x2 = - _x - direct * _y * tanx2 * (num - i) / num
            _y2 = - _y - direct * _x * tany2 * (i) / num
        else:
            _x2 = - _x - direct * _y * tanx2 * (i) / num
            _y2 = - _y - direct * _x * tany2 * (num - i) / num
        cor2[i, 0, 0] = cor2[i - 1, 0, 0] + _x2
        cor2[i, 0, 1] = cor2[i - 1, 0, 1] + _y2

    cor = np.append(cor1, cor2[::-1], axis=0)
    return cor.astype(int)


def choose_point(i, dis, dis_center, endpoints, remains, max_value, image, thresh, center):
    if np.all(dis[i, :] == max_value):
        return None
    if i == 32:
        print('abc')
    chose_point = np.argmin(dis[i, :])
    img_try = np.zeros_like(image)
    cv2.line(img_try, endpoints[i, 1:3], endpoints[chose_point, 1:3], 1, 1)

    if (chose_point not in remains) \
        or ((not (0.8 < dis_center[i] / dis_center[chose_point] < 1.2)) and (dis_center[i] > thresh or dis_center[chose_point] > thresh)) \
        or (angle_at_point(endpoints[i, 3:], endpoints[i, 1:3], endpoints[chose_point, 1:3]) < 80) \
        or np.sum(image * img_try) > 2:
        dis[i, chose_point] = max_value
        chose_point = choose_point(i, dis, dis_center, endpoints, remains, max_value, image, thresh, center)

    return chose_point

if __name__ == '__main__':
    # image_list = glob.glob('/home/khietdang/Documents/khiet/treeRing/predictions_segmentation_from_bigDistance/*.tif')
    image_list = ['/home/khietdang/Documents/khiet/treeRing/predictions_segmentation_from_bigDistance/4 E 3 t_8µm_x50 2.tif']
    # image_list = ['/home/khietdang/Documents/khiet/treeRing/bad_results/29(8)_x50_8µm_bigDistance-1-lbl-sizeFilt-Closing.tif']
    crop_size = 1024
    for image_path in image_list:
        image = tifffile.imread(image_path)

        image[image == 255] = 1
        one_indice = np.where(image == 1)
        center = int(np.mean(one_indice[0])), int(np.mean(one_indice[1]))

        pith_whole = tifffile.imread(image_path.replace('predictions_segmentation_from_bigDistance', 'predictions_pith'))
        # pith_whole = tifffile.imread('/home/khietdang/Documents/khiet/treeRing/bad_results/29(8)_x50_8 µm_pith.tif')

        pith_whole[pith_whole >= 0.5] = 1
        pith_whole[pith_whole < 0.5] = 0
        
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
                chose_point = choose_point(i, dis, dis_center, endpoints, remains, max_value, 
                                                   image1, 0.10 * image.shape[0], [center[1], center[0]])
                
                if chose_point is None:
                    dis[i, :] = copy.deepcopy(max_value)
                    dis[:, i] = copy.deepcopy(max_value)
                    remains.remove(i)
                    continue
                
                if dis[i, chose_point] / dis_center[i] < 1.2:
                    cv2.line(image1, endpoints[i, 1:3], endpoints[chose_point, 1:3], 1, 1)
                else:
                    # mid = (endpoints[i, 1] + endpoints[chose_point, 1]) / 2, (endpoints[i, 2] + endpoints[chose_point, 2]) / 2
                    # radius = (dis_center[i] + dis_center[chose_point]) / 2
                    # dis_mid_center = np.sqrt((mid[0] - center[1]) ** 2 + (mid[1] - center[0]) ** 2)
                    # ratio = radius / dis_mid_center
                    # third_point = (dis)
                    # center2, radius = findCircle(endpoints[i, 1], endpoints[i, 2], center[1], center[0], endpoints[chose_point, 1], endpoints[chose_point, 2])
                    # center3 = (2 * center2[0] - center[1], 2 * center2[1] - center[0])
                    # cor = curva_b(endpoints[i, 1], endpoints[i, 2], center3[0], center3[1], endpoints[chose_point, 1], endpoints[chose_point, 2])
                    # cor = np.array(cor)[:, None, :]
                    cor = plot_curve(endpoints[i, 1:3], endpoints[chose_point, 1:3], [center[1], center[0]], thresh=0.01*image.shape[0])
                    image1 = cv2.polylines(image1, cor, True, 1, 1)

                dis[i, :] = copy.deepcopy(max_value)
                dis[:, i] = copy.deepcopy(max_value)
                dis[chose_point, :] = copy.deepcopy(max_value)
                dis[:, chose_point] = copy.deepcopy(max_value)
                remains.remove(i)
                remains.remove(chose_point)
                print(i, chose_point)

            # image_final = np.zeros_like(image_combined)
            # contours, hierarchy = cv2.findContours(image1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            # for i in range(0, len(contours)):
            #     if hierarchy[0][i][3] == -1:
            #         hull = cv2.convexHull(contours[i])
            #         if cv2.pointPolygonTest(hull, [center[1], center[0]], True) > 0:
            #             cv2.drawContours(image_final, [hull], 0, 1, 1)

        image_final = np.zeros_like(image_combined)
        num_label, labels = cv2.connectedComponents(image1)
        for i in range(1, num_label):
            image2 = np.zeros_like(image_combined)
            image2[labels == i] = 1
            contours, _ = cv2.findContours(image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cv2.pointPolygonTest(contours[0], [center[1], center[0]], True) > 0:
                cv2.drawContours(image2, contours, 0, 1, cv2.FILLED)
                contours, _ = cv2.findContours(image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_final, contours, 0, 1, 1)

        contours, _ = cv2.findContours(pith_whole.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        longest_contour = np.argmax(np.array([len(contour) for contour in contours]))
        cv2.drawContours(image_final, contours, longest_contour, 1, 1)

        # plt.subplot(1, 3, 1)
        # plt.imshow(image, cmap='gray')
        # plt.subplot(1, 3, 2)
        # plt.imshow(labels, cmap='gray')
        for i in [32, 34]:
            plt.plot(endpoints[i, 1], endpoints[i, 2], 'bo')
        # plt.subplot(1, 3, 3)
        plt.imshow(image_final)
        plt.plot(int(center[1]), int(center[0]), 'bo')
        plt.show()

        image_final[image_final == 1] = 255
        # image1[image1 == 1] = 255
        tifffile.imwrite('/home/khietdang/Documents/khiet/treeRing/endpoints_bigDistance_pith/' + os.path.basename(image_path),
                         image_final.astype(np.uint8))
        
        input_image = tifffile.imread(image_path.replace('predictions_segmentation_from_bigDistance', 'input'))
        input_image[image_final == 255] = 0
        tifffile.imwrite('/home/khietdang/Documents/khiet/treeRing/final_bigDistance_pith/' + os.path.basename(image_path),
                         input_image.astype(np.uint8))