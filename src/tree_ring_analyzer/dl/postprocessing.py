import copy
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, median_filter
from skimage.draw import circle_perimeter
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import active_contour
from skimage.transform import hough_circle, hough_circle_peaks
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import tifffile



def activeContour(image_path, pith_path, output_path, thickness):
    image = tifffile.imread(image_path)
    image = (image - np.min(image)) * 255 / (np.max(image) - np.min(image))
    fsize = int((image.shape[0] + image.shape[1]) * 0.001) * 2 + 1
    image = median_filter(image, size=fsize)

    shapeOriginal = image.shape
    image = cv2.resize(image, (int(image.shape[1] / 5), int(image.shape[0] / 5)))
    
    pith = tifffile.imread(os.path.join(pith_path, os.path.basename(image_path)))
    pith_whole = cv2.resize(pith, (image.shape[1], image.shape[0]))
    pith_whole[pith_whole >= 0.5] = 1
    pith_whole[pith_whole < 0.5] = 0
    pith_clear = binary_dilation(pith_whole)
    image = (image * (1 - pith_clear)).astype(np.uint8)

    one_indice = np.where(pith_whole == 1)
    center_pith = np.mean(one_indice[0]), np.mean(one_indice[1])
    center = int(image.shape[0] / 2), int(image.shape[1] / 2)
    contours, _ = cv2.findContours(pith_whole.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    longest_contour = np.argmax(np.array([len(contour) for contour in contours]))
    pithContour = contours[longest_contour]
    rad_min = np.mean(np.sqrt((pithContour[:, 0, 1] - center[0]) ** 2 + (pithContour[:, 0, 0] - center[1]) ** 2))

    rings = []
    radius = int(0.95 * min(center[0], image.shape[0] - center[0], center[1], image.shape[1] - center[1]))
    i = 1
    plt.imshow(image)
    while True:
        s = np.linspace(0, 2 * np.pi, int(np.pi * radius * 2))
        r = center[0] + radius * np.sin(s)
        c = center[1] + radius * np.cos(s)
        init = np.array([r, c]).T
        snake = active_contour(
            image,
            init,
            alpha=radius / 100,
            beta=0.5,
            gamma=0.001,
            max_num_iter=1000,
            boundary_condition='periodic'
        )

        center = int(np.mean(snake[:, 0])), int(np.mean(snake[:, 1]))
        radius = np.min(np.sqrt((center[0] - snake[:, 0]) ** 2 + (center[1] - snake[:, 1]) ** 2))
        polygon_value = cv2.pointPolygonTest(snake[:, None, ::-1].astype(np.int32), (center_pith[1], center_pith[0]), True)
        if radius > rad_min and polygon_value > 0:
            rings.append((snake[:, ::-1] * 5).astype(np.int32))
            radius = radius - int(0.025 * image.shape[0])
            plt.plot(init[:, 1], init[:, 0], 'r--')
            plt.plot(snake[:, 1], snake[:, 0], 'b')
        
        if radius <= rad_min or i >= 5 or polygon_value < 0:
            break

        i += 1
    plt.show()
    raise ValueError
    image_final = np.zeros((shapeOriginal[0], shapeOriginal[1]))

    pith[pith >= 0.5] = 1
    pith[pith < 0.5] = 0
    contours, _ = cv2.findContours(pith.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    longest_contour = np.argmax(np.array([len(contour) for contour in contours]))
    rings.append(contours[longest_contour][:, 0, :])

    for ring in rings:
        cv2.drawContours(image_final, [ring[:, None, :]], 0, 1, thickness=thickness)

    image_final[image_final == 1] = 255
    tifffile.imwrite(os.path.join(output_path, os.path.basename(image_path)), image_final.astype(np.uint8))
    return image_final.astype(np.uint8), rings



def clustering(imat_path):
    image = tifffile.imread()
    image = binary_dilation(image, iterations=30).astype(np.uint8)
    height, width = image.shape[0], image.shape[1]
    image = cv2.resize(image, (int(height / 10), int(height / 10)))

    one_indice = np.where(image == 1)
    center = np.mean(one_indice[0]), np.mean(one_indice[1])
    dis = (center[0] - one_indice[0]) ** 2 + (center[1] - one_indice[1]) ** 2

    X = dis[:, None]
    X = StandardScaler().fit_transform(X)
    
    model = DBSCAN(eps=0.2, min_samples=2).fit(X)

    label = model.labels_
    
    re = np.zeros_like(image)
    for i in range(0, 5):
        re[one_indice[0][label == i], one_indice[1][label == i]] = i + 1
        _center = np.mean(one_indice[0][label == i]), np.mean(one_indice[1][label == i])


    plt.imshow(re)
    plt.show()



def hough(image_path, output_path):
    image = tifffile.imread(image_path)
    image = binary_dilation(image, iterations=30).astype(np.uint8)
    height, width = image.shape[0], image.shape[1]
    image = cv2.resize(image, (int(width / 10), int(height / 10)))

    hough_radii = np.arange(int(width / 300), int(width / 30), 5)
    hough_res = hough_circle(image, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=100, num_peaks=100)
    hough_image_circle = np.zeros_like(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
        hough_image_circle[circy, circx] = 255

    hough_image_circle = cv2.resize(hough_image_circle, (width, height))

    tifffile.imwrite(os.path.join(output_path, os.path.basename(image_path)), hough_image_circle)



def interpolate(image_path):
    pred_seg = tifffile.imread(image_path)

    image = binary_dilation(pred_seg, iterations=30).astype(np.uint8)
    height, width = image.shape[0], image.shape[1]
    image = cv2.resize(image, (int(height / 10), int(width / 10)))

    one_indice = np.where(image == 1)
    one_choose = np.random.randint(0, len(one_indice[0]), int(len(one_indice[0])))
    one_indice = (one_indice[0][one_choose], one_indice[1][one_choose])

    zero_indice = np.where(image == 0)
    zero_choose = np.random.randint(0, len(zero_indice[0]), len(one_indice[0]))
    zero_indice = (zero_indice[0][zero_choose], zero_indice[1][zero_choose])

    px = np.concatenate((one_indice[0], zero_indice[0]))
    py = np.concatenate((one_indice[1], zero_indice[1]))

    x = np.arange(0, image.shape[0])
    y =  np.arange(0, image.shape[1])
    X, Y = np.meshgrid(x,y)

    T = griddata((px, py), image[px, py], (X, Y), method='linear').T

    plt.subplot(1, 2, 1)
    plt.imshow(image, 'gray')
    plt.subplot(1, 2, 2)
    plt.imshow(T, 'gray')
    plt.show()



def morphologyOval(image_path, output_path):
    image = tifffile.imread(image_path)
    image = binary_dilation(image, iterations=30).astype(np.uint8)
    height, width = image.shape[0], image.shape[1]
    image = cv2.resize(image, (512, 512))

    one_indice = np.where(image == 1)
    center = int(np.average(one_indice[1])), int(np.average(one_indice[0]))
    
    solid_circle = np.zeros_like(image)
    for ksize in [8, 16, 32, 64, 128]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize))  # Adjust size as needed

        mask1 = np.zeros_like(image)
        cv2.circle(mask1, center, ksize * 2, 1, -1)
        mask2 = np.zeros_like(image)
        cv2.circle(mask2, center, ksize * 4, 1, -1)
        mask = mask1 + (1 - mask2)

        solid_circle += cv2.morphologyEx(image * mask, cv2.MORPH_CLOSE, kernel)

    solid_circle = cv2.resize(solid_circle, (height, width))
    
    plt.imshow(solid_circle)
    plt.show()
    plt.close()
    
    tifffile.imwrite(os.path.join(output_path, os.path.basename(image_path)), solid_circle)



def measureDistance(points):
    x1 = points[:, 0][:, None]
    x2 = points[:, 0][None, :]
    y1 = points[:, 1][:, None]
    y2 = points[:, 1][None, :]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)



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



def choose_point(i, dis, dis_center, endpoints, remains, max_value, image, thresh, center):
    if np.all(dis[i, :] == max_value):
        return None
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



def recta(x1, y1, x2, y2):
    a = (y1 - y2) / (x1 - x2)
    b = y1 - a * x1
    return (a, b)



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



def endpoints(image_path, pith_folder, output_folder, thickness=1):
    image = tifffile.imread(image_path)

    image[image == 255] = 1
    one_indice = np.where(image == 1)
    center = int(np.mean(one_indice[0])), int(np.mean(one_indice[1]))

    pith_whole = tifffile.imread(os.path.join(pith_folder, os.path.basename(image_path)))

    pith_whole[pith_whole >= 0.5] = 1
    pith_whole[pith_whole < 0.5] = 0
    
    pith_clear = binary_dilation(pith_whole)

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

        dis = measureDistance(endpoints[:, 1:3])
        max_value = np.max(dis) + 1
        dis[dis == 0] = max_value
        
        remains = list(range(0, len(dis)))
        while len(remains) >= 2:
            i = remains[0]
            chose_point = choose_point(i, dis, dis_center, endpoints, remains, max_value, 
                                                image1, 0.10 * image.shape[0], [center[1], center[0]])
            
            if chose_point is None:
                dis[i, :] = copy.deepcopy(max_value)
                dis[:, i] = copy.deepcopy(max_value)
                remains.remove(i)
                continue
            
            cv2.line(image1, endpoints[i, 1:3], endpoints[chose_point, 1:3], 1, 1)

            dis[i, :] = copy.deepcopy(max_value)
            dis[:, i] = copy.deepcopy(max_value)
            dis[chose_point, :] = copy.deepcopy(max_value)
            dis[:, chose_point] = copy.deepcopy(max_value)
            remains.remove(i)
            remains.remove(chose_point)


    image_final = np.zeros_like(image_combined)
    num_label, labels = cv2.connectedComponents(image1)
    for i in range(1, num_label):
        image2 = np.zeros_like(image_combined)
        image2[labels == i] = 1
        contours, _ = cv2.findContours(image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cv2.pointPolygonTest(contours[0], [center[1], center[0]], True) > 0:
            cv2.drawContours(image2, contours, 0, 1, cv2.FILLED)
            contours, _ = cv2.findContours(image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_final, contours, 0, 1, thickness=thickness)

    contours, _ = cv2.findContours(pith_whole.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    longest_contour = np.argmax(np.array([len(contour) for contour in contours]))
    cv2.drawContours(image_final, contours, longest_contour, 1, thickness=thickness)

    image_final[image_final == 1] = 255
    tifffile.imwrite(os.path.join(output_folder, os.path.basename(image_path)), image_final.astype(np.uint8))
    
    return image_final