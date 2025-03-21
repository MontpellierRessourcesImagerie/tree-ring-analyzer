import tifffile
import cv2
from scipy.ndimage import binary_dilation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import math

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

def identify_ellipse(center, p1, p2):
    h, k = center
    x1, y1 = p1
    x2, y2 = p2
    
    a = max(abs(x1 - h), abs(x2 - h))  # Approximate semi-major axis
    b = max(abs(y1 - k), abs(y2 - k))  # Approximate semi-minor axis
    
    return (h, k, a, b)

def plot_ellipse(h, k, a, b):
    theta = np.linspace(0, 2 * np.pi, 300)
    x = h + a * np.cos(theta)
    y = k + b * np.sin(theta)
    return np.append(x[:, None, None], y[:, None, None], axis=-1)

def plot_curve(point1, point2, center, num=100000):
    x1, y1 = point1
    x2, y2 = point2
    cx, cy = center

    grad1 = cx - x1, cy - y1
    grad2 = cx - x2, cy - y2

    nx1, ny1 = - grad1[1], grad1[0]
    nx2, ny2 = - grad2[1], grad2[0]

    tanx1 = nx1 / ny1 / 2
    tany1 = ny1 / nx1 / 2

    tanx2 = nx2 / ny2 / 2
    tany2 = ny2 / nx2 / 2
    
    cor = np.zeros((num, 1, 2))
    cor[0, 0, 0] = x1
    cor[0, 0, 1] = y1
    cor[-1, 0, 0] = x2
    cor[-1, 0, 1] = y2
    for i in range(1, num - 1):
        _x = (x2 - cor[i - 1, 0, 0]) / (num - i)
        _y = (y2 - cor[i - 1, 0, 1]) / (num - i)

        if i <= num / 2:
            _x += _y * tanx1 * (num / 2 - i) /(num / 2)
            _y += _x * tany1 * (num / 2 - i) /(num / 2)
        else:
            _x += _y * tanx2 * (i - num / 2) /(num / 2)
            _y += _x * tany2 * (i - num / 2) /(num / 2)

        cor[i, 0, 0] = cor[i - 1, 0, 0] + _x
        cor[i, 0, 1] = cor[i - 1, 0, 1] + _y

    return cor


if __name__ == '__main__':
    image = np.zeros((7000, 7000))
    point1 = (5000, 2000)
    point2 = (6000, 6000)
    center = (3500, 3500)
    
    cor = plot_curve(point1, point2, center)
    image = cv2.polylines(image, cor[:50000].astype(int), True, 1, 10)
    image = cv2.polylines(image, cor[50000:].astype(int), True, 2, 10)

    plt.imshow(image)
    plt.plot(point1[0], point1[1], 'bo')
    plt.plot(point2[0], point2[1], 'ro')
    plt.plot(center[0], center[1], 'go')
    plt.show()