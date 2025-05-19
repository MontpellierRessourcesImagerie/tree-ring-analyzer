import tensorflow as tf
from tensorflow.keras import layers
import tifffile
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray
import glob
import copy
from skimage.filters import threshold_otsu
from tree_ring_analyzer.dl.train import bce_dice_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



if __name__ == "__main__":
   input_folder = '/home/khietdang/Documents/khiet/treeRing/Luidmila/50 tilias'
   prediction_folder = '/home/khietdang/Documents/khiet/treeRing/Luidmila/predictions_50 tilias_bigDisRingAugGrayNormal16'
   checkpoint = '/home/khietdang/Documents/khiet/tree-ring-analyzer/models/pithGrayNormal16.h5'
   output_folder = f'/home/khietdang/Documents/khiet/treeRing/Luidmila/predictions_50 tilias_{os.path.basename(checkpoint)[:-3]}'
   os.makedirs(output_folder, exist_ok=True)
   list_input = glob.glob(os.path.join(input_folder, '*.tif'))
   batch_size = 8
   model = tf.keras.models.load_model(checkpoint, custom_objects={'bcl': bce_dice_loss(bce_coef=0.5)})

   for im_name in list_input:
      im_data = tifffile.imread(im_name)
      im_data = (0.299 * im_data[:, :, 0] + 0.587 * im_data[:, :, 1] + 0.114 * im_data[:, :, 2])[:, :, None]
      prediction_ring = tifffile.imread(os.path.join(prediction_folder, os.path.basename(im_name)))
      shape = im_data.shape[0], im_data.shape[1]

      ret = threshold_otsu(prediction_ring)
      ring_indices = np.where(prediction_ring > ret)
      center = int(np.mean(ring_indices[0])), int(np.mean(ring_indices[1]))
      
      crop_size = int(0.1 * max(shape[1], shape[0])) * 2
      crop_img = im_data[center[0] - int(crop_size / 2):center[0] + int(crop_size / 2),
                        center[1] - int(crop_size / 2):center[1] + int(crop_size / 2)]

      crop_img = cv2.resize(crop_img, (256, 256))[None, :, :, None]
      crop_img = crop_img / 255
      pred_pith = model.predict(crop_img)

      pred_final = np.zeros((im_data.shape[0], im_data.shape[1]))
      pred_pith = cv2.resize(pred_pith[0], (crop_size, crop_size))
      pred_final[center[0] - int(crop_size / 2):center[0] + int(crop_size / 2),
                        center[1] - int(crop_size / 2):center[1] + int(crop_size / 2)] = copy.deepcopy(pred_pith)

      tifffile.imwrite(os.path.join(output_folder, os.path.basename(im_name)), pred_final)

