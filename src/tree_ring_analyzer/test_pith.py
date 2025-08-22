import tensorflow as tf
from tensorflow.keras import layers
import tifffile
import os
import numpy as np
import cv2
from skimage.color import rgb2gray
import glob
import copy
from skimage.filters import threshold_otsu
from tree_ring_analyzer.dl.train import bce_dice_loss
from tree_ring_analyzer.segmentation import TreeRingSegmentation

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
      
      tree_ring_segmentation = TreeRingSegmentation(resize=5, pithWhole=False)
      tree_ring_segmentation.shape = im_data.shape[0], im_data.shape[1]
      tree_ring_segmentation.predictionRing = prediction_ring

      tree_ring_segmentation.createMask(im_data)

      tree_ring_segmentation.predictPith(model, im_data)
      pred_final = tree_ring_segmentation.pith

      tifffile.imwrite(os.path.join(output_folder, os.path.basename(im_name)), pred_final)

