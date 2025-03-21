import tensorflow as tf
from tensorflow.keras import layers
import tifffile
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.3)(p)
   return f, p

def attention_gate(g, s, num_filters):
    Wg = layers.Conv2D(num_filters, 3, padding="same")(g)
    Wg = layers.BatchNormalization()(Wg)
 
    Ws = layers.Conv2D(num_filters, 3, padding="same")(s)
    Ws = layers.BatchNormalization()(Ws)
 
    out = layers.Activation("relu")(Wg + Ws)
    out = layers.Conv2D(num_filters, 3, padding="same")(out)
    out = layers.Activation("sigmoid")(out)
 
    return out * s
    
def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   s = attention_gate(x, conv_features, n_filters)
   # concatenate
   x = layers.concatenate([x, s])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)
   return x

def build_unet_model():
   inputs = layers.Input(shape=(256,256,3))
   # encoder: contracting path - downsample
   # 1 - downsample
   f1, p1 = downsample_block(inputs, 64)
   # 2 - downsample
   f2, p2 = downsample_block(p1, 128)

   # 3 - downsample
   f3, p3 = downsample_block(p2, 256)
   # 4 - downsample
   f4, p4 = downsample_block(p3, 512)
   # 5 - bottleneck
   bottleneck = double_conv_block(p4, 1024)
   # decoder: expanding path - upsample
   # 6 - upsample
   u6 = upsample_block(bottleneck, f4, 512)
   # 7 - upsample
   u7 = upsample_block(u6, f3, 256)
   # 8 - upsample
   u8 = upsample_block(u7, f2, 128)
   # 9 - upsample
   u9 = upsample_block(u8, f1, 64)
   # outputs
   outputs = layers.Conv2D(1, (1,1), padding="same", activation = "sigmoid")(u9)
   # unet model with Keras Functional API
   unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
   return unet_model

if __name__ == "__main__":
    input_folder = '/home/khietdang/Documents/khiet/treeRing/input'
    checkpoint = '/home/khietdang/Documents/khiet/tree-ring-analyzer/src/models/pith.h5'
    output_folder = f'/home/khietdang/Documents/khiet/treeRing/predictions_{os.path.basename(checkpoint)[:-3]}'
    os.makedirs(output_folder, exist_ok=True)
    list_input = glob.glob(os.path.join(input_folder, '*.tif'))
    crop_size = 1024
    batch_size = 8
    model = build_unet_model()
    model.load_weights(checkpoint)

    for im_name in list_input:
        im_data = tifffile.imread(im_name)
        pred_otherrings = tifffile.imread(im_name.replace('input', 'predictions_segmentation_from_bigDistance'))

        one_indices = np.where(pred_otherrings == 255)
        center = int(np.mean(one_indices[0])), int(np.mean(one_indices[1]))
        
        crop_img = im_data[center[0] - int(crop_size / 2):center[0] + int(crop_size / 2),
                           center[1] - int(crop_size / 2):center[1] + int(crop_size / 2)]

        crop_img = cv2.resize(crop_img, (256, 256))[None, :, :, :]
        crop_img = crop_img / 255
        pred_pith = model.predict(crop_img)

        tifffile.imwrite(os.path.join(output_folder, os.path.basename(im_name)[:-4] + f'_{center[0]}_{center[1]}.tif'), pred_pith[0])

