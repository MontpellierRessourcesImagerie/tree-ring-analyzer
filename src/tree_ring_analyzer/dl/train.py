import tensorflow as tf
import numpy as np
import tifffile
import os



def read_images(img_path, label_path):
    def _load_image(path):
        path = path.numpy().decode("utf-8")  # Convert TensorFlow tensor to Python string
        img = tifffile.imread(path)  # Read TIFF image
        img = img.astype(np.float32)  # Convert image to float32 for TensorFlow compatibility
        return img

    img = tf.py_function(func=_load_image, inp=[img_path], Tout=tf.float32)  # Use tf.py_function
    img.set_shape([None, None, None])  # Set output shape for TensorFlow dataset compatibility

    seg = tf.py_function(func=_load_image, inp=[label_path], Tout=tf.float32)  # Use tf.py_function
    seg.set_shape([None, None, None])  # Set output shape for TensorFlow dataset compatibility

    return img / 255, seg



class Training:


    def __init__(self, input_path, label_path):
        self.inputPath = input_path
        self.labelPath = label_path
        self.batchSize = 8
        self.bufferSize = 512
        self.trainDataset = None
        self._createDataset()


    def _createDataset(self):
        train_input_paths = [os.path.join(self.inputPath, path) for path in os.listdir(self.inputPath) if path.endswith(".tif")]
        train_mask_paths = [os.path.join(self.labelPath, path) for path in os.listdir(self.labelPath) if path.endswith(".tif")]
        train_path_dataset = tf.data.Dataset.from_tensor_slices((train_input_paths, train_mask_paths))
        self.trainDataset = train_path_dataset.map(lambda img_path, label: (read_images(img_path, label)), num_parallel_calls=tf.data.AUTOTUNE)