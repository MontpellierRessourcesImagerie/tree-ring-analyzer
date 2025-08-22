import tensorflow as tf
import numpy as np
import tifffile
import os
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import json


def read_images(img_path, label_path, channel):


    def _load_image(path):
        path = path.numpy().decode("utf-8")  # Convert TensorFlow tensor to Python string
        img = tifffile.imread(path)  # Read TIFF image
        img = img.astype(np.float32)  # Convert image to float32 for TensorFlow compatibility
        if len(img.shape) == 2:
            img = img[:, :, None]
        if channel == 1 and img.shape[-1] == 3:
            img = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])[:, :, None]
        return img


    img = tf.py_function(func=_load_image, inp=[img_path], Tout=tf.float32)  # Use tf.py_function
    img.set_shape([None, None, None])  # Set output shape for TensorFlow dataset compatibility

    seg = tf.py_function(func=_load_image, inp=[label_path], Tout=tf.float32)  # Use tf.py_function
    seg.set_shape([None, None, 1])  # Set output shape for TensorFlow dataset compatibility
    if tf.reduce_max(seg) == 255:
        seg = seg / 255

    return img, seg


@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + 1) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1)


@tf.keras.utils.register_keras_serializable()
def dice_dm_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    intersection = tf.reduce_sum(tf.sqrt(tf.maximum(y_true * y_pred, epsilon)))
    return 1 - (2. * intersection + 1) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1)


@tf.keras.utils.register_keras_serializable()
def bce_dice_loss(bce_coef=0.5):
    @tf.keras.utils.register_keras_serializable()
    def bcl(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        dice = dice_loss(y_true, y_pred)
        return bce_coef * bce + (1.0 - bce_coef) * dice
    return bcl


@tf.keras.utils.register_keras_serializable()
def dice_mse_loss(mse_coef=0.3):
    def dml(y_true, y_pred):
        dice = dice_dm_loss(y_true, y_pred)
        mse = keras.losses.MeanSquaredError()(y_true, y_pred)
        return dice + mse * mse_coef
    return dml


@tf.keras.utils.register_keras_serializable()
def rdloss(y_true, y_pred):
    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=(1, 2, 3))
    y_pred = tf.math.sigmoid(y_pred - tf.math.reduce_mean(y_pred))
    y_true = y_true / tf.math.reduce_max(y_true)
    rd = - 0.5 * (2 * tf.math.reduce_sum(y_pred *  y_true)) / (tf.math.reduce_sum(y_pred ** 2) + tf.math.reduce_sum(y_true ** 2))
    return mse + rd
    


class Training:


    def __init__(self, input_path, label_path, val_input_path, val_label_path,
                 name, loss='mse', metrics=['accuracy', 'precision', 'recall', 'mse'], numEpochs=100, channel=1):
        self.inputPath = input_path
        self.labelPath = label_path
        self.valInputPath = val_input_path
        self.valLabelPath = val_label_path
        self.name = name
        self.loss = loss
        self.metrics = metrics
        self.numEpochs = numEpochs
        self.channel = channel

        self.trainDataset = None
        self.valDataset = None
        self.model = None
        self.stepsPerEpoch = None
        self.checkpoint_callback1 = ModelCheckpoint(f'./models/{name}.h5',  # Filename
                                                   monitor='val_loss',  # Metric to monitor
                                                   save_best_only=True,  # Save only if it improves
                                                   mode='min',  # Minimize the loss
                                                   verbose=1)
        self.checkpoint_callback2 = ModelCheckpoint(f'./models/{name}.keras',  # Filename
                                                   monitor='val_loss',  # Metric to monitor
                                                   save_best_only=True,  # Save only if it improves,
                                                   mode='min',  # Minimize the loss
                                                   verbose=1)
        self.checkpoint_callback3 = EarlyStopping(monitor='val_loss',
                                                    patience=10,
                                                    verbose=0,
                                                    mode='min')

        
        self.batchSize = 8
        self.bufferSize = 256
        self.learningRate = 0.001
        
        self._createDataset()


    def _createDataset(self):
        train_input_paths = [os.path.join(self.inputPath, path) for path in os.listdir(self.inputPath) if path.endswith(".tif")]
        train_mask_paths = [os.path.join(self.labelPath, path) for path in os.listdir(self.labelPath) if path.endswith(".tif")]
        self.stepsPerEpoch = len(train_input_paths) // self.batchSize

        train_path_dataset = tf.data.Dataset.from_tensor_slices((train_input_paths, train_mask_paths))
        trainDataset = train_path_dataset.map(lambda img_path, label: (read_images(img_path, label, self.channel)), num_parallel_calls=tf.data.AUTOTUNE)
        trainDataset = trainDataset.shuffle(self.bufferSize).batch(self.batchSize)
        self.trainDataset = trainDataset.prefetch(tf.data.AUTOTUNE)

        val_input_paths = [os.path.join(self.valInputPath, path) for path in os.listdir(self.valInputPath) if path.endswith(".tif")]
        val_mask_paths = [os.path.join(self.valLabelPath, path) for path in os.listdir(self.valLabelPath) if path.endswith(".tif")]
        val_path_dataset = tf.data.Dataset.from_tensor_slices((val_input_paths, val_mask_paths))
        valDataset = val_path_dataset.map(lambda img_path, label: (read_images(img_path, label, self.channel)), num_parallel_calls=tf.data.AUTOTUNE)
        valDataset = valDataset.shuffle(self.bufferSize).batch(self.batchSize)
        self.valDataset = valDataset.prefetch(tf.data.AUTOTUNE)

    
    def _compileModel(self, model):
        self.model = model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learningRate),
                           loss=self.loss,
                           metrics=self.metrics)
        
    
    def fit(self, model):
        self._compileModel(model)
        model_history = self.model.fit(self.trainDataset,
                                epochs=self.numEpochs,
                                steps_per_epoch=self.stepsPerEpoch,
                                validation_data=self.valDataset, 
                                verbose=2,
                                callbacks=[self.checkpoint_callback1, self.checkpoint_callback2, self.checkpoint_callback3]
                                )
        
        json.dump(model_history.history, open(f'./history/{self.name}.json', 'w'))

        