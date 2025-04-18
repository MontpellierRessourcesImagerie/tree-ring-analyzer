from tree_ring_analyzer.dl.train3 import Training, bce_dice_loss
from tree_ring_analyzer.dl.model import AttentionUnet
import tensorflow as tf
from tensorflow import keras
import glob
import os
from sklearn.model_selection import train_test_split



if __name__ == '__main__':
    input_path = '/home/khietdang/Documents/khiet/treeRing/predictions_bigDisRingAugGray'
    mask_path = '/home/khietdang/Documents/khiet/treeRing/big_dis_otherrings'

    list_images = glob.glob(os.path.join(input_path, '*.tif'))
    list_masks = [image_path.replace(input_path, mask_path) for image_path in list_images]

    train_images, test_images, train_masks, test_masks = train_test_split(list_images, list_masks, test_size=0.2, shuffle=True, random_state=42)
    train_images, val_images, train_masks, val_masks = train_test_split(train_images, train_masks, test_size=0.2, shuffle=True, random_state=42)

    unet_model = AttentionUnet(input_size=(256, 256, 1),
                               filter_num=[7, 14, 28, 56, 112],
                               n_labels=1,
                               output_activation='linear',
                            #    output_activation='sigmoid',
                               ).model

    train = Training(train_images, train_masks,
                     val_images, val_masks,
                     name='thirdModel',
                     loss='mse',
                    #  loss=bce_dice_loss(bce_coef=0.3),
                    numEpochs=30000,
                     )
    
    train.fit(unet_model)

    