from tree_ring_analyzer.dl.train import Training, bce_dice_loss, dice_mse_loss, dice_loss
from tree_ring_analyzer.dl.model import Unet
import random
import numpy as np
import tensorflow as tf

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


if __name__ == '__main__':
   train_input_path = "/home/khietdang/Documents/khiet/treeRing/tile_big_dis_otherrings/train/x"
   train_mask_path = "/home/khietdang/Documents/khiet/treeRing/tile_big_dis_otherrings/train/y"
   val_input_path = "/home/khietdang/Documents/khiet/treeRing/tile_big_dis_otherrings/val/x"
   val_mask_path = "/home/khietdang/Documents/khiet/treeRing/tile_big_dis_otherrings/val/y"

   input_size = (256, 256, 1)

   unet_model = Unet(input_size=input_size,
                     filter_num=[32, 64, 128, 256, 512],
                     n_labels=1,
                     output_activation='linear',
                  #  output_activation='sigmoid',
                     attention=False,
                     ).model

   train = Training(train_input_path, 
                  train_mask_path,
                  val_input_path,
                  val_mask_path,
                  name='bigDisRingAugGrayNormalWH32',
                  loss='mse',
                  # loss=bce_dice_loss(bce_coef=0.5),
                  numEpochs=30,
                  channel = input_size[-1]
                  )
   
   train.fit(unet_model)

    