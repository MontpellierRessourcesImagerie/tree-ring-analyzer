from tree_ring_analyzer.dl.train import Training, bce_dice_loss
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
   filter_num = [16, 24, 40, 80, 960] # [7, 14, 28, 56, 112], [16, 24, 40, 80, 960], [64, 128, 256, 512, 1024]
   output_activation = 'linear' # linear, sigmoid
   attention=True
   loss = 'mse' # bce_dice_loss(bce_coef=0.5)
   name = 'bigDisRingAugGrayWH16RD'
   numEpochs = 30 #30, 100
   input_size = (256, 256, 1)

   unet_model = Unet(input_size=input_size,
                     filter_num=filter_num,
                     n_labels=1,
                     output_activation=output_activation,
                     attention=attention,
                     ).model

   train = Training(train_input_path, 
                  train_mask_path,
                  val_input_path,
                  val_mask_path,
                  name=name,
                  loss = loss,
                  numEpochs=30,
                  channel = input_size[-1]
                  )
   
   train.fit(unet_model)

    