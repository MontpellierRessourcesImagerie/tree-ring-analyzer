from tree_ring_analyzer.dl.train import Training, bce_dice_loss
from tree_ring_analyzer.dl.model import AttentionUnet
import tensorflow as tf
from tensorflow import keras



if __name__ == '__main__':
    train_input_path = "/home/khietdang/Documents/khiet/treeRing/tile_big_dis_otherrings/train/x"
    train_mask_path = "/home/khietdang/Documents/khiet/treeRing/tile_big_dis_otherrings/train/y"
    val_input_path = "/home/khietdang/Documents/khiet/treeRing/tile_big_dis_otherrings/val/x"
    val_mask_path = "/home/khietdang/Documents/khiet/treeRing/tile_big_dis_otherrings/val/y"

    unet_model = AttentionUnet(input_size=(256, 256, 3),
                               filter_num=[7, 14, 28, 56, 112],
                               n_labels=1,
                               output_activation='linear',
                            #    activation='sigmoid',
                               ).model

    train = Training(train_input_path, 
                     train_mask_path,
                     val_input_path,
                     val_mask_path,
                     name='bigDisRingAug2',
                     loss='mse',
                    #  loss=bce_dice_loss(bce_coef=0.3),
                    numEpochs=10,
                     )
    
    train.fit(unet_model)

    