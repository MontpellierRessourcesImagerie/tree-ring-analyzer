from tree_ring_analyzer.dl.train import Training, bce_dice_loss
from tree_ring_analyzer.dl.model import AttentionUnet
import tensorflow as tf
from tensorflow import keras



if __name__ == '__main__':
    train_input_path = "/home/khietdang/Documents/khiet/treeRing/pith/train/x"
    train_mask_path = "/home/khietdang/Documents/khiet/treeRing/pith/train/y"
    val_input_path = "/home/khietdang/Documents/khiet/treeRing/pith/val/x"
    val_mask_path = "/home/khietdang/Documents/khiet/treeRing/pith/val/y"

    unet_model = AttentionUnet(input_size=(256, 256, 1),
                               filter_num=[7, 14, 28, 56, 112],
                               n_labels=1,
                              #  output_activation='linear',
                               output_activation='sigmoid',
                               ).model

    train = Training(train_input_path, 
                     train_mask_path,
                     val_input_path,
                     val_mask_path,
                     name='pithGray',
                     # loss='mse',
                     loss=bce_dice_loss(bce_coef=0.3),
                    numEpochs=100,
                     )
    
    train.fit(unet_model)

    