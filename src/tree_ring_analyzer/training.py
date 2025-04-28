from tree_ring_analyzer.dl.train import Training, bce_dice_loss
from tree_ring_analyzer.dl.model import Unet
import tensorflow as tf
from tensorflow import keras



if __name__ == '__main__':
    train_input_path = "/home/khietdang/Documents/khiet/treeRing/tile_big_dis_otherrings/train/x"
    train_mask_path = "/home/khietdang/Documents/khiet/treeRing/tile_big_dis_otherrings/train/y"
    val_input_path = "/home/khietdang/Documents/khiet/treeRing/tile_big_dis_otherrings/val/x"
    val_mask_path = "/home/khietdang/Documents/khiet/treeRing/tile_big_dis_otherrings/val/y"

    unet_model = Unet(input_size=(256, 256, 1),
                               filter_num=[16, 24, 40, 80, 960],
                               n_labels=1,
                               output_activation='linear',
                            #    output_activation='sigmoid',
                               attention=False,
                               ).model
    
    # unet_model = tf.keras.models.load_model('/home/khietdang/Documents/khiet/tree-ring-analyzer/models/bigDisRingAugGrayEH.keras')

    train = Training(train_input_path, 
                     train_mask_path,
                     val_input_path,
                     val_mask_path,
                     name='bigDisRingAugGrayNormal',
                     loss='mse',
                    #  loss=bce_dice_loss(bce_coef=0.5),
                    numEpochs=100,
                     )
    
    train.fit(unet_model)

    