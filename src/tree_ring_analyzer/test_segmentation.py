from tree_ring_analyzer.segmentation import TreeRingSegmentation
import tifffile
import tensorflow as tf
import matplotlib.pyplot as plt



if __name__ == '__main__':
    image_path = '/home/khietdang/Documents/khiet/treeRing/input/1E_4milieu8microns_x40.tif'
    checkpoint_ring_path = '/home/khietdang/Documents/khiet/tree-ring-analyzer/models/bigDistance.keras'
    checkpoint_pith_path = '/home/khietdang/Documents/khiet/tree-ring-analyzer/models/pith.keras'

    image = tifffile.imread(image_path)

    model_ring = tf.keras.models.load_model(checkpoint_ring_path)

    model_pith = tf.keras.models.load_model(checkpoint_pith_path)

    treeRingSegment = TreeRingSegmentation(model_ring, model_pith)
    treeRingSegment.segmentImage(image)
    
    result = treeRingSegment.maskRings

    plt.imshow(result)
    plt.show()



