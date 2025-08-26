from tree_ring_analyzer.tiles.tiler import ImageTiler2D
from tree_ring_analyzer._tests import utils
from tree_ring_analyzer.dl.model import Unet
from tree_ring_analyzer import INPUT_SIZE, FILTER_NUM, N_LABELS

import random
import numpy as np


def test_demo_unet():
    # Generate a random 2D shapes in ([10, 99999], [10, 99999]).
    shape = (random.randint(256, 9999), random.randint(256, 9999))

    # Input image.
    input_img = utils.make_checkboard(shape, 256, 'grayscale')

    # Tiling
    tiles_manager = ImageTiler2D(INPUT_SIZE[0], 10, input_img.shape)
    tiles = np.array(tiles_manager.image_to_tiles(input_img, False))

    # Reconstruction
    reconstruct_img = tiles_manager.tiles_to_image(tiles)

    assert reconstruct_img.shape == input_img.shape

    difference = np.abs(input_img.astype(np.float32) - reconstruct_img.astype(np.float32)).astype(np.uint16)
    assert np.percentile(difference, 75) == 0

    # Model
    model = Unet(input_size=INPUT_SIZE, filter_num=FILTER_NUM, n_labels=N_LABELS, output_activation='sigmoid').model
    
    # Inference result
    predictions = model.predict(tiles[:, :, :, None])
    probability_map = tiles_manager.tiles_to_image(predictions[:, :, :, 0])

    assert probability_map.shape == input_img.shape

