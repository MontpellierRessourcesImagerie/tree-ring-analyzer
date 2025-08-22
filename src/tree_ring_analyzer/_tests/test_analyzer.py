from tree_ring_analyzer.segmentation import TreeRingSegmentation, Evaluation
from tree_ring_analyzer._tests import utils
from tree_ring_analyzer.dl.model import Unet
from tree_ring_analyzer import FILTER_NUM, N_LABELS, INPUT_SIZE

import numpy as np

def test_analyzer():
    # Input image.
    input_img = utils.make_checkboard((1000, 1000), 256, 'grayscale')[:, :, None]
    input_img = input_img.astype(np.uint8)

    # Model
    modelRing = Unet(input_size=INPUT_SIZE, filter_num=FILTER_NUM, n_labels=N_LABELS, output_activation='linear').model
    modelPith = Unet(input_size=INPUT_SIZE, filter_num=FILTER_NUM, n_labels=N_LABELS, output_activation='sigmoid').model

    treeRingSegment = TreeRingSegmentation()
    treeRingSegment.segmentImage(modelRing, modelPith, input_img)

    assert treeRingSegment.maskRings.shape == input_img[:, :, 0].shape


def test_evaluation():
    # Mask and prediction
    mask = np.random.randint(0, 2, INPUT_SIZE[:-1], dtype=np.uint8)
    prediction = np.random.randint(0, 2, INPUT_SIZE[:-1], dtype=np.uint8)

    # Fake rings
    numRings = np.random.randint(1, 5)
    gtRings = []
    predictedRings = []
    for i in range(0, numRings):
        length = np.random.randint(3, 9999)
        gtRings.append(np.random.randint(0, INPUT_SIZE[0], (length, 2)))
        predictedRings.append(np.random.randint(0, INPUT_SIZE[0], (length, 2)))

    # Evaluation
    evaluation = Evaluation(mask, prediction, gtRings=gtRings, predictedRings=predictedRings)
    hausdorff = evaluation.evaluateHausdorff()
    mAR = evaluation.evaluatemAR()
    arand = evaluation.evaluateARAND()
    _recall, _precision, _f1, _acc = evaluation.evaluateRPFA()

    assert hausdorff >= 0
    assert mAR >= 0 and mAR <= 1
    assert arand >= 0 and arand <= 1
    assert _recall >= 0 and _recall <= 1
    assert _precision >= 0 and _precision <= 1
    assert _f1 >= 0 and _f1 <= 1
    assert _acc >= 0 and _acc <= 1
    