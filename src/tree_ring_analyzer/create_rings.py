import glob
import tifffile
import os
import numpy as np

list_anno = glob.glob('/home/khietdang/Documents/khiet/INBD/dataset/EH/annotations/*.tiff')
ring_path = '/home/khietdang/Documents/khiet/INBD/dataset/EH/rings'
for anno_path in list_anno:
    anno = tifffile.imread(anno_path)
    anno[anno != 0] = 1
    anno = 1 - anno

    anno[anno == 1] = 255

    tifffile.imwrite(os.path.join(ring_path, os.path.basename(anno_path).split('.')[0] + '.jpg'), anno.astype(np.uint8))
