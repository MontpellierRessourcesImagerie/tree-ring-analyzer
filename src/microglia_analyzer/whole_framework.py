from brightest_path_lib.algorithm import AStarSearch
import copy
import cv2
import glob
import math
import multiprocessing
from multiprocessing import Pool
import numpy as np
import os
from scipy.ndimage import binary_dilation
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu
import tensorflow as tf
from tensorflow.keras import layers
import tifffile
import matplotlib.pyplot as plt

def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.3)(p)
   return f, p

def attention_gate(g, s, num_filters):
    Wg = layers.Conv2D(num_filters, 3, padding="same")(g)
    Wg = layers.BatchNormalization()(Wg)
 
    Ws = layers.Conv2D(num_filters, 3, padding="same")(s)
    Ws = layers.BatchNormalization()(Ws)
 
    out = layers.Activation("relu")(Wg + Ws)
    out = layers.Conv2D(num_filters, 3, padding="same")(out)
    out = layers.Activation("sigmoid")(out)
 
    return out * s
    
def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   s = attention_gate(x, conv_features, n_filters)
   # concatenate
   x = layers.concatenate([x, s])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)
   return x

def build_unet_model(activation):
   inputs = layers.Input(shape=(256,256,3))
   # encoder: contracting path - downsample
   # 1 - downsample
   f1, p1 = downsample_block(inputs, 64)
   # 2 - downsample
   f2, p2 = downsample_block(p1, 128)

   # 3 - downsample
   f3, p3 = downsample_block(p2, 256)
   # 4 - downsample
   f4, p4 = downsample_block(p3, 512)
   # 5 - bottleneck
   bottleneck = double_conv_block(p4, 1024)
   # decoder: expanding path - upsample
   # 6 - upsample
   u6 = upsample_block(bottleneck, f4, 512)
   # 7 - upsample
   u7 = upsample_block(u6, f3, 256)
   # 8 - upsample
   u8 = upsample_block(u7, f2, 128)
   # 9 - upsample
   u9 = upsample_block(u8, f1, 64)
   # outputs
   outputs = layers.Conv2D(1, (1,1), padding="same", activation = activation)(u9)
   # unet model with Keras Functional API
   unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
   return unet_model   

def make_gradient_patch(patch_size, overlap, direction):
    """
    Builds a patch containing coefficients between 0 and 1 so it can be used to merge patches.
    The area not included in the overlap is filled with 1s, while a linear gradient from 1 to 0 is applied to the overlap area.
    The direction of the gradient is defined by the direction parameter, respecting the following order: [0: LEFT, 1: BOTTOM, 2: RIGHT, 3: TOP]
    The provided `patch_size` is the final size of the patch, including the overlap.

    Args:
        patch_size: (int) Size of the patch (height and width), overlap included.
        overlap: (int) Overlap between patches (in pixels).
        direction: (int) Direction of the gradient. 0: LEFT, 1: BOTTOM, 2: RIGHT, 3: TOP
    
    Returns:
        patch: (np.ndarray) Patch containing the coefficients.
    """
    gradient = np.linspace(0, 1, overlap)
    flat = np.ones(patch_size - overlap, np.float32)
    line = np.concatenate((gradient, flat))
    patch = np.tile(line, (patch_size, 1))
    for _ in range(direction):
        patch = np.rot90(patch)
    return patch

def normalize(image, lower_bound=0.0, upper_bound=1.0, dtype=np.float32):
    """
    Normalizes the value of an image between `lower_bound` and `upper_bound`.
    Works whatever the number of channels.
    The normalization is not applied in place.

    Args:
        image: (np.ndarray) Image to normalize.
        lower_bound: (float) Lower bound of the normalization.
        upper_bound: (float) Upper bound of the normalization.
        dtype: (np.dtype) Type of the output image.
    
    Returns:
        img: (np.ndarray) Normalized image.
    """
    img = image.astype(np.float32)
    # If the image contains only zeros.
    if np.abs(np.max(img)) < 1e-5 and np.abs(np.min(img)) < 1e-5:
        return img
    # If the image contains more than one value.
    if np.max(img) - np.min(img) > 1e-6:
        img -= np.min(img)
    img /= np.max(img)
    img *= (upper_bound - lower_bound)
    img += lower_bound
    return img.astype(dtype)

class ImageTiler2D(object):

    def __init__(self, patch_size, overlap, shape, blending='gradient'):
        # if len(shape) != 2:
        #     raise ValueError("This class is only suitable for 2D images.")
        if (shape[0] < patch_size) or (shape[1] < patch_size):
            raise ValueError("The input image must be at least as large as a patch.")
        if (overlap > int(patch_size / 2)):
            raise ValueError("Overlap must be smaller than half the patch size.")
        if (patch_size == 0) or (shape[0] == 0) or (shape[1] == 0):
            raise ZeroDivisionError("Patch size and image size must be non-zero.")
        # Size of the patches (height and width), overlap included.
        self.patch_size     = patch_size
        # Overlap between patches (in pixels).
        self.overlap        = overlap
        # Step between each patch.
        self.step           = patch_size - overlap
        # Shape of the images that we will want to cut or assemble.
        self.shape          = shape
        # Layout of the patches, which is a list of Patch2D objects.
        self.layout         = None
        # Number of patches on each axis, represented as a tuple (nY, nX).
        self.grid_size      = None
        # Patches containing coefficients to merge the patches.
        self.blending_coefs = None
        # Blending method to use for the merging of the patches. ('gradient', 'flat')
        self.blending       = blending.lower()
        # -----
        self._process_grid_size()
        self._process_cutting_layout()
        self._make_coefs()

    def image_to_tiles(self, image, use_normalize=True, lower_bound=0.0, upper_bound=1.0, dtype=np.float32):
        """
        Takes an image and cuts it into tiles according to the layout processed for this shape.
        The image is not modified. Works with any number of channels.
        The produced tiles have the possibility to be normalized in any desired range.

        Args:
            image: (np.ndarray) Image to cut into patches.
            use_normalize: (bool) Whether to normalize the image or not (not destructive).
            lower_bound: (float) Lower bound of the normalization.
            upper_bound: (float) Upper bound of the normalization.
            dtype: (np.dtype) Type of the output tiles.

        Returns:
            patches: (list) List of patches (np.ndarray) cut from the image.
        """
        tgt_shape = image.shape[:2]
        # if tgt_shape != self.shape:
        #     print(tgt_shape, self.shape)
        #     raise ValueError("Image's shape does not match the expected shape.")
        if use_normalize:
            image = normalize(image, lower_bound, upper_bound, dtype)
        patches = []
        for patch in self.layout:
            ul, lr = patch.ul_corner, patch.lr_corner
            patches.append(image[ul[0]:lr[0], ul[1]:lr[1]].copy())
        return patches
    
    def _make_coefs_gradients(self):
        """
        For every patch, creates a gradient map being the same size as the patch.
        It contains values between 0 and 1. This new patch has to be multiplied with the original patch to merge them.
        It intends to create a smooth blending between patches.
        The patches produced here consist in arrays full of 1s, with a (linear) gradient from 1 to 0 in the overlap area.
        Summing all these tiles into an image, at their position, results in an image where each pixel is 1.0.
        """
        coefs = []
        for patch in self.layout:
            gradient = np.ones((self.patch_size, self.patch_size), np.float32)
            for n in range(len(patch.has_neighbour)):
                if patch.has_neighbour[n]:
                    # By multiplying, we can handle quad-connexions.
                    gradient = np.multiply(gradient, make_gradient_patch(self.patch_size, patch.overlaps[n], n))
            coefs.append(gradient)
        self.blending_coefs = coefs
    
    def _make_coefs_flats(self):
        """
        For every patch, creates a coefficients map being the same size as the patch.
        It contains values between 0 and 1. This new patch has to be multiplied with the original patch to merge them.
        It intends to create a smooth blending between patches.
        The patches produced here consist flat areas (no gradient).
        The value contained in each pixel is: 1.0 / (number of patches sharing this pixel).
        Summing all these tiles into an image, at their position, results in an image where each pixel is 1.0.
        """
        canvas = np.zeros(self.shape, np.float32)
        stamp = np.ones((self.patch_size, self.patch_size), np.float32)
        for p in self.layout:
            canvas[p.ul_corner[0]:p.lr_corner[0], p.ul_corner[1]:p.lr_corner[1]] += stamp
        canvas = np.ones_like(canvas) / canvas
        self.blending_coefs = self.image_to_tiles(canvas, False)

    def _make_coefs(self):
        """
        Triggers the generation of blending patches according to the chosen method.
        """
        if self.blending == 'gradient':
            self._make_coefs_gradients()
        elif self.blending == 'flat':
            self._make_coefs_flats()
        else:
            raise ValueError("Unknown blending method.")
    
    def get_layout(self):
        """
        Returns the layout of the patches.
        It is a list of Patch2D objects.
        """
        return self.layout
    
    def get_grid_size(self):
        """
        Returns the number of patches on each axis.
        It is a tuple (nY, nX).
        """
        return self.grid_size
    
    def _process_grid_size(self):
        """
        Processes the final number of tiles on each axis, taking into account the overlap.
        """
        height, width = self.shape[0], self.shape[1]
        self.grid_size = (
            math.ceil((height - self.overlap) / self.step),
            math.ceil((width - self.overlap) / self.step)
        )

    def _process_cutting_layout(self):
        self.layout = []
        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                self.layout.append(Patch2D(
                    self.patch_size, 
                    self.overlap, 
                    (y, x), 
                    self.shape, 
                    self.grid_size
                ))
    
    def tiles_to_image(self, patches):
        """
        Takes a list of tiles (images) and uses the coefs maps to fusion them into a single image with a smooth blending.
        The goal is to assemble them with seamless blending, respecting the overlap.
        The order in the list must match the order of the layout.
        Input patches are not modified. Works with any number of channels.

        Args:
            patches: (list) List of patches (np.ndarray) to merge.

        Returns:
            canvas: (np.ndarray) Merged
        """
        if len(patches) != len(self.layout):
            raise ValueError("The number of patches does not match the layout.")
        if patches[0].shape[:2] != (self.patch_size, self.patch_size):
            raise ValueError("The shape of the patches does not match the expected shape.")
        copies = [i.copy().astype(np.float32) for i in patches]
        new_shape = self.shape
        n_channels = 1
        if len(patches[0].shape) == 3:
            new_shape += (patches[0].shape[2],)
            n_channels = patches[0].shape[2]
        canvas = np.zeros(new_shape, np.float32)
        for i, p in enumerate(self.layout):
            coef = np.stack([self.blending_coefs[i]] * n_channels, axis=-1) if n_channels > 1 else self.blending_coefs[i]
            copies[i] *= coef
            canvas[p.ul_corner[0]:p.lr_corner[0], p.ul_corner[1]:p.lr_corner[1]] += copies[i]
        return canvas.astype(patches[0].dtype)
    
class Patch2D(object):

    def __init__(self, patch_size, overlap, indices, shape, grid):
        """
        Builds a representation of a patch in a 2D image.
        The patch is defined by its upper left and lower right corners, as well as its overlap with its neighbours.
        The overlap is defined as the number of pixels that are shared with the neighbour.
        Coordinates are in the Python order: (y, x), with Y being upside-down.

        Args:
            patch_size: (int) Size of the patch (height and width), overlap included.
            overlap: (int) Overlap between patches.
            indices: (tuple) Indices of the patch in the grid (vertical and horizontal index in the grid of patches).
            shape: (tuple) Shape of the image.
            grid: (tuple) Number of patches on each axis.
        """
        # Total height and width of patches, including overlap (necessarily a square).
        self.patch_size    = patch_size
        # Minimum overlap between patches, in number of pixels.
        self.overlap       = overlap
        # Indices (y, x) of this patch within the grid of patches.
        self.indices       = indices
        # Shape of the global image (height, width).
        self.shape         = shape
        # Grid size (number of patches on each axis).
        self.grid          = grid
        # Step between each patch.
        self.step          = patch_size - overlap
        # Does the current patch have a neighbour on each side?
        self.has_neighbour = [False, False, False, False]
        # Overlap size with each neighbour.
        self.overlaps      = [0    , 0    , 0    , 0]
        # Upper left corner of the patch.
        self.ul_corner     = None
        # Lower right corner of the patch.
        self.lr_corner     = None
        # -----
        self._process_patch()
        self._check_neighbours()
        self._process_overlap()
    
    def __str__(self):
        return f"Patch2D({self.patch_size} > {self.ul_corner}, {self.lr_corner}, {self.overlaps})"

    def _process_patch(self):
        """
        From the indices on the grid, determines the upper-left and lower-right corners of the patch.
        If the patch is on an edge, the overlap is increased to conserve a constant patch size.
        The upper-left corner is processed from the lower-right corner.
        On both axes, the lower coordinate is included, while the upper one is excluded.
        It implies that the last patch will contain indices corresponding to the shape of the image.
        """
        height, width = self.shape[0], self.shape[1]
        y, x = self.indices[0] * self.step, self.indices[1] * self.step
        lower_right = (
            min(y + self.patch_size, height), 
            min(x + self.patch_size, width)
        )
        upper_left = (
            lower_right[0] - self.patch_size,
            lower_right[1] - self.patch_size
        )
        self.lr_corner = lower_right
        self.ul_corner = upper_left
    
    def _check_neighbours(self):
        """
        Determines the presence of neighbours for the current patch.
        For the left and top edges, we just hav eto check if we are touching the index 0.
        For the right and bottom edges, we have to check if the indices match the grid size.
        Note that if the overlap is set to 0, the patches won't ever have any neighbour.
        """
        y, x = self.indices
        self.has_neighbour[0] = (self.overlap > 0) and (x > 0)
        self.has_neighbour[1] = (self.overlap > 0) and (y < self.grid[0] - 1)
        self.has_neighbour[2] = (self.overlap > 0) and (x < self.grid[1] - 1)
        self.has_neighbour[3] = (self.overlap > 0) and (y > 0)

    def _process_overlap(self):
        """
        According to the presence of neighbours, determines the overlap size with each neighbour.
        The overlap size varies depending on the position of the patch in the image. If we are on an edge, the overlap is increased.
        In the case of the bottom and right edges, we also check whether the next patch would exceed the image size.
        Otherwise, the overlap wouldn't be symmetric.
        """
        y, x = self.indices[0] * self.step, self.indices[1] * self.step
        if self.has_neighbour[0]:
            self.overlaps[0] = x + self.overlap - self.ul_corner[1]
        if self.has_neighbour[1]:
            self.overlaps[1] = max(-self.shape[0] + y + 2 * self.patch_size, self.overlap)
        if self.has_neighbour[2]:
            self.overlaps[2] = max(-self.shape[1] + x + 2 * self.patch_size, self.overlap)
        if self.has_neighbour[3]:
            self.overlaps[3] = y + self.overlap - self.ul_corner[0]
    
    def as_napari_rectangle(self):
        """
        Returns this bounding-box as a Napari rectangle, which is a numpy array:
        [
            [yMin, xMin],
            [yMin, xMax],
            [xMax, yMin],
            [xMax, yMax]
        ]
        """
        return np.array([
            [self.ul_corner[0], self.ul_corner[1]],
            [self.ul_corner[0], self.lr_corner[1]],
            [self.lr_corner[0], self.lr_corner[1]],
            [self.lr_corner[0], self.ul_corner[1]]
        ])

def plot_half_ring(image, peak1, peak2, dark_point, light_point):
    start_point = np.array([light_point, peak1])
    goal_point = np.array([light_point, peak2])
    search_algorithm = AStarSearch(image, start_point=start_point, goal_point=goal_point)
    brightest_path =search_algorithm.search()

    result = np.array(search_algorithm.result)[:, 1]
    brightest_path = np.array(search_algorithm.result)[:, 0]

    result = np.append(start_point[1], np.append(result, goal_point[1]))
    brightest_path = np.append(dark_point, np.append(brightest_path, dark_point))

    cor = np.stack([result[:, None], brightest_path[:, None]], axis=-1)
    return cor

if __name__ == '__main__':
    # Parameters
    input_folder = '/home/khietdang/Documents/khiet/treeRing/input'
    # input_folder = '/home/khietdang/Documents/khiet/treeRing/input/12 E 1 b_8µm_x50.tif'
    output_folder = '/home/khietdang/Documents/khiet/treeRing/output'
    checkpoint_ring_path = '/home/khietdang/Documents/khiet/tree-ring-analyzer/src/models/bigDistance.keras'
    checkpoint_pith_path = '/home/khietdang/Documents/khiet/tree-ring-analyzer/src/models/pith.h5'
    batch_size = 8
    patch_size = 256
    overlap = patch_size - 196

    # Load image paths and create output folder
    if os.path.isdir(input_folder):
        image_list = glob.glob(os.path.join(input_folder, '*.tif'))
    else:
        image_list = [input_folder]

    os.makedirs(output_folder, exist_ok=True)

    # Load models
    model_ring = build_unet_model(activation='linear')
    model_ring.load_weights(checkpoint_ring_path)

    model_pith = build_unet_model(activation='sigmoid')
    model_pith.load_weights(checkpoint_pith_path)

    for image_path in image_list:
        print(image_path)

        # Load image
        image = tifffile.imread(image_path)
        shape = image.shape

        # Ring prediction
        ## Tiling
        tiles_manager = ImageTiler2D(patch_size, overlap, shape)
        tiles = tiles_manager.image_to_tiles(image)
        tiles = np.array(tiles)

        ## Prediction
        prediction_ring = np.squeeze(model_ring.predict(tiles, batch_size=batch_size, verbose=0))

        ## Reconstruction
        tiles_manager = ImageTiler2D(patch_size, overlap, shape[:2])
        prediction_ring = tiles_manager.tiles_to_image(prediction_ring)

        # Pith prediction
        ## Center identification from ring prediction
        ret = threshold_otsu(prediction_ring)
        ring_indices = np.where(prediction_ring > ret)
        center = int(np.mean(ring_indices[0])), int(np.mean(ring_indices[1]))

        ## Cropping
        crop_size = int(0.1 * max(shape[1], shape[0])) * 2
        crop_img = image[center[0] - int(crop_size / 2):center[0] + int(crop_size / 2),
                         center[1] - int(crop_size / 2):center[1] + int(crop_size / 2)]
        crop_img = cv2.resize(crop_img, (patch_size, patch_size))[None, :, :, :]
        crop_img = crop_img / 255

        ## Prediction
        prediction_crop_pith = model_pith.predict(crop_img, batch_size=1, verbose=0)
        prediction_crop_pith = cv2.resize(prediction_crop_pith[0], (crop_size, crop_size))
        prediction_pith = np.zeros((shape[0], shape[1]))
        prediction_pith[center[0] - int(crop_size / 2):center[0] + int(crop_size / 2),
                        center[1] - int(crop_size / 2):center[1] + int(crop_size / 2)] = copy.deepcopy(prediction_crop_pith)
        
        ## Binarization
        pith = np.zeros((shape[0], shape[1]))
        pith[prediction_pith > 0.5] = 1

        # Tracing
        ## Resize
        prediction_ring = cv2.resize(prediction_ring, (int(shape[1] / 10), int(shape[0] / 10)))
        height, width = prediction_ring.shape
        pith = cv2.resize(pith, (width, height))

        ## Center identification from pith
        one_indice = np.where(pith == 1)
        center = int(np.mean(one_indice[0])), int(np.mean(one_indice[1]))

        ## Delete pith area from ring prediction
        pith_dilated = binary_dilation(pith, iterations=int(0.005 * width))
        prediction_ring = prediction_ring * (1 - pith_dilated)

        ## Create dark line
        min_value = np.min(prediction_ring)
        dark_point = center[0]
        prediction_ring[dark_point, :] = copy.deepcopy(min_value)

        ## Find start and goal points
        light_part = np.mean(prediction_ring[dark_point - 10:dark_point + 10, :], axis=0)
        ret = threshold_otsu(light_part)
        peaks1, _ = find_peaks(light_part[:center[1]], height=ret, distance=0.05 * width)
        peaks1 = peaks1[::-1]
        peaks2, _ = find_peaks(light_part[center[1]:], height=ret, distance=0.05 * width)
        peaks2 = peaks2 + center[1]

        if len(peaks1) < len(peaks2):
            a = copy.deepcopy(peaks1)
            peaks1 = copy.deepcopy(peaks2)
            peaks2 = copy.deepcopy(a)

        ## Catch up the pairs of start points and goal points
        peaks1_center = np.abs(peaks1 - center[1])
        peaks2_center = np.abs(peaks2 - center[1])
        num_pair = min(len(peaks1), len(peaks2))
        data = []
        remains = list(np.arange(0, len(peaks1)))
        for i in range(0, num_pair, 1):
            difference = np.abs(peaks2_center[i] - peaks1_center)
            sort = np.argsort(difference)
            j = sort[np.isin(sort, remains)]
            if len(j):
                j = j[0]
                data.append((prediction_ring, peaks1[j], peaks2[i], dark_point, dark_point - 1))
                data.append((prediction_ring, peaks1[j], peaks2[i], dark_point, dark_point + 1))
                remains.remove(j)

        for j in remains:
            if 0 <= 2 * center[1] - peaks1[j] < width:
                data.append((prediction_ring, 2 * center[1] - peaks1[j], peaks1[j], dark_point, dark_point - 1))
                data.append((prediction_ring, 2 * center[1] - peaks1[j], peaks1[j], dark_point, dark_point + 1))

        ## Tracing half ring for each pair
        with Pool(multiprocessing.cpu_count()) as pool:
            results = pool.starmap(plot_half_ring, data)

        ## Check each half ring and draw
        prediction_final = np.zeros_like(prediction_ring, dtype=np.uint8)
        add = False
        i = 0
        while i < len(results):
            image_i = np.zeros_like(prediction_ring, dtype=np.uint8)
            cv2.polylines(image_i, results[i], True, 1, 1)
            cv2.polylines(image_i, results[i + 1], True, 1, 1)
            j = i + 2
            while j < len(results):
                image_j = np.zeros_like(prediction_ring, dtype=np.uint8)
                cv2.polylines(image_j, results[j], True, 1, 1)
                cv2.polylines(image_j, results[j + 1], True, 1, 1)
                image_ij = np.bitwise_and(image_i, image_j)
                sum_ij1 = np.sum(image_ij[:dark_point])
                sum_ij2 = np.sum(image_ij[dark_point + 1:])
                if sum_ij1 >= 1 and sum_ij2 >= 1:
                    add = True
                    if len(results[i]) < len(results[j]):
                        prediction_final = np.bitwise_or(prediction_final, image_i)
                    else:
                        prediction_final = np.bitwise_or(prediction_final, image_j)
                    results.pop(j)
                    results.pop(j)
                elif sum_ij1 >= 1 and sum_ij2 == 0:
                    add = True
                    if len(results[i]) < len(results[j]):
                        prediction_final = np.bitwise_or(prediction_final, image_i)
                        prediction_final[dark_point + 1:] = np.bitwise_or(prediction_final[dark_point + 1:], image_j[dark_point + 1:])
                        opposite_ring = np.stack([results[j + 1][:, :, 0][:, :, None], 
                                             2 * dark_point - results[j + 1][:, :, 1][:, :, None]], axis=-1)
                        cv2.polylines(prediction_final, opposite_ring.astype(np.int32), True, 1, 1)
                    else:
                        prediction_final = np.bitwise_or(prediction_final, image_j)
                        prediction_final[dark_point + 1:] = np.bitwise_or(prediction_final[dark_point + 1:], image_i[dark_point + 1:])
                        opposite_ring = np.stack([results[i + 1][:, :, 0][:, :, None],
                                            2 * dark_point - results[i + 1][:, :, 1][:, :, None]], axis=-1)
                        cv2.polylines(prediction_final, opposite_ring.astype(np.int32), True, 1, 1)
                    results.pop(j)
                    results.pop(j)
                elif sum_ij1 == 0 and sum_ij2 >= 1:
                    add = True
                    if len(results[i]) < len(results[j]):
                        prediction_final = np.bitwise_or(prediction_final, image_i)
                        prediction_final[:dark_point] = np.bitwise_or(prediction_final[:dark_point], image_j[:dark_point])
                        opposite_ring = np.stack([results[j][:, :, 0][:, :, None], 
                                                  2 * dark_point - results[j][:, :, 1][:, :, None]], axis=-1)
                        cv2.polylines(prediction_final, opposite_ring.astype(np.int32), True, 1, 1)
                    else:
                        prediction_final = np.bitwise_or(prediction_final, image_j)
                        prediction_final[:dark_point] = np.bitwise_or(prediction_final[:dark_point], image_i[:dark_point])
                        opposite_ring = np.stack([results[i][:, :, 0][:, :, None], 
                                             2 * dark_point - results[i][:, :, 1][:, :, None]], axis=-1)
                        cv2.polylines(prediction_final, opposite_ring.astype(np.int32), True, 1, 1)
                    
                    results.pop(j)
                    results.pop(j)
                j += 2
            if not add:
                prediction_final = np.bitwise_or(prediction_final, image_i)

            add = False
            i += 2
            
        ## Draw the pith ring
        contours, _ = cv2.findContours(pith.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        longest_contour = np.argmax(np.array([len(contour) for contour in contours]))
        cv2.drawContours(prediction_final, contours, longest_contour, 1, 1)

        ## Resize and draw over the original image
        prediction_final = cv2.resize(prediction_final, (shape[1], shape[0]))
        image[prediction_final == 1] = 0

        # Save result
        tifffile.imwrite(os.path.join(output_folder, os.path.basename(image_path)), image.astype(np.uint8))
