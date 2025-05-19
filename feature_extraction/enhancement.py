import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
import math
import os
from pathlib import Path
from scipy.ndimage import label
import pyodbc
import uuid


def gabor_filter_enhancement(
    image, orientation_map_block, frequency_map_block, mask, block_size=16
):
    enhanced_image = image.astype(np.float32)
    (h_block, w_block) = orientation_map_block.shape
    (h_img, w_img) = image.shape

    for r in range(h_block):
        for c in range(w_block):
            if (
                r >= orientation_map_block.shape[0]
                or c >= orientation_map_block.shape[1]
                or orientation_map_block[r, c] == -1
                or r >= frequency_map_block.shape[0]
                or c >= frequency_map_block.shape[1]
                or frequency_map_block[r, c] <= 1e-5
            ):
                continue

            orientation_angle = orientation_map_block[r, c]
            frequency = frequency_map_block[r, c]
            wavelength = 1.0 / (frequency + 1e-6)

            sigma_x = 0.5 * wavelength
            sigma_y = 0.5 * wavelength
            gamma = 0.5
            psi = 0

            ksize = max(15, int(2.0 * wavelength))
            if ksize % 2 == 0:
                ksize += 1

            gabor_kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma_x,
                orientation_angle,
                wavelength,
                gamma,
                psi,
                ktype=cv2.CV_32F,
            )
            gabor_kernel -= np.mean(gabor_kernel)

            y_start, y_end = r * block_size, min(h_img, (r + 1) * block_size)
            x_start, x_end = c * block_size, min(w_img, (c + 1) * block_size)

            img_block = image[y_start:y_end, x_start:x_end]

            if img_block.size > 0 and np.sum(mask[y_start:y_end, x_start:x_end]) > (
                img_block.size * 0.1
            ):
                filtered_block = cv2.filter2D(
                    img_block.astype(np.float32),
                    -1,
                    gabor_kernel,
                    borderType=cv2.BORDER_REPLICATE,
                )
                enhanced_image[y_start:y_end, x_start:x_end] = filtered_block

    enhanced_image_norm = cv2.normalize(enhanced_image, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced_image_norm.astype(np.uint8)
