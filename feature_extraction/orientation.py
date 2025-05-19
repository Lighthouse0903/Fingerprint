import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
import math
import os
from pathlib import Path
from scipy.ndimage import label
import pyodbc
import uuid


def calculate_orientation_field(image, mask, block_size=16):
    (h, w) = image.shape
    rows_omap = max(0, h // block_size)
    cols_omap = max(0, w // block_size)
    if rows_omap == 0 or cols_omap == 0:
        return np.array([[]], dtype=np.float32)

    orientation_map = np.zeros((rows_omap, cols_omap), dtype=np.float32)

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    for r in range(rows_omap):
        for c in range(cols_omap):
            y_start, y_end = r * block_size, min(h, (r + 1) * block_size)
            x_start, x_end = c * block_size, min(w, (c + 1) * block_size)

            block_mask = mask[y_start:y_end, x_start:x_end]
            if np.sum(block_mask) < (block_size * block_size * 0.1):
                orientation_map[r, c] = -1
                continue

            Gxx_sum = np.sum(sobel_x[y_start:y_end, x_start:x_end] ** 2)
            Gyy_sum = np.sum(sobel_y[y_start:y_end, x_start:x_end] ** 2)
            Gxy_sum = np.sum(
                sobel_x[y_start:y_end, x_start:x_end]
                * sobel_y[y_start:y_end, x_start:x_end]
            )

            denominator = Gxx_sum - Gyy_sum
            if abs(denominator) < 1e-6 and abs(2 * Gxy_sum) < 1e-6:
                theta = 0
            elif abs(denominator) < 1e-6:
                theta = np.pi / 2
            else:
                theta = 0.5 * np.arctan2(2 * Gxy_sum, denominator)

            orientation_map[r, c] = (theta + np.pi / 2) % np.pi  #
    return orientation_map
