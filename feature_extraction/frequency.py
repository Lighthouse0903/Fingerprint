import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
import math
import os
from pathlib import Path
from scipy.ndimage import label
import pyodbc
import uuid


def calculate_frequency_field(image, orientation_map_block, mask, block_size=16):

    (h_block, w_block) = orientation_map_block.shape
    default_freq = 1.0 / 8.0
    frequency_map = np.full_like(orientation_map_block, default_freq, dtype=np.float32)

    for r in range(h_block):
        for c in range(w_block):
            y_center = int((r + 0.5) * block_size)
            x_center = int((c + 0.5) * block_size)

            if (
                y_center >= mask.shape[0]
                or x_center >= mask.shape[1]
                or mask[y_center, x_center] == 0
                or r >= orientation_map_block.shape[0]
                or c >= orientation_map_block.shape[1]
                or orientation_map_block[r, c] == -1
            ):
                frequency_map[r, c] = 0
                continue
    return frequency_map
