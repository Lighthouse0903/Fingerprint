import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
import math
import os
from pathlib import Path
from scipy.ndimage import label
import pyodbc
import uuid


def create_segmented_mask(image, block_size=16, threshold_ratio=0.1):
    (h, w) = image.shape
    mask = np.zeros_like(image, dtype=bool)
    if np.std(image) < 1e-6 and np.mean(image) < 1e-6:
        variance_threshold = threshold_ratio
    elif np.std(image) < 1e-6:
        variance_threshold = threshold_ratio
    else:
        variance_threshold = np.std(image) ** 2 * threshold_ratio

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y : y + block_size, x : x + block_size]
            if block.size == 0:
                continue
            if np.std(block) ** 2 > variance_threshold:
                mask[y : y + block_size, x : x + block_size] = True

    if mask.shape != image.shape:
        print(
            "Cảnh báo: Kích thước mask không khớp với ảnh gốc trong create_segmented_mask"
        )

    mask = remove_small_objects(
        mask, min_size=max(1, block_size * block_size * 2), connectivity=2
    )
    mask = remove_small_holes(
        mask, area_threshold=max(1, block_size * block_size * 2), connectivity=2
    )
    return mask.astype(np.uint8) * 255
