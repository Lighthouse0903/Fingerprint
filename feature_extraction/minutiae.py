import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
import math
import os
from pathlib import Path
from scipy.ndimage import label
import pyodbc
import uuid


def compute_crossing_number(window):
    pixels = [
        int(p > 0)
        for p in [
            window[1, 2],
            window[0, 2],
            window[0, 1],
            window[0, 0],
            window[1, 0],
            window[2, 0],
            window[2, 1],
            window[2, 2],
        ]
    ]
    cn = 0
    for i in range(8):
        cn += abs(pixels[i] - pixels[(i + 1) % 8])
    return cn // 2


def extract_minutiae(
    skeleton_img, roi_mask, orientation_map_block, block_size, border_margin=15
):
    minutiae = []
    (rows, cols) = skeleton_img.shape
    skeleton_padded = np.pad(
        skeleton_img.astype(np.uint8), 1, mode="constant", constant_values=0
    )

    for r_padded in range(1, rows + 1):
        for c_padded in range(1, cols + 1):
            r, c = r_padded - 1, c_padded - 1

            if roi_mask[r, c] == 0:
                continue
            if (
                r < border_margin
                or r >= rows - border_margin
                or c < border_margin
                or c >= cols - border_margin
            ):
                continue

            if skeleton_padded[r_padded, c_padded] == 1:
                window = skeleton_padded[
                    r_padded - 1 : r_padded + 2, c_padded - 1 : c_padded + 2
                ]
                cn = compute_crossing_number(window)

                minutia_type = None
                if cn == 1:
                    minutia_type = "ending"
                elif cn == 3:
                    minutia_type = "bifurcation"

                if minutia_type:
                    block_r = min(r // block_size, orientation_map_block.shape[0] - 1)
                    block_c = min(c // block_size, orientation_map_block.shape[1] - 1)

                    if block_r < 0 or block_c < 0:
                        continue

                    orientation_rad = orientation_map_block[block_r, block_c]

                    if orientation_rad != -1:
                        minutiae.append(
                            {
                                "x": c,
                                "y": r,
                                "type": minutia_type,
                                "orientation": orientation_rad,
                            }
                        )
    return minutiae


def remove_false_minutiae(
    minutiae_list,
    skeleton_img,
    roi_mask,
    orientation_map_block,
    block_size,
    min_distance_between_minutiae=8,
    spur_max_length=10,
    short_ridge_max_length=12,
    min_ridge_length=15,
    max_angle_diff=np.pi / 4,
    border_margin=10,
):
    if not minutiae_list:
        return []

    rows, cols = skeleton_img.shape
    filtered_minutiae = minutiae_list.copy()
    to_remove_indices = [False] * len(minutiae_list)

    def trace_ridge(y, x, max_length, visited):
        length = 0
        stack = [(y, x)]
        visited.add((y, x))
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        while stack and length < max_length:
            cy, cx = stack.pop()
            length += 1
            for dy, dx in directions:
                ny, nx = cy + dy, cx + dx
                if (
                    0 <= ny < rows
                    and 0 <= nx < cols
                    and skeleton_img[ny, nx]
                    and (ny, nx) not in visited
                ):
                    stack.append((ny, nx))
                    visited.add((ny, nx))
        return length

    def is_orientation_consistent(m, neighbor_minutiae, max_angle_diff):
        if not neighbor_minutiae:
            return True
        angles = [abs(m["orientation"] - n["orientation"]) for n in neighbor_minutiae]
        angles = [min(a, np.pi - a) for a in angles]
        return all(a < max_angle_diff for a in angles)

    for i in range(len(minutiae_list)):
        if to_remove_indices[i]:
            continue
        m1 = minutiae_list[i]
        for j in range(i + 1, len(minutiae_list)):
            if to_remove_indices[j]:
                continue
            m2 = minutiae_list[j]
            dist = np.sqrt((m1["x"] - m2["x"]) ** 2 + (m1["y"] - m2["y"]) ** 2)
            if dist < min_distance_between_minutiae:
                r1, c1 = m1["y"], m1["x"]
                r2, c2 = m2["y"], m2["x"]
                window_size = 5
                count1 = np.sum(
                    skeleton_img[
                        max(0, r1 - window_size) : r1 + window_size + 1,
                        max(0, c1 - window_size) : c1 + window_size + 1,
                    ]
                )
                count2 = np.sum(
                    skeleton_img[
                        max(0, r2 - window_size) : r2 + window_size + 1,
                        max(0, c2 - window_size) : c2 + window_size + 1,
                    ]
                )
                to_remove_indices[i if count1 < count2 else j] = True

    for i, m in enumerate(minutiae_list):
        if to_remove_indices[i]:
            continue
        y, x = m["y"], m["x"]
        if (
            y < border_margin
            or y >= rows - border_margin
            or x < border_margin
            or x >= cols - border_margin
            or roi_mask[y, x] == 0
        ):
            to_remove_indices[i] = True

    labeled_skeleton, num_features = label(skeleton_img)
    for i, m in enumerate(minutiae_list):
        if to_remove_indices[i]:
            continue
        y, x = m["y"], m["x"]
        visited = set()
        ridge_length = trace_ridge(y, x, min_ridge_length, visited)

        if ridge_length < min_ridge_length:
            to_remove_indices[i] = True
            continue

        if m["type"] == "ending":
            for m_bif in minutiae_list:
                if (
                    m_bif["type"] != "bifurcation"
                    or to_remove_indices[minutiae_list.index(m_bif)]
                ):
                    continue
                dist = np.sqrt((m["x"] - m_bif["x"]) ** 2 + (m["y"] - m_bif["y"]) ** 2)
                if dist > spur_max_length:
                    continue
                visited = set()
                queue = [(m["y"], m["x"])]
                found = False
                while queue:
                    cy, cx = queue.pop(0)
                    if (cy, cx) == (m_bif["y"], m_bif["x"]):
                        found = True
                        break
                    visited.add((cy, cx))
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = cy + dy, cx + dx
                            if dy == 0 and dx == 0:
                                continue
                            if (
                                0 <= ny < rows
                                and 0 <= nx < cols
                                and skeleton_img[ny, nx]
                                and (ny, nx) not in visited
                            ):
                                queue.append((ny, nx))
                if found:
                    to_remove_indices[i] = True
                    break

    temp_minutiae = [m for i, m in enumerate(minutiae_list) if not to_remove_indices[i]]
    to_remove_indices = [False] * len(temp_minutiae)
    for i in range(len(temp_minutiae)):
        if to_remove_indices[i]:
            continue
        m1 = temp_minutiae[i]
        if m1["type"] != "ending":
            continue
        for j in range(i + 1, len(temp_minutiae)):
            if to_remove_indices[j]:
                continue
            m2 = temp_minutiae[j]
            if m2["type"] != "ending":
                continue
            dist = np.sqrt((m1["x"] - m2["x"]) ** 2 + (m1["y"] - m2["y"]) ** 2)
            if dist < short_ridge_max_length:
                angle_diff = abs(m1["orientation"] - m2["orientation"])
                angle_diff = min(angle_diff, np.pi - angle_diff)
                if abs(angle_diff - np.pi) < np.pi / 3:
                    to_remove_indices[i] = True
                    to_remove_indices[j] = True
                    break

    temp_minutiae = [m for i, m in enumerate(temp_minutiae) if not to_remove_indices[i]]
    to_remove_indices = [False] * len(temp_minutiae)
    for i, m in enumerate(temp_minutiae):
        if to_remove_indices[i]:
            continue
        neighbors = [
            n
            for n in temp_minutiae
            if n != m and np.sqrt((m["x"] - n["x"]) ** 2 + (m["y"] - n["y"]) ** 2) < 20
        ]
        if not is_orientation_consistent(m, neighbors, max_angle_diff):
            to_remove_indices[i] = True

    final_minutiae = [
        m for i, m in enumerate(temp_minutiae) if not to_remove_indices[i]
    ]

    return final_minutiae
