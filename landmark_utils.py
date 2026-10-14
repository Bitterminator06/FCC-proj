"""
Landmark Processing and Visualization Utilities.
Handles invariant coordinate normalization and OpenCV skeletal mesh rendering.
"""

import copy
import itertools
from typing import List, Tuple
import cv2
import numpy as np
from config import HAND_SKELETON_CONNECTIONS

def pre_process_landmark(landmark_points: List[List[int]]) -> List[float]:
    """
    Normalizes 21 2D hand landmark pixel coordinates to be translation & scale invariant:
    1. Shifts all points relative to the wrist (landmark 0) -> Translation invariance.
    2. Flattens into a 42-element 1D list [x0, y0, x1, y1, ...].
    3. Divides by maximum absolute value -> Scale invariance bounded in [-1.0, 1.0].
    
    Args:
        landmark_points: List of 21 [x, y] integer pixel coordinates.
        
    Returns:
        42-element list of normalized floats.
    """
    temp_landmarks = copy.deepcopy(landmark_points)

    # 1. Translate relative to wrist (index 0)
    base_x, base_y = temp_landmarks[0][0], temp_landmarks[0][1]
    for idx in range(len(temp_landmarks)):
        temp_landmarks[idx][0] -= base_x
        temp_landmarks[idx][1] -= base_y

    # 2. Flatten coordinates
    flattened = list(itertools.chain.from_iterable(temp_landmarks))

    # 3. Scale normalization
    max_value = max(map(abs, flattened))
    if max_value == 0:
        max_value = 1.0

    return [val / max_value for val in flattened]


def draw_hand_mesh(frame: np.ndarray, landmarks, color_joint=(0, 255, 255), color_line=(0, 200, 0)):
    """
    Renders 21 joints and connecting skeletal lines onto an OpenCV video frame.
    
    Args:
        frame: OpenCV image frame (BGR).
        landmarks: MediaPipe normalized landmarks list.
        color_joint: BGR tuple for joint circles.
        color_line: BGR tuple for skeletal lines.
    """
    h, w, _ = frame.shape
    pts: List[Tuple[int, int]] = []
    
    for lm in landmarks:
        cx = min(int(lm.x * w), w - 1)
        cy = min(int(lm.y * h), h - 1)
        pts.append((cx, cy))
        cv2.circle(frame, (cx, cy), 4, color_joint, -1)

    for start_idx, end_idx in HAND_SKELETON_CONNECTIONS:
        cv2.line(frame, pts[start_idx], pts[end_idx], color_line, 2)
