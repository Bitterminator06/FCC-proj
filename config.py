"""
Global Configuration and Constants for ASL Recognition System.
Centralizes API settings, model paths, thresholds, and skeletal geometry.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DATASET_CSV_PATH = os.path.join(DATA_DIR, "real_asl_dataset.csv")
MODEL_PKL_PATH = os.path.join(MODELS_DIR, "asl_model.pkl")
HAND_LANDMARKER_TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

# Network & Cloud API
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
API_INFERENCE_URL = os.getenv("ASL_API_URL", f"http://{DEFAULT_API_HOST}:{DEFAULT_API_PORT}/predict")

# Model & Features
INPUT_FEATURE_COUNT = 42  # 21 landmarks * (x, y) relative normalized coordinates
TOTAL_ALPHABET_CLASSES = 26

# Recognition & Debounce Parameters
CONFIDENCE_THRESHOLD = 0.55
CONSECUTIVE_FRAMES_REQUIRED = 6
COOLDOWN_SECONDS = 0.8

# Standard 21 Hand Connections for skeletal drawing
HAND_SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]
