"""
Local Data Collector for ASL Alphabet Hand Landmarks.
Allows recording personalized landmark samples using the MediaPipe Tasks API.
Appends normalized 42-feature samples to data/real_asl_dataset.csv or a custom file.
"""

import csv
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import DATA_DIR, HAND_LANDMARKER_TASK_PATH
from landmark_utils import draw_hand_mesh, pre_process_landmark

COLLECT_DATA_FILE = os.path.join(DATA_DIR, "custom_user_landmarks.csv")
RECORD_LIMIT_PER_KEY = 100

def initialize_csv(filepath: str):
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label"] + [f"feat_{i}" for i in range(42)])

def get_sample_counts(filepath: str):
    counts = {}
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    counts[row[0]] = counts.get(row[0], 0) + 1
    return counts

def main():
    initialize_csv(COLLECT_DATA_FILE)

    base_options = python.BaseOptions(model_asset_path=HAND_LANDMARKER_TASK_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    current_target = "A"
    recording = False
    samples_recorded = 0

    print("=" * 60)
    print("ASL Custom Landmark Collector (42 Features)")
    print(" - [A-Z]   : Change target letter")
    print(" - [SPACE] : Toggle recording samples")
    print(" - [Q]     : Exit")
    print("=" * 60)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        hand_present = bool(detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0)
        norm_vector = None

        if hand_present:
            hand_landmarks = detection_result.hand_landmarks[0]
            draw_hand_mesh(frame, hand_landmarks)

            pixel_points = [
                [min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)]
                for lm in hand_landmarks
            ]
            norm_vector = pre_process_landmark(pixel_points)

        if recording and hand_present and norm_vector:
            with open(COLLECT_DATA_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([current_target] + norm_vector)
            samples_recorded += 1
            if samples_recorded >= RECORD_LIMIT_PER_KEY:
                recording = False
                print(f"[Done] Reached {RECORD_LIMIT_PER_KEY} samples for {current_target}")

        counts = get_sample_counts(COLLECT_DATA_FILE)
        curr_count = counts.get(current_target, 0)
        status_color = (0, 0, 255) if recording else (0, 255, 0)
        status_text = (
            f"RECORDING ({samples_recorded}/{RECORD_LIMIT_PER_KEY})"
            if recording
            else "IDLE (Press SPACE to record)"
        )

        cv2.putText(frame, f"Target: {current_target} (Total: {curr_count})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, "Keys: [A-Z] letter | [Space] record | [Q] quit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("ASL Landmark Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        elif key == 32:  # Space
            recording = not recording
            if recording:
                samples_recorded = 0
        elif (ord("a") <= key <= ord("z")) or (ord("A") <= key <= ord("Z")):
            current_target = chr(key).upper()
            recording = False
            samples_recorded = 0

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()
