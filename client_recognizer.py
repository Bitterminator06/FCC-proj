"""
Real-time Local Client for ASL Recognition with Cloud Inference.
Features:
  - Local OpenCV webcam feed with MediaPipe Tasks Vision API (HandLandmarker).
  - Normalizes 21 2D landmarks into a 42-feature vector (translation and scale invariant).
  - Communicates asynchronously/RESTfully with Cloud Inference Endpoint (/predict).
  - Robust debounce mechanism with sentence formation buffer and HUD overlay.
"""

import time
from collections import Counter, deque
from typing import Dict, List, Optional
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests

from config import (
    API_INFERENCE_URL,
    CONFIDENCE_THRESHOLD,
    CONSECUTIVE_FRAMES_REQUIRED,
    COOLDOWN_SECONDS,
    HAND_LANDMARKER_TASK_PATH,
)
from landmark_utils import draw_hand_mesh, pre_process_landmark


class ASLClient:
    """Manages webcam capture, MediaPipe detection, API communication, and HUD rendering."""

    def __init__(self, api_url: str = API_INFERENCE_URL):
        self.api_url = api_url
        self.sentence = ""
        self.recent_predictions = deque(maxlen=CONSECUTIVE_FRAMES_REQUIRED)
        self.last_added_time = 0.0
        self.last_confirmed_letter = ""
        self.current_prediction = "None"
        self.current_confidence = 0.0
        self.network_latency_ms = 0.0
        self.top3_candidates: Dict[str, float] = {}

        # Initialize MediaPipe HandLandmarker
        base_options = python.BaseOptions(model_asset_path=HAND_LANDMARKER_TASK_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def send_inference_request(self, vector: List[float]) -> None:
        """Sends preprocessed 42-element landmark payload to the inference service."""
        t_start = time.time()
        try:
            resp = requests.post(self.api_url, json={"landmarks": vector}, timeout=0.35)
            self.network_latency_ms = (time.time() - t_start) * 1000
            if resp.status_code == 200:
                data = resp.json()
                self.current_prediction = data.get("letter", "?")
                self.current_confidence = data.get("confidence", 0.0)
                self.top3_candidates = data.get("all_probabilities", {})
                self.recent_predictions.append(self.current_prediction)
            else:
                self.current_prediction = "Error"
        except requests.exceptions.RequestException:
            self.network_latency_ms = -1.0
            self.current_prediction = "API Offline"

    def update_sentence_buffer(self, hand_present: bool) -> None:
        """Debounces and commits high-confidence predictions to the sentence buffer."""
        now = time.time()
        if not hand_present:
            self.recent_predictions.clear()
            self.current_prediction = "No Hand"
            self.current_confidence = 0.0
            self.top3_candidates = {}
            return

        if len(self.recent_predictions) == CONSECUTIVE_FRAMES_REQUIRED:
            most_common, count = Counter(self.recent_predictions).most_common(1)[0]
            if count >= (CONSECUTIVE_FRAMES_REQUIRED - 1) and self.current_confidence >= CONFIDENCE_THRESHOLD:
                time_since_last = now - self.last_added_time
                if (time_since_last > COOLDOWN_SECONDS) or (
                    most_common != self.last_confirmed_letter and time_since_last > 0.4
                ):
                    self.sentence += most_common
                    self.last_confirmed_letter = most_common
                    self.last_added_time = now
                    self.recent_predictions.clear()

    def render_hud(self, frame: cv2.Mat) -> None:
        """Renders header dashboard, confidence stats, and sentence buffer."""
        h, w, _ = frame.shape

        # Header overlay
        header = frame.copy()
        cv2.rectangle(header, (0, 0), (w, 100), (20, 20, 20), -1)
        cv2.addWeighted(header, 0.75, frame, 0.25, 0, frame)

        pred_color = (0, 255, 0) if self.current_confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        cv2.putText(frame, f"Sign: {self.current_prediction}", (20, 48), cv2.FONT_HERSHEY_DUPLEX, 1.3, pred_color, 2)
        cv2.putText(
            frame,
            f"Conf: {self.current_confidence * 100:.1f}%",
            (260, 48),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (220, 220, 220),
            2,
        )

        latency_str = f"API: {self.network_latency_ms:.1f}ms" if self.network_latency_ms >= 0 else "API Offline"
        latency_color = (100, 255, 100) if (0 <= self.network_latency_ms < 150) else (50, 100, 255)
        cv2.putText(frame, latency_str, (w - 240, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, latency_color, 2)

        if self.top3_candidates:
            top_str = "Top: " + " | ".join([f"{k}: {int(v * 100)}%" for k, v in self.top3_candidates.items()])
            cv2.putText(frame, top_str, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Footer overlay
        footer = frame.copy()
        cv2.rectangle(footer, (0, h - 80), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(footer, 0.8, frame, 0.2, 0, frame)

        display_sentence = self.sentence if self.sentence else "[Start signing letters...]"
        cv2.putText(frame, "Sentence: " + display_sentence, (20, h - 35), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(
            frame,
            "[Space] Space  |  [Backspace] Undo  |  [C] Clear  |  [Q] Quit",
            (20, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (160, 160, 160),
            1,
        )

    def run(self) -> None:
        """Main event loop for video capture and processing."""
        print("=" * 60)
        print("Starting ASL Real-Time Client...")
        print(f"Targeting Inference Endpoint: {self.api_url}")
        print("Controls: [Space] Add space | [Backspace] Delete | [C] Clear | [Q] Quit")
        print("=" * 60)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera device 0.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror view for intuitive user experience
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.detector.detect(mp_image)

            hand_present = bool(detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0)
            if hand_present:
                hand_landmarks = detection_result.hand_landmarks[0]
                draw_hand_mesh(frame, hand_landmarks)

                pixel_points = [
                    [min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)]
                    for lm in hand_landmarks
                ]
                norm_vector = pre_process_landmark(pixel_points)
                self.send_inference_request(norm_vector)

            self.update_sentence_buffer(hand_present)
            self.render_hud(frame)

            cv2.imshow("Real-Time ASL Recognizer (Cloud Connected)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key == 32:  # Spacebar
                self.sentence += " "
                self.last_confirmed_letter = " "
                self.last_added_time = time.time()
            elif key == 8:  # Backspace
                self.sentence = self.sentence[:-1]
            elif key in (ord("c"), ord("C")):
                self.sentence = ""

        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()


if __name__ == "__main__":
    client = ASLClient()
    client.run()
