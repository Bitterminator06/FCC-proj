"""
FastAPI Cloud Inference API for Static ASL Alphabet Recognition.
Provides low-latency microservice endpoints for hand landmark classification,
plus cloud observability metrics (Prometheus/CloudWatch style telemetry)
and architectural cost/bandwidth simulation.
"""

import os
import time
import pickle
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import MODEL_PKL_PATH, INPUT_FEATURE_COUNT

# Global model state & Cloud Telemetry Metrics
ml_state: Dict = {"model": None, "classes": []}

server_start_time = time.time()
metrics_state = {
    "total_requests": 0,
    "successful_inferences": 0,
    "failed_requests": 0,
    "total_latency_ms": 0.0,
    "min_latency_ms": float("inf"),
    "max_latency_ms": 0.0,
    "total_payload_bytes_received": 0,
    "bandwidth_saved_bytes_vs_raw_video": 0
}

# 1 HD frame (1280x720x3 bytes JPEG ~ 50KB, raw ~ 2.76 MB)
EQUIVALENT_RAW_FRAME_BYTES = 50 * 1024  # 50 KB compressed per video frame

def load_inference_model():
    """Loads the serialized classifier and label metadata."""
    if os.path.exists(MODEL_PKL_PATH):
        with open(MODEL_PKL_PATH, "rb") as f:
            bundle = pickle.load(f)
            ml_state["model"] = bundle.get("model")
            ml_state["classes"] = bundle.get("classes", [])
        print(f"[API] Loaded model successfully with {len(ml_state['classes'])} alphabet classes.")
    else:
        print(f"[API Warning] Model artifact not found at {MODEL_PKL_PATH}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown hooks."""
    load_inference_model()
    yield

# Initialize FastAPI service
app = FastAPI(
    title="ASL Alphabet Cloud Inference API",
    description="Edge-Cloud hybrid microservice demonstrating Cloud Computing fundamentals (PaaS, Elasticity, Observability).",
    version="2.5.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas
class LandmarkPayload(BaseModel):
    landmarks: List[float] = Field(
        ...,
        description=f"Exactly {INPUT_FEATURE_COUNT} normalized coordinate floats (x, y for 21 joints)"
    )

class PredictionResponse(BaseModel):
    letter: str
    confidence: float
    all_probabilities: Optional[Dict[str, float]] = None
    server_execution_time_ms: float
    instance_id: str

INSTANCE_ID = os.getenv("INSTANCE_ID", f"worker-{os.getpid()}")

@app.get("/", summary="Service Health & Cloud Metadata")
def health_check():
    """Returns service health status, cloud deployment tier, and model status."""
    uptime_seconds = round(time.time() - server_start_time, 2)
    return {
        "status": "online",
        "service": "ASL Alphabet Landmark Inference API",
        "cloud_tier": "PaaS (Platform-as-a-Service)",
        "instance_id": INSTANCE_ID,
        "uptime_seconds": uptime_seconds,
        "model_loaded": ml_state["model"] is not None,
        "classes_count": len(ml_state["classes"]),
        "input_features_required": INPUT_FEATURE_COUNT
    }

@app.get("/labels", summary="Supported Classes")
def get_supported_labels():
    """Returns the list of 26 recognized sign letters."""
    if not ml_state["classes"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model classes not loaded."
        )
    return {"classes": ml_state["classes"]}

@app.get("/metrics", summary="Cloud Telemetry & Observability")
def get_cloud_metrics():
    """
    Cloud Observability Endpoint (similar to Prometheus / AWS CloudWatch).
    Demonstrates request rates, latency distributions, and Edge vs Cloud bandwidth savings.
    """
    tot = metrics_state["total_requests"]
    avg_lat = round(metrics_state["total_latency_ms"] / tot, 2) if tot > 0 else 0.0
    bytes_rx = metrics_state["total_payload_bytes_received"]
    saved_bytes = metrics_state["bandwidth_saved_bytes_vs_raw_video"]
    savings_pct = round((saved_bytes / (saved_bytes + bytes_rx)) * 100, 2) if (saved_bytes + bytes_rx) > 0 else 0.0

    return {
        "cloud_instance_id": INSTANCE_ID,
        "uptime_seconds": round(time.time() - server_start_time, 2),
        "requests": {
            "total": tot,
            "successful": metrics_state["successful_inferences"],
            "failed": metrics_state["failed_requests"],
            "success_rate_pct": round((metrics_state["successful_inferences"] / tot) * 100, 2) if tot > 0 else 100.0
        },
        "latency_ms": {
            "average": avg_lat,
            "min": round(metrics_state["min_latency_ms"], 2) if tot > 0 else 0.0,
            "max": round(metrics_state["max_latency_ms"], 2)
        },
        "bandwidth_analysis": {
            "actual_landmarks_received_bytes": bytes_rx,
            "actual_landmarks_received_kb": round(bytes_rx / 1024, 2),
            "estimated_raw_video_bytes": saved_bytes + bytes_rx,
            "estimated_raw_video_mb": round((saved_bytes + bytes_rx) / (1024 * 1024), 2),
            "bandwidth_reduction_pct": f"{savings_pct}%"
        }
    }

@app.post("/predict", response_model=PredictionResponse, summary="Perform Real-Time Inference")
def predict_landmark_sign(payload: LandmarkPayload):
    """Classifies a 42-element hand landmark vector into an ASL letter."""
    start_t = time.perf_counter()
    metrics_state["total_requests"] += 1

    model = ml_state["model"]
    classes = ml_state["classes"]

    if model is None:
        metrics_state["failed_requests"] += 1
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is not currently loaded on server."
        )

    raw_features = payload.landmarks
    if len(raw_features) != INPUT_FEATURE_COUNT:
        metrics_state["failed_requests"] += 1
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected {INPUT_FEATURE_COUNT} landmark coordinates, got {len(raw_features)}"
        )

    # 42 floats in JSON ~ 350-400 bytes
    payload_size = len(raw_features) * 8 + 60
    metrics_state["total_payload_bytes_received"] += payload_size
    metrics_state["bandwidth_saved_bytes_vs_raw_video"] += (EQUIVALENT_RAW_FRAME_BYTES - payload_size)

    features_matrix = np.array(raw_features, dtype=np.float32).reshape(1, -1)

    try:
        probabilities = model.predict_proba(features_matrix)[0]
        max_idx = int(np.argmax(probabilities))
        predicted_letter = str(classes[max_idx])
        confidence = float(probabilities[max_idx])

        top3_indices = np.argsort(probabilities)[::-1][:3]
        top3_dict = {
            str(classes[idx]): round(float(probabilities[idx]), 4)
            for idx in top3_indices
        }

        exec_ms = round((time.perf_counter() - start_t) * 1000, 2)
        metrics_state["successful_inferences"] += 1
        metrics_state["total_latency_ms"] += exec_ms
        metrics_state["min_latency_ms"] = min(metrics_state["min_latency_ms"], exec_ms)
        metrics_state["max_latency_ms"] = max(metrics_state["max_latency_ms"], exec_ms)

        return PredictionResponse(
            letter=predicted_letter,
            confidence=round(confidence, 4),
            all_probabilities=top3_dict,
            server_execution_time_ms=exec_ms,
            instance_id=INSTANCE_ID
        )
    except Exception as e:
        metrics_state["failed_requests"] += 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    from config import DEFAULT_API_HOST, DEFAULT_API_PORT
    uvicorn.run(app, host=DEFAULT_API_HOST, port=DEFAULT_API_PORT)
