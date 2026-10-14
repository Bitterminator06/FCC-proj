"""
Train High-Accuracy ASL Hand Landmark Classifier.
Trains an optimized Random Forest on the benchmark dataset.
Saves model bundle to models/asl_model.pkl.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from config import DATASET_CSV_PATH, INPUT_FEATURE_COUNT, MODEL_PKL_PATH, MODELS_DIR

def train(data_path: str = DATASET_CSV_PATH, model_save_path: str = MODEL_PKL_PATH):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset not found at {data_path}. Run download_real_data.py first.")

    print(f"[Training] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    y = df["label"].values
    X = df.drop(columns=["label"]).values.astype(np.float32)

    classes = sorted(np.unique(y).tolist())
    print(f"[Training] Samples: {len(X)}, Features: {X.shape[1]}, Classes ({len(classes)}): {classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"[Training] Training Random Forest on {len(X_train)} samples...")
    clf = RandomForestClassifier(n_estimators=120, max_depth=20, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print(f"BENCHMARK TEST ACCURACY: {acc * 100:.2f}%")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    bundle = {
        "model": clf,
        "classes": classes,
        "n_features": INPUT_FEATURE_COUNT
    }
    with open(model_save_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"[Training] Model successfully saved to: {model_save_path}")

if __name__ == "__main__":
    train()
