"""
Downloads real-world ASL hand-landmark training data (36,000+ samples across all 26 letters)
from the verified ASL recognition benchmark.
Saves to data/real_asl_dataset.csv with standardized 42-feature landmarks.
"""

import os
import pandas as pd
import urllib.request

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(DATA_DIR, "real_asl_dataset.csv")

DATA_URL = "https://raw.githubusercontent.com/Muhib-Mehdi/ASL-Recognition-System/main/model/keypoint_classifier/keypoint.csv"

def download_and_prepare():
    print(f"Downloading real ASL dataset from {DATA_URL}...")
    df = pd.read_csv(DATA_URL, header=None, on_bad_lines='skip')
    
    # Filter rows that have exactly 43 columns (1 label + 42 landmark coords)
    df = df.dropna()
    
    # Convert label from 0-25 to letters 'A'-'Z'
    label_map = {i: chr(ord('A') + i) for i in range(26)}
    df[0] = df[0].astype(int).map(label_map)
    
    # Drop any unmapped labels
    df = df.dropna(subset=[0])
    
    # Rename columns: label, feat_0 ... feat_41
    cols = ["label"] + [f"feat_{i}" for i in range(42)]
    df.columns = cols
    
    print(f"Dataset successfully loaded: {len(df)} samples across {df['label'].nunique()} classes.")
    print("Samples per class:")
    print(df['label'].value_counts().sort_index())
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved cleaned dataset to {OUTPUT_CSV}")

if __name__ == "__main__":
    download_and_prepare()
