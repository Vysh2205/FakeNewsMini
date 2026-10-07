import os
import json
import numpy as np
import pandas as pd

def generate_audio_dataset():
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    dataset_path = os.path.join(dataset_dir, "audio_deepfake_dataset.csv")
    os.makedirs(dataset_dir, exist_ok=True)

    np.random.seed(42)
    n_real = 150
    n_fake = 150
    n_total = n_real + n_fake

    # Feature columns: 20 MFCC means, 10 MFCC stds, Spectral Centroid, ZCR, Chroma, Spectral Rolloff, Spectral Bandwidth
    feature_names = [f"mfcc_mean_{i}" for i in range(1, 21)] + \
                    [f"mfcc_std_{i}" for i in range(1, 11)] + \
                    ["spec_centroid_mean", "spec_centroid_std", "zcr_mean", "zcr_std",
                     "chroma_mean", "chroma_std", "spec_rolloff_mean", "spec_bandwidth_mean",
                     "duration_sec"]

    data = []

    # 1. Generate Real Human Voice Acoustic Features (Label = 0)
    # Real voices have natural micro-tremors, continuous spectral variation, organic background noise
    for i in range(n_real):
        row = {}
        # MFCC Means (natural human vocal tract resonances)
        for k in range(1, 21):
            base = -15.0 if k == 1 else (10.0 / k)
            row[f"mfcc_mean_{k}"] = float(np.random.normal(base, 2.5))
        # MFCC Stds (higher natural dynamic range)
        for k in range(1, 11):
            row[f"mfcc_std_{k}"] = float(np.random.normal(4.5 + (1.5 / k), 1.2))
        
        row["spec_centroid_mean"] = float(np.random.normal(1600.0, 300.0))
        row["spec_centroid_std"] = float(np.random.normal(450.0, 80.0))
        row["zcr_mean"] = float(np.random.normal(0.08, 0.02))
        row["zcr_std"] = float(np.random.normal(0.04, 0.01))
        row["chroma_mean"] = float(np.random.normal(0.45, 0.08))
        row["chroma_std"] = float(np.random.normal(0.25, 0.05))
        row["spec_rolloff_mean"] = float(np.random.normal(3200.0, 500.0))
        row["spec_bandwidth_mean"] = float(np.random.normal(1800.0, 250.0))
        row["duration_sec"] = float(np.random.uniform(5.0, 60.0))
        row["label"] = 0  # Real Voice
        data.append(row)

    # 2. Generate AI-Generated / Deepfake Audio Features (Label = 1)
    # Synthetic voices (TTS, Voice Conversion) exhibit phase artifacts, overly smooth pitch curves, robotic high-frequency cutoffs
    for i in range(n_fake):
        row = {}
        for k in range(1, 21):
            base = -25.0 if k == 1 else (4.0 / k)
            row[f"mfcc_mean_{k}"] = float(np.random.normal(base, 1.8))
        for k in range(1, 11):
            row[f"mfcc_std_{k}"] = float(np.random.normal(2.1 + (0.8 / k), 0.6))  # Reduced variance (over-smoothness)
        
        row["spec_centroid_mean"] = float(np.random.normal(2400.0, 450.0))       # Higher synthetic spectral brightness
        row["spec_centroid_std"] = float(np.random.normal(210.0, 45.0))         # Lower spectral variance
        row["zcr_mean"] = float(np.random.normal(0.14, 0.03))                   # High-frequency switching/buzzy artifacts
        row["zcr_std"] = float(np.random.normal(0.02, 0.008))
        row["chroma_mean"] = float(np.random.normal(0.60, 0.06))
        row["chroma_std"] = float(np.random.normal(0.12, 0.03))
        row["spec_rolloff_mean"] = float(np.random.normal(4800.0, 650.0))
        row["spec_bandwidth_mean"] = float(np.random.normal(2300.0, 300.0))
        row["duration_sec"] = float(np.random.uniform(5.0, 60.0))
        row["label"] = 1  # AI-Generated Deepfake Voice
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(dataset_path, index=False)
    print(f"Audio Deepfake Dataset created successfully at {dataset_path} with {len(df)} records ({n_real} Real Voice, {n_fake} Deepfake AI).")

if __name__ == "__main__":
    generate_audio_dataset()
