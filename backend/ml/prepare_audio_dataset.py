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
    # Real speech: Natural lower vocal resonances, moderate spectral centroids (1000-2200Hz), low zero crossing rates
    for i in range(n_real):
        row = {}
        # MFCC Means (librosa speech scale: -180 for c0/c1, decaying organic components)
        row["mfcc_mean_1"] = float(np.random.normal(-160.0, 25.0))
        for k in range(2, 21):
            row[f"mfcc_mean_{k}"] = float(np.random.normal(12.0 / k, 4.0 / np.sqrt(k)))
        # MFCC Stds (higher dynamic range in natural speech)
        for k in range(1, 11):
            row[f"mfcc_std_{k}"] = float(np.random.normal(15.0 + (5.0 / k), 3.5))

        row["spec_centroid_mean"] = float(np.random.normal(1500.0, 250.0))
        row["spec_centroid_std"] = float(np.random.normal(400.0, 60.0))
        row["zcr_mean"] = float(np.random.normal(0.04, 0.015))
        row["zcr_std"] = float(np.random.normal(0.025, 0.008))
        row["chroma_mean"] = float(np.random.normal(0.22, 0.04))
        row["chroma_std"] = float(np.random.normal(0.18, 0.03))
        row["spec_rolloff_mean"] = float(np.random.normal(3000.0, 450.0))
        row["spec_bandwidth_mean"] = float(np.random.normal(1600.0, 200.0))
        row["duration_sec"] = float(np.random.uniform(3.0, 45.0))
        row["label"] = 0  # Real Voice
        data.append(row)

    # 2. Generate AI-Generated / Deepfake Audio Features (Label = 1)
    # AI synthetic speech (neural vocoders): flatter MFCC spectrum, higher spectral centroids (>2500Hz), elevated ZCR (>0.10)
    for i in range(n_fake):
        row = {}
        row["mfcc_mean_1"] = float(np.random.normal(-50.0, 30.0))
        for k in range(2, 21):
            row[f"mfcc_mean_{k}"] = float(np.random.normal(3.0 / k, 2.0))
        for k in range(1, 11):
            row[f"mfcc_std_{k}"] = float(np.random.normal(6.0 + (2.0 / k), 1.8))  # Lower std = over-smoothed synthetic voice

        row["spec_centroid_mean"] = float(np.random.normal(3800.0, 600.0))
        row["spec_centroid_std"] = float(np.random.normal(180.0, 35.0))
        row["zcr_mean"] = float(np.random.normal(0.18, 0.04))
        row["zcr_std"] = float(np.random.normal(0.012, 0.005))
        row["chroma_mean"] = float(np.random.normal(0.35, 0.05))
        row["chroma_std"] = float(np.random.normal(0.09, 0.02))
        row["spec_rolloff_mean"] = float(np.random.normal(5500.0, 700.0))
        row["spec_bandwidth_mean"] = float(np.random.normal(2600.0, 300.0))
        row["duration_sec"] = float(np.random.uniform(3.0, 45.0))
        row["label"] = 1  # AI-Generated Deepfake Voice
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(dataset_path, index=False)
    print(f"Audio Deepfake Dataset created successfully at {dataset_path} with {len(df)} records ({n_real} Real Voice, {n_fake} Deepfake AI).")

if __name__ == "__main__":
    generate_audio_dataset()
