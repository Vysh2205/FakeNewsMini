import os
import numpy as np
import pandas as pd

def generate_image_dataset():
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    dataset_path = os.path.join(dataset_dir, "image_forensic_dataset.csv")
    os.makedirs(dataset_dir, exist_ok=True)

    np.random.seed(42)
    n_real = 150
    n_fake = 150
    n_total = n_real + n_fake

    # Feature columns: ELA stats, RGB stats, Laplacian variance, FFT noise ratio, Color covariance, Exif indicator
    feature_names = [
        "ela_mean", "ela_std", "ela_max",
        "r_mean", "r_std", "g_mean", "g_std", "b_mean", "b_std",
        "rg_cov", "gb_cov", "rb_cov",
        "laplacian_var", "fft_high_freq_energy", "noise_residual_std",
        "exif_present", "aspect_ratio", "width", "height"
    ]

    data = []

    # 1. Generate Real Camera Photos & Human Portraits (Label = 0: REAL)
    # Real photos & passport portraits: organic skin tones, ELA mean 2.5-35.0, laplacian var 25-3000, color std 20-60
    for i in range(n_real):
        row = {}
        row["ela_mean"] = float(np.random.normal(12.0, 5.0))
        row["ela_std"] = float(np.random.normal(8.0, 3.0))
        row["ela_max"] = float(np.random.normal(120.0, 30.0))

        row["r_mean"] = float(np.random.normal(135.0, 30.0))
        row["r_std"] = float(np.random.normal(45.0, 12.0))
        row["g_mean"] = float(np.random.normal(115.0, 25.0))
        row["g_std"] = float(np.random.normal(38.0, 10.0))
        row["b_mean"] = float(np.random.normal(105.0, 28.0))
        row["b_std"] = float(np.random.normal(35.0, 11.0))

        row["rg_cov"] = float(np.random.normal(1800.0, 500.0))
        row["gb_cov"] = float(np.random.normal(1600.0, 450.0))
        row["rb_cov"] = float(np.random.normal(1700.0, 480.0))

        row["laplacian_var"] = float(np.random.normal(450.0, 300.0))
        row["fft_high_freq_energy"] = float(np.random.normal(0.08, 0.03))
        row["noise_residual_std"] = float(np.random.normal(18.0, 5.0))

        row["exif_present"] = 1.0 if np.random.rand() > 0.6 else 0.0
        row["aspect_ratio"] = float(np.random.choice([0.75, 1.0, 1.33, 1.5, 1.77]))
        row["width"] = float(np.random.normal(1024.0, 300.0))
        row["height"] = float(np.random.normal(1024.0, 300.0))

        row["label"] = 0  # 0 = REAL
        data.append(row)

    # 2. Generate Manipulated / AI Deepfake Images (Label = 1: FAKE)
    # AI/Manipulated images: Abnormally flat ELA (< 0.5), artificial zero variance (< 4.0), high frequency FFT grid noise (> 0.45)
    for i in range(n_fake):
        row = {}
        row["ela_mean"] = float(np.random.normal(0.2, 0.1))       # Ultra-flat AI generated ELA
        row["ela_std"] = float(np.random.normal(0.1, 0.05))
        row["ela_max"] = float(np.random.normal(8.0, 3.0))

        row["r_mean"] = float(np.random.normal(140.0, 30.0))
        row["r_std"] = float(np.random.normal(3.5, 1.0))         # Artificial flat color variance
        row["g_mean"] = float(np.random.normal(135.0, 28.0))
        row["g_std"] = float(np.random.normal(3.0, 0.8))
        row["b_mean"] = float(np.random.normal(130.0, 32.0))
        row["b_std"] = float(np.random.normal(2.5, 0.7))

        row["rg_cov"] = float(np.random.normal(50.0, 15.0))
        row["gb_cov"] = float(np.random.normal(40.0, 12.0))
        row["rb_cov"] = float(np.random.normal(45.0, 14.0))

        row["laplacian_var"] = float(np.random.normal(5.0, 1.5))     # Synthetic grid flat residual
        row["fft_high_freq_energy"] = float(np.random.normal(0.55, 0.08)) # High frequency grid noise
        row["noise_residual_std"] = float(np.random.normal(1.2, 0.4))

        row["exif_present"] = 0.0
        row["aspect_ratio"] = float(np.random.choice([1.0, 1.33]))
        row["width"] = float(np.random.normal(1024.0, 100.0))
        row["height"] = float(np.random.normal(1024.0, 100.0))

        row["label"] = 1  # 1 = FAKE
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(dataset_path, index=False)
    print(f"Image Forensic Dataset generated at {dataset_path} with {len(df)} samples ({n_real} Real, {n_fake} Deepfake/Manipulated).")

if __name__ == "__main__":
    generate_image_dataset()
