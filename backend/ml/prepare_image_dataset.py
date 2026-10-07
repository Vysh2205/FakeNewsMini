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
    # Real human face photos & camera images: organic skin color distributions, laplacian var 30-3000, ELA mean 3.0-35.0
    for i in range(n_real):
        row = {}
        row["ela_mean"] = float(np.random.normal(16.0, 6.0))
        row["ela_std"] = float(np.random.normal(10.0, 3.5))
        row["ela_max"] = float(np.random.normal(150.0, 30.0))

        row["r_mean"] = float(np.random.normal(135.0, 30.0))
        row["r_std"] = float(np.random.normal(48.0, 12.0))
        row["g_mean"] = float(np.random.normal(115.0, 25.0))
        row["g_std"] = float(np.random.normal(42.0, 10.0))
        row["b_mean"] = float(np.random.normal(105.0, 28.0))
        row["b_std"] = float(np.random.normal(40.0, 11.0))

        row["rg_cov"] = float(np.random.normal(2100.0, 600.0))
        row["gb_cov"] = float(np.random.normal(1900.0, 550.0))
        row["rb_cov"] = float(np.random.normal(2000.0, 580.0))

        row["laplacian_var"] = float(np.random.normal(600.0, 350.0)) # Soft face skin to sharp landscapes
        row["fft_high_freq_energy"] = float(np.random.normal(0.08, 0.03))
        row["noise_residual_std"] = float(np.random.normal(20.0, 6.0))

        row["exif_present"] = 1.0 if np.random.rand() > 0.5 else 0.0
        row["aspect_ratio"] = float(np.random.choice([1.0, 1.33, 1.5, 1.77]))
        row["width"] = float(np.random.normal(1280.0, 400.0))
        row["height"] = float(np.random.normal(960.0, 300.0))

        row["label"] = 0  # 0 = REAL
        data.append(row)

    # 2. Generate Manipulated / AI Deepfake Images (Label = 1: FAKE)
    # AI/Manipulated images: Abnormally flat ELA (< 1.5), stripped color variance, high frequency FFT grid artifacts
    for i in range(n_fake):
        row = {}
        row["ela_mean"] = float(np.random.normal(0.6, 0.3))       # Ultra-flat AI generated ELA
        row["ela_std"] = float(np.random.normal(0.3, 0.15))
        row["ela_max"] = float(np.random.normal(15.0, 5.0))

        row["r_mean"] = float(np.random.normal(140.0, 30.0))
        row["r_std"] = float(np.random.normal(15.0, 3.0))        # Over-smoothed skin variance
        row["g_mean"] = float(np.random.normal(135.0, 28.0))
        row["g_std"] = float(np.random.normal(12.0, 2.5))
        row["b_mean"] = float(np.random.normal(130.0, 32.0))
        row["b_std"] = float(np.random.normal(10.0, 2.0))

        row["rg_cov"] = float(np.random.normal(300.0, 80.0))
        row["gb_cov"] = float(np.random.normal(250.0, 70.0))
        row["rb_cov"] = float(np.random.normal(280.0, 75.0))

        row["laplacian_var"] = float(np.random.normal(18.0, 5.0))    # Extremely blurred/smoothed
        row["fft_high_freq_energy"] = float(np.random.normal(0.40, 0.08)) # High frequency grid noise
        row["noise_residual_std"] = float(np.random.normal(3.0, 1.0))

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
