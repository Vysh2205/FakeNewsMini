import os
import io
import pickle
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def extract_image_features(file_bytes: bytes):
    """
    Extracts multi-domain image forensic features:
    1. Error Level Analysis (ELA) statistics (mean, std, max)
    2. Spatial 2D FFT High-to-Low frequency energy ratio
    3. Laplacian optical variance (focus & blur consistency)
    4. High-pass sensor noise variance (PRNU pattern approximation)
    5. Color Saturation distribution statistics
    """
    pil_img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img_np = np.array(pil_img)
    height, width, channels = img_np.shape

    # 1. Error Level Analysis (ELA) at 90% JPEG quality
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=90)
    buf.seek(0)
    ela_img = Image.open(buf).convert('RGB')
    ela_np = np.array(ela_img)

    ela_diff = np.abs(img_np.astype(np.float32) - ela_np.astype(np.float32))
    ela_mean = float(np.mean(ela_diff))
    ela_std = float(np.std(ela_diff))
    ela_max = float(np.max(ela_diff))

    # 2. 2D FFT Spatial Frequency Energy Ratio
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1e-8)

    cy, cx = height // 2, width // 2
    r = max(10, min(height, width) // 8)
    y, x = np.ogrid[:height, :width]
    mask_low = (x - cx)**2 + (y - cy)**2 <= r**2

    low_freq_energy = float(np.mean(magnitude_spectrum[mask_low]))
    high_freq_energy = float(np.mean(magnitude_spectrum[~mask_low]))
    freq_ratio = float(high_freq_energy / (low_freq_energy + 1e-8))

    # 3. Optical Focus & Edge Sharpness (Laplacian)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 4. Sensor PRNU High-pass Noise Estimation
    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_residual = gray.astype(np.float32) - blur_gray.astype(np.float32)
    noise_std = float(np.std(noise_residual))

    # 5. HSV Saturation Distribution
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    sat_mean = float(np.mean(sat))
    sat_std = float(np.std(sat))

    features = [
        ela_mean, ela_std, ela_max,
        freq_ratio, high_freq_energy, low_freq_energy,
        laplacian_var, noise_std,
        sat_mean, sat_std
    ]
    return np.array(features, dtype=np.float32), (height, width, channels)

def generate_synthetic_benchmark_dataset():
    X = []
    y = []
    np.random.seed(42)

    # 60 Genuine Camera Photo Samples (Class 0: REAL)
    for _ in range(60):
        w, h = 400, 400
        base = np.random.randint(40, 220, (h, w, 3), dtype=np.uint8)
        noise = np.random.normal(0, np.random.uniform(5, 15), (h, w, 3)).astype(np.int16)
        real_img_np = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        pil = Image.fromarray(real_img_np)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=np.random.randint(85, 98))
        feat, _ = extract_image_features(buf.getvalue())
        X.append(feat)
        y.append(0)  # 0 = REAL

    # 60 Deepfake / Manipulated Samples (Class 1: FAKE)
    for _ in range(60):
        w, h = 400, 400
        base = np.zeros((h, w, 3), dtype=np.uint8)
        base[:, :, 0] = np.linspace(50, 200, w, dtype=np.uint8)
        base[:, :, 1] = np.linspace(200, 50, h, dtype=np.uint8)[:, None]
        base[:, :, 2] = 128
        
        pil = Image.fromarray(base).filter(ImageFilter.GaussianBlur(radius=np.random.uniform(2, 5)))
        draw = ImageDraw.Draw(pil)
        draw.rectangle([w//4, h//4, 3*w//4, 3*h//4], fill=(255, 0, 128))
        
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=70)
        feat, _ = extract_image_features(buf.getvalue())
        X.append(feat)
        y.append(1)  # 1 = FAKE

    return np.array(X), np.array(y)

def train_and_save_image_model():
    print("Generating Image Forensics Training Benchmark...", flush=True)
    X, y = generate_synthetic_benchmark_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Model Training Complete!", flush=True)
    print(f"Accuracy:  {acc * 100:.2f}%", flush=True)
    print(f"Precision: {prec * 100:.2f}%", flush=True)
    print(f"Recall:    {rec * 100:.2f}%", flush=True)
    print(f"F1 Score:  {f1 * 100:.2f}%", flush=True)

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "image_classifier.pkl")
    scaler_path = os.path.join(models_dir, "image_scaler.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Saved image model to: {model_path}", flush=True)
    print(f"Saved scaler to: {scaler_path}", flush=True)

if __name__ == "__main__":
    train_and_save_image_model()
