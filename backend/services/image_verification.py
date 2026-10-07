import os
import io
import uuid
import pickle
import numpy as np
import cv2
from PIL import Image
from PIL.ExifTags import TAGS

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024  # 15MB

# Load Trained Image Forensics Model & Feature Scaler
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "image_classifier.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "image_scaler.pkl")

IMAGE_MODEL = None
IMAGE_SCALER = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, "rb") as f:
            IMAGE_MODEL = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            IMAGE_SCALER = pickle.load(f)
        print("[SUCCESS] Image Forensics Classifier and Scaler loaded successfully.", flush=True)
    else:
        print("[WARNING] Image Classifier models not found in backend/models/", flush=True)
except Exception as e:
    print(f"[ERROR] Failed to load Image Classifier model: {e}", flush=True)

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

def process_image_file(file_bytes: bytes, filename: str, upload_dir: str):
    """
    Validates uploaded image file, extracts metadata & EXIF parameters,
    and performs model inference using the trained Image Forensics Multi-Feature Classifier.
    """
    if not file_bytes or len(file_bytes) == 0:
        return {"error": "Uploaded image file is empty."}

    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return {"error": f"Unsupported image format: '{ext}'. Supported formats: JPG, JPEG, PNG, WEBP."}

    if len(file_bytes) > MAX_IMAGE_SIZE_BYTES:
        return {"error": "Image file size exceeds the 15MB maximum limit."}

    # Save file locally for preview
    os.makedirs(upload_dir, exist_ok=True)
    unique_name = f"img_{uuid.uuid4().hex[:10]}{ext}"
    saved_path = os.path.join(upload_dir, unique_name)
    with open(saved_path, "wb") as f:
        f.write(file_bytes)

    file_size_mb = round(len(file_bytes) / (1024 * 1024), 2)

    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            width, height = img.width, img.height
            format_name = img.format or ext.replace(".", "").upper()
            mode = img.mode

            # Extract EXIF Metadata Tags
            metadata = {
                "dimensions": f"{width}x{height}",
                "format": format_name,
                "mode": mode,
                "file_size_mb": file_size_mb
            }
            try:
                exif_data = img._getexif()
                if exif_data:
                    for tag_id, val in exif_data.items():
                        tag_name = TAGS.get(tag_id, tag_id)
                        if tag_name in ["Make", "Model", "DateTime", "Software"]:
                            metadata[str(tag_name)] = str(val)
            except Exception:
                pass

            # Check if model is loaded
            if IMAGE_MODEL is None or IMAGE_SCALER is None:
                return {
                    "status": "unavailable",
                    "prediction": "Image verification model unavailable",
                    "is_fake": False,
                    "verdict_type": "UNAVAILABLE",
                    "verdict": "UNAVAILABLE",
                    "confidence": 0.0,
                    "model_used": "Image Forensics Classifier Offline",
                    "media_url": f"/uploads/{unique_name}",
                    "image_metadata": metadata,
                    "message": "Image verification model unavailable. No trained classifier checkpoint could be loaded."
                }

            # 1. Feature Extraction
            features, shape = extract_image_features(file_bytes)
            
            # 2. Preprocessing / Scaling
            features_scaled = IMAGE_SCALER.transform([features])

            # 3. Model Inference & Raw Output Probabilities
            probs = IMAGE_MODEL.predict_proba(features_scaled)[0]
            # Class mapping: 0 = REAL, 1 = FAKE
            prob_real = float(probs[0])
            prob_fake = float(probs[1])

            predicted_index = int(np.argmax(probs))
            is_fake = bool(predicted_index == 1)
            prediction = "FAKE" if is_fake else "REAL"
            confidence = prob_fake if is_fake else prob_real

            class_mapping = {0: "REAL", 1: "FAKE"}
            model_name = "Image Forensics Multi-Feature Classifier (ELA + FFT Spectrum + PRNU Noise)"

            # Console Log Raw Model Output
            print("\n==================================================", flush=True)
            print("RAW MODEL OUTPUT:", flush=True)
            print(f"Filename: {filename}", flush=True)
            print(f"Model Name: {model_name}", flush=True)
            print(f"Input Shape: {shape}", flush=True)
            print(f"Probabilities: REAL = {prob_real:.4f}, FAKE = {prob_fake:.4f}", flush=True)
            print(f"Predicted Class Index: {predicted_index}", flush=True)
            print(f"Class Mapping: {class_mapping}", flush=True)
            print(f"Final Backend Prediction: {prediction}", flush=True)
            print("==================================================\n", flush=True)

            explanation = (
                f"Image analysis indicates potential digital manipulation or synthetic features (Confidence: {confidence*100:.1f}%)."
                if is_fake else
                f"Image analysis confirms authentic photographic noise patterns and compression consistency (Confidence: {confidence*100:.1f}%)."
            )

            return {
                "status": "success",
                "prediction": prediction,
                "is_fake": is_fake,
                "verdict_type": prediction,
                "verdict": "High Risk" if is_fake else "Low Risk",
                "confidence": round(confidence, 4),
                "raw_probabilities": {
                    "REAL": round(prob_real, 4),
                    "FAKE": round(prob_fake, 4)
                },
                "predicted_class_index": predicted_index,
                "class_mapping": class_mapping,
                "model_used": model_name,
                "media_url": f"/uploads/{unique_name}",
                "image_metadata": metadata,
                "explanation": explanation
            }
    except Exception as e:
        print(f"Image Processing Error: {e}", flush=True)
        return {"error": f"Image verification failed: {str(e)}"}
