import os
import io
import uuid
import pickle
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from PIL import Image
from PIL.ExifTags import TAGS
from backend.services.image_verification import extract_image_features, IMAGE_MODEL, IMAGE_SCALER

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
MAX_VIDEO_SIZE_BYTES = 50 * 1024 * 1024  # 50MB

def process_video_file(file_bytes: bytes, filename: str, upload_dir: str):
    """
    Validates uploaded video file, extracts representative keyframes using OpenCV,
    performs multi-frame image forensics feature extraction and model inference,
    aggregates frame probabilities, and logs raw model predictions.
    """
    if not file_bytes or len(file_bytes) == 0:
        return {"error": "Uploaded video file is empty."}

    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return {"error": f"Unsupported video format: '{ext}'. Supported formats: MP4, MOV, AVI."}

    if len(file_bytes) > MAX_VIDEO_SIZE_BYTES:
        return {"error": "Video file size exceeds the 50MB maximum limit."}

    os.makedirs(upload_dir, exist_ok=True)
    unique_name = f"vid_{uuid.uuid4().hex[:10]}{ext}"
    saved_path = os.path.join(upload_dir, unique_name)

    with open(saved_path, "wb") as f:
        f.write(file_bytes)

    file_size_mb = round(len(file_bytes) / (1024 * 1024), 2)
    video_metadata = {
        "file_name": filename,
        "file_size": f"{file_size_mb} MB",
        "format": ext.replace(".", "").upper()
    }

    extracted_frames = []
    frame_probabilities = []
    model_name = "Multi-Frame Video Forensics Classifier (ELA + 2D FFT + PRNU Noise)"
    class_mapping = {0: "REAL", 1: "FAKE"}

    if not cv2:
        return {
            "status": "unavailable",
            "prediction": "Video Verification Inconclusive",
            "is_fake": False,
            "verdict_type": "UNAVAILABLE",
            "verdict": "UNAVAILABLE",
            "confidence": 0.0,
            "model_used": "OpenCV Library Offline",
            "media_url": f"/uploads/{unique_name}",
            "video_metadata": video_metadata,
            "message": "Video processing requires OpenCV library for frame extraction."
        }

    try:
        cap = cv2.VideoCapture(saved_path)
        if not cap.isOpened():
            return {"error": "Failed to open uploaded video file for processing."}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = round(total_frames / fps, 1) if fps > 0 else 0.0

        video_metadata.update({
            "dimensions": f"{width}x{height}",
            "duration_seconds": duration,
            "total_frames": total_frames,
            "fps": round(fps, 1)
        })

        # Extract 10 to 16 representative frames evenly spaced across video
        num_samples = min(16, max(4, total_frames // 10 if total_frames > 0 else 8))
        if total_frames > 0:
            frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        else:
            frame_indices = [0]

        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                continue

            # Save sample keyframe preview for frontend gallery (first 4 frames)
            if i < 4:
                frame_name = f"frame_{unique_name}_{i}.jpg"
                frame_path = os.path.join(upload_dir, frame_name)
                cv2.imwrite(frame_path, frame_bgr)
                extracted_frames.append(f"/uploads/{frame_name}")

            # Preprocessing: Convert BGR to RGB JPEG bytes for feature extraction
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            buf = io.BytesIO()
            pil_frame.save(buf, format="JPEG", quality=95)
            frame_bytes = buf.getvalue()

            # Perform model inference if model is available
            if IMAGE_MODEL is not None and IMAGE_SCALER is not None:
                feat, _ = extract_image_features(frame_bytes)
                feat_scaled = IMAGE_SCALER.transform([feat])
                probs = IMAGE_MODEL.predict_proba(feat_scaled)[0]
                frame_probabilities.append(probs)

        cap.release()

        # Handle case where model is unavailable
        if IMAGE_MODEL is None or IMAGE_SCALER is None or len(frame_probabilities) == 0:
            return {
                "status": "unavailable",
                "prediction": "Video Verification Inconclusive",
                "is_fake": False,
                "verdict_type": "UNAVAILABLE",
                "verdict": "UNAVAILABLE",
                "confidence": 0.0,
                "model_used": "Video Forensics Classifier Offline",
                "media_url": f"/uploads/{unique_name}",
                "video_metadata": video_metadata,
                "extracted_frames": extracted_frames,
                "message": "Video verification model unavailable. No trained classifier checkpoint could be loaded."
            }

        # Multi-Frame Prediction Aggregation: Average Probabilities across analyzed frames
        avg_probs = np.mean(frame_probabilities, axis=0)
        prob_real = float(avg_probs[0])
        prob_fake = float(avg_probs[1])

        predicted_index = int(np.argmax(avg_probs))
        is_fake = bool(predicted_index == 1)
        prediction = "FAKE" if is_fake else "REAL"
        confidence = prob_fake if is_fake else prob_real

        # Print / Log Raw Video Model Output
        print("\n==================================================", flush=True)
        print("RAW VIDEO MODEL OUTPUT:", flush=True)
        print(f"Video Filename: {filename}", flush=True)
        print(f"Frames Analyzed: {len(frame_probabilities)}", flush=True)
        print(f"Model Name: {model_name}", flush=True)
        print(f"Probabilities: REAL = {prob_real:.4f}, FAKE = {prob_fake:.4f}", flush=True)
        print(f"Predicted Class Index: {predicted_index}", flush=True)
        print(f"Class Mapping: {class_mapping}", flush=True)
        print(f"Final Video Prediction: {prediction}", flush=True)
        print("==================================================\n", flush=True)

        explanation = (
            f"Multi-frame forensic analysis detected synthetic blurring or compression anomalies across {len(frame_probabilities)} keyframes (Confidence: {confidence*100:.1f}%)."
            if is_fake else
            f"Multi-frame forensic analysis confirms consistent photographic noise patterns and frame compression across {len(frame_probabilities)} keyframes (Confidence: {confidence*100:.1f}%)."
        )

        return {
            "status": "success",
            "content_type": "video",
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
            "frames_analyzed": len(frame_probabilities),
            "media_url": f"/uploads/{unique_name}",
            "extracted_frames": extracted_frames,
            "video_metadata": video_metadata,
            "explanation": explanation
        }

    except Exception as e:
        print(f"Video Processing Error: {e}", flush=True)
        return {"error": f"Video verification failed: {str(e)}"}
