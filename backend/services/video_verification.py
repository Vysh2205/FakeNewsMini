import os
import io
import uuid
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
MAX_VIDEO_SIZE_BYTES = 50 * 1024 * 1024  # 50MB

def process_video_file(file_bytes: bytes, filename: str, upload_dir: str):
    """
    Validates uploaded video file, extracts technical metadata and keyframes,
    and returns explicit status indicating that a Deep 3D-CNN (PyTorch EfficientNet-B0 / ResNet50)
    checkpoint trained on video deepfake benchmarks (FaceForensics++ / DFDC) is required.
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

    if cv2:
        try:
            cap = cv2.VideoCapture(saved_path)
            if cap.isOpened():
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

                # Extract 4 representative keyframes for preview
                frame_indices = np.linspace(0, max(0, total_frames - 1), 4, dtype=int) if total_frames > 0 else [0]
                for i, idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame_bgr = cap.read()
                    if ret and frame_bgr is not None:
                        frame_name = f"frame_{unique_name}_{i}.jpg"
                        frame_path = os.path.join(upload_dir, frame_name)
                        cv2.imwrite(frame_path, frame_bgr)
                        extracted_frames.append(f"/uploads/{frame_name}")
                cap.release()
        except Exception as e:
            print(f"OpenCV Keyframe Extraction Error: {e}", flush=True)

    print("\n==================================================", flush=True)
    print("VIDEO VERIFICATION DIAGNOSTIC AUDIT LOG:", flush=True)
    print(f"Filename: {filename}", flush=True)
    print(f"Metadata: {video_metadata}", flush=True)
    print("Status: UNAVAILABLE / INCONCLUSIVE", flush=True)
    print("Required Model: PyTorch Deep 3D-CNN / EfficientNet trained on FaceForensics++ / DFDC.", flush=True)
    print("==================================================\n", flush=True)

    return {
        "status": "unavailable",
        "prediction": "Video Verification Inconclusive",
        "is_fake": False,
        "verdict_type": "UNAVAILABLE",
        "verdict": "UNAVAILABLE",
        "confidence": 0.0,
        "model_used": "Deep CNN Checkpoint Needed (PyTorch EfficientNet-B0 / ResNet50)",
        "frames_analyzed": len(extracted_frames),
        "media_url": f"/uploads/{unique_name}",
        "extracted_frames": extracted_frames,
        "video_metadata": video_metadata,
        "message": "Video verification model is currently unavailable/inconclusive. Reliable video deepfake detection requires a Deep 3D-CNN / EfficientNet frame classifier trained on video deepfake benchmark datasets (FaceForensics++ / DFDC)."
    }
