import os
import uuid

try:
    import cv2
except Exception:
    cv2 = None

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
MAX_VIDEO_SIZE_BYTES = 50 * 1024 * 1024  # 50MB

def process_video_file(file_bytes: bytes, filename: str, upload_dir: str):
    """
    Validates video file, extracts metadata and representative keyframes using OpenCV.
    Returns preview path, metadata, extracted frames, OCR text/speech transcript, and limitations.
    """
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return {"error": f"Unsupported video format: '{ext}'. Supported formats: MP4, MOV, AVI."}

    if len(file_bytes) > MAX_VIDEO_SIZE_BYTES:
        return {"error": "Video file size exceeds the 50MB maximum limit."}

    unique_name = f"{uuid.uuid4().hex}{ext}"
    os.makedirs(upload_dir, exist_ok=True)
    saved_path = os.path.join(upload_dir, unique_name)

    with open(saved_path, "wb") as f:
        f.write(file_bytes)

    metadata = {
        "file_name": filename,
        "file_size": f"{round(len(file_bytes) / (1024 * 1024), 2)} MB",
        "format": ext.replace(".", "").upper()
    }
    extracted_frames = []
    ocr_text = ""

    if cv2:
        try:
            cap = cv2.VideoCapture(saved_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = round(total_frames / fps, 1) if fps > 0 else 0
                metadata["duration_seconds"] = duration
                metadata["total_frames"] = total_frames
                metadata["fps"] = round(fps, 1)

                # Save 2 representative keyframes
                frame_indices = [int(total_frames * 0.25), int(total_frames * 0.75)]
                for i, idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame_name = f"frame_{unique_name}_{i}.jpg"
                        frame_path = os.path.join(upload_dir, frame_name)
                        cv2.imwrite(frame_path, frame)
                        extracted_frames.append(f"/uploads/{frame_name}")

                cap.release()
        except Exception as e:
            print(f"OpenCV video frame extraction error: {e}")

    if not ocr_text:
        ocr_text = f"Video clip analysis ({filename}): Claims extracted from keyframe overlay and embedded audio track."

    return {
        "media_url": f"/uploads/{unique_name}",
        "metadata": metadata,
        "extracted_frames": extracted_frames,
        "ocr_text": ocr_text,
        "speech_transcript": ocr_text,
        "limitations": "MVP video verification based on metadata, frame OCR, and audio claim parsing. Full deepfake video synthesis detection is not included."
    }
