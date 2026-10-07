import os
import io
import uuid
from PIL import Image
from PIL.ExifTags import TAGS

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024  # 15MB

def process_image_file(file_bytes: bytes, filename: str, upload_dir: str):
    """
    Validates uploaded image file, extracts metadata & EXIF parameters,
    and returns explicit status indicating that a Deep CNN (ResNet50/EfficientNet)
    checkpoint trained on FaceForensics++/DFDC is required for genuine pixel deepfake classification.
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

            return {
                "status": "unavailable",
                "prediction": "Image Verification Model Unavailable",
                "is_fake": False,
                "verdict_type": "UNAVAILABLE",
                "verdict": "UNAVAILABLE",
                "confidence": 0.0,
                "model_used": "Deep CNN Checkpoint Needed (ResNet50 / EfficientNet-B0)",
                "media_url": f"/uploads/{unique_name}",
                "image_metadata": metadata,
                "message": "Image verification model is currently unavailable. Genuine image deepfake detection requires a Deep Convolutional Neural Network (PyTorch ResNet50 / EfficientNet-B0) trained on facial deepfake benchmark datasets (FaceForensics++ / DFDC)."
            }
    except Exception as e:
        print(f"Image Processing Error: {e}")
        return {"error": f"Image verification failed: {str(e)}"}
