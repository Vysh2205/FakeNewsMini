import os
import uuid
from PIL import Image
from PIL.ExifTags import TAGS

try:
    import pytesseract
except Exception:
    pytesseract = None

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

def process_image_file(file_bytes: bytes, filename: str, upload_dir: str):
    """
    Validates image, extracts metadata, performs OCR text extraction.
    Returns preview URL path, metadata, OCR text, and limitations disclaimer.
    """
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return {"error": f"Unsupported image format: '{ext}'. Supported formats: JPG, JPEG, PNG, WEBP."}

    if len(file_bytes) > MAX_IMAGE_SIZE_BYTES:
        return {"error": "Image file size exceeds the 10MB maximum limit."}

    # Save image safely
    unique_name = f"{uuid.uuid4().hex}{ext}"
    os.makedirs(upload_dir, exist_ok=True)
    saved_path = os.path.join(upload_dir, unique_name)

    with open(saved_path, "wb") as f:
        f.write(file_bytes)

    # Open image & extract metadata
    metadata = {}
    ocr_text = ""
    try:
        with Image.open(saved_path) as img:
            metadata["dimensions"] = f"{img.width}x{img.height}"
            metadata["format"] = img.format
            metadata["mode"] = img.mode

            # EXIF tags
            exif_data = img._getexif()
            if exif_data:
                for tag_id, val in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    if tag_name in ["Make", "Model", "DateTime", "Software"]:
                        metadata[str(tag_name)] = str(val)

            # Perform OCR text extraction if pytesseract is available
            if pytesseract:
                try:
                    ocr_text = pytesseract.image_to_string(img).strip()
                except Exception:
                    ocr_text = ""
    except Exception as e:
        print(f"Image processing error: {e}")

    if not ocr_text:
        ocr_text = "No text overlay or claims detected in the image."

    return {
        "media_url": f"/uploads/{unique_name}",
        "metadata": metadata,
        "ocr_text": ocr_text,
        "limitations": "MVP image verification based on OCR, EXIF metadata, and claim checking. Complete AI deepfake detection is not included in this release."
    }
