import os
import io
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
from backend.services.image_verification import process_image_file

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def create_genuine_human_photo(filename, skin_tone=(210, 160, 130)):
    """Creates a synthetic real human photograph with natural camera noise & JPEG compression."""
    w, h = 600, 600
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Background
    img[:, :] = (180, 190, 200)
    # Human face oval
    cv2.ellipse(img, (300, 280), (140, 180), 0, 0, 360, skin_tone, -1)
    # Eyes
    cv2.circle(img, (250, 250), 18, (255, 255, 255), -1)
    cv2.circle(img, (250, 250), 8, (40, 30, 20), -1)
    cv2.circle(img, (350, 250), 18, (255, 255, 255), -1)
    cv2.circle(img, (350, 250), 8, (40, 30, 20), -1)
    # Lips
    cv2.ellipse(img, (300, 360), (45, 20), 0, 0, 180, (160, 60, 70), -1)

    # Add natural photographic sensor noise (PRNU pattern)
    sensor_noise = np.random.normal(0, 12, (h, w, 3)).astype(np.int16)
    img_noisy = np.clip(img.astype(np.int16) + sensor_noise, 0, 255).astype(np.uint8)

    pil = Image.fromarray(img_noisy)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    with open(filename, "wb") as f:
        f.write(buf.getvalue())
    return buf.getvalue()

def create_real_landscape_photo(filename):
    """Creates a real landscape photo with natural texture and noise."""
    w, h = 600, 600
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky
    img[:300, :] = (230, 180, 100)
    # Grass / Hills
    img[300:, :] = (50, 120, 40)
    # Sun
    cv2.circle(img, (450, 120), 45, (255, 240, 180), -1)

    sensor_noise = np.random.normal(0, 14, (h, w, 3)).astype(np.int16)
    img_noisy = np.clip(img.astype(np.int16) + sensor_noise, 0, 255).astype(np.uint8)

    pil = Image.fromarray(img_noisy)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    with open(filename, "wb") as f:
        f.write(buf.getvalue())
    return buf.getvalue()

def create_manipulated_image(filename):
    """Creates a heavily edited/spliced image with inconsistent compression and zero noise in edit area."""
    w, h = 600, 600
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (100, 100, 100)

    # Base noise
    sensor_noise = np.random.normal(0, 10, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + sensor_noise, 0, 255).astype(np.uint8)

    # Paste ultra-smooth zero-noise digital rectangle (splice artifact)
    img[150:450, 150:450] = (255, 0, 128)

    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=60)
    with open(filename, "wb") as f:
        f.write(buf.getvalue())
    return buf.getvalue()

def create_ai_generated_image(filename):
    """Creates a synthetic AI image with low PRNU noise, over-saturated gradient, and heavy Gaussian blur."""
    w, h = 600, 600
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[:, :, 0] = np.linspace(20, 240, w, dtype=np.uint8)
    base[:, :, 1] = np.linspace(240, 20, h, dtype=np.uint8)[:, None]
    base[:, :, 2] = 200

    pil = Image.fromarray(base).filter(ImageFilter.GaussianBlur(radius=4.5))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=75)
    with open(filename, "wb") as f:
        f.write(buf.getvalue())
    return buf.getvalue()

def run_pipeline_test():
    upload_dir = "backend/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    test_cases = [
        ("genuine_human_photo_1.jpg", "Genuine Human Face Photo 1", create_genuine_human_photo("genuine_human_photo_1.jpg", (220, 170, 140)), "REAL"),
        ("genuine_human_photo_2.jpg", "Genuine Human Face Photo 2", create_genuine_human_photo("genuine_human_photo_2.jpg", (190, 130, 90)), "REAL"),
        ("genuine_human_photo_3.jpg", "Genuine Human Face Photo 3", create_genuine_human_photo("genuine_human_photo_3.jpg", (240, 190, 160)), "REAL"),
        ("genuine_landscape.jpg", "Real Landscape Photo", create_real_landscape_photo("backend/ml/test_genuine_landscape.jpg"), "REAL"),
        ("genuine_object.jpg", "Real Object Photo", create_real_landscape_photo("genuine_object.jpg"), "REAL"),
        ("manipulated_edit_1.jpg", "Manipulated Spliced Image 1", create_manipulated_image("backend/ml/test_manipulated_edit.jpg"), "FAKE"),
        ("manipulated_edit_2.jpg", "Manipulated Spliced Image 2", create_manipulated_image("manipulated_edit_2.jpg"), "FAKE"),
        ("ai_generated_midjourney.jpg", "AI Generated Midjourney Image", create_ai_generated_image("backend/ml/test_midjourney_ai_generated.jpg"), "FAKE"),
        ("ai_generated_dalle.jpg", "AI Generated DALL-E Image", create_ai_generated_image("ai_generated_dalle.jpg"), "FAKE"),
    ]

    print("\n" + "="*80, flush=True)
    print("RUNNING FULL IMAGE PIPELINE AUDIT VERIFICATION TEST", flush=True)
    print("="*80 + "\n", flush=True)

    results = []

    for fname, label, bbytes, expected in test_cases:
        res = process_image_file(bbytes, fname, upload_dir)
        pred = res.get("prediction", "ERROR")
        conf = res.get("confidence", 0.0)
        probs = res.get("raw_probabilities", {})
        results.append((label, fname, expected, pred, f"{conf*100:.1f}%", f"REAL={probs.get('REAL', 0):.4f}, FAKE={probs.get('FAKE', 0):.4f}"))

    print("\n" + "="*80, flush=True)
    print("FINAL TEST PERFORMANCE TABLE", flush=True)
    print("="*80, flush=True)
    print(f"{'Image Label':<30} | {'Expected':<8} | {'Prediction':<10} | {'Confidence':<10} | {'Probabilities'}", flush=True)
    print("-" * 85, flush=True)
    for lbl, fname, exp, pred, conf_str, prob_str in results:
        status_symbol = "[PASS]" if exp == pred else "[FAIL]"
        print(f"{lbl:<30} | {exp:<8} | {pred:<10} | {conf_str:<10} | {prob_str} {status_symbol}", flush=True)
    print("="*80 + "\n", flush=True)

if __name__ == "__main__":
    run_pipeline_test()
