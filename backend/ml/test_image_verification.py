import os
import sys
import numpy as np
from PIL import Image

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.image_verification import process_image_file

def generate_test_images():
    # 1. Genuine Photo Simulation (Natural gradient, organic camera-like noise pattern)
    width, height = 800, 600
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    r = np.uint8((0.5 + 0.5 * np.sin(2 * np.pi * xx)) * 255)
    g = np.uint8((0.5 + 0.5 * np.cos(2 * np.pi * yy)) * 255)
    b = np.uint8((0.5 + 0.5 * np.sin(2 * np.pi * (xx + yy))) * 255)
    noise = np.uint8(np.random.normal(12, 5, (height, width, 3)))
    rgb_arr = np.clip(np.stack([r, g, b], axis=2).astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)

    real_img = Image.fromarray(rgb_arr)
    real_path = os.path.join(os.path.dirname(__file__), "test_genuine_photo.jpg")
    real_img.save(real_path, "JPEG", quality=95)

    # 2. Manipulated / Synthetic Deepfake Image Simulation (Midjourney / AI tag in filename + smooth flat patches)
    fake_img = Image.fromarray(np.uint8(np.random.uniform(50, 200, (512, 512, 3))))
    fake_path = os.path.join(os.path.dirname(__file__), "test_midjourney_synthetic.jpg")
    fake_img.save(fake_path, "JPEG", quality=70)

    return real_path, fake_path

def test_image_pipeline():
    print("==================================================")
    print("  IMAGE FORENSIC MODEL INFERENCE TEST             ")
    print("==================================================\n")

    real_path, fake_path = generate_test_images()
    upload_dir = os.path.join(backend_dir, "uploads")

    # 1. Test Genuine Photo
    with open(real_path, "rb") as f:
        real_bytes = f.read()

    print("--- TESTING GENUINE CAMERA PHOTO SAMPLE ---")
    res_real = process_image_file(real_bytes, "test_genuine_photo.jpg", upload_dir)
    print("Result:")
    print(f"  Image         : test_genuine_photo.jpg")
    print(f"  Prediction    : {res_real.get('prediction')}")
    print(f"  Is Fake       : {res_real.get('is_fake')}")
    print(f"  Confidence    : {res_real.get('confidence')}")
    print(f"  Probabilities : {res_real.get('probabilities')}")
    print(f"  Label Mapping : {res_real.get('label_mapping')}")
    print(f"  Indicators    : {res_real.get('forensic_indicators')}\n")

    # 2. Test Deepfake / Manipulated Image
    with open(fake_path, "rb") as f:
        fake_bytes = f.read()

    print("--- TESTING MANIPULATED / AI DEEPFAKE SAMPLE ---")
    res_fake = process_image_file(fake_bytes, "test_midjourney_synthetic.jpg", upload_dir)
    print("Result:")
    print(f"  Image         : test_midjourney_synthetic.jpg")
    print(f"  Prediction    : {res_fake.get('prediction')}")
    print(f"  Is Fake       : {res_fake.get('is_fake')}")
    print(f"  Confidence    : {res_fake.get('confidence')}")
    print(f"  Probabilities : {res_fake.get('probabilities')}")
    print(f"  Label Mapping : {res_fake.get('label_mapping')}")
    print(f"  Indicators    : {res_fake.get('forensic_indicators')}\n")

    # Clean up test files
    if os.path.exists(real_path): os.remove(real_path)
    if os.path.exists(fake_path): os.remove(fake_path)

if __name__ == "__main__":
    test_image_pipeline()
