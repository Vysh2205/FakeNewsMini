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
    upload_dir = os.path.join(backend_dir, "uploads")
    
    # 1. Genuine Personal JPEG (user photo in artifacts if present)
    user_photo_path = r"C:\Users\vyshn\.gemini\antigravity\brain\08abe981-3b0a-4824-98f8-d52691308a53\.user_uploaded\media_1791395885575.jpg"
    if not os.path.exists(user_photo_path):
        user_photo_path = os.path.join(os.path.dirname(__file__), "test_genuine_portrait.jpg")
        img = Image.fromarray(np.uint8(np.random.normal(128, 30, (800, 600, 3)).clip(0, 255)))
        img.save(user_photo_path, "JPEG", quality=90)

    # 2. Another Genuine Photograph (Camera landscape photo)
    photo2_path = os.path.join(os.path.dirname(__file__), "test_genuine_landscape.jpg")
    img2 = Image.fromarray(np.uint8(np.random.normal(120, 40, (1920, 1080, 3)).clip(0, 255)))
    img2.save(photo2_path, "JPEG", quality=95)

    # 3. Known Manipulated Image (Edited / Spliced image)
    manipulated_path = os.path.join(os.path.dirname(__file__), "test_manipulated_edit.jpg")
    img3_arr = np.uint8(np.random.uniform(10, 240, (512, 512, 3)))
    # Insert sharp artificial spliced block
    img3_arr[100:300, 100:300] = 255
    img3 = Image.fromarray(img3_arr)
    img3.save(manipulated_path, "JPEG", quality=40)

    # 4. Known AI-Generated / Deepfake Image
    ai_path = os.path.join(os.path.dirname(__file__), "test_midjourney_ai_generated.jpg")
    img4_arr = np.uint8(np.ones((1024, 1024, 3)) * 128)
    img4 = Image.fromarray(img4_arr)
    img4.save(ai_path, "JPEG", quality=100)

    return user_photo_path, photo2_path, manipulated_path, ai_path

def test_four_image_cases():
    print("==================================================")
    print("  IMAGE FORENSIC ML PIPELINE MULTI-CASE TEST      ")
    print("==================================================\n")

    p1, p2, p3, p4 = generate_test_images()
    upload_dir = os.path.join(backend_dir, "uploads")

    cases = [
        ("A. Genuine Personal JPEG Photo", p1, "test_genuine_portrait.jpg"),
        ("B. Another Genuine Photograph", p2, "test_genuine_landscape.jpg"),
        ("C. Known Manipulated / Spliced Image", p3, "test_manipulated_edit.jpg"),
        ("D. Known AI-Generated / Deepfake Image", p4, "test_midjourney_ai_generated.jpg")
    ]

    results_table = []

    for label, path, fname in cases:
        with open(path, "rb") as f:
            bytes_data = f.read()

        print(f"--- RUNNING TEST CASE: {label} ---")
        res = process_image_file(bytes_data, fname, upload_dir)
        pred = res.get("prediction")
        conf = res.get("confidence")
        probs = res.get("probabilities")
        is_fake = res.get("is_fake")

        print(f"  Filename      : {fname}")
        print(f"  Prediction    : {pred}")
        print(f"  Is Fake       : {is_fake}")
        print(f"  Confidence    : {conf}")
        print(f"  Probabilities : {probs}\n")

        results_table.append({
            "case": label,
            "filename": fname,
            "prediction": pred,
            "confidence": f"{conf*100:.1f}%",
            "probs": probs
        })

    print("==================================================")
    print("  SUMMARY TEST TABLE                              ")
    print("==================================================")
    print(f"{'Test Case':38s} | {'Prediction':32s} | {'Confidence':10s}")
    print("-" * 88)
    for r in results_table:
        print(f"{r['case']:38s} | {r['prediction']:32s} | {r['confidence']:10s}")

if __name__ == "__main__":
    test_four_image_cases()
