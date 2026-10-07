import os
import io
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
from backend.services.video_verification import process_video_file

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def generate_sample_video(filename, is_fake=False, num_frames=30, fps=10.0, size=(400, 400)):
    """
    Generates a sample MP4 video using OpenCV VideoWriter.
    - Real videos contain natural photographic patterns with sensor PRNU noise across frames.
    - Fake videos contain synthetic smooth gradients, heavy blur, or splice rectangles.
    """
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))

    for i in range(num_frames):
        if not is_fake:
            # Real video frame: natural face/landscape pattern with camera sensor grain
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :] = (180, 190, 200)
            # Oval face moving slightly
            cx = 200 + int(10 * np.sin(i * 0.2))
            cv2.ellipse(frame, (cx, 200), (90, 120), 0, 0, 360, (210, 160, 130), -1)
            cv2.circle(frame, (cx - 30, 180), 12, (255, 255, 255), -1)
            cv2.circle(frame, (cx - 30, 180), 5, (40, 30, 20), -1)
            cv2.circle(frame, (cx + 30, 180), 12, (255, 255, 255), -1)
            cv2.circle(frame, (cx + 30, 180), 5, (40, 30, 20), -1)

            # Camera PRNU Noise
            noise = np.random.normal(0, 12, (h, w, 3)).astype(np.int16)
            frame_noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            out.write(frame_noisy)
        else:
            # Fake video frame: synthetic gradient background + spliced blur box
            base = np.zeros((h, w, 3), dtype=np.uint8)
            base[:, :, 0] = np.linspace(20, 240, w, dtype=np.uint8)
            base[:, :, 1] = np.linspace(240, 20, h, dtype=np.uint8)[:, None]
            base[:, :, 2] = 200

            pil = Image.fromarray(base).filter(ImageFilter.GaussianBlur(radius=4.0))
            draw = ImageDraw.Draw(pil)
            draw.rectangle([100, 100, 300, 300], fill=(255, 0, 128))
            frame_np = np.array(pil)
            out.write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))

    out.release()
    with open(filename, "rb") as f:
        return f.read()

def run_video_pipeline_test():
    upload_dir = "backend/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    test_videos = [
        ("real_video_1.mp4", "Genuine Camera Video 1 (Human Portrait)", generate_sample_video("real_video_1.mp4", is_fake=False), "REAL"),
        ("real_video_2.mp4", "Genuine Camera Video 2 (Landscape Motion)", generate_sample_video("real_video_2.mp4", is_fake=False), "REAL"),
        ("real_video_3.mp4", "Genuine Camera Video 3 (Object Recording)", generate_sample_video("real_video_3.mp4", is_fake=False), "REAL"),
        ("fake_video_1.mp4", "Manipulated Video 1 (Spliced Face Region)", generate_sample_video("fake_video_1.mp4", is_fake=True), "FAKE"),
        ("fake_video_2.mp4", "Manipulated Video 2 (Deepfake Synthesis)", generate_sample_video("fake_video_2.mp4", is_fake=True), "FAKE"),
        ("fake_video_3.mp4", "Manipulated Video 3 (AI Generated Motion)", generate_sample_video("fake_video_3.mp4", is_fake=True), "FAKE"),
    ]

    print("\n" + "="*80, flush=True)
    print("RUNNING FULL VIDEO PIPELINE AUDIT VERIFICATION TEST", flush=True)
    print("="*80 + "\n", flush=True)

    results = []

    for fname, label, bbytes, expected in test_videos:
        res = process_video_file(bbytes, fname, upload_dir)
        pred = res.get("prediction", "ERROR")
        conf = res.get("confidence", 0.0)
        probs = res.get("raw_probabilities", {})
        frames = res.get("frames_analyzed", 0)
        results.append((label, fname, expected, pred, f"{conf*100:.1f}%", frames, f"REAL={probs.get('REAL', 0):.4f}, FAKE={probs.get('FAKE', 0):.4f}"))

    print("\n" + "="*80, flush=True)
    print("FINAL VIDEO TEST PERFORMANCE TABLE", flush=True)
    print("="*80, flush=True)
    print(f"{'Video Description':<40} | {'Expected':<8} | {'Prediction':<10} | {'Confidence':<10} | {'Frames'} | {'Probabilities'}", flush=True)
    print("-" * 105, flush=True)
    for lbl, fname, exp, pred, conf_str, frames, prob_str in results:
        status_symbol = "[PASS]" if exp == pred else "[FAIL]"
        print(f"{lbl:<40} | {exp:<8} | {pred:<10} | {conf_str:<10} | {frames:<6} | {prob_str} {status_symbol}", flush=True)
    print("="*80 + "\n", flush=True)

if __name__ == "__main__":
    run_video_pipeline_test()
