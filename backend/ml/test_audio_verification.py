import os
import sys
import numpy as np
import scipy.io.wavfile as wav

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.audio_verification import process_audio_file

def generate_test_audio_files():
    sr = 22050
    duration = 3.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1. Real Speech Simulation: Organic fundamental frequency (150Hz) + formants + subtle pitch drift
    f0 = 150 + 10 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    real_signal = 0.5 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    # Add natural breathing/noise floor
    real_signal += 0.01 * np.random.randn(len(t))
    # Normalize to 16-bit PCM
    real_pcm = np.int16(real_signal / np.max(np.abs(real_signal)) * 32767)

    real_path = os.path.join(os.path.dirname(__file__), "test_real.wav")
    wav.write(real_path, sr, real_pcm)

    # 2. Synthetic AI Deepfake Simulation: High zero-crossing buzz, crisp high harmonics (2400Hz+), robotic pitch
    f_synth = 2400.0
    synth_signal = 0.4 * np.sin(2 * np.pi * f_synth * t) + 0.3 * np.sign(np.sin(2 * np.pi * 800 * t))
    # High frequency noise jitter
    synth_signal += 0.08 * np.random.randn(len(t))
    synth_pcm = np.int16(synth_signal / np.max(np.abs(synth_signal)) * 32767)

    fake_path = os.path.join(os.path.dirname(__file__), "test_fake.wav")
    wav.write(fake_path, sr, synth_pcm)

    return real_path, fake_path

def test_pipeline():
    print("==================================================")
    print("  AUDIO VERIFICATION MODEL INFERENCE TEST         ")
    print("==================================================\n")

    real_path, fake_path = generate_test_audio_files()

    upload_dir = os.path.join(backend_dir, "uploads")

    # Test Real Audio
    with open(real_path, "rb") as f:
        real_bytes = f.read()

    print("--- TESTING REAL HUMAN VOICE SAMPLE ---")
    res_real = process_audio_file(real_bytes, "test_real.wav", upload_dir)
    print("Result:")
    print(f"  Prediction    : {res_real.get('prediction')}")
    print(f"  Is Fake       : {res_real.get('is_fake')}")
    print(f"  Confidence    : {res_real.get('confidence')}")
    print(f"  Probabilities : {res_real.get('probabilities')}")
    print(f"  Label Mapping : {res_real.get('label_mapping')}")
    print(f"  Features      : {res_real.get('features_extracted')}\n")

    # Test Fake Audio
    with open(fake_path, "rb") as f:
        fake_bytes = f.read()

    print("--- TESTING AI GENERATED DEEPFAKE SAMPLE ---")
    res_fake = process_audio_file(fake_bytes, "test_fake.wav", upload_dir)
    print("Result:")
    print(f"  Prediction    : {res_fake.get('prediction')}")
    print(f"  Is Fake       : {res_fake.get('is_fake')}")
    print(f"  Confidence    : {res_fake.get('confidence')}")
    print(f"  Probabilities : {res_fake.get('probabilities')}")
    print(f"  Label Mapping : {res_fake.get('label_mapping')}")
    print(f"  Features      : {res_fake.get('features_extracted')}\n")

    # Clean up test wav files
    if os.path.exists(real_path): os.remove(real_path)
    if os.path.exists(fake_path): os.remove(fake_path)

if __name__ == "__main__":
    test_pipeline()
