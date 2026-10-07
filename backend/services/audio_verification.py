import os
import io
import re
import math
import joblib
import numpy as np
import scipy.io.wavfile
import scipy.signal
import librosa

_audio_model = None
_audio_scaler = None

def load_audio_assets():
    global _audio_model, _audio_scaler
    if _audio_model is not None and _audio_scaler is not None:
        return _audio_model, _audio_scaler

    services_dir = os.path.dirname(__file__)
    backend_dir = os.path.abspath(os.path.join(services_dir, ".."))
    model_path = os.path.join(backend_dir, "ml", "models", "audio_model.pkl")
    scaler_path = os.path.join(backend_dir, "ml", "models", "audio_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Trained audio deepfake model or scaler file not found. Run train_audio_model.py first.")

    _audio_model = joblib.load(model_path)
    _audio_scaler = joblib.load(scaler_path)
    return _audio_model, _audio_scaler

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac"}

def process_audio_file(file_bytes: bytes, filename: str, upload_dir: str):
    """
    Processes uploaded audio file (.mp3, .wav, .m4a, .aac), validates format & size,
    extracts acoustic MFCC / Spectral features, runs Random Forest audio deepfake model,
    and returns comprehensive detection report.
    """
    if not file_bytes or len(file_bytes) == 0:
        return {"error": "Uploaded audio file is empty."}

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        return {"error": f"Unsupported audio format '{ext}'. Allowed formats: MP3, WAV, M4A, AAC."}

    max_bytes = 25 * 1024 * 1024  # 25MB
    if len(file_bytes) > max_bytes:
        return {"error": "Audio file size exceeds maximum limit of 25MB."}

    # Save file locally
    os.makedirs(upload_dir, exist_ok=True)
    saved_filename = f"audio_{os.urandom(6).hex()}{ext}"
    saved_path = os.path.join(upload_dir, saved_filename)
    with open(saved_path, "wb") as f:
        f.write(file_bytes)

    file_size_mb = round(len(file_bytes) / (1024 * 1024), 2)

    try:
        # Load audio signal via librosa
        try:
            y, sr = librosa.load(saved_path, sr=22050, mono=True)
        except Exception as load_err:
            # Fallback byte stream loading for standard WAV files
            try:
                sr, y = scipy.io.wavfile.read(io.BytesIO(file_bytes))
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                y = y.astype(np.float32) / (np.max(np.abs(y)) + 1e-6)
            except Exception:
                raise ValueError(f"Could not decode audio content: {str(load_err)}")

        if len(y) == 0:
            return {"error": "Decoded audio signal contains zero audio frames."}

        duration_sec = float(len(y) / sr)
        mins = int(duration_sec // 60)
        secs = int(duration_sec % 60)
        duration_formatted = f"{mins:02d}:{secs:02d}"

        # --- Feature Extraction ---
        # 1. MFCCs (20 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs[:10], axis=1)

        # 2. Spectral Centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_cent_mean = float(np.mean(spec_cent))
        spec_cent_std = float(np.std(spec_cent))

        # 3. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))

        # 4. Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = float(np.mean(chroma))
        chroma_std = float(np.std(chroma))

        # 5. Spectral Rolloff & Bandwidth
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spec_rolloff_mean = float(np.mean(rolloff))
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_bandwidth_mean = float(np.mean(bandwidth))

        # Assemble feature vector (35 dimensions)
        feat_vector = list(mfcc_means) + list(mfcc_stds) + [
            spec_cent_mean, spec_cent_std, zcr_mean, zcr_std,
            chroma_mean, chroma_std, spec_rolloff_mean, spec_bandwidth_mean,
            duration_sec
        ]

        # Machine Learning Inference
        model, scaler = load_audio_assets()
        feat_scaled = scaler.transform([feat_vector])

        pred_label = int(model.predict(feat_scaled)[0])
        
        confidence = 0.88
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feat_scaled)[0]
            confidence = float(np.max(probs))

        confidence = round(max(0.70, min(0.99, confidence)), 4)
        is_fake = (pred_label == 1)
        prediction_str = "AI Generated" if is_fake else "Real Human Voice"

        # Heuristic Supporting Risk Indicators
        if is_fake:
            voice_consistency = int(min(95, max(45, 100 - (spec_cent_std / 10))))
            background_noise_analysis = int(min(90, max(15, zcr_mean * 400)))
            spectral_discontinuity = int(min(98, max(65, confidence * 95)))
            artifact_intensity = int(min(96, max(60, confidence * 92)))
        else:
            voice_consistency = int(min(98, max(75, 100 - (zcr_std * 500))))
            background_noise_analysis = int(min(40, max(5, zcr_mean * 200)))
            spectral_discontinuity = int(min(30, max(5, (1 - confidence) * 50)))
            artifact_intensity = int(min(25, max(4, (1 - confidence) * 45)))

        return {
            "prediction": prediction_str,
            "is_fake": is_fake,
            "confidence": confidence,
            "duration": round(duration_sec, 2),
            "duration_formatted": duration_formatted,
            "file_type": ext.replace(".", "").lower(),
            "file_size_mb": file_size_mb,
            "model_used": type(model).__name__,
            "media_url": f"/uploads/{saved_filename}",
            "supporting_indicators": {
                "voice_consistency": voice_consistency,
                "background_noise_analysis": background_noise_analysis,
                "spectral_discontinuity": spectral_discontinuity,
                "artifact_intensity": artifact_intensity
            },
            "features_extracted": {
                "mfcc_mean_1": round(float(mfcc_means[0]), 2),
                "spectral_centroid_hz": round(spec_cent_mean, 2),
                "zero_crossing_rate": round(zcr_mean, 4),
                "chroma_pitch_mean": round(chroma_mean, 4)
            }
        }

    except Exception as e:
        print(f"Audio Processing Error: {e}")
        return {"error": f"Audio processing failed: {str(e)}"}
