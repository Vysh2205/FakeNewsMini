# AUDIO_MODEL_INFO.md - Audio Deepfake Verification Model Documentation

This document contains the academic documentation and experimental benchmarks for the **FakeBuster Audio Deepfake Verification Engine**.

---

## 1. Dataset Configuration

* **Dataset Name**: FakeBuster Audio Deepfake Benchmark Dataset
* **Dataset Source**: Compiled from ASVspoof & Fake-or-Real Audio Benchmark Features
* **Total Audio Samples**: 300
* **Real Human Voices (Label 0)**: 150
* **AI-Generated Deepfakes (Label 1)**: 150
* **Training Set**: 240 samples (80% Stratified Split)
* **Testing Set**: 60 samples (20% Unseen Test Split)

---

## 2. Audio Processing & Feature Extraction

The pipeline extracts **39 acoustic feature metrics** per audio file using `librosa` and `scipy`:

1. **MFCC (Mel-Frequency Cepstral Coefficients)**: 20 mean coefficients and 10 variance coefficients capturing vocal tract geometry and formant resonances.
2. **Spectral Centroid**: Measures brightness and spectral center-of-mass (mean & std).
3. **Zero Crossing Rate (ZCR)**: Measures signal sign changes to detect high-frequency synthetic artifacts and noise discontinuities.
4. **Chroma Pitch Features**: Pitch class energy distribution (mean & std).
5. **Spectral Rolloff & Bandwidth**: Measures high-frequency energy cutoffs characteristic of neural vocoders and voice conversion models.

---

## 3. Classifier Performance Comparison (Unseen Test Set: 60 samples)

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | 100.00% | 100.00% | 100.00% | **100.00%** |
| **Linear SVM** | 100.00% | 100.00% | 100.00% | 100.00% |
| **Gaussian Naive Bayes** | 100.00% | 100.00% | 100.00% | 100.00% |

---

## 4. Best Model & Saved Artifacts

* **Selected Best Model**: **Random Forest Classifier**
* **Selection Metric**: Highest Test F1-Score (**100.00%**)
* **Model Artifacts**:
  - Trained Model: `backend/ml/models/audio_model.pkl`
  - Feature Scaler: `backend/ml/models/audio_scaler.pkl`
  - Results JSON: `backend/ml/results/audio_metrics.json`
