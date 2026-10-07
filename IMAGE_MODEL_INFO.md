# IMAGE_MODEL_INFO.md - Image Deepfake Verification Model Documentation

This document contains the academic documentation, label mappings, and benchmark metrics for the **FakeBuster Image Forensic Verification Engine**.

---

## 1. Dataset & Label Mapping

* **Dataset Name**: FakeBuster Image Forensic Benchmark Dataset
* **Total Image Samples**: 300
* **Real Photos (Label 0)**: 150
* **Manipulated / AI Deepfakes (Label 1)**: 150
* **Training Split**: 240 samples (80% Stratified Split)
* **Testing Split**: 60 samples (20% Unseen Test Split)

### Label Encoding Mapping:
* **`0`** = **REAL** (Genuine Original Camera Photo)
* **`1`** = **FAKE** (Manipulated / AI-Generated Deepfake Image)

---

## 2. Image Forensic Feature Extraction

The pipeline extracts **19 image forensic metrics** per uploaded image file:

1. **Error Level Analysis (ELA)**: Re-compresses image at 90% JPEG quality to measure localized compression artifacts (`ela_mean`, `ela_std`, `ela_max`).
2. **RGB Color Channel Statistics**: Measures channel means, standard deviations, and cross-channel covariance matrices (`rg_cov`, `gb_cov`, `rb_cov`).
3. **High-Frequency Noise Residual**: Laplacian variance (`laplacian_var`) to measure sensor blur and synthetic smoothing.
4. **Frequency Domain Energy**: FFT high-frequency noise ratio (`fft_high_freq_energy`) to detect checkerboard artifacts from generative neural networks.
5. **EXIF Metadata Tags**: Preserved camera sensor tags (`exif_present`, `dimensions`, `format`).

---

## 3. Acoustic & Visual Feature Importances

Top 10 image feature weights calculated by Random Forest MDI:

| Feature Name | Feature Importance Weight (%) |
| :--- | :---: |
| `rg_cov` | 18.70% |
| `rb_cov` | 16.15% |
| `gb_cov` | 12.64% |
| `ela_mean` | 11.04% |
| `ela_max` | 9.65% |
| `width` | 9.40% |
| `noise_residual_std` | 7.59% |
| `fft_high_freq_energy` | 5.45% |
| `laplacian_var` | 3.40% |
| `ela_std` | 2.98% |


---

## 4. Classifier Performance Comparison (Unseen Test Set: 60 samples)

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | True Pos | True Neg | False Pos | False Neg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | 100.00% | 100.00% | 100.00% | **100.00%** | 30 | 30 | 0 | 0 |
| **Linear SVM** | 100.00% | 100.00% | 100.00% | 100.00% | 30 | 30 | 0 | 0 |
| **Gaussian Naive Bayes** | 100.00% | 100.00% | 100.00% | 100.00% | 30 | 30 | 0 | 0 |

---

## 5. Model Artifacts

* Trained Model: `backend/ml/models/image_model.pkl`
* Feature Scaler: `backend/ml/models/image_scaler.pkl`
* Results JSON: `backend/ml/results/image_metrics.json`
