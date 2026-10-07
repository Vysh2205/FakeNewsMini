# AUDIO_MODEL_INFO.md - Audio Deepfake Verification Model Documentation

This document contains the academic documentation, tuned hyper-parameters, and feature weights for the **FakeBuster Audio Deepfake Verification Engine**.

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

## 2. Model Parameters & Hyperparameter Tuning

The audio verification pipeline employs a **Tuned Random Forest Ensemble** with balanced class weighting:

* **Number of Trees (`n_estimators`)**: `200`
* **Maximum Tree Depth (`max_depth`)**: `12`
* **Minimum Split Samples (`min_samples_split`)**: `4`
* **Class Weighting (`class_weight`)**: `"balanced"`
* **Feature Normalization**: Fitted `StandardScaler` (applied strictly to training split)

---

## 3. Acoustic Feature Importances & Weights

Top 10 acoustic feature weights calculated by Random Forest mean decrease in impurity (MDI):

| Feature Name | Feature Importance Weight (%) |
| :--- | :---: |
| `mfcc_mean_1` | 20.55% |
| `spec_centroid_std` | 14.45% |
| `mfcc_std_1` | 11.90% |
| `mfcc_std_2` | 8.52% |
| `mfcc_std_5` | 6.51% |
| `mfcc_std_8` | 5.65% |
| `mfcc_std_7` | 4.94% |
| `chroma_std` | 4.32% |
| `mfcc_std_4` | 4.02% |
| `mfcc_std_10` | 4.02% |


---

## 4. Classifier Performance Comparison (Unseen Test Set: 60 samples)

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | 100.00% | 100.00% | 100.00% | **100.00%** |
| **Linear SVM** | 100.00% | 100.00% | 100.00% | 100.00% |
| **Gaussian Naive Bayes** | 100.00% | 100.00% | 100.00% | 100.00% |

---

## 5. Best Model & Saved Artifacts

* **Selected Best Model**: **Random Forest Classifier**
* **Selection Metric**: Highest Test F1-Score (**100.00%**)
* **Model Artifacts**:
  - Trained Model: `backend/ml/models/audio_model.pkl`
  - Feature Scaler: `backend/ml/models/audio_scaler.pkl`
  - Results JSON: `backend/ml/results/audio_metrics.json`
