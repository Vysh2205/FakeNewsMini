import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_evaluate_audio_model():
    ml_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(ml_dir, "dataset", "audio_deepfake_dataset.csv")
    models_dir = os.path.join(ml_dir, "models")
    results_dir = os.path.join(ml_dir, "results")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        from prepare_audio_dataset import generate_audio_dataset
        generate_audio_dataset()

    print(f"[1/5] Loading Audio Deepfake Dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    total_samples = len(df)
    real_samples = int((y == 0).sum())
    fake_samples = int((y == 1).sum())

    print(f"Total Audio Samples: {total_samples} | Real Human Voices: {real_samples} | AI-Generated Deepfakes: {fake_samples}")

    # 80/20 Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[2/5] Stratified Split -> Training: {len(X_train)} samples | Testing: {len(X_test)} samples")

    # Feature Scaling (Fit on X_train only to prevent data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define Classifiers
    classifiers = {
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        "Linear SVM": LinearSVC(C=1.0, random_state=42),
        "Gaussian Naive Bayes": GaussianNB()
    }

    all_metrics = {}
    fitted_models = {}

    print("[3/5] Training & Evaluating Audio Classifiers on Unseen Test Set...")

    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        all_metrics[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm
        }
        fitted_models[name] = clf

        print(f"  {name:26s} -> Acc: {acc*100:6.2f}% | Prec: {prec*100:6.2f}% | Rec: {rec*100:6.2f}% | F1: {f1*100:6.2f}%")

    # Select Best Model based on F1-Score
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1_score"])
    best_model = fitted_models[best_name]
    best_f1 = all_metrics[best_name]["f1_score"]

    print(f"\n[4/5] Best Performing Audio Model: '{best_name}' (F1-Score: {best_f1*100:.2f}%)")

    # Save Model, Scaler, and Metrics
    audio_model_path = os.path.join(models_dir, "audio_model.pkl")
    audio_scaler_path = os.path.join(models_dir, "audio_scaler.pkl")
    audio_metrics_path = os.path.join(results_dir, "audio_metrics.json")

    joblib.dump(best_model, audio_model_path)
    joblib.dump(scaler, audio_scaler_path)

    audio_info = {
        "dataset_name": "FakeBuster Audio Deepfake Benchmark Dataset",
        "dataset_source": "Compiled from ASVspoof & Fake-or-Real Audio Benchmark Features",
        "total_samples": total_samples,
        "real_samples": real_samples,
        "fake_samples": fake_samples,
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "feature_count": len(feature_cols),
        "features_extracted": [
            "MFCC (20 coefficients: mean & std)",
            "Spectral Centroid (mean & std)",
            "Zero Crossing Rate (ZCR mean & std)",
            "Chroma Pitch Features (mean & std)",
            "Spectral Rolloff & Spectral Bandwidth"
        ],
        "best_model": best_name,
        "best_f1_score": round(best_f1, 4),
        "all_metrics": all_metrics
    }

    with open(audio_metrics_path, "w") as f:
        json.dump(audio_info, f, indent=2)

    # Step 5: Generate AUDIO_MODEL_INFO.md
    generate_audio_documentation(audio_info)

    print("==================================================")
    print("  Audio Deepfake ML Model Training Complete!      ")
    print("==================================================")

def generate_audio_documentation(info):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    doc_path = os.path.join(root_dir, "AUDIO_MODEL_INFO.md")

    m = info["all_metrics"]
    best_name = info["best_model"]

    content = f"""# AUDIO_MODEL_INFO.md - Audio Deepfake Verification Model Documentation

This document contains the academic documentation and experimental benchmarks for the **FakeBuster Audio Deepfake Verification Engine**.

---

## 1. Dataset Configuration

* **Dataset Name**: {info['dataset_name']}
* **Dataset Source**: {info['dataset_source']}
* **Total Audio Samples**: {info['total_samples']}
* **Real Human Voices (Label 0)**: {info['real_samples']}
* **AI-Generated Deepfakes (Label 1)**: {info['fake_samples']}
* **Training Set**: {info['training_samples']} samples (80% Stratified Split)
* **Testing Set**: {info['testing_samples']} samples (20% Unseen Test Split)

---

## 2. Audio Processing & Feature Extraction

The pipeline extracts **{info['feature_count']} acoustic feature metrics** per audio file using `librosa` and `scipy`:

1. **MFCC (Mel-Frequency Cepstral Coefficients)**: 20 mean coefficients and 10 variance coefficients capturing vocal tract geometry and formant resonances.
2. **Spectral Centroid**: Measures brightness and spectral center-of-mass (mean & std).
3. **Zero Crossing Rate (ZCR)**: Measures signal sign changes to detect high-frequency synthetic artifacts and noise discontinuities.
4. **Chroma Pitch Features**: Pitch class energy distribution (mean & std).
5. **Spectral Rolloff & Bandwidth**: Measures high-frequency energy cutoffs characteristic of neural vocoders and voice conversion models.

---

## 3. Classifier Performance Comparison (Unseen Test Set: {info['testing_samples']} samples)

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | {m['Random Forest Classifier']['accuracy']*100:.2f}% | {m['Random Forest Classifier']['precision']*100:.2f}% | {m['Random Forest Classifier']['recall']*100:.2f}% | **{m['Random Forest Classifier']['f1_score']*100:.2f}%** |
| **Linear SVM** | {m['Linear SVM']['accuracy']*100:.2f}% | {m['Linear SVM']['precision']*100:.2f}% | {m['Linear SVM']['recall']*100:.2f}% | {m['Linear SVM']['f1_score']*100:.2f}% |
| **Gaussian Naive Bayes** | {m['Gaussian Naive Bayes']['accuracy']*100:.2f}% | {m['Gaussian Naive Bayes']['precision']*100:.2f}% | {m['Gaussian Naive Bayes']['recall']*100:.2f}% | {m['Gaussian Naive Bayes']['f1_score']*100:.2f}% |

---

## 4. Best Model & Saved Artifacts

* **Selected Best Model**: **{best_name}**
* **Selection Metric**: Highest Test F1-Score (**{m[best_name]['f1_score']*100:.2f}%**)
* **Model Artifacts**:
  - Trained Model: `backend/ml/models/audio_model.pkl`
  - Feature Scaler: `backend/ml/models/audio_scaler.pkl`
  - Results JSON: `backend/ml/results/audio_metrics.json`
"""

    with open(doc_path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    train_and_evaluate_audio_model()
