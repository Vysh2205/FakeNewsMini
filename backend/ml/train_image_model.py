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

def train_and_evaluate_image_model():
    ml_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(ml_dir, "dataset", "image_forensic_dataset.csv")
    models_dir = os.path.join(ml_dir, "models")
    results_dir = os.path.join(ml_dir, "results")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        from prepare_image_dataset import generate_image_dataset
        generate_image_dataset()

    print(f"[1/5] Loading Image Forensic Dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    total_samples = len(df)
    real_samples = int((y == 0).sum())
    fake_samples = int((y == 1).sum())

    print(f"Total Image Samples: {total_samples} | Real Photos (Label 0): {real_samples} | Manipulated/Deepfake (Label 1): {fake_samples}")

    # 80/20 Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[2/5] Stratified Split -> Training: {len(X_train)} samples | Testing: {len(X_test)} samples")

    # Feature Scaling (Fit on X_train only to prevent data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Classifiers
    classifiers = {
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=4, min_samples_leaf=2, random_state=42, class_weight="balanced"
        ),
        "Linear SVM": LinearSVC(C=0.8, max_iter=2000, random_state=42),
        "Gaussian Naive Bayes": GaussianNB(var_smoothing=1e-8)
    }

    all_metrics = {}
    fitted_models = {}

    print("[3/5] Training & Evaluating Image Forensic Classifiers on Unseen Test Set...")

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
            "confusion_matrix": cm,
            "tp": cm[1][1],
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0]
        }
        fitted_models[name] = clf

        print(f"  {name:26s} -> Acc: {acc*100:6.2f}% | Prec: {prec*100:6.2f}% | Rec: {rec*100:6.2f}% | F1: {f1*100:6.2f}%")

    # Select Best Model based on F1-Score
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1_score"])
    best_model = fitted_models[best_name]
    best_f1 = all_metrics[best_name]["f1_score"]

    # Compute Feature Importances
    feature_ranking = []
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        for idx in sorted_indices[:10]:
            feature_ranking.append({
                "feature": feature_cols[idx],
                "importance": round(float(importances[idx]), 4)
            })

    print(f"\n[4/5] Best Performing Image Model: '{best_name}' (F1-Score: {best_f1*100:.2f}%)")

    # Save Model, Scaler, and Metrics
    image_model_path = os.path.join(models_dir, "image_model.pkl")
    image_scaler_path = os.path.join(models_dir, "image_scaler.pkl")
    image_metrics_path = os.path.join(results_dir, "image_metrics.json")

    joblib.dump(best_model, image_model_path)
    joblib.dump(scaler, image_scaler_path)

    image_info = {
        "dataset_name": "FakeBuster Image Forensic Benchmark Dataset",
        "dataset_source": "Compiled from ELA Compression, Color Covariance & Frequency Spectrum Residuals",
        "total_samples": total_samples,
        "real_samples": real_samples,
        "fake_samples": fake_samples,
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "feature_count": len(feature_cols),
        "label_mapping": {
            "0": "REAL (Genuine Original Photo)",
            "1": "FAKE (Manipulated / AI Deepfake Image)"
        },
        "top_features": feature_ranking,
        "best_model": best_name,
        "best_f1_score": round(best_f1, 4),
        "all_metrics": all_metrics
    }

    with open(image_metrics_path, "w") as f:
        json.dump(image_info, f, indent=2)

    # Generate IMAGE_MODEL_INFO.md
    generate_image_documentation(image_info)

    print("==================================================")
    print("  Image Forensic ML Model Training Complete!      ")
    print("==================================================")

def generate_image_documentation(info):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    doc_path = os.path.join(root_dir, "IMAGE_MODEL_INFO.md")

    m = info["all_metrics"]
    best_name = info["best_model"]
    top_feats = info.get("top_features", [])

    feat_table_rows = ""
    for f in top_feats:
        feat_table_rows += f"| `{f['feature']}` | {f['importance']*100:.2f}% |\n"

    content = f"""# IMAGE_MODEL_INFO.md - Image Deepfake Verification Model Documentation

This document contains the academic documentation, label mappings, and benchmark metrics for the **FakeBuster Image Forensic Verification Engine**.

---

## 1. Dataset & Label Mapping

* **Dataset Name**: {info['dataset_name']}
* **Total Image Samples**: {info['total_samples']}
* **Real Photos (Label 0)**: {info['real_samples']}
* **Manipulated / AI Deepfakes (Label 1)**: {info['fake_samples']}
* **Training Split**: {info['training_samples']} samples (80% Stratified Split)
* **Testing Split**: {info['testing_samples']} samples (20% Unseen Test Split)

### Label Encoding Mapping:
* **`0`** = **REAL** (Genuine Original Camera Photo)
* **`1`** = **FAKE** (Manipulated / AI-Generated Deepfake Image)

---

## 2. Image Forensic Feature Extraction

The pipeline extracts **{info['feature_count']} image forensic metrics** per uploaded image file:

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
{feat_table_rows}

---

## 4. Classifier Performance Comparison (Unseen Test Set: {info['testing_samples']} samples)

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | True Pos | True Neg | False Pos | False Neg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | {m['Random Forest Classifier']['accuracy']*100:.2f}% | {m['Random Forest Classifier']['precision']*100:.2f}% | {m['Random Forest Classifier']['recall']*100:.2f}% | **{m['Random Forest Classifier']['f1_score']*100:.2f}%** | {m['Random Forest Classifier']['tp']} | {m['Random Forest Classifier']['tn']} | {m['Random Forest Classifier']['fp']} | {m['Random Forest Classifier']['fn']} |
| **Linear SVM** | {m['Linear SVM']['accuracy']*100:.2f}% | {m['Linear SVM']['precision']*100:.2f}% | {m['Linear SVM']['recall']*100:.2f}% | {m['Linear SVM']['f1_score']*100:.2f}% | {m['Linear SVM']['tp']} | {m['Linear SVM']['tn']} | {m['Linear SVM']['fp']} | {m['Linear SVM']['fn']} |
| **Gaussian Naive Bayes** | {m['Gaussian Naive Bayes']['accuracy']*100:.2f}% | {m['Gaussian Naive Bayes']['precision']*100:.2f}% | {m['Gaussian Naive Bayes']['recall']*100:.2f}% | {m['Gaussian Naive Bayes']['f1_score']*100:.2f}% | {m['Gaussian Naive Bayes']['tp']} | {m['Gaussian Naive Bayes']['tn']} | {m['Gaussian Naive Bayes']['fp']} | {m['Gaussian Naive Bayes']['fn']} |

---

## 5. Model Artifacts

* Trained Model: `backend/ml/models/image_model.pkl`
* Feature Scaler: `backend/ml/models/image_scaler.pkl`
* Results JSON: `backend/ml/results/image_metrics.json`
"""

    with open(doc_path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    train_and_evaluate_image_model()
