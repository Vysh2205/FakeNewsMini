import os
import re
import json
import joblib
import numpy as np

_model = None
_vectorizer = None
_model_info = None

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_ml_assets():
    global _model, _vectorizer, _model_info
    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer, _model_info

    ml_dir = os.path.dirname(__file__)
    best_model_path = os.path.join(ml_dir, "models", "best_model.pkl")
    vectorizer_path = os.path.join(ml_dir, "models", "tfidf_vectorizer.pkl")
    model_info_path = os.path.join(ml_dir, "models", "model_info.json")

    if not os.path.exists(best_model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Trained ML model or vectorizer file not found. Run train_model.py first.")

    _model = joblib.load(best_model_path)
    _vectorizer = joblib.load(vectorizer_path)

    if os.path.exists(model_info_path):
        with open(model_info_path, "r") as f:
            _model_info = json.load(f)
    else:
        _model_info = {"best_model": type(_model).__name__, "all_metrics": {}}

    return _model, _vectorizer, _model_info

def predict_fake_news(text: str):
    """
    Performs real-time fake news inference using the trained TF-IDF + Classifier model.
    """
    try:
        model, vectorizer, info = load_ml_assets()
        processed = clean_text(text)
        
        if not processed:
            return {
                "prediction": "Uncertain",
                "is_fake": False,
                "confidence": 0.50,
                "model_used": info.get("best_model", "Trained ML Model"),
                "metrics": info.get("all_metrics", {})
            }

        # Vectorize text using saved TF-IDF vectorizer
        tfidf_feat = vectorizer.transform([processed])
        pred_label = int(model.predict(tfidf_feat)[0])

        # Confidence calculation
        confidence = 0.85
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(tfidf_feat)[0]
            confidence = float(np.max(probs)) if 'np' in globals() else float(max(probs))
        elif hasattr(model, "decision_function"):
            df_val = float(model.decision_function(tfidf_feat)[0])
            confidence = round(1.0 / (1.0 + float(np.exp(-abs(df_val)))) if 'np' in globals() else 0.88, 4)

        confidence = round(max(0.65, min(0.99, confidence)), 4)
        is_fake = (pred_label == 1)
        prediction_str = "Fake" if is_fake else "Real"
        model_name = info.get("best_model", type(model).__name__)

        return {
            "prediction": prediction_str,
            "is_fake": is_fake,
            "confidence": confidence,
            "model_used": model_name,
            "all_metrics": info.get("all_metrics", {})
        }
    except Exception as e:
        print(f"ML Prediction Error: {e}")
        return {
            "error": f"ML Prediction Failed: {str(e)}",
            "prediction": "Uncertain",
            "is_fake": False,
            "confidence": 0.50,
            "model_used": "Fallback Classifier"
        }
