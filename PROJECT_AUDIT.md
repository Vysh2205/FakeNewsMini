# PROJECT_AUDIT.md - Academic Audit of FakeBuster System

**Project Name**: FakeBuster – Real-Time News Verification System  
**Repository**: [https://github.com/Vysh2205/FakeNewsMini](https://github.com/Vysh2205/FakeNewsMini)  
**Deployment**: [https://fakenewsmini-fakebuster.surge.sh/](https://fakenewsmini-fakebuster.surge.sh/)  
**Audit Date**: October 6, 2026  

---

## 1. Executive Summary & Architecture Overview

FakeBuster is designed as a multimodal AI-assisted news verification system consisting of:
- **Frontend**: React.js (TypeScript, Vite, Tailwind CSS, Lucide Icons).
- **Backend API**: FastAPI (Python 3.10) with Uvicorn server.
- **Machine Learning Engine**: TF-IDF Vectorization paired with Multinomial Naive Bayes, Linear SVM, and Random Forest Classifiers trained using `scikit-learn`.
- **Database**: SQLite (`backend/fakenews.db`) via SQLAlchemy for history persistence and real-time analytics.
- **Multimodal & NLP Services**: Scraper engine (`newspaper3k` + `BeautifulSoup`), Multi-Tier Google/MyMemory Translation, Claim Extraction, Fact-Checking integration, and Image OCR.

---

## 2. Inconsistencies & Issues Identified During Audit

### A. Data Leakage & Artificial 100% Accuracy in ML Training
* **Identified Issue**: In `backend/ml/prepare_dataset.py`, a base set of 34 unique records was replicated 10 times to reach 340 records. During the 80/20 train/test split, identical rows were present in both the training set and the test set.
* **Impact**: Caused strict data leakage where test samples were copies of training samples, yielding artificial 100.00% accuracy metrics across all 3 classifiers.
* **Fix**: Replace replicated rows with a diverse dataset of 300+ unique, distinct, non-duplicate real and fake news records compiled from benchmark corpora (ISOT, WELFake, Reuters, AP News).

### B. TF-IDF Fit/Transform Order
* **Identified Issue**: Vectorizer fitting must occur exclusively on `X_train`, followed by `transform(X_test)` to strictly prevent feature vocabulary leakage from the test set.
* **Fix**: Enforce `vectorizer.fit_transform(X_train)` and `vectorizer.transform(X_test)`.

### C. Fallback Behavior & Error Handling
* **Identified Issue**: If an ML model failed to load, legacy code sometimes fell back to default assumptions or random estimates rather than explicit error reporting.
* **Fix**: Remove all random fallbacks. Return structured JSON error responses (`{"error": "ML model is currently unavailable"}`) and display explicit frontend notices.

### D. Model Confidence vs. Factual Truth Disclaimer
* **Identified Issue**: High classification confidence (e.g. 96%) could be misconstrued by users as a 96% guarantee of absolute factual truth.
* **Fix**: Re-label confidence clearly as "Model Classification Confidence" and attach an explicit disclaimer in the UI:
  > *"FakeBuster is an AI-based decision-support tool. Model confidence represents classification probability and does not guarantee factual truth."*

### E. URL Extraction Failures
* **Identified Issue**: On paywalled or heavily blocked sites, if extracted text was under 40 words, running ML classification produced unreliable results.
* **Fix**: When article extraction yields insufficient text (< 40 words), inform the user:
  > *"Unable to extract sufficient article content from this URL. Please paste the article text manually."*

### F. Terminology Transparency for Risk Indicators
* **Identified Issue**: Metrics like Clickbait, Emotional Language, Source Reliability, and Evidence Quality could be mistaken for separate deep neural networks.
* **Fix**: Re-label them transparently as **"Supporting Risk Indicators (Heuristic Measures)"** to ensure academic clarity.

---

## 3. Current System Components Audit

| Component | Status | Audit Findings |
| :--- | :--- | :--- |
| **React Frontend** | Functional | Clean UI. Needs updated disclaimers and URL extraction error notices. |
| **FastAPI Backend** | Functional | Handles `/api/verify/text`, `/api/verify/url`, `/api/verify/image`, `/api/history`, `/api/analytics`. |
| **ML Models** | Functional | `MultinomialNB`, `LinearSVC`, `RandomForestClassifier` trained via `joblib`. |
| **Database** | Functional | SQLite database `fakenews.db` tracks real history records and analytics. |
| **Scraper** | Functional | 3-Tier fallback (Newspaper3k -> BeautifulSoup -> Slug Metadata). Updated to handle extraction threshold checks. |

---

## 4. Recommended Fixes Action Plan

1. **Rebuild Dataset**: Generate 300+ unique, non-duplicate real and fake news records in `fake_news_dataset.csv`.
2. **Retrain ML Pipeline**: Separate fit/transform to prevent leakage. Compute empirical Accuracy, Precision, Recall, F1-Score, and Confusion Matrices on unseen test set.
3. **Persist Artifacts**: Update `best_model.pkl`, `tfidf_vectorizer.pkl`, `metrics.json`, and `confusion_matrix.png`.
4. **Update API & Frontend**: Enforce explicit error responses, confidence disclaimers, URL length validation, and Supporting Risk Indicators.
5. **Generate Academic Documentation**: Produce `RESEARCH_PAPER_VALUES.md` with un-hardcoded metrics.
6. **Re-deploy**: Build dist and deploy to Surge.sh & GitHub Pages.
