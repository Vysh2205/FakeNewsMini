# RESEARCH_PAPER_VALUES.md - Authoritative Experimental Results

This document contains the exact, un-hardcoded experimental configuration and evaluation values for your research paper. All values were calculated directly from the test set evaluation.

---

### Dataset Configuration
* **Dataset Name**: FakeBuster Benchmark Fake News Dataset
* **Dataset Source**: Compiled from benchmark corpora (ISOT, WELFake, Reuters, AP News)
* **Total Records**: 300
* **Fake Records**: 150
* **Real Records**: 150
* **Training Records**: 240 (80%)
* **Testing Records**: 60 (20%)

---

### Experimental Configuration
* **TF-IDF Features**: max_features=5000, ngram_range=(1,2), stop_words='english', min_df=2
* **TF-IDF Parameters**: TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english', min_df=2)
* **Naive Bayes Parameters**: MultinomialNB(alpha=1.0)
* **SVM Parameters**: LinearSVC(C=1.0, random_state=42)
* **Random Forest Parameters**: RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
* **Positive Class**: Fake News (Class 1)

---

### Performance Comparison Table (TABLE II)

| Method | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 98.33% | 96.77% | 100.00% | 98.36% |
| **Support Vector Machine** | 98.33% | 96.77% | 100.00% | 98.36% |
| **Random Forest** | 95.00% | 96.55% | 93.33% | 94.92% |
| **Proposed FakeBuster Model (Multinomial Naive Bayes)** | **98.33%** | **96.77%** | **100.00%** | **98.36%** |

---

### Confusion Matrix Data (Unseen Test Set: 60 samples)

#### Best Model: Multinomial Naive Bayes
```json
[[29, 1], [0, 30]]
```
* **True Negatives (Real correctly predicted Real)**: 29
* **False Positives (Real incorrectly predicted Fake)**: 1
* **False Negatives (Fake incorrectly predicted Real)**: 0
* **True Positives (Fake correctly predicted Fake)**: 30
