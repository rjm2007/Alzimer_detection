# 🧠 Early Alzheimer’s Detection from Speech Patterns (NLP + ML)

### Project Overview
This project aims to **detect early signs of Alzheimer’s disease** using **speech and linguistic patterns**.  
We utilize the `addetector_dataset.csv` dataset containing **1010 samples** and **66 extracted features** — a mix of **acoustic (MFCCs)** and **linguistic embeddings**.

The final system can even take a **user-typed text input**, extract linguistic cues (like disfluency, coherence, and length), and **predict Alzheimer’s likelihood**.

---

## 📘 1. Dataset Description

| Feature Group | Description |
|----------------|-------------|
| **duration_sec**, **chunk_count** | Speech length and fragmentation count |
| **mfcc_1 – mfcc_13** | Acoustic speech features capturing tone and frequency |
| **linguistic_feat_1 – linguistic_feat_50** | Linguistic embeddings / textual statistics |
| **label** | Target variable — `0 = Healthy`, `1 = Alzheimer’s` |

---

## 🧹 2. Clean Dataset

After preprocessing, we generate:

> `cleaned_addetector_dataset.csv`

This cleaned dataset will:
- Remove redundant or unnecessary columns  
- Scale features using StandardScaler  
- Handle missing/null values  
- Optionally reduce linguistic dimensions using **mean or PCA aggregation**

---

## ⚙️ 3. Workflow Overview

### **Notebook-Based Pipeline**
All tasks will be implemented within `.ipynb` notebooks for easier presentation and visualization.

**Notebooks:**
1. `01_Preprocessing.ipynb` — Cleaning, feature reduction, and dataset preparation  
2. `02_Model_Training.ipynb` — Ensemble + stacking models with hyperparameter tuning  
3. `03_Evaluation_and_Prediction.ipynb` — Final metrics, confusion matrix, and user text-based prediction  

---

## 🧩 4. Feature Reduction Strategy

Since `linguistic_feat_1 – linguistic_feat_50` are highly correlated embeddings:

### **Approach 1 — Mean Aggregation**
Compute one feature:
```python
df["linguistic_mean"] = df[[f"linguistic_feat_{i}" for i in range(1, 51)]].mean(axis=1)
```
✅ Retains overall linguistic signal while reducing complexity.  
✅ Works best for smaller datasets (like ours).

### **Approach 2 — PCA (Principal Component Analysis)**
Extract top components explaining 95% variance:
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
reduced_feats = pca.fit_transform(df[linguistic_features])
```
✅ Keeps most information, reduces redundancy, ideal for model interpretability.

---

## 🧠 5. Model Selection

We’ll use **stacking and ensemble-based learning** for robustness.

| Model | Purpose |
|--------|----------|
| **Random Forest** | Handles nonlinear relations and feature importance |
| **XGBoost / LightGBM** | High performance with small datasets |
| **Logistic Regression** | Lightweight, interpretable baseline |
| **StackingClassifier** | Combines all above for the best F1 and ROC-AUC |

---

## 🧪 6. Hyperparameter Optimization

- Use **Optuna** or **GridSearchCV**
- Parameters tuned:
  - `max_depth`, `n_estimators`, `learning_rate` for XGBoost/LightGBM
  - `C`, `penalty` for Logistic Regression
  - `max_features`, `min_samples_split` for Random Forest

---

## 📈 7. Model Training and Evaluation

Metrics:
- Accuracy  
- Precision / Recall / F1-score  
- ROC-AUC  
- Confusion Matrix  

Feature importance visualization will be done via **SHAP values** and **permutation importance**.

---

## ✍️ 8. User Text-Based Prediction (Simulated Input)

Instead of real audio, users can **type a sentence**, which will be converted into a simplified **linguistic feature vector** using NLP preprocessing.

### Example Flow:
```python
Enter text: hi... ho ar u...

Predicted Output → Alzheimer's Detected
Confidence → 0.89
```

### How It Works:
1. The text is analyzed using NLP:
   - Sentence length
   - Pauses (“...” count)
   - Word diversity
   - Grammatical completeness
   - Average word length
2. These linguistic patterns are transformed into a numerical feature vector.
3. The trained ensemble model predicts whether the text reflects **healthy or Alzheimer-like** linguistic patterns.

---

## 📂 9. Folder Structure

```
AlzheimerSpeechDetection/
│
├── data/
│   ├── addetector_dataset.csv
│   └── cleaned_addetector_dataset.csv
│
├── notebooks/
│   ├── 01_Preprocessing.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_Evaluation_and_Prediction.ipynb
│
├── models/
│   └── final_model.pkl
│
├── results/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
└── README.md
```

---

## 🧰 10. Tech Stack

- **Language:** Python 3.10+
- **Libraries:**
  - pandas, numpy, scikit-learn  
  - xgboost, lightgbm, optuna  
  - shap, matplotlib, seaborn  
  - nltk, textstat (for text-based user input)

---

## 🎯 11. Final Output

- A **stacked ensemble classifier** that predicts Alzheimer’s vs Healthy speech patterns.  
- A **text-input prediction cell** allowing real-time evaluation.  
- Clean, reproducible `.ipynb` notebooks for presentation and model interpretation.

---

## 🚀 12. Future Enhancements

- Extend to real audio input via automatic speech recognition (ASR).  
- Fine-tune transformer models (BERT-based linguistic embedding).  
- Build a multi-class cognitive detection model (Normal / MCI / Alzheimer’s).  
