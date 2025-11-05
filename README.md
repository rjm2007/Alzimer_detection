# 🧠 Early Alzheimer’s Detection from Speech Patterns (NLP + ML)

### 🔍 Project Overview
This project predicts early signs of **Alzheimer’s disease** based on **speech patterns** — using both **linguistic** (text-based) and **acoustic** (MFCC) features.  
The system can take **user-typed input** like:  
> "Hi... umm... ho ar u..."  
and detect disfluency or irregular language patterns that may indicate cognitive decline.

Built with **Python, scikit-learn, XGBoost, LightGBM**, and **ensemble learning techniques**, this end-to-end workflow covers data preprocessing, model training, evaluation, and real-time prediction.

---

## ⚙️ Project Pipeline

| Step | Notebook | Description |
|------|-----------|--------------|
| 1️⃣ | `01_Preprocessing.ipynb` | Data cleaning, feature engineering, PCA for linguistic features, MFCC stats |
| 2️⃣ | `02_Model_Training.ipynb` | Ensemble + stacking model training and comparison |
| 3️⃣ | `03_Evaluation_and_Prediction.ipynb` | Model evaluation and user text-based Alzheimer’s prediction |

---

## 📘 1. Dataset Description

| Feature Group | Description |
|----------------|-------------|
| **duration_sec**, **chunk_count** | Basic speech metrics (length, segmentation) |
| **mfcc_1 – mfcc_13** | Acoustic features (Mel-Frequency Cepstral Coefficients) |
| **linguistic_feat_1 – linguistic_feat_50** | Text-based linguistic embeddings |
| **label** | Target variable — 0 = Healthy, 1 = Alzheimer’s |

📂 Original dataset: `data/addetector_dataset.csv`  
📂 Cleaned dataset after preprocessing: `data/cleaned_addetector_dataset.csv`

---

## 🧹 2. Preprocessing Highlights (`01_Preprocessing.ipynb`)

### 🧩 Key Steps:
- Removed nulls & duplicates  
- Scaled features using **StandardScaler**  
- **Linguistic Features → PCA (Top 10 components)** to preserve semantic richness  
- **MFCC Features → Mean, Std, Var** to capture tone dynamics  
- Train-test split (80–20 stratified)  

### 🧾 Output:
- Clean, reduced dataset → `data/cleaned_addetector_dataset.csv`  
- Feature count reduced from 66 → 18 (optimized for interpretability)  

---

## 🤖 3. Model Training (`02_Model_Training.ipynb`)

### 🧠 Models Used:
| Type | Model | Purpose |
|------|--------|----------|
| Base | Logistic Regression | Lightweight baseline |
| Base | Random Forest | Robust, interpretable ensemble |
| Base | XGBoost | Gradient-boosted high performer |
| Base | LightGBM | Efficient gradient boosting |
| Ensemble | Voting Classifier | Averages model probabilities |
| Ensemble | Stacking Classifier | Meta-learner improves final accuracy |

### ⚙️ Training Setup
- Used **class_weight='balanced'** to handle class imbalance  
- Evaluated models with: Accuracy, Precision, Recall, F1, ROC-AUC  
- Saved **best model automatically** to `models/<best_model>_best.pkl`  

### 📊 Example Results

| Model | Accuracy | F1 Score | ROC-AUC |
|--------|-----------|-----------|----------|
| Logistic Regression | 0.68 | 0.60 | 0.71 |
| Random Forest | 0.73 | 0.65 | 0.77 |
| XGBoost | 0.76 | 0.70 | 0.80 |
| LightGBM | 0.77 | 0.72 | 0.81 |
| **Voting Ensemble** | 0.80 | 0.74 | 0.84 |
| **Stacking Ensemble** | **0.83** | **0.78** | **0.88** |

🧾 Final Model: `models/Stacking_Ensemble_best.pkl`  
📈 Metrics: `results/metrics.json`

---

## 💬 4. Prediction & Evaluation (`03_Evaluation_and_Prediction.ipynb`)

### ✍️ Live Text Prediction
The user can input text such as:
```
Hi... um... I forget what I was saying...
```
The system extracts simplified linguistic signals and predicts:
```
🧠 Alzheimer’s Detected
Confidence: 0.82
```

### 🧩 Linguistic Cues Extracted:
- Word count  
- Unique word ratio  
- Pause count (“...”)  
- Average word length  
- Readability score  
- Derived linguistic PCA embeddings  

### 📊 Model Evaluation
- Confusion Matrix visualization  
- ROC Curve & AUC  
- Performance barplots (Accuracy, F1, Recall, Precision)  

---

## 🧩 5. Explainability (Optional)
Use **SHAP** to explain model behavior and feature importance:
```python
import shap
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
shap.summary_plot(shap_values, X)
```

Helps visualize which linguistic or acoustic cues most influence Alzheimer’s detection.

---

## 📦 6. Folder Structure

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
│   └── Stacking_Ensemble_best.pkl
│
├── results/
│   ├── metrics.json
│   └── confusion_matrix.png
│
└── README.md
```

---

## 🧠 7. Key Learnings
✅ PCA preserved linguistic expressiveness  
✅ MFCC statistics captured subtle tone variations  
✅ Stacking ensembles boosted F1 and recall performance  
✅ Text-based simulation provided a deployable prototype for real-world scenarios  

---

## 🧩 8. Future Improvements
- Integrate **real audio preprocessing** (using `librosa`)  
- Add **speech-to-text (ASR)** pipeline (Google, Whisper, or Vosk)  
- Enhance **linguistic feature extraction** using transformer models (BERT-based embeddings)  
- Deploy app publicly on **Hugging Face Spaces** or a lightweight web framework  

---

## 👨‍💻 Author
- **Developer:** [Your Name]  
- **Tools Used:** Python, scikit-learn, XGBoost, LightGBM, SHAP  

---

### 🎯 Final Output Example

| Input Text | Prediction | Confidence |
|-------------|-------------|-------------|
| Hi how are you today? | ✅ Healthy Speech Pattern | 0.15 |
| Hi... umm... ho ar u... hmm I forget | 🧠 Alzheimer’s Detected | 0.82 |

---

**✅ Final Deliverables:**
- Trained ensemble Alzheimer’s classifier  
- Text-based predictor  
- Documentation & notebooks ready for submission or demo  

---

 
