# ❤️ Heart Disease Prediction Using Machine Learning

> Predicting the presence of heart disease from clinical patient data using 
> Logistic Regression, Random Forest, and SVM — with full EDA, feature analysis, 
> and model comparison.

---

## 📌 Project Overview

Heart disease is the leading cause of death globally. Early prediction using patient 
clinical data can significantly improve outcomes. This project builds and compares 
three machine learning models to predict heart disease presence with high accuracy.

### Key Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~85% | ~0.91 |
| **Random Forest** | **~88%** | **~0.94** |
| SVM | ~87% | ~0.93 |

> 🏆 **Random Forest** achieved the best performance across all metrics.

---

## 📊 Dataset

- **Source:** UCI Heart Disease Dataset via [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Samples:** 303 patients
- **Features:** 13 clinical attributes
- **Target:** 0 = No Disease, 1 = Disease

### Features Used

| Feature | Description |
|---------|-------------|
| age | Patient age in years |
| sex | Gender (1=Male, 0=Female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels coloured by fluoroscopy |
| thal | Thalassemia type |

---

## 🔬 Methodology

1. **Data Loading & Inspection** — check shape, types, missing values
2. **Exploratory Data Analysis (EDA)** — distribution plots, correlation heatmap
3. **Preprocessing** — train/test split (80/20), StandardScaler normalisation
4. **Model Training** — Logistic Regression, Random Forest, SVM
5. **Evaluation** — Accuracy, ROC-AUC, Cross-validation, Confusion Matrix
6. **Visualisation** — ROC curves, feature importance, model comparison

---

## 📈 Visualisations

- Target class distribution
- Age distribution by disease status
- Feature correlation heatmap
- Model performance comparison (bar chart)
- Confusion matrix (best model)
- ROC curves for all models
- Feature importance (Random Forest)

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/lekshmi45/heart-disease-prediction
cd heart-disease-prediction

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# 3. Download dataset from Kaggle and place heart.csv in project folder

# 4. Create plots folder
mkdir plots

# 5. Run the project
python heart_disease_prediction.py
```

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── heart_disease_prediction.py   # Main ML pipeline
├── heart.csv                     # Dataset (download from Kaggle)
├── plots/                        # Generated visualisations
│   ├── 01_target_distribution.png
│   ├── 02_age_distribution.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_model_comparison.png
│   ├── 05_confusion_matrix.png
│   ├── 06_roc_curves.png
│   └── 07_feature_importance.png
└── README.md
```

---

## 💡 Key Findings

- **Random Forest** outperforms linear models, confirming nonlinear relationships in clinical data
- **Chest pain type (cp)** and **maximum heart rate (thalach)** are the strongest predictors
- **Age alone is not the best predictor** — combination of features matters most
- Model generalises well with consistent cross-validation scores

---

## 👩‍💻 Author

**Lekshmi S.S** — ML Engineer | EEG Signal Processing | Biomedical AI  
  
🔗 [LinkedIn](https://www.linkedin.com/in/lekshmi-s-s-506701216)  
🐙 [GitHub](https://github.com/lekshmi45)
