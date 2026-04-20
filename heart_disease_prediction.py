# ❤️ Heart Disease Prediction Using Machine Learning
# Author: Lekshmi S.S | github.com/lekshmi45
# Dataset: UCI Heart Disease Dataset (Kaggle)

# ============================================================
# STEP 1 — IMPORT LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries loaded successfully")

# ============================================================
# STEP 2 — LOAD DATA
# ============================================================
df = pd.read_csv('heart.csv')

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# ============================================================
# STEP 3 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# 3a. Target Distribution
plt.figure(figsize=(6,4))
df['target'].value_counts().plot(kind='bar', color=['#2ecc71','#e74c3c'])
plt.title('Heart Disease Distribution\n0 = No Disease | 1 = Disease')
plt.xlabel('Condition')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/01_target_distribution.png', dpi=150)
plt.show()
print("✅ Plot saved: target distribution")

# 3b. Age Distribution by Disease
plt.figure(figsize=(8,4))
df[df['target']==1]['age'].hist(alpha=0.7, label='Disease', color='#e74c3c', bins=20)
df[df['target']==0]['age'].hist(alpha=0.7, label='No Disease', color='#2ecc71', bins=20)
plt.title('Age Distribution by Heart Disease')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('plots/02_age_distribution.png', dpi=150)
plt.show()

# 3c. Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('plots/03_correlation_heatmap.png', dpi=150)
plt.show()
print("✅ EDA plots saved")

# 3d. Key statistics
print("\n📊 Key Statistics:")
print(f"Average age of patients: {df['age'].mean():.1f}")
print(f"Heart disease prevalence: {df['target'].mean()*100:.1f}%")
print(f"Male patients: {(df['sex']==1).sum()} | Female: {(df['sex']==0).sum()}")

# ============================================================
# STEP 4 — FEATURE ENGINEERING & PREPROCESSING
# ============================================================

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature columns: {list(X.columns)}")

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ============================================================
# STEP 5 — MODEL BUILDING
# ============================================================

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

results = {}

print("\n🤖 Training Models...\n")
for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:,1]
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    results[name] = {
        'model': model,
        'accuracy': acc,
        'auc': auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    print(f"{'='*40}")
    print(f"Model: {name}")
    print(f"  Accuracy:     {acc*100:.2f}%")
    print(f"  ROC-AUC:      {auc:.4f}")
    print(f"  CV Score:     {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ============================================================
# STEP 6 — EVALUATION & VISUALISATION
# ============================================================

# 6a. Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(8,5))
model_names = list(results.keys())
accuracies = [results[m]['accuracy']*100 for m in model_names]
aucs = [results[m]['auc'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35
bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#3498db')
bars2 = ax.bar(x + width/2, [a*100 for a in aucs], width, label='AUC Score (%)', color='#e74c3c')

ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylabel('Score (%)')
ax.set_ylim(0, 110)
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('plots/04_model_comparison.png', dpi=150)
plt.show()

# 6b. Best model confusion matrix
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best = results[best_model_name]

plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title(f'Confusion Matrix — {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plots/05_confusion_matrix.png', dpi=150)
plt.show()

# 6c. ROC Curves
plt.figure(figsize=(8,6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", linewidth=2)

plt.plot([0,1],[0,1],'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — All Models')
plt.legend()
plt.tight_layout()
plt.savefig('plots/06_roc_curves.png', dpi=150)
plt.show()

# 6d. Feature Importance (Random Forest)
rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=True)

plt.figure(figsize=(8,6))
importances_sorted.plot(kind='barh', color='#9b59b6')
plt.title('Feature Importance — Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('plots/07_feature_importance.png', dpi=150)
plt.show()

# ============================================================
# STEP 7 — FINAL RESULTS SUMMARY
# ============================================================
print("\n" + "="*50)
print("📊 FINAL RESULTS SUMMARY")
print("="*50)
for name, res in results.items():
    print(f"\n{name}:")
    print(f"  Accuracy : {res['accuracy']*100:.2f}%")
    print(f"  ROC-AUC  : {res['auc']:.4f}")
    print(f"  CV Score : {res['cv_mean']*100:.2f}% ± {res['cv_std']*100:.2f}%")

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Accuracy: {best['accuracy']*100:.2f}%")
print(f"   AUC: {best['auc']:.4f}")

print("\n✅ Project Complete! All plots saved to /plots folder")
