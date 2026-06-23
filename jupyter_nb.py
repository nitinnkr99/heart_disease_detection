#cell_1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
import joblib
import warnings
warnings.filterwarnings('ignore')

# File is actually CSV despite .xls extension
df = pd.read_csv('heart.xls')
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nClass distribution:")
print(df['target'].value_counts())
df.head()

#cell_2
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Class distribution
counts = df['target'].value_counts()
axes[0].bar(['Heart Disease', 'No Disease'], counts.values, color=['#E24B4A', '#1D9E75'], width=0.5)
axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 3, str(v), ha='center', fontweight='bold')

# 2. Age distribution by class
axes[1].hist(df[df['target']==1]['age'], bins=20, alpha=0.7, color='#E24B4A', label='Heart Disease')
axes[1].hist(df[df['target']==0]['age'], bins=20, alpha=0.7, color='#1D9E75', label='No Disease')
axes[1].set_title('Age Distribution by Class', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Age')
axes[1].legend()

# 3. Correlation heatmap
sns.heatmap(df.corr(), ax=axes[2], cmap='coolwarm', annot=False, linewidths=0.5)
axes[2].set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("EDA complete.")


#cell_3
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")
print(f"Features         : {X_train.shape[1]}")


#cell_4
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':                 SVC(probability=True, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred   = model.predict(X_test_sc)
    y_prob   = model.predict_proba(X_test_sc)[:, 1]
    cv_score = cross_val_score(model, X_train_sc, y_train, cv=5).mean()
    results[name] = {
        'model':    model,
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc':  roc_auc_score(y_test, y_prob),
        'cv_score': cv_score,
        'y_pred':   y_pred,
        'y_prob':   y_prob
    }
    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  CV Score : {cv_score:.4f}")
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Heart Disease']))
    
#cell_4b

# Random Forest with constraints to prevent overfitting
rf_tuned = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,          # limit tree depth
    min_samples_split=10, # need 10 samples to split
    min_samples_leaf=4,   # need 4 samples at leaf
    random_state=42
)
rf_tuned.fit(X_train_sc, y_train)
y_pred_rf  = rf_tuned.predict(X_test_sc)
y_prob_rf  = rf_tuned.predict_proba(X_test_sc)[:, 1]
cv_rf      = cross_val_score(rf_tuned, X_train_sc, y_train, cv=5).mean()

print(f"Tuned Random Forest")
print(f"Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob_rf):.4f}")
print(f"CV Score : {cv_rf:.4f}")
print(classification_report(y_test, y_pred_rf, target_names=['No Disease','Heart Disease']))

# Update results with tuned model
results['Random Forest'] = {
    'model':    rf_tuned,
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'roc_auc':  roc_auc_score(y_test, y_prob_rf),
    'cv_score': cv_rf,
    'y_pred':   y_pred_rf,
    'y_prob':   y_prob_rf
}

#cell_5
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, (name, res) in enumerate(results.items()):
    # Confusion matrix
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0][i],
                xticklabels=['No Disease', 'Heart Disease'],
                yticklabels=['No Disease', 'Heart Disease'])
    axes[0][i].set_title(f'{name}\nConfusion Matrix', fontweight='bold')
    axes[0][i].set_ylabel('Actual')
    axes[0][i].set_xlabel('Predicted')

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[1][i].plot(fpr, tpr, color='#185FA5', lw=2, label=f"AUC = {res['roc_auc']:.3f}")
    axes[1][i].plot([0,1],[0,1], 'k--', lw=1)
    axes[1][i].set_title(f'{name}\nROC Curve', fontweight='bold')
    axes[1][i].set_xlabel('False Positive Rate')
    axes[1][i].set_ylabel('True Positive Rate')
    axes[1][i].legend(loc='lower right')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary table
print("\n===== MODEL COMPARISON SUMMARY =====")
summary = pd.DataFrame({
    name: {
        'Accuracy': f"{r['accuracy']:.4f}",
        'ROC-AUC':  f"{r['roc_auc']:.4f}",
        'CV Score': f"{r['cv_score']:.4f}"
    }
    for name, r in results.items()
}).T
print(summary)

#cell_6
best_name  = max(results, key=lambda k: results[k]['roc_auc'])
best_model = results[best_name]['model']
print(f"Best model: {best_name}  (ROC-AUC: {results[best_name]['roc_auc']:.4f})")

joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler,     'scaler.pkl')
print("Saved: best_model.pkl and scaler.pkl")    




