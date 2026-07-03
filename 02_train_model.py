"""
02_train_model.py
------------------
Trains two classification models (Logistic Regression, Random Forest) to
predict customer churn, evaluates them with Accuracy, Precision, Recall,
F1, and a Confusion Matrix, then saves the better model + feature list.

Run:
    python 02_train_model.py
Output:
    model.pkl              (trained model)
    feature_columns.pkl    (column order the model expects)
    eda_charts/confusion_matrix.png
    model_metrics.txt
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

df = pd.read_csv("telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])
df = df.drop(columns=["customerID"])

# Target
y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn"])

# One-hot encode categoricals
X = pd.get_dummies(X, drop_first=True)
feature_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# --- Model 1: Logistic Regression ---
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
log_reg.fit(X_train_scaled, y_train)
pred_lr = log_reg.predict(X_test_scaled)
results["Logistic Regression"] = {
    "accuracy": accuracy_score(y_test, pred_lr),
    "precision": precision_score(y_test, pred_lr),
    "recall": recall_score(y_test, pred_lr),
    "f1": f1_score(y_test, pred_lr),
}

# --- Model 2: Random Forest ---
rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
results["Random Forest"] = {
    "accuracy": accuracy_score(y_test, pred_rf),
    "precision": precision_score(y_test, pred_rf),
    "recall": recall_score(y_test, pred_rf),
    "f1": f1_score(y_test, pred_rf),
}

# --- Pick best model by recall (catching at-risk customers matters most for churn) ---
best_name = max(results, key=lambda k: results[k]["recall"])
best_model = rf if best_name == "Random Forest" else log_reg
best_pred = pred_rf if best_name == "Random Forest" else pred_lr

print("=== Model Comparison ===")
for name, m in results.items():
    print(f"{name}: Accuracy={m['accuracy']:.3f}  Precision={m['precision']:.3f}  "
          f"Recall={m['recall']:.3f}  F1={m['f1']:.3f}")
print(f"\nBest model selected: {best_name} (highest recall — prioritizes catching churners)")

# Confusion matrix chart
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title(f"Confusion Matrix - {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("eda_charts/confusion_matrix.png", dpi=120)
plt.close()

# Feature importance (if Random Forest is best)
if best_name == "Random Forest":
    importances = pd.Series(rf.feature_importances_, index=feature_columns)
    top_features = importances.sort_values(ascending=False).head(8)
    plt.figure(figsize=(7, 4))
    top_features.sort_values().plot(kind="barh", color="#4C72B0")
    plt.title("Top Churn Drivers (Feature Importance)")
    plt.tight_layout()
    plt.savefig("eda_charts/feature_importance.png", dpi=120)
    plt.close()
    top_features.to_csv("top_features.csv", header=["importance"])

# Save model + metadata
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)
if best_name == "Logistic Regression":
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

with open("model_metrics.txt", "w") as f:
    f.write(f"Best model: {best_name}\n\n")
    for name, m in results.items():
        f.write(f"{name}:\n")
        for k, v in m.items():
            f.write(f"  {k}: {v:.3f}\n")
    f.write(f"\nConfusion Matrix ({best_name}):\n{cm}\n")
    f.write(f"\n{classification_report(y_test, best_pred, target_names=['No Churn', 'Churn'])}")

print("\nSaved: model.pkl, feature_columns.pkl, model_metrics.txt, eda_charts/confusion_matrix.png")
