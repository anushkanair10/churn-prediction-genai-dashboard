"""
05_export_for_powerbi.py
--------------------------
Scores every customer with the trained model and exports a clean CSV
that Power BI can import directly to build the churn-risk dashboard.

Run:
    python 05_export_for_powerbi.py
Output:
    powerbi_churn_export.csv
"""

import pickle
import pandas as pd

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

scaler = None
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    pass

df = pd.read_csv("telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

X = df.drop(columns=["customerID", "Churn"])
X = pd.get_dummies(X, drop_first=True)
X = X.reindex(columns=feature_columns, fill_value=0)  # align columns exactly

X_for_pred = scaler.transform(X) if scaler is not None else X
df["PredictedChurnProb"] = model.predict_proba(X_for_pred)[:, 1]
df["PredictedChurn"] = model.predict(X_for_pred)
df["RiskTier"] = pd.cut(
    df["PredictedChurnProb"], bins=[0, 0.33, 0.66, 1.0],
    labels=["Low", "Medium", "High"]
)
df["AtRiskRevenue"] = df["MonthlyCharges"] * (df["PredictedChurn"] == 1)

export_cols = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "Contract", "InternetService", "TechSupport", "OnlineSecurity",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
    "PredictedChurnProb", "PredictedChurn", "RiskTier", "AtRiskRevenue",
]
df[export_cols].to_csv("powerbi_churn_export.csv", index=False)

print(f"Exported {len(df)} rows to powerbi_churn_export.csv")
print(f"Total at-risk monthly revenue: ${df['AtRiskRevenue'].sum():,.2f}")
print(f"High-risk customers: {(df['RiskTier'] == 'High').sum()}")
