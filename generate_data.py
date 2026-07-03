"""
generate_data.py
-----------------
Creates a realistic synthetic telco customer churn dataset with the SAME
column structure as the popular Kaggle "Telco Customer Churn" dataset.

Why this exists: it lets you build and test the entire project right now,
without needing internet access to download data. Later, you can swap in
the real Kaggle dataset (same column names) with zero code changes.

Run:
    python generate_data.py
Output:
    telco_churn.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 7043  # same size as the real Kaggle dataset

gender = np.random.choice(["Male", "Female"], n)
senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])
tenure = np.random.randint(0, 73, n)

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
)
paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    n,
)
internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
phone_service = np.random.choice(["Yes", "No"], n, p=[0.90, 0.10])

def dependent_service(base_yes_prob):
    return np.random.choice(["Yes", "No"], n, p=[base_yes_prob, 1 - base_yes_prob])

online_security = dependent_service(0.29)
online_backup = dependent_service(0.34)
device_protection = dependent_service(0.34)
tech_support = dependent_service(0.29)
streaming_tv = dependent_service(0.38)
streaming_movies = dependent_service(0.39)
multiple_lines = dependent_service(0.42)

monthly_charges = np.round(np.random.uniform(18, 120, n), 2)
total_charges = np.round(monthly_charges * tenure + np.random.uniform(-50, 50, n), 2)
total_charges = np.clip(total_charges, 0, None)

# ---- Build churn probability from realistic drivers (this creates the
# patterns your EDA and model will actually discover) ----
churn_prob = np.full(n, 0.03)
churn_prob += np.where(contract == "Month-to-month", 0.18, 0)
churn_prob += np.where(contract == "One year", 0.02, 0)
churn_prob += np.where(internet_service == "Fiber optic", 0.08, 0)
churn_prob += np.where(tech_support == "No", 0.05, 0)
churn_prob += np.where(online_security == "No", 0.04, 0)
churn_prob += np.where(tenure < 12, 0.12, 0)
churn_prob += np.where(tenure > 48, -0.08, 0)
churn_prob += np.where(monthly_charges > 80, 0.05, 0)
churn_prob += np.where(paperless_billing == "Yes", 0.02, 0)
churn_prob += np.where(payment_method == "Electronic check", 0.04, 0)
churn_prob = np.clip(churn_prob, 0.02, 0.90)

churn = np.where(np.random.rand(n) < churn_prob, "Yes", "No")

df = pd.DataFrame({
    "customerID": [f"{i:04d}-{''.join(np.random.choice(list('ABCDEFGHIJ'), 5))}" for i in range(n)],
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn,
})

df.to_csv("telco_churn.csv", index=False)
print(f"Created telco_churn.csv with {len(df)} rows.")
print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
