"""
04_qa_tests.py
---------------
Applies QA rigor to the model: sanity-checks that predictions behave the
way a real business stakeholder would expect. This is the same mindset as
functional testing — you're not just trusting the model, you're verifying it.

Run:
    python 04_qa_tests.py
Output:
    qa_test_results.txt (prints PASS/FAIL for each test)
"""

import pickle
import pandas as pd
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

scaler = None
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    pass  # Random Forest was best model — no scaler needed

def make_input(overrides):
    base = {col: 0 for col in feature_columns}
    base.update(overrides)
    row = pd.DataFrame([base])[feature_columns]
    return row

def predict(row):
    if scaler is not None:
        row = scaler.transform(row)
    return model.predict(row)[0], model.predict_proba(row)[0][1]

tests = []

# Test 1: A brand-new, month-to-month, high-charge customer with no tech
# support SHOULD be flagged as high risk.
row = make_input({
    "tenure": 1, "MonthlyCharges": 95,
    "Contract_One year": 0, "Contract_Two year": 0,  # implies month-to-month
    "TechSupport_Yes": 0,
})
pred, prob = predict(row)
tests.append(("High-risk profile flagged as churn", pred == 1, f"predicted={pred}, prob={prob:.2f}"))

# Test 2: A loyal, 2-year-contract, long-tenure customer SHOULD be flagged
# as low risk.
row = make_input({
    "tenure": 60, "MonthlyCharges": 50,
    "Contract_Two year": 1,
    "TechSupport_Yes": 1,
})
pred, prob = predict(row)
tests.append(("Low-risk profile flagged as no-churn", pred == 0, f"predicted={pred}, prob={prob:.2f}"))

# Test 3: Model output is a valid probability (between 0 and 1)
_, prob = predict(make_input({"tenure": 12, "MonthlyCharges": 60}))
tests.append(("Predicted probability is within [0,1]", 0 <= prob <= 1, f"prob={prob:.2f}"))

# Test 4: Model predictions are deterministic (same input -> same output)
row = make_input({"tenure": 24, "MonthlyCharges": 70})
p1, _ = predict(row)
p2, _ = predict(row)
tests.append(("Model is deterministic for identical input", p1 == p2, f"run1={p1}, run2={p2}"))

# Test 5: No missing feature columns break prediction
row = make_input({})
try:
    predict(row)
    ok = True
    detail = "no exception raised"
except Exception as e:
    ok = False
    detail = str(e)
tests.append(("Model handles all-default input without crashing", ok, detail))

print("=== QA Test Results ===")
lines = []
for name, passed, detail in tests:
    status = "PASS" if passed else "FAIL"
    line = f"[{status}] {name} ({detail})"
    print(line)
    lines.append(line)

with open("qa_test_results.txt", "w") as f:
    f.write("\n".join(lines))
