# AI-Powered Customer Churn Prediction & Insight System

Predicts which telecom customers are likely to churn, explains *why* in
plain English using an LLM, and visualizes risk for business stakeholders
in Power BI.

## Why this project

Most churn-prediction projects stop at a model and an accuracy score. This
one goes one step further: it turns raw model output into a decision a
business person can actually act on — using prompt engineering with
Google's Gemini API (free tier, no credit card required) to translate
feature importances into a plain English explanation and a recommended
retention action, per customer.

## Pipeline

```
telco_churn.csv
      │
      ▼
01_eda.py  ──────────► eda_charts/*.png  (churn patterns by contract,
                                            tenure, charges, tech support)
      │
      ▼
02_train_model.py ───► model.pkl  (Logistic Regression vs Random Forest,
                                    evaluated on Accuracy/Precision/
                                    Recall/F1 + Confusion Matrix)
      │
      ▼
03_genai_insights.py ► genai_customer_insights.txt  (LLM-generated,
                        plain-English explanation + retention action
                        per at-risk customer)
      │
      ▼
04_qa_tests.py ──────► qa_test_results.txt  (functional tests validating
                        the model behaves as expected on known profiles)
      │
      ▼
05_export_for_powerbi.py ► powerbi_churn_export.csv  (feeds the
                            Power BI dashboard)
```

## Key Results

- **Best model:** see `model_metrics.txt` for Accuracy / Precision /
  Recall / F1 and the confusion matrix.
- **Top churn drivers:** month-to-month contracts, low tenure, and lack
  of tech support — see `eda_charts/feature_importance.png`.
- **Business impact:** total at-risk monthly revenue and high-risk
  customer count are printed by `05_export_for_powerbi.py` and visualized
  in the Power BI dashboard.

## Tech Stack

Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn) · Google Gemini
API for GenAI insight generation · Power BI (DAX, Data Modeling) for the
stakeholder-facing dashboard.

## How to run

See `TUTORIAL.docx` for exact, click-by-click setup instructions
(Python install, virtual environment, API key setup, Power BI import).

Quick version:
```bash
pip install -r requirements.txt
python generate_data.py          # or drop in the real Kaggle Telco dataset
python 01_eda.py
python 02_train_model.py
python 04_qa_tests.py
export GEMINI_API_KEY=your_key_here        # Windows: set GEMINI_API_KEY=your_key_here
python 03_genai_insights.py
python 05_export_for_powerbi.py
```

## Dashboard

Import `powerbi_churn_export.csv` into Power BI Desktop to build the
churn-risk dashboard (see TUTORIAL.docx, Step 8).

## Sample GenAI Output

> *"This customer is on a month-to-month contract with only 3 months of
> tenure and no tech support — a profile strongly associated with early
> churn. Recommend proactively offering a discounted annual plan with a
> complimentary tech support add-on to increase switching cost and
> improve support experience."*

## Note on the dataset

`telco_churn.csv` in this repo is synthetically generated
(`generate_data.py`) to mirror the structure and churn patterns of the
well-known Kaggle "Telco Customer Churn" dataset, so the project runs
immediately without needing external downloads. Swap in the real Kaggle
dataset (same column names) for production-grade results — see
TUTORIAL.docx, Step 3, Option B.
