"""
03_genai_insights.py
---------------------
Uses an LLM (Google Gemini, via the free Gemini API) with prompt
engineering to turn raw model outputs into plain-English, business-ready
explanations.

This is the "GenAI/LLM exposure" part of the project. This is real
prompt engineering: we feed the model structured data (customer profile +
top churn-driving features) and ask the LLM to reason over it and produce
a business recommendation.

WHY GEMINI: Google's Gemini API has a genuinely free, permanent tier
(1,500 requests/day on Gemini 2.5 Flash) that does not require a credit
card and does not expire. This script only makes 5 calls, so it costs
$0 and stays far under the daily limit.

SETUP (one-time):
    1. Get a free API key: https://aistudio.google.com/app/apikey
       (sign in with a Google account, click "Create API key" — no
       credit card requested)
    2. Run: pip install google-genai
    3. Set your key as an environment variable (see TUTORIAL.docx, Step 6,
       for exact steps), OR paste it into the API_KEY variable below (NOT
       recommended for anything you'll push to GitHub).

Run:
    python 03_genai_insights.py
Output:
    genai_customer_insights.txt
"""

import os
import pickle
import pandas as pd
from google import genai

# --- Load your API key from an environment variable (safer than hardcoding) ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit(
        "No API key found. Set the GEMINI_API_KEY environment variable first.\n"
        "See TUTORIAL.docx, Step 6, for exact instructions."
    )

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash"  # free-tier model, 1,500 requests/day, no cost

# --- Load model + data ---
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

top_features = []
try:
    top_features_df = pd.read_csv("top_features.csv", index_col=0)
    top_features = top_features_df.index.tolist()[:5]
except FileNotFoundError:
    pass  # Logistic Regression run — no feature_importance file, that's fine

df = pd.read_csv("telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

# Pick 5 example at-risk customers to generate insights for
sample_customers = df[df["Churn"] == "Yes"].sample(5, random_state=1)

def build_prompt(row):
    profile = {
        "tenure_months": int(row["tenure"]),
        "contract": row["Contract"],
        "monthly_charges": float(row["MonthlyCharges"]),
        "internet_service": row["InternetService"],
        "tech_support": row["TechSupport"],
        "online_security": row["OnlineSecurity"],
        "payment_method": row["PaymentMethod"],
    }
    return f"""You are a churn analyst assistant. Given this customer profile:
{profile}

And knowing the top churn drivers in our model are: {top_features if top_features else ['contract type', 'tenure', 'tech support']}.

In 2-3 sentences:
1. Explain the most likely reason this customer is at risk of churning.
2. Recommend ONE specific retention action for this customer.

Keep it business-friendly, no jargon, no bullet points."""

results = []
print("Generating GenAI insights for 5 at-risk customers...\n")
for idx, row in sample_customers.iterrows():
    prompt = build_prompt(row)
    response = client.models.generate_content(model=MODEL, contents=prompt)
    explanation = response.text
    results.append(f"Customer {row['customerID']}:\n{explanation}\n")
    print(f"Customer {row['customerID']}:\n{explanation}\n{'-'*60}")

with open("genai_customer_insights.txt", "w") as f:
    f.write("\n".join(results))

print("Saved: genai_customer_insights.txt")
