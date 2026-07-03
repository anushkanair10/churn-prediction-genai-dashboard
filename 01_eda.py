"""
01_eda.py
---------
Exploratory Data Analysis on the churn dataset.
Generates 4 charts (saved as PNGs) and prints key insights to the terminal.

Run:
    python 01_eda.py
Output:
    eda_charts/*.png
    insights.txt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("eda_charts", exist_ok=True)
sns.set_style("whitegrid")

df = pd.read_csv("telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

insights = []
overall_churn = (df["Churn"] == "Yes").mean()
insights.append(f"Overall churn rate: {overall_churn:.1%}")

# 1. Churn by contract type
plt.figure(figsize=(6, 4))
contract_churn = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean())
contract_churn.sort_values(ascending=False).plot(kind="bar", color="#4C72B0")
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.savefig("eda_charts/churn_by_contract.png", dpi=120)
plt.close()
top_contract = contract_churn.idxmax()
insights.append(
    f"{top_contract} customers churn at {contract_churn.max():.1%}, "
    f"vs {contract_churn.min():.1%} for {contract_churn.idxmin()} customers."
)

# 2. Churn by tenure buckets
plt.figure(figsize=(6, 4))
df["tenure_bucket"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 72],
                              labels=["0-12mo", "13-24mo", "25-48mo", "49-72mo"])
tenure_churn = df.groupby("tenure_bucket", observed=True)["Churn"].apply(lambda x: (x == "Yes").mean())
tenure_churn.plot(kind="bar", color="#DD8452")
plt.title("Churn Rate by Tenure")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.savefig("eda_charts/churn_by_tenure.png", dpi=120)
plt.close()
insights.append(
    f"New customers (0-12mo) churn at {tenure_churn.iloc[0]:.1%}, "
    f"nearly {tenure_churn.iloc[0] / tenure_churn.iloc[-1]:.1f}x the rate of "
    f"long-tenure customers (49-72mo) at {tenure_churn.iloc[-1]:.1%}."
)

# 3. Monthly charges distribution by churn
plt.figure(figsize=(6, 4))
sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, common_norm=False)
plt.title("Monthly Charges Distribution by Churn Status")
plt.tight_layout()
plt.savefig("eda_charts/charges_by_churn.png", dpi=120)
plt.close()
avg_charge_churn = df[df["Churn"] == "Yes"]["MonthlyCharges"].mean()
avg_charge_stay = df[df["Churn"] == "No"]["MonthlyCharges"].mean()
insights.append(
    f"Customers who churned paid ${avg_charge_churn:.2f}/month on average, "
    f"vs ${avg_charge_stay:.2f}/month for those who stayed."
)

# 4. Churn by tech support
plt.figure(figsize=(6, 4))
support_churn = df.groupby("TechSupport")["Churn"].apply(lambda x: (x == "Yes").mean())
support_churn.plot(kind="bar", color="#55A868")
plt.title("Churn Rate by Tech Support Subscription")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.savefig("eda_charts/churn_by_techsupport.png", dpi=120)
plt.close()
insights.append(
    f"Customers WITHOUT tech support churn at {support_churn.get('No', 0):.1%}, "
    f"vs {support_churn.get('Yes', 0):.1%} for those with tech support."
)

with open("insights.txt", "w") as f:
    f.write("\n".join(insights))

print("EDA complete. Charts saved to eda_charts/. Insights:")
for i in insights:
    print(" -", i)
