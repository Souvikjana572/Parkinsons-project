import os
import pandas as pd

# ==============================
# Settings (edit these)
# ==============================
CSV_PATH = "model_comparison_results.csv"
OUR_METHOD_NAME = "Our Method (BiLSTM+Attention)"

# Put your final trained model scores here:
OUR_METHOD_RESULTS = [
    {
        "Dataset": "Voice Recording",
        "Model": OUR_METHOD_NAME,
        "Accuracy": 0.99,
        "Precision": 1.00,
        "Recall": 0.99,
        "F1": 0.99
    },
    {
        "Dataset": "Telemonitoring CSV",
        "Model": OUR_METHOD_NAME,
        "Accuracy": 0.96,
        "Precision": 0.96,
        "Recall": 0.96,
        "F1": 0.96
    }
]

# ==============================
# Add/Update our method rows
# ==============================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ {CSV_PATH} not found. Run evaluate_models.py first.")

df = pd.read_csv(CSV_PATH)

our_df = pd.DataFrame(OUR_METHOD_RESULTS)

# Remove old rows of our method if already present
df = df[~((df["Model"] == OUR_METHOD_NAME) & (df["Dataset"].isin(our_df["Dataset"])))]

# Append new rows
df = pd.concat([df, our_df], ignore_index=True)

# Save
df.to_csv(CSV_PATH, index=False)

print("✅ Our method added/updated successfully!")
print("✅ Updated file:", CSV_PATH)
