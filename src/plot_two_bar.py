import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "model_comparison_results.csv"
OUT_PATH = "two_dataset_model_comparison_colored.png"

df = pd.read_csv(CSV_PATH)

# 🔹 Clean model names (remove hidden spaces)
df["Model"] = df["Model"].str.strip()

VOICE_NAME = "Voice Recording"
CSV_NAME = "Telemonitoring CSV"

voice_df = df[df["Dataset"] == VOICE_NAME].copy()
csv_df = df[df["Dataset"] == CSV_NAME].copy()

if voice_df.empty or csv_df.empty:
    raise ValueError("One of the datasets is missing in the CSV.")

# 🔹 Get ALL models across both datasets
all_models = sorted(df["Model"].unique())

# 🔹 Reindex safely (this prevents dropping HPDHM)
voice_df = voice_df.set_index("Model").reindex(all_models)
csv_df = csv_df.set_index("Model").reindex(all_models)

models = all_models
voice_acc = voice_df["Accuracy"].values
csv_acc = csv_df["Accuracy"].values

x = np.arange(len(models))
width = 0.6

# 🎨 Define fixed color per model
colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# -------- Voice Dataset --------
axes[0].bar(x, voice_acc, width, color=colors)
axes[0].set_title("Voice Dataset")
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=30, ha="right")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0.8, 1.0)

# -------- CSV Dataset --------
axes[1].bar(x, csv_acc, width, color=colors)
axes[1].set_title("Telemonitoring CSV Dataset")
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, rotation=30, ha="right")
axes[1].set_ylim(0.8, 1.0)

plt.suptitle("Model-wise Performance Comparison Across Two Datasets", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", OUT_PATH)
