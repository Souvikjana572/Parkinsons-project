import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "model_comparison_results.csv"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

OUR_METHOD_NAME = "Our Method (BiLSTM+Attention)"  # change if your name is different

df_sorted = df.copy()

# Ensure our method exists
if OUR_METHOD_NAME not in df_sorted["Model"].values:
    raise ValueError(
        f"❌ Our method '{OUR_METHOD_NAME}' not found in CSV.\n"
        f"Please add it to model_comparison_results.csv first."
    )

# =========================
# Sort models by mean accuracy (across datasets)
# =========================
model_order = (
    df_sorted.groupby("Model")["Accuracy"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

df_sorted["Model"] = pd.Categorical(df_sorted["Model"], categories=model_order, ordered=True)
df_sorted = df_sorted.sort_values("Model")

# =========================
# 1) BAR GRAPH
# =========================
plt.figure(figsize=(12, 6))
sns.barplot(data=df_sorted, x="Model", y="Accuracy", hue="Dataset")

ax = plt.gca()
for tick in ax.get_xticklabels():
    if tick.get_text() == OUR_METHOD_NAME:
        tick.set_fontweight("bold")

plt.xticks(rotation=30, ha="right")
plt.title("Accuracy Comparison: Our Method vs Other Models (Sorted)")
plt.tight_layout()

bar_path = os.path.join(OUT_DIR, "our_method_vs_others_bar.png")
plt.savefig(bar_path, dpi=300)
plt.show()
print("✅ Saved:", bar_path)

# =========================
# 2) LINE COMPARISON
# =========================
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df_sorted,
    x="Model",
    y="Accuracy",
    hue="Dataset",
    marker="o"
)

ax = plt.gca()
for tick in ax.get_xticklabels():
    if tick.get_text() == OUR_METHOD_NAME:
        tick.set_fontweight("bold")

plt.xticks(rotation=30, ha="right")
plt.title("Line Comparison: Our Method vs Other Models")
plt.tight_layout()

line_path = os.path.join(OUT_DIR, "our_method_vs_others_line.png")
plt.savefig(line_path, dpi=300)
plt.show()
print("✅ Saved:", line_path)

# =========================
# 3) HEATMAP (Accuracy matrix)
# =========================
pivot_acc = df_sorted.pivot(index="Model", columns="Dataset", values="Accuracy")

plt.figure(figsize=(8, max(4, len(pivot_acc) * 0.6)))
sns.heatmap(
    pivot_acc,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    linewidths=0.5
)

plt.title("Accuracy Heatmap: Model vs Dataset")
plt.tight_layout()

heatmap_path = os.path.join(OUT_DIR, "our_method_vs_others_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.show()
print("✅ Saved:", heatmap_path)
