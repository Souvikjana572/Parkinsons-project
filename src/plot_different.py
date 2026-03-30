import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "model_comparison_results.csv"
OUT_DIR = "results_separate"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Use exact dataset names as they appear in your CSV:
VOICE_DATASET = "Voice Recording"
CSV_DATASET = "Telemonitoring CSV"


def plot_for_dataset(dataset_name: str, file_prefix: str):
    df_d = df[df["Dataset"] == dataset_name].copy()

    if df_d.empty:
        raise ValueError(f"Dataset '{dataset_name}' not found in CSV. Check spelling in CSV.")

    # Sort models by accuracy (best first) for THIS dataset only
    model_order = df_d.sort_values("Accuracy", ascending=False)["Model"].tolist()
    df_d["Model"] = pd.Categorical(df_d["Model"], categories=model_order, ordered=True)
    df_d = df_d.sort_values("Model")

    # =========================
    # 1) BAR GRAPH (Accuracy)
    # =========================
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_d, x="Model", y="Accuracy")

    plt.ylim(0.8, 1.0)   # 🔥 Start Y-axis from 0.8

    plt.xticks(rotation=30, ha="right")
    plt.title(f"Accuracy Comparison (Bar) - {dataset_name}")
    plt.tight_layout()


    bar_path = os.path.join(OUT_DIR, f"{file_prefix}_bar.png")
    plt.savefig(bar_path, dpi=300)
    plt.show()
    print("✅ Saved:", bar_path)

    # =========================
    # 2) LINE GRAPH (Accuracy)
    # =========================
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_d, x="Model", y="Accuracy", marker="o")
    plt.xticks(rotation=30, ha="right")
    plt.title(f"Accuracy Comparison (Line) - {dataset_name}")
    plt.tight_layout()

    line_path = os.path.join(OUT_DIR, f"{file_prefix}_line.png")
    plt.savefig(line_path, dpi=300)
    plt.show()
    print("✅ Saved:", line_path)

    # =========================
    # 3) HEATMAP (Accuracy only, same models)
    # =========================
    # Heatmap should show model vs metric for THIS dataset
    heat_df = df_d.set_index("Model")[["Accuracy"]]

    plt.figure(figsize=(6, max(4, len(heat_df) * 0.5)))
    sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5)
    plt.title(f"Accuracy Heatmap - {dataset_name}")
    plt.tight_layout()

    heat_path = os.path.join(OUT_DIR, f"{file_prefix}_heatmap.png")
    plt.savefig(heat_path, dpi=300)
    plt.show()
    print("✅ Saved:", heat_path)


# ✅ Run same plotting pipeline TWICE (separate)
plot_for_dataset(VOICE_DATASET, "voice")
plot_for_dataset(CSV_DATASET, "csv")
