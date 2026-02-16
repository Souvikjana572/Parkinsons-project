import os
import pandas as pd

from telemonitor_models import get_results as tele_results
from voice_models import get_results as voice_results


def main():
    tele = tele_results()     # Telemonitoring baseline models
    voice = voice_results()   # Voice baseline models

    rows = []

    # Telemonitoring CSV results
    for model, metrics in tele.items():
        rows.append([
            "Telemonitoring CSV",
            model,
            metrics["Accuracy"],
            metrics["Precision"],
            metrics["Recall"],
            metrics["F1"]
        ])

    # Voice Recording results
    for model, metrics in voice.items():
        rows.append([
            "Voice Recording",
            model,
            metrics["Accuracy"],
            metrics["Precision"],
            metrics["Recall"],
            metrics["F1"]
        ])

    df = pd.DataFrame(rows, columns=["Dataset", "Model", "Accuracy", "Precision", "Recall", "F1"])

    df.to_csv("model_comparison_results.csv", index=False)

    print("\n✅ Baseline results saved to model_comparison_results.csv")
    print(df)


if __name__ == "__main__":
    main()
