import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "telemonitoring", "parkinsons_updrs.data")

df = pd.read_csv(DATA_PATH)

# Remove non-feature columns
features = df.drop(columns=["motor_UPDRS", "total_UPDRS"])

# Correlation matrix
corr = features.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr,
    cmap="magma",
    square=True,
    cbar=True
)

plt.title("Correlation Heatmap of Telemonitoring Features")
plt.tight_layout()
plt.savefig("telemonitor_correlation_heatmap.png", dpi=300)
plt.show()