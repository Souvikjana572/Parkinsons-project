import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load MFCC features
X = np.load("data/voice_features.npy")

df = pd.DataFrame(X, columns=[f"MFCC_{i+1}" for i in range(X.shape[1])])

corr = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr,
    cmap="magma",
    square=True,
    cbar=True
)

plt.title("Correlation Heatmap of MFCC Features (Voice Dataset)")
plt.tight_layout()
plt.savefig("voice_mfcc_correlation_heatmap.png", dpi=300)
plt.show()