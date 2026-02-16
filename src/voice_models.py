import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


from utils import compute_metrics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X_PATH = os.path.join(BASE_DIR, "data", "voice_features.npy")
y_PATH = os.path.join(BASE_DIR, "data", "voice_labels.npy")

X = np.load(X_PATH)
y = np.load(y_PATH)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF)": SVC(kernel="rbf"),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}


results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = compute_metrics(y_test, preds)

def get_results():
    return results
