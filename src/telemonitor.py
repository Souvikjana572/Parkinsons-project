import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier


def updrs_to_class(value):
    if value < 20:
        return 0
    elif value < 40:
        return 1
    else:
        return 2


def evaluate():
   
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "telemonitoring", "parkinsons_updrs.data")

    print("📂 Loading dataset from:", DATA_PATH)

    df = pd.read_csv(DATA_PATH)

   
    df["target"] = df["total_UPDRS"].apply(updrs_to_class)

    # Features + labels
    X = df.drop(columns=["motor_UPDRS", "total_UPDRS", "target"])
    y = df["target"]

   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

   
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

   
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["Mild", "Moderate", "Severe"],
        zero_division=0
    )

    print("\n📊 MODEL EVALUATION RESULTS (Telemonitoring CSV)\n")
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-score  : {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\n📋 Classification Report:\n")
    print(report)


if __name__ == "__main__":
    evaluate()
