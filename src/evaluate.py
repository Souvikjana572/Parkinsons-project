import os
import torch
from torch.utils.data import DataLoader

from dataset import ParkinsonDataset
from model import ParkinsonModel
from config import *

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate():
    # --------------------------
    # Paths (robust absolute path)
    # --------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "voice")
    MODEL_PATH = os.path.join(BASE_DIR, "src", "parkinson_model.pth")

    # --------------------------
    # Load dataset
    # --------------------------
    dataset = ParkinsonDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"✅ Total samples in dataset: {len(dataset)}")
    print(f"📂 Loading model from: {MODEL_PATH}")

    # --------------------------
    # Load trained model
    # --------------------------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\n❌ Model file not found at:\n{MODEL_PATH}\n\n"
            f"👉 Fix: Make sure your training saves the model here:\n"
            f"   models/parkinson_model.pth\n"
        )

    model = ParkinsonModel().to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.eval()

    all_preds = []
    all_labels = []

    # --------------------------
    # Inference loop
    # --------------------------
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            outputs = model(x)

            # Case 1: outputs shape = (batch, 2)  (2-class logits)
            if outputs.dim() == 2 and outputs.size(1) == 2:
                preds = torch.argmax(outputs, dim=1)

            # Case 2: outputs shape = (batch, 1) or (batch,) (binary logits)
            else:
                outputs = outputs.squeeze()
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # --------------------------
    # Metrics (binary)
    # --------------------------
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # --------------------------
    # Print results
    # --------------------------
    print("\n📊 MODEL EVALUATION RESULTS\n")
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-score  : {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    evaluate()
