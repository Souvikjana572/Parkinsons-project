import os
import numpy as np
import librosa

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DIR = os.path.join(BASE_DIR, "data", "voice")

FEATURES = []
LABELS = []

def extract_mfcc(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

for folder in os.listdir(VOICE_DIR):
    folder_path = os.path.join(VOICE_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    # Healthy = 0, Parkinson = 1
    label = 1 if "Parkinson" in folder else 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                file_path = os.path.join(root, file)
                try:
                    mfcc = extract_mfcc(file_path)
                    FEATURES.append(mfcc)
                    LABELS.append(label)
                except Exception as e:
                    print("❌ Skipping:", file_path, e)

X = np.array(FEATURES)
y = np.array(LABELS)

print("\n✅ Extracted features:", X.shape)
print("✅ Labels:", y.shape)

np.save(os.path.join(BASE_DIR, "data", "voice_features.npy"), X)
np.save(os.path.join(BASE_DIR, "data", "voice_labels.npy"), y)

print("💾 Saved voice_features.npy and voice_labels.npy")
