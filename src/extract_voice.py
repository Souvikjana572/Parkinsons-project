import os
import numpy as np
import librosa

# ==============================
# Paths
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DIR = os.path.join(BASE_DIR, "data", "voice")

SAVE_FEATURES = os.path.join(BASE_DIR, "data", "voice_features.npy")
SAVE_LABELS = os.path.join(BASE_DIR, "data", "voice_labels.npy")

# ==============================
# Config
# ==============================
N_MFCC = 40
SUPPORTED_EXTS = (".wav", ".mp3", ".flac", ".ogg")

FEATURES = []
LABELS = []

def extract_mfcc(file_path, n_mfcc=40):
    """
    Extract MFCC features from an audio file.
    Returns fixed-length vector by taking mean over time.
    """
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean


def get_label_from_folder(folder_name: str):
    """
    Healthy folders -> 0
    Parkinson folder -> 1
    """
    if "Parkinson" in folder_name or "parkinson" in folder_name:
        return 1
    return 0


if __name__ == "__main__":
    if not os.path.exists(VOICE_DIR):
        raise FileNotFoundError(f"❌ Voice dataset folder not found: {VOICE_DIR}")

    print("Scanning dataset recursively in:", VOICE_DIR)

    total_files = 0
    total_used = 0

    # Walk over each class folder
    for folder in os.listdir(VOICE_DIR):
        folder_path = os.path.join(VOICE_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        label = get_label_from_folder(folder)

        # Recursively scan inside folder
        for root, _, files in os.walk(folder_path):
            for file in files:
                total_files += 1

                if not file.lower().endswith(SUPPORTED_EXTS):
                    continue

                file_path = os.path.join(root, file)

                try:
                    mfcc_vec = extract_mfcc(file_path, n_mfcc=N_MFCC)
                    FEATURES.append(mfcc_vec)
                    LABELS.append(label)
                    total_used += 1
                except Exception as e:
                    print(f"❌ Skipping {file_path} بسبب error: {e}")

    X = np.array(FEATURES)
    y = np.array(LABELS)

    print(f"Total files scanned: {total_files}")
    print(f"Total valid audio samples found: {total_used}")

    print("Extracted features shape:", X.shape)
    print("Labels shape:", y.shape)

    # Save
    np.save(SAVE_FEATURES, X)
    np.save(SAVE_LABELS, y)

    print("\n✅ Voice features saved to:", SAVE_FEATURES)
    print("✅ Voice labels saved to:", SAVE_LABELS)
