import os
import torch
from torch.utils.data import Dataset
from preprocess import extract_mfcc

class ParkinsonDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        print("Scanning dataset recursively in:", root_dir)

        for root, _, files in os.walk(root_dir):
            root_lower = root.lower()

            if "healthy control" in root_lower:
                label = 0
            elif "parkinson" in root_lower:
                label = 1
            else:
                continue

            for file in files:
                if file.lower().endswith(".wav"):
                    self.samples.append(
                        (os.path.join(root, file), label)
                    )

        print("Total valid .wav samples found:", len(self.samples))

        if len(self.samples) == 0:
            raise RuntimeError("No audio files found. Check dataset structure.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        mfcc = extract_mfcc(file_path)

        return (
            torch.tensor(mfcc, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )
