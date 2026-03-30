---

# 🧠 AI-Based Parkinson’s Disease Detection using Voice Signals

## 📌 Overview

Parkinson’s Disease (PD) is a progressive neurological disorder that affects movement and speech. Early detection is crucial for effective treatment and management. This project presents a **non-invasive AI-based approach** to detect Parkinson’s Disease using **voice signal analysis**.

The model leverages **deep learning techniques (BiLSTM + Attention)** to capture temporal dependencies in speech data and identify patterns associated with Parkinson’s disease.

---

## 🎯 Objectives

* Detect Parkinson’s Disease using voice recordings
* Build a robust deep learning model for classification
* Analyze voice features such as jitter, shimmer, and pitch
* Improve early diagnosis using non-invasive methods

---

## 🧪 Dataset

* Voice dataset containing biomedical speech features
* Includes attributes like:

  * Jitter
  * Shimmer
  * Fundamental Frequency (Fo)
  * Harmonic-to-Noise Ratio (HNR)

*(You can add dataset link here if needed)*

---

## 🧠 Model Architecture

* **Bidirectional LSTM (BiLSTM):**

  * Captures forward and backward temporal dependencies
* **Attention Mechanism:**

  * Focuses on the most relevant parts of the sequence
* **Hybrid Model (HPDHM):**

  * Combines BiLSTM + Attention for improved performance

---

## ⚙️ Technologies & Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow
* Keras

---

## 📊 Features

* Data preprocessing and normalization
* Feature correlation heatmap visualization
* Deep learning model training
* Performance evaluation (Accuracy, Loss)
* Comparison with traditional ML models

---

## 📈 Results

* Achieved **~99% accuracy** on voice dataset
* Achieved **~96% accuracy** on clinical dataset
* Outperformed traditional machine learning models

---

## 🚀 How to Run the Project

```bash
# Clone the repository
git clone https://github.com/your-username/parkinson-detection.git

# Navigate to project folder
cd parkinson-detection

# Install dependencies
pip install -r requirements.txt

# Run the model
python main.py
```

---

## 📷 Visualizations

* Heatmaps for feature importance
* Training vs Validation accuracy graphs
* Confusion matrix

---

## 🔮 Future Work

* Integration with real-time voice input
* Mobile/web-based diagnostic application
* Use of Transformer-based models
* Larger and more diverse datasets

---

## 🤝 Contribution

Contributions are welcome! Feel free to fork the repository and submit a pull request.

---

## 📜 License

This project is for academic and research purposes.

---

## 👨‍💻 Author

**Udity Banerjee**


---
