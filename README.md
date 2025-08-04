# ECG Arrhythmia Classification Using Lightweight Machine Learning Models

## 🔍 Overview
This project presents a lightweight and interpretable arrhythmia classification system based on ECG (Electrocardiogram) signals. It overcomes the limitations of traditional and deep learning models by delivering accurate classification with low computational cost—ideal for real-time wearable health monitoring devices.

---

## 🧠 Motivation
Cardiac arrhythmias can lead to severe conditions like stroke, heart attack, or sudden cardiac death. Early detection and classification are critical for effective treatment.

While deep learning models offer high accuracy, they:
- Consume high computational resources
- Lack transparency (black-box)
- Are unsuitable for wearable devices

This project aims to provide an efficient and explainable solution using:
- Heartbeat dynamics-based feature extraction
- Lightweight and ensemble machine learning models
- Dimensionality reduction (PCA)
- Class imbalance handling (SMOTE + ENN)

---

## 🎯 Key Features
- ✅ **High Accuracy** with lightweight models like KNN, SVM, and Random Forest
- 🔁 **Advanced Ensemble Models**: AdaBoost, Gradient Boosting, XGBoost, CatBoost
- 📉 **PCA** for dimensionality reduction and improved model efficiency
- ⚖️ **SMOTE and ENN** for class imbalance handling
- 🧾 **Evaluated using multiple metrics**: Accuracy, Precision, Recall, F1-Score
- 🖥️ **Low memory and high-speed**—ideal for wearables

---

## 🗃️ Dataset
- **MIT-BIH Arrhythmia Database**
  - 48 half-hour ECG recordings from 47 patients
  - Each heartbeat labeled with arrhythmia types
  - Used for training, testing, and validation

---

## ⚙️ Tech Stack
- **Language**: Python 3.9
- **Libraries**: 
  - `scikit-learn`
  - `xgboost`, `catboost`
  - `imblearn`
  - `numpy`, `pandas`, `matplotlib`, `seaborn`
- **Platform**: Jupyter Notebook
- **Hardware Used**: Intel i7, 16GB RAM

---

## 📈 Model Performance (Test Data)

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| KNN                  | 90.38%   | 90.93%    | 90.38% | 90.38%   |
| CatBoost             | 90.86%   | 91.78%    | 90.86% | 90.96%   |
| XGBoost              | 89.50%   | 91.21%    | 89.50% | 89.64%   |
| SVM                  | 87.87%   | 88.33%    | 87.87% | 87.97%   |
| Random Forest        | 85.69%   | 89.83%    | 85.69% | 85.97%   |
| Logistic Regression  | 72.06%   | 72.43%    | 72.06% | 71.87%   |
| Gaussian NB          | 58.67%   | 59.70%    | 58.67% | 54.50%   |
| AdaBoost             | 69.73%   | 69.34%    | 69.73% | 69.47%   |

---

## ✅ Objectives

### 1. Build a Lightweight, Real-Time System
- Suitable for wearable health devices with limited memory and power
- Efficient classification with minimal overhead

### 2. Increase Classification Accuracy
- Detect subtle morphological changes in ECG signals
- Use ensemble models for better performance

### 3. Handle Class Imbalance
- Apply SMOTE for synthetic sampling of minority classes
- Use ENN to eliminate noisy samples

### 4. Improve Model Interpretability
- Replace deep black-box models with transparent ML techniques
- Ensure clinicians can understand model predictions

---

## 💻 How to Use

### 🔧 Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sridharchinthaparthi/ecg-ml-classifier.git
   cd ecg-ml-classifier
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate    # On Windows
   # source venv/bin/activate  # On macOS/Linux
   ```

3. **Upgrade pip and install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the Jupyter Notebook or Python scripts** for training and evaluation.

---

## 🧪 Future Work
- Incorporate advanced denoising methods (e.g., wavelet transform)
- Extract time-frequency and frequency-domain features
- Explore hybrid ML-DL architectures with fewer parameters
- Validate system with real-world wearable device datasets
- Optimize system for edge computing devices

---

## 📝 Note
This project is under active development. Stay tuned for the complete code and enhancements.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact
For questions or collaborations, please reach out via GitHub issues or email.