# Liver Disease Detection from Ultrasound Images Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)

> **⚠️ Educational Notice:** This application is developed for academic and research demonstration only. It is not a medical device and must **NOT** be used for diagnosis, treatment, or clinical decision-making.

---

## 📌 Project Overview & Objective

This college / mini-project provides an end-to-end, transparent, and reproducible **Traditional Machine Learning Pipeline** for classifying B-mode liver ultrasound images into pathological classes (such as **Normal**, **Fatty Liver**, and **Cirrhosis**).

Unlike black-box Deep Learning (CNNs), this project highlights classical Computer Vision and statistical pattern recognition techniques:
- **Image Preprocessing:** Grayscale conversion, Gaussian spatial denoising, and Histogram Equalization.
- **Feature Engineering:** Directional gradients via **HOG (Histogram of Oriented Gradients)** and second-order parenchymal texture statistics via **GLCM (Gray-Level Co-occurrence Matrix)**.
- **Classifiers:** Support Vector Machines (**SVM** with RBF kernel), **Random Forest**, and **K-Nearest Neighbors (KNN)** wrapped in standard scaling pipelines.

---

## 🏗️ Architecture & Workflow

```text
                        ULTRASOUND IMAGE
                               ↓
                   [ STEP 1: PREPROCESSING ]
                 Resize (128x128) → Grayscale
            → Gaussian Blur → Histogram Equalization
                      → Pixel Normalization
                               ↓
                 [ STEP 2: FEATURE EXTRACTION ]
             ┌─────────────────┴─────────────────┐
             ↓                                   ↓
    HOG Feature Vector                  GLCM Texture Matrix
  (8,100 Gradient Bins)                (60 Texture Metrics)
             └─────────────────┬─────────────────┘
                               ↓
                    COMBINED FEATURE VECTOR
                     (8,160 Feature Values)
                               ↓
                  [ STEP 3: MACHINE LEARNING ]
             ┌─────────────────┼─────────────────┐
             ↓                 ↓                 ↓
      SVM (RBF Kernel)    Random Forest         KNN
             └─────────────────┼─────────────────┘
                               ↓
                     [ STEP 4: PREDICTION ]
            Predicted Dataset Class + Confidence Estimate
                               ↓
                   [ STEP 5: EVALUATION ]
            Accuracy | Precision | Recall | F1-Score
             Confusion Matrix | Classification Report
```

---

## 🛠️ Technology Stack

| Module | Technologies Used | Purpose |
| :--- | :--- | :--- |
| **User Interface** | [Streamlit](https://streamlit.io/) | Interactive multi-tab web dashboard |
| **Image Processing** | [OpenCV](https://opencv.org/), [Pillow](https://python-pillow.org/), [NumPy](https://numpy.org/) | Multi-stage image preprocessing pipeline |
| **Feature Extraction** | [Scikit-Image](https://scikit-image.org/) | HOG gradient extraction & GLCM texture analysis |
| **Machine Learning** | [Scikit-Learn](https://scikit-learn.org/) | StandardScaler, SVM, Random Forest, KNN, Metrics |
| **Data Processing** | [Pandas](https://pandas.pydata.org/) | Tabular representation and metrics structuring |
| **Visualizations** | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) | Heatmaps, bar charts, and feature plots |
| **Model Persistence** | [Joblib](https://joblib.readthedocs.io/) | Model pipeline serialization (`.pkl`) |

---

## 📁 Directory Structure

```text
Liver Disease Detection from Ultrasound Images/
│
├── app.py                     # Main Streamlit Web Application (7 Interactive Tabs)
├── preprocessing.py           # 5-Stage Image Preprocessing Pipeline
├── feature_extraction.py      # HOG + GLCM Texture Extraction
├── model_trainer.py           # ML Model Training, Comparison, & Evaluation
├── dataset_utils.py           # Dynamic Dataset Discovery & Feature Batching
├── ui_components.py           # Custom CSS, Metric Cards, Heatmap & Plot Helpers
├── requirements.txt           # Python Package Dependencies
├── run_app.bat                # Windows Batch launcher script
├── run_app.ps1                # PowerShell launcher script
├── dataset/                   # Dataset root (auto-discovers class folders)
│   ├── Normal/                # Normal liver ultrasound images
│   ├── Fatty_Liver/           # Steatosis / fatty liver ultrasound images
│   └── Cirrhosis/             # Cirrhotic liver ultrasound images
└── models/                    # Saved model weights & configuration
    └── liver_ultrasound_model.pkl
```

---

## 🚀 Getting Started

### 1. Installation

Ensure you have Python 3.10 or higher installed:

```bash
pip install -r requirements.txt
```

### 2. Running the Application

Launch the Streamlit web dashboard:

```bash
python -m streamlit run app.py
```

Or on Windows, simply double-click `run_app.bat`.

The application will open in your web browser at `http://localhost:8501`.

---

## 🖥️ Application Features & Tabs

### 🏠 Tab 1: Overview
- Complete project objectives and academic context.
- Technology stack overview and mathematical foundation of HOG and GLCM.

### 📁 Tab 2: Dataset Management
- Dynamic dataset directory loader. Automatically detects all class folders inside `dataset/`.
- Real-time class distribution bar charts and per-class sample image gallery preview.
- Built-in **Generate Sample Educational Dataset** button to create realistic synthetic ultrasound textures for instant testing.

### 🖼️ Tab 3: Preprocessing Pipeline
- Interactive demonstration of the 5-stage transformation:
  1. **Original RGB** Ultrasound Scan
  2. **Resize** to standard $(128 \times 128)$
  3. **Grayscale Conversion**
  4. **Gaussian Denoising** ($5 \times 5$ kernel)
  5. **Histogram Equalization** (contrast enhancement)
  6. **Pixel Normalization** to $[0.0, 1.0]$
- Side-by-side visual comparison and pixel intensity distribution histograms.

### 🔍 Tab 4: Feature Extraction
- **HOG Gradients:** Visualizes vector gradients ($9$ orientation bins, $8 \times 8$ pixel cells, $2 \times 2$ blocks).
- **GLCM Texture Metrics:** Computes Contrast, Dissimilarity, Homogeneity, Energy, and Correlation across multiple distances and angles.
- Displays first 20 features of the combined $8,160$-element vector.

### 🤖 Tab 5: Model Training
- One-click batch dataset feature extraction with progress indicator.
- Configurable Train/Test split (e.g. 80/20) with automated stratification fallback.
- Algorithm selection: **SVM (RBF kernel)**, **Random Forest**, or **KNN**.
- Auto-saves the trained `StandardScaler` pipeline to `models/liver_ultrasound_model.pkl`.

### 📊 Tab 6: Evaluation & Comparison
- Dynamically calculated metric cards: **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
- High-resolution **Confusion Matrix Heatmap** labeled with actual dataset class names.
- Complete per-class **Classification Report** table.
- **Compare All Models:** Simultaneously trains SVM, Random Forest, and KNN on the identical split and produces a comparative performance table and bar chart.

### 🔮 Tab 7: Prediction & Inference
- Upload new ultrasound images (`JPG`, `JPEG`, `PNG`, `BMP`, `TIFF`) or select test samples.
- Visual execution across preprocessing, feature extraction, and ML classification.
- Prominent **Prediction Result Card** displaying Predicted Dataset Class, Algorithm, and Model Confidence Estimate percentage.

---

## 📜 Academic Integrity & Disclaimer

This project is created strictly for academic mini-project demonstration. All metrics are calculated live from the provided dataset without hardcoding. Prediction outputs are presented as **"Predicted Dataset Class"** and model confidence estimates, with explicit clarification that the system does not perform medical diagnosis.
