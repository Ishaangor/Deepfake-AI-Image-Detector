# 🛡️ Universal Deepfake & AI Image Detector

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning and digital forensics platform designed to detect **both classic deepfakes** (FaceSwap, StyleGAN) and **modern text-to-image generative AI** (Google Gemini / Imagen 3, Midjourney, Stable Diffusion, Flux, and DALL-E).

---

## 🌟 Key Features

* **Multi-Generator Detection:** Overcomes traditional dataset domain bias by detecting both GAN-based artifacts and modern Latent Diffusion Model (LDM) synthesis patterns.
* **Dual-Engine Consensus Ensemble:**
  * **Universal XceptionNet:** Fine-tuned convolutional network capturing spatial texture and face manipulation fingerprints (**ROC-AUC: 0.9441**, **Accuracy: 87.40%**).
  * **Swin Vision Transformer (`umm-maybe/AI-image-detector`):** Hierarchical transformer evaluating patch-level attention to catch subtle diffusion smoothing.
* **Automated Face Localization & Smart Crop:** Powered by OpenCV Haar Cascade with 15% margin padding to preserve face geometry without distortion.
* **Forensic Explainability Suite:**
  * **2D Fourier Frequency Spectrum (FFT):** Visualizes power spectrum roll-off and periodic checkerboard anomalies.
  * **Error Level Analysis (ELA):** Highlights JPEG compression discrepancies across image regions.
* **Interactive Web Application:** Built with Streamlit featuring real-time gauge metrics, side-by-side analysis, and threshold customization.
* **Command-Line Interface (CLI):** Fast, standalone evaluation tool for batch workflows.

---

## 📊 Benchmark Results

| Model / Architecture | Test Accuracy | Test ROC-AUC | FAKE Precision | REAL Precision |
| :--- | :---: | :---: | :---: | :---: |
| **Universal XceptionNet** | **87.40%** | **0.9441** | **89.0%** | **85.0%** |
| **Dual-Engine Consensus** | **91.80%** | **0.9620** | **92.5%** | **91.0%** |

### Performance on Challenging Samples
* **Google Gemini (Imagen 3) Faces:** Correctly identified as **FAKE (99.7%–100%)**.
* **Modern Prompt Diffusion Faces:** Correctly identified as **FAKE (95%+ confidence)**.
* **Classic StyleGAN Faces:** Correctly identified as **FAKE (80%–90%)**.
* **Authentic Human Photography:** Correctly identified as **REAL (88%–100%)**.

---

## 🏗️ Project Architecture

```
df_cdac/
├── app.py                     # Streamlit interactive web application
├── predict.py                 # CLI single-image prediction tool
├── train_universal.py         # Universal model training pipeline
├── train_xception.py          # Baseline XceptionNet training script
├── evaluate_model.py          # Model evaluation & metrics benchmark
├── prepare_dataset.py         # Dataset preparation & split script
├── create_final_dataset.py    # Dataset balancing & merge utility
├── models/
│   └── best_xception.keras    # Pre-trained universal detector weights
├── test_images/               # Curated sample validation images
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Ishaangor/Deepfake-AI-Image-Detector.git
cd Deepfake-AI-Image-Detector
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
Then open your browser at `http://localhost:8501`.

### 4. Command-Line Prediction
```bash
python predict.py "path/to/image.jpg"
```

---

## 🔬 Forensic Methodologies Explained

### 1. Error Level Analysis (ELA)
ELA analyzes the rate of lossy compression error across an image. Authentic camera photographs display an even distribution of compression noise. Spliced deepfakes or generated AI patches display noticeable brightness variations in the ELA heatmap.

### 2. 2D Fast Fourier Transform (FFT)
Natural photographs follow a smooth $1/f^\alpha$ radial decay in frequency domain space. Generative models leave artificial periodic grid patterns (GAN checkerboards) or high-frequency spectral roll-offs (Diffusion denoising steps) that are exposed in the 2D power spectrum.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
