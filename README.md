# Deep Learning Image Prediction - Nike vs Adidas Classification

A deep learning project that classifies shoe images as Nike or Adidas using a CNN built with TensorFlow/Keras.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## 🎯 Overview

Binary image classification using deep learning to distinguish between Nike and Adidas shoe images. Implemented as a Jupyter notebook compatible with Google Colab.

## ✨ Features

- **Binary Image Classification**: Automatically classifies images as Nike or Adidas
- **High Accuracy**: ~100% training accuracy by epoch 12
- **Automatic Data Preprocessing**: Organizes and normalizes images
- **Google Colab Compatible**: Easy cloud-based execution with Google Drive integration
- **Confidence Scoring**: Shows prediction confidence percentages

## 📊 Dataset

- **Training Set**: 100 images (50 Nike + 50 Adidas)
- **Test Set**: 40 images (20 Nike + 20 Adidas)
- **Image Size**: 224×224 pixels
- **Batch Size**: 32

## 🏗️ Model Architecture

```
Input (224×224×3)
  ↓
Conv2D (32) + ReLU → MaxPooling2D
  ↓
Conv2D (64) + ReLU → MaxPooling2D
  ↓
Conv2D (128) + ReLU → MaxPooling2D
  ↓
Flatten → Dense (128) + ReLU → Dense (2) + Softmax
  ↓
Output (Nike/Adidas probability)
```

**Specs**: Adam optimizer | Sparse Categorical Crossentropy | 20 epochs

## 🚀 Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib

### Setup

```bash
pip install tensorflow numpy matplotlib
```

**For Google Colab:**
1. Upload notebook to Colab
2. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Place dataset in: `/MyDrive/NIKE_vs_ADIDAS-master/`

## 📖 Usage

1. **Mount Drive** (Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Run Notebook**: Execute cells sequentially
   - Automatically organizes images into class folders
   - Trains model for 20 epochs
   - Generates predictions on test images

3. **Make Predictions**:
   ```python
   predictions = model.predict(img_array)
   confidence = 100 * np.max(score)
   ```

## 📈 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~100% |
| Sample Predictions | 73%+ confidence |
| Best Performance | Epoch 12-20 |

## 🛠️ Technologies

- TensorFlow/Keras
- Python
- NumPy
- Matplotlib
- Google Colab

## 📁 Files

- `Deep_Learning_Image_detection.ipynb` - Main notebook
- `README.md` - This file

## ⚙️ Configuration

```python
IMG_SIZE = 224
batch_size = 32
epochs = 20
data_dir = "/content/drive/MyDrive/NIKE_vs_ADIDAS-master"
```

## ✍️ Author

**Shrinivas Suresh Hegde**

---

**Note**: Designed for Google Colab with GPU access. For local execution, ensure sufficient computational resources.
