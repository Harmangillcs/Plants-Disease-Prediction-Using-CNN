# Plant Disease Prediction using CNN (Deep Learning Project)

This project implements a **Convolutional Neural Network (CNN)** to classify plant diseases from images. The model predicts the disease type or identifies healthy plants using a trained deep learning model.

---

## Dataset

The dataset used for this project comes from **Kaggle**:

[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

---

## Features

- Upload an image of a plant leaf.
- Predict the disease type or healthy status using a CNN model.
- Built using **TensorFlow**, **Keras**, and **Streamlit**.
- Model weights are stored on Hugging Face Hub and loaded at runtime.
- Lightweight repository (model not included in the repo).

---

## Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/Harmangillcs/Plants-Disease-Prediction-Using-CNN.git
cd Plants-Disease-Prediction-Using-CNN
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
# or
source .venv/bin/activate      # Mac/Linux
```

3.Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a .env file in the project root and add your Hugging Face token:
```bash
HF_TOKEN=your_huggingface_token
```
5.Run the Streamlit app:
```bash
streamlit run main.py
```
## Live Demo

You can try the deployed app here: [Plants Disease Classifier](https://plants-disease-prediction-using-cnn-3.onrender.com/)

---


