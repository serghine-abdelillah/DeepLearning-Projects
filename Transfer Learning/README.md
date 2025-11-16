# 🫁 COVID-19 Lung Segmentation using CNN
This project focuses on implementing a Convolutional Neural Network (CNN) for **medical image segmentation** of **COVID-19 chest X-ray scans**. The goal is to automatically segment infected lung regions, helping assist clinical decision support systems during the pandemic.
## 📌 Project Overview
- Use chest X-ray images and corresponding mask annotations
- Build and train a deep learning segmentation model
- Evaluate the model using standard segmentation metrics
- Visualize predictions vs. ground-truth masks
## 🧠 Model Architecture
A custom **U-Net-style CNN** segmentation model was developed using:
- **Convolutional Layers**
- **MaxPooling & Upsampling**
- **Skip Connections**
- **Sigmoid Output** for binary mask prediction
## 🗂 Dataset
- COVID-19 Radiography Dataset
- Contains:
  - X-ray Images (COVID cases)
  - Ground-truth segmentation masks

### Please Note that the Dataset is not included in the repo so please download if from the link available in the notebook.

## 🧪 Evaluation Metrics
The segmentation performance was assessed using:
| Metric                            | Description                                       | Values         |
| --------------------------------- | ------------------------------------------------- | -------------- |
| **Dice Coefficient**              | Measures overlap between predicted and true masks |  **95%**           |
| **IoU (Intersection over Union)** | Ratio of common area vs. union area               |  **90%**           |
| **Pixel-wise Accuracy**           | Correctly segmented pixels percentage             |  **98%**           |

## 🛠 Tools & Technologies
- **Python**
- **TensorFlow / Keras**
- **NumPy, Matplotlib**
- **scikit-learn**

  <!--- ======== Consider TO add the running section  --->

## 📊 Results Visualization
The notebook displays:

✅ Original Image
✅ Ground-truth Mask
✅ Predicted Mask

Allowing qualitative evaluation of segmentation success.
