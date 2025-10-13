# 🛡️ Malicious URL Detection

This project is a deep learning–powered web application that detects whether a given URL is benign or malicious using a deep learning model (LSTM).
It analyzes URL patterns and structural features to help users identify phishing or harmful websites before visiting them.

## 🚀 Features
- 🔍 Real-time URL classification using a trained LSTM model
- 🧠 Extracts more than 20 URL-based features such as length, special characters, and domain patterns
- 🌐 Built with Gradio for an intuitive web interface
- 🧾 Predicts with clear labels:
- - ✅ Benign — Safe to browse
- - ⚠️ Malicious — Potentially harmful

## 🧩 How It Works
- The model analyzes URLs based on several features, including:
- Presence of IP address in URL
- Use of shortening services (e.g., bit.ly, tinyurl)
- Number of dots, slashes, or special symbols
- Suspicious keywords (e.g., login, paypal, bank, free)
- URL and domain lengths

These extracted features are then passed into a trained LSTM neural network, which outputs a binary classification.
## 🛠️ Tech Stack
| Component                  | Description                  |
| -------------------------- | ---------------------------- |
| **Python**                 | Core language                |
| **TensorFlow / Keras**     | Model training and inference |
| **Gradio**                 | Web UI interface             |
| **tldextract, urllib, re** | URL feature extraction       |
| **LSTM (Deep Learning)**   | Detection model architecture |

## 📦 Installation
```bash

git clone https://github.com/serghine-abdelillah/DeepLearning-Projects.git
```
```bash
cd Malicious URL Detection
```
```bash
cd Malicious_URL_Detector
```
```bash
pip install -r requirements.txt
```
