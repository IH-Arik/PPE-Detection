# Construction Site PPE & Hazard Detection

A multi-platform AI solution for real-time monitoring of Personal Protective Equipment (PPE) and safety hazards on construction sites.

## 🏗 Project Structure

This repository is organized into two primary applications and shared resources:

### 📱 [Mobile App](./mobile-app)
- **Tech Stack**: React Native, Expo, TFLite (GPU Accelerated), Vision Camera.
- **Features**: Real-time camera feed, high-performance detection overlay, stats tracking (FPS, Latency, RAM).
- **Setup**: `npm install` -> `npx expo prebuild` -> `npx expo run:android`

### 🌐 [Web App](./web-app)
- **Tech Stack**: Streamlit, YOLO.
- **Features**: Image/Video upload processing, webcam support, detailed detection summaries.
- **Setup**: `pip install -r requirements.txt` -> `streamlit run app.py`

### 📂 [Shared Assets](./assets)
- Trained models (`.tflite`, `.pt`, `.onnx`).
- Configuration files and label mappings.
- Sample calibration and test data.

### 📜 [Scripts](./scripts)
- Post-processing and model conversion utilities.
- Environment setup and inspection tools.

## 🚀 Getting Started

1. Clone the repository.
2. Follow the setup instructions in each sub-directory's README.

## 🛠 Features

- **Real-time Performance**: Runs inference at 24+ FPS on modern mobile hardware using GPU delegation.
- **PPE Classes**: Detects boots, helmets, vests, gloves, and glasses (or their absence).
- **Safety Diagnostics**: Integrated monitoring for system latency and memory usage.

---
Powered by [YOLO](https://github.com/ultralytics/ultralytics) and [Fast TFLite](https://github.com/mrousavy/react-native-fast-tflite).
