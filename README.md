# 🚦 Traffic Signal Violation Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?style=for-the-badge&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?style=for-the-badge&logo=nvidia)

### An AI-powered traffic violation detection system using YOLOv8, EasyOCR, and Streamlit

**Capstone Project by Pranav Gedela**

</div>

---

## 📌 Overview

This project implements an automated traffic violation detection system that analyzes traffic camera feeds in real-time to detect traffic violations, read number plates, and auto-flag violations for enforcement authorities.

### Problem Statement
Manual traffic monitoring is inefficient and error-prone. This system automates the detection of:
- Vehicles jumping red lights at intersections
- Motorcycle riders not wearing helmets
- Automatic identification of violators via license plate recognition

---

## 🎯 Key Features

| Feature | Description | Technology |
|---------|-------------|------------|
| 🔴 Red Light Detection | Detects traffic light color using HSV | OpenCV |
| 🛑 Stop Line Detection | Adaptive stop line using Hough Transform | OpenCV |
| 🚗 Vehicle Tracking | Multi-object tracking across frames | YOLOv8 + ByteTrack |
| ⛑️ Helmet Detection | Detects helmet/no-helmet on riders | Custom YOLOv8 |
| 🔢 Plate Recognition | Reads license plates of violators | EasyOCR (GPU) |
| 📸 Evidence Saving | Auto-saves screenshots of violations | OpenCV |
| 📝 CSV Logging | Timestamped violation records | Pandas |
| 📊 Dashboard | Real-time enforcement dashboard | Streamlit + Plotly |
| ⚡ GPU Support | CUDA accelerated inference | PyTorch + CUDA |
| 🧵 Threading | Non-blocking OCR processing | Python Threading |

---

## 🏗️ System Architecture

**Step 1:** Traffic Camera Feed is processed frame by frame

**Step 2:** YOLOv8 detects and tracks all vehicles in the frame

**Step 3A — Red Light Module:**
- Traffic light color detected using HSV color space
- Stop line detected using Hough Transform
- If vehicle crosses stop line during RED → Violation triggered
- License plate read using EasyOCR

**Step 3B — Helmet Module:**
- YOLOv8 detects helmet / no-helmet on riders
- License plate bounding box detected simultaneously
- If no-helmet detected → Violation triggered
- License plate read using EasyOCR

**Step 4:** Violation Logger saves screenshot + CSV record

**Step 5:** Streamlit Dashboard displays all violations in real-time

---

## 🛠️ Tech Stack

- **Object Detection** — Ultralytics YOLOv8
- **OCR** — EasyOCR with GPU support
- **Computer Vision** — OpenCV 4.8
- **Dashboard** — Streamlit + Plotly
- **Deep Learning** — PyTorch + CUDA 11.8
- **Language** — Python 3.10

---

## 📂 Project Structure
traffic-signal-violation-detection/
│
├── detectors/
│   ├── init.py
│   ├── helmet.py
│   ├── plate.py
│   └── red_light.py
│
├── Traffic-Violation-Detection/
│   ├── traffic_video.mp4
│   └── haarcascade_russian_plate_number.xml
│
├── smart-traffic-monitor/
│
├── dashboard.py
├── main.py
├── helmet_images.py
├── helmet_main.py
├── violation_logger.py
├── violations_log.csv
├── helmet_violations_log.csv
└── requirements.txt

---

## ⚙️ Installation

### Prerequisites
- Python 3.10
- NVIDIA GPU with CUDA (recommended)
- Tesseract OCR

### 1. Clone the repository
```bash
git clone https://github.com/Pranavv1039/traffic-signal-violation-detection-.git
cd traffic-signal-violation-detection-
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install ultralytics
pip install easyocr
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install streamlit plotly pandas
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install Tesseract OCR
Download from: https://github.com/UB-Mannheim/tesseract/wiki

Install to default path: `C:\Program Files\Tesseract-OCR\`

### 5. Train Helmet Model
```bash
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(
    data='helmet-lincense-plate-detection-24/data.yaml',
    epochs=15,
    imgsz=416,
    device=0,
    name='helmet_model4'
)
"
```

---

## 🚀 Usage

### Module 1 — Red Light Violation Detection
```bash
python main.py
```
- Processes traffic video
- Detects red light violations
- Reads license plates
- Saves evidence screenshots

### Module 2 — Helmet Violation Detection on Images
```bash
python helmet_images.py
```
- Processes images from media folder
- Detects helmetless riders
- Reads license plates

### Module 3 — Helmet Detection on Webcam
```bash
python helmet_main.py
```
- Real-time helmet detection using webcam

### Module 4 — Enforcement Dashboard
```bash
streamlit run dashboard.py
```
- Opens at http://localhost:8501
- Shows all violations with charts
- Evidence screenshot viewer
- Downloadable CSV reports

---

## 📊 Results

### Helmet Detection Model Performance
| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| License Plate | 0.997 | 1.000 | 0.995 | 0.866 |
| Helmet | 0.939 | 0.941 | 0.946 | 0.701 |
| No Helmet | 0.793 | 0.800 | 0.893 | 0.603 |
| **Overall** | **0.910** | **0.914** | **0.945** | **0.723** |

### System Performance
| Module | Speed | Accuracy |
|--------|-------|----------|
| Vehicle Detection | ~15-20 FPS | YOLOv8n |
| Red Light Detection | Real-time | HSV Color |
| Helmet Detection | ~10-15 FPS | mAP50: 0.945 |
| Plate Reading | ~0.5s/plate | EasyOCR GPU |

---

## 📚 References

1. **FarzadNekouee/Traffic-Violation-Detection** — Red light detection logic and stop line detection using classical image processing
2. **rumbleFTW/smart-traffic-monitor** — Helmet violation pipeline. Published at IEEE ICCCI 2023, Best Paper Award
3. **ankandrew/fast-alpr** — Fast Automatic License Plate Recognition framework
4. **meryemsakin/helmet-detection-yolov8** — YOLOv8 helmet detection reference
5. **ThanhSan97/Helmet-Violation-Detection-Using-YOLO-and-VGG16** — 3-model pipeline architecture reference
6. **Roboflow Dataset** — helmet-lincense-plate-detection by cdio-zmfmj (3702 images, 3 classes)

---

## 👨‍💻 Author

**Pranav Gedela**

- GitHub: [@Pranavv1039](https://github.com/Pranavv1039)

---

## 📄 License

This project is developed for educational purposes as part of a Capstone Project.

---

<div align="center">

🚦 Built with ❤️ using YOLOv8 + EasyOCR + Streamlit

</div>
