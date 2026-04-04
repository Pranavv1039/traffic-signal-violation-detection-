# 🚦 Traffic Signal Violation Detection System

**Capstone Project by Pranav Gedela**

## Overview
AI-powered traffic violation detection system that analyzes traffic camera feeds to detect Red Light Jumping, Helmetless Riding, and reads Number Plates automatically.

## Features
- 🔴 Red Light Violation Detection
- ⛑️ Helmetless Rider Detection
- 🔢 Automatic Number Plate Recognition
- 📊 Real-time Enforcement Dashboard
- 📸 Evidence Screenshot Saving
- 📝 CSV Violation Logging
- ⚡ GPU Accelerated

## Tech Stack
- YOLOv8 — Vehicle and Helmet Detection
- EasyOCR — License Plate Reading
- OpenCV — Video Processing
- Streamlit + Plotly — Dashboard
- PyTorch + CUDA — Deep Learning

## Usage

### Red Light Detection
python main.py

### Helmet Detection on Images
python helmet_images.py

### Helmet Detection on Webcam
python helmet_main.py

### Dashboard
streamlit run dashboard.py

## Helmet Model Performance
| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| License Plate | 0.997 | 1.000 | 0.995 |
| Helmet | 0.939 | 0.941 | 0.946 |
| No Helmet | 0.793 | 0.800 | 0.893 |

## References
1. FarzadNekouee/Traffic-Violation-Detection
2. rumbleFTW/smart-traffic-monitor (ICCCI 2023)
3. ankandrew/fast-alpr
4. Roboflow Dataset - helmet-lincense-plate-detection

## Author
Pranav Gedela
GitHub: https://github.com/Pranavv1039
