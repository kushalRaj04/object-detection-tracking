# Real-Time Object Detection and Tracking

## Overview
A real-time computer vision application that detects and tracks objects from a webcam feed using YOLO, Deep SORT, and OpenCV.

## Features
- Real-time webcam video processing
- Object detection using a pretrained YOLO model
- Object tracking with Deep SORT
- Bounding boxes and persistent tracking IDs
- Flask backend with live video streaming
- Web dashboard displaying tracking statistics

## Technologies
- Python
- OpenCV
- Ultralytics YOLO
- Deep SORT
- Flask
- HTML, CSS, JavaScript

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/object-detection-tracking.git
   cd object-detection-tracking
   ```

2. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install the required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

4. Run the application:
   ```powershell
   python app.py
   ```

5. Open the local address displayed in the terminal in your browser.

## Notes
- A working webcam is required for webcam-based detection.
- The YOLO model weights are not included in this repository and may be downloaded automatically by Ultralytics.
- Model predictions depend on the pretrained model's supported classes and confidence threshold.