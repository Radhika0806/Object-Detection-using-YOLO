# Object Detection using YOLOv8

A real-time object detection application built with Streamlit and YOLOv8. The app detects objects such as license plates in uploaded images and videos, rendering bounding box annotations with confidence scores directly in the browser.

## Overview

This application leverages the YOLOv8 architecture from Ultralytics to perform object detection on user-uploaded media. It supports both image and video inputs, processing each frame to identify and annotate detected objects with labeled bounding boxes and confidence percentages.

## Tech Stack

| Component        | Technology                  |
|------------------|-----------------------------|
| Language         | Python                      |
| Detection Model  | YOLOv8 (Ultralytics)       |
| Computer Vision  | OpenCV                      |
| Web Framework    | Streamlit                   |
| Image Processing | PIL (Pillow), NumPy         |

## Features

- Object detection on static images with supported formats: JPG, PNG, WEBP
- Video processing with frame-by-frame detection for MP4 and MKV files
- Bounding box annotations rendered with per-object confidence scores
- Real-time visual feedback displayed directly in the Streamlit interface
- Supports custom-trained YOLO weights for domain-specific detection tasks (e.g., license plates)
- Automatic GPU utilization when available, with CPU fallback

## Installation

```bash
git clone <repository-url>
cd ObjectDetection
pip install -r requirements.txt
```

Ensure the YOLOv8 model weights file (`best.pt`) is present in the project directory.

## Usage

```bash
streamlit run yolo_application.py
```

The application opens in your default browser. Upload an image or video file through the interface to run detection. Annotated results are displayed directly on the page.

## Project Structure

```
ObjectDetection/
|-- yolo_application.py    # Streamlit application entry point
|-- best.pt                # Trained YOLOv8 model weights
|-- requirements.txt       # Python dependencies
|-- temp/                  # Temporary uploaded file storage
|-- output/                # Processed output files
```

## Screenshots

| Upload Interface | Detection Result |
|-----------------|------------------|
| ![Upload](screenshots/upload.png) | ![Detection](screenshots/detection.png) |

> **To add screenshots:** Run `streamlit run yolo_application.py`, upload an image, and save screenshots of the interface and detection output in the `screenshots/` folder.

## How It Works

1. User uploads an image or video through the Streamlit interface
2. The uploaded file is temporarily saved and passed to the YOLOv8 model
3. For images: objects are detected, bounding boxes with confidence scores are drawn, and the annotated image is displayed
4. For videos: each frame is processed individually, annotated, and assembled into an output video
5. All results are rendered directly within the Streamlit application

## Supported Formats

| Type  | Formats         |
|-------|-----------------|
| Image | JPG, PNG, WEBP  |
| Video | MP4, MKV        |
