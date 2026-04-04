[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vaniatharv/AccidentLens/blob/main/Accident_detection.ipynb)
AccidentLens is a deep learning system that automatically detects road accidents in video footage. It combines **EfficientNetB0** (for spatial feature extraction per frame) with an **LSTM network** (for temporal pattern analysis across frames) to classify video clips as either **Accident** or **Normal Driving**.
---
## Table of Contents
- [Mission](#mission)
- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline](#pipeline)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Training](#model-training)
- [Prediction](#prediction)
---
## Mission
AccidentLens is driven by a commitment to **saving lives through intelligent automation**. Road accidents claim millions of lives every year, and delayed detection is a key factor in poor emergency response outcomes. Our mission is to harness the power of deep learning to enable **real-time, automated accident detection** from video footage — reducing response times, supporting traffic management systems, and ultimately making roads safer for everyone.

By combining state-of-the-art computer vision with temporal sequence modelling, AccidentLens aims to provide an accessible, accurate, and scalable tool that can be integrated into existing surveillance infrastructure with minimal friction.

---
## Overview
Road accident detection is a critical component of intelligent transportation systems and public safety. AccidentLens automates this process by analyzing surveillance or dashcam video using a two-stage deep learning pipeline:
1. **Spatial Feature Extraction** — EfficientNetB0 (pretrained on ImageNet) processes individual frames and produces compact feature vectors.
2. **Temporal Classification** — A stacked LSTM network learns the temporal dynamics across frame sequences and outputs a binary prediction.
---
## Architecture
```
Video Input
    │
    ▼
Frame Extraction (every 10th frame)
    │
    ▼
EfficientNetB0 (ImageNet weights, no top, average pooling)
    │  → 1280-dimensional feature vector per frame
    ▼
Sequence Padding (pad_sequences)
    │
    ▼
LSTM (128 units, return_sequences=True)  → Dropout(0.2)
    │
LSTM (64 units)  → Dropout(0.2)
    │
Dense (1, sigmoid)
    │
    ▼
Binary Classification: Accident (1) / Normal Driving (0)
```
---
## Pipeline
### 1. Frame Extraction
Videos are sampled at every 10th frame using OpenCV. Extracted frames are saved as JPEG images, organised by label (`accident` / `normal`).
### 2. Feature Extraction
Each frame is resized to **224×224**, preprocessed, and passed through **EfficientNetB0** (with global average pooling) to produce a **1280-dimensional** feature vector.
### 3. Dataset Preparation
- Feature sequences for each video are padded to a uniform length using `pad_sequences`.
- Labels are encoded as binary values: `1` for accident, `0` for normal.
- An 80/20 stratified train/test split is applied.
### 4. Model Training
The LSTM model is trained for **10 epochs** with a batch size of **8**, using:
- Optimizer: Adam (lr = 0.001)
- Loss: Binary Cross-Entropy
- Metric: Accuracy
- Validation split: 20% of training data
### 5. Prediction
A trained model is loaded and used to classify new video clips. The pipeline extracts frames, computes features, pads the sequence, and outputs a label with a confidence score.
---
## Requirements
Install all dependencies before running the notebook:
```bash
pip install tensorflow keras opencv-python tqdm
```
| Package | Purpose |
|---|---|
| `tensorflow` / `keras` | Model building and training |
| `opencv-python` | Video frame extraction |
| `tqdm` | Progress bars |
| `scikit-learn` | Train/test splitting |
| `numpy` / `pandas` | Data handling |
> **Note:** The notebook is designed to run on **Google Colab** with files stored in **Google Drive**.
---
## Dataset
The dataset should be organised in Google Drive as follows:
```
MyDrive/
├── train.csv                  # CSV with columns: filename, case (accident/normal)
└── train/
    └── training_videos/       # Raw .mp4 video files
```
The `train.csv` file must contain at minimum:
- `filename` — the video file name (e.g., `accident_0001.mp4`)
- `case` — the label (`accident` or `normal`)
---
## Usage
1. Open the notebook in Google Colab using the badge at the top.
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Ensure `train.csv` and training videos are in the expected Drive paths.
4. Run all cells sequentially to:
   - Extract frames from videos
   - Extract CNN features from frames
   - Prepare the dataset
   - Train and evaluate the LSTM model
   - Save the trained model to Drive
---
## Model Training
After dataset preparation, the LSTM model is defined and compiled:
```python
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(timesteps, 1280)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
```
The trained model is saved as:
```
/content/drive/MyDrive/model.h5
```
---
## Prediction
To classify a new video:
```python
video_path = "/content/drive/MyDrive/train/training_videos/accident_0020.mp4"
predict_video(video_path)
# Output: Prediction: Accident (Confidence: 0.87)
```
The `predict_video` function:
1. Extracts frames from the video (every 10th frame).
2. Passes each frame through EfficientNetB0 to get features.
3. Pads the feature sequence to the required input length.
4. Returns `"Accident"` if the model confidence > 0.5, otherwise `"Normal Driving"`.
---
## License
This project is open source. Feel free to use and adapt it for research or safety applications.
