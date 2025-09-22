# predict_video.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

IMG_SIZE = 224
SEQUENCE_LENGTH = 60
THRESHOLD = 0.5

# Load models
video_model = load_model("model/mobilenet_lstm_improved.h5")
image_model = load_model("model/mobilenet_image_model.h5")

def extract_frames(video_path, sequence_length=SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if frame_count >= sequence_length:
        interval = frame_count // sequence_length
        for i in range(sequence_length):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            success, frame = cap.read()
            if success:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
    else:
        for _ in range(frame_count):
            success, frame = cap.read()
            if success:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        black_frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        while len(frames) < sequence_length:
            frames.append(black_frame)

    cap.release()
    return np.array(frames)

def predict_video(video_path):
    frames = extract_frames(video_path)
    frames = preprocess_input(frames)
    input_sequence = np.expand_dims(frames, axis=0)  # (1, 60, 224, 224, 3)

    prediction = video_model.predict(input_sequence)[0][0]
    label = "REAL" if prediction >= THRESHOLD else "FAKE"
    confidence = round(float(prediction), 4)

    print(f"[Video] Prediction: {label} (confidence: {confidence})")
    return {"label": label, "confidence": confidence}

THRESHOLD = 0.5

def predict_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image could not be read")

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img.astype("float32"))
        input_array = np.expand_dims(img, axis=0)

        prediction = image_model.predict(input_array)[0][0]

        if prediction >= THRESHOLD:
            label = "REAL"
            confidence = prediction
        else:
            label = "FAKE"
            confidence = 1 - prediction

        confidence = round(float(confidence), 4)

        print(f"[Image] Prediction: {label} (confidence: {confidence})")
        return {"label": label, "confidence": confidence}

    except Exception as e:
        print(f"❌ Error in image prediction: {e}")
        return {"label": "Error", "confidence": 0.0}
