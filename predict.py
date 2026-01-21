# predict.py
import cv2
import numpy as np
import tensorflow as tf
import os

IMG_SIZE = 224
SEQUENCE_LENGTH = 60
MODEL_PATH = "model/mobilenet_lstm_sequence.h5"
IMAGE_MODEL_PATH = "model/mobilenet_image_model.h5"  # optional if you trained separate image model

video_model = tf.keras.models.load_model(MODEL_PATH)
# image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)  # uncomment if available

def extract_frames_for_prediction(video_path, seq_len=SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if frame_count <= 0:
        cap.release()
        raise ValueError("No frames found")
    interval = max(1, frame_count // seq_len)
    for i in range(0, frame_count, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if len(frames) >= seq_len:
            break
    cap.release()
    while len(frames) < seq_len:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
    arr = np.array(frames).astype("float32") / 255.0
    return arr

def predict_video(video_path, threshold=0.5):
    seq = extract_frames_for_prediction(video_path)
    input_seq = np.expand_dims(seq, axis=0)
    preds = video_model.predict(input_seq)
    score = float(np.mean(preds))
    label = "fake" if score >= threshold else "real"
    return {"label": label, "score": score}

def predict_image(image_path, threshold=0.5):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image read error")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = np.expand_dims(img.astype("float32")/255.0, axis=0)
    # if using image_model:
    # score = float(image_model.predict(arr)[0][0])
    # label = "fake" if score >= threshold else "real"
    # else fallback to using video model on repeated frames:
    seq = np.repeat(arr, SEQUENCE_LENGTH, axis=0)  # shape (60, H, W, C)
    seq = np.reshape(seq, (1, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    preds = video_model.predict(seq)
    score = float(np.mean(preds))
    label = "fake" if score >= threshold else "real"
    return {"label": label, "score": score}

if __name__ == "__main__":
    print(predict_video("uploads/test_video.mp4"))
    print(predict_image("uploads/test_image.jpg"))
