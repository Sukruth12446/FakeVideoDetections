import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# -----------------------------
# CONSTANTS
# -----------------------------
IMG_SIZE = 224
SEQUENCE_LENGTH = 60
VIDEO_THRESHOLD = 0.52
IMAGE_THRESHOLD = 0.69  # tuned threshold for CNN

# Load pretrained models
video_model = load_model("model/mobilenet_lstm_bilstm_v2.h5")
image_model = load_model("model/mobilenet_image_model.h5")

# Load Haarcascade and fallback DNN detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# DNN detector (more reliable if Haar fails)
proto_path = cv2.data.haarcascades.replace("haarcascades/", "") + "deploy.prototxt"
model_path = cv2.data.haarcascades.replace("haarcascades/", "") + "res10_300x300_ssd_iter_140000.caffemodel"
if os.path.exists(proto_path) and os.path.exists(model_path):
    dnn_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
else:
    dnn_net = None

# -----------------------------
# FRAME EXTRACTION (VIDEO)
# -----------------------------
def extract_frames(video_path, sequence_length=SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if frame_count <= 0:
        cap.release()
        raise ValueError("No frames found in video.")

    interval = max(1, frame_count // sequence_length)
    for i in range(0, frame_count, interval):
        if len(frames) >= sequence_length:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = cap.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = frame[y:y + h, x:x + w]
        else:
            face = frame

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        frames.append(face)

    black_frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    while len(frames) < sequence_length:
        frames.append(black_frame)

    cap.release()
    return np.array(frames[:sequence_length])

# -----------------------------
# ADAPTIVE THRESHOLD
# -----------------------------
def get_adaptive_threshold(predictions, mode="video"):
    avg_pred = np.mean(predictions)
    if mode == "video":
        if avg_pred > 0.8:
            return 0.55
        elif avg_pred < 0.3:
            return 0.45
        else:
            return VIDEO_THRESHOLD
    else:  # image mode
        if avg_pred > 0.85:
            return 0.55
        elif avg_pred < 0.35:
            return 0.35
        else:
            return IMAGE_THRESHOLD

# -----------------------------
# VIDEO PREDICTION
# -----------------------------
def predict_video(video_path):
    try:
        frames = extract_frames(video_path)
        frames = frames.astype("float32") / 255.0  # normalize
        input_sequence = np.expand_dims(frames, axis=0)

        preds = video_model.predict(input_sequence)
        prediction = np.mean(preds)

        dynamic_thresh = get_adaptive_threshold(preds.flatten(), mode="video")
        label = "REAL" if prediction >= dynamic_thresh else "FAKE"
        confidence = round(float(abs(prediction - 0.5) * 2 * 100), 2)

        print(f"[Video] Prediction: {label} (Confidence: {confidence}%)")
        return {"label": label, "confidence": confidence}

    except Exception as e:
        print(f"❌ Error in video prediction: {e}")
        return {"label": "Error", "confidence": 0.0}

# -----------------------------
# IMAGE PREDICTION
# -----------------------------
def detect_face_dnn(img):
    """Use DNN face detector as fallback."""
    if dnn_net is None:
        return None
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    dnn_net.setInput(blob)
    detections = dnn_net.forward()
    if detections.shape[2] > 0:
        confidence = detections[0, 0, 0, 2]
        if confidence > 0.5:
            box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            return [x1, y1, x2 - x1, y2 - y1]
    return None

def normalize_brightness(image):
    """Normalize brightness to avoid lighting bias."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def predict_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image could not be read.")

        # --- Face detection (simple) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 100))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = img[y:y+h, x:x+w]
        else:
            # No face detected → use full image
            face = img

        # --- EXACT same preprocessing as training ---
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float32") / 255.0  # SAME as training

        face = np.expand_dims(face, axis=0)

        # --- PREDICTION ---
        raw = image_model.predict(face)[0][0]

        # MobileNet binary output → 0 = fake, 1 = real
        threshold = 0.67

        label = "REAL" if raw >= threshold else "FAKE"
        confidence = round(abs(raw - 0.5) * 2 * 100, 2)

        print(f"[Image] Prediction: {label} | Raw={raw:.4f}")
        return {"label": label, "confidence": confidence}

    except Exception as e:
        print(f"❌ Error in image prediction: {e}")
        return {"label": "Error", "confidence": 0.0}

# TESTING (Optional)
# -----------------------------
if __name__ == "__main__":
    print("---- Deepfake Detection System ----")
    # Example usage:
    # print(predict_video("uploads/test_video.mp4"))
    # print(predict_image("uploads/test_image.jpg"))    # print(predict_image(TEST_IMAGE))