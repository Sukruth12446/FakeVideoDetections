import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np

# Load pretrained DFDC model
def load_dfdc_model():
    model = torch.hub.load('selimsef/dfdc_deepfake_challenge', 'model')
    model.eval()
    return model

# Run prediction on a single video
def predict_dfdc(video_path):
    model = load_dfdc_model()

    # Extract a few frames from the video
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in range(10):   # sample 10 frames
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()

    if not frames:
        return {"label": "Error", "confidence": 0.0}

    # Convert frames to tensors
    frames = np.stack(frames)
    frames = np.transpose(frames, (0, 3, 1, 2)) / 255.0
    frames_tensor = torch.tensor(frames, dtype=torch.float32)

    with torch.no_grad():
        preds = model(frames_tensor)
        avg_pred = torch.sigmoid(preds).mean().item()

    label = "REAL" if avg_pred > 0.5 else "FAKE"
    confidence = round(float(avg_pred), 4)
    return {"label": label, "confidence": confidence}
