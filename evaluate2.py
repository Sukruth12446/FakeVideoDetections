# evaluate_model.py
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

MODEL_PATH = "model/mobilenet_lstm_sequence.h5"
SEQUENCES_DIR = "dataset/sequences"

def load_val():
    data = np.load(os.path.join(SEQUENCES_DIR, "val.npz"))
    X_val, y_val = data["X"].astype("float32")/255.0, data["y"]
    return X_val, y_val

def evaluate():
    X_val, y_val = load_val()
    model = tf.keras.models.load_model(MODEL_PATH)
    preds = model.predict(X_val, verbose=1)
    preds_bin = (preds.flatten() >= 0.5).astype(int)
    print("Accuracy:", accuracy_score(y_val, preds_bin))
    print("\nClassification Report:")
    print(classification_report(y_val, preds_bin, target_names=["real","fake"]))
    cm = confusion_matrix(y_val, preds_bin)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["real","fake"], yticklabels=["real","fake"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate()
