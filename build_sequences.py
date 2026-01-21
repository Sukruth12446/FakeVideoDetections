# build_sequences.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image

FRAMES_PATH = "dataset/frames"        # input frames: frames/real, frames/fake
SEQUENCES_OUT = "dataset/sequences"  # output npz files
SEQUENCE_LENGTH = 60
IMG_SIZE = 224
VAL_RATIO = 0.15
RANDOM_STATE = 42

def list_frame_files(cls_folder):
    files = sorted(glob(os.path.join(cls_folder, "*.jpg")))
    # group by video prefix (everything before _frame_)
    groups = {}
    for f in files:
        base = os.path.basename(f)
        if "_frame_" in base:
            prefix = base.split("_frame_")[0]
        else:
            # fallback group per file (treat single images like tiny sequence)
            prefix = base
        groups.setdefault(prefix, []).append(f)
    return list(groups.values())

def make_sequences_for_class(class_name, label):
    folder = os.path.join(FRAMES_PATH, class_name)
    groups = list_frame_files(folder)
    sequences = []
    labels = []
    for g in groups:
        # ensure frames are sorted
        g = sorted(g)
        # create non-overlapping sequences
        for i in range(0, max(1, len(g) - SEQUENCE_LENGTH + 1), SEQUENCE_LENGTH):
            chunk = g[i:i + SEQUENCE_LENGTH]
            if len(chunk) < SEQUENCE_LENGTH:
                # pad using last frame
                while len(chunk) < SEQUENCE_LENGTH:
                    chunk.append(chunk[-1])
            arr = np.zeros((SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            for j, fpath in enumerate(chunk):
                img = Image.open(fpath).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                arr[j] = np.array(img)
            sequences.append(arr)
            labels.append(label)
    return sequences, labels

def build_and_save():
    X = []
    y = []
    for cls, label in [("real", 0), ("fake", 1)]:
        seqs, labels = make_sequences_for_class(cls, label)
        X.extend(seqs)
        y.extend(labels)
        print(f"Found {len(seqs)} sequences for class {cls}")
    X = np.array(X)
    y = np.array(y)
    print("Total sequences:", len(X))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_RATIO, random_state=RANDOM_STATE, stratify=y)
    os.makedirs(SEQUENCES_OUT, exist_ok=True)
    np.savez_compressed(os.path.join(SEQUENCES_OUT, "train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(SEQUENCES_OUT, "val.npz"), X=X_val, y=y_val)
    print("Saved sequences to", SEQUENCES_OUT)

if __name__ == "__main__":
    build_and_save()
