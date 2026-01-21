import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    TimeDistributed, GlobalAveragePooling2D, LSTM,
    Dense, Dropout, Input
)
from tensorflow.keras.optimizers import Adam

# -------------------------
# CONSTANTS
# -------------------------
IMG_SIZE = 224
SEQUENCE_LENGTH = 60           # Number of frames given to LSTM
NUM_CLASSES = 1                # Binary output (real=0, fake=1)
LEARNING_RATE = 1e-5

# -------------------------
# CHECK GPU
# -------------------------
print("🔍 Checking for GPU...")
print(tf.config.list_physical_devices("GPU"))

# -------------------------
# LOAD BASE MODEL
# -------------------------
mobilenet = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze all layers first
mobilenet.trainable = False

# Unfreeze last 30 layers for fine-tuning
for layer in mobilenet.layers[-30:]:
    layer.trainable = True


# -------------------------
# BUILD LSTM MODEL
# -------------------------
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)),

    # Apply MobileNetV2 on each frame
    TimeDistributed(mobilenet),
    TimeDistributed(GlobalAveragePooling2D()),

    # LSTM layers
    LSTM(128, return_sequences=True),
    Dropout(0.5),

    LSTM(64),
    Dropout(0.5),

    # Final output for binary (SIGMOID)
    Dense(NUM_CLASSES, activation='sigmoid')
])

# -------------------------
# COMPILE THE MODEL
# -------------------------
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------
# MODEL SUMMARY
# -------------------------
model.summary()

# -------------------------
# SAVE MODEL
# -------------------------
os.makedirs("model", exist_ok=True)
model.save("model/mobilenet_lstm_bilstm_v2.h5")
print("\n MobileNet + LSTM model successfully saved!")
