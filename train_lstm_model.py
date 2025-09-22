from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, GlobalAveragePooling2D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
import tensorflow as tf

# Set constants
IMG_SIZE = 224
SEQUENCE_LENGTH = 60  # Number of frames per video
NUM_CLASSES = 1  # Binary classification (Real vs Fake)
LEARNING_RATE = 1e-5  # Lower learning rate for fine-tuning

# Load MobileNetV2 as feature extractor
mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Unfreeze the last 30 layers for fine-tuning
for layer in mobilenet.layers[-30:]:
    layer.trainable = True

# Build the model
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)),

    TimeDistributed(mobilenet),
    TimeDistributed(GlobalAveragePooling2D()),

    LSTM(128, return_sequences=True),
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),

    Dense(NUM_CLASSES, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Summary
model.summary()

# Save the model (optional)
model.save("model/mobilenet_lstm_improved.h5")
print("✅ Improved MobileNet + LSTM model has been saved.")
