from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, GlobalAveragePooling2D, Input
import tensorflow as tf

# Constants
IMG_SIZE = 224
SEQUENCE_LENGTH = 60  # number of frames the model will expect
NUM_CLASSES = 1       # binary classification (Fake or Real)

# Load base model (MobileNetV2)
mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
mobilenet.trainable = False

# Wrap in TimeDistributed for sequence input
model = Sequential([
    TimeDistributed(mobilenet, input_shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)),
    TimeDistributed(GlobalAveragePooling2D()),
    LSTM(64, return_sequences=False),
    Dense(NUM_CLASSES, activation='sigmoid')  # use sigmoid for binary classification
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save it
model.save('model/mobilenet_lstm.h5')

print("✅ MobileNet + LSTM model created and saved.")

