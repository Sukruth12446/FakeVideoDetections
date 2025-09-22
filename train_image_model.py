# train_image_model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 224

# Build a simple CNN model for image classification (REAL vs FAKE)
model = Sequential([
    MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Save the model (you'll need to train it first with your dataset)
model.save("model/mobilenet_image_model.h5")

print("✅ Image model saved successfully.")
