from tensorflow.keras.models import load_model
import numpy as np

# Load the trained MobileNet model
model = load_model("model/mobilenet_image_model.h5")

# Generate dummy test data (10 random "images" of 224x224x3)
# Normally, values are between 0 and 1 for normalized images
X_test = np.random.rand(10, 224, 224, 3)
y_test = np.random.randint(0, 2, size=(10, 1))  # random 0 or 1 as labels

# Evaluate model performance
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"✅ Model testing completed.")
print(f"📊 Accuracy: {accuracy * 100:.2f}%")
print(f"📉 Loss: {loss:.4f}")

