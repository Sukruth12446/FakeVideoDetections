
from tensorflow.keras.models import load_model
import joblib
import os
import sys

print("🔍 Deepfake Detection Accuracy Checker\n")

# ============================
# Step 1: Load your trained model
# ============================
MODEL_PATH = r"C:\Users\admin\PycharmProjects\FakeVideoDetection\model\mobilenet_lstm_improved.h5" # Change if your model file name differs
HISTORY_PATH = "history.pkl"               # Optional: training history file

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ Model file not found: {MODEL_PATH}")
    print("💡 Please ensure the model file is in the same folder or update MODEL_PATH.")
    sys.exit()

# Load model
model = load_model(MODEL_PATH)
print(f"✅ Model loaded successfully from: {MODEL_PATH}")

# ============================
# Step 2: Load training history (if available)
# ============================
if os.path.exists(HISTORY_PATH):
    print("\n📂 Found saved training history, loading...\n")
    history = joblib.load(HISTORY_PATH)

    # Safely extract metrics from history
    train_acc = history.history.get('accuracy', [None])[-1]
    val_acc = history.history.get('val_accuracy', [None])[-1]
    train_loss = history.history.get('loss', [None])[-1]
    val_loss = history.history.get('val_loss', [None])[-1]

    print("📊 Training Summary (from history):")
    print("----------------------------------------")
    print(f"Training Accuracy   : {train_acc * 100:.2f}%" if train_acc else "Training Accuracy   : Not available")
    print(f"Validation Accuracy : {val_acc * 100:.2f}%" if val_acc else "Validation Accuracy : Not available")
    print(f"Training Loss       : {train_loss:.4f}" if train_loss else "Training Loss       : Not available")
    print(f"Validation Loss     : {val_loss:.4f}" if val_loss else "Validation Loss     : Not available")
    print("----------------------------------------")

else:
    print("\n⚠️ No training history file found.")
    print("   To view accuracy, save it during training using this example:\n")
    print("""
    Example (during model training):
    --------------------------------
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

    import joblib
    joblib.dump(history, "history.pkl")

    Then run this script again to see the accuracy.
    --------------------------------
    """)

# ============================
# Step 3: Display model summary
# ============================
print("\n🧠 Model Summary:")
print("----------------------------------------")
model.summary()
print("----------------------------------------")
print("✅ Accuracy check completed.\n")
