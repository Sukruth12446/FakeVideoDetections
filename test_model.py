from tensorflow.keras.models import load_model

model = load_model("model/mobilenet_lstm_model.h5")
model.summary()

