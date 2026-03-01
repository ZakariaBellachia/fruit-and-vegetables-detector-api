import tensorflow as tf
import os
# Load your .h5 model
MODEL_PATH = os.path.join(os.getcwd(), "model", "fruit_vegetable_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Create the converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: optimize for size and speed
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()
# Save the converted model to a file
with open("fruit_vegetable_model.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ TFLite model saved as fruit_vegetable_model.tflite")       