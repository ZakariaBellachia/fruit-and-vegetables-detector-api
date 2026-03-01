import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import io
import random
import numpy as np
app = FastAPI()
# Fix Python random
random.seed(42)
# Fix numpy random
np.random.seed(42)
# Fix tensorflow random
tf.random.set_seed(42)
from PIL import Image

# ----------------------------------------
# LOAD THE tensorflow  MODEL ONCE
# ----------------------------------------
import os

MODEL_PATH = os.path.join(os.getcwd(), "model", "fruit_vegetable_model.tflite")
# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details() # only once, no need to call in endpoint
# label map (index → label)
label_map = {'Apple': 0, 'Banana': 1, 'Broccoli': 2, 'Carrots': 3, 'Cauliflower': 4, 'Chili': 5, 
             'Coconut': 6, 'Cucumber': 7, 'Custard apple': 8, 'Dates': 9, 'Garlic': 10, 'Grape': 11, 
             'Green Lemon': 12, 'Jackfruit': 13, 'Kiwi': 14, 'Mango': 15, 'Okra': 16, 'Onion': 17, 'Orange': 18, 
             'Papaya': 19, 'Peanut': 20, 'Pineapple': 21, 'Pomegranate': 22, 'Star Fruit': 23, 'Strawberry': 24, 
             'Sweet Potato': 25, 'Watermelon': 26}

# ============================================
# ROUTES
# ============================================
@app.get("/")
def read_root():
    return {"message": "Welcome to the fruits and vegetables Images Detection API!"}

@app.get("/about")
def read_about():
    return {"description": "This API detects fruits and vegetables using a trained CNN model."}
@app.post("/predict")
async def predict_fruits_and_vegetables(file: UploadFile = File(...)):
    try:
        # Read file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize to match training size
        image = image.resize((224, 224))  # adjust to your model input

        # Convert to numpy + normalize
        image_arr = np.array(image, dtype=np.float32) / 255.0
        image_arr = np.expand_dims(image_arr, axis=0)

        # Run TFLite prediction
        interpreter.set_tensor(input_details[0]["index"], image_arr)
        interpreter.invoke()  

        # Get output from TFLite model
        predictions = interpreter.get_tensor(output_details[0]["index"])

        # Compute class and confidence
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        # Map index → label
        label_map_inv = {v: k for k, v in label_map.items()}
        predicted_label = label_map_inv.get(predicted_class, "Unknown")
        
        return {
            "predicted_label": predicted_label,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok", "model_loaded": True}