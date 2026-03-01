# 🍎 Fruit and Vegetables Detector Web App

## 1️⃣ Project Overview
A deep learning web app that classifies images of fruits and vegetables.

- Built with **TensorFlow / Keras**  
- Fully deployed on the web, accessible **without installation**  
- Supports multiple fruits and vegetables with high accuracy

---

## 2️⃣ Features
- Predicts the **class of an uploaded image**  
- Shows **confidence score** for each prediction  
- Easy to access via a browser — **no setup required**

---

## 3️⃣ Try It Online
You can try the app directly here:

🌐 [Fruit & Vegetables Detector Web App](https://fruit-frontend-1-0.onrender.com)

**Steps:**  
1. Open the website  
2. Upload an image of a fruit or vegetable  
3. View the **predicted class** and **confidence score** instantly

---
## 4️⃣ Example Usage (for developers)

If you want to run locally or integrate the model via Python:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('model.h5')

# Prepare the image
img = image.load_img('apple.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make prediction
pred = model.predict(img_array)
predicted_class = np.argmax(pred, axis=1)[0]

print(predicted_class)

---
##5️⃣ Model Details

Architecture: VGG16

Number of classes: 27 (Apple, Banana, Carrot…)

Input size: 224x224

Dataset: Fruits and Vegetables images (collected from [https://www.kaggle.com/datasets/shadikfaysal/fruit-and-vegetables-ssm/data])
Note: Some irrelevant classes were removed for clarity

---
##6️⃣ License

This project is licensed under the Apache License 2.0 – see the LICENSE
 file for details.
 
