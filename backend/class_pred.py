# classifier.py

import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("backend\models\garbage_tf_model.h5")
CLASS_NAMES = ["Biodegradable","Ewaste","hazardous","Non Biodegradable","Pharmaceutical and Biomedical Waste"]

def classify_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return predicted_class, confidence
