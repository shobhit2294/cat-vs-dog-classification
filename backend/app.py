from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model.keras")


# Image preprocessing
def preprocess_image(img):
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Home route
@app.route("/")
def home():
    return "Cat vs Dog Prediction API is running"


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["file"]

    try:
        img = Image.open(file).convert("RGB")
    except:
        return jsonify({"error": "Invalid image file"})

    img = preprocess_image(img)

    prediction = model.predict(img)

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        result = "Dog"
    else:
        result = "Cat"

    print("Prediction:", result)

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)