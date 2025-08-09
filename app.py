"""
Flask application for predicting class of Animals in a zoo
"""

import os
import traceback
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np


# Just for confirmation
print(f"Templates directory: {os.path.join(os.getcwd(), 'templates')}")

app = Flask(__name__)
CORS(app)

# Ensure the templates directory is set correctly
@app.route('/')
def home():
    return render_template('index.html')

# 🦁 Load the zoo ensemble model
model = joblib.load("zoo_classifier_ensemble.pkl")  # Make sure the filename matches!

# Set these to your actual values from your training data!
LEGS_MEAN = 2.841584
LEGS_STD = 2.033385
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    """
    data = request.json
    features = data.get('features', None)

    print(f"Received features: {features}")

    def encode(val):
        if isinstance(val, str):
            if val.lower() == "yes":
                return 1
            if val.lower() == "no":
                return 0
        return val

    if features is not None:
        features = [encode(f) for f in features]

    if features is None or len(features) != 16:
        return jsonify({'error': 'Expected 16 features (legs as integer).'}), 400

    try:
        # Scale legs (index 12)
        legs = float(features[12])
        legs_scaled = (legs - LEGS_MEAN) / LEGS_STD
        features[12] = legs_scaled  # Replace legs with legs_scaled

        features = np.array(features, dtype=float).reshape(1, -1)
        prediction = model.predict(features)[0]

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            print(f"Prediction: {prediction}, Probabilities: {proba}")
        else:
            proba = None
            print(f"Prediction: {prediction}, No probability available.")

        class_labels = {
            1: "Mammal",
            2: "Bird",
            3: "Reptile",
            4: "Fish",
            5: "Amphibian",
            6: "Bug",
            7: "Invertebrate"
        }

        result_label = class_labels.get(int(prediction), f"Class {prediction}")

        response = {"prediction": result_label}

        if proba is not None:
            response["confidence"] = f"{round(float(max(proba)) * 100, 2)}%"

        return jsonify(response)

    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during prediction. '
        'Please check your input and try again.'}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5050)
