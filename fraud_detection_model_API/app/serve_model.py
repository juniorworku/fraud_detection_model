from flask import request, jsonify
import pickle
import numpy as np
from app import app
from app.utils.preprocessing import preprocess_input

# Load the trained model
model_path = 'app/model/model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = preprocess_input(data)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})
