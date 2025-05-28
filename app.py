from flask import Flask, request, jsonify, send_from_directory
from classifierWeb import predict_sign
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    result = predict_sign(img)
    if result is None:
        return jsonify({'prediction': 'No hand detected'})
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
