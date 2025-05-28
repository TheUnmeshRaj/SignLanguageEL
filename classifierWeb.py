import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load model once
with open('models/model.p', 'rb') as f:
    model = pickle.load(f)['model']

# Setup Mediapipe once
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels = {0: 'peace', 1: 'ok', 2: 'hi', 3: 'thank you', 4: 'you'}

def predict_sign(image_bgr):
    """Given a BGR image (OpenCV style), returns predicted sign label or None."""
    H, W, _ = image_bgr.shape
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None  # No hand detected

    # Only using the first detected hand for simplicity
    handLms = results.multi_hand_landmarks[0]
    x_list = [lm.x for lm in handLms.landmark]
    y_list = [lm.y for lm in handLms.landmark]

    features = []
    xmin, ymin = min(x_list), min(y_list)
    for lm in handLms.landmark:
        features.append(lm.x - xmin)
        features.append(lm.y - ymin)

    pred = model.predict([np.asarray(features)])
    return labels[int(pred[0])]
