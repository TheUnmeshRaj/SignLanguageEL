import pickle
import cv2
import mediapipe as mp
import numpy as np


with open('models/model.p', 'rb') as f:
    model = pickle.load(f)['model']


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open camera")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

labels = {0: 'peace', 1: 'ok', 2: 'hi', 3: 'thank you', 4: 'fuck you'}

def draw_rounded_rect(img, pt1, pt2, color, thickness=2, r=20, d=20):
    x1, y1 = pt1
    x2, y2 = pt2
    
    cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
    cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
    cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
    cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
    
    cv2.ellipse(img, (x1+r, y1+r), (r,r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y1+r), (r,r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+r, y2-r), (r,r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y2-r), (r,r), 0, 0, 90, color, thickness)
    
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                handLms,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

        for handLms in results.multi_hand_landmarks:
            x_list = [lm.x for lm in handLms.landmark]
            y_list = [lm.y for lm in handLms.landmark]

            features = []
            xmin, ymin = min(x_list), min(y_list)
            xmax, ymax = max(x_list), max(y_list)

            for lm in handLms.landmark:
                features.append(lm.x - xmin)
                features.append(lm.y - ymin)

            x1 = int(xmin * W) - 15
            y1 = int(ymin * H) - 15
            x2 = int(xmax * W) + 15
            y2 = int(ymax * H) + 15

            pred = model.predict([np.asarray(features)])
            label = labels[int(pred[0])]

            
            overlay = frame.copy()
            alpha = 0.6
            box_width = 140
            box_height = 40
            cv2.rectangle(overlay, (x1, y1 - box_height), (x1 + box_width, y1), (30, 30, 30), -1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            
            draw_rounded_rect(frame, (x1, y1), (x2, y2), (0, 255, 150), thickness=3, r=15)

            
            cv2.putText(frame, label, (x1 + 10, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    
    # frame =  cv2.flip(frame, 1)  
    cv2.imshow("Modern Sign Language Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
