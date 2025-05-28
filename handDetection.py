import cv2
import mediapipe as mp

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
mpStyle = mp.solutions.drawing_styles

hand = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.2)

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if not success:
        break
    image = cv2.flip(image, 1)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hand.process(imageRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS,
                                  mpStyle.get_default_hand_landmarks_style(),
                                  mpStyle.get_default_hand_connections_style())
    cv2.imshow("Hand Detection", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
