import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
 
# Load trained model
model = load_model("landmarks_model.h5")
 
# Load label encoder
labels = [str(i) for i in range(10)] + [chr(65 + i) for i in range(26)]  # 0-9 and A-Z
 
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
 
cap = cv2.VideoCapture(0)
 
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
 
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
 
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
 
                # Extract hand landmark coordinates
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])  # 63 values
 
                landmarks = np.array(landmarks).reshape(1, -1)  # Reshape for prediction
 
                # Predict using model
                prediction = model.predict(landmarks)[0]
                predicted_label = labels[np.argmax(prediction)]
 
                # Display prediction
                cv2.putText(frame, predicted_label, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
        # Show frame
        cv2.imshow("Sign Language Recognition", frame)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
cap.release()
cv2.destroyAllWindows()