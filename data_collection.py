import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


DATASET_PATH = r"asl_dataset"  
OUTPUT_FILE = "landmarks_dataset_option.csv"


data = []


with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
    for label in os.listdir(DATASET_PATH):  # Label folders (0-9, A-Z)
        label_path = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(label_path):
            continue  # not a directory

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue  # image cant be read

           
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Get absolute coordinates
                    h, w, _ = image.shape  # Get image dimensions
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0
                    
                    # Find bounding box around the hand
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min, y_min = min(x_min, x), min(y_min, y)
                        x_max, y_max = max(x_max, x), max(y_max, y)
                    
                    # Expand bounding box slightly
                    padding = 20
                    x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                    x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

                    # Normalize landmarks relative to the bounding box
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        x_rel = (lm.x * w - x_min) / (x_max - x_min)  
                        y_rel = (lm.y * h - y_min) / (y_max - y_min)  
                        z_rel = lm.z  

                        landmarks.extend([x_rel, y_rel, z_rel])

                    
                    data.append([label] + landmarks)


df = pd.DataFrame(data, columns=["label"] + [f"{axis}{i}" for i in range(21) for axis in "xyz"])
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved extracted landmarks to {OUTPUT_FILE}")
