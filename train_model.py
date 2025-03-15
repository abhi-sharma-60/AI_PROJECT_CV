import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("landmarks_dataset_option.csv")

# Extract features (X) and labels (y)
X = df.iloc[:, 1:].values  # Landmark coordinates
y = df.iloc[:, 0].values  # Labels

# Encode labels (0-9 -> 0-9, A-Z -> 10-35)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Converts labels into numeric form
y = to_categorical(y)  # Convert to one-hot encoding

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(36, activation='softmax')  # 36 classes (0-9, A-Z)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save trained model
model.save("landmarks_model.h5")
print("Model training complete and saved as landmarks_model.h5")
