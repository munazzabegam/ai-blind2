import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import cv2
import face_recognition
import pickle

# Dataset path
dataset_path = "dataset"

# Store encodings & labels
encodings = []
labels = []
label_map = {}

# Load dataset and process images
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    label_map[person_name] = len(label_map)  # Assign numeric label

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        image = face_recognition.load_image_file(img_path)
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) > 0:
            encodings.append(face_encodings[0])
            labels.append(label_map[person_name])

# Convert to NumPy arrays
X = np.array(encodings)
y = np.array(labels)

# Build Neural Network for Classification
model = Sequential([
    Dense(512, activation='relu', input_shape=(128,)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=100, batch_size=16)

# Save model & label map
model.save("face_model.h5")
with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("Training complete!")
