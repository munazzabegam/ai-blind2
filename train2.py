import os
import numpy as np
import face_recognition
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Dataset directory
DATASET_DIR = "dataset"

# Load dataset
images = []
labels = []

print("[INFO] Loading dataset...")
for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                images.append(encodings[0])
                labels.append(person_name)
        except Exception as e:
            print(f"[WARN] Skipped {image_name}: {e}")

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
y = to_categorical(encoded_labels)
X = np.array(images)

# Normalize embeddings (important!)
X = X / np.linalg.norm(X, axis=1, keepdims=True)

# Save label map
label_map = {name: idx for idx, name in enumerate(label_encoder.classes_)}
with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("[INFO] Training model...")

# Define model
model = Sequential([
    Input(shape=(128,)),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(label_map), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_face_model.keras", monitor='val_accuracy', save_best_only=True)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=8,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Save final model
model.save("face_model_final.keras")

print("[INFO] Training complete.")
print("[INFO] Best model saved as 'best_face_model.keras'. Final model saved as 'face_model_final.keras'.")
print("[INFO] Labels saved as 'label_map.pkl'")
