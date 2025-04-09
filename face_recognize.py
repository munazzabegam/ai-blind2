import cv2
import face_recognition
import numpy as np
import pickle
import tensorflow as tf
import pyttsx3
import threading
from scipy.spatial.distance import cosine

# Load trained model & label map
model = tf.keras.models.load_model("face_model.h5")
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)
inv_label_map = {v: k for k, v in label_map.items()}  # Reverse mapping

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Distance threshold for recognizing unknown faces
FACE_DISTANCE_THRESHOLD = 0.5  

# Focal length for distance estimation (calibration required)
KNOWN_HEIGHT = 20  # Average face height in cm
FOCAL_LENGTH = 650  # Adjust based on testing

# Store last announced person to prevent repetition
last_announced_person = None

def calculate_distance(known_height, focal_length, face_height_pixels):
    """Calculate the estimated distance from the camera."""
    if face_height_pixels > 0:
        return (known_height * focal_length) / face_height_pixels
    return None  # Avoid division by zero

def speak(text):
    """Give voice feedback for recognized person & distance."""
    global last_announced_person
    if text != last_announced_person:
        last_announced_person = text
        engine.say(text)
        engine.runAndWait()

# Open webcam
video_capture = cv2.VideoCapture(0)

def recognize_faces(frame):
    """Process the frame for AI-powered face recognition."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        face_height = bottom - top
        distance = calculate_distance(KNOWN_HEIGHT, FOCAL_LENGTH, face_height)

        # Predict using Neural Network
        encoding = np.expand_dims(encoding, axis=0)  # Reshape for model
        predictions = model.predict(encoding)
        label_index = np.argmax(predictions)
        confidence = np.max(predictions)

        # Check if it's a known or unknown person
        if confidence < FACE_DISTANCE_THRESHOLD:
            name = "Unknown"
        else:
            name = inv_label_map.get(label_index, "Unknown")

        results.append(((top, right, bottom, left), name, distance))

    return results

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue  # Skip if the frame isn't read correctly

    # Run face recognition in a separate thread for better FPS
    thread = threading.Thread(target=recognize_faces, args=(frame,))
    thread.start()
    thread.join()
    
    recognized_faces = recognize_faces(frame)

    for (top, right, bottom, left), name, distance in recognized_faces:
        # Draw bounding box and display name + distance
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({distance:.2f} cm)", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Speak out name & distance only once
        if name != "Unknown":
            text = f"{name} is {distance:.2f} centimeters away."
            threading.Thread(target=speak, args=(text,)).start()

    cv2.imshow("AI Face Recognition with Distance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
