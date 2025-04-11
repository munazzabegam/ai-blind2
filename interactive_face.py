import cv2
import face_recognition
import numpy as np
import tensorflow as tf
import pickle
import pyttsx3
import threading
import speech_recognition as sr
from datetime import datetime

# Load face recognition model
model = tf.keras.models.load_model("face_model.h5")
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

# Constants
KNOWN_WIDTH = 14  # cm (avg human face width)
FOCAL_LENGTH = 600

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 160)

# Setup recognition variables
last_recognized = ""
spoken_names = set()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def calculate_distance(known_width, focal_length, face_width_pixels):
    return (known_width * focal_length) / face_width_pixels

# Voice command listener (background thread)
def voice_listener():
    global last_recognized
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                print("[LISTENING] Say something...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)

            query = recognizer.recognize_google(audio).lower()
            print("You said:", query)

            if "who is in front" in query:
                if last_recognized:
                    speak(f"{last_recognized} is in front")
                else:
                    speak("No one is recognized in front.")
        except Exception:
            pass  # Ignore errors (silent or misheard)

# Start voice command listener in background
threading.Thread(target=voice_listener, daemon=True).start()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        face_width = right - left
        distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, face_width)

        # Predict using neural network
        encoding_input = np.expand_dims(encoding, axis=0)
        predictions = model.predict(encoding_input, verbose=0)
        label_index = np.argmax(predictions)
        confidence = np.max(predictions)

        name = inv_label_map.get(label_index, "Unknown") if confidence > 0.6 else "Unknown"
        if name != "Unknown":
            label_text = f"{name} at {int(distance)} centimeters"
            last_recognized = name
            if name not in spoken_names:
                speak(label_text)
                spoken_names.add(name)
        else:
            label_text = "Unknown"

        # Draw rectangle & text
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Interactive Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
