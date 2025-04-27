import cv2
import numpy as np
import pyttsx3
import torch
import os
import sys
import face_recognition
import threading
import speech_recognition as sr
import base64
import requests
import time
import pickle
import queue
from fer import FER  # Emotion detection
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model

# Global frame
global_frame = None

# YOLOv5 Path
YOLO_PATH = os.path.join(os.getcwd(), "yolov5")
sys.path.append(YOLO_PATH)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# --- Correctly load face recognition model ---
model = load_model("best_face_model.keras")
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

# Distance estimation
KNOWN_WIDTH = 14  # cm
FOCAL_LENGTH = 600

def calculate_distance(known_width, focal_length, face_width_pixels):
    return (known_width * focal_length) / face_width_pixels

# Voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)
spoken_names = set()
last_recognized = ""
gemini_active = False

# Speech queue
speech_queue = queue.Queue()

def speak_once(text):
    if text not in spoken_names:
        spoken_names.add(text)
        speech_queue.put(text)

def speak(text):
    speech_queue.put(text)

def speech_thread():
    while True:
        text = speech_queue.get()
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

threading.Thread(target=speech_thread, daemon=True).start()

# Load YOLOv5
device = select_device('')
yolo_model = DetectMultiBackend("yolov5s.pt", device=device)
yolo_stride = yolo_model.stride
yolo_names = yolo_model.names

# Webcam
cap = cv2.VideoCapture(0)

# Gemini API setup
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

def ask_gemini(question):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        # --- Force Gemini to answer short and simple ---
        prompt = f"Answer very simply and clearly in 2-3 lines maximum: {question}"
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        if 'candidates' in result:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Gemini API error."
    except Exception as e:
        return f"Error contacting Gemini: {e}"

detector = FER()

def detect_emotion(frame):
    emotions = detector.top_emotion(frame)
    if emotions:
        return emotions[0]
    return "neutral"

def listen_command():
    global last_recognized, gemini_active
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                print("Listening...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            query = recognizer.recognize_google(audio).lower()
            print("You said:", query)

            if "amy start" in query:
                gemini_active = True
                speak("Gemini activated.")
                print("Gemini activated")

            elif "amy stop" in query:
                gemini_active = False
                speak("Gemini deactivated.")
                print("Gemini deactivated")

            elif "who is in front" in query:
                if last_recognized:
                    speak(f"{last_recognized} is in front.")
                else:
                    speak("No one recognized in front.")

            elif "what is the emotion" in query:
                if global_frame is not None:
                    emotion = detect_emotion(global_frame)
                    speak(f"The emotion is {emotion}.")
                else:
                    speak("No frame available for emotion detection.")

            elif gemini_active:
                # Let Gemini answer any general questions only when active
                response = ask_gemini(query)
                print("Gemini:", response)
                speak(response[:300])

        except Exception as e:
            print("Voice recognition error:", e)

threading.Thread(target=listen_command, daemon=True).start()

# FPS counter
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    global_frame = frame.copy()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --------- FACE RECOGNITION ---------
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        face_width = right - left
        distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, face_width)

        encoding_input = np.expand_dims(encoding, axis=0)
        encoding_input = encoding_input / np.linalg.norm(encoding_input, axis=1, keepdims=True)
        predictions = model.predict(encoding_input)
        label_index = np.argmax(predictions)
        confidence = np.max(predictions)

        name = inv_label_map.get(label_index, "Unknown") if confidence > 0.6 else "Unknown"
        if name != "Unknown":
            last_recognized = f"{name} at {int(distance)} centimeters"

        # ------ Crop the face for emotion detection ------
        face_crop = rgb_frame[top:bottom, left:right]
        if face_crop.size != 0:
            emotion = detect_emotion(face_crop)
        else:
            emotion = "neutral"

        # ------ Draw face box with name and emotion ------
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        text = f"{name} ({distance:.2f} cm)"
        cv2.putText(frame, text, (left, top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        emotion_text = f"{emotion}"
        cv2.putText(frame, emotion_text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        speak_once(f"{name} is {int(distance)} centimeters in front")

    # --------- YOLO DETECTION ---------
    img = letterbox(frame, new_shape=640, stride=yolo_stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    pred = yolo_model(img_tensor, augment=False)
    det = non_max_suppression(pred, 0.25, 0.45)[0]

    if det is not None and len(det):
        det[:, :4] = scale_boxes(img.shape[1:], det[:, :4], frame.shape).round()
        for *xyxy, conf, cls in det:
            label = yolo_names[int(cls)]
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Face Recognition + YOLOv5 + Gemini + Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
