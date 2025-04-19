import cv2
import numpy as np
import pyttsx3
import torch
import os
import sys
import face_recognition
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))


import threading
import speech_recognition as sr
import base64
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# Add YOLOv5 path
YOLO_PATH = os.path.join(os.getcwd(), "yolov5")
sys.path.append(YOLO_PATH)

# Load face recognition model & labels
model = load_model("face_model.h5")
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

# Distance parameters
KNOWN_WIDTH = 14  # cm
FOCAL_LENGTH = 600

def calculate_distance(known_width, focal_length, face_width_pixels):
    return (known_width * focal_length) / face_width_pixels

# Voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)
spoken_names = set()
last_recognized = ""

def speak_once(text):
    if text not in spoken_names:
        engine.say(text)
        engine.runAndWait()
        spoken_names.add(text)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load YOLOv5
device = select_device('')
yolo_model = DetectMultiBackend("yolov5s.pt", device=device)
yolo_stride = yolo_model.stride
yolo_names = yolo_model.names

# Setup webcam
cap = cv2.VideoCapture(0)

# Function to convert image to base64
def image_to_base64(frame):
    # Encode the image in PNG format (you can use JPG if preferred)
    _, buffer = cv2.imencode('.png', frame)  # `frame` is a NumPy array representing the image
    img_bytes = buffer.tobytes()  # Convert to byte format
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')  # Base64 encode and decode to string
    return img_base64

# Voice command thread
def listen_command():
    global last_recognized
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
            query = recognizer.recognize_google(audio).lower()
            print("You said:", query)
            if "who is in front" in query:
                if last_recognized:
                    speak(f"{last_recognized} is in front")
                else:
                    speak("No one is recognized in front right now.")
        except Exception as e:
            pass  # No output on silent/misrecognition

# Start voice listener in background
threading.Thread(target=listen_command, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---------- FACE RECOGNITION ----------
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        face_width = right - left
        distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, face_width)

        encoding_input = np.expand_dims(encoding, axis=0)
        predictions = model.predict(encoding_input)
        label_index = np.argmax(predictions)
        confidence = np.max(predictions)

        name = inv_label_map.get(label_index, "Unknown") if confidence > 0.6 else "Unknown"
        if name != "Unknown":
            last_recognized = f"{name} at {int(distance)} centimeters"

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        text = f"{name} ({distance:.2f} cm)"
        cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        speak_once(f"{name} is {int(distance)} centimeters in front")

    # ---------- YOLO OBJECT DETECTION ----------
    img = letterbox(frame, new_shape=640, stride=yolo_stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3xHxW
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

    # ---------- Display ----------
    cv2.imshow("Face Recognition + YOLOv5 + Voice", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
