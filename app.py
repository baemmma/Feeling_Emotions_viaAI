import cv2
import torch
import streamlit as st
from torchvision import transforms
from PIL import Image
from model import FaceEmotionModel

# Streamlit UI
st.markdown("<h1 style='text-align: center; font-weight: bold;'>Feeling Emotions via AI</h1>", unsafe_allow_html=True)
st.subheader("Created by BAHRI Emna")
st.write("emnabahri@yahoo.fr")

# Load Model and Define Emotions
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = FaceEmotionModel(num_classes=8)
emotions = {0: "anger", 1: "contempt", 2: "disgust", 3: "fear", 4: "happy", 5: "neutral", 6: "sad", 7: "surprise"}
model.load_state_dict(torch.load('./best_trained_model_weights.pt', map_location=torch.device('cpu')))
model.eval()

# Initialize OpenCV Face Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open Video Capture
cap = cv2.VideoCapture(0)

stframe = st.empty()
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        face = transform(face)
        face = face.unsqueeze(0)

        with torch.no_grad():
            outputs = model(face)
            predicted_class_index = torch.argmax(outputs, dim=1).item()

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotions[predicted_class_index], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    stframe.image(frame, channels="BGR")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
