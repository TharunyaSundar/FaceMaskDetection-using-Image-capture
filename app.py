import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from PIL import Image

# Load the face detector and mask detection models
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

try:
    model = load_model("mask_detector.keras")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

maskNet = load_model("mask_detector.keras")

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.shape[0] > 0 and face.shape[1] > 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# Add tab navigation
tabs = st.tabs(["Detection", "About the Project"])

with tabs[0]:
    # Detection Page
    st.title("Face Mask Detection")
    st.write("Capture an image using your camera and the system will predict if a mask is being worn.")

    # Capture image from the browser camera
    image_file = st.camera_input("Capture Image")

    if image_file is not None:
        # Open the image as a PIL object
        image = Image.open(image_file)
        frame = np.array(image)

        # Convert the image from RGB to BGR (required by OpenCV)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run mask detection on the captured image
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # Draw boxes and predictions on the image
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # Draw rectangle and label on the image
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Display the image with the mask prediction result
        st.image(frame, channels="BGR")

with tabs[1]:
    # About the Project Page
    st.title("About the Face Mask Detection Project")
    st.write("""
    This project is a real-time face mask detection system using deep learning.

    **Key Features:**
    - Detects faces and identifies whether the person is wearing a mask or not.
    - Uses OpenCV for face detection and TensorFlow/Keras for mask classification.

    **How It Works:**
    - A pre-trained face detector is used to locate faces in the image.
    - Each detected face is passed through a mask classification model, which predicts if a mask is being worn.

    **Technologies Used:**
    - Python
    - Streamlit
    - TensorFlow/Keras
    - OpenCV

    **Why the Change from Video Feed to Image Capture:**
    In the local version of this project, we capture video frames continuously, allowing the system to perform mask detection in real-time. However, for the deployed version of the app, I opted for an image capture approach using Streamlit's `st.camera_input()` functionality. This choice ensures that the deployment version of the app is more efficient for users, as it captures a single image and performs mask detection only on that image.

    The main reason for this shift is that capturing a video stream from a browser and processing each frame in real-time can be resource-intensive and challenging to manage in the web environment. By capturing a single image, the deployment version is simplified and optimized for accessibility, while still delivering accurate predictions on whether individuals are wearing masks.

    **Instructions:**
    - Navigate to the "Detection" tab to capture a photo from your webcam.
    - After capturing the photo, the system will predict if a mask is being worn.

    **Acknowledgments:**
    - The mask detector model was trained on a public dataset of face images with and without masks.
    """)
