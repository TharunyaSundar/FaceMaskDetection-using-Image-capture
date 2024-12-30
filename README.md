# Face Mask Detection Using Image Capture

## Overview

This project implements a real-time face mask detection system using deep learning. The model detects faces in images and classifies whether the person is wearing a mask or not. Instead of continuously streaming video, this version allows users to capture an image using their webcam and then performs face mask detection on that image.

## Features

- **Image Capture**: Capture an image from your webcam to perform face mask detection.
- **Face Mask Detection**: Classifies if a person is wearing a mask or not using a pre-trained deep learning model.
- **Streamlit Interface**: A simple and user-friendly interface built with Streamlit to interact with the model.

## Requirements

Before running the project, make sure to install the necessary dependencies. You can do this by running the following command:

pip install -r requirements.txt

The `requirements.txt` includes the necessary libraries for image processing, deep learning, and the Streamlit interface. Key libraries include:

- `tensorflow` for the mask detection model.
- `opencv-python` for image processing and webcam access.
- `streamlit` for the web interface.

## Setup

1. Clone the repository:

   git clone https://github.com/TharunyaSundar/FaceMaskDetection-using-Image-capture.git
   cd FaceMaskDetection-using-Image-capture

2. It's highly recommended to set up a virtual environment to avoid conflicts with other Python projects on your local machine. You can create a virtual environment with the following commands:

For Windows:
python -m venv venv
venv\Scripts\activate

For MacOS/Linux:
python3 -m venv venv
source venv/bin/activate

3. Install the required dependencies:

   pip install -r requirements.txt

4. Ensure you have a working webcam connected to your computer.

## Usage

1. Run the application:

   streamlit run app.py

2. Open the application in your browser. You should see a user interface where you can:
   - **Capture an image** using your webcam.
   - Once you capture an image, the system will detect faces in the image and classify whether a mask is being worn.

3. The detection results (with bounding boxes around faces and mask predictions) will be displayed on the captured image.

## How It Works

1. **Face Detection**: The face detection model detects faces in the captured image using OpenCV's pre-trained model. Each detected face is passed through a mask classification model.
   
2. **Mask Classification**: The mask classification model (trained with TensorFlow/Keras) determines whether the detected person is wearing a mask or not. The model outputs probabilities for "mask" or "no mask".

3. **Web Interface**: The interface is built with Streamlit, allowing users to interact with the webcam and visualize detection results in real-time.

## Technologies Used

- **Python**: Programming language used for the project.
- **TensorFlow/Keras**: Used to build and load the mask detection model.
- **OpenCV**: For image processing and face detection.
- **Streamlit**: For creating the user interface to interact with the webcam and display results.

## Acknowledgments

- The face mask detection model is based on a pre-trained model using a publicly available dataset for face mask detection.
- Thanks to the developers of `OpenCV`, `TensorFlow`, and `Streamlit` for providing the libraries used in this project.
