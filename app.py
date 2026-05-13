import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import av
import numpy as np
import streamlit as st

from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from tensorflow.keras.models import load_model

# ---------------- MODEL ---------------- #

@st.cache_resource
def load_emotion_model():
    return load_model(
        "emotion_recognition_model.h5",
        compile=False
    )

emotion_model = load_emotion_model()

# ---------------- LABELS ---------------- #

EMOTION_LABELS = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]

# ---------------- FACE DETECTOR ---------------- #

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

# ---------------- EMOTION FUNCTION ---------------- #

def detect_emotions(frame):

    # Handle grayscale safely
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    detected_emotions = []

    for (x, y, w, h) in faces:

        roi_gray = gray[y:y+h, x:x+w]

        roi_gray = cv2.resize(roi_gray, (48, 48))

        roi = roi_gray.astype("float32") / 255.0

        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = emotion_model.predict(
            roi,
            verbose=0
        )[0]

        label = EMOTION_LABELS[np.argmax(prediction)]

        detected_emotions.append(label)

        # Rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Emotion text
        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    return frame, detected_emotions

# ---------------- STREAMLIT ---------------- #

st.title("Real-Time Face Emotion Recognition")

# ---------------- IMAGE UPLOAD ---------------- #

st.header("Upload Image")

uploaded_file = st.file_uploader(
    "Choose Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image = np.array(image)

    result, emotions = detect_emotions(image)

    st.image(result)

    if emotions:
        st.success(
            "Detected Emotion: " +
            ", ".join(emotions)
        )
    else:
        st.warning("No face detected")

# ---------------- LIVE WEBCAM ---------------- #

st.header("Live Webcam Emotion Detection")

class EmotionDetector(VideoTransformerBase):

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img, _ = detect_emotions(img)

        return img

webrtc_streamer(
    key="emotion-detection",
    video_transformer_factory=EmotionDetector
)