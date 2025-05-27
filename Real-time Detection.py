import cv2 # OpenCV for webcam access and face detection
import numpy as np # For numerical operations, especially with image arrays
from tensorflow.keras.models import load_model # To load our saved deep learning model
import os # To check if files exist

# --- Debugging CWD in VS Code (Keep this for now) ---
print(f"Script's current working directory: {os.getcwd()}")
expected_path = 'E:\\SATHVIK\\study\\Projects\\Face Emotion Recognition' # Note double backslashes
if os.getcwd() != expected_path:
    print("WARNING: Script is NOT running from the expected project directory.")
    print(f"Please launch VS Code from or change terminal directory to {expected_path}")
# --- End of debugging part ---

# --- Configuration ---
# Use the ABSOLUTE PATH to your model file.
# This ensures the model is found regardless of the script's current working directory.
MODEL_PATH = 'E:/SATHVIK/study/Projects/Face Emotion Recognition/emotion_recognition_model.h5' # Use forward slashes here for Python

# Path to the Haar Cascade XML file for face detection.
# This file is typically installed with OpenCV.
# cv2.data.haarcascades provides the correct path to the installed cascades.
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Define emotion labels. This order MUST match the order your model was trained on
# (i.e., the order LabelEncoder assigned numerical values to).
# Standard FER2013 dataset emotions are often in this alphabetical order.
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- Step 1: Load the Trained Emotion Recognition Model ---
print("Attempting to load the emotion recognition model...")
try:
    # Load the model from the specified .h5 file
    emotion_model = load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully!")
except Exception as e:
    # If the model file is not found or corrupted, print an error and exit
    print(f"ERROR: Could not load model from '{MODEL_PATH}'.")
    print(f"Please ensure the file exists at the specified absolute path. Error: {e}")
    exit() # Exit the script if the model can't be loaded, as it's essential.

# --- Step 2: Load the Face Cascade Classifier ---
print("Attempting to load face detection cascade...")
# This cascade file is used by OpenCV to detect human faces in an image.
face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)
if face_detector.empty():
    # If the cascade file is not found, print an error and exit
    print(f"ERROR: Could not load face cascade classifier from '{FACE_CASCADE_PATH}'.")
    print("Please ensure OpenCV's 'haarcascade_frontalface_default.xml' is correctly installed or provide its full path.")
    exit() # Exit the script if face detection fails.
print("Face cascade loaded successfully!")

# --- Step 3: Initialize Webcam ---
print("Initializing webcam...")
# 0 refers to the default webcam. If you have multiple cameras, you might try 1, 2, etc.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    # If the webcam cannot be opened, print an error and exit
    print("ERROR: Could not open webcam. Please check if it's connected and not in use.")
    exit()
print("Webcam initialized. Starting real-time detection.")
print("Press 'q' to quit the application.")

# --- Step 4: Real-time Emotion Detection Loop ---
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam. Exiting...")
        break # Exit the loop if frame reading fails

    # Flip the frame horizontally for a mirrored view (more intuitive for user)
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale for face detection (Haar cascades work better on grayscale)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    # detectMultiScale returns a list of rectangles (x, y, w, h) for each detected face.
    #   scaleFactor: How much the image size is reduced at each image scale. (e.g., 1.1 means reducing by 10% each time)
    #   minNeighbors: How many neighbors each candidate rectangle should have to retain it. Higher values reduce false positives.
    #   minSize: Minimum possible object size. Objects smaller than this are ignored.
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Extract the Region of Interest (ROI) - the face itself
        face_roi = gray_frame[y:y + h, x:x + w]

        # Resize the face ROI to 48x48 pixels, which is the input size our model expects
        resized_face = cv2.resize(face_roi, (48, 48))

        # Prepare the face image for prediction:
        # 1. Convert to a NumPy array (already is, but explicit for clarity)
        # 2. Expand dimensions: The model expects a batch of images (even if it's just one).
        #    So, (48, 48) becomes (1, 48, 48, 1) for (batch_size, height, width, channels).
        input_image = np.expand_dims(resized_face, axis=0) # Add batch dimension
        input_image = np.expand_dims(input_image, -1)     # Add channel dimension (1 for grayscale)

        # 3. Normalize pixel values: Scale from 0-255 to 0-1, just like during training.
        input_image = input_image.astype('float32') / 255.0

        # Make a prediction using the loaded emotion model
        # model.predict returns probabilities for each emotion class. [0] gets the probabilities for the single image.
        predictions = emotion_model.predict(input_image, verbose=0)[0]

        # Get the index of the emotion with the highest probability
        predicted_label_index = np.argmax(predictions)

        # Get the actual emotion label (e.g., 'happy', 'angry') using the index
        predicted_emotion_text = EMOTION_LABELS[predicted_label_index]

        # --- Display results on the frame ---
        # Draw the predicted emotion text above the face rectangle
        cv2.putText(frame, predicted_emotion_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # Green text

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green rectangle

    # Display the final frame with detections in a window
    cv2.imshow('Real-time Face Emotion Detection', frame)

    # Check for 'q' key press to quit the application
    # cv2.waitKey(1) waits for 1ms for a key event. 0xFF is a bitmask for keyboard input.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Step 5: Clean Up ---
# Release the webcam resource
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
print("Application closed.")