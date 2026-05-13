# Real-Time Face Emotion Recognition

## Overview
This project enables real-time facial emotion recognition using a webcam, detecting emotions such as angry, disgust, fear, happy, neutral, sad, and surprise. It employs OpenCV for face detection with Haar cascades and a pre-trained Convolutional Neural Network (CNN) model (`emotion_recognition_model.h5`) built with TensorFlow for emotion prediction. The system is suitable for applications like sentiment analysis, human-computer interaction, and psychological research.

## Prerequisites
- Python 3.8+ installed on your system
- A functional webcam (default device, index 0) for live video capture
- Required Python libraries:
  - `opencv-python` (for webcam access and face detection)
  - `numpy` (for numerical operations)
  - `tensorflow` (for loading and running the CNN model)
- Install dependencies manually:
  ```bash
  pip install opencv-python numpy tensorflow
  ```
- The pre-trained model file `emotion_recognition_model.h5` placed in the project directory (update `MODEL_PATH` in `Real-time Detection.py` to `./emotion_recognition_model.h5` or your preferred path)
- OpenCV’s Haar cascade file (`haarcascade_frontalface_default.xml`), typically included with `opencv-python`
- FER2013 dataset for model training (optional), downloadable from Kaggle: [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013). Extract it to an `images` folder in the project directory with `train` and `test` subfolders.

## How to Run
1. Ensure your webcam is connected and functional.
2. Save the project code in a file named `Real-time Detection.py`.
3. Place `emotion_recognition_model.h5` in the project directory and update `MODEL_PATH` in `Real-time Detection.py` (e.g., `MODEL_PATH = './emotion_recognition_model.h5'`).
4. (Optional) For training, download the FER2013 dataset from Kaggle, extract it to `images/train` and `images/test`, and run `trainmodel.ipynb` to generate `emotion_recognition_model.h5`.
5. Open a terminal or command prompt.
6. Navigate to the directory containing `Real-time Detection.py`.
7. Install the required libraries (see Prerequisites).
8. Run the program:
   ```bash
   python "Real-time Detection.py"
   ```
9. The webcam will display a live feed with detected faces outlined in green and predicted emotions labeled above.
10. Press the `q` key to exit.

## Application Flow
- The program initializes the default webcam using OpenCV.
- Frames are flipped horizontally for a mirror-like view and converted to grayscale for face detection.
- OpenCV’s Haar cascade classifier detects faces, returning bounding rectangles.
- Each detected face is resized to 48x48 pixels, normalized, and fed into the CNN model.
- The model predicts probabilities for seven emotions, displaying the highest-scoring emotion above the face.
- The processed frame, with green rectangles and emotion labels, is shown in a window named "Real-time Face Emotion Detection".
- The loop exits on `q` key press, with webcam resources cleaned up.

## Code Structure
- **Main Script**: `Real-time Detection.py` manages webcam capture, face detection, emotion prediction, and display.
- **Model Training (Optional)**: `trainmodel.ipynb` (Jupyter Notebook) trains the CNN model on the FER2013 dataset, saving it as `emotion_recognition_model.h5`.
- **Webcam Capture**: Uses `cv2.VideoCapture(0)` for default webcam access.
- **Face Detection**: Uses `cv2.CascadeClassifier` with Haar cascades for frontal face detection.
- **Emotion Prediction**: Loads the CNN model with `tensorflow.keras.models.load_model` to predict emotions from 48x48 grayscale images.
- **Exit and Cleanup**: Terminates on `q` key press, releasing webcam and closing windows.

## Example Output
```
[Webcam Window: "Real-time Face Emotion Detection"]
- Displays live webcam feed
- Draws green rectangles around detected faces
- Labels predicted emotions above each face (e.g., "happy", "sad")
```

## Limitations
- Requires a functional webcam; fails if no camera is detected.
- Depends on `emotion_recognition_model.h5` being in the specified path.
- Haar cascade detection may miss non-frontal faces, occluded faces, or faces in poor lighting.
- Model accuracy (~61% validation accuracy) may struggle with emotions like disgust or fear due to FER2013 dataset imbalances.
- Limited to effective detection of one face at a time; multiple faces may yield inconsistent results.

## Potential Improvements
- Add webcam availability check:
  ```python
  if not cap.isOpened():
      print("Error: Webcam not detected")
      exit()
  ```
- Use advanced face detectors (e.g., MediaPipe, MTCNN) for better robustness:
  ```python
  import mediapipe as mp
  mp_face_detection = mp.solutions.face_detection
  ```
- Enhance model accuracy with data augmentation during training:
  ```python
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1)
  ```
- Display confidence scores with emotion labels:
  ```python
  confidence = predictions[predicted_label_index]
  cv2.putText(frame, f"{predicted_emotion_text}: {confidence:.2f}", ...)
  ```
- Record output video with emotion labels:
  ```python
  out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame.shape[1], frame.shape[0]))
  out.write(frame)
  out.release()
  ```

## License
This project is unlicensed and free to use or modify.