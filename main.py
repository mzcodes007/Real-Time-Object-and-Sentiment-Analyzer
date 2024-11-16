import torch
import cv2
from ultralytics import YOLO
from deepface import DeepFace
import warnings
import threading
import time

# Suppress warnings from YOLO and DeepFace
warnings.filterwarnings("ignore")

# Check if CUDA is available and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the pretrained YOLO model (use a smaller model for speed)
model = YOLO('yolov8n.pt')  # Using the nano model for faster performance

# Detector backend for DeepFace
detector_backend = 'retinaface'  # Try different backends if needed

# VideoStream class to read frames in a separate thread
class VideoStream:
    def __init__(self, src=0, width=640, height=480):  # Try higher resolution
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()

# Define the recognize_emotion function
def recognize_emotion(frame, results):
    try:
        face_analysis = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=True,  # Enforce face detection
            detector_backend=detector_backend
        )
        print("Face analysis results:", face_analysis)
        # Ensure face_analysis is a list
        if isinstance(face_analysis, list):
            results.extend(face_analysis)
        else:
            results.append(face_analysis)
    except Exception as e:
        print(f"Error in emotion recognition: {e}")

# Initialize the video stream
video_stream = VideoStream(src=0)

# Check if the camera opened successfully
if not video_stream.ret:
    print("Error: Could not open camera.")
    exit()

# Create a named window
cv2.namedWindow('Real-time Object Detection and Emotion Recognition', cv2.WINDOW_NORMAL)

# Initialize frame counter
frame_count = 0

# Initialize cache for emotion results
last_emotion_results = []

# Loop for real-time object detection and emotion recognition
while True:
    # Capture frame-by-frame
    ret, frame = video_stream.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame_count += 1

    # Run YOLO object detection on the frame
    results = model(frame, device=device, verbose=False, classes=[0])  # Only detect 'person' class

    # Process detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box

    # Perform emotion recognition every 5 frames
    if frame_count % 5 == 0:
        try:
            # Run emotion recognition
            emotion_results = []
            recognize_emotion(frame.copy(), emotion_results)
            last_emotion_results = emotion_results
        except Exception as e:
            print(f"Error in emotion recognition: {e}")
            last_emotion_results = []
    else:
        # Use the last emotion results
        emotion_results = last_emotion_results

    # Draw emotion results
    for face in emotion_results:
        x = face['region']['x']
        y = face['region']['y']
        w = face['region']['w']
        h = face['region']['h']
        dominant_emotion = face['dominant_emotion']
        # Draw the face bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue bounding box
        # Draw the emotion label
        cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)  # Blue emotion label text
        print(f"Detected emotion: {dominant_emotion}")

    # Display the resulting frame
    cv2.imshow('Real-time Object Detection and Emotion Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
video_stream.stop()
cv2.destroyAllWindows()
