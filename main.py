import torch
import cv2
from ultralytics import YOLO
from deepface import DeepFace
import warnings
import threading
import time

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO('yolov8n.pt')  # Using the nano model for faster performance

detector_backend = 'retinaface'  # Try different backends if needed

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

video_stream = VideoStream(src=0)

if not video_stream.ret:
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow('Real-time Object Detection and Emotion Recognition', cv2.WINDOW_NORMAL)

frame_count = 0

last_emotion_results = []

while True:
    ret, frame = video_stream.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame_count += 1

    results = model(frame, device=device, verbose=False, classes=[0])  # Only detect 'person' class

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box

    if frame_count % 5 == 0:
        try:
            emotion_results = []
            recognize_emotion(frame.copy(), emotion_results)
            last_emotion_results = emotion_results
        except Exception as e:
            print(f"Error in emotion recognition: {e}")
            last_emotion_results = []
    else:
        emotion_results = last_emotion_results

    for face in emotion_results:
        x = face['region']['x']
        y = face['region']['y']
        w = face['region']['w']
        h = face['region']['h']
        dominant_emotion = face['dominant_emotion']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  
        cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)  
        print(f"Detected emotion: {dominant_emotion}")

    cv2.imshow('Real-time Object Detection and Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.stop()
cv2.destroyAllWindows()
