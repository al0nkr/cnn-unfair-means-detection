from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import datetime
import matplotlib.pyplot as plt

model = YOLO(r"G:\cnn-detector\runs\detect\train9\weights\best.pt")
video_path = r'G:\cnn-detector\cctv.mp4'
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object
output_path = 'path_to_output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
alerts_dir = 'alerts'
os.makedirs(alerts_dir, exist_ok=True)

# Initialize variables for alert logging
alert_classes = ["turningHead", "usingPhone","usingIpad"]
consecutive_count = 0
alert_threshold = 15

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model.predict(frame, imgsz=1280)

    # Draw bounding boxes and labels on the frame
    detected = False
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            if model.names[cls] in alert_classes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )
                detected = True

    # Check for consecutive detections
    if detected:
        consecutive_count += 1
        if consecutive_count >= alert_threshold:
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            utc_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            alert_path = os.path.join(alerts_dir, f'{model.names[cls]}_{frame_number}_{utc_timestamp}.jpg')
            cv2.imwrite(alert_path, frame)
            consecutive_count = 0  # Reset count after saving alert
    else:
        consecutive_count = 0

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
