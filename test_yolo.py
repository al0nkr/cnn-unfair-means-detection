from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


### train 7 - yolo v8m 50 ep 640 imgsz
### train 9 - yolo 11l 50 ep 640 imgsz
### train 3 - yolo 11m 70 ep 1280 imgsz
# Load the YOLOv8 model
model = YOLO(r"G:\cnn-detector\runs\detect\train9\weights\best.pt")

# Load the sample image
image_path = r'G:\cnn-detector\new_students_data\train\images\frame_1217_jpg.rf.a98d965cd5e600813998621ed776e467.jpg'
image = Image.open(image_path)

# Perform inference
results = model.predict(image)
for result in results:
    result.show()
print(results)
# Extract bounding boxes and classes
bboxes = results[0].boxes.cpu().numpy()  # Bounding boxes
print(bboxes)
# Open the input video
# video_path = r'G:\cnn-detector\cctv.mp4'
# cap = cv2.VideoCapture(video_path)

# # Define the codec and create VideoWriter object
# output_path = 'path_to_output_video.mp4'
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# # Process each frame
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform inference
#     results = model.predict(frame,imgsz=1280,conf=0.25)

#     # Draw bounding boxes and labels on the frame
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = box.xyxy[0].int().tolist()
#             conf = box.conf[0]
#             cls = int(box.cls[0])
#             label = f"{model.names[cls]} {conf:.2f}"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(
#                 frame,
#                 label,
#                 (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.9,
#                 (0, 255, 0),
#                 2
#             )

#     # Write the frame to the output video
#     out.write(frame)

#     # Display the frame
#     cv2.imshow('Output', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()