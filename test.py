# import torch
# import cv2
# import numpy as np
# from torchvision import models, transforms
# from torchvision.models.detection import fasterrcnn_resnet50_fpn

# # Load the trained action detection model's state dictionary
# state_dict = torch.load('action_detection_model89.67%.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# # Define the action detection model architecture
# action_model = models.resnet50(pretrained=True)
# num_ftrs = action_model.fc.in_features
# action_model.fc = torch.nn.Linear(num_ftrs, 6)
# action_model.load_state_dict(state_dict)
# action_model.eval()

# # Load the object detection model
# object_detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
# object_detection_model.eval()

# # Define the preprocessing transform
# preprocess = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Function to perform action detection on a single frame
# def detect_action(frame, action_model, object_detection_model):
#     # Perform object detection
#     transform = transforms.ToTensor()
#     frame_tensor = transform(frame).unsqueeze(0)
#     with torch.no_grad():
#         detections = object_detection_model(frame_tensor)[0]

#     # Filter detections for person class (class_id = 1) with confidence > 0.8
#     person_detections = []
#     for i, label in enumerate(detections['labels']):
#         if label == 1 and detections['scores'][i] > 0.7:
#             person_detections.append(detections['boxes'][i])

#     action_detections = []
#     for bbox in person_detections:
#         x1, y1, x2, y2 = map(int, bbox)
#         person_crop = frame[y1:y2, x1:x2]

#         # Preprocess the cropped person image
#         input_tensor = preprocess(person_crop).unsqueeze(0)

#         # Perform action detection
#         with torch.no_grad():
#             outputs = action_model(input_tensor)
#             probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()

#         # Get the class with the highest probability
#         print(probabilities)
#         class_id = np.argmax(probabilities)
#         confidence = probabilities[class_id]

#         action_detections.append((x1, y1, x2 - x1, y2 - y1, class_id, confidence))

#     return action_detections

# def draw_detections(frame, detections):
#     for detection in detections:
#         x, y, w, h, class_id, confidence = detection
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, f'Class: {class_id}, Conf: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     return frame

# # Load a single frame
# frame = cv2.imread(r'G:\cnn-detector\new_students_data\valid\images\frame_0049_jpg.rf.f6377ac7a71c8463cf311d62860a7e7c.jpg')

# # Detect actions in the frame
# detections = detect_action(frame, action_model, object_detection_model)

# # Draw detections on the frame
# frame = draw_detections(frame, detections)

# # Save or display the frame
# cv2.imshow('Detections', frame)
# cv2.imwrite('output_frame.jpg', frame)

# cv2.waitKey(0)
# cv2.destroyAllWindows()