from ultralytics import YOLO
# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolo11m.pt")

# Display model information (optional)
model.info()

if __name__ == '__main__':
    # Train the model on the COCO8 example dataset for 100 epochs
    model.train(data=r"G:\cnn-detector\new_students_data\data.yaml", epochs=70, imgsz=1280, batch=0.85,device = 0)

    # Run inference with the YOLOv8n model on the 'bus.jpg' image
    results = model(r"G:\cnn-detector\new_students_data\valid\images\frame_0017_jpg.rf.76f15f0a0054559e64004056ea8fa425.jpg")