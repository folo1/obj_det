from ultralytics import YOLO


# Load a COCO-pretrained YOLO11n model
model = YOLO("yolov8n.pt")

#results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Train the model on the COCO8 example dataset for 100 epochs
results = model(0, show = True)
