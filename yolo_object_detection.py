from ultralytics import YOLO


# Load a COCO-pretrained YOLO11n model
model = YOLO("yolov8n.pt")

#results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Train the model on the COCO8 example dataset for 100 epochs
results = model(0, stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    #boxes = result.boxes  # Boxes object for bounding box outputs
    #masks = result.masks  # Masks object for segmentation masks outputs
    #keypoints = result.keypoints  # Keypoints object for pose outputs
    #probs = result.probs  # Probs object for classification outputs
    %obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    #result.save(filename="result.jpg")  # save to disk
