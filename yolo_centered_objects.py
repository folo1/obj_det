from ultralytics import YOLO
import cv2

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolov8n.pt",verbose = False)

# return a generator of Results objects
results = model(0, stream=True, verbose=False)  

d_index = 0
# Process results generator
for result in results:
    print(d_index)
    d_index += 1
    boxes = result.boxes  # Boxes object for bounding box outputs
    class_ids = boxes.cls # class id (not names)
    class_locs = boxes.xyxy # upper left and lower right points.
    
    img = result.orig_img.copy()
    for index in range(0,len(class_ids)):
        # get the class_id for the current detection
        id = class_ids[index]
        # get the center of the current detection

        loc = class_locs[index]
        center_x = int((loc[2]+loc[0])/2)
        center_y = int((loc[3]+loc[1])/2)

        # show a dot in the center with a class label
        color = (0,50,0)
        white = (255,255,255)
        cv2.rectangle(img,(center_x-2,center_y-2),(center_x+2,center_y+2),color,4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # white outline
        cv2.putText(img, result.names[int(id)], (center_x, center_y), font, 2, white, 5, cv2.LINE_AA)
        # custom color inside
        cv2.putText(img, result.names[int(id)], (center_x, center_y), font, 2, color, 2, cv2.LINE_AA)

        # print the class name and center location of each detection
        print("%14s: center x,y = %5d,%5d" 
              % (result.names[int(id)],center_x,center_y))
    
    # display to screen
    cv2.imshow('yolo', img) 
    # quit on 'Q' key presed 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    pass

