from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

#_--------------For Webcam-----------------
#cap = cv2.VideoCapture(0)     #Object representing video capture device, 0 for internal webcam, 1 for external webcam
#Setting the resolution of the webcam
#cap.set(3, 1280)  #Set width
#cap.set(4, 720)  #Set height

#_--------------For Video File-------------
cap = cv2.VideoCapture('Videos/cars.mp4')


model = YOLO("../Yolo_weights/yolov8l.pt")    #Loading pre-trained YOLOv8 'nano' model


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dot", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair dryer", "toothbrush"
              ]


mask = cv2.imread("Project 1- Car Counter/mask.png")

#Tracking
tracker = Sort(max_age = 20, min_hits = 3, iou_threshold= 0.3)


while True:                 #Infinite loop to continuously capture frames from the webcam
    success, img = cap.read()  #success will be 1 if frame was captured successfully, 0 otherwise, and the loop will end. Img will be a numpy array of the captured frame
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream = True) #Perform inference on the captured frame, stream=True allows for real-time processing
    
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes     #Return a list of detected bounding boxes

        for box in boxes:   
            
            #Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)     #Convert coordinates from tensors to integers
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1
            bbox = x1, y1, w, h
            
            
            #Confidence
            conf = math.ceil((box.conf[0]*100))/100
            
            #Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and conf > 0.3:
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0,x1), max(35, y1)), scale = 0.6, thickness = 1, offset =3)
                cvzone.cornerRect(img, (x1, y1, w, h), l = 8)

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        print(result)
    
    cv2.imshow('Image', img)    #Display the captured frame in a window named 'Image'
    cv2.imshow('ImageRegion', imgRegion)  #Display the masked region in a window named 'ImageRegion'

    if cv2.waitKey(1) != -1:
        break

cap.release()
cv2.destroyAllWindows()