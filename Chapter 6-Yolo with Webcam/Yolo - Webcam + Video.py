from ultralytics import YOLO
import cv2
import cvzone
import math

#_--------------For Webcam-----------------
#cap = cv2.VideoCapture(0)     #Object representing video capture device, 0 for internal webcam, 1 for external webcam
#Setting the resolution of the webcam
#cap.set(3, 1280)  #Set width
#cap.set(4, 720)  #Set height

#_--------------For Video File-------------
cap = cv2.VideoCapture('Videos/PUBG.mp4')


model = YOLO("../Yolo_weights/yolov8n.pt")    #Loading pre-trained YOLOv8 'nano' model


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


while True:                 #Infinite loop to continuously capture frames from the webcam
    success, img = cap.read()  #success will be 1 if frame was captured successfully, 0 otherwise, and the loop will end. Img will be a numpy array of the captured frame
    results = model(img, stream = True) #Perform inference on the captured frame, stream=True allows for real-time processing
    
    for r in results:
        boxes = r.boxes     #Return a list of detected bounding boxes

        for box in boxes:   
            
            #Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)     #Convert coordinates from tensors to integers
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1
            bbox = x1, y1, w, h
            cvzone.cornerRect(img, bbox)
            
            #Confidence
            conf = math.ceil((box.conf[0]*100))/100
            
            #Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0,x1), max(35, y1)), scale = 1, thickness = 1)


    
    cv2.imshow('Image', img)    #Display the captured frame in a window named 'Image'
    cv2.waitKey(1)

    
