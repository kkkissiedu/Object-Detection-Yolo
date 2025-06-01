from ultralytics import YOLO
import cv2


model = YOLO('../Yolo_weights/yolov8l.pt')  # Load a pre-trained YOLOv8 'nano' model 
results = model("Chapter 5-Running Yolo/Images/razaak.png", show = True)  #path to image
cv2.waitKey(0)   #Display the image until a key is pressed