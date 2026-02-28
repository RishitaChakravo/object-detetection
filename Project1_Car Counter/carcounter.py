from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

cap = cv2.VideoCapture('../Videos/video.mp4')

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

model = YOLO('../Yolo-Weights/yolov8n.pt')
mask = cv2.imread("mask.png")
mask = cv2.resize(mask, (1280, 720))
limits = [400, 415, 1200, 415] 
totalcnts = []

# Tracking
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    if not success:
        break
    imgRegion = cv2.bitwise_and(img, mask)
    
    results = model(imgRegion, stream=True)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            bbox = int(x1), int(y1), int(w), int(h)

            conf = (math.ceil(box.conf[0]*100))/100
            cls = int(box.cls[0])
            classname = classNames[cls]

            if classname == "car" or classname == "truck" or classname == "bus" or classname == "motorbike" and conf > 0.3:  
                # cvzone.cornerRect(img, bbox, l=5, rt=5)  
                cvzone.putTextRect(img, f'{classname} {conf}', (max(0,x1), max(0,y1-20)), scale=0.6, thickness=1)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    result_Tracker = tracker.update(detections)
    cv2.line(img,(limits[0], limits[1]),(limits[2], limits[3]), (0, 0, 255), 5)

    for result in result_Tracker:
        x1, y1,x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=5, rt=2, colorR=(255,0,255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0,x1), max(0,y1-20)), scale=2, thickness=10, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img,(cx, cy),5,(255, 0, 255), cv2.FILLED)

        if limits[0]< cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalcnts.count(id) == 0:
                totalcnts.append(id)
                cv2.line(img,(limits[0], limits[1]),(limits[2], limits[3]), (0, 255, 0), 5)

    cvzone.putTextRect(img, f'{int(len(totalcnts))}', (50,50))
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)