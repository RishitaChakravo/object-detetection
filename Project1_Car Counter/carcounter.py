from ultralytics import YOLO
import cv2
import cvzone
import math

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
while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    if not success:
        break
    imgRegion = cv2.bitwise_and(img, mask)
    
    results = model(imgRegion, stream=True)
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
                cvzone.cornerRect(img, bbox, l=5)  
                cvzone.putTextRect(img, f'{classname} {conf}', (max(0,x1), max(0,y1-20)), scale=0.6, thickness=1)

    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)