from ultralytics import YOLO
import cv2
# without cn\v2 there is no delay so when i run the file no output because it opens and closes to quickly

model = YOLO('../Yolo-Weights/yolov8n.pt')
results = model("Images/3.png",show=True)

annotated_frame = results[0].plot()

cv2.imshow("Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()