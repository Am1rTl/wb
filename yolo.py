from ultralytics import YOLO
import cv2
from PIL import ImageGrab
import numpy as np
import math

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person"]

while True:
    img = ImageGrab.grab(bbox=(960, 0, 1920, 1080))
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    results = model(img_np, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if cls == 0:  # Проверка, что cls равно 0 (единственный класс "person")
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img_np, classNames[cls], org, font, fontScale, color, thickness)

    # Изменение размера окна на 500x500 пикселей
    img_np = cv2.resize(img_np, (960, 1080))

    cv2.imshow('Screen Capture', img_np)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

