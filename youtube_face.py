import math
import time
import cv2
import cvzone
from ultralytics import YOLO

confidence = 0.6

#cap = cv2.VideoCapture(1)  # For Webcam
#cap.set(3, 640)
#cap.set(4, 480)
cap = cv2.VideoCapture("videoplayback.mp4a")  # Corrected for video file
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("yolov8n.pt")

classNames = ["fake", "real"]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        break  # Break if no frame is returned (video ends)

    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            if conf > confidence:

                if classNames[cls] == 'real':
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                   colorB=color)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    # Display FPS on the video
    cvzone.putTextRect(img, f'FPS: {int(fps)}', (20, 50), scale=2, thickness=4, colorR=(255, 255, 255), colorB=(0, 0, 0))

    # Show the image with bounding boxes
    cv2.imshow("Image", img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
