import cv2
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
print("Camera frame:", ok, frame.shape if ok else None)
cap.release()
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model("https://ultralytics.com/images/bus.jpg")
for b in results[0].boxes:
    cls_id = int(b.cls[0])
    conf = float(b.conf[0])
    print(model.names[cls_id], f"{conf:.2f}")
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

i = 0
while i < 50:  # 50 кадров для проверки
    ok, frame = cap.read()
    if not ok:
        print("No frame")
        break
    res = model(frame, verbose=False)
    for box in res[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{model.names[cls_id]} {conf:.2f}", (x1, max(0,y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imwrite(f"frame_{i:03d}.jpg", frame)  # просто сохраняем
    i += 1

cap.release()
print("Saved frames to current folder")
cv2.namedWindow("YOLO Live", cv2.WINDOW_NORMAL)
cv2.imshow("YOLO Live", frame)
