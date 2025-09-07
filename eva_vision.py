from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

from main import describe_scene  # импорт один раз, в начале файла

names = model.names  # если модель загружена через YOLOv5 или YOLOv8
model = YOLO("yolov8n.pt")  # пример
names = model.names

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes
    detected = []

    for box in boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        detected.append(label)

    description = ", ".join(set(detected)) if detected else "ничего не видно"
    print(f"EVA видит: {description}")

    gpt_response = describe_scene(description)
    print(f"EVA говорит: {gpt_response}")

    annotated_frame = results[0].plot()
    cv2.imshow("EVA sees", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # теперь это внутри цикла — всё ок!

cap.release()
cv2.destroyAllWindows()


