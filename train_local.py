from ultralytics import YOLO

model = YOLO("yolov8s.pt")   # hoặc yolov8n.pt, yolov8m.pt

results = model.train(
    data="dataset/data.yaml",
    epochs=60,
    imgsz=640,
    batch=8,                 # giảm nếu máy yếu (4 hoặc 8)
    name="fruit5_yolov8s",
    patience=20,
    plots=True
)