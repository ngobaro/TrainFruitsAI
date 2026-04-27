from ultralytics import YOLO

# Train model
model = YOLO("yolov8s.pt")        # yolov8n.pt nếu máy yếu, yolov8m.pt nếu muốn chính xác hơn

results = model.train(
    data="dataset/data.yaml",     # đường dẫn tương đối
    epochs=60,
    imgsz=640,
    batch=8,                      # Windows local: điều chỉnh theo VRAM của bạn
    name="fruit4_yolov8s",
    patience=20,
    plots=True,
    pretrained=True,
    device=0                      # dùng GPU (nếu có), dùng 'cpu' nếu không có GPU
)