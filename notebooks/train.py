from ultralytics import YOLO

# Load pretrained model
model = YOLO("models/yolo26n.pt")

# Train on tennis ball dataset
results = model.train(
    data="data/dataset/data.yaml",
    epochs=25,
    imgsz=640
)