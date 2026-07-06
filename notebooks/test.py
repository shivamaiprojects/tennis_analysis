from ultralytics import YOLO

model = YOLO("models/yolo26n.pt")

metrics = model.val(
    data="data/dataset/data.yaml",
    split="test"
)