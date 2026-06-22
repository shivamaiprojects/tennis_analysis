from ultralytics import YOLO

model = YOLO(r"models\yolo26n.pt")

result = model.predict(r"D:\projects\tennis_analysis\data\input_video.mp4", save=True)

print(result)

for box in result[0].boxes:
    print(box)
