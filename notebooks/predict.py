from ultralytics import YOLO

model = YOLO(r"D:\projects\tennis_analysis\models\yolo26n.pt")

result = model.track(r"D:\projects\tennis_analysis\data\input\input_video.mp4", save=True)

# print(result)

# for box in result[0].boxes:
#     print(box)


