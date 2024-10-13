import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "/home/jupyter/yolov5ByteTrack/best.pt")  # or yolov5n - yolov5x6, custom

# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  

python detect.py --weights "/home/jupyter/yolov5ByteTrack/best.pt" --source "/home/jupyter/yolov5ByteTrack/0a13ae67-6a72-44e4-ab3d-0feab065a482_1_7_1_9_2024-08-26T03-42-41_043000200049.mp4"                           # webcam
