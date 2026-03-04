from ultralytics import YOLO
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="melihuzunoglu/ppe-detection", filename="best.pt")
model = YOLO(model_path)
results = model.predict(source="your_image.jpg", conf=0.25, save=True)