from ultralytics import YOLO

# Load the model
model = YOLO("https://huggingface.co/yihong1120/Construction-Hazard-Detection/resolve/main/models/yolo11/pt/yolo11n.pt")

# Print class information
print(f"Total number of classes: {len(model.names)}")
print("\nClass names:")
for idx, class_name in model.names.items():
    print(f"  {idx}: {class_name}")
