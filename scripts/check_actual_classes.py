from ultralytics import YOLO

# Load the model
model = YOLO("https://huggingface.co/yihong1120/Construction-Hazard-Detection/resolve/main/models/yolo11/pt/yolo11n.pt")

# Print all class names with their IDs
print("Actual model class names:")
for idx, class_name in model.names.items():
    print(f"  {idx}: {class_name}")

print(f"\nCurrent ALLOWED_CLASSES: {0, 1, 2, 3, 4, 5, 8}")
print("These correspond to:")
for class_id in [0, 1, 2, 3, 4, 5, 8]:
    if class_id in model.names:
        print(f"  {class_id}: {model.names[class_id]}")
    else:
        print(f"  {class_id}: NOT FOUND")
