from ultralytics import YOLO
import os

# Load the PyTorch model
print("Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")

# Export the model to TFLite format
print("Exporting to TFLite format...")
try:
    # First try without INT8 quantization
    model.export(format="tflite", imgsz=640, optimize=True)
    print("✅ Successfully exported to TFLite!")
    
    # Check if the file was created
    if os.path.exists("yolov8n.tflite"):
        size_mb = os.path.getsize("yolov8n.tflite") / (1024 * 1024)
        print(f"✅ TFLite model created: yolov8n.tflite ({size_mb:.2f} MB)")
        
        # Compare with original size
        pt_size = os.path.getsize("yolov8n.pt") / (1024 * 1024)
        reduction = ((pt_size - size_mb) / pt_size * 100)
        print(f"Size reduction: {reduction:.1f}%")
    else:
        print("❌ TFLite file was not created")
        
except Exception as e:
    print(f"❌ Export failed: {e}")
    print("Trying alternative export method...")
    
    try:
        # Try with different parameters
        model.export(format="tflite", imgsz=640)
        print("✅ Alternative export successful!")
    except Exception as e2:
        print(f"❌ Alternative export also failed: {e2}")