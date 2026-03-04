from ultralytics import YOLO
import os

# Convert ONNX to TFLite using compatible TensorFlow
print("Converting ONNX to TFLite in Docker...")

try:
    # Use YOLO to convert
    model = YOLO('yolov8n.onnx')
    model.export(format='tflite', imgsz=640)
    print("TFLite conversion completed in Docker!")
except Exception as e:
    print(f"Conversion failed: {e}")