import torch
from ultralytics import YOLO
import os

def simple_convert():
    """
    Simple conversion using YOLO's built-in export functionality
    """
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    
    print("Exporting to TFLite format...")
    try:
        # Export directly to TFLite
        model.export(format='tflite', imgsz=640, optimize=True)
        print("Successfully exported to TFLite!")
        
        # Check if the file was created
        tflite_file = 'yolov8n.tflite'
        if os.path.exists(tflite_file):
            size_mb = os.path.getsize(tflite_file) / (1024 * 1024)
            print(f"TFLite model created: {tflite_file} ({size_mb:.2f} MB)")
            return True
        else:
            print("TFLite file not found after export")
            return False
            
    except Exception as e:
        print(f"Error during export: {e}")
        return False

if __name__ == "__main__":
    print("Starting simple YOLOv8 to TFLite conversion...")
    print("=" * 50)
    
    success = simple_convert()
    
    if success:
        print("=" * 50)
        print("Conversion completed successfully!")
        
        # Compare file sizes
        pt_size = os.path.getsize('yolov8n.pt') / (1024 * 1024)
        tflite_size = os.path.getsize('yolov8n.tflite') / (1024 * 1024)
        
        print(f"Original model size: {pt_size:.2f} MB")
        print(f"TFLite model size: {tflite_size:.2f} MB")
        print(f"Size reduction: {((pt_size - tflite_size) / pt_size * 100):.1f}%")
    else:
        print("Conversion failed.")
