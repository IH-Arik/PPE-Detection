"""
Test YOLOv8s model to see available classes
"""

from ultralytics import YOLO

def test_yolov8s():
    """Test YOLOv8s model and show classes"""
    print("🔍 Testing YOLOv8s Model...")
    print("=" * 50)
    
    try:
        # Load YOLOv8s model
        model_path = "src/models/yolov8s.pt"
        print(f"Loading model from: {model_path}")
        
        model = YOLO(model_path)
        
        print(f"✅ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Total number of classes: {len(model.names)}")
        print()
        
        print("📋 Available Classes:")
        print("-" * 30)
        
        for idx, class_name in model.names.items():
            print(f"  {idx: {class_name}")
        
        print()
        print("🎯 Test Detection:")
        print("-" * 30)
        
        # Test with a sample image (create a dummy one if needed)
        import numpy as np
        import cv2
        
        # Create a simple test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (100, 100, 100)  # Gray background
        
        print("Running detection on test image...")
        results = model(test_image, conf=0.25)
        
        print(f"Detection results: {len(results)}")
        
        if results:
            for i, result in enumerate(results):
                print(f"Result {i+1}:")
                print(f"  - Boxes: {len(result.boxes) if result.boxes else 0}")
                if result.boxes:
                    for j, box in enumerate(result.boxes):
                        class_id = int(box.cls)
                        class_name = model.names[class_id]
                        confidence = float(box.conf)
                        print(f"    Box {j+1}: {class_name} (conf: {confidence:.3f})")
        
        print()
        print("✅ YOLOv8s test completed!")
        
    except Exception as e:
        print(f"❌ Error testing YOLOv8s: {e}")
        print("Make sure the model file exists at src/models/yolov8s.pt")

if __name__ == "__main__":
    test_yolov8s()
