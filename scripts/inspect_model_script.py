
import torch
import sys
import os

def inspect_model(model_path):
    print(f"Inspecting model at: {model_path}")
    try:
        # Try loading as a full model or dictionary
        content = torch.load(model_path, map_location='cpu')
        print(f"Type of content: {type(content)}")
        
        if isinstance(content, dict):
            print("Keys in dictionary:", content.keys())
            if 'model' in content:
                print("Found 'model' key.")
                # print(content['model']) # Might be too large
            if 'names' in content:
                print("Classes found in 'names':")
                print(content['names'])
            if 'epoch' in content:
                print(f"Epoch: {content['epoch']}")
                
        elif hasattr(content, 'names'):
             print("Model has 'names' attribute:")
             print(content.names)
        else:
            print("Content is not a dict and doesn't have 'names' attribute directly visible.")
            print(content)

    except Exception as e:
        print(f"Error loading model: {e}")
        # specific check for YOLO via ultralytics if generic torch load fails or is opaque
        try:
            from ultralytics import YOLO
            print("Attempting to load via ultralytics YOLO...")
            model = YOLO(model_path)
            print("Successfully loaded with YOLO.")
            print("Model names:", model.names)
        except ImportError:
            print("Ultralytics not installed or failed to load.")
        except Exception as e2:
            print(f"YOLO load failed: {e2}")

if __name__ == "__main__":
    inspect_model(r"c:\Users\IH ARIK\Desktop\ARIK\PPE Detection\src\models\trained.pt")
