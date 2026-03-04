
import torch
import sys
import os

def inspect_model(model_path):
    # Redirect stderr to stdout to capture everything
    sys.stderr = sys.stdout
    print(f"Inspecting model at: {model_path}")
    
    try:
        # 1. Try generic torch load
        content = torch.load(model_path, map_location='cpu')
        
        names = None
        
        # Check if content itself has names
        if hasattr(content, 'names'):
            names = content.names
        # Check if content is a dict and has 'names' key
        elif isinstance(content, dict) and 'names' in content:
            names = content['names']
        # Check if content is a dict and has 'model' key which might have names
        elif isinstance(content, dict) and 'model' in content:
            model_part = content['model']
            if hasattr(model_part, 'names'):
                names = model_part.names
            elif isinstance(model_part, dict) and 'names' in model_part:
                names = model_part['names']

        if names:
            print("\n--- MODEL CLASSES FOUND ---")
            # Handle different formats of names (list vs dict)
            if isinstance(names, dict):
                for k, v in names.items():
                    print(f"{k}: {v}")
            elif isinstance(names, list):
                for i, v in enumerate(names):
                    print(f"{i}: {v}")
            else:
                print(f"Names found but unknown format: {names}")
            print("--- END MODEL CLASSES ---\n")
            return

        print("Generic load did not find 'names'. Trying Ultralytics YOLO...")

    except Exception as e:
        print(f"Generic load failed: {e}")

    # 2. Try Ultralytics YOLO
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("\n--- MODEL CLASSES FOUND (via YOLO) ---")
        print(model.names) # Should print the names dict
        print("--- END MODEL CLASSES ---\n")
    except ImportError:
        print("Ultralytics library not found.")
    except Exception as e:
        print(f"YOLO load failed: {e}")

if __name__ == "__main__":
    inspect_model(r"c:\Users\IH ARIK\Desktop\ARIK\PPE Detection\src\models\trained.pt")
