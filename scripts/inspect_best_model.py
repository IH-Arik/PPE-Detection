
import torch
import sys
import os
from ultralytics import YOLO

def inspect_model(model_path):
    print(f"Inspecting model at: {model_path}")
    try:
        model = YOLO(model_path)
        print("\n--- MODEL CLASSES FOUND (via YOLO) ---")
        print(model.names) 
        print("--- END MODEL CLASSES ---\n")
    except Exception as e:
        print(f"YOLO load failed: {e}")

if __name__ == "__main__":
    inspect_model(r"c:\Users\IH ARIK\Desktop\ARIK\PPE Detection\src\models\best.pt")
