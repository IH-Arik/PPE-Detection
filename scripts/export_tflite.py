"""Convert yolo26.pt to TFLite for Android deployment"""
from ultralytics import YOLO

model = YOLO("src/models/yolo26.pt")

# Export to TFLite (float16 — lightweight, no calibration needed)
result = model.export(format="tflite", half=True, imgsz=320)
print(f"\n✅ TFLite exported to: {result}")

import os
if result and os.path.exists(result):
    size_kb = os.path.getsize(result) / 1024
    print(f"📦 Size: {size_kb:.1f} KB ({size_kb/1024:.1f} MB)")
    print(f"\n📱 Ready for Android! Copy this file to your Android app's assets folder.")
