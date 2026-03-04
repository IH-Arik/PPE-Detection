"""
PPE Detection — Training + End2End TFLite Export
=================================================
Produces a model with BUILT-IN NMS that outputs [1, 300, 6]
where each row = [x1, y1, x2, y2, confidence, class_id]
with normalized 0-1 coordinates.

This is exactly what index.tsx expects:
  - Input:  320x320 RGB float32
  - Output: [1, 300, 6] (end2end NMS baked in)

Labels (11 classes, must match data.yaml order):
  0: boots
  1: glasses
  2: gloves
  3: helmet
  4: no boots
  5: no glasses
  6: no gloves
  7: no helmet
  8: no vest
  9: person
  10: vest
"""

from ultralytics import YOLO

# ──────────────────────────────────────────────
# 1. TRAINING
# ──────────────────────────────────────────────
model = YOLO("yolo11n.pt")  # YOLOv11-nano base

results = model.train(
    data="/content/PPEv5-17/data.yaml",
    epochs=200,
    imgsz=320,               # Must match INPUT_SIZE in index.tsx
    batch=32,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,                # Final LR = lr0 * lrf (cosine decay)
    warmup_epochs=5,
    warmup_momentum=0.5,
    weight_decay=0.0005,
    device=[0, 1],
    patience=30,             # Early stopping patience

    # ── Augmentation for small object / PPE scenarios ──
    hsv_h=0.015,             # Hue jitter
    hsv_s=0.7,               # Saturation jitter
    hsv_v=0.4,               # Value/brightness jitter
    degrees=10.0,            # Rotation ±10°
    translate=0.15,          # Translation ±15%
    scale=0.5,               # Scale +50%
    shear=5.0,               # Shear ±5°
    flipud=0.1,              # Vertical flip 10%
    fliplr=0.5,              # Horizontal flip 50%
    mosaic=1.0,              # Mosaic augmentation
    mixup=0.1,               # Mixup augmentation
    copy_paste=0.1,          # Copy-paste augmentation (helps small objects)
    erasing=0.2,             # Random erasing

    # ── Losses ──
    box=7.5,                 # Box loss gain
    cls=0.5,                 # Classification loss gain
    dfl=1.5,                 # Distribution focal loss gain

    # ── Other ──
    cos_lr=True,             # Cosine LR schedule
    close_mosaic=20,         # Disable mosaic last 20 epochs
    amp=True,                # Mixed precision training
    workers=8,
    project="ppe_training",
    name="yolo11n_ppe_e2e",
    save=True,
    save_period=25,          # Checkpoint every 25 epochs
    val=True,
    plots=True,
    verbose=True,
)

# ──────────────────────────────────────────────
# 2. EXPORT — End2End TFLite with built-in NMS
# ──────────────────────────────────────────────
# Load the best checkpoint from training
best_model = YOLO("ppe_training/yolo11n_ppe_e2e/weights/best.pt")

# Export to TFLite with:
#   - nms=True        → bakes NMS into the model graph (end2end)
#   - half=True       → float16 quantization (smaller + faster on mobile)
#   - imgsz=320       → fixed input size matching index.tsx
#   - max_det=300     → matches MAX_DETECTIONS in index.tsx
#
# Output shape will be [1, 300, 6]:
#   [x1, y1, x2, y2, confidence, class_id]
#   coordinates are normalized 0–1
best_model.export(
    format="tflite",
    imgsz=320,
    half=True,                # float16 quantization
    nms=True,                 # ← CRITICAL: bakes NMS into graph → [1, 300, 6]
    max_det=300,              # matches MAX_DETECTIONS in index.tsx
    agnostic_nms=False,
    simplify=True,
)

print("\n✅ Done! Your TFLite model is ready.")
print("📁 Look for: ppe_training/yolo11n_ppe_e2e/weights/best_float16.tflite")
print("📋 Output shape: [1, 300, 6] → [x1, y1, x2, y2, conf, class_id]")
print("🚀 Copy it to: assets/models/best_float16.tflite")