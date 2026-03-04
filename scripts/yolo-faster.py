# ===========================
# YOLO-FastestV2 PPE Training + TFLite Export
# ===========================
# Ultra-lightweight detector (~250K params, ~300fps on mobile)
# GitHub: https://github.com/dog-qiuqiu/Yolo-FastestV2
#
# 🔧 This script is designed to run on Kaggle with dual GPUs.
#    It clones the YOLO-FastestV2 repo, prepares a Darknet-style
#    dataset from the Roboflow PPEv5 dataset, generates anchors,
#    trains the model, and exports to ONNX → TFLite.
# ===========================

import os
import sys
import glob
import shutil
import subprocess

# ===========================
# 0. Configuration
# ===========================
DATASET_ROOT = "/kaggle/working/PPEv5-17"       # Roboflow dataset path (YOLO format)
WORK_DIR     = "/kaggle/working"
REPO_DIR     = os.path.join(WORK_DIR, "Yolo-FastestV2")

# PPE Classes (11 classes)
CLASS_NAMES = [
    "boots", "glasses", "gloves", "helmet",
    "no boots", "no glasses", "no gloves", "no helmet",
    "no vest", "person", "vest"
]

# Training hyperparameters
EPOCHS       = 100
BATCH_SIZE   = 64
IMG_SIZE     = 352          # YOLO-FastestV2 default (must be multiple of 32)
LR           = 0.001
STEPS        = "50,80"      # Learning rate decay steps

# ===========================
# 1. Clone YOLO-FastestV2 repo
# ===========================
print("📦 Step 1: Cloning YOLO-FastestV2 repository...")
if not os.path.exists(REPO_DIR):
    subprocess.run(
        ["git", "clone", "https://github.com/dog-qiuqiu/Yolo-FastestV2.git"],
        cwd=WORK_DIR, check=True
    )
else:
    print("   ↳ Repository already exists, skipping clone.")

# Install dependencies
# NOTE: The repo's requirements.txt pins ancient torchvision==0.10.0 which
# is no longer available. Kaggle already has PyTorch & torchvision installed,
# so we only install the extra packages the repo actually needs.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "opencv-python", "tqdm", "pyaml",         # YOLO-FastestV2 deps
     "onnx", "onnxsim", "onnx2tf", "tensorflow"],  # export deps
    check=True
)

# Patch deprecated numpy aliases (np.float, np.int removed in NumPy 1.24+)
print("\n🔧 Patching deprecated NumPy aliases in YOLO-FastestV2 source...")
for py_file in glob.glob(os.path.join(REPO_DIR, "**", "*.py"), recursive=True):
    with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    original = content
    content = content.replace("np.float)", "np.float64)")
    content = content.replace("np.float,", "np.float64,")
    content = content.replace("np.int)", "np.int64)")
    content = content.replace("np.int,", "np.int64,")
    if content != original:
        with open(py_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"   ↳ Patched: {os.path.basename(py_file)}")

# Patch datasets.py: fix label path derivation for Roboflow filenames
# The original code uses img_path.split(".")[0] + ".txt" which breaks on
# filenames with multiple dots like "img_jpg.rf.abc123.jpg" → "img_jpg.txt"
# Fix: use os.path.splitext() which correctly handles this.
datasets_py = os.path.join(REPO_DIR, "utils", "datasets.py")
with open(datasets_py, "r", encoding="utf-8") as f:
    ds_content = f.read()
ds_content = ds_content.replace(
    'label_path = img_path.split(".")[0] + ".txt"',
    'label_path = os.path.splitext(img_path)[0] + ".txt"'
)
with open(datasets_py, "w", encoding="utf-8") as f:
    f.write(ds_content)
print("   ↳ Patched: datasets.py (label path fix for Roboflow filenames)")

# Patch loss.py: fix torch dtype clamp issue on newer PyTorch
def patch_loss_for_new_torch():
    loss_file = os.path.join(REPO_DIR, "utils", "loss.py")
    if not os.path.exists(loss_file):
        print(f"   WARNING: loss.py not found: {loss_file}")
        return

    with open(loss_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    old_line = "indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))"
    if old_line not in content:
        print("   -> loss.py already patched or source line changed")
        return

    new_block = (
        "max_gj = int(gain[3].item()) - 1\n"
        "            max_gi = int(gain[2].item()) - 1\n"
        "            gj = gj.clamp(0, max_gj)\n"
        "            gi = gi.clamp(0, max_gi)\n"
        "            indices.append((b, a, gj, gi))"
    )
    content = content.replace(old_line, new_block)
    with open(loss_file, "w", encoding="utf-8") as f:
        f.write(content)
    print("   -> Patched: loss.py (torch clamp dtype fix)")

# Patch train.py: enable DataParallel for dual GPU training
def patch_train_dual_gpu():
    train_file = os.path.join(REPO_DIR, "train.py")
    with open(train_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # 1. Add DataParallel after model creation
    old_model_line = 'summary(model, input_size=(3, cfg["height"], cfg["width"]))'
    if old_model_line in content and "DataParallel" not in content:
        new_model_block = (
            'summary(model, input_size=(3, cfg["height"], cfg["width"]))\n'
            '\n'
            '    # Enable multi-GPU training\n'
            '    if torch.cuda.device_count() > 1:\n'
            '        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")\n'
            '        model = torch.nn.DataParallel(model)'
        )
        content = content.replace(old_model_line, new_model_block)

        # 2. Fix model save to handle DataParallel's .module attribute
        old_save = 'torch.save(model.state_dict()'
        new_save = 'torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict()'
        content = content.replace(old_save, new_save)

        with open(train_file, "w", encoding="utf-8") as f:
            f.write(content)
        print("   -> Patched: train.py (dual GPU DataParallel)")
    else:
        print("   -> train.py already patched for DataParallel or source changed")

# ===========================
# 2. Prepare Darknet-style dataset
# ===========================
# Roboflow YOLO-format datasets already have images/ and labels/
# subdirectories. YOLO-FastestV2 expects images and .txt labels
# in the SAME directory (Darknet format).
# ===========================
print("\n📁 Step 2: Preparing dataset in Darknet format...")

data_dir = os.path.join(REPO_DIR, "data", "ppe")
# Clean up old data from previous runs
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.makedirs(data_dir, exist_ok=True)


def prepare_split(split_name):
    """
    Copy images and labels into a single flat directory
    and generate a path listing file (train.txt / val.txt).
    """
    src_img_dir = os.path.join(DATASET_ROOT, split_name, "images")
    src_lbl_dir = os.path.join(DATASET_ROOT, split_name, "labels")

    dst_dir = os.path.join(data_dir, split_name)
    os.makedirs(dst_dir, exist_ok=True)

    # Copy images
    img_files = sorted(
        glob.glob(os.path.join(src_img_dir, "*.jpg")) +
        glob.glob(os.path.join(src_img_dir, "*.png"))
    )

    paths = []
    missing_labels = 0
    for img_path in img_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]

        dst_img = os.path.join(dst_dir, basename + ext)
        dst_lbl = os.path.join(dst_dir, basename + ".txt")
        src_lbl = os.path.join(src_lbl_dir, basename + ".txt")

        shutil.copy2(img_path, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            # Create empty label file (image with no objects)
            open(dst_lbl, "w").close()
            missing_labels += 1

        paths.append(dst_img)

    if missing_labels > 0:
        print(f"   ⚠️  {missing_labels} images had no labels (empty .txt created)")

    # Write path listing file
    txt_file = os.path.join(data_dir, f"{split_name}.txt")
    with open(txt_file, "w") as f:
        f.write("\n".join(paths) + "\n")

    print(f"   ↳ {split_name}: {len(paths)} images prepared → {txt_file}")
    return txt_file


train_txt = prepare_split("train")
val_txt   = prepare_split("valid")

# ===========================
# 3. Create .names file
# ===========================
names_file = os.path.join(data_dir, "ppe.names")
with open(names_file, "w") as f:
    f.write("\n".join(CLASS_NAMES) + "\n")
print(f"   ↳ Class names written to {names_file}")

# ===========================
# 4. Generate anchor biases
# ===========================
print("\n⚓ Step 4: Generating anchor biases...")

def generate_anchors(train_txt_path, num_anchors=6, img_w=IMG_SIZE, img_h=IMG_SIZE):
    """
    Custom anchor generation that properly handles empty label files.
    Reads all bounding box w/h from labels, runs k-means to find anchors.
    """
    import numpy as np

    # Collect all bounding box widths and heights (in pixels)
    wh_list = []
    with open(train_txt_path, "r") as f:
        img_paths = [line.strip() for line in f if line.strip()]

    for img_path in img_paths:
        label_path = os.path.splitext(img_path)[0] + ".txt"
        if not os.path.exists(label_path):
            continue
        with open(label_path, "r") as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) >= 5:
                    w = float(parts[3]) * img_w
                    h = float(parts[4]) * img_h
                    if w > 0 and h > 0:
                        wh_list.append([w, h])

    if len(wh_list) < num_anchors:
        print("   ⚠️  Too few valid boxes, using default PPE anchors")
        return "10,13, 16,30, 33,23, 30,61, 62,45, 59,119"

    data = np.array(wh_list, dtype=np.float64)
    print(f"   ↳ Collected {len(data)} bounding boxes for clustering")

    # K-means clustering
    np.random.seed(42)
    indices = np.random.choice(len(data), num_anchors, replace=False)
    centroids = data[indices]

    for iteration in range(300):
        # Compute IoU-based distance (1 - IoU)
        dists = np.zeros((len(data), num_anchors))
        for i in range(num_anchors):
            c_w, c_h = centroids[i]
            inter_w = np.minimum(data[:, 0], c_w)
            inter_h = np.minimum(data[:, 1], c_h)
            inter_area = inter_w * inter_h
            box_area = data[:, 0] * data[:, 1]
            centroid_area = c_w * c_h
            iou = inter_area / (box_area + centroid_area - inter_area + 1e-9)
            dists[:, i] = 1 - iou

        # Assign to nearest centroid
        assignments = np.argmin(dists, axis=1)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(num_anchors):
            mask = assignments == i
            if mask.sum() > 0:
                new_centroids[i] = data[mask].mean(axis=0)
            else:
                new_centroids[i] = centroids[i]

        if np.allclose(centroids, new_centroids, atol=0.01):
            break
        centroids = new_centroids

    # Sort by area
    areas = centroids[:, 0] * centroids[:, 1]
    order = np.argsort(areas)
    centroids = centroids[order]

    # Format as comma-separated string
    anchor_str = ", ".join(f"{c[0]:.2f},{c[1]:.2f}" for c in centroids)
    return anchor_str

anchors_line = generate_anchors(train_txt)
print(f"   ↳ Anchors: {anchors_line}")

# ===========================
# 5. Create .data configuration file
# ===========================
print("\n📝 Step 5: Creating training configuration...")

data_config = os.path.join(REPO_DIR, "data", "ppe.data")
config_content = f"""[name]
model_name=ppe_fastestv2

[train-configure]
epochs={EPOCHS}
steps={STEPS}
batch_size={BATCH_SIZE}
subdivisions=1
learning_rate={LR}

[model-configure]
pre_weights=None
classes={len(CLASS_NAMES)}
width={IMG_SIZE}
height={IMG_SIZE}
anchor_num=3
anchors={anchors_line}

[data-configure]
train={train_txt}
val={val_txt}
names={names_file}
"""

with open(data_config, "w") as f:
    f.write(config_content)
print(f"   ↳ Config written to {data_config}")

# ===========================
# 6. Train the model
# ===========================
print("\n🚀 Step 6: Starting YOLO-FastestV2 training...")
print(f"   ↳ Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | ImgSize: {IMG_SIZE}x{IMG_SIZE}")
print(f"   ↳ Classes: {len(CLASS_NAMES)} | LR: {LR}")


patch_loss_for_new_torch()
patch_train_dual_gpu()
subprocess.run(
    [sys.executable, "train.py", "--data", data_config],
    cwd=REPO_DIR, check=True
)

# Find the best trained weights
# YOLO-FastestV2 saves to: weights/{model_name}-{epoch}-epoch-{AP}ap-model.pth
weight_files = []
for search_dir in [os.path.join(REPO_DIR, "weights"), REPO_DIR, WORK_DIR]:
    weight_files.extend(glob.glob(os.path.join(search_dir, "*.pth")))
# Filter to only model weights (exclude backbone)
weight_files = [w for w in weight_files if "backbone" not in w]
weight_files = sorted(weight_files)
if weight_files:
    best_weights = weight_files[-1]  # latest checkpoint
    print(f"   ↳ Best weights: {best_weights}")
else:
    raise FileNotFoundError("No .pth weight files found after training!")

# ===========================
# 7. Export to ONNX
# ===========================
print("\n📤 Step 7: Exporting to ONNX...")
onnx_output = os.path.join(WORK_DIR, "yolo-fastestv2-ppe.onnx")

# Inline export — handles DataParallel "module." prefix in state_dict keys
sys.path.insert(0, REPO_DIR)
import importlib
detector_module = importlib.import_module("model.detector")

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
export_model = detector_module.Detector(len(CLASS_NAMES), 3, True, True).to(device)

# Load weights, stripping "module." prefix if trained with DataParallel
state_dict = torch.load(best_weights, map_location=device)
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[new_key] = v
export_model.load_state_dict(new_state_dict)
export_model.eval()

test_data = torch.rand(1, 3, IMG_SIZE, IMG_SIZE).to(device)
torch.onnx.export(
    export_model, test_data, onnx_output,
    export_params=True, opset_version=11, do_constant_folding=True
)
print(f"   ↳ ONNX exported: {onnx_output}")

# Simplify ONNX
onnx_simplified = os.path.join(WORK_DIR, "yolo-fastestv2-ppe-sim.onnx")
subprocess.run(
    [sys.executable, "-m", "onnxsim", onnx_output, onnx_simplified],
    check=True
)
print(f"   ↳ ONNX (simplified): {onnx_simplified}")

# ===========================
# 8. Convert ONNX → TFLite
# ===========================
print("\n📱 Step 8: Converting ONNX → TFLite (int8 quantized)...")

tflite_dir = os.path.join(WORK_DIR, "tflite_output")

# Use onnx2tf for conversion
subprocess.run(
    [sys.executable, "-m", "onnx2tf",
     "-i", onnx_simplified,
     "-o", tflite_dir,
     "-oiqt"],   # output int8 quantized tflite
    check=True
)

# Find the generated .tflite file
tflite_files = glob.glob(os.path.join(tflite_dir, "**", "*.tflite"), recursive=True)
if tflite_files:
    final_tflite = tflite_files[0]
    dst_tflite = os.path.join(WORK_DIR, "yolo-fastestv2-ppe-int8.tflite")
    shutil.copy2(final_tflite, dst_tflite)
    print(f"   ↳ TFLite model: {dst_tflite}")
    print(f"   ↳ Size: {os.path.getsize(dst_tflite) / 1024:.1f} KB")
else:
    print("⚠️  No .tflite file found. Check onnx2tf output above.")

# ===========================
# 9. Summary
# ===========================
print("\n" + "=" * 50)
print("✅ YOLO-FastestV2 PPE Training & Export Complete!")
print("=" * 50)
print(f"  Model:     YOLO-FastestV2 (~250K params)")
print(f"  Classes:   {len(CLASS_NAMES)} ({', '.join(CLASS_NAMES)})")
print(f"  Input:     {IMG_SIZE}x{IMG_SIZE}")
print(f"  Weights:   {best_weights}")
print(f"  ONNX:      {onnx_simplified}")
if tflite_files:
    print(f"  TFLite:    {dst_tflite}")
print("=" * 50)
