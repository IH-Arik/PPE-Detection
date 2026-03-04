import subprocess
import os
import shutil

def step1_install_libraries():
    """Step 1: Install required libraries with compatible versions"""
    print("Step 1: Installing required libraries...")
    
    # Install onnx2tf (more reliable than onnx-tf)
    try:
        subprocess.run(['pip', 'install', 'onnx2tf'], check=True, capture_output=True)
        print("✅ onnx2tf installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install onnx2tf: {e}")
        return False

def step2_onnx_to_tensorflow():
    """Step 2: Convert ONNX to TensorFlow using onnx2tf"""
    print("\nStep 2: Converting ONNX to TensorFlow...")
    
    onnx_file = 'yolov8n.onnx'
    tf_model_dir = 'tf_model'
    
    if not os.path.exists(onnx_file):
        print(f"❌ ONNX file {onnx_file} not found!")
        return False
    
    # Clean up previous attempts
    if os.path.exists(tf_model_dir):
        shutil.rmtree(tf_model_dir)
    
    try:
        # Use onnx2tf command
        cmd = [
            'python', '-c', 
            f'''
import onnx2tf
import os

print("Converting ONNX to TensorFlow...")
onnx2tf.convert(
    input_onnx_file_path="{onnx_file}",
    output_folder_path="{tf_model_dir}",
    copy_onnx_input_output_names_to_tflite=True,
    output_signature_names=["outputs"],
    non_verbose=True
)

print("✅ ONNX to TensorFlow conversion completed!")
'''
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ TensorFlow model created successfully!")
            print(result.stdout)
            return True
        else:
            print(f"❌ Conversion failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Conversion timed out")
        return False
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False

def step3_tensorflow_to_tflite():
    """Step 3: Convert TensorFlow to TFLite"""
    print("\nStep 3: Converting TensorFlow to TFLite...")
    
    tf_model_dir = 'tf_model'
    tflite_file = 'yolov8n.tflite'
    
    if not os.path.exists(tf_model_dir):
        print(f"❌ TensorFlow model directory {tf_model_dir} not found!")
        return False
    
    try:
        cmd = [
            'python', '-c',
            f'''
import tensorflow as tf
import os

print("Converting TensorFlow to TFLite...")

# Find the SavedModel directory
saved_model_path = None
for root, dirs, files in os.walk("{tf_model_dir}"):
    if "saved_model.pb" in files:
        saved_model_path = root
        break

if not saved_model_path:
    print("❌ SavedModel not found!")
    exit(1)

print(f"Found SavedModel at: {{saved_model_path}}")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

try:
    tflite_model = converter.convert()
    print("✅ TFLite conversion successful!")
    
    with open("{tflite_file}", "wb") as f:
        f.write(tflite_model)
    
    print(f"✅ TFLite model saved as: {tflite_file}")
    
except Exception as e:
    print(f"❌ TFLite conversion error: {{e}}")
    print("Trying without optimizations...")
    
    # Try without optimizations
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_model = converter.convert()
    
    with open("{tflite_file}", "wb") as f:
        f.write(tflite_model)
    
    print(f"✅ TFLite model saved (no optimizations): {tflite_file}")
'''
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ TFLite conversion completed!")
            print(result.stdout)
            return True
        else:
            print(f"❌ TFLite conversion failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ TFLite conversion timed out")
        return False
    except Exception as e:
        print(f"❌ Error during TFLite conversion: {e}")
        return False

def check_onnx_model_info():
    """Check ONNX model inputs and structure"""
    print("\nChecking ONNX model structure...")
    
    try:
        cmd = [
            'python', '-c',
            '''
import onnx

onnx_model = onnx.load("yolov8n.onnx")

print("ONNX Model Information:")
print("=" * 40)
print(f"Inputs: {len(onnx_model.graph.input)}")
for i, input_info in enumerate(onnx_model.graph.input):
    print(f"  Input {i}: {input_info.name}")
    print(f"    Shape: {[d.dim_value for d in input_info.type.tensor_type.shape.dim]}")
    print(f"    Type: {input_info.type.tensor_type.elem_type}")

print(f"\nOutputs: {len(onnx_model.graph.output)}")
for i, output_info in enumerate(onnx_model.graph.output):
    print(f"  Output {i}: {output_info.name}")
    print(f"    Shape: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in output_info.type.tensor_type.shape.dim]}")
    print(f"    Type: {output_info.type.tensor_type.elem_type}")
'''
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        
    except Exception as e:
        print(f"❌ Error checking ONNX model: {e}")

def main():
    print("YOLOv8n ONNX to TFLite Conversion")
    print("=" * 50)
    
    # Check ONNX model first
    check_onnx_model_info()
    
    # Step 1: Install libraries
    if not step1_install_libraries():
        print("❌ Failed to install required libraries")
        return
    
    # Step 2: ONNX to TensorFlow
    if not step2_onnx_to_tensorflow():
        print("❌ Failed to convert ONNX to TensorFlow")
        return
    
    # Step 3: TensorFlow to TFLite
    if not step3_tensorflow_to_tflite():
        print("❌ Failed to convert TensorFlow to TFLite")
        return
    
    # Success!
    print("\n" + "=" * 50)
    print("🎉 CONVERSION COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 50)
    
    # Show file sizes
    files_to_check = ['yolov8n.pt', 'yolov8n.onnx', 'yolov8n.tflite']
    print("\nFile Sizes:")
    for file in files_to_check:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"  {file}: {size_mb:.2f} MB")
    
    if os.path.exists('yolov8n.tflite'):
        pt_size = os.path.getsize('yolov8n.pt') / (1024 * 1024)
        tflite_size = os.path.getsize('yolov8n.tflite') / (1024 * 1024)
        reduction = ((pt_size - tflite_size) / pt_size * 100)
        print(f"\nSize reduction: {reduction:.1f}%")
        print(f"✅ Your TFLite model is ready for mobile deployment!")

if __name__ == "__main__":
    main()
