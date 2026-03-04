import subprocess
import os

def convert_with_onnx2tf():
    """
    Use the onnx2tf command line tool that's already installed
    """
    onnx_file = 'yolov8n.onnx'
    tflite_file = 'yolov8n.tflite'
    
    if not os.path.exists(onnx_file):
        print(f"ONNX file {onnx_file} not found!")
        return False
    
    print(f"Converting {onnx_file} to TFLite using onnx2tf...")
    
    try:
        # Use onnx2tf as Python module
        cmd = [
            'python', '-m', 'onnx2tf',
            '-i', onnx_file,
            '-o', 'onnx2tf_output',
            '-oh5',  # Output H5 format as well
            '-cotof',  # Copy ONNX input/output names to TFLite
            '-osd'  # Output signature definitions
        ]
        
        print("Running command:", ' '.join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("onnx2tf conversion successful!")
            
            # Check for the generated TFLite file
            tflite_path = os.path.join('onnx2tf_output', 'model_float32.tflite')
            if os.path.exists(tflite_path):
                # Copy to our desired location
                import shutil
                shutil.copy2(tflite_path, tflite_file)
                print(f"TFLite model copied to {tflite_file}")
                return True
            else:
                print("TFLite file not found in expected location")
                print("Available files in output directory:")
                for root, dirs, files in os.walk('onnx2tf_output'):
                    for file in files:
                        if file.endswith('.tflite'):
                            print(f"  Found: {os.path.join(root, file)}")
                            # Copy the first TFLite file we find
                            shutil.copy2(os.path.join(root, file), tflite_file)
                            return True
                return False
        else:
            print(f"onnx2tf failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("Conversion timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

def main():
    print("Starting ONNX to TFLite conversion with onnx2tf...")
    print("=" * 50)
    
    success = convert_with_onnx2tf()
    
    if success:
        print("=" * 50)
        print("Conversion completed successfully!")
        
        # Show file sizes
        if os.path.exists('yolov8n.tflite'):
            tflite_size = os.path.getsize('yolov8n.tflite') / (1024 * 1024)
            pt_size = os.path.getsize('yolov8n.pt') / (1024 * 1024)
            onnx_size = os.path.getsize('yolov8n.onnx') / (1024 * 1024)
            
            print(f"Original PT model: {pt_size:.2f} MB")
            print(f"ONNX model: {onnx_size:.2f} MB")
            print(f"TFLite model: {tflite_size:.2f} MB")
            print(f"Size reduction from PT: {((pt_size - tflite_size) / pt_size * 100):.1f}%")
    else:
        print("Conversion failed.")

if __name__ == "__main__":
    main()
