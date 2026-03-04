import os

def create_docker_files():
    """Create Docker files for TFLite conversion"""
    
    dockerfile_content = '''FROM tensorflow/tensorflow:2.12.0

WORKDIR /workspace

# Install required packages
RUN pip install ultralytics onnx onnx2tf

# Copy your ONNX file
COPY yolov8n.onnx /workspace/

# Create conversion script
COPY convert_script.py /workspace/

# Run conversion
CMD ["python", "convert_script.py"]'''
    
    convert_script = '''from ultralytics import YOLO
import os

# Convert ONNX to TFLite using compatible TensorFlow
print("Converting ONNX to TFLite in Docker...")

try:
    # Use YOLO to convert
    model = YOLO('yolov8n.onnx')
    model.export(format='tflite', imgsz=640)
    print("TFLite conversion completed in Docker!")
except Exception as e:
    print(f"Conversion failed: {e}")'''
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    with open('convert_script.py', 'w') as f:
        f.write(convert_script)
    
    print("Docker files created successfully!")

def create_conda_script():
    """Create conda setup script"""
    
    conda_script = '''@echo off
echo Creating conda environment for TFLite conversion...
echo.

REM Create conda environment
conda create -n tflite_env python=3.9 -y

REM Activate environment and install packages
call conda activate tflite_env
pip install tensorflow==2.12.0 ultralytics onnx onnx2tf

REM Navigate to project directory and convert
cd /d "c:\\Users\\IH ARIK\\Desktop\\ARIK\\PPE Detection"
python newconverter.py

echo.
echo TFLite conversion should be completed!
pause'''
    
    with open('setup_tflite.bat', 'w') as f:
        f.write(conda_script)
    
    print("Conda setup script created: setup_tflite.bat")

def main():
    print("FINAL TFLITE CONVERSION SOLUTIONS")
    print("=" * 50)
    
    # Check current files
    print("Current files:")
    files = ['yolov8n.onnx', 'yolov8n_inference.py']
    for file in files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"  {file}: {size_mb:.2f} MB")
    
    print("\nSOLUTION OPTIONS:")
    print("=" * 30)
    
    print("\n1. ONNX RUNTIME (Already Working)")
    print("   - Use: python yolov8n_inference.py")
    print("   - Pros: Works now, no setup needed")
    print("   - Cons: Not TFLite format")
    
    print("\n2. DOCKER SOLUTION")
    create_docker_files()
    print("   Commands:")
    print("   docker build -t yolov8-tflite .")
    print("   docker run -v %cd%:/workspace yolov8-tflite")
    print("   - Pros: Isolated, reliable")
    print("   - Cons: Requires Docker")
    
    print("\n3. CONDA ENVIRONMENT")
    create_conda_script()
    print("   Run: setup_tflite.bat")
    print("   - Pros: Native Python environment")
    print("   - Cons: Requires conda")
    
    print("\n4. MANUAL TENSORFLOW SETUP")
    print("   Commands:")
    print("   conda create -n tflite python=3.9")
    print("   conda activate tflite")
    print("   pip install tensorflow==2.12.0 ultralytics")
    print("   python newconverter.py")
    
    print("\n" + "=" * 50)
    print("RECOMMENDATION:")
    print("=" * 50)
    print("1. For immediate use: yolov8n_inference.py (ONNX Runtime)")
    print("2. For TFLite: Use Docker or conda environment")
    print("3. Docker is most reliable option")
    
    print("\nFILES CREATED:")
    files = ['Dockerfile', 'convert_script.py', 'setup_tflite.bat']
    for file in files:
        if os.path.exists(file):
            print(f"  {file}")

if __name__ == "__main__":
    main()
