import subprocess
import os

def create_conda_environment():
    """
    Create a conda environment with compatible TensorFlow version
    """
    print("Creating conda environment for TFLite conversion...")
    
    try:
        # Check if conda is available
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Conda not found. Please install Anaconda or Miniconda first.")
            return False
        
        print("✅ Conda found")
        
        # Create new environment
        commands = [
            ['conda', 'create', '-n', 'tflite_env', 'python=3.9', '-y'],
            ['conda', 'activate', 'tflite_env', '&&', 'pip', 'install', 'tensorflow==2.12.0'],
            ['conda', 'activate', 'tflite_env', '&&', 'pip', 'install', 'ultralytics', 'onnx', 'onnx2tf']
        ]
        
        for i, cmd in enumerate(commands, 1):
            print(f"\nStep {i}: {' '.join(cmd)}")
            
            if i == 1:
                # Create environment
                result = subprocess.run(cmd, capture_output=True, text=True)
            else:
                # For pip install in conda env
                cmd_str = ' '.join(cmd)
                result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Step {i} completed")
            else:
                print(f"❌ Step {i} failed: {result.stderr}")
        
        print("\n" + "="*50)
        print("CONDA ENVIRONMENT SETUP INSTRUCTIONS:")
        print("="*50)
        print("1. Open new terminal/command prompt")
        print("2. Run: conda activate tflite_env")
        print("3. Navigate to your project directory")
        print("4. Run: python newconverter.py")
        print("\nOr use this command:")
        print("conda activate tflite_env && cd \"c:\\Users\\IH ARIK\\Desktop\\ARIK\\PPE Detection\" && python newconverter.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Conda setup failed: {e}")
        return False

def create_docker_solution():
    """
    Create Docker solution for TFLite conversion
    """
    print("Creating Docker solution...")
    
    dockerfile_content = '''
FROM tensorflow/tensorflow:2.12.0

WORKDIR /workspace

# Install required packages
RUN pip install ultralytics onnx onnx2tf

# Copy your ONNX file
COPY yolov8n.onnx /workspace/

# Create conversion script
COPY convert_script.py /workspace/

# Run conversion
CMD ["python", "convert_script.py"]
'''
    
    convert_script = '''
from ultralytics import YOLO
import os

# Convert ONNX to TFLite using compatible TensorFlow
print("Converting ONNX to TFLite in Docker...")

try:
    # Use YOLO to convert
    model = YOLO('yolov8n.onnx')
    model.export(format='tflite', imgsz=640)
    print("✅ TFLite conversion completed in Docker!")
except Exception as e:
    print(f"❌ Conversion failed: {e}")
'''
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    with open('convert_script.py', 'w') as f:
        f.write(convert_script)
    
    print("✅ Docker files created!")
    
    print("\nDOCKER INSTRUCTIONS:")
    print("="*30)
    print("1. Build Docker image:")
    print("   docker build -t yolov8-tflite .")
    print("\n2. Run conversion:")
    print("   docker run -v $(pwd):/workspace yolov8-tflite")
    print("\n3. TFLite file will be created in your directory")

def main():
    print("Local TFLite Environment Setup")
    print("=" * 50)
    
    print("Choose your preferred solution:")
    print("1. Conda Environment (Recommended)")
    print("2. Docker Solution")
    print("3. Use existing ONNX Runtime (Already working)")
    
    # Create Docker solution
    create_docker_solution()
    
    # Try conda setup
    print("\n" + "="*50)
    print("SETTING UP CONDA ENVIRONMENT...")
    print("="*50)
    create_conda_environment()
    
    print("\n" + "="*50)
    print("CURRENT STATUS:")
    print("="*50)
    
    files = ['yolov8n.onnx', 'yolov8n_inference.py', 'Dockerfile', 'convert_script.py']
    for file in files:
        if os.path.exists(file):
            size_kb = os.path.getsize(file) / 1024
            print(f"✅ {file}: {size_kb:.1f} KB")
    
    print("\n💡 BEST OPTIONS:")
    print("1. Use conda environment (most reliable)")
    print("2. Use Docker (if conda not available)")
    print("3. Use yolov8n_inference.py (ONNX Runtime - already working)")

if __name__ == "__main__":
    main()
