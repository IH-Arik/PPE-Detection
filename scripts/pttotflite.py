import torch
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import os

def convert_yolo_to_tflite(pt_path, tflite_path=None):
    """
    Convert YOLOv8 PyTorch model to TensorFlow Lite format.
    
    Args:
        pt_path (str): Path to the .pt model file
        tflite_path (str): Output path for .tflite file (optional)
    """
    print(f"Loading YOLOv8 model from: {pt_path}")
    
    # Load the YOLOv8 model
    model = YOLO(pt_path)
    
    # Export to ONNX format first
    onnx_path = pt_path.replace('.pt', '.onnx')
    print(f"Exporting to ONNX: {onnx_path}")
    model.export(format='onnx', imgsz=640, optimize=True)
    
    # Convert ONNX to TensorFlow SavedModel
    try:
        import onnx
        import onnx2tf
        
        print("Converting ONNX to TensorFlow SavedModel...")
        saved_model_path = pt_path.replace('.pt', '_saved_model')
        
        # Convert ONNX to TensorFlow
        onnx_model = onnx.load(onnx_path)
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=saved_model_path,
            copy_onnx_input_output_names_to_tflite=True,
            output_signature_names=['outputs'],
            output_tflite_file=tflite_path if tflite_path else pt_path.replace('.pt', '.tflite')
        )
        
        print(f"Successfully converted to TFLite: {tflite_path if tflite_path else pt_path.replace('.pt', '.tflite')}")
        
        # Clean up intermediate files
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        if os.path.exists(saved_model_path):
            import shutil
            shutil.rmtree(saved_model_path)
            
    except ImportError:
        print("Required packages not found. Installing...")
        os.system("pip install onnx onnx2tf")
        print("Please run the script again after installation.")
        return False
    
    return True

def convert_with_tflite_converter(pt_path, tflite_path=None):
    """
    Alternative method using TensorFlow's TFLiteConverter directly.
    This method requires the model to be in TensorFlow format first.
    """
    try:
        # Load YOLO model
        model = YOLO(pt_path)
        
        # Export to TensorFlow format
        tf_model_path = pt_path.replace('.pt', '_tf_model')
        model.export(format='tf', imgsz=640)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        output_path = tflite_path if tflite_path else pt_path.replace('.pt', '.tflite')
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"Successfully converted to TFLite: {output_path}")
        
        # Clean up
        import shutil
        if os.path.exists(tf_model_path):
            shutil.rmtree(tf_model_path)
            
        return True
        
    except Exception as e:
        print(f"Error in conversion: {e}")
        return False

def main():
    # Path to your YOLOv8n.pt model
    pt_model_path = "yolov8n.pt"
    
    # Output path for TFLite model
    tflite_model_path = "yolov8n.tflite"
    
    if not os.path.exists(pt_model_path):
        print(f"Error: Model file {pt_model_path} not found!")
        return
    
    print("Starting YOLOv8 to TFLite conversion...")
    print("=" * 50)
    
    # Try the alternative conversion method first (more reliable)
    success = convert_with_tflite_converter(pt_model_path, tflite_model_path)
    
    if not success:
        print("Alternative method failed. Trying primary method...")
        success = convert_yolo_to_tflite(pt_model_path, tflite_model_path)
    
    if success:
        print("=" * 50)
        print("Conversion completed successfully!")
        print(f"Input model: {pt_model_path}")
        print(f"Output model: {tflite_model_path}")
        
        # Check file sizes
        pt_size = os.path.getsize(pt_model_path) / (1024 * 1024)  # MB
        tflite_size = os.path.getsize(tflite_model_path) / (1024 * 1024)  # MB
        
        print(f"Original model size: {pt_size:.2f} MB")
        print(f"TFLite model size: {tflite_size:.2f} MB")
        print(f"Size reduction: {((pt_size - tflite_size) / pt_size * 100):.1f}%")
    else:
        print("Conversion failed. Please check the error messages above.")

if __name__ == "__main__":
    main()