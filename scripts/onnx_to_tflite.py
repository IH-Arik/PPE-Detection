import onnx
import tensorflow as tf
import numpy as np
import os

def convert_onnx_to_tflite():
    """
    Convert the existing ONNX file to TFLite using a different approach
    """
    onnx_file = 'yolov8n.onnx'
    tflite_file = 'yolov8n.tflite'
    
    if not os.path.exists(onnx_file):
        print(f"ONNX file {onnx_file} not found!")
        return False
    
    print(f"Converting {onnx_file} to TFLite...")
    
    try:
        # Try using onnx-tf
        print("Using onnx-tf for conversion...")
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_file)
        
        # Convert to TensorFlow
        from onnx_tf.backend import prepare
        tf_rep = prepare(onnx_model)
        
        # Export to SavedModel
        saved_model_dir = 'temp_tf_model'
        tf_rep.export_graph(saved_model_dir)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(tflite_file, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Successfully converted to {tflite_file}")
        
        # Clean up
        import shutil
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)
        
        return True
        
    except ImportError:
        print("onnx-tf not available. Installing...")
        os.system("pip install onnx-tf")
        print("Please run the script again after installation.")
        return False
            
    except Exception as e:
        print(f"Error in conversion: {e}")
        return False

def main():
    print("Starting ONNX to TFLite conversion...")
    print("=" * 50)
    
    success = convert_onnx_to_tflite()
    
    if success:
        print("=" * 50)
        print("Conversion completed successfully!")
        
        # Check file sizes
        if os.path.exists('yolov8n.tflite'):
            tflite_size = os.path.getsize('yolov8n.tflite') / (1024 * 1024)
            pt_size = os.path.getsize('yolov8n.pt') / (1024 * 1024)
            
            print(f"Original model size: {pt_size:.2f} MB")
            print(f"TFLite model size: {tflite_size:.2f} MB")
            print(f"Size reduction: {((pt_size - tflite_size) / pt_size * 100):.1f}%")
    else:
        print("Conversion failed.")

if __name__ == "__main__":
    main()
