import onnx
import tensorflow as tf
import numpy as np
import os
import subprocess

def convert_onnx_to_tflite_direct():
    """
    Direct conversion from ONNX to TFLite using available tools
    """
    onnx_file = 'yolov8n.onnx'
    tflite_file = 'yolov8n.tflite'
    
    print(f"Converting {onnx_file} to TFLite...")
    
    # Method 1: Try using TensorFlow's experimental ONNX converter
    try:
        print("Method 1: Using TensorFlow's built-in converter...")
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_file)
        
        # Get model info
        input_name = onnx_model.graph.input[0].name
        input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        
        print(f"Input: {input_name}, Shape: {input_shape}")
        
        # Try to convert using tf.lite.experimental
        if hasattr(tf.lite, 'experimental') and hasattr(tf.lite.experimental, 'convert_from_onnx_model'):
            converter = tf.lite.experimental.convert_from_onnx_model(onnx_file)
            with open(tflite_file, 'wb') as f:
                f.write(converter)
            print("✅ Method 1 successful!")
            return True
        else:
            print("❌ Experimental ONNX converter not available")
            
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Use subprocess with onnx2tf (if available in PATH)
    try:
        print("Method 2: Using onnx2tf command...")
        
        # Try different ways to call onnx2tf
        commands = [
            ['onnx2tf', '-i', onnx_file, '-o', 'temp_tf_model'],
            ['python', '-m', 'onnx2tf', '-i', onnx_file, '-o', 'temp_tf_model'],
            ['python', '-c', f'import onnx2tf; onnx2tf.convert(input_onnx_file_path="{onnx_file}", output_folder_path="temp_tf_model")']
        ]
        
        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    print("✅ ONNX to TensorFlow conversion successful!")
                    
                    # Now convert TensorFlow to TFLite
                    if os.path.exists('temp_tf_model'):
                        converter = tf.lite.TFLiteConverter.from_saved_model('temp_tf_model')
                        converter.optimizations = [tf.lite.Optimize.DEFAULT]
                        
                        tflite_model = converter.convert()
                        with open(tflite_file, 'wb') as f:
                            f.write(tflite_model)
                        
                        print("✅ TensorFlow to TFLite conversion successful!")
                        
                        # Clean up
                        import shutil
                        if os.path.exists('temp_tf_model'):
                            shutil.rmtree('temp_tf_model')
                        
                        return True
                    
            except Exception as e:
                print(f"Command {cmd} failed: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    # Method 3: Manual conversion using onnxruntime
    try:
        print("Method 3: Manual conversion using onnxruntime...")
        
        import onnxruntime as ort
        
        # Create inference session
        session = ort.InferenceSession(onnx_file)
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"Input: {input_name}, Shape: {input_shape}")
        print(f"Outputs: {output_names}")
        
        # Create a representative dataset for quantization
        def representative_dataset():
            for _ in range(100):
                # Create random input data
                data = np.random.rand(*input_shape).astype(np.float32)
                yield [data]
        
        # This method would require more complex implementation
        print("❌ Method 3 requires more complex implementation")
        
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    return False

def create_simple_tflite_converter():
    """
    Create a simple TFLite converter that works
    """
    print("Creating simple TFLite converter...")
    
    try:
        # Create a basic TensorFlow model that can be converted
        # This is a workaround to demonstrate TFLite conversion works
        
        input_shape = (640, 640, 3)
        inputs = tf.keras.Input(shape=input_shape, name='input')
        
        # Simple CNN structure
        x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(10, name='output')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save as demonstration
        with open('demo_model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print("✅ Demo TFLite model created successfully!")
        print("This proves TFLite conversion works with your TensorFlow setup.")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo conversion failed: {e}")
        return False

def main():
    print("ONNX to TFLite Conversion - Final Attempt")
    print("=" * 50)
    
    # Check if ONNX file exists
    if not os.path.exists('yolov8n.onnx'):
        print("❌ yolov8n.onnx not found!")
        return
    
    # Show file info
    size_mb = os.path.getsize('yolov8n.onnx') / (1024 * 1024)
    print(f"📁 ONNX file: yolov8n.onnx ({size_mb:.2f} MB)")
    
    # Try direct conversion
    if convert_onnx_to_tflite_direct():
        print("\n🎉 SUCCESS! TFLite conversion completed!")
        
        # Show final file info
        if os.path.exists('yolov8n.tflite'):
            tflite_size = os.path.getsize('yolov8n.tflite') / (1024 * 1024)
            print(f"📁 TFLite file: yolov8n.tflite ({tflite_size:.2f} MB)")
            
            # Compare sizes
            pt_size = os.path.getsize('yolov8n.pt') / (1024 * 1024) if os.path.exists('yolov8n.pt') else 0
            if pt_size > 0:
                reduction = ((pt_size - tflite_size) / pt_size * 100)
                print(f"📊 Size reduction: {reduction:.1f}%")
    else:
        print("\n❌ Direct conversion failed. Trying demo...")
        create_simple_tflite_converter()
        
        print("\n💡 ALTERNATIVE SOLUTIONS:")
        print("1. Use online converter: https://convertmodel.com")
        print("2. Use Google Colab with TF 2.12.x")
        print("3. Use Docker: tensorflow/tensorflow:2.12.0")
        print("4. Your ONNX file is ready for any of these methods")

if __name__ == "__main__":
    main()
