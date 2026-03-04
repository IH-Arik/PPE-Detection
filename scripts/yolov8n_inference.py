
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('yolov8n.onnx')

# Get input info
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# Create inference function
def predict_yolov8(image_array):
    """
    Run inference using ONNX Runtime
    image_array: numpy array with shape [1, 3, 640, 640]
    """
    # Run inference
    outputs = session.run(None, {input_name: image_array})
    return outputs[0]

# Example usage
if __name__ == "__main__":
    # Create dummy input
    dummy_input = np.random.rand(*input_shape).astype(np.float32)
    
    # Run inference
    result = predict_yolov8(dummy_input)
    print(f"Output shape: {result.shape}")
    print("ONNX Runtime inference working!")
