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