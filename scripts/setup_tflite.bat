@echo off
echo Creating conda environment for TFLite conversion...
echo.

REM Create conda environment
conda create -n tflite_env python=3.9 -y

REM Activate environment and install packages
call conda activate tflite_env
pip install tensorflow==2.12.0 ultralytics onnx onnx2tf

REM Navigate to project directory and convert
cd /d "c:\Users\IH ARIK\Desktop\ARIK\PPE Detection"
python newconverter.py

echo.
echo TFLite conversion should be completed!
pause