"""
Webcam diagnostic script to troubleshoot camera issues
"""

import cv2
import sys

def test_webcam():
    """Test webcam functionality"""
    print("🔍 Testing Webcam Access...")
    print("=" * 50)
    
    # Test different camera indices
    for camera_index in range(3):  # Test cameras 0, 1, 2
        print(f"\n📹 Testing camera {camera_index}...")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Camera {camera_index}: Unable to open")
            continue
        
        # Try to read a frame
        ret, frame = cap.read()
        
        if not ret:
            print(f"❌ Camera {camera_index}: Unable to read frame")
            cap.release()
            continue
        
        if frame is None:
            print(f"❌ Camera {camera_index}: Frame is None")
            cap.release()
            continue
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera {camera_index}: Working!")
        print(f"   - Resolution: {width}x{height}")
        print(f"   - FPS: {fps}")
        print(f"   - Frame shape: {frame.shape}")
        
        # Save a test frame
        test_file = f"webcam_test_{camera_index}.jpg"
        cv2.imwrite(test_file, frame)
        print(f"   - Test image saved: {test_file}")
        
        cap.release()
        
        # If we found a working camera, we can stop testing
        if camera_index == 0:
            print(f"\n🎉 Primary camera (0) is working!")
            return camera_index
    
    print("\n❌ No working cameras found!")
    return None

def check_opencv_version():
    """Check OpenCV installation"""
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"OpenCV build info: {cv2.getBuildInformation()[:200]}...")

def main():
    """Main diagnostic function"""
    print("📹 Webcam Diagnostic Tool")
    print("=" * 50)
    
    check_opencv_version()
    print()
    
    working_camera = test_webcam()
    
    if working_camera is not None:
        print(f"\n✅ Solution: Use camera index {working_camera} in your app")
        print("Update this line in app.py:")
        print(f"cap = cv2.VideoCapture({working_camera})")
    else:
        print("\n🔧 Troubleshooting steps:")
        print("1. Check if camera is connected")
        print("2. Close other apps using camera (Zoom, Teams, etc.)")
        print("3. Restart your computer")
        print("4. Check camera permissions in system settings")
        print("5. Try external USB camera")

if __name__ == "__main__":
    main()
