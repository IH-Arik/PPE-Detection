
import cv2
import time

def test_camera():
    print("Testing camera access...")
    try:
        # Try index 0 with DSHOW
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("ERROR: Could not open camera (index 0) with DSHOW.")
            # Try index 1
            print("Trying index 1 with DSHOW...")
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("ERROR: Could not open camera (index 1) with DSHOW.")
                return
            else:
                print("Success: Opened camera (index 1).")
        else:
            print("Success: Opened camera (index 0).")

        ret, frame = cap.read()
        if ret:
            print(f"Success: Read frame of shape {frame.shape}")
            cv2.imwrite("camera_test.jpg", frame)
            print("Saved camera_test.jpg")
        else:
            print("ERROR: Failed to read frame.")
        
        cap.release()
    except Exception as e:
        print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    test_camera()
