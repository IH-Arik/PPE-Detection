"""
Configuration file for Hazard Detection System
"""

class Config:
    """Application configuration settings"""
    
    # Model URLs
    CONSTRUCTION_HAZARD_MODEL = "https://huggingface.co/yihong1120/Construction-Hazard-Detection/resolve/main/models/yolo11/pt/yolo11n.pt"
    PPE_DETECTION_MODEL_REPO = "melihuzunoglu/ppe-detection"
    PPE_DETECTION_MODEL_FILE = "best.pt"
    
    # Default settings
    DEFAULT_CONFIDENCE_THRESHOLD = 0.25
    DEFAULT_IMAGE_SIZE = 640
    
    # Supported file formats
    SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'webp']
    SUPPORTED_VIDEO_FORMATS = ['mp4', 'avi', 'mov', 'mkv']
    
    # Detection classes to filter
    ALLOWED_CLASSES = {0, 1, 2, 3, 4, 5, 7}  # Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Vest
    
    # Color scheme for bounding boxes (BGR format for OpenCV)
    COLORS = {
        'green': (0, 255, 0),      # Proper safety equipment
        'red': (0, 0, 255),        # Safety violations
        'light_blue': (255, 255, 0), # People (Yellow in BGR = Light blue in RGB)
        'white': (255, 255, 255)   # Text color
    }
    
    # Class color mapping
    CLASS_COLORS = {
        0: 'green',    # Hardhat
        1: 'green',    # Mask
        2: 'red',      # NO-Hardhat
        3: 'red',      # NO-Mask
        4: 'red',      # NO-Safety Vest
        5: 'light_blue', # Person
        7: 'green'     # Safety Vest
    }
    
    # UI Settings
    PAGE_TITLE = "🚧 Hazard Detection System"
    PAGE_ICON = "🚧"
    PAGE_LAYOUT = "wide"
    
    # Processing settings
    MAX_WEBCAM_FRAMES = 100
    WEBCAM_FRAME_DELAY = 0.1
    MAX_RECENT_DETECTIONS = 100
    
    # File size limits (in MB)
    MAX_IMAGE_SIZE = 50
    MAX_VIDEO_SIZE = 500
