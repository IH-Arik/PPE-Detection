"""
Utility functions for hazard detection system
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO
from src.config.config import Config


def draw_custom_bboxes(image: np.ndarray, boxes: List, model: YOLO) -> np.ndarray:
    """
    Draw bounding boxes with custom colors based on class types
    
    Args:
        image: Input image (BGR format)
        boxes: List of detection boxes
        model: YOLO model for class names
        
    Returns:
        Annotated image with bounding boxes
    """
    annotated_image = image.copy()
    
    for box in boxes:
        # Get box coordinates and properties
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls)
        class_name = model.names[class_id]
        confidence = float(box.conf)
        
        # Determine color based on class
        color_name = Config.CLASS_COLORS.get(class_id, 'green')
        color = Config.COLORS[color_name]
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{class_name}: {confidence:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated_image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        
        # Draw label text
        cv2.putText(annotated_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.COLORS['white'], 1)
    
    return annotated_image


def filter_detections(results: List, model: YOLO) -> Tuple[List, List[Dict[str, Any]]]:
    """
    Filter detections to only include allowed classes
    
    Args:
        results: YOLO detection results
        model: YOLO model for class names
        
    Returns:
        Tuple of (filtered_boxes, detection_info)
    """
    filtered_boxes = []
    detection_info = []
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id in Config.ALLOWED_CLASSES:
                filtered_boxes.append(box)
                detection_info.append({
                    'class': model.names[class_id],
                    'confidence': float(box.conf),
                    'class_id': class_id
                })
    
    return filtered_boxes, detection_info


def calculate_detection_summary(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics for detections
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Dictionary with summary statistics
    """
    if not detections:
        return {
            'total_detections': 0,
            'class_counts': {},
            'class_confidences': {},
            'average_confidence': 0.0
        }
    
    class_counts = {}
    class_confidences = {}
    
    for det in detections:
        class_name = det['class']
        confidence = det['confidence']
        
        if class_name not in class_counts:
            class_counts[class_name] = 0
            class_confidences[class_name] = []
        
        class_counts[class_name] += 1
        class_confidences[class_name].append(confidence)
    
    # Calculate average confidences
    avg_confidences = {}
    for class_name, confidences in class_confidences.items():
        avg_confidences[class_name] = np.mean(confidences)
    
    total_detections = len(detections)
    overall_avg_confidence = np.mean([det['confidence'] for det in detections])
    
    return {
        'total_detections': total_detections,
        'class_counts': class_counts,
        'class_confidences': avg_confidences,
        'average_confidence': overall_avg_confidence
    }


def validate_file_format(filename: str, file_type: str = 'image') -> bool:
    """
    Validate if file format is supported
    
    Args:
        filename: Name of the file to validate
        file_type: Type of file ('image' or 'video')
        
    Returns:
        True if format is supported, False otherwise
    """
    if file_type == 'image':
        supported_formats = Config.SUPPORTED_IMAGE_FORMATS
    elif file_type == 'video':
        supported_formats = Config.SUPPORTED_VIDEO_FORMATS
    else:
        return False
    
    file_extension = filename.lower().split('.')[-1]
    return file_extension in supported_formats


def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    if max(height, width) <= max_size:
        return image
    
    if height > width:
        new_height = max_size
        new_width = int(width * (max_size / height))
    else:
        new_width = max_size
        new_height = int(height * (max_size / width))
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def format_detection_summary(summary: Dict[str, Any]) -> str:
    """
    Format detection summary for display
    
    Args:
        summary: Detection summary dictionary
        
    Returns:
        Formatted string
    """
    if summary['total_detections'] == 0:
        return "No hazards detected."
    
    lines = [f"**Total Detections**: {summary['total_detections']}"]
    lines.append(f"**Average Confidence**: {summary['average_confidence']:.2f}")
    lines.append("")
    lines.append("**Detections by Class**:")
    
    for class_name, count in summary['class_counts'].items():
        avg_conf = summary['class_confidences'][class_name]
        lines.append(f"- **{class_name}**: {count} (avg conf: {avg_conf:.2f})")
    
    return "\n".join(lines)
