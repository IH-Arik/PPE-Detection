"""
Fixed webcam function for PPE Detection app
"""

import streamlit as st
import cv2
import time

def draw_all_boxes(image, boxes, model):
    """Draw bounding boxes for all detected classes"""
    annotated_image = image.copy()
    
    # Identify person class ID dynamically
    person_class_id = None
    if hasattr(model, 'names'):
        for k, v in model.names.items():
            if 'person' in str(v).lower():
                person_class_id = k
                break

    for box in boxes:
        # Get box coordinates and properties
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls)
        # Safety check
        if 0 <= class_id < len(model.names):
            class_name = model.names[class_id]
        else:
            class_name = f"Class {class_id}"
            
        confidence = float(box.conf)
        
        # Person styling
        if person_class_id is not None and class_id == person_class_id:
            color = (255, 255, 0)  # Light blue in BGR
        elif "no" in class_name.lower():
            color = (0, 0, 255) # Red for violations
        else:
            color = (0, 255, 0) # Green for PPE
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"{class_name} {confidence:.2f}" if person_class_id is None or class_id != person_class_id else "Person"
            
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated_image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        
        # Draw label text
        cv2.putText(annotated_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return annotated_image

def process_webcam_fixed(confidence_threshold, model, ALLOWED_CLASSES):
    """Fixed webcam processing function"""
    st.subheader("📹 Live Webcam Detection")
    
    # Initialize session state for webcam control
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    
    # Create placeholders
    frame_placeholder = st.empty()
    summary_placeholder = st.empty()
    
    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📹 Start Webcam", type="primary", disabled=st.session_state.webcam_active):
            st.session_state.webcam_active = True
            st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if not st.session_state.cap.isOpened():
                st.error("❌ Unable to access webcam. Please check your camera permissions.")
                st.session_state.webcam_active = False
                st.session_state.cap = None
                return
            
            st.success("📹 Webcam started! Click 'Stop' to end.")
    
    with col2:
        if st.button("⏹ Stop Webcam", disabled=not st.session_state.webcam_active):
            st.session_state.webcam_active = False
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.warning("📹 Webcam stopped.")
            return
    
    # Process frames if webcam is active
    if st.session_state.webcam_active and st.session_state.cap:
        try:
            recent_detections = []
            
            for i in range(100):  # Process 100 frames max
                if not st.session_state.webcam_active:
                    break
                
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.error("❌ Failed to read from webcam")
                    break
                
                # Process frame
                results = model(frame, conf=confidence_threshold)
                
                # Separate detections: persons vs PPE items
                person_boxes = []
                ppe_detections = []
                
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = model.names[class_id]
                        confidence = float(box.conf)
                        
                        # Identify person class ID dynamically
                        is_person = 'person' in str(class_name).lower()
                        
                        # Only show bounding boxes for persons
                        if is_person:
                            person_boxes.append(box)
                        
                        # Log all other PPE detections
                        else:
                            ppe_detections.append({
                                'class': class_name,
                                'confidence': confidence,
                                'class_id': class_id
                            })
                
                # Annotate frame with all boxes
                # Collect all boxes from results
                all_boxes = []
                for result in results:
                    for box in result.boxes:
                        all_boxes.append(box)
                        
                if all_boxes:
                    annotated_frame = draw_all_boxes(frame.copy(), all_boxes, model)
                else:
                    annotated_frame = frame.copy()
                
                # Log PPE detections to console
                if ppe_detections:
                    ppe_list = [f'{d["class"]}({d["confidence"]:.2f})' for d in ppe_detections]
                    print(f"PPE Detected: {ppe_list}")
                
                # Store all detections for summary
                all_detections = person_boxes + ppe_detections
                
                # Convert BGR to RGB for display
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(annotated_rgb, channels="RGB", width=800)
                
                # Update recent detections
                recent_detections.extend(all_detections)
                if len(recent_detections) > 100:
                    recent_detections = recent_detections[-100:]
                
                # Update summary every 10 frames
                if i % 10 == 0 and len(recent_detections) > 0:
                    update_summary(recent_detections, summary_placeholder)
                
                # Small delay
                time.sleep(0.1)
                
        except Exception as e:
            st.error(f"Error in webcam processing: {e}")
        finally:
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.session_state.webcam_active = False
    
    else:
        st.info("📹 Click 'Start Webcam' to begin live detection.")

def update_summary(detections, placeholder):
    """Update detection summary"""
    if not detections:
        return
    
    # Count detections by class
    class_counts = {}
    class_confidences = {}
    
    for det in detections:
        # Handle both box objects and dict objects
        if hasattr(det, 'cls'):
            # This is a box object (person)
            class_name = 'Person'
            confidence = float(det.conf)
        else:
            # This is a dict (PPE detection)
            class_name = det['class']
            confidence = det['confidence']
        
        if class_name not in class_counts:
            class_counts[class_name] = 0
            class_confidences[class_name] = []
        
        class_counts[class_name] += 1
        class_confidences[class_name].append(confidence)
    
    # Display summary
    with placeholder.container():
        st.subheader("📊 Detection Summary")
        for class_name, count in class_counts.items():
            avg_conf = sum(class_confidences[class_name]) / len(class_confidences[class_name])
            st.write(f"**{class_name}**: {count} detections (avg confidence: {avg_conf:.2f})")
        st.write(f"**Total Detections**: {len(detections)}")
