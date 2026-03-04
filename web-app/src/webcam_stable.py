"""
Stable webcam processing for Streamlit - no threading
Uses st.camera_input or direct OpenCV loop with st.empty()
"""

import streamlit as st
import cv2
import time


def process_webcam_stable(confidence_threshold, model, ALLOWED_CLASSES):
    """Webcam processing using a simple main-thread loop"""
    st.subheader("📹 Live Webcam Detection")

    # Identify person class ID dynamically
    person_class_id = None
    if hasattr(model, 'names'):
        for k, v in model.names.items():
            if 'person' in str(v).lower():
                person_class_id = k
                break

    # Session state
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False

    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        start = st.button("📹 Start Webcam", type="primary")
    with col2:
        stop = st.button("⏹ Stop Webcam")

    if stop:
        st.session_state.webcam_active = False
        st.warning("📹 Webcam stopped.")
        return

    if start:
        st.session_state.webcam_active = True

    if not st.session_state.webcam_active:
        st.info("📹 Click 'Start Webcam' to begin detection.")
        return

    # Placeholders
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    status_placeholder.info("📹 Opening webcam...")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback: try without DSHOW
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Unable to access webcam. Check camera permissions.")
        st.session_state.webcam_active = False
        return

    status_placeholder.success("✅ Webcam active — detecting hazards...")

    frame_count = 0
    try:
        while st.session_state.webcam_active:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read frame from webcam.")
                break

            person_count = 0
            ppe_count = 0

            # Run detection every 2nd frame for speed
            if frame_count % 2 == 0:
                try:
                    results = model(frame, conf=confidence_threshold, verbose=False)
                    annotated_frame = frame.copy()

                    for result in results:
                        for box in result.boxes:
                            class_id = int(box.cls)
                            if class_id not in ALLOWED_CLASSES:
                                continue

                            class_name = model.names[class_id]
                            conf = float(box.conf)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            # Color logic
                            if person_class_id is not None and class_id == person_class_id:
                                color = (255, 255, 0)  # Cyan (BGR)
                                label = "Person"
                                person_count += 1
                            elif "no" in class_name.lower():
                                color = (0, 0, 255)  # Red
                                label = f"{class_name} {conf:.2f}"
                                ppe_count += 1
                            else:
                                color = (0, 255, 0)  # Green
                                label = f"{class_name} {conf:.2f}"
                                ppe_count += 1

                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                            cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
                            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                except Exception as e:
                    annotated_frame = frame.copy()
                    print(f"Detection error: {e}")
            else:
                annotated_frame = frame.copy()

            # Display
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
            status_placeholder.info(
                f"📹 Frame {frame_count} | Persons: {person_count} | PPE items: {ppe_count}"
            )

            frame_count += 1
            time.sleep(0.03)  # ~30 fps cap

    except Exception as e:
        st.error(f"❌ Webcam error: {e}")
    finally:
        cap.release()
        st.session_state.webcam_active = False
