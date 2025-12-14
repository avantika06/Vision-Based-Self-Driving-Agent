import cv2
import streamlit as st
import tempfile
import numpy as np
import torch

# --- FIXES FOR PYTORCH 2.6+ ---
if hasattr(torch, 'classes'):
    torch.classes.__path__ = [] 

try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except (ImportError, AttributeError):
    pass

from ultralytics import YOLO
from agent import DrivingAgent

# --- HELPER: Traffic Light Color Detection ---
def get_traffic_light_color(frame, box):
    """
    Analyzes the pixels inside the bounding box to detect Red or Green.
    Returns: 'Red', 'Green', 'Yellow', or 'Unknown'
    """
    x1, y1, x2, y2 = map(int, box)
    
    # Crop the traffic light from the frame
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return "Unknown"

    # Convert to HSV color space for better color filtering
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define color ranges (These need tuning based on video lighting)
    # Red has two ranges in HSV (0-10 and 170-180)
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([180, 255, 255])
    
    green_lower = np.array([40, 100, 100]) # Green is roughly 40-80 hue
    green_upper = np.array([90, 255, 255])
    
    yellow_lower = np.array([15, 100, 100]) # Yellow is roughly 15-35 hue
    yellow_upper = np.array([35, 255, 255])

    # Create masks
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.add(mask_red1, mask_red2)
    
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Count pixels
    red_pixels = cv2.countNonZero(mask_red)
    green_pixels = cv2.countNonZero(mask_green)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    
    # Threshold to decide color (must be at least 5% of the area to count)
    total_pixels = roi.shape[0] * roi.shape[1]
    threshold = total_pixels * 0.05

    if red_pixels > threshold and red_pixels > green_pixels:
        return "Red"
    elif green_pixels > threshold and green_pixels > red_pixels:
        return "Green"
    elif yellow_pixels > threshold:
        return "Yellow"
    
    return "Unknown"

# --- MAIN APP ---
st.set_page_config(page_title="AI Self-Driving Agent", layout="wide")
st.title("🚗 Vision-Based Self-Driving Agent v2.0")

# Load Model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

try:
    model = load_model()
    agent = DrivingAgent()
except Exception as e:
    st.error(f"Error: {e}")

# Sidebar
conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.45, 0.05)

uploaded_file = st.file_uploader("Upload Dashcam Video", type=['mp4', 'mov'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    col1, col2 = st.columns([0.75, 0.25])
    with col1: st_frame = st.empty()
    with col2:
        st.subheader("Agent Telemetry")
        status = st.empty()
        reason_display = st.empty()
        speed_bar = st.progress(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Perception (YOLO)
        results = model(frame, conf=conf_threshold, verbose=False)[0]
        
        detections = []
        annotated_frame = frame.copy()

        if results.boxes is not None:
            for box in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls_id = box
                cls_id = int(cls_id)
                
                # Check for Traffic Light Color
                extra_info = None
                if cls_id == 9: # Traffic Light
                    color = get_traffic_light_color(frame, (x1, y1, x2, y2))
                    extra_info = color
                    
                    # Draw Color on Frame for Debugging
                    cv2.putText(annotated_frame, f"Light: {color}", (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                detections.append([cls_id, conf, x1, y1, x2, y2, extra_info])

        # 2. Reasoning (Agent)
        height, width, _ = frame.shape
        action, reason = agent.decide_action(detections, width, height)

        # 3. Visualization
        # Draw all YOLO boxes first
        for d in detections:
            # We draw simpler boxes ourselves or use YOLO's plotter if preferred
            # Let's use YOLO's plotter for convenience, but we already copied frame
            pass 
        annotated_frame = results.plot() # Overwrite with YOLO's nice drawing

        # Draw Heads-Up Display
        color_map = {
            "BRAKE": (0, 0, 255),       # Red
            "SLOW DOWN": (0, 165, 255), # Orange
            "ACCELERATE": (0, 255, 0),  # Green
            "STEER LEFT": (255, 255, 0),
            "STEER RIGHT": (255, 255, 0)
        }
        hud_color = color_map.get(action, (255, 255, 255))

        cv2.putText(annotated_frame, f"CMD: {action}", (30, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, hud_color, 4)
        cv2.putText(annotated_frame, f"WHY: {reason}", (30, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        st_frame.image(annotated_frame, channels="BGR")
        
        status.markdown(f"## **{action}**")
        reason_display.info(f"{reason}")
        
        # Speedometer Logic
        if action == "ACCELERATE": speed_bar.progress(80)
        elif action == "SLOW DOWN": speed_bar.progress(40)
        elif action == "BRAKE": speed_bar.progress(0)
        else: speed_bar.progress(50)

    cap.release()