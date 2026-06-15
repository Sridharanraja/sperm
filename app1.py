import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import math
from collections import defaultdict
from ultralytics import YOLO

# --- Page Configuration ---
st.set_page_config(page_title="Sperm Analysis AI", layout="wide")
st.title("🔬 Sperm Detection & Motility Tracker")

# --- Load Model ---
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Settings")
model_path = st.sidebar.text_input("Model Path", "./best_15-06_300.pt")
confidence = st.sidebar.slider("Detection Confidence", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Speed Calibration")
fps_input = st.sidebar.number_input("Video FPS", value=30.0, step=1.0)

st.sidebar.subheader("📏 Calibration Factor")
calib_mode = st.sidebar.radio("Set Scale By:", ["Enter Manually", "Calculate from Ruler Slide"])

if calib_mode == "Calculate from Ruler Slide":
    known_um = st.sidebar.number_input("Known Distance (µm)", value=50.0, step=10.0)
    measured_px = st.sidebar.number_input("Pixels Measured", value=100.0, step=10.0)
    calib_factor = known_um / measured_px if measured_px > 0 else 0.5
    st.sidebar.success(f"✅ Scale: **{calib_factor:.3f} µm/px**")
else:
    calib_factor = st.sidebar.number_input("Calibration (µm per pixel)", value=0.500, format="%.3f")

model = load_model(model_path)

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "wmv","asf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type in ['jpg', 'jpeg', 'png', 'bmp']:
        # --- IMAGE PROCESSING ---
        st.subheader("Image Analysis: Sperm Count")
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        if model:
            results = model.predict(img_cv2, conf=confidence)
            total_count = len(results[0].boxes)
            
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(res_rgb, use_container_width=True)
            with col2:
                st.metric(label="Total Sperm Count", value=total_count)

    elif file_type in ['mp4', 'avi', 'mov', 'wmv', 'asf']:
        # --- VIDEO PROCESSING ---
        st.subheader("Video Analysis: Live Tracking")
        
        if st.button("Start Video Tracking"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}')
            tfile.write(file_bytes)
            tfile.flush()

            cap = cv2.VideoCapture(tfile.name)
            frame_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            track_history = defaultdict(list)
            speed_history = defaultdict(float)

            while cap.isOpened():
                success, frame = cap.read()
                if not success: break

                results = model.track(frame, persist=True, conf=confidence, tracker="bytetrack.yaml", verbose=False)
                annotated_frame = frame.copy()
                
                current_count = 0
                total_speed = 0.0
                tracked_count = 0

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    current_count = len(track_ids)

                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = box
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                        # Track Movement
                        track = track_history[track_id]
                        track.append((cx, cy))
                        if len(track) > 30: track.pop(0)

                        # Calculate Speed
                        if len(track) > 1:
                            dist_px = math.hypot(track[-1][0] - track[-2][0], track[-1][1] - track[-2][1])
                            speed = (dist_px * calib_factor) * fps_input
                            speed_history[track_id] = (speed_history[track_id] * 0.8) + (speed * 0.2)

                        curr_v = speed_history[track_id]
                        total_speed += curr_v
                        tracked_count += 1

                        # Draw Visuals (Green for all sperm)
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"ID:{track_id} {curr_v:.1f}um/s", (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Update UI
                avg_v = (total_speed / tracked_count) if tracked_count > 0 else 0.0
                frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with metrics_placeholder.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Live Count", current_count)
                    m2.metric("Total Unique Tracks", len(track_history))
                    m3.metric("Avg Speed (µm/s)", f"{avg_v:.1f}")

            cap.release()
