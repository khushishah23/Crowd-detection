import streamlit as st
import cv2
import torch
import tempfile
import numpy as np
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
import pygame
from datetime import datetime
import logging
import plotly.graph_objects as go
import plotly.express as px

# Initialize alert sound
pygame.mixer.init()
ALERT_SOUND = "alert.wav"
alert_playing = False

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.classes = [0]  # Only person class
    return model

model = load_model()
tracker = DeepSort(max_age=30)
log_data = []

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('crowd_detection.log')
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def stop_alert():
    global alert_playing
    if alert_playing:
        pygame.mixer.music.stop()
        alert_playing = False
        logger.info("Alert stopped manually")

def detect_and_display(frame, threshold, stframe, live_data):
    global alert_playing
    global log_data

    frame = cv2.resize(frame, (320, 240))
    results = model(frame)
    detections = results.pred[0]
    people_dets = []
    positions = []

    for *xyxy, conf, cls in detections:
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, xyxy)
            people_dets.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), 'person'))
            positions.append(((x1 + x2) // 2, (y1 + y2) // 2))

    tracks = tracker.update_tracks(people_dets, frame=frame)
    people_count = 0

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        color = (0, 255, 0) if len(tracks) <= threshold else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        people_count += 1

    alert_triggered = False
    if people_count > threshold:
        cv2.putText(frame, "ALERT: CROWD LIMIT EXCEEDED!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        if not alert_playing:
            pygame.mixer.music.load(ALERT_SOUND)
            pygame.mixer.music.play(-1)
            alert_playing = True
            alert_triggered = True
            logger.warning("Crowd threshold exceeded")
    else:
        if alert_playing:
            pygame.mixer.music.stop()
            alert_playing = False
            logger.info("Crowd back under threshold")

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_data.append({'Time': timestamp, 'People Count': people_count, 'Alert Triggered': alert_triggered})
    logger.info(f"People count: {people_count}")

    live_data['Time'].append(timestamp)
    live_data['People Count'].append(people_count)
    live_data['Alert Triggered'].append(alert_triggered)
    live_data['Positions'].append(positions)
    current_status = "Alert" if alert_triggered else "Normal"
    if 'Status' not in live_data:
        live_data['Status'] = []
    live_data['Status'].append(current_status)

    stframe.image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), channels="RGB")

def process_video(cap, threshold, stop_button, frame_limit, frame_skip):
    stframe = st.empty()
    frame_count = 0
    live_data = {'Time': [], 'People Count': [], 'Alert Triggered': [], 'Positions': []}
    line_chart_placeholder = st.empty()
    heatmap_placeholder = st.empty()
    pie_chart_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        detect_and_display(frame, threshold, stframe, live_data)

        if len(live_data['People Count']) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=live_data['Time'], y=live_data['People Count'],
                                     mode='lines+markers', name="People Count",
                                     line=dict(color='royalblue', width=4)))
            fig.update_layout(title="Live People Count", xaxis_title="Time", yaxis_title="People Count",
                              plot_bgcolor='rgb(25, 25, 25)', paper_bgcolor='rgb(25, 25, 25)',
                              font=dict(color="white"))
            line_chart_placeholder.plotly_chart(fig, use_container_width=True)

        if len(live_data['Positions']) > 5:
            heatmap_data = np.zeros((480, 640))
            for positions in live_data['Positions'][-20:]:
                for x, y in positions:
                    if 0 <= x < 640 and 0 <= y < 480:
                        for i in range(max(0, y - 30), min(480, y + 30)):
                            for j in range(max(0, x - 30), min(640, x + 30)):
                                dist = np.sqrt((i - y) ** 2 + (j - x) ** 2)
                                if dist < 30:
                                    heatmap_data[i, j] += 30 - dist
            fig_heatmap = go.Figure(data=go.Heatmap(z=heatmap_data, colorscale='Inferno'))
            fig_heatmap.update_layout(title="Crowd Density Heatmap",
                                      plot_bgcolor='rgb(25, 25, 25)', paper_bgcolor='rgb(25, 25, 25)',
                                      font=dict(color="white"))
            heatmap_placeholder.plotly_chart(fig_heatmap, use_container_width=True)

        if len(live_data['Status']) > 0:
            status_counts = {"Normal": 0, "Alert": 0}
            for status in live_data['Status']:
                status_counts[status] += 1
            fig_pie = go.Figure(data=[go.Pie(labels=list(status_counts.keys()),
                                             values=list(status_counts.values()),
                                             marker_colors=['#2ecc71', '#e74c3c'])])
            fig_pie.update_layout(title="Monitoring Status Distribution",
                                  plot_bgcolor='rgb(25, 25, 25)', paper_bgcolor='rgb(25, 25, 25)',
                                  font=dict(color="white"))
            pie_chart_placeholder.plotly_chart(fig_pie, use_container_width=True)

        if stop_button or frame_count >= frame_limit:
            stop_alert()
            st.success(f"✅ Detection stopped at {frame_count} frames.")
            logger.info(f"Stopped after {frame_count} frames.")
            break

    cap.release()

# -------------------- Streamlit UI --------------------
st.title("🧠 Real-Time Crowd Detection Dashboard")

with st.sidebar:
    st.header("Configuration")
    mode = st.radio("Select Input Mode", ["Webcam", "Upload Video"])
    threshold = st.slider("People Threshold", 1, 20, 5)
    frame_limit = st.slider("Frame Limit", 10, 300, 100)
    frame_skip = st.slider("Frame Skip (higher = faster)", 1, 10, 5)

if mode == "Webcam":
    stop_button = st.button("⏹️ Stop Webcam Detection")
    if st.button("🎥 Start Webcam Detection"):
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            st.error("💥 Webcam not detected.")
        else:
            process_video(cap, threshold, stop_button, frame_limit, frame_skip)
            stop_alert()

elif mode == "Upload Video":
    uploaded_file = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        if cap.isOpened():
            stop_button = st.button("⏹️ Stop Video Detection")
            if st.button("📺 Start Video Processing"):
                process_video(cap, threshold, stop_button, frame_limit, frame_skip)
                stop_alert()
        else:
            st.error("💥 Could not open the uploaded video.")
