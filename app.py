
import streamlit as st
import cv2
import tempfile
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import os
import torch
from ultralytics import YOLO

# Import our modules
from src.config import (
    TEAM_A, TEAM_B, NEW_REFEREE_CLASS_ID, NEW_COACH_CLASS_ID, 
    NEW_HOOP_CLASS_ID, BALL_CLASS_ID, NON_PLAYER_OBJECTS, FPS
)
from src.tracker import StablePlayerTracker
from src.analytics import calculate_speed, detect_ball_acquisition
from src.visualization import generate_heatmap, draw_bbox, draw_text
from src.utils import get_video_properties, resize_frame

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="Basketball Analytics Pro",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #41444b;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #00ADB5;
    }
    .metric-label {
        font-size: 14px;
        color: #a0a0a0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏀 Basketball Analytics Pro")
st.markdown("### advanced tracking, speed analysis & heatmap generation")

# --- SIDEBAR CONFIG ---
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model Path", "best11.pt")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)
device_option = st.sidebar.selectbox("Device", ["cuda", "cpu"])

# --- HELPERS ---
@st.cache_resource
def load_models(model_path, device_option):
    try:
        yolo_model = YOLO(model_path)
    except Exception as e:
        yolo_model = None
        st.error(f"Failed to load YOLO model: {e}")

    try:
        import torchreid
        reid_model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        device = torch.device(device_option if torch.cuda.is_available() else "cpu")
        reid_model.to(device)
        reid_model.eval()
    except ImportError:
        reid_model = None
        device = "cpu"
    except Exception as e:
        st.warning(f"ReID model loading warning: {e}")
        reid_model = None
        device = "cpu"

    return yolo_model, reid_model, device

# --- MAIN APP LOGIC ---

uploaded_file = st.file_uploader("Upload Gameplay Video (MP4)", type=["mp4", "avi"])

if uploaded_file is not None:
    # Save uploaded file to temp
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Display Video
    col1, col2 = st.columns([2, 1])
    with col1:
        st.video(video_path)
    with col2:
        props = get_video_properties(video_path)
        if props:
            st.markdown("### Video Info")
            st.json(props)

    if st.button("Start Analysis", type="primary"):
        yolo_model, reid_model, device = load_models(model_path, device_option)
        
        if yolo_model is None:
            st.error("YOLO model not found. Please check path.")
            st.stop()
            
        tracker = StablePlayerTracker(reid_model, device)
        cap = cv2.VideoCapture(video_path)
        
        st_frame = st.empty()
        st_progress = st.progress(0)
        st_status = st.empty()
        
        output_video_path = "processed_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, FPS, (640, 384))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        
        latest_speeds = {}
        current_possession = None
        possession_counts = {"Team A": 0, "Team B": 0}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            tracker.current_frame = frame_idx
            
            # Resize for consistent processing
            frame_resized = resize_frame(frame, width=640, height=384)
            output_frame = frame_resized.copy()
            
            # Periodic Cleanup
            if frame_idx % 15 == 0:
                tracker.cleanup()
                
            # YOLO Tracking
            results = yolo_model.track(frame_resized, persist=True, tracker="bytetrack.yaml", verbose=False, conf=conf_threshold)[0]
            
            ball_pos = None # For calculating possession
            
            if results.boxes.id is not None:
                ids = results.boxes.id.cpu().numpy().astype(int)
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                
                for track_id, box, cls in zip(ids, boxes, classes):
                    x1, y1, x2, y2 = map(int, box)
                    
                     # Non-player/static objects
                    if cls in NON_PLAYER_OBJECTS:
                        obj_config = NON_PLAYER_OBJECTS[cls]
                        draw_bbox(output_frame, (x1,y1,x2,y2), obj_config['color'], obj_config['label'])
                        
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        
                        if cls == NEW_HOOP_CLASS_ID:
                             tracker.record_hoop_position(center)
                        elif cls == BALL_CLASS_ID:
                             ball_pos = center
                        continue
                        
                    # Filter for players/ref/coach
                    valid_classes = [TEAM_A['class_id'], TEAM_B['class_id'], NEW_REFEREE_CLASS_ID, NEW_COACH_CLASS_ID]
                    if cls not in valid_classes:
                        continue
                        
                    crop = frame_resized[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    
                    static_id = tracker.assign_id(track_id, cls, crop)
                    if static_id is None: continue
                    
                    # Determine Label & Color
                    label = f"ID: {static_id}"
                    color = (255, 255, 255)
                    
                    if cls == TEAM_A['class_id']:
                        color = TEAM_A['color']
                        label = f"{TEAM_A['name']} {static_id}"
                        
                        # Speed Check
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        tracker.player_history[static_id].append(center)
                        tracker.record_position(static_id, center)
                        
                        speed_kph = 0
                        if len(tracker.player_history[static_id]) >= 2:
                             p1 = tracker.player_history[static_id][-2]
                             p2 = tracker.player_history[static_id][-1]
                             speed_kph = calculate_speed(p1, p2)
                        
                        tracker.record_speed(static_id, speed_kph)
                        latest_speeds[static_id] = speed_kph
                        label += f" | {speed_kph:.1f} km/h"
                        
                    elif cls == TEAM_B['class_id']:
                         color = TEAM_B['color']
                         label = f"{TEAM_B['name']} {static_id}"
                         
                         center = ((x1 + x2) // 2, (y1 + y2) // 2)
                         tracker.player_history[static_id].append(center)
                         tracker.record_position(static_id, center)
                         
                         speed_kph = 0
                         if len(tracker.player_history[static_id]) >= 2:
                             p1 = tracker.player_history[static_id][-2]
                             p2 = tracker.player_history[static_id][-1]
                             speed_kph = calculate_speed(p1, p2)
                             
                         tracker.record_speed(static_id, speed_kph)
                         latest_speeds[static_id] = speed_kph
                         label += f" | {speed_kph:.1f} km/h"

                    # Draw
                    draw_bbox(output_frame, (x1, y1, x2, y2), color, label)
            

            # Ball Acquisition Check
            current_player_positions = {sid: tracker.player_history[sid][-1] if tracker.player_history[sid] else None 
                                        for sid in list(tracker.assigned_ids.values())}
            
            possession_id = detect_ball_acquisition(current_player_positions, ball_pos)
            
            if possession_id:
                # Determine team
                if possession_id in TEAM_A['id_range']:
                    current_possession = "Team A"
                elif possession_id in TEAM_B['id_range']:
                    current_possession = "Team B"
            
            # Update counters based on sticky possession
            if current_possession:
                possession_counts[current_possession] += 1
                draw_text(output_frame, f"Possession: {current_possession}", (10, 30), color=(0, 255, 0), scale=0.8)

            out.write(output_frame)
            
            # UI Updates
            st_frame.image(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            st_progress.progress(min(frame_idx / total_frames, 1.0))
            st_status.text(f"Processing Frame {frame_idx}/{total_frames}")
            
        cap.release()
        out.release()
        
        st.success("Analysis Complete!")
        
        # --- ANALYTICS DASHBOARD ---
        st.markdown("## 📊 Game Analytics")
        
        # --- NEW: TEAM STATS SECTION ---
        st.subheader("Team Performance")
        col_team1, col_team2, col_team3 = st.columns(3)
        
        # 1. Possession Analysis
        total_possession_frames = possession_counts["Team A"] + possession_counts["Team B"]
        
        if total_possession_frames > 0:
            team_a_poss_pct = (possession_counts["Team A"] / total_possession_frames) * 100
            team_b_poss_pct = (possession_counts["Team B"] / total_possession_frames) * 100
            
            # Pie Chart
            fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
            ax_pie.pie([team_a_poss_pct, team_b_poss_pct], labels=["Team A", "Team B"], 
                      colors=['#EE82EE', '#FF0000'], autopct='%1.1f%%', startangle=90)
            ax_pie.set_title("Possession Distribution", color='white')
            fig_pie.patch.set_alpha(0) 
            
            with col_team1:
                st.pyplot(fig_pie)
                
            # Possession Dominance
            leader = "Balanced"
            if team_a_poss_pct > team_b_poss_pct:
                leader = "Team A"
            elif team_b_poss_pct > team_a_poss_pct:
                leader = "Team B"
                
            st.metric("Possession Leader", leader)
                
        else:
             with col_team1:
                st.info("No possession detected.")

        # 2. Team Average Speeds
        all_speeds = tracker.get_all_speed_data()
        
        team_a_speeds = []
        team_b_speeds = []
        
        # Valid player IDs for filtering
        valid_player_ids = list(TEAM_A['id_range']) + list(TEAM_B['id_range'])
        
        for pid, speeds in all_speeds.items():
            if not speeds: continue
            
            # Strict filtering: Only Players 1-10
            if pid not in valid_player_ids:
                continue
                
            vals = [s[1] for s in speeds]
            avg_p_speed = np.mean(vals)
            
            if pid in TEAM_A['id_range']:
                team_a_speeds.append(avg_p_speed)
            elif pid in TEAM_B['id_range']:
                team_b_speeds.append(avg_p_speed)
                
        avg_speed_a = np.mean(team_a_speeds) if team_a_speeds else 0.0
        avg_speed_b = np.mean(team_b_speeds) if team_b_speeds else 0.0
        
        with col_team2:
             st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Team A Avg Speed</div>
                <div class="metric-value">{avg_speed_a:.2f} km/h</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_team3:
             st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Team B Avg Speed</div>
                <div class="metric-value">{avg_speed_b:.2f} km/h</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # 3. Heatmaps (Static)
        st.subheader("Player Movement Heatmaps")
        hm_col1, hm_col2 = st.columns(2)
        
        team_a_positions = []
        team_b_positions = []
        
        for static_id, positions in tracker.position_data.items():
            if static_id in TEAM_A['id_range']:
                team_a_positions.extend(positions)
            elif static_id in TEAM_B['id_range']:
                team_b_positions.extend(positions)
        
        with hm_col1:
            if team_a_positions:
                hm_path_a = generate_heatmap(team_a_positions, "Team A Coverage", tracker.hoop_positions, "heatmap_team_a.png")
                if hm_path_a: st.image(hm_path_a)
            else:
                st.info("Not enough data for Team A Heatmap")

        with hm_col2:
            if team_b_positions:
                hm_path_b = generate_heatmap(team_b_positions, "Team B Coverage", tracker.hoop_positions, "heatmap_team_b.png")
                if hm_path_b: st.image(hm_path_b)
            else:
                 st.info("Not enough data for Team B Heatmap")
                 
        # 4. Player Stats Table (Filtered)
        with st.expander("Detailed Player Stats (Players Only)"):
            stats = []
            for pid, speeds in all_speeds.items():
                if not speeds: continue
                if pid not in valid_player_ids: continue # Filter non-players
                
                vals = [s[1] for s in speeds]
                avg = np.mean(vals)
                mx = np.max(vals)
                
                team = "Unknown"
                if pid in TEAM_A['id_range']: team = "Team A"
                elif pid in TEAM_B['id_range']: team = "Team B"
                
                stats.append({"Player ID": pid, "Team": team, "Avg Speed (km/h)": round(avg, 2), "Max Speed (km/h)": round(mx, 2)})
                
            if stats:
                df_stats = pd.DataFrame(stats)
                st.dataframe(df_stats, use_container_width=True)
            
        # 5. Download Processed Video
        with open(output_video_path, 'rb') as f:
             st.download_button("Download Annotated Video", f, file_name="analyzed_game.mp4")

