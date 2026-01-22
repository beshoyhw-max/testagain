import streamlit as st
import cv2
import time
import os
import glob
from camera_manager import CameraManager
from PIL import Image

# Set Page Config
st.set_page_config(
    page_title="Enterprise Phone Detection",
    page_icon="📢",
    layout="wide"
)

# Custom CSS for Enterprise look
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .alert-box {
        background-color: #ff4b4b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    div[data-testid="stImage"] img {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📢 Enterprise Phone Monitor System")

# --- Load Camera Manager (Singleton) ---
@st.cache_resource
def get_camera_manager():
    return CameraManager()

try:
    manager = get_camera_manager()
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.header("⚙️ Global Controls")

# Detection Sensitivity
conf_threshold = st.sidebar.slider(
    "Detection Sensitivity", 
    0.1, 1.0, 0.25,
    help="Lower = more sensitive (more detections, more false positives)"
)
manager.update_global_conf(conf_threshold)

st.sidebar.markdown("---")

# Time-Based Thresholds
st.sidebar.subheader("📱 Phone Detection")
phone_duration = st.sidebar.slider(
    "Alert after continuous use (seconds)",
    min_value=1,
    max_value=30,
    value=5,
    step=1,
    help="Person must be using phone continuously for this duration before alert"
)
manager.update_phone_duration(phone_duration)

st.sidebar.markdown("---")

st.sidebar.subheader("😴 Sleep Detection")
sleep_duration = st.sidebar.slider(
    "Alert after sleeping (seconds)",
    min_value=3,
    max_value=60,
    value=10,
    step=1,
    help="Person must be sleeping continuously for this duration before alert"
)
manager.update_sleep_duration(sleep_duration)

st.sidebar.markdown("---")

st.sidebar.subheader("📸 Evidence Cooldown")
cooldown_duration = st.sidebar.slider(
    "Cooldown between screenshots (seconds)",
    min_value=30,
    max_value=300,
    value=120,
    step=10,
    help="Minimum time between saving evidence for the same person"
)
manager.update_cooldown_duration(cooldown_duration)

st.sidebar.markdown("---")
st.sidebar.info(f"Active Cameras: {len(manager.get_active_cameras())}")

# --- Navigation ---
page_selection = st.radio(
    "Navigate", 
    ["🔴 Live Dashboard", "📸 Evidence Log", "⚙️ Configuration"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# --- Page 1: Live Dashboard ---
if page_selection == "🔴 Live Dashboard":
    # 1. Global Alert System
    active_cams = manager.get_active_cameras()
    
    # Check for any active alerts across all cameras
    alert_texting = []
    alert_sleeping = []
    
    for cam in active_cams.values():
        status = cam.get_status()
        if status == "texting":
            alert_texting.append(cam.camera_name)
        elif status == "sleeping":
            alert_sleeping.append(cam.camera_name)
    
    alert_placeholder = st.empty()
    if alert_texting:
        names_str = ", ".join(alert_texting)
        alert_placeholder.markdown(
            f'<div class="alert-box">⚠️ ALERT: PHONE DETECTED IN: {names_str}</div>', 
            unsafe_allow_html=True
        )
    elif alert_sleeping:
        names_str = ", ".join(alert_sleeping)
        alert_placeholder.markdown(
            f'<div class="alert-box" style="background-color: #6a0dad;">💤 ALERT: SLEEP DETECTED IN: {names_str}</div>', 
            unsafe_allow_html=True
        )
    else:
        alert_placeholder.empty()

    # 2. Camera Grid
    if not active_cams:
        st.warning("No cameras configured. Go to Configuration tab.")
    else:
        cols = st.columns(2)
        
        cam_containers = {}
        cam_ids = list(active_cams.keys())
        
        for idx, cam_id in enumerate(cam_ids):
            col_idx = idx % 2
            with cols[col_idx]:
                cam = active_cams[cam_id]
                st.subheader(f"📹 {cam.camera_name}")
                frame_view = st.empty()
                status_text = st.empty()
                
                cam_containers[cam_id] = {
                    "frame": frame_view,
                    "status": status_text,
                    "thread": cam
                }

        # Auto-Refresh Loop
        run_monitor = st.checkbox("Start Live Monitor", value=True, key="run_live_monitor")
        
        if run_monitor:
            placeholder = st.empty()
            with placeholder.container():
                pass

            while True:
                loop_texting = []
                loop_sleeping = []
                
                for cam_id, container in list(cam_containers.items()):
                    thread = container["thread"]
                    frame = thread.get_frame()
                    status = thread.get_status()
                    
                    # Update Alert Logic
                    if status == "texting":
                        loop_texting.append(thread.camera_name)
                    elif status == "sleeping":
                        loop_sleeping.append(thread.camera_name)
                    
                    # Update Status Text
                    if status == "texting":
                        container["status"].markdown(":red[**PHONE DETECTED**]")
                    elif status == "sleeping":
                        container["status"].markdown(":blue[**SLEEP DETECTED**]")
                    elif status == "safe":
                        container["status"].markdown(":green[**SAFE**]")
                    elif status == "disconnected":
                        container["status"].markdown(":orange[**CONNECTING...**]")
                    else:
                        container["status"].write(status)
                        
                    # Update Frame
                    if frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        container["frame"].image(frame_rgb, width="stretch")
                    else:
                        container["frame"].info("No Signal")
                
                # Update Global Alert inside loop
                if loop_texting:
                    names_str = ", ".join(loop_texting)
                    alert_placeholder.markdown(
                        f'<div class="alert-box">⚠️ ALERT: PHONE DETECTED IN: {names_str}</div>', 
                        unsafe_allow_html=True
                    )
                elif loop_sleeping:
                    names_str = ", ".join(loop_sleeping)
                    alert_placeholder.markdown(
                        f'<div class="alert-box" style="background-color: #6a0dad;">💤 ALERT: SLEEP DETECTED IN: {names_str}</div>', 
                        unsafe_allow_html=True
                    )
                else:
                    alert_placeholder.empty()
                
                # Sleep to limit UI refresh rate
                time.sleep(0.1)

# --- Page 2: Evidence Log ---
elif page_selection == "📸 Evidence Log":
    st.subheader("Infraction History")
    
    if st.button("Refresh Gallery"):
        pass
        
    image_files = glob.glob("detections/*.jpg")
    image_files.sort(key=os.path.getmtime, reverse=True)
    
    if not image_files:
        st.info("No evidence collected yet.")
    else:
        cols = st.columns(4)
        for idx, img_path in enumerate(image_files):
            with cols[idx % 4]:
                image = Image.open(img_path)
                st.image(image, width="stretch")
                st.caption(os.path.basename(img_path))

# --- Page 3: Configuration ---
elif page_selection == "⚙️ Configuration":
    st.header("Camera Management")
    
    # List Existing
    st.subheader("Active Cameras")
    
    active_cams = manager.get_active_cameras()
    
    if not active_cams:
        st.info("No cameras configured yet.")
    else:
        for cam_id, cam in list(active_cams.items()):
            col1, col2, col3 = st.columns([1, 3, 1])
            col1.write(f"**ID: {cam_id}**")
            col2.write(f"{cam.camera_name} ({cam.source})")
            if col3.button("Remove", key=f"del_{cam_id}"):
                manager.remove_camera(cam_id)
                st.rerun()
            
    st.markdown("---")
    st.subheader("Add New Camera")
    
    with st.form("add_cam_form"):
        new_name = st.text_input("Camera Name (e.g., Meeting Room 1)")
        new_source = st.text_input("Source (URL or ID)", value="0")
        submitted = st.form_submit_button("Add Camera")
        
        if submitted:
            if new_name:
                manager.add_camera(new_name, new_source)
                st.success(f"Added {new_name}")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Name is required.")

    st.markdown("---")
    st.info("💡 Use '0', '1' for local webcams. Use 'rtsp://...' for IP cameras.")