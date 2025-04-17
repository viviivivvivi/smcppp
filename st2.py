import streamlit as st
import cv2
import pandas as pd
import numpy as np
import time
from datetime import datetime
import requests
import threading
import json
from PIL import Image
import io
from pls3 import CombinedSystem, AlertService




class SecuritySystemClient:
    def __init__(self, api_url="http://127.0.0.1:8000"):
        self.api_url = api_url
        self.connected = False
        self.detection_history = {
            "violence": 0,
            "suspicious": 0,
            "weapon": 0,
            "normal": 0
        }
        self.alert_history = []
        
    def connect(self):
        try:
            response = requests.get(f"{self.api_url}/status", timeout=1)
            if response.status_code == 200:
                self.connected = True
                return True
            return False
        except:
            self.connected = False
            return False
            
    def get_frame(self):
        if not self.connected:
            return None

        try:
            response = requests.get(f"{self.api_url}/frame", timeout=1)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                return np.array(img)
            return None
        except:
            return None

            
    def get_stats(self):
        if not self.connected:
            return self.detection_history
            
        try:
            response = requests.get(f"{self.api_url}/stats", timeout=1)
            if response.status_code == 200:
                self.detection_history = response.json()
            return self.detection_history
        except:
            return self.detection_history
            
    def get_alerts(self):
        if not self.connected:
            return self.alert_history
            
        try:
            response = requests.get(f"{self.api_url}/alerts", timeout=1)
            if response.status_code == 200:
                self.alert_history = response.json()
            return self.alert_history
        except:
            return self.alert_history
            
    def send_config(self, config):
        if not self.connected:
            return False
            
        try:
            response = requests.post(f"{self.api_url}/config", json=config, timeout=1)
            return response.status_code == 200
        except:
            return False
            
    def send_alert(self, alert_type, message=""):
        if not self.connected:
            return False
            
        try:
            response = requests.get(f"{self.api_url}/send_alert?type={alert_type}&message={message}", timeout=1)
            return response.status_code == 200
        except:
            return False

    def sound_detection(self,sound_value,message ):
        response = requests.get(f"{self.api_url}/sound_detected", params={"type": sound_value, "message": message}, timeout=3)  # Timeout increased to 3 seconds
        if response.status_code == 200:

            response_data = response.json()

            return True
        else:
            print(f"Failed to send sound detection: {response.status_code}")
            return False


def generate_frame():

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    

    font = cv2.FONT_HERSHEY_SIMPLEX

    if np.random.random() < 0.05:
        x1, y1 = np.random.randint(100, 500), np.random.randint(100, 400)
        w, h = np.random.randint(50, 150), np.random.randint(50, 150)
        x2, y2 = x1 + w, y1 + h
        
        color = (0, 0, 255) if np.random.random() < 0.1 else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = "Person"
        cv2.putText(frame, label, (x1, y1-10), font, 0.5, color, 2)
    
    return frame



def main():
    st.set_page_config(
        page_title="Security Monitoring Dashboard",
        page_icon="🔒",
        layout="wide"
    )

    if 'client' not in st.session_state:
        st.session_state.client = SecuritySystemClient()
    
    if 'camera_status' not in st.session_state:
        st.session_state.camera_status = False

    # Title and intro
    st.title("🔒 Security Monitoring System")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("System Settings")
        
        # Connection settings
        st.subheader("Connection")
        api_url = st.text_input("API URL", "http://127.0.0.1:8000")
        
        col1= st.columns(1)
        connect_button = st.button("Connect")
   
        
        if connect_button:
            st.session_state.client = SecuritySystemClient(api_url)
            if st.session_state.client.connect():
                st.success("Connected successfully!")

        
        # Camera settings
        st.subheader("Camera")
        camera_source = st.selectbox(
            "Camera Source", 
            ["0", "1", "2", "rtsp://example.com/stream"], 
            index=1
        )
        
        camera_toggle = st.checkbox("Enable Camera", value=False)
        if camera_toggle != st.session_state.camera_status:
            st.session_state.camera_status = camera_toggle

            st.session_state.client.send_config({"camera": {"enabled": camera_toggle, "source": camera_source}})
        
        # Alert settings
        st.subheader("Alert Settings")
        alert_settings = {
            "esp32": st.checkbox("Enable ESP32 Alerts", value=True),
        }
        
        if st.button("Save Alert Settings"):
            if not st.session_state.demo_mode:
                success = st.session_state.client.send_config({"alerts": alert_settings})
                if success:
                    st.success("Settings saved!")
                else:
                    st.error("Failed to save settings")
            else:
                st.info("Settings would be saved in live mode")
        
        # Manual alert trigger
        st.subheader("Manual Control")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Test Alert"):
                st.session_state.client.send_alert("test", "Manual test alert")

        
        with col2:
            if st.button("Clear Stats"):
                st.session_state.client.send_config({"clear_stats": True})


    # Main content area
    col1, col2 = st.columns([3, 1])
     
    # Video feed column
    with col1:
        st.header("Live Video Feed")
        video_placeholder = st.empty()
        
        st.markdown("---")
        
        # System status indicators
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            # if st.session_state.demo_mode:
            #     st.info("Demo Mode Active")
            if st.session_state.client.connected:
                st.success("System Connected")
            else:
                st.error("System Disconnected")
        with status_col2:
            if st.session_state.camera_status:
                st.success("Camera Active")
            else:
                st.warning("Camera Inactive")
        with status_col3:
            status_indicator = st.empty()
    
    # Stats column
    with col2:
        st.header("Detection Stats")
        stats_placeholder = st.empty()
        
        st.header("Recent Alerts")
        alerts_container = st.container()
    
   
    while True:
   
        frame = st.session_state.client.get_frame()
        if frame is None:
            frame = generate_frame() 
        stats = st.session_state.client.get_stats()
        alerts = st.session_state.client.get_alerts()
        
        if st.session_state.camera_status:
            video_placeholder.image(frame, channels="RGB", use_column_width=True)
        else:
            video_placeholder.warning("Camera is currently disabled")
        
        if "violence" in stats and stats["violence"] > 0:
            status_indicator.error("⚠️ Violence Detected")
        elif "weapon" in stats and stats["weapon"] > 0:
            status_indicator.error("⚠️ Weapon Detected")
        elif "suspicious" in stats and stats["suspicious"] > 0:
            status_indicator.warning("⚠ Suspicious Activity")
        else:
            status_indicator.success("✓ Normal Activity")
        

        stats_df = pd.DataFrame({
            'Detection Type': list(stats.keys()),
            'Count': list(stats.values())
        })
        stats_placeholder.bar_chart(stats_df.set_index('Detection Type'))
        

        alerts_container.empty()  
        for i, alert in enumerate(reversed(alerts[-5:])):
            alert_type = alert.get("type", "unknown")
            message = alert.get("message", "")
            timestamp = alert.get("timestamp", "")

            if alert_type in ["violence", "weapon"]:
                alerts_container.error(f"🚨 {timestamp} - {alert_type.upper()}: {message}")
            elif alert_type == "suspicious":
                alerts_container.warning(f"⚠️ {timestamp} - {alert_type.upper()}: {message}")
            else:
                alerts_container.info(f"ℹ️ {timestamp} - {alert_type.upper()}: {message}")

                time.sleep(0.1)
        

        if not st.session_state.camera_status:
            break

if __name__ == "__main__":
    main() 