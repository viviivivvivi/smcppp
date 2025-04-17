import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from tensorflow import keras
from tensorflow.keras import layers
from flask import Flask, request, jsonify, Response
import requests
import threading
import time
from datetime import datetime
from twilio.rest import Client
import json
from flask_cors import CORS

class PersonDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        people_boxes = []

        if results.pose_landmarks:
            h, w, _ = frame.shape
            landmarks = results.pose_landmarks.landmark
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in landmarks:
                if lm.visibility > 0.5:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            people_boxes.append([x_min, y_min, x_max, y_max])
        return people_boxes

    def extract_132d_keypoints(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
            return np.array(keypoints) 
        return None

class IntegratedDetector:
    def __init__(self, standard_model_path="yolov8n.pt", custom_model_path="model1/train/weights/best.pt"):
        self.standard_model = YOLO(standard_model_path)
        self.custom_model = YOLO(custom_model_path)
        self.activity_classes = ["normal", "suspicious", "weapon"]  

    def detect_persons(self, frame):
        results = self.standard_model(frame, classes=0)
        people_boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                if conf > 0.5:
                    people_boxes.append([int(x1), int(y1), int(x2), int(y2)])
        return people_boxes

    def detect_activities(self, frame, people_boxes=None):
        activities = []
        if not people_boxes:
            results = self.custom_model(frame)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    if conf > 0.4:
                        activities.append((cls_id, conf, [int(x1), int(y1), int(x2), int(y2)]))
        else:
            for box in people_boxes:
                x_min, y_min, x_max, y_max = box
                padding = 20
                x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                x_max, y_max = min(frame.shape[1], x_max + padding), min(frame.shape[0], y_max + padding)
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    continue
                results = self.custom_model(roi)
                for r in results:
                    for det_box in r.boxes:
                        rx1, ry1, rx2, ry2 = det_box.xyxy[0].cpu().numpy()
                        conf = float(det_box.conf[0].cpu().numpy())
                        cls_id = int(det_box.cls[0].cpu().numpy())
                        if conf > 0.4:
                            x1, y1 = int(rx1 + x_min), int(ry1 + y_min)
                            x2, y2 = int(rx2 + x_min), int(ry2 + y_min)
                            activities.append((cls_id, conf, [x1, y1, x2, y2]))
        return activities

    def process_frame(self, frame):
        people_boxes = self.detect_persons(frame)
        activities = self.detect_activities(frame, people_boxes)
        return people_boxes, activities


class AlertService:
    def __init__(self):
        self.last_alert_time = 0
        self.alert_cooldown = 30  
        self.twilio_enabled = False #still false and may be opened if we get to stage 4 :)
        self.esp32_enabled = True
        self.esp32_ip = "192.168.100.77"
        
        #Twilio
        self.twilio_sid = "oAC050bc8d5b85372bb9a6178bfd7f24d52"
        self.twilio_token = "3fc8f71e039cfbf04cf0ae68faca5101"
        self.twilio_from = "+19786969043"
        self.twilio_to = "+628112444595"
 
    def send_alert(self, alert_type, details=""):
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False 

        self.last_alert_time = current_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"!!ALERT!!: {alert_type} detected at {timestamp}. {details}"
        
        print(f"🚨 {message}")
        
        if self.esp32_enabled and alert_type in ["violence", "weapon"]:
            self._send_to_esp32(alert_type)
            
        if self.twilio_enabled and alert_type in ["violence", "weapon", "high_sound_violence", "high_sound_suspicious"]:
            self._send_to_twilio(message)
            
        return True

    def _send_to_esp32(self, alert_type):
        try:
            requests.get(f"http://{self.esp32_ip}/alert?type={alert_type}", timeout=0.5)
            print(f"ESP32 alert sent: {alert_type}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to alert ESP32: {e}")

    def _send_to_twilio(self, message):
        try:
            client = Client(self.twilio_sid, self.twilio_token)
            client.messages.create(body=message, from_=self.twilio_from, to=self.twilio_to)
            print(f"Twilio message sent: {message}")
        except Exception as e:
            print(f"Failed to send Twilio message: {e}")


class CombinedSystem:
    def __init__(self, standard_model_path, custom_model_path, lstm_model_path):
        self.detector = IntegratedDetector(standard_model_path, custom_model_path)
        self.person_pose = PersonDetector()
        self.sequence_length = 30
        self.feature_buffer = []
        self.lstm = self._load_lstm_model(lstm_model_path)
        self.lstm_classes = ["non-violence", "violence"]
        self.alert_service = AlertService()
        self.ubidots_service = UbidotsService("BBUS-5QUctLYAhVGEfAQxGrSSM9Zciv4g0m")
        self.alert_history = []
        self.latest_frame = None
        self.cap=cv2.VideoCapture(1)


        
        # Detection history 
        self.detection_history = {
            "violence": 0,
            "suspicious": 0,
            "weapon": 0,
            "normal": 0
        }

    def _load_lstm_model(self, lstm_model_path):
        try:
            model = keras.models.load_model(lstm_model_path)
            return model
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            fallback_model = keras.Sequential([
                layers.Input(shape=(self.sequence_length, 132)),
                layers.LSTM(128),
                layers.Dense(2, activation='softmax')  
            ])
            return fallback_model

    def update_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

       
            result_frame, action_label, activities= self.process_frame(frame)  

            self.latest_frame = result_frame

   

    def start_update_loop(self):
        threading.Thread(target=self.update_loop, daemon=True).start()


    def process_frame(self, frame):
        people_boxes, activities = self.detector.process_frame(frame)
        keypoints = self.person_pose.extract_132d_keypoints(frame)
        if keypoints is not None:
            self.feature_buffer.append(keypoints)
            if len(self.feature_buffer) > self.sequence_length:
                self.feature_buffer.pop(0)

        action_label = "Unknown"
        if len(self.feature_buffer) == self.sequence_length:
            input_tensor = np.array([self.feature_buffer])  
            predictions = self.lstm.predict(input_tensor, verbose=0)
            action_id = np.argmax(predictions[0])
            action_label = self.lstm_classes[action_id]
        
        # Handle alerts based on detections
        self.handle_detections(action_label, activities)
 

        
        result_frame = self.visualize_results(frame, people_boxes, activities, action_label)

        #send to ubi
        self.ubidots_service.send_detection_data(self.detection_history, action_label, activities)

        return result_frame, action_label, activities



    def handle_detections(self, action_label, activities):
        if action_label == "violence":
            self.detection_history["violence"] += 1
            self.alert_service.send_alert("violence", "Violent behavior detected")

            self.alert_history.append({
                "type": "violence",
                "message": "Violent behavior detected",
                "timestamp": datetime.now().isoformat()
            })
        
        # Logic for YOLO
        for cls_id, conf, _ in activities:
            if cls_id == 2:  # Weapon
                self.detection_history["weapon"] += 1 #detection_history= fully counts all activity including the one that isn't in the current frame
                if conf > 0.7:
                    self.alert_service.send_alert("weapon", f"Weapon detected (confidence: {conf:.2f})")
                    self.alert_history.append({
                        "type": "Weapon",
                        "message": "Weapon detected",
                        "timestamp": datetime.now().isoformat()
                    })
            elif cls_id == 1:  # Suspicious
                self.detection_history["suspicious"] += 1
                if conf > 0.7:  
                    self.alert_service.send_alert("suspicious", f"Suspicious activity detected (confidence: {conf:.2f})")
                    self.alert_history.append({
                        "type": "Suspicious",
                        "message": "Suspicious activity detected",
                        "timestamp": datetime.now().isoformat()
                    })
            elif cls_id == 0:  # Normal
                self.detection_history["normal"] += 1

    def visualize_results(self, frame, people_boxes, activities, action_label):
        result = frame.copy()
        #person boxes
        for box in people_boxes:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        #YOLO activity detections
        for cls_id, conf, (x1, y1, x2, y2) in activities:
            label = f"{self.detector.activity_classes[cls_id]}: {conf:.2f}"
            # Red=weapon, yellow=suspicious, blue=normal
            color = (0, 0, 255) if cls_id == 2 else (0, 255, 255) if cls_id == 1 else (255, 0, 0)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        #LSTM prediction
        lstm_color = (0, 0, 255) if action_label == "violence" else (0, 255, 0)
        cv2.putText(result, f"Action: {action_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, lstm_color, 2)
        
        return result

    
class UbidotsService:
    def __init__(self, token):
        self.token = token
        self.base_url = "https://industrial.api.ubidots.com/api/v1.6/devices/"

        self.headers = {
            "X-Auth-Token": token,
            "Content-Type": "application/json"
        }

        self.device_id = "67de277cc6ae7e0b18c2d1a1"  
        self.last_sent_time = 0
        self.send_interval = 3 

    def send_detection_data(self, detection_history, action_label, activities):
        current_time = time.time()
        if current_time - self.last_sent_time < self.send_interval:
            return False
            
        self.last_sent_time = current_time
        
        # Count activities in current frame
        activity_counts = {"normal": 0, "suspicious": 0, "weapon": 0}
        for cls_id, _, _ in activities:
            if cls_id == 0:
                activity_counts["normal"] += 1
            elif cls_id == 1:
                activity_counts["suspicious"] += 1
            elif cls_id == 2:
                activity_counts["weapon"] += 1

        dataubi = {
            "violence_detected": 1 if action_label == "violence" else 0,
            "normal_activities": activity_counts["normal"],
            "suspicious_activities": activity_counts["suspicious"],
            "weapon_detected": activity_counts["weapon"],
            "total_normal": detection_history["normal"],
            "total_suspicious": detection_history["suspicious"], 
            "total_violence": detection_history["violence"],
            "total_weapon": detection_history["weapon"]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}{self.device_id}",
                headers=self.headers,
                data=json.dumps(dataubi),
                timeout=2
            )
            
            if response.status_code == 200:
                print(f"Data sent to Ubidots: {dataubi}")
                return True
            else:
                print(f"Failed to send data to Ubidots. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error sending data to Ubidots: {e}")
            return False


class FlaskServer:
    def __init__(self, system):
        self.app = Flask(__name__)
        CORS(self.app)
        self.system = system
        self.current_action_label = "None" 
        
        @self.app.route('/sound_detected')
        def sound_detected():
            sound_value = request.args.get('sound')
            if sound_value is not None:
                sound_value = int(sound_value)
                print(f"Sound detected from ESP32: {sound_value}")
                
                if sound_value > 1940:  
                    if self.current_action_label == "suspicious":
                        self.system.alert_service.send_alert("high_sound_suspicious", f"High sound level ({sound_value}dB) with suspicious activity")
                    elif self.current_action_label == "violence":
                        self.system.alert_service.send_alert("high_sound_violence", f"High sound level ({sound_value}dB) with violence detected")
            
            return "Received", 200
        
        @self.app.route('/send_alert')
        def send_alert():
            alert_type = request.args.get('type', 'general')
            message = request.args.get('message', '')
            success = self.system.alert_service.send_alert(alert_type, message)
            return jsonify({"success": success}), 200

        @self.app.route("/status", methods=["GET"])
        def status():
            return jsonify({"status": "ok"}), 200

        @self.app.route('/frame')
        def get_frame():
            success, encoded_frame = cv2.imencode('.jpg', system.latest_frame)
            return Response(encoded_frame.tobytes(), mimetype='image/jpeg')

        @self.app.route('/stats')
        def get_stats():
            return jsonify(system.detection_history), 200

        @self.app.route('/alerts')
        def get_alerts():
            return jsonify(system.alert_history), 200
    
    def update_action_label(self, action_label):
        self.current_action_label = action_label

    def run(self):
        threading.Thread(target=lambda: self.app.run(host='0.0.0.0', port=8000, debug=False)).start()

def main():
    standard_model_path = "yolov8n.pt"
    custom_model_path = "best.pt"
    lstm_model_path = "best_lstm_model.h5"
   


    system = CombinedSystem(standard_model_path, custom_model_path, lstm_model_path)
    system.start_update_loop()
        
    server = FlaskServer(system)
    server.run()
        
    cap = cv2.VideoCapture(1)
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
           break
        result_frame, action_label, _ = system.process_frame(frame)
        cv2.imshow("Activity Recognition", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()