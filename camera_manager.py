import cv2
import threading
import time
import json
import os
import datetime
from detector import PhoneDetector
from ultralytics import YOLO

# Force TCP connection (critical for Huawei cameras and general RTSP stability)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class VideoReader:
    """
    Dedicated thread for reading frames from a video source.
    Ensures that we always have the latest frame available, discarding older ones.
    """
    def __init__(self, source, camera_name="Unknown"):
        self.source = source
        self.camera_name = camera_name
        
        self.cap = None
        self.frame = None
        self.last_read_time = 0
        self.running = False
        self.connected = False
        
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        
    def start(self):
        self.running = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

    def update(self):
        print(f"[{self.camera_name}] VideoReader started for source: {self.source}")
        
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.connected = False
                print(f"[{self.camera_name}] Connecting to source...")
                
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None

                self.cap = cv2.VideoCapture(self.source)
                
                is_webcam = isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit())
                
                if is_webcam:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                
                self.fps = 30
                self.is_file = False
                if not is_webcam and isinstance(self.source, str) and not self.source.startswith("rtsp"):
                     if os.path.exists(self.source):
                         self.is_file = True
                         self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                         if self.fps <= 0: 
                             self.fps = 30
                         print(f"[{self.camera_name}] File detected. FPS: {self.fps}")

                if not self.cap.isOpened():
                    print(f"[{self.camera_name}] Connection failed. Retrying in 5s...")
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                    time.sleep(5)
                    continue
                
                print(f"[{self.camera_name}] Connected successfully.")
                self.connected = True

            try:
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
                    self.last_read_time = time.time()
                    
                    if self.is_file:
                        time.sleep(1.0 / self.fps)
                        
                else:
                    print(f"[{self.camera_name}] Stream read failed.")
                    if self.cap:
                        try:
                            self.cap.release()
                        except:
                            pass
                        self.cap = None
                    self.connected = False
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"[{self.camera_name}] Error reading frame: {e}")
                self.connected = False
                if self.cap:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                time.sleep(1)

    def get_frame(self):
        return self.frame, self.last_read_time
        
    def is_connected(self):
        return self.connected and (time.time() - self.last_read_time < 3.0)


class CameraThread(threading.Thread):
    def __init__(self, camera_config, conf_threshold=0.25, 
                 phone_duration=5.0, sleep_duration=10.0, cooldown_duration=120.0,
                 skip_frames=5):
        """
        Camera thread with TIME-BASED detection thresholds.
        
        Args:
            phone_duration: Seconds of continuous phone use before alert
            sleep_duration: Seconds of continuous sleep before alert
            cooldown_duration: Seconds between evidence screenshots
            skip_frames: Process every Nth frame (higher = faster, less accurate)
        """
        super().__init__()
        self.camera_id = camera_config['id']
        self.camera_name = camera_config['name']
        self.source = camera_config['source']
        
        print(f"[{self.camera_name}] Initializing private detector with time-based thresholds...")
        print(f"  → Phone alert after: {phone_duration}s continuous use")
        print(f"  → Sleep alert after: {sleep_duration}s continuous sleep")
        print(f"  → Cooldown period: {cooldown_duration}s")
        
        self.detector = PhoneDetector(
            model_path='yolo26n.pt',
            pose_model_path='yolo26n-pose.pt',
            model_instance=None,
            pose_model_instance=None,
            lock=None,
            phone_duration_threshold=phone_duration,
            sleep_duration_threshold=sleep_duration,
            cooldown_seconds=cooldown_duration
        )
        print(f"[{self.camera_name}] Detector ready.")
        
        self.conf_threshold = conf_threshold
        self.phone_duration = phone_duration
        self.sleep_duration = sleep_duration
        self.cooldown_duration = cooldown_duration
        self.skip_frames = skip_frames
        
        self.running = False
        self.latest_processed_frame = None
        self.status = "safe"
        self.last_update_time = 0
        self.last_processed_timestamp = 0
        
        self.reader = VideoReader(self.source, self.camera_name)
        
    def run(self):
        self.running = True
        print(f"[{self.camera_name}] Starting processing thread...")
        self.reader.start()
        
        frame_count = 0
        
        while self.running:
            if not self.reader.is_connected():
                self.status = "disconnected"
                time.sleep(0.5)
                continue
            
            raw_frame, timestamp = self.reader.get_frame()
            
            if raw_frame is None or timestamp == self.last_processed_timestamp:
                time.sleep(0.01)
                continue
            
            self.last_processed_timestamp = timestamp

            try:
                processed_frame, status, is_saved = self.detector.process_frame(
                    raw_frame, 
                    frame_count, 
                    skip_frames=self.skip_frames,
                    save_screenshots=True,
                    conf_threshold=self.conf_threshold,
                    camera_name=self.camera_name
                )
                
                self.latest_processed_frame = processed_frame
                self.status = status
                self.last_update_time = time.time()
                
            except Exception as e:
                print(f"[{self.camera_name}] Error in processing: {e}")
                import traceback
                traceback.print_exc()
            
            frame_count += 1
            time.sleep(0.01)

        print(f"[{self.camera_name}] Processing thread stopped.")
        self.reader.stop()

    def stop(self):
        self.running = False
        self.join(timeout=2.0)

    def get_frame(self):
        return self.latest_processed_frame

    def get_status(self):
        if time.time() - self.last_update_time > 3.0:
            return "disconnected"
        return self.status
    
    def update_thresholds(self, conf=None, phone_dur=None, sleep_dur=None, cooldown=None, skip_frames=None):
        """Update detection thresholds on the fly."""
        if conf is not None:
            self.conf_threshold = conf
        if phone_dur is not None:
            self.phone_duration = phone_dur
            self.detector.phone_duration_threshold = phone_dur
        if sleep_dur is not None:
            self.sleep_duration = sleep_dur
            self.detector.sleep_duration_threshold = sleep_dur
        if cooldown is not None:
            self.cooldown_duration = cooldown
            self.detector.cooldown_seconds = cooldown
        if skip_frames is not None:
            self.skip_frames = skip_frames


class CameraManager:
    def __init__(self, config_file="cameras.json"):
        """
        Camera Manager with TIME-BASED detection configuration.
        """
        self.config_file = config_file
        self.cameras = {}
        
        # Global thresholds (can be overridden per camera)
        self.global_conf = 0.25
        self.global_phone_duration = 5.0
        self.global_sleep_duration = 10.0
        self.global_cooldown = 120.0
        self.global_skip_frames = 5
        
        print("=" * 70)
        print("Initializing Camera Manager - TIME-BASED DETECTION")
        print("Architecture: Private models per thread")
        print("=" * 70)
        
        self.load_config_and_start()

    def load_config_and_start(self):
        if not os.path.exists(self.config_file):
            default_config = [
                {"id": 0, "name": "Webcam Main", "source": 0}
            ]
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
        
        with open(self.config_file, 'r') as f:
            configs = json.load(f)
            
        for conf in configs:
            self.add_camera_thread(conf)

    def add_camera_thread(self, config):
        """Add a camera thread with current global thresholds."""
        source = config['source']
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        config['source'] = source

        cam_id = config['id']
        if cam_id in self.cameras:
            print(f"Camera {cam_id} already running.")
            return
        
        print(f"\n{'='*60}")
        print(f"Starting Camera {cam_id}: {config['name']}")
        print(f"{'='*60}")
        
        thread = CameraThread(
            config,
            conf_threshold=self.global_conf,
            phone_duration=self.global_phone_duration,
            sleep_duration=self.global_sleep_duration,
            cooldown_duration=self.global_cooldown,
            skip_frames=self.global_skip_frames
        )
        thread.start()
        self.cameras[cam_id] = thread
        
        print(f"[Camera {cam_id}] Started successfully.\n")

    def add_camera(self, name, source):
        existing_ids = [c.camera_id for c in self.cameras.values()]
        new_id = max(existing_ids) + 1 if existing_ids else 0
        
        new_config = {"id": new_id, "name": name, "source": source}
        
        self.save_config_append(new_config)
        self.add_camera_thread(new_config)

    def remove_camera(self, cam_id):
        if cam_id in self.cameras:
            print(f"Removing camera {cam_id}...")
            self.cameras[cam_id].stop()
            del self.cameras[cam_id]
            self.save_config_remove(cam_id)
            print(f"Camera {cam_id} removed.")

    def save_config_append(self, new_config):
        try:
            with open(self.config_file, 'r') as f:
                configs = json.load(f)
            configs.append(new_config)
            with open(self.config_file, 'w') as f:
                json.dump(configs, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def save_config_remove(self, cam_id):
        try:
            with open(self.config_file, 'r') as f:
                configs = json.load(f)
            configs = [c for c in configs if c['id'] != cam_id]
            with open(self.config_file, 'w') as f:
                json.dump(configs, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get_active_cameras(self):
        return self.cameras

    def update_global_conf(self, conf):
        """Update confidence threshold for all cameras."""
        self.global_conf = conf
        for cam in self.cameras.values():
            cam.update_thresholds(conf=conf)
    
    def update_phone_duration(self, duration):
        """Update phone detection duration for all cameras."""
        self.global_phone_duration = duration
        for cam in self.cameras.values():
            cam.update_thresholds(phone_dur=duration)
    
    def update_sleep_duration(self, duration):
        """Update sleep detection duration for all cameras."""
        self.global_sleep_duration = duration
        for cam in self.cameras.values():
            cam.update_thresholds(sleep_dur=duration)
    
    def update_cooldown_duration(self, duration):
        """Update cooldown duration for all cameras."""
        self.global_cooldown = duration
        for cam in self.cameras.values():
            cam.update_thresholds(cooldown=duration)
    
    def update_skip_frames(self, skip_frames):
        """Update skip frames for all cameras."""
        self.global_skip_frames = skip_frames
        for cam in self.cameras.values():
            cam.update_thresholds(skip_frames=skip_frames)
