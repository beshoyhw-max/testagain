import time
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import os

class SleepDetector:
    def __init__(self, pose_model_path='yolo26n-pose.pt', 
                 mp_model_path='models/face_landmarker.task', 
                 pose_model_instance=None):
        """
        Initialize Sleep Detector.
        
        IMPROVED: Now supports both shared and private pose models.
        - If pose_model_instance is None: loads its own private model (RECOMMENDED)
        - If pose_model_instance is provided: uses shared model (legacy)
        
        Private models provide better thread isolation.
        """
        
        # 1. Load YOLO Pose
        if pose_model_instance:
            # Legacy: Use shared pose model
            print("    → Using shared pose model")
            self.pose_model = pose_model_instance
        else:
            # IMPROVED: Load private pose model
            print(f"    → Loading private pose model from {pose_model_path}")
            self.pose_model = YOLO(pose_model_path)
            print("    → Private pose model loaded")

        # 2. Load MediaPipe Face Landmarker
        print(f"    → Loading MediaPipe Face Landmarker from {mp_model_path}")
        if not os.path.exists(mp_model_path):
            print(f"    → Warning: {mp_model_path} not found. Face detection disabled.")
            self.detector = None
        else:
            try:
                base_options = python.BaseOptions(model_asset_path=mp_model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1
                )
                self.detector = vision.FaceLandmarker.create_from_options(options)
                print("    → MediaPipe Face Landmarker loaded")
            except Exception as e:
                print(f"    → Error loading MediaPipe: {e}")
                self.detector = None

        # === RESEARCH-BACKED CONFIGURATION ===
        
        # EAR (Eye Aspect Ratio) Settings
        self.EAR_THRESHOLD = 0.22           # Below this = eyes closed
        self.EAR_CONSEC_FRAMES = 3          # Min consecutive frames for "closed"
        
        # PERCLOS Settings (Percentage of Eye Closure)
        self.PERCLOS_WINDOW = 30            # Frames to calculate PERCLOS (~1 sec at 30fps)
        self.PERCLOS_DROWSY = 0.40          # 40% eyes closed = drowsy
        self.PERCLOS_SLEEP = 0.70           # 70% eyes closed = likely sleeping
        
        # Scale-Invariant Posture Thresholds (ratios of shoulder_width)
        self.HEAD_TILT_RATIO = 0.15         # 15% of shoulder width = tilted
        self.HEAD_DROP_RATIO = 0.50         # 50% of shoulder width = slumped
        self.TORSO_COLLAPSE_RATIO = 0.40    # Torso < 40% of shoulder width = collapsed
        
        # Head Motion
        self.HEAD_MOTION_THRESHOLD = 0.05   # Normalized std deviation
        self.MOTION_BUFFER_SIZE = 30        # Frames to track motion
        
        # Multi-Signal Scoring Weights (sum to 1.0)
        self.SCORE_WEIGHTS = {
            'perclos_high': 0.30,       # PERCLOS > 40%
            'eyes_closed': 0.25,        # Current EAR < threshold
            'head_still': 0.15,         # Low head motion
            'head_dropped': 0.15,       # Pitch indicates head down
            'head_tilted': 0.10,        # Roll indicates tilt
            'sleep_posture': 0.05,      # Shoulders visible, no face
        }
        
        # Scoring Thresholds
        self.SCORE_DROWSY = 0.35            # Score >= 0.35 = drowsy
        self.SCORE_SLEEPING = 0.60          # Score >= 0.60 = sleeping
        
        # Temporal Smoothing
        self.SMOOTHING_WINDOW = 10          # Frames for majority voting
        
        # Resolution Gate
        self.MIN_FACE_SIZE = 48             # Skip MediaPipe if crop smaller
        
        # Per-person state tracking
        self.state = {}

    def process_crop(self, crop, id_key="unknown", keypoints=None, crop_origin=(0,0)):
        """
        Process person crop for sleep/drowsiness detection.
        
        Uses multi-signal fusion combining:
        - PERCLOS (eye closure percentage over time)
        - Head stillness
        - Posture analysis
        - Temporal smoothing
        
        Returns: (status, details) where status is 'awake', 'drowsy', or 'sleeping'
        """
        if crop.size == 0:
            return "awake", {"score": 0.0, "reason": "empty_crop"}

        current_time = time.time()

        # Initialize state for new person
        if id_key not in self.state:
            self.state[id_key] = {
                'last_seen': current_time,
                'head_positions': [],
                'ear_history': [],           # For dynamic threshold
                'ear_closed_history': [],    # Boolean: was eye closed? (for PERCLOS)
                'recent_states': [],         # For temporal smoothing
                'last_active_time': current_time,
            }

        state = self.state[id_key]
        state['last_seen'] = current_time

        # === COLLECT SIGNALS ===
        signals = {
            'perclos_high': False,
            'eyes_closed': False,
            'head_still': False,
            'head_dropped': False,
            'head_tilted': False,
            'sleep_posture': False,
        }
        
        details = {
            'score': 0.0,
            'signals': {},
            'source': 'none',
        }

        # --- SIGNAL 1 & 2: Eye Analysis (MediaPipe) ---
        avg_ear = None
        current_threshold = self.EAR_THRESHOLD
        is_head_still = False
        face_detected = False
        
        run_mediapipe = (
            self.detector is not None and 
            crop.shape[0] >= self.MIN_FACE_SIZE and 
            crop.shape[1] >= self.MIN_FACE_SIZE
        )
            
        if run_mediapipe:
            try:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                detection_result = self.detector.detect(mp_image)

                if detection_result.face_landmarks:
                    face_detected = True
                    landmarks = detection_result.face_landmarks[0]
                    
                    # Track head position for stillness
                    nose_landmark = landmarks[1]
                    nose_position = (
                        nose_landmark.x * crop.shape[1], 
                        nose_landmark.y * crop.shape[0]
                    )
                    state['head_positions'].append(nose_position)
                    if len(state['head_positions']) > self.MOTION_BUFFER_SIZE:
                        state['head_positions'].pop(0)

                    # Calculate EAR
                    left_ear = self._calculate_ear(landmarks, [33, 160, 158, 133, 153, 144])
                    right_ear = self._calculate_ear(landmarks, [362, 385, 387, 263, 373, 380])
                    avg_ear = (left_ear + right_ear) / 2.0

                    # Dynamic threshold based on person's baseline
                    state['ear_history'].append(avg_ear)
                    if len(state['ear_history']) > 300:
                        state['ear_history'].pop(0)

                    if len(state['ear_history']) > 10:
                        baseline_ear = np.percentile(state['ear_history'], 90)
                        dynamic = baseline_ear * 0.75  # 75% of open-eye baseline
                        current_threshold = min(self.EAR_THRESHOLD, max(0.12, dynamic))

                    # Is eye currently closed?
                    eye_closed = avg_ear < current_threshold
                    signals['eyes_closed'] = eye_closed
                    
                    # Track for PERCLOS
                    state['ear_closed_history'].append(eye_closed)
                    if len(state['ear_closed_history']) > self.PERCLOS_WINDOW:
                        state['ear_closed_history'].pop(0)
                    
                    # Calculate PERCLOS
                    if len(state['ear_closed_history']) >= 10:  # Need minimum samples
                        closed_count = sum(state['ear_closed_history'])
                        perclos = closed_count / len(state['ear_closed_history'])
                        signals['perclos_high'] = perclos >= self.PERCLOS_DROWSY
                        details['perclos'] = perclos
                    
                    # Head stillness (normalized)
                    is_head_still = self._is_head_still_normalized(state['head_positions'], crop.shape)
                    signals['head_still'] = is_head_still
                    
                    details['ear'] = avg_ear
                    details['threshold'] = current_threshold
                    details['source'] = 'mediapipe'
                    
            except Exception as e:
                pass  # Fall through to posture check

        # --- SIGNAL 3, 4, 5: Posture Analysis (YOLO Pose) ---
        kpts = None
        shoulder_width = None
        
        if keypoints is not None:
            cx, cy = crop_origin
            kpts = keypoints.copy()
            kpts[:, 0] -= cx
            kpts[:, 1] -= cy
        else:
            # Fallback - run pose on crop
            pose_results = self.pose_model.predict(crop, verbose=False, conf=0.5)
            if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                keypoints_data = pose_results[0].keypoints.xy.cpu().numpy()
                if len(keypoints_data) > 0:
                    kpts = keypoints_data[0]

        if kpts is not None:
            posture_signals = self._check_sleep_posture_scaled(kpts, crop.shape)
            signals['head_dropped'] = posture_signals.get('head_dropped', False)
            signals['head_tilted'] = posture_signals.get('head_tilted', False)
            signals['sleep_posture'] = posture_signals.get('sleep_posture', False)
            
            # Override to awake if writing detected
            if posture_signals.get('is_writing', False):
                state['last_active_time'] = current_time
                state['recent_states'].append('awake')
                if len(state['recent_states']) > self.SMOOTHING_WINDOW:
                    state['recent_states'].pop(0)
                return "awake", {"score": 0.0, "reason": "writing_detected", "source": "yolo-pose"}
            
            details['posture'] = posture_signals
            if details['source'] == 'none':
                details['source'] = 'yolo-pose'

        # === COMPUTE WEIGHTED SCORE ===
        score = 0.0
        active_signals = []
        for signal_name, is_active in signals.items():
            if is_active and signal_name in self.SCORE_WEIGHTS:
                score += self.SCORE_WEIGHTS[signal_name]
                active_signals.append(signal_name)
        
        details['score'] = score
        details['signals'] = signals
        details['active_signals'] = active_signals

        # === DETERMINE RAW STATE ===
        if score >= self.SCORE_SLEEPING:
            raw_state = 'sleeping'
        elif score >= self.SCORE_DROWSY:
            raw_state = 'drowsy'
        else:
            raw_state = 'awake'
            state['last_active_time'] = current_time

        # === TEMPORAL SMOOTHING ===
        state['recent_states'].append(raw_state)
        if len(state['recent_states']) > self.SMOOTHING_WINDOW:
            state['recent_states'].pop(0)
        
        # Majority voting for smoothed state
        if len(state['recent_states']) >= 5:
            from collections import Counter
            state_counts = Counter(state['recent_states'])
            smoothed_state = state_counts.most_common(1)[0][0]
        else:
            smoothed_state = raw_state
        
        details['raw_state'] = raw_state
        details['smoothed_state'] = smoothed_state

        return smoothed_state, details

    def _is_head_still(self, positions):
        """Check if head movement is minimal (indicator of sleep) - legacy method."""
        if len(positions) < 10: 
            return False
        positions_array = np.array(positions)
        x_std = np.std(positions_array[:, 0])
        y_std = np.std(positions_array[:, 1])
        return (x_std + y_std) < 15.0  # Pixel-based threshold
    
    def _is_head_still_normalized(self, positions, crop_shape):
        """
        Check if head movement is minimal, normalized by crop size.
        This makes the threshold work consistently regardless of camera distance.
        """
        if len(positions) < 10: 
            return False
        
        positions_array = np.array(positions)
        x_std = np.std(positions_array[:, 0])
        y_std = np.std(positions_array[:, 1])
        
        # Normalize by crop dimensions
        crop_size = max(crop_shape[0], crop_shape[1])
        if crop_size == 0:
            return False
            
        normalized_motion = (x_std + y_std) / crop_size
        return normalized_motion < self.HEAD_MOTION_THRESHOLD
    
    def _check_sleep_posture_scaled(self, kpts, crop_shape):
        """
        Scale-invariant posture check using ratios relative to shoulder_width.
        
        Returns dict with boolean flags for each posture signal:
        - head_dropped: nose significantly below shoulders
        - head_tilted: eyes at different heights
        - sleep_posture: shoulders visible but face not
        - is_writing: hands positioned for writing/reading
        """
        def has_pt(idx): 
            return kpts[idx][0] > 0 and kpts[idx][1] > 0
        
        result = {
            'head_dropped': False, 
            'head_tilted': False, 
            'sleep_posture': False,
            'is_writing': False,
            'details': {}
        }
        
        # Calculate reference measurements
        has_shoulders = has_pt(5) and has_pt(6)
        if not has_shoulders:
            return result  # Can't do scale-invariant checks without shoulders
            
        shoulder_width = abs(kpts[6][0] - kpts[5][0])
        if shoulder_width < 10:  # Too small to be reliable
            return result
            
        shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
        result['details']['shoulder_width'] = shoulder_width
        
        # Check if face is visible
        has_face = has_pt(0) or has_pt(1) or has_pt(2) or has_pt(3) or has_pt(4)
        
        # === SLEEP POSTURE: Shoulders visible but no face ===
        if has_shoulders and not has_face:
            # Additional check: shoulders should be near top of crop (person leaning forward)
            if shoulder_y < crop_shape[0] * 0.35:
                result['sleep_posture'] = True
                result['details']['reason'] = 'head_buried'
        
        # === HEAD DROPPED: Nose significantly below shoulder line ===
        if has_pt(0) and has_shoulders:
            nose_y = kpts[0][1]
            drop_distance = nose_y - shoulder_y
            normalized_drop = drop_distance / shoulder_width
            
            if normalized_drop > self.HEAD_DROP_RATIO:
                result['head_dropped'] = True
                result['details']['drop_ratio'] = normalized_drop
        
        # === HEAD TILTED: Eye heights significantly different ===
        if has_pt(1) and has_pt(2):
            eye_tilt = abs(kpts[1][1] - kpts[2][1])
            normalized_tilt = eye_tilt / shoulder_width
            
            if normalized_tilt > self.HEAD_TILT_RATIO:
                result['head_tilted'] = True
                result['details']['tilt_ratio'] = normalized_tilt
        
        # === WRITING DETECTION: Hands below shoulders, head slightly forward ===
        has_wrists = has_pt(9) or has_pt(10)
        if has_shoulders and has_wrists:
            wrist_y = 0
            if has_pt(9) and has_pt(10): 
                wrist_y = max(kpts[9][1], kpts[10][1])
            elif has_pt(9): 
                wrist_y = kpts[9][1]
            else: 
                wrist_y = kpts[10][1]
            
            wrist_drop = wrist_y - shoulder_y
            normalized_wrist = wrist_drop / shoulder_width
            
            # Hands significantly below shoulders = working
            if normalized_wrist > 0.5:
                # Additional check: if nose is visible and only slightly forward, it's writing
                if has_pt(0):
                    nose_y = kpts[0][1]
                    nose_drop = (nose_y - shoulder_y) / shoulder_width
                    if 0.1 < nose_drop < 0.4:  # Slight forward lean, not collapsed
                        result['is_writing'] = True
                        result['details']['activity'] = 'writing_or_reading'
        
        return result

    def _check_sleep_posture(self, kpts, crop_shape):
        """
        Check for sleep-indicative postures using pose keypoints.
        
        Patterns detected:
        - Head buried (shoulders visible, face not)
        - Slumped forward
        - Head tilted significantly
        - Collapsed posture
        
        Also detects active postures (writing/reading).
        """
        def has_pt(idx): 
            return kpts[idx][0] > 0 and kpts[idx][1] > 0
        
        result = {'is_sleeping': False, 'is_writing': False, 'reason': None, 'details': {}}
        crop_height = crop_shape[0]
        
        has_shoulders = has_pt(5) and has_pt(6)
        has_face = has_pt(0) or has_pt(1) or has_pt(2) or has_pt(3) or has_pt(4)

        if has_shoulders and not has_face:
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            if shoulder_y < crop_height * 0.25:
                result['is_sleeping'] = True
                result['reason'] = "head_buried_high_shoulders"
                result['details']['shoulder_height_ratio'] = shoulder_y / crop_height
                return result
            else:
                return result
        
        if has_pt(0) and has_shoulders:
            nose_y = kpts[0][1]
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            if nose_y > shoulder_y + 30:
                result['is_sleeping'] = True
                result['reason'] = "slumped_forward"
                result['details']['slump_distance'] = nose_y - shoulder_y
                return result
        
        has_wrists = has_pt(9) or has_pt(10)
        if has_shoulders and has_wrists:
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            wrist_y = 0
            if has_pt(9) and has_pt(10): 
                wrist_y = max(kpts[9][1], kpts[10][1])
            elif has_pt(9): 
                wrist_y = kpts[9][1]
            else: 
                wrist_y = kpts[10][1]
            
            if wrist_y > shoulder_y + 80:
                result['is_writing'] = True
                result['reason'] = "hands_on_desk"
                result['details']['hand_position'] = "active"
                if has_pt(0):
                    nose_y = kpts[0][1]
                    if nose_y > shoulder_y + 10 and nose_y < shoulder_y + 50: 
                        return result
        
        if has_pt(1) and has_pt(2):
            tilt = abs(kpts[1][1] - kpts[2][1])
            if tilt > 40:
                result['is_sleeping'] = True
                result['reason'] = "head_tilted"
                result['details']['tilt_amount'] = tilt
                return result
        
        if has_shoulders and (has_pt(11) or has_pt(12)):
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            hip_y = 0
            if has_pt(11) and has_pt(12): 
                hip_y = (kpts[11][1] + kpts[12][1]) / 2
            elif has_pt(11): 
                hip_y = kpts[11][1]
            else: 
                hip_y = kpts[12][1]
            torso_length = abs(hip_y - shoulder_y)
            if torso_length < 60:
                result['is_sleeping'] = True
                result['reason'] = "collapsed_posture"
                result['details']['torso_length'] = torso_length
                return result
        
        if has_pt(0) and has_shoulders and (has_pt(7) or has_pt(8)):
            nose_y = kpts[0][1]
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            if nose_y > shoulder_y + 20 and nose_y < shoulder_y + 60:
                elbow_y = 0
                if has_pt(7) and has_pt(8): 
                    elbow_y = (kpts[7][1] + kpts[8][1]) / 2
                elif has_pt(7): 
                    elbow_y = kpts[7][1]
                else: 
                    elbow_y = kpts[8][1]
                if elbow_y > shoulder_y and elbow_y < shoulder_y + 100:
                    result['is_writing'] = True
                    result['reason'] = "reading_posture"
                    return result
        
        return result

    def _calculate_ear(self, landmarks, indices):
        """
        Calculate Eye Aspect Ratio (EAR) for drowsiness detection.
        Lower values indicate closed eyes.
        """
        def dist(i1, i2):
            p1 = landmarks[i1]
            p2 = landmarks[i2]
            return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
        vertical_1 = dist(indices[1], indices[5])
        vertical_2 = dist(indices[2], indices[4])
        horizontal = dist(indices[0], indices[3])
        if horizontal == 0: 
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def close(self):
        """Clean up MediaPipe resources."""
        if self.detector: 
            self.detector.close()
