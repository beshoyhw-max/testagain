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

        # Config
        self.EAR_THRESHOLD = 0.22
        self.SLEEP_TIME_THRESHOLD = 5.0
        self.MIN_FACE_SIZE = 48 # Pixels - Skip MediaPipe if crop is smaller than this
        
        self.HEAD_MOTION_THRESHOLD = 15.0
        self.MOTION_BUFFER_SIZE = 30

        self.state = {}

    def process_crop(self, crop, id_key="unknown", keypoints=None, crop_origin=(0,0)):
        """
        Process person crop for sleep/drowsiness detection.
        
        Uses MediaPipe for eye analysis and YOLO pose for posture analysis.
        """
        if crop.size == 0:
            return "awake", {}

        current_time = time.time()

        if id_key not in self.state:
            self.state[id_key] = {
                'closed_start': None,
                'status': 'awake',
                'last_seen': current_time,
                'head_positions': [],
                'last_active_time': current_time,
                'ear_history': []
            }

        state = self.state[id_key]
        state['last_seen'] = current_time

        # --- STEP 1: EYES CHECK (MediaPipe) ---
        # Resolution Gate
        run_mediapipe = True
        if self.detector is None:
            run_mediapipe = False
        elif crop.shape[0] < self.MIN_FACE_SIZE or crop.shape[1] < self.MIN_FACE_SIZE:
            run_mediapipe = False
            
        if run_mediapipe:
            try:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                detection_result = self.detector.detect(mp_image)

                if detection_result.face_landmarks:
                    landmarks = detection_result.face_landmarks[0]
                    nose_landmark = landmarks[1]
                    nose_position = (nose_landmark.x * crop.shape[1], 
                                   nose_landmark.y * crop.shape[0])

                    state['head_positions'].append(nose_position)
                    if len(state['head_positions']) > self.MOTION_BUFFER_SIZE:
                        state['head_positions'].pop(0)

                    left_ear = self._calculate_ear(landmarks, [33, 160, 158, 133, 153, 144])
                    right_ear = self._calculate_ear(landmarks, [362, 385, 387, 263, 373, 380])
                    avg_ear = (left_ear + right_ear) / 2.0

                    if 'ear_history' not in state: 
                        state['ear_history'] = []
                    state['ear_history'].append(avg_ear)
                    if len(state['ear_history']) > 300:
                        state['ear_history'].pop(0)

                    current_threshold = self.EAR_THRESHOLD
                    if len(state['ear_history']) > 10:
                        baseline_ear = np.percentile(state['ear_history'], 90)
                        dynamic = baseline_ear * 0.8
                        current_threshold = min(self.EAR_THRESHOLD, max(0.12, dynamic))

                    is_head_still = self._is_head_still(state['head_positions'])

                    if avg_ear < current_threshold:
                        if state['closed_start'] is None:
                            state['closed_start'] = current_time

                        duration = current_time - state['closed_start']
                        if duration > self.SLEEP_TIME_THRESHOLD:
                            return "sleeping", {
                                "ear": avg_ear, 
                                "duration": duration, 
                                "still": is_head_still, 
                                "source": "mediapipe"
                            }
                        else:
                            return "drowsy", {
                                "ear": avg_ear, 
                                "duration": duration, 
                                "still": is_head_still, 
                                "source": "mediapipe"
                            }
                    else:
                        state['closed_start'] = None
                        state['last_active_time'] = current_time
                        return "awake", {
                            "ear": avg_ear, 
                            "still": is_head_still, 
                            "source": "mediapipe"
                        }
            except Exception as e:
                # Silently fall through to posture check
                pass

        # --- STEP 2: ENHANCED POSTURE CHECK (YOLO) ---
        kpts = None
        if keypoints is not None:
            cx, cy = crop_origin
            kpts = keypoints.copy()
            kpts[:, 0] -= cx
            kpts[:, 1] -= cy
        else:
            # Fallback - run pose on crop (less efficient)
            # IMPROVED: No lock needed with private pose model
            pose_results = self.pose_model.predict(crop, verbose=False, conf=0.5)
            if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                keypoints_data = pose_results[0].keypoints.xy.cpu().numpy()
                if len(keypoints_data) > 0:
                    kpts = keypoints_data[0]

        if kpts is not None:
            posture_result = self._check_sleep_posture(kpts, crop.shape)
            if posture_result['is_sleeping']:
                return "sleeping", {
                    "reason": posture_result['reason'], 
                    "source": "yolo-pose", 
                    "details": posture_result
                }
            if posture_result['is_writing']:
                state['last_active_time'] = current_time
                return "awake", {
                    "reason": "writing_detected", 
                    "source": "yolo-pose", 
                    "details": posture_result
                }

        return "awake", {"reason": "no_face_no_posture", "source": "fallback"}

    def _is_head_still(self, positions):
        """Check if head movement is minimal (indicator of sleep)."""
        if len(positions) < 10: 
            return False
        positions_array = np.array(positions)
        x_std = np.std(positions_array[:, 0])
        y_std = np.std(positions_array[:, 1])
        return (x_std + y_std) < self.HEAD_MOTION_THRESHOLD

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

        # IMPROVED: Face definition
        has_nose = has_pt(0)
        has_eyes = has_pt(1) or has_pt(2)

        # 1. Head Buried / High Shoulders
        # Fixed: allow check even if face points are noisy, rely on shoulder position
        if has_shoulders:
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            shoulder_width = abs(kpts[5][0] - kpts[6][0])

            # Threshold increased to 0.32 to catch people slumped on desk (shoulders high)
            if shoulder_y < crop_height * 0.32:
                # Safeguard: If face is clearly upright, do NOT classify as buried
                # "Upright" means nose is significantly above shoulders
                is_clearly_upright = False
                if has_nose:
                    nose_y = kpts[0][1]
                    # If nose is high above shoulders (large negative difference)
                    # For ID 2 (Awake): Diff is -72px (approx -0.5 * shoulder_width)
                    # For ID 1 (Sleep): Diff is -22px (approx -0.2 * shoulder_width)
                    if shoulder_width > 0 and (nose_y - shoulder_y) < (-0.4 * shoulder_width):
                        is_clearly_upright = True

                if not is_clearly_upright:
                    result['is_sleeping'] = True
                    result['reason'] = "head_buried_high_shoulders"
                    result['details']['shoulder_height_ratio'] = shoulder_y / crop_height
                    return result
        
        # 2. Slumped Forward (Nose below shoulders)
        if has_pt(0) and has_shoulders:
            nose_y = kpts[0][1]
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2

            # Use relative threshold instead of 30px
            threshold = 30
            if crop_height > 100:
                threshold = crop_height * 0.1

            if nose_y > shoulder_y + threshold:
                result['is_sleeping'] = True
                result['reason'] = "slumped_forward"
                result['details']['slump_distance'] = nose_y - shoulder_y
                return result
        
        # 3. Active Check (Hands)
        has_wrists = has_pt(9) or has_pt(10)
        if has_shoulders and has_wrists:
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            shoulder_width = abs(kpts[5][0] - kpts[6][0])
            wrist_y = 0
            if has_pt(9) and has_pt(10): 
                wrist_y = max(kpts[9][1], kpts[10][1])
            elif has_pt(9): 
                wrist_y = kpts[9][1]
            else: 
                wrist_y = kpts[10][1]
            
            # Use relative threshold
            thresh_wrist = 80
            if shoulder_width > 0:
                thresh_wrist = shoulder_width * 0.8

            if wrist_y > shoulder_y + thresh_wrist:
                result['is_writing'] = True
                result['reason'] = "hands_on_desk"
                result['details']['hand_position'] = "active"
                if has_pt(0):
                    nose_y = kpts[0][1]
                    if nose_y > shoulder_y + 10 and nose_y < shoulder_y + 50: 
                        return result
        
        # 4. Head Tilt
        if has_pt(1) and has_pt(2):
            tilt = abs(kpts[1][1] - kpts[2][1])
            if tilt > 40: # This is still absolute, but tilt is angle-proxy
                result['is_sleeping'] = True
                result['reason'] = "head_tilted"
                result['details']['tilt_amount'] = tilt
                return result
        
        # 5. Collapsed Posture (Background filtering)
        if has_shoulders and (has_pt(11) or has_pt(12)):
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            shoulder_width = abs(kpts[5][0] - kpts[6][0])

            hip_y = 0
            if has_pt(11) and has_pt(12): 
                hip_y = (kpts[11][1] + kpts[12][1]) / 2
            elif has_pt(11): 
                hip_y = kpts[11][1]
            else: 
                hip_y = kpts[12][1]
            torso_length = abs(hip_y - shoulder_y)

            # FIXED: Use relative ratio check instead of absolute pixels
            # ID 4 (Background): Ratio ~3.35 (Safe)
            # Collapsed/Hunched: Ratio < 0.8 (Trigger)
            if shoulder_width > 0:
                ratio = torso_length / shoulder_width
                if ratio < 0.8:
                    result['is_sleeping'] = True
                    result['reason'] = "collapsed_posture"
                    result['details']['torso_ratio'] = ratio
                    return result
            # Fallback for very small crops if shoulder width is weird
            elif crop_height > 100 and torso_length < 60:
                 # Only use pixel check for decent sized crops
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
