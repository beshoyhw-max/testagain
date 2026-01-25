import cv2
import time
import os
import math
import datetime
from ultralytics import YOLO
import threading
from sleep_detector import SleepDetector

class PhoneDetector:
    def __init__(self, model_path='yolo26n.pt', pose_model_path='yolo26n-pose.pt', 
                 output_dir="detections", 
                 phone_duration_threshold=5.0,
                 sleep_duration_threshold=10.0,
                 cooldown_seconds=120.0,
                 reset_gap=5.0,
                 model_instance=None, pose_model_instance=None, lock=None,
                 enable_face_recognition=False):
        """
        Initialize Phone Detector with FIXED face recognition integration.
        """
        self.face_rec_debug_counter = 0
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Load models
        self.lock = lock
        
        if model_instance:
            print("  → Using shared detection model (with lock)")
            self.model = model_instance
        else:
            print(f"  → Loading private detection model from {model_path}")
            self.model = YOLO(model_path)
            print("  → Private detection model loaded")
            
        print("  → Initializing sleep detector...")
        if pose_model_instance:
            self.sleep_detector = SleepDetector(pose_model_instance=pose_model_instance)
        else:
            self.sleep_detector = SleepDetector(pose_model_path=pose_model_path)
        print("  → Sleep detector initialized")

        # Initialize face recognition if enabled
        self.face_recognizer = None
        self.enable_face_recognition = enable_face_recognition
        
        if enable_face_recognition:
            try:
                from face_recognizer import FaceRecognizer
                self.face_recognizer = FaceRecognizer()
                num_people = len(self.face_recognizer.list_known_people())
                print(f"  → Face recognition ENABLED")
                print(f"     • Registered people: {num_people}")
                if num_people > 0:
                    print(f"     • Names: {', '.join(self.face_recognizer.list_known_people())}")
            except Exception as e:
                print(f"  → Face recognition initialization failed: {e}")
                print(f"  → Continuing without face recognition")
                self.face_recognizer = None

        self.PHONE_CLASS_ID = 67
        self.PERSON_CLASS_ID = 0
        
        # TIME-BASED THRESHOLDS
        self.phone_duration_threshold = phone_duration_threshold
        self.sleep_duration_threshold = sleep_duration_threshold
        self.cooldown_seconds = cooldown_seconds
        self.reset_gap = reset_gap
        
        # Tracking state
        self.violation_timers = {}
        self.cooldowns = {}
        self.last_display_data = []
        
        # FIXED: Person identification with better caching
        self.person_identities = {}  # track_id -> name
        self.identity_confidences = {}  # track_id -> confidence
        self.identity_last_check = {}  # track_id -> timestamp
        self.identity_check_interval = 1.0  # Check every 1 second (was 2.0)
        self.identity_min_confidence = 0.40  # Minimum confidence to accept (lowered from implicit 0.363)
        
        print(f"  → Time-based thresholds configured:")
        print(f"     • Phone alert after: {phone_duration_threshold}s")
        print(f"     • Sleep alert after: {sleep_duration_threshold}s")
        print(f"     • Reset gap: {reset_gap}s")
        print(f"     • Face recognition check interval: {self.identity_check_interval}s")

    def process_frame(self, frame, frame_count, skip_frames=5, save_screenshots=True, 
                     conf_threshold=0.25, camera_name="Unknown", enable_sleep_detection=True):
        """
        Process frame with FIXED face recognition display.
        """
        current_time = time.time()
        
        # Periodic cleanup
        if frame_count % 1000 == 0:
            cutoff = current_time - (self.cooldown_seconds * 2)
            self.cooldowns = {k: v for k, v in self.cooldowns.items() if v > cutoff}
            
            stale_cutoff = current_time - 10.0
            self.violation_timers = {
                k: v for k, v in self.violation_timers.items() 
                if v['last_seen'] > stale_cutoff
            }
            
            # Clean stale identities (persons not seen in 15 seconds)
            identity_cutoff = current_time - 15.0
            active_track_ids = set(t[0] for t in self.violation_timers.keys())
            self.person_identities = {
                k: v for k, v in self.person_identities.items()
                if (k in active_track_ids or 
                    self.identity_last_check.get(k, 0) > identity_cutoff)
            }
            self.identity_confidences = {
                k: v for k, v in self.identity_confidences.items()
                if k in self.person_identities
            }

        global_status = "safe"
        screenshot_saved_global = False

        # --- INFERENCE STEP ---
        if frame_count % skip_frames == 0:
            self.last_display_data = []
            
            # 1. Detection + Tracking
            classes_to_track = [self.PERSON_CLASS_ID, self.PHONE_CLASS_ID]
            
            if self.lock:
                with self.lock:
                    results = self.model.track(frame, classes=classes_to_track, 
                                             conf=conf_threshold, persist=True, 
                                             verbose=False, imgsz=1280)
            else:
                results = self.model.track(frame, classes=classes_to_track, 
                                         conf=conf_threshold, persist=True, 
                                         verbose=False, imgsz=1280)
            
            person_boxes = []
            phone_boxes = []

            if len(results) > 0 and results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    coords = box.xyxy[0].cpu().numpy()

                    if cls_id == self.PERSON_CLASS_ID:
                        if box.id is not None:
                            track_id = int(box.id.item())
                            person_boxes.append((*coords, track_id))
                    elif cls_id == self.PHONE_CLASS_ID:
                        conf = float(box.conf[0].item())
                        phone_boxes.append((*coords, conf))

            # 2. Pose Estimation
            pose_keypoints_map = {}
            if hasattr(self.sleep_detector, 'pose_model'):
                try:
                    pose_results = self.sleep_detector.pose_model(frame, verbose=False, conf=0.5)
                    
                    if len(pose_results) > 0 and pose_results[0].boxes:
                        pose_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                        pose_kpts = pose_results[0].keypoints.xy.cpu().numpy()
                        pose_keypoints_map = self._associate_pose_to_persons(
                            person_boxes, pose_boxes, pose_kpts
                        )
                except Exception as e:
                    print(f"Pose Inference Error: {e}")

            # 3. Associate phones to persons
            phone_map = self._associate_phones_to_persons(
                person_boxes, phone_boxes, pose_keypoints_map
            )

            # 4. Face Recognition with VERBOSE DEBUGGING
            if self.face_recognizer:
                for p_box in person_boxes:
                    x1, y1, x2, y2, track_id = p_box
                    
                    last_check = self.identity_last_check.get(track_id, 0)
                    time_since_check = current_time - last_check
                    
                    should_check = (track_id not in self.person_identities or 
                                   time_since_check > self.identity_check_interval)
                    
                    if should_check:
                        self.face_rec_debug_counter += 1
                        
                        try:
                            # Calculate bbox info
                            h, w = frame.shape[:2]
                            person_width = x2 - x1
                            person_height = y2 - y1
                            
                            # Use padding
                            pad = 20
                            fx1 = max(0, int(x1 - pad))
                            fy1 = max(0, int(y1 - pad))
                            fx2 = min(w, int(x2 + pad))
                            fy2 = min(h, int(y2 + pad))
                            
                            roi = frame[fy1:fy2, fx1:fx2]
                            roi_h, roi_w = roi.shape[:2]
                            
                            # DEBUG: Print bbox details every 30 checks
                            if self.face_rec_debug_counter % 30 == 1:
                                print(f"\n  🔍 FACE REC DEBUG - ID {track_id}:")
                                print(f"     Frame size: {w}x{h}")
                                print(f"     Person bbox: ({x1},{y1}) to ({x2},{y2}) = {person_width}x{person_height}px")
                                print(f"     ROI (with pad): ({fx1},{fy1}) to ({fx2},{fy2}) = {roi_w}x{roi_h}px")
                            
                            # Try recognition
                            person_name, confidence = self.face_recognizer.recognize_face(
                                frame, (fx1, fy1, fx2, fy2)
                            )
                            
                            self.identity_last_check[track_id] = current_time
                            
                            # DEBUG: Always print results
                            if self.face_rec_debug_counter % 30 == 1:
                                print(f"     Result: {person_name} (confidence={confidence:.3f})")
                                
                                # Save ROI for inspection
                                if confidence == 0.0:
                                    debug_path = f"debug_roi_id{track_id}_frame{frame_count}.jpg"
                                    cv2.imwrite(debug_path, roi)
                                    print(f"     💾 Saved ROI to {debug_path}")
                                    print(f"     → Register using this image!")
                            
                            # Handle recognition results
                            if person_name != "Unknown" and confidence >= self.identity_min_confidence:
                                old_name = self.person_identities.get(track_id, "Unknown")
                                self.person_identities[track_id] = person_name
                                self.identity_confidences[track_id] = confidence
                                
                                if old_name != person_name:
                                    print(f"  ✓✓✓ ID {track_id}: RECOGNIZED as {person_name} (conf={confidence:.3f})")
                            
                            elif confidence > 0.0:
                                # Face detected but not matched
                                if self.face_rec_debug_counter % 30 == 1:
                                    print(f"     Face detected but score below threshold")
                                    print(f"     Best score: {confidence:.3f}, Threshold: {self.identity_min_confidence:.3f}")
                            
                            else:
                                # No face detected at all
                                if self.face_rec_debug_counter % 30 == 1:
                                    print(f"     ⚠ No face detected in ROI")
                                    print(f"     This means YuNet couldn't find a face in the {roi_w}x{roi_h}px region")
                                        
                        except Exception as e:
                            print(f"  ❌ Face recognition exception for ID {track_id}: {e}")
                            import traceback
                            traceback.print_exc()

            # 5. Process each person with TIME-BASED logic
            current_detections = set()
            
            for p_box in person_boxes:
                x1, y1, x2, y2, track_id = map(int, p_box)

                # Determine current frame status
                has_phone = phone_map.get(track_id, False)
                is_sleeping = False
                
                if not has_phone and enable_sleep_detection:
                    # Check for sleep
                    h, w, _ = frame.shape
                    pad = 20
                    cx1 = max(0, x1 - pad)
                    cy1 = max(0, y1 - pad)
                    cx2 = min(w, x2 + pad)
                    cy2 = min(h, y2 + pad)
                    person_crop = frame[cy1:cy2, cx1:cx2]
                    
                    if person_crop.size > 0:
                        sleep_key = f"{camera_name}_id_{track_id}"
                        kpts = pose_keypoints_map.get(track_id)
                        
                        sleep_status, sleep_details = self.sleep_detector.process_crop(
                            person_crop,
                            id_key=sleep_key,
                            keypoints=kpts,
                            crop_origin=(cx1, cy1)
                        )
                        
                        if sleep_status in ("sleeping", "drowsy"):
                            is_sleeping = True

                # --- TIME-BASED VIOLATION TRACKING ---
                if has_phone:
                    key = (track_id, "texting")
                    current_detections.add(key)
                    
                    if key not in self.violation_timers:
                        self.violation_timers[key] = {
                            'start_time': current_time,
                            'last_seen': current_time
                        }
                    else:
                        self.violation_timers[key]['last_seen'] = current_time
                
                if is_sleeping:
                    key = (track_id, "sleeping")
                    current_detections.add(key)
                    
                    if key not in self.violation_timers:
                        self.violation_timers[key] = {
                            'start_time': current_time,
                            'last_seen': current_time
                        }
                    else:
                        self.violation_timers[key]['last_seen'] = current_time

            # --- RESET TIMERS for violations not currently detected ---
            for key, timer_data in list(self.violation_timers.items()):
                if key not in current_detections:
                    gap = current_time - timer_data['last_seen']
                    if gap > self.reset_gap:
                        del self.violation_timers[key]

            # --- DETERMINE STATUS AND SAVE EVIDENCE ---
            for p_box in person_boxes:
                x1, y1, x2, y2, track_id = map(int, p_box)
                
                status = "safe"
                color = (0, 255, 0)
                
                # FIXED: Always show person name if available
                person_name = self.person_identities.get(track_id, None)
                person_conf = self.identity_confidences.get(track_id, 0.0)
                
                if person_name:
                    # Show name with confidence
                    label = f"{person_name} ({person_conf:.2f})"
                else:
                    label = f"ID: {track_id}"
                
                # Check texting timer
                texting_key = (track_id, "texting")
                sleeping_key = (track_id, "sleeping")
                
                texting_duration = 0.0
                sleeping_duration = 0.0
                
                if texting_key in self.violation_timers:
                    texting_duration = current_time - self.violation_timers[texting_key]['start_time']
                    
                    if texting_duration >= self.phone_duration_threshold:
                        status = "texting"
                        color = (0, 0, 255)
                        global_status = "texting"
                        label += f" | PHONE {texting_duration:.1f}s"
                    else:
                        color = (0, 165, 255)
                        label += f" | Phone {texting_duration:.1f}s/{self.phone_duration_threshold:.0f}s"
                
                elif sleeping_key in self.violation_timers:
                    sleeping_duration = current_time - self.violation_timers[sleeping_key]['start_time']
                    
                    if sleeping_duration >= self.sleep_duration_threshold:
                        status = "sleeping"
                        color = (255, 0, 0)
                        if global_status != "texting":
                            global_status = "sleeping"
                        label += f" | SLEEP {sleeping_duration:.1f}s"
                    else:
                        color = (0, 255, 255)
                        label += f" | Sleep {sleeping_duration:.1f}s/{self.sleep_duration_threshold:.0f}s"

                # --- SAVE EVIDENCE ---
                if save_screenshots and status != "safe":
                    key = (track_id, status)
                    last_save_time = self.cooldowns.get(key, 0)
                    
                    if (current_time - last_save_time) > self.cooldown_seconds:
                        type_str = "PHONE" if status == "texting" else "SLEEP"
                        duration_str = f"{texting_duration if status == 'texting' else sleeping_duration:.1f}s"
                        
                        self.save_evidence(
                            frame, x1, y1, x2, y2, camera_name, 
                            type_str, track_id=track_id, duration=duration_str,
                            person_name=person_name
                        )
                        screenshot_saved_global = True
                        self.cooldowns[key] = current_time
                        label += " [SAVED]"

                self.last_display_data.append((x1, y1, x2, y2, color, status, label))

        # --- DRAWING ---
        for (x1, y1, x2, y2, color, status, label) in self.last_display_data:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # FIXED: Better label rendering with background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_w, label_h = label_size
            
            # Draw background for label
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if status == "texting":
                global_status = "texting"
            elif status == "sleeping" and global_status != "texting":
                global_status = "sleeping"

        return frame, global_status, screenshot_saved_global

    def _associate_phones_to_persons(self, person_boxes, phone_boxes, pose_keypoints_map):
        """Scale-invariant geometric association."""
        mapping = {}
        if not phone_boxes: 
            return mapping

        for ph in phone_boxes:
            ph_x1, ph_y1, ph_x2, ph_y2, _ = ph
            ph_cx = (ph_x1 + ph_x2) / 2
            ph_cy = (ph_y1 + ph_y2) / 2

            best_person_id = None
            min_normalized_dist = float('inf')

            for p in person_boxes:
                p_x1, p_y1, p_x2, p_y2, p_id = p
                
                person_width = p_x2 - p_x1
                person_height = p_y2 - p_y1

                pad_x = person_width * 0.4
                pad_y = person_height * 0.3
                
                if not (p_x1 - pad_x <= ph_cx <= p_x2 + pad_x and 
                       p_y1 - pad_y <= ph_cy <= p_y2 + pad_y):
                    continue

                kpts = pose_keypoints_map.get(p_id)
                raw_dist = float('inf')

                if kpts is not None:
                    wrists = []
                    if kpts[9][0] > 0: 
                        wrists.append(kpts[9])
                    if kpts[10][0] > 0: 
                        wrists.append(kpts[10])

                    if wrists:
                        for w_pt in wrists:
                            d = math.hypot(ph_cx - w_pt[0], ph_cy - w_pt[1])
                            if d < raw_dist: 
                                raw_dist = d

                        eye_y = 0
                        if kpts[1][1] > 0: 
                            eye_y = kpts[1][1]
                        elif kpts[2][1] > 0: 
                            eye_y = kpts[2][1]

                        if eye_y > 0 and ph_cy < eye_y - (person_height * 0.1):
                            raw_dist += person_height * 0.5

                if raw_dist == float('inf'):
                    p_cx = (p_x1 + p_x2) / 2
                    p_chest_y = p_y1 + person_height * 0.4
                    raw_dist = math.hypot(ph_cx - p_cx, ph_cy - p_chest_y)

                normalized_dist = raw_dist / person_height

                if normalized_dist < min_normalized_dist:
                    min_normalized_dist = normalized_dist
                    best_person_id = p_id

            MAX_NORMALIZED_DISTANCE = 0.6
            
            if best_person_id is not None and min_normalized_dist < MAX_NORMALIZED_DISTANCE:
                mapping[best_person_id] = True

        return mapping

    def _associate_pose_to_persons(self, person_boxes, pose_boxes, pose_kpts):
        """Associate pose detections with tracked persons."""
        mapping = {}
        if len(person_boxes) == 0 or len(pose_boxes) == 0:
            return mapping
        
        for p_box in person_boxes:
            px1, py1, px2, py2, track_id = p_box
            p_area = (px2 - px1) * (py2 - py1)
            best_iou = 0
            best_idx = -1

            for i, (pox1, poy1, pox2, poy2) in enumerate(pose_boxes):
                ix1 = max(px1, pox1)
                iy1 = max(py1, poy1)
                ix2 = min(px2, pox2)
                iy2 = min(py2, poy2)

                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    po_area = (pox2 - pox1) * (poy2 - poy1)
                    union_area = p_area + po_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0

                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i

            if best_idx != -1 and best_iou > 0.3:
                mapping[track_id] = pose_kpts[best_idx]
                
        return mapping

    def save_evidence(self, frame, x1, y1, x2, y2, camera_name="Unknown", 
                     detection_type="PHONE", track_id=None, duration="N/A", person_name=None):
        """Save evidence screenshot with duration and person name metadata."""
        evidence_img = frame.copy()
        box_color = (0, 0, 255) if detection_type == "PHONE" else (255, 0, 0)
        cv2.rectangle(evidence_img, (x1, y1), (x2, y2), box_color, 3)
        
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.rectangle(evidence_img, (0, 0), (evidence_img.shape[1], 40), (0,0,0), -1)
        
        # Build header with person name if available
        if person_name:
            header_text = f"{detection_type} {duration} | {person_name} | {camera_name} | {ts}"
        else:
            header_text = f"{detection_type} {duration} | {camera_name} | {ts}"
            if track_id is not None: 
                header_text += f" | ID: {track_id}"

        cv2.putText(evidence_img, header_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        timestamp_fn = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        safe_cam_name = "".join([c for c in camera_name 
                                if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
        
        # Include person name in filename if available
        if person_name:
            safe_person_name = "".join([c for c in person_name 
                                       if c.isalnum() or c in ('_', '-')]).strip()
            filename = os.path.join(
                self.output_dir, 
                f"evidence_{detection_type.lower()}_{safe_person_name}_{safe_cam_name}_{timestamp_fn}.jpg"
            )
        else:
            id_str = f"_id{track_id}" if track_id is not None else ""
            filename = os.path.join(
                self.output_dir, 
                f"evidence_{detection_type.lower()}_{safe_cam_name}_{timestamp_fn}{id_str}.jpg"
            )
        
        cv2.imwrite(filename, evidence_img)
        print(f"📸 EVIDENCE SAVED: {filename} (Duration: {duration})")
