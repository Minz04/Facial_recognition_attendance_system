import os
import re
import cv2
import time
import numpy as np
import pandas as pd
import traceback

from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtGui import QImage
from deepface import DeepFace

# Import config (từ thư mục gốc)
import config # <--- THÊM DÒNG NÀY

try:
    from yolo_face_detector import YOLOFaceDetector
    YOLO_AVAILABLE = True
except ImportError:
    print("[LỖI WORKER] Không thể nhập YOLOFaceDetector.")
    YOLO_AVAILABLE = False
    class YOLOFaceDetector:
        def __init__(self, *args, **kwargs): self.model = None
        def detect_faces(self, *args, **kwargs): return [], []

# Các biến cấu hình từ config.py
# DEEPFACE_MODEL_NAME = 'ArcFace'
# DEEPFACE_DETECTOR_BACKEND = 'mtcnn'
# DEEPFACE_ALIGNMENT = True
# DEEPFACE_DISTANCE_METRIC = 'cosine'
# DEEPFACE_RECOGNITION_THRESHOLD = 0.4
# DATABASE_FOLDER_WORKER = "database" 
# FRAME_SKIP_RATE = 3 
# MIN_FACE_SIZE_WORKER = 50  
# QUALITY_THRESHOLD_WORKER = 60 
# RECOGNITION_COOLDOWN_SEC = 1.5

class RecognitionSignals(QObject):
    frame_ready = pyqtSignal(QImage)
    recognition_result = pyqtSignal(np.ndarray, str, str)
    no_recognition = pyqtSignal()
    error = pyqtSignal(str)

class RecognitionWorker(QThread):
    def __init__(self, yolo_weights_path: str, yolo_repo_path: str,
                 db_path: str, yolo_confidence: float = 0.324, parent=None):
        super().__init__(parent)
        self.signals = RecognitionSignals()
        self.running = False
        self._prevent_run = False
        # Lưu đường dẫn DB được truyền vào (đã từ config)
        self.database_path = db_path # Thay thế cho global DATABASE_FOLDER_WORKER
        self.frame_skip_counter = 0
        self._initialize_models(yolo_weights_path, yolo_repo_path, yolo_confidence)

    def _initialize_models(self, yolo_weights_path, yolo_repo_path, yolo_confidence):
        if not YOLO_AVAILABLE:
            self._prevent_run = True
            return
        try:
            self.yolo_detector = YOLOFaceDetector(
                yolo_repo_path=yolo_repo_path,
                model_weights_path=yolo_weights_path,
                confidence_threshold=yolo_confidence
            )
            if self.yolo_detector.model is None: raise ValueError("Model YOLO không tải được.")
            dummy_img = np.zeros((112, 112, 3), dtype=np.uint8)
            # Sử dụng các biến từ config cho DeepFace
            DeepFace.represent(dummy_img, model_name=config.DEEPFACE_MODEL_NAME, enforce_detection=False)
            print("[WORKER] Models initialized successfully")
        except Exception as e:
            print(f"[LỖI WORKER] Khởi tạo Models thất bại: {e}")
            self._prevent_run = True
            
    def _is_face_large_enough(self, bbox_coords):
        width = bbox_coords[2] - bbox_coords[0]
        height = bbox_coords[3] - bbox_coords[1]
        return min(width, height) >= config.MIN_FACE_SIZE_WORKER # Dùng biến từ config

    def _calculate_face_quality(self, face_crop_bgr):
        # Logic này bạn tự định nghĩa và các ngưỡng (200, 100, 50, ...)
        # hiện đang là giá trị cứng. Nếu muốn cấu hình, bạn có thể thêm vào config.py
        if face_crop_bgr is None or face_crop_bgr.size == 0: return 0
        try:
            gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
            blur_score_raw = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_quality = 0
            if blur_score_raw > 200: blur_quality = 50
            elif blur_score_raw > 100: blur_quality = 35
            elif blur_score_raw > 50: blur_quality = 20
            elif blur_score_raw > 20: blur_quality = 10
            brightness_score_raw = np.mean(gray)
            brightness_quality = 0
            if 80 <= brightness_score_raw <= 180: brightness_quality = 50
            elif 60 <= brightness_score_raw <= 200: brightness_quality = 35
            elif 40 <= brightness_score_raw <= 220: brightness_quality = 20
            elif 20 <= brightness_score_raw <= 240: brightness_quality = 10
            return blur_quality + brightness_quality
        except Exception: return 0

    def _initialize_camera(self):
        cap = None
        try:
            # Sử dụng config.WEBCAM_INDEX hoặc duyệt qua
            # Hiện tại bạn đang duyệt qua [0, 1, -1], có thể cấu hình cái này nếu muốn
            for cam_id in [config.WEBCAM_INDEX, 1, -1]: # Có thể bắt đầu với config.WEBCAM_INDEX
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened(): break
            if not cap or not cap.isOpened(): raise IOError("Không thể mở camera.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        except Exception as e:
            self.signals.error.emit(f"Lỗi camera: {str(e)}")
            if cap: cap.release()
            return None

    def _emit_frame_for_display(self, frame_to_emit_bgr):
        try:
            frame_rgb = cv2.cvtColor(frame_to_emit_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.signals.frame_ready.emit(q_img.copy())
        except Exception as e: print(f"[LỖI WORKER] _emit_frame_for_display: {e}")

    def reload_embeddings(self):
        print("[WORKER] Yêu cầu reload embeddings (xóa file .pkl)...")
        deleted_any = False
        try:
            # Sử dụng self.database_path thay vì DATABASE_FOLDER_WORKER
            for item_name in os.listdir(self.database_path):
                is_deepface_pkl = False
                if item_name.endswith(".pkl"):
                    # Các mẫu DeepFace thường dùng để lưu embeddings
                    if item_name.startswith("representations_") or \
                       item_name.startswith("ds_model_") or \
                       "face_recognition_embedding" in item_name:
                        is_deepface_pkl = True
                
                if is_deepface_pkl:
                    pkl_path = os.path.join(self.database_path, item_name)
                    if os.path.isfile(pkl_path): 
                        try:
                            os.remove(pkl_path)
                            print(f"[WORKER] Đã xóa file representation: {pkl_path}")
                            deleted_any = True
                        except Exception as e_del:
                            print(f"[WORKER] Lỗi khi xóa {pkl_path}: {e_del}")
            
            if not deleted_any:
                print(f"[WORKER] Không tìm thấy file .pkl nào khớp với mẫu trong thư mục: {self.database_path}")
            else:
                print(f"[WORKER] Hoàn tất reload embeddings.")

        except FileNotFoundError:
            print(f"[WORKER] Thư mục database '{self.database_path}' không tồn tại khi reload embeddings.")
        except Exception as e:
            print(f"[WORKER] Lỗi khi duyệt thư mục database '{self.database_path}' để xóa .pkl: {e}")
            traceback.print_exc()

    def run(self):
        if self._prevent_run:
            self.signals.error.emit("Worker không thể chạy do lỗi khởi tạo model.")
            return
        self.running = True
        cap = self._initialize_camera()
        if cap is None:
            self.running = False
            return
        last_recognition_times = {}
        while self.running:
            try:
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    time.sleep(0.05)
                    continue
                self.frame_skip_counter += 1
                if self.frame_skip_counter % config.FRAME_SKIP_RATE != 0: # Dùng biến từ config
                    if self.frame_skip_counter == 1: self._emit_frame_for_display(frame_bgr) # Vẫn hiển thị 1 frame đầu của chu kỳ
                    self.msleep(1)
                    continue
                if self.frame_skip_counter > 100 * config.FRAME_SKIP_RATE : self.frame_skip_counter = 0 # Đặt lại counter
                frame_for_display = frame_bgr.copy()
                
                bboxes, cropped_faces_bgr = self.yolo_detector.detect_faces(frame_bgr) # Cắt khuôn mặt
                best_match_overall = None
                found_known_person_this_frame = False
                
                for bbox, face_crop in zip(bboxes, cropped_faces_bgr):
                    if face_crop is None or face_crop.size == 0: continue
                    if not self._is_face_large_enough(bbox): continue
                    quality_score = self._calculate_face_quality(face_crop)
                    if quality_score < config.QUALITY_THRESHOLD_WORKER: # Dùng biến từ config
                        cv2.rectangle(frame_for_display, tuple(bbox[:2]), tuple(bbox[2:]), (0, 165, 255), 2)
                        cv2.putText(frame_for_display, f"Low Quality ({quality_score})", (bbox[0], bbox[1] - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
                        continue
                    try:
                        dfs_results = DeepFace.find(img_path=face_crop, db_path=self.database_path, # Dùng self.database_path
                                                    model_name=config.DEEPFACE_MODEL_NAME, # Dùng biến từ config
                                                    distance_metric=config.DEEPFACE_DISTANCE_METRIC, # Dùng biến từ config
                                                    detector_backend=config.DEEPFACE_DETECTOR_BACKEND, # Dùng biến từ config
                                                    align=config.DEEPFACE_ALIGNMENT, # Dùng biến từ config
                                                    enforce_detection=False, silent=True)
                        if dfs_results and isinstance(dfs_results, list) and len(dfs_results) > 0 and not dfs_results[0].empty:
                            df_person = dfs_results[0].iloc[0]
                            dist_col_expected = f"{config.DEEPFACE_MODEL_NAME}_{config.DEEPFACE_DISTANCE_METRIC}" # Dùng biến từ config
                            current_distance = df_person.get(dist_col_expected, df_person.get('distance'))
                            if current_distance is None: continue
                            if current_distance < config.DEEPFACE_RECOGNITION_THRESHOLD: # Dùng biến từ config
                                found_known_person_this_frame = True
                                identity_path_df = df_person['identity']
                                person_folder_name_df = os.path.basename(os.path.dirname(identity_path_df))
                                current_time_cooldown = time.time()
                                if person_folder_name_df in last_recognition_times and \
                                   (current_time_cooldown - last_recognition_times[person_folder_name_df]) < config.RECOGNITION_COOLDOWN_SEC: # Dùng biến từ config
                                    continue
                                last_recognition_times[person_folder_name_df] = current_time_cooldown
                                match_id_name = re.match(r"^(\d+)_?(.*)", person_folder_name_df)
                                display_name_df = person_folder_name_df
                                if match_id_name:
                                    raw_name = match_id_name.group(2) if match_id_name.group(2) else "N/A"
                                    display_name_df = raw_name.replace('_', ' ')
                                if best_match_overall is None or current_distance < best_match_overall[3]:
                                     best_match_overall = (face_crop.copy(), display_name_df, person_folder_name_df, current_distance)
                                cv2.rectangle(frame_for_display, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 2)
                                label = f"{display_name_df} ({current_distance:.2f})"
                                cv2.putText(frame_for_display, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            else:
                                cv2.rectangle(frame_for_display, tuple(bbox[:2]), tuple(bbox[2:]), (0, 0, 255), 2)
                                cv2.putText(frame_for_display, "Unknown", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        else:
                            cv2.rectangle(frame_for_display, tuple(bbox[:2]), tuple(bbox[2:]), (0, 0, 255), 2)
                            cv2.putText(frame_for_display, "Unknown", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    except Exception as e_find:
                        print(f"[LỖI WORKER] DeepFace.find: {e_find}")
                        cv2.rectangle(frame_for_display, tuple(bbox[:2]), tuple(bbox[2:]), (255, 0, 0), 2)
                        cv2.putText(frame_for_display, "ErrorRec", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                if best_match_overall:
                    crop_sig, name_sig, folder_sig, _ = best_match_overall
                    self.signals.recognition_result.emit(crop_sig, name_sig, folder_sig)
                elif not found_known_person_this_frame:
                    current_time_no_rec = time.time()
                    should_emit_no_recognition = True
                    if last_recognition_times:
                        # Thời gian clear last_recognition_times cũng dựa vào cooldown (gấp đôi)
                        if any((current_time_no_rec - t) < config.RECOGNITION_COOLDOWN_SEC * 2 for t in last_recognition_times.values()): # Dùng biến từ config
                            should_emit_no_recognition = False
                    if should_emit_no_recognition:
                        self.signals.no_recognition.emit()
                        last_recognition_times.clear()
                self._emit_frame_for_display(frame_for_display)
                self.msleep(10)
            except Exception as e_loop:
                print(f"[LỖI WORKER] Vòng lặp chính: {e_loop}")
                traceback.print_exc()
                self.signals.error.emit(f"Lỗi worker: {str(e_loop)}")
                self.msleep(500)
        if cap: cap.release()
        print("[WORKER] Luồng nhận diện đã dừng.")
    def stop(self):
        self.running = False
        print("[WORKER] Yêu cầu dừng luồng nhận diện...")