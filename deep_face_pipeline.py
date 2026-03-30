import cv2
import os
import time
import numpy as np
import config as server_config

# Cấu hình biến môi trường trước khi import DeepFace
os.environ['DEEPFACE_HOME'] = "D:/AI_MODELS" 

from deepface import DeepFace
from deepface.modules import verification 
from yolo_face_detector import YOLOFaceDetector 

class FaceRecognitionPipeline:
    def __init__(self, yolo_model_weights_path, db_path,
                 yolo_repo_path='yolov5', 
                 df_model_name=getattr(server_config, 'DEEPFACE_MODEL_NAME', 'Facenet512'), 
                 df_distance_metric=getattr(server_config, 'DEEPFACE_DISTANCE_METRIC', 'euclidean_l2'), 
                 use_alignment=getattr(server_config, 'USE_FACE_ALIGNMENT', True),
                 alignment_detector_backend=getattr(server_config, 'ALIGNMENT_DETECTOR_BACKEND', 'mtcnn'),
                 yolo_confidence=getattr(server_config, 'YOLO_CONFIDENCE_THRESHOLD', 0.30),
                 custom_threshold=getattr(server_config, 'DEEPFACE_RECOGNITION_THRESHOLD', 0.80)
                 ): 
        """
        Khởi tạo pipeline nhận diện khuôn mặt.
        """
        self.yolo_detector = YOLOFaceDetector(
            yolo_repo_path=yolo_repo_path,
            model_weights_path=yolo_model_weights_path,
            confidence_threshold=yolo_confidence
        )
        self.db_path = db_path
        self.df_model_name = df_model_name
        self.df_distance_metric = df_distance_metric
        self.use_alignment = use_alignment
        self.alignment_detector_backend = alignment_detector_backend

        # Kiểm tra database
        if not os.path.exists(self.db_path):
            print(f"Warning: Database path '{self.db_path}' does not exist.")
        else:
            print(f"DeepFace database path set to: {self.db_path}")

        # [LOGIC MỚI] Xử lý ngưỡng (Thủ công hoặc Tự động)
        if custom_threshold is not None:
            self.threshold = custom_threshold
            print(f"Configuration: Model={self.df_model_name} | Metric={self.df_distance_metric}")
            print(f"-> MANUAL Threshold set to: {self.threshold}")
        else:
            # Tự động lấy ngưỡng chuẩn từ thư viện nếu không nhập tay
            self.threshold = verification.find_threshold(self.df_model_name, self.df_distance_metric)
            print(f"Configuration: Model={self.df_model_name} | Metric={self.df_distance_metric}")
            print(f"-> Auto-calculated Threshold: {self.threshold}")

        if self.use_alignment:
            print(f"Face alignment enabled using '{self.alignment_detector_backend}'.")
        else:
            print("Face alignment disabled.")

    def _add_padding(self, img, bbox, padding_pct=0.15):
        """
        [QUAN TRỌNG] Hàm mở rộng vùng crop để lấy thêm tóc/cằm
        Giúp Facenet nhận diện tốt hơn.
        """
        h_img, w_img = img.shape[:2]
        x1, y1, x2, y2 = bbox
        w_box = x2 - x1
        h_box = y2 - y1

        # Tính toán vùng mở rộng
        pad_w = int(w_box * padding_pct)
        pad_h = int(h_box * padding_pct)

        # Đảm bảo không vượt quá kích thước ảnh gốc (Clamping)
        new_x1 = max(0, x1 - pad_w)
        new_y1 = max(0, y1 - pad_h)
        new_x2 = min(w_img, x2 + pad_w)
        new_y2 = min(h_img, y2 + pad_h)

        # Trả về ảnh đã cắt
        return img[new_y1:new_y2, new_x1:new_x2]

    def _get_aligned_face(self, face_image_bgr):
        """
        Căn chỉnh khuôn mặt (Align) dùng DeepFace
        """
        try:
            # Thử backend chính (MTCNN/RetinaFace...)
            extracted_data = DeepFace.extract_faces(
                img_path=face_image_bgr,
                detector_backend=self.alignment_detector_backend,
                enforce_detection=True, # Bắt buộc tìm thấy mặt để align chuẩn
                align=True
            )
            if extracted_data and len(extracted_data) > 0:
                face_rgb = extracted_data[0]['face']
                # DeepFace trả về float (0-1), cần convert sang uint8 (0-255) cho OpenCV
                return cv2.cvtColor((face_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        except:
            # Fallback sang opencv nếu backend chính lỗi (dự phòng)
            if self.alignment_detector_backend != 'opencv':
                try:
                    extracted_data = DeepFace.extract_faces(
                        img_path=face_image_bgr, detector_backend='opencv', enforce_detection=True, align=True
                    )
                    if extracted_data and len(extracted_data) > 0:
                        face_rgb = extracted_data[0]['face']
                        return cv2.cvtColor((face_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                except: pass
        return None

    def process_image(self, image_path_or_array, output_image_path=None):
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None: return None, []
        else:
            image = image_path_or_array.copy()

        if self.yolo_detector.model is None: return image, []

        bboxes, _ = self.yolo_detector.detect_faces(image) # Chỉ lấy bbox
        recognition_results = []
        image_to_draw_on = image.copy()

        for bbox in bboxes:
            # [CẢI TIẾN] Cắt ảnh với Padding (mở rộng 15%)
            face_with_padding = self._add_padding(image, bbox, padding_pct=0.15)
            
            if face_with_padding.size == 0: continue

            face_to_recognize = face_with_padding
            
            # Alignment (Căn chỉnh lại khuôn mặt đã cắt)
            if self.use_alignment:
                aligned = self._get_aligned_face(face_with_padding)
                # Nếu align thành công thì dùng ảnh align, không thì dùng ảnh padding
                if aligned is not None: face_to_recognize = aligned
            
            name = "Unknown"
            distance_val = -1.0
            
            try:
                # Tìm kiếm khuôn mặt trong DB
                dfs = DeepFace.find(img_path=face_to_recognize,
                                    db_path=self.db_path,
                                    model_name=self.df_model_name,
                                    distance_metric=self.df_distance_metric,
                                    enforce_detection=False, # Đã crop rồi nên False
                                    silent=True)
                
                if isinstance(dfs, list) and len(dfs) > 0 and not dfs[0].empty:
                    best_match = dfs[0].iloc[0]
                    identity_path = best_match['identity']
                    
                    # Fix tên folder/file để lấy nhãn
                    possible_name_1 = os.path.basename(os.path.dirname(identity_path))
                    possible_name_2 = os.path.splitext(os.path.basename(identity_path))[0]
                    
                    if possible_name_1 and possible_name_1 != "database" and possible_name_1 != os.path.basename(self.db_path):
                        name = possible_name_1
                    else:
                        name = possible_name_2

                    # Lấy giá trị khoảng cách
                    if 'distance' in best_match.index:
                        distance_val = best_match['distance']
                    else:
                        distance_val = best_match.get(f"{self.df_model_name}_{self.df_distance_metric}", -1)

                    # So sánh với ngưỡng (Threshold)
                    if distance_val > self.threshold:
                        name = "Unknown"
                
                recognition_results.append({"bbox": bbox, "name": name, "distance": distance_val})
                
                # Vẽ Box và Tên
                label = f"{name} ({distance_val:.2f})" if distance_val != -1 else name
                x1, y1, x2, y2 = bbox
                color = (0, 165, 255) if name == "Unknown" else (0, 255, 0)
                cv2.rectangle(image_to_draw_on, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image_to_draw_on, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            except Exception: 
                recognition_results.append({"bbox": bbox, "name": "Error", "distance": -1})

        if output_image_path: cv2.imwrite(output_image_path, image_to_draw_on)
        return image_to_draw_on, recognition_results

    def process_video(self, video_source, output_video_path=None, 
                      display_width=640, display_height=480,
                      camera_capture_width=640, camera_capture_height=480):
        """
        Nhận diện khuôn mặt từ video/camera.
        """
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_capture_height)
        
        # Thử set FPS
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print(f"Error: Cannot open video source {video_source}")
            return

        out = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (display_width, display_height))

        print("Starting video recognition... Press 'Esc' to exit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream.")
                break

            # Resize hiển thị cho nhẹ
            frame_resized = cv2.resize(frame, (display_width, display_height))

            # Gọi hàm xử lý ảnh tĩnh cho frame này
            frame_with_boxes, results = self.process_image(frame_resized)

            cv2.imshow("Face Recognition System", frame_with_boxes)
            
            if out is not None:
                out.write(frame_with_boxes)

            if cv2.waitKey(1) & 0xFF == 27: # Esc
                break

        cap.release()
        if out: out.release()
        cv2.destroyAllWindows()