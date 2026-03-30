import torch
import cv2
import config as server_config
import pathlib
import sys

class FaceDetector:
    def __init__(self):
        self.weights_path = server_config.YOLO_MODEL_WEIGHTS_PATH
        self.conf_threshold = server_config.YOLO_CONFIDENCE_THRESHOLD
        
        print(f"Loading YOLOv5 model from: {self.weights_path}")
        
        try:
            # Fix lỗi Path trên Windows
            pathlib.PosixPath = pathlib.WindowsPath
            
            # Load model
            self.model = torch.hub.load(
                r'D:\DeepFace_YOLOv5s\yolov5',  
                'custom', 
                path=self.weights_path,         
                source='local'                  
            )
            
            self.model.conf = self.conf_threshold 
            print("LOG: YOLOv5 model loaded successfully (Local Mode).")

        except Exception as e:
            print(f"LỖI: Không thể tải mô hình YOLOv5. Chi tiết: {e}")
            self.model = None

    def detect_faces(self, frame):
        """
        Phát hiện khuôn mặt trong frame.
        """
        if frame is None or self.model is None:
            return []

        # [FIX QUAN TRỌNG]: Chuyển BGR (OpenCV) -> RGB (YOLO/Torch)
        # Nếu không chuyển, độ chính xác nhận diện gần như bằng 0
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Thực hiện suy luận
        results = self.model(img_rgb)

        face_boxes = []
        
        if len(results.pred) > 0:
            predictions = results.pred[0] 
            
            for *xyxy, conf, cls in predictions:
                x1, y1, x2, y2 = [int(x.item()) for x in xyxy]
                
                # Lọc kích thước nhỏ
                face_width = x2 - x1
                # Nếu config không có biến này, mặc định là 0
                min_size = getattr(server_config, 'MIN_FACE_SIZE_WORKER', 0)
                
                if face_width >= min_size:
                    face_boxes.append([x1, y1, x2, y2])
        
        # [DEBUG LOG] Uncomment dòng dưới nếu muốn thấy log detect liên tục
        # if len(face_boxes) > 0: print(f"DEBUG: YOLO Detected {len(face_boxes)} faces")
        
        return face_boxes