from deepface import DeepFace
import numpy as np
import cv2  # <--- Đừng quên import cv2
import config as server_config

class FaceEncoder:
    def __init__(self):
        self.model_name = server_config.DEEPFACE_MODEL_NAME
        self.detector_backend = 'mtcnn' 
        print(f"Initializing DeepFace model: {self.model_name}")

    def encode_face(self, face_image):
        """
        Mã hóa ảnh khuôn mặt. Tự động chuyển BGR -> RGB để khớp với DeepFace.
        """
        if face_image is None or face_image.size == 0:
            return None
        
        try:
            # [FIX QUAN TRỌNG] Chuyển hệ màu từ BGR (OpenCV) sang RGB (DeepFace)
            # Nếu không có dòng này, vector sẽ bị sai lệch hoàn toàn.
            img_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            embedding_objs = DeepFace.represent(
                img_path=img_rgb,  # Dùng ảnh RGB
                model_name=self.model_name,
                enforce_detection=False, 
                detector_backend=self.detector_backend
            )
            
            embedding = np.array(embedding_objs[0]['embedding'])
            
            # Chuẩn hóa L2 (Bắt buộc cho Facenet512 + Euclidean)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding

        except Exception as e:
            print(f"Lỗi Encode: {e}")
            return None