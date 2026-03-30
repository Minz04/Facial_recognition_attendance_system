# test_webcam_recognition.py

import cv2
import time
import os
import sys

# Thêm PROJECT_ROOT vào sys.path để import các module custom
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config as server_config
from supabase import create_client
from server.ModuleAI.face_detector import FaceDetector
from server.ModuleAI.face_encoder import FaceEncoder
from server.ModuleAI.face_matcher import FaceMatcher
from server.utils.helpers import crop_face_from_image # Dùng helper để cắt

# --- Cấu hình Webcam (0 là webcam chính) ---
WEBCAM_INDEX = 0 
WINDOW_NAME = "AI & DB Validation Test - PRESS 'Q' TO QUIT"
FONT = cv2.FONT_HERSHEY_SIMPLEX

def initialize_components():
    """Khởi tạo Supabase và các module AI"""
    print("LOG: Khởi tạo AI Modules và kết nối DB...")
    
    # Khởi tạo Supabase Client
    supabase_url = server_config.SUPABASE_URL
    supabase_key = server_config.SUPABASE_KEY
    supabase_client = create_client(supabase_url, supabase_key)
    
    # Khởi tạo các module
    detector = FaceDetector()
    encoder = FaceEncoder()
    matcher = FaceMatcher(supabase_client) # Matcher sẽ tự tải vector từ DB
    
    # Tải vector (Nếu có lỗi ở đây, DB vector của bạn sai)
    try:
        matcher.load_known_faces() 
        print(f"LOG: Tải dữ liệu thành công. Đã có {len(matcher.known_encodings)} vector.")
    except Exception as e:
        print(f"LỖI DB: Không thể tải vector. Vui lòng kiểm tra lại cấu trúc DB và dữ liệu vector: {e}")
        return None, None, None
        
    return detector, encoder, matcher


def run_webcam_test(detector, encoder, matcher):
    """Chạy vòng lặp nhận diện trực tiếp từ Webcam."""
    
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"LỖI: Không thể mở Webcam Index {WEBCAM_INDEX}. Kiểm tra lại camera.")
        return
        
    print("\n--- BẮT ĐẦU TEST NHẬN DIỆN ---")
    print("Hãy nhìn thẳng vào camera...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        
        # 1. Phát hiện Khuôn mặt (YOLO/MTCNN)
        face_boxes = detector.detect_faces(frame)
        
        status_text = "Status: No Face"
        color = (0, 0, 255) # Đỏ
        
        if face_boxes:
            bbox = face_boxes[0]
            
            # 2. Cắt Khuôn mặt
            face_image = crop_face_from_image(frame, bbox)
            
            # 3. Mã hóa và So khớp (ArcFace)
            face_encoding = encoder.encode_face(face_image)
            
            if face_encoding is not None:
                matched_person, confidence = matcher.match_face(face_encoding)
                
                if matched_person:
                    # Nhận diện thành công
                    name = matched_person['full_name']
                    status_text = f"MATCH: {name} ({confidence:.2f})"
                    color = (0, 255, 0) # Xanh lá
                else:
                    # Nhận diện thất bại (Unknown)
                    status_text = f"UNKNOWN ({confidence:.2f})"
                    color = (0, 165, 255) # Cam

            # Vẽ bounding box lên màn hình hiển thị
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        
        # Hiển thị trạng thái
        cv2.putText(display_frame, status_text, (20, 40), FONT, 1, color, 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, display_frame)

        # Thoát bằng phím Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detector, encoder, matcher = initialize_components()
    
    if detector and matcher:
        run_webcam_test(detector, encoder, matcher)