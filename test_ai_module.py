import cv2
import numpy as np
import os
import sys

# Thêm thư mục gốc của project vào Python path để import các module
# Đường dẫn đã được điều chỉnh để trỏ đến thư mục gốc của project,
# đảm bảo các import server.ModuleAI.* và database.supabase_client hoạt động
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import các module AI từ cấu trúc thư mục mới
from server.ModuleAI.face_detector import FaceDetector
from server.ModuleAI.face_encoder import FaceEncoder
from server.ModuleAI.face_matcher import FaceMatcher
# from server.utils.helpers import crop_face_from_image, image_to_jpeg_bytes, img_bytes_to_cv2 
# Các hàm này không được sử dụng trực tiếp trong test_ai_module hiện tại,
# nên có thể bỏ qua hoặc giữ lại nếu bạn có kế hoạch dùng chúng.

# Import supabase_client và config từ thư mục gốc của project
import database.supabase_client as supabase_client
import config

if __name__ == "__main__":
    print("===================================")
    print("   Starting AI Core Modules Test   ")
    print("===================================")

    # 1. Khởi tạo các module
    print("\n--- Initializing AI Modules ---")
    detector = FaceDetector()
    encoder = FaceEncoder()
    matcher = FaceMatcher() # FaceMatcher sẽ tự động gọi load_known_faces() trong __init__

    # 2. Kiểm tra dữ liệu khuôn mặt đã biết
    print("\n--- Checking Known Faces ---")
    if not matcher.known_face_encodings.size:
        print("WARNING: No approved face encodings found in Supabase DB.")
        print("To test matching, please ensure you have approved users with face encodings in Supabase.")
        print("You can manually insert a user and an encoding for them in Supabase, then set 'is_approved' to true for the user, and 'approved' to true for the face_encoding.")
        print("Skipping face matching test.")
        # Exit hoặc tiếp tục mà không kiểm tra matching
        # exit() # Có thể thoát nếu muốn
    else:
        print(f"Successfully loaded {len(matcher.known_face_encodings)} known faces.")

    # 3. Tải một ảnh mẫu để test
    print("\n--- Loading Test Image ---")
    # Đặt test_face.jpg vào thư mục gốc của project (cùng cấp với thư mục 'ModuleAI')
    test_image_path = os.path.join(config.PROJECT_ROOT, "test_face.jpg") 
    if not os.path.exists(test_image_path):
        print(f"ERROR: Test image '{test_image_path}' not found.")
        print("Please place an image with at least one face named 'test_face.jpg' in the project root for testing.")
        exit()
    
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"ERROR: Could not load image '{test_image_path}'. Check if it's a valid image file.")
        exit()
    print(f"Test image '{test_image_path}' loaded successfully. Shape: {test_image.shape}")

    # 4. Phát hiện khuôn mặt
    print("\n--- Detecting Faces ---")
    # detect_faces giờ trả về bboxes (x1,y1,x2,y2) và cropped_faces
    bboxes, cropped_faces_from_detector = detector.detect_faces(test_image)
    print(f"Found {len(bboxes)} face(s).")

    if bboxes:
        for i, (bbox, cropped_face_img) in enumerate(zip(bboxes, cropped_faces_from_detector)):
            print(f"\nProcessing Face {i+1}:")
            x1, y1, x2, y2 = bbox
            print(f"  Bounding Box (x1, y1, x2, y2): ({x1}, {y1}, {x2}, {y2})")
            
            # Hiển thị ảnh khuôn mặt đã cắt bởi YOLO detector
            if cropped_face_img is not None and cropped_face_img.size > 0:
                cv2.imshow(f"Detected Face {i+1} (YOLO Crop)", cropped_face_img)
            else:
                print(f"  Warning: Cropped face {i+1} from detector is None or empty.")
                continue # Bỏ qua nếu không có ảnh cắt hợp lệ

            # 5. Mã hóa khuôn mặt
            print("  Encoding face...")
            # face_encoder.py của bạn nhận ảnh đã cắt, không cần face_location gốc
            encoding = encoder.encode_face(cropped_face_img)
            
            if encoding is not None:
                print(f"  Face {i+1} encoded. Shape: {encoding.shape}")

                # 6. So sánh khuôn mặt (chỉ nếu có known faces)
                if matcher.known_face_encodings.size:
                    print("  Matching face...")
                    matched_info, score = matcher.match_embedding(encoding)
                    if matched_info:
                        print(f"  Face {i+1} matched: {matched_info['full_name']} ({matched_info['student_id']}), Score: {score:.4f}")
                    else:
                        print(f"  Face {i+1} is UNKNOWN. Best score: {score:.4f}")
                else:
                    print("  Skipping face matching (no known faces loaded).")
            else:
                print(f"  Failed to encode Face {i+1}.")
    else:
        print("No faces detected in the test image.")

    print("\n--- Test Complete. Press any key to close image windows. ---")
    cv2.waitKey(0)
    cv2.destroyAllWindows()