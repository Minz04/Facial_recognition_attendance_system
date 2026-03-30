import os
from dotenv import load_dotenv

load_dotenv()

# 1. CẤU HÌNH SUPABASE
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Cảnh báo: Không tìm thấy SUPABASE_URL hoặc SUPABASE_KEY trong file .env.")

# Cấu hình dự án
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CLIENT_ID = "Camera-ND1"

# 2. CẤU HÌNH AI CORE (YOLOv5 & DeepFace)
# Cấu hình YOLOv5
YOLO_REPO_PATH = os.path.join(PROJECT_ROOT, 'yolov5')
YOLO_MODEL_WEIGHTS_PATH = os.path.join(YOLO_REPO_PATH, 'runs', 'train', 'train_face_detection_v3', 'weights', 'best.pt')
YOLO_CONFIDENCE_THRESHOLD = 0.30

# Cấu hình DeepFace
DEEPFACE_MODEL_NAME = 'Facenet512'
DEEPFACE_DISTANCE_METRIC = 'euclidean_l2'
DEEPFACE_RECOGNITION_THRESHOLD = 0.80
USE_FACE_ALIGNMENT = True
ALIGNMENT_DETECTOR_BACKEND = 'mtcnn'

# 3. CẤU HÌNH MẠNG VÀ SERVER
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5500


# 4. CẤU HÌNH LUỒNG CAMERA CHO AI
CAMERA_IP = "192.168.33.80"
RTSP_USERNAME = "adminRTSP"
RTSP_PASSWORD = "baominh55555"

RTSP_URL_LOW_RES = f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{CAMERA_IP}:554/stream2"

CAMERA_FPS = 15  

# 5. CẤU HÌNH LOGIC ĐIỂM DANH
VOTING_FRAMES = 10
VOTING_THRESHOLD = 6
ATTENDANCE_COOLDOWN_SEC = 900

MIN_FACE_SIZE_WORKER = 30 

# 4. CẤU HÌNH ROI 
ROI_CONFIG = {
"X_START_PERCENT": 0.10,
"X_END_PERCENT": 0.80,
"Y_START_PERCENT": 0.15,
"Y_END_PERCENT": 1.00
}