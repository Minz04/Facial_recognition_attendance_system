import os
import sys

# --- 1. CẤU HÌNH MÔI TRƯỜNG (PHẢI ĐẶT TRÊN CÙNG) ---
os.environ["OPENCV_LOG_LEVEL"] = "OFF" # TẮT LOG OPENCV
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0" # TẮT LOG VIDEOIO
# FFMPEG: Timeout 5s, dùng TCP để ổn định
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
# Cấu hình DeepFace Home (Nếu cần)
os.environ['DEEPFACE_HOME'] = "D:/AI_MODELS" 

import eventlet
eventlet.monkey_patch() # Vá lỗi xung đột luồng (Quan trọng cho SocketIO)

import time 
import threading
from datetime import datetime
from collections import deque
import cv2
import pytz
from datetime import datetime, timedelta 

from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
from dotenv import load_dotenv
from flasgger import Swagger 
from supabase import create_client


# --- THÊM PROJECT ROOT VÀO SYS.PATH ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config as server_config
from camera_stream import CameraStream
from server.utils.offline_manager import save_offline_record, read_and_clear_queue, restore_queue

# Import AI Modules
from server.ModuleAI.face_detector import FaceDetector
from server.ModuleAI.face_encoder import FaceEncoder
from server.ModuleAI.face_matcher import FaceMatcher

# Import API Blueprints
from server.api.face_management_routes import face_management_bp 
from server.api.attendance_routes import attendance_bp

# Import Utils
from server.utils.helpers import crop_face_from_image
from server.utils.db_helpers import (
    get_active_schedule_for_student, 
    determine_status,
    get_or_create_device,   
    update_device_heartbeat 
)

# Import Global State
from server.app_state import (
    CLIENT_VOTING_QUEUE,
    VOTING_LOCK,
    last_attendance_records # Chỉ dùng lưu người ĐÃ ĐIỂM DANH THÀNH CÔNG
)
# Lưu ý: last_attendance_records import từ app_state để dùng chung global

import gc 

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Khai báo biến toàn cục (Thay bằng key thật của bạn)
API_SECRET_KEY = "secret-key-244466666" 

# --- Cấu hình kiểm tra API Key từ mọi request ---x
@app.before_request
def check_api_key():
    # 1. Bỏ qua cho request OPTIONS
    if request.method == 'OPTIONS':
        return
    
    # 2. Bỏ qua các đường dẫn công khai (Whitelist) 
    if request.path.startswith('/static') or \
       request.path.startswith('/swagger') or \
       request.path.startswith('/flasgger_static') or \
       request.path.startswith('/apispec_1.json') or \
       request.path == '/ping' or \
       request.path == '/' or \
       request.path.startswith('/socket.io'): 
        return

    # 3. Kiểm tra chìa khóa trong Header
    key = request.headers.get('x-api-key')
    if key != API_SECRET_KEY:
        return jsonify({"message": "Truy cập bị từ chối: Sai API Key!"}), 401

# --- CẤU HÌNH SWAGGER (BẢO MẬT) ---

# 1. Template
swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Hệ thống Điểm danh AI",
        "description": "API Documentation (Yêu cầu nhập x-api-key)",
        "version": "1.0.0"
    },
    "securityDefinitions": {
        "APIKeyHeader": {
            "type": "apiKey",
            "name": "x-api-key", 
            "in": "header"
        }
    },
    "security": [
        {
            "APIKeyHeader": []
        }
    ]
}

# 2. Config
swagger_config = {
    "headers": [], 
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/swagger/" 
}

# 3. Khởi tạo (QUAN TRỌNG: Phải truyền template vào đây)
swagger = Swagger(app, config=swagger_config, template=swagger_template)

# Cấu hình Ping Timeout 60s để Client không bị báo Offline khi Camera xử lý lâu
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', ping_timeout=60, ping_interval=25)

# BIẾN TOÀN CỤC LƯU ID THIẾT BỊ
CURRENT_DEVICE_UUID = None


# 1. HÀM KHỞI TẠO ỨNG DỤNG
def create_app():
    app.config.from_object(server_config) 
    
    # --- [CẬP NHẬT CONFIG CHO DEEPFACE (MTCNN)] ---
    # Đảm bảo config lấy đúng từ file config.py mới sửa
    app.config['DEEPFACE_MODEL_NAME'] = getattr(server_config, 'DEEPFACE_MODEL_NAME', 'Facenet512')
    app.config['DEEPFACE_DISTANCE_METRIC'] = getattr(server_config, 'DEEPFACE_DISTANCE_METRIC', 'euclidean_l2')
    app.config['DEEPFACE_RECOGNITION_THRESHOLD'] = getattr(server_config, 'DEEPFACE_RECOGNITION_THRESHOLD', 0.80)
    app.config['ALIGNMENT_DETECTOR_BACKEND'] = getattr(server_config, 'ALIGNMENT_DETECTOR_BACKEND', 'mtcnn')
    # -----------------------------------------------

    supabase_url = server_config.SUPABASE_URL
    supabase_key = server_config.SUPABASE_KEY
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL or SUPABASE_KEY not found.")
    app.supabase_client = create_client(supabase_url, supabase_key)
    print("LOG: Supabase client initialized.")
    
    # --- ĐỊNH DANH THIẾT BỊ ---
    global CURRENT_DEVICE_UUID
    try:
        device_name = getattr(server_config, 'CLIENT_ID', 'Unknown_Device')
        CURRENT_DEVICE_UUID = get_or_create_device(app.supabase_client, device_name)
        if CURRENT_DEVICE_UUID:
            print(f"LOG: Thiết bị đã định danh thành công. ID: {CURRENT_DEVICE_UUID}")
        else:
            print("CẢNH BÁO: Không thể định danh thiết bị (Có thể do lỗi mạng).")
    except Exception as e:
        print(f"LỖI KHỞI TẠO THIẾT BỊ: {e}")
    # ---------------------------

    with app.app_context():
        # Khởi tạo FaceDetector (Lưu ý: FaceDetector có thể cần cập nhật để dùng MTCNN nếu bạn muốn detect bằng MTCNN thay vì YOLO, 
        # nhưng thường ta dùng YOLO để detect cho nhanh, rồi dùng MTCNN để align bên trong FaceEncoder/FaceMatcher)
        app.face_detector = FaceDetector()
        
        # Khởi tạo FaceEncoder & Matcher
        app.face_encoder = FaceEncoder()
        app.face_matcher = FaceMatcher(app.supabase_client) 
        app.face_matcher.load_known_faces() 
        
    print(f"LOG: AI Core initialized. Model: {app.config['DEEPFACE_MODEL_NAME']} | Backend: {app.config['ALIGNMENT_DETECTOR_BACKEND']}")

    app.register_blueprint(face_management_bp, url_prefix='/face_management')
    app.register_blueprint(attendance_bp, url_prefix='/attendance')
    
    return app


# 2. LUỒNG HEARTBEAT
# --- LUỒNG HEARTBEAT (ĐÃ FIX GIỜ TRỰC TIẾP) ---
def heartbeat_thread(app, state):
    print("LOG: Heartbeat service started (Direct Fix Mode).")
    
    while True:
        try:
            # Lấy dữ liệu từ biến chung
            uuid = state.get("uuid")
            is_alive = state.get("is_alive", False)

            if uuid and is_alive:
                # --- LOGIC TÍNH GIỜ VN THỦ CÔNG (HARDCODE) ---
                # 1. Lấy giờ UTC gốc
                utc_now = datetime.utcnow()
                # 2. Cộng 7 tiếng để ra giờ VN chuẩn
                vn_now = utc_now + timedelta(hours=7)
                # 3. Tạo chuỗi String (Bắt buộc dùng String để DB không trừ giờ)
                time_str = vn_now.strftime('%Y-%m-%d %H:%M:%S.%f')
                
                # print(f"DEBUG: Đang gửi giờ VN lên DB: {time_str}")

                # --- UPDATE TRỰC TIẾP (Không qua db_helpers) ---
                app.supabase_client.table('thiet_bi').update({
                    'ngay_cap_nhat': time_str, 
                    'trang_thai': 'Hoạt động' 
                }).eq('id', uuid).execute()
                
                print(f"LOG: Heartbeat OK -> {uuid} at {time_str}") 
            else:
                # Debug (có thể comment lại cho đỡ rối)
                # print(f"Skip Heartbeat. UUID={uuid}, Alive={is_alive}")
                pass
                
        except Exception as e:
            print(f"WARN: Heartbeat error: {e}")
            
        eventlet.sleep(30)


# 3. HÀM XỬ LÝ LUỒNG CAMERA (ĐÃ TỐI ƯU LOGIC COOLDOWN & CHỐNG HEVC ERROR)
# --- COPY VÀO main_server.py (Thay thế hàm cũ) ---
def camera_processing_thread(app, state):
    global CLIENT_VOTING_QUEUE, last_attendance_records

    # Bộ nhớ tạm để lưu các cảnh báo (Tránh spam log "Không có lịch")
    last_warning_records = {} 
    
    vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    gc_counter = 0 
    
    with app.app_context():
        # --- [HEARTBEAT SETUP] ---
        device_uuid = get_or_create_device(app.supabase_client, "Camera-ND1")
        state["uuid"] = device_uuid
        print(f"LOG: Camera Thread đã lấy UUID thiết bị: {device_uuid}")
        
        face_detector = app.face_detector
        face_encoder = app.face_encoder
        face_matcher = app.face_matcher
        config = app.config
        
        print(f"LOG: Server đang chạy CameraStream...")
        
        # Khởi tạo Stream Camera
        stream = CameraStream(config['RTSP_URL_LOW_RES'], shared_state=state)
        
        print("LOG: Server AI đang chờ khởi động...")
        eventlet.sleep(2)
        print("LOG: Server AI đã sẵn sàng xử lý.")
        
        read_interval = 1 / config['CAMERA_FPS']
        client_id = "Camera-ND1"
        
        while True:
            start_time = time.time()
            
            # Dọn rác RAM định kỳ
            gc_counter += 1
            if gc_counter % 200 == 0: 
                gc.collect()
                gc_counter = 0

            # Đọc Camera
            ret, frame_large = stream.read()
            
            if ret and not state["is_alive"]: state["is_alive"] = True

            # 1. XỬ LÝ MẤT KẾT NỐI CAMERA
            if not ret or frame_large is None:
                if state["is_alive"]: state["is_alive"] = False
                
                lost_seconds = time.time() - stream.last_read_time
                if lost_seconds > 8.0:
                    if int(lost_seconds) % 5 == 0:
                        print(f"Server: Mất kết nối với Camera. (Trong: {lost_seconds:.1f}s)")
                    socketio.emit('attendance_result', {
                        "status": "warning", 
                        "message": "Đang kết nối lại Camera...",
                        "student_name": "OFFLINE"
                    })
                eventlet.sleep(0.5) 
                continue
            
            # 2. LỌC ẢNH LỖI/MỜ
            try:
                gray_check = cv2.cvtColor(frame_large, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray_check, cv2.CV_64F).var()
                if blur_score < 40: continue 
            except: continue

            # --- AI PIPELINE ---
            frame = cv2.resize(frame_large, (640, 480))

            (h, w) = frame.shape[:2]
            roi = config['ROI_CONFIG']
            frame_roi = frame[int(h*roi['Y_START_PERCENT']):int(h*roi['Y_END_PERCENT']), 
                              int(w*roi['X_START_PERCENT']):int(w*roi['X_END_PERCENT'])]
            
            # Detect & Encode & Match
            face_boxes_roi = face_detector.detect_faces(frame_roi)
            
            matched_user_id = None 
            matched_person = None 
            bbox_roi = None 

            if face_boxes_roi:
                matched_user_id = "UNKNOWN"
                bbox_roi = face_boxes_roi[0]
                x, y, x2, y2 = bbox_roi 
                face_crop = frame_roi[y:y2, x:x2] 
                
                face_encoding = face_encoder.encode_face(face_crop)

                if face_encoding is not None:
                    matched_person, confidence = face_matcher.match_face(face_encoding)
                    matched_user_id = matched_person['user_id'] if matched_person else "UNKNOWN"

            # --- LOGIC QUYẾT ĐỊNH (VOTING THÔNG MINH) ---
            with VOTING_LOCK:
                # Khởi tạo hàng đợi nếu chưa có
                if client_id not in CLIENT_VOTING_QUEUE:
                    CLIENT_VOTING_QUEUE[client_id] = deque(maxlen=config['VOTING_FRAMES'])
                
                # 1. Nếu không có mặt -> Xóa buffer ngay (Reset)
                if matched_user_id is None:
                    CLIENT_VOTING_QUEUE[client_id].clear()
                    eventlet.sleep(0.01)
                    continue

                # 2. Thêm phiếu bầu mới vào hàng đợi
                CLIENT_VOTING_QUEUE[client_id].append(matched_user_id)
                
                # --- PHÂN TÍCH HÀNG ĐỢI ---
                queue_len = len(CLIENT_VOTING_QUEUE[client_id])
                
                # Đếm số lượng phiếu là "UNKNOWN"
                unknown_count = CLIENT_VOTING_QUEUE[client_id].count("UNKNOWN")
                
                # Tìm ứng viên tiềm năng (Người được vote nhiều nhất trừ Unknown)
                from collections import Counter
                valid_votes = [uid for uid in CLIENT_VOTING_QUEUE[client_id] if uid != "UNKNOWN"]
                
                top_candidate_id = None
                candidate_votes = 0
                
                if valid_votes:
                    counts = Counter(valid_votes)
                    # Lấy người có số phiếu cao nhất
                    top_candidate_id, candidate_votes = counts.most_common(1)[0]

                result_to_push = None
                should_clear_buffer = False

                # === [QUAN TRỌNG] LUẬT QUYẾT ĐỊNH ===

                # TRƯỜNG HỢP 1: NGƯỜI LẠ (Unknown chiếm đa số)
                # Nếu Unknown chiếm hơn 40% hàng đợi -> Chặn ngay lập tức
                if queue_len >= 5 and (unknown_count / queue_len) > 0.4:
                    
                    # Chỉ hiển thị cảnh báo nếu Unknown thực sự áp đảo (> 70%) để đỡ nháy
                    if (unknown_count / queue_len) > 0.7:
                        result_to_push = {
                            "status": "warning", 
                            "message": "Người lạ / Chưa đăng ký", 
                            "student_name": "UNKNOWN"
                        }
                    # Lưu ý: Không xóa buffer ngay để hệ thống tiếp tục theo dõi ổn định
                
                # TRƯỜNG HỢP 2: NGƯỜI QUEN (Đủ phiếu bầu & Không bị Unknown lấn át)
                elif top_candidate_id and candidate_votes >= config['VOTING_THRESHOLD']:
                    
                    user_id = top_candidate_id
                    # Lấy tên hiển thị
                    student_name = matched_person['full_name'] if (matched_person and matched_person['user_id'] == user_id) else "Đang lấy tên..."
                    
                    now = datetime.now(vn_tz)
                    should_clear_buffer = True # Đánh dấu để xóa buffer sau khi gửi socket

                    # --- [LOGIC XỬ LÝ DATABASE & COOLDOWN] ---
                    last_success = last_attendance_records.get(user_id)
                    
                    # A. Check Cooldown
                    if last_success and (now - last_success).total_seconds() < config['ATTENDANCE_COOLDOWN_SEC']:
                        result_to_push = {
                            "status": "cooldown", "message": "Bạn đã điểm danh rồi", "student_name": student_name
                        }
                    
                    # B. Check Warning (Warning Cooldown)
                    elif user_id in last_warning_records and (now - last_warning_records[user_id]).total_seconds() < 10:
                        result_to_push = {
                            "status": "warning", "message": "Không tìm thấy lịch học!", "student_name": student_name
                        }
                    
                    # C. Điểm danh thành công (Ghi DB)
                    else:
                        try:
                            # Check DB xem có lịch không
                            active_schedule = get_active_schedule_for_student(app.supabase_client, user_id)
                            
                            if active_schedule:
                                # --- CÓ LỊCH HỌC -> ĐIỂM DANH ---
                                tkb_id = active_schedule['id']
                                status = determine_status(active_schedule['gio_bat_dau'])
                                
                                # Upload ảnh minh chứng
                                evidence_url = None
                                try:
                                    if bbox_roi is not None:
                                        evidence_img = frame_roi.copy()
                                        ex, ey, ex2, ey2 = bbox_roi 
                                        cv2.rectangle(evidence_img, (ex, ey), (ex2, ey2), (0, 255, 0), 2)
                                        _, buffer = cv2.imencode(".jpg", evidence_img)
                                        path = f"evidence/{user_id}_{int(time.time())}.jpg"
                                        app.supabase_client.storage.from_("rollcall_thumbnails").upload(
                                            path=path, file=buffer.tobytes(), file_options={"content-type": "image/jpeg"}
                                        )
                                        evidence_url = app.supabase_client.storage.from_("rollcall_thumbnails").get_public_url(path)
                                except: pass

                                # Upsert vào bảng diem_danh
                                app.supabase_client.table("diem_danh").upsert({
                                    "sinh_vien_id": user_id, "thoi_khoa_bieu_id": tkb_id, "status": status,
                                    "is_manual": False, "anh_nhan_dien": evidence_url, "thiet_bi_id": state["uuid"]
                                }, on_conflict="sinh_vien_id, thoi_khoa_bieu_id").execute()
                                
                                # Cập nhật trạng thái thành công
                                last_attendance_records[user_id] = now
                                if user_id in last_warning_records: del last_warning_records[user_id]
                                
                                result_to_push = {
                                    "status": "attended",
                                    "message": f"Điểm danh thành công ({status})",
                                    "student_name": student_name,
                                    "thumbnail": evidence_url 
                                }
                                print(f"LOG: >>> SUCCESS: {student_name} <<<")
                            
                            else:
                                # --- KHÔNG CÓ LỊCH HỌC ---
                                result_to_push = {
                                    "status": "warning", "message": "Không tìm thấy lịch học!", "student_name": student_name
                                }
                                last_warning_records[user_id] = now

                        except Exception as db_err:
                            # --- CHẾ ĐỘ OFFLINE ---
                            print(f"LỖI DB/MẠNG: {db_err}. Chuyển sang lưu Offline.")
                            try:
                                if not os.path.exists("server/temp_evidence"): os.makedirs("server/temp_evidence")
                                local_filename = f"server/temp_evidence/{user_id}_{int(time.time())}.jpg"
                                cv2.imwrite(local_filename, frame_roi) # Lưu ảnh tạm

                                offline_data = {
                                    "sinh_vien_id": user_id, "timestamp": now.isoformat(),
                                    "local_image_path": local_filename, "student_name": student_name
                                }
                                save_offline_record(offline_data)
                                
                                result_to_push = {
                                    "status": "offline_saved", "message": "Mất mạng - Đã lưu Offline",
                                    "student_name": student_name
                                }
                                last_attendance_records[user_id] = now
                            except Exception as e2: print(f"Lỗi ghi offline: {e2}")

                # TRƯỜNG HỢP 3: ĐANG PHÂN TÍCH (Chưa đủ phiếu)
                else:
                    if queue_len > 2:
                        percent = int((candidate_votes / config['VOTING_FRAMES']) * 100)
                        result_to_push = {
                            "status": "voting_in_progress", 
                            "message": f"Đang xác thực... ({percent}%)", 
                            "student_name": "ĐANG TÌM..." 
                        }

            # --- GỬI SOCKET & XỬ LÝ DELAY ---
            if result_to_push:
                socketio.emit('attendance_result', result_to_push)
                
                # Nếu đã CHỐT KẾT QUẢ -> Xóa buffer & Ngủ
                if should_clear_buffer:
                    with VOTING_LOCK:
                        CLIENT_VOTING_QUEUE[client_id].clear()
                    
                    # Delay để người dùng kịp đọc thông báo
                    delay_time = 3.0 if result_to_push['status'] == 'attended' else 1.5
                    eventlet.sleep(delay_time)
                    continue

            # Điều chỉnh FPS
            elapsed_time = time.time() - start_time
            sleep_duration = read_interval - elapsed_time
            eventlet.sleep(sleep_duration if sleep_duration > 0 else 0.01)
    
    stream.stop()


# 4. LUỒNG ĐỒNG BỘ DỮ LIỆU DATABASE KHI OFFLINE
def sync_offline_data_thread(app):
    """
    Luồng chạy ngầm: Kiểm tra file offline_queue.json và đẩy lên Server khi có mạng.
    """
    print("LOG: Offline Sync Service started.")
    
    while True:
        eventlet.sleep(30) # Chạy mỗi 30 giây
        
        # 1. Lấy dữ liệu từ kho
        queue = read_and_clear_queue()
        if not queue:
            continue # Kho rỗng, ngủ tiếp
            
        print(f"SYNC: Phát hiện {len(queue)} bản ghi offline. Đang đồng bộ...")
        
        failed_records = []
        
        with app.app_context():
            vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
            
            for record in queue:
                try:
                    # Parse lại thời gian từ chuỗi lưu trữ
                    timestamp_str = record['timestamp']
                    # Chuyển string -> datetime
                    check_time = datetime.fromisoformat(timestamp_str)
                    if check_time.tzinfo is None:
                        check_time = vn_tz.localize(check_time)
                    
                    user_id = record['sinh_vien_id']
                    local_img_path = record['local_image_path']
                    
                    # 2. Kiểm tra lại lịch học (Lazy Validation)
                    active_schedule = get_active_schedule_for_student(app.supabase_client, user_id, check_time)
                    
                    if active_schedule:
                        # Có lịch -> Xử lý như bình thường
                        tkb_id = active_schedule['id']
                        start_time_str = active_schedule['gio_bat_dau']
                        status = determine_status(start_time_str) # (Cần sửa hàm này để nhận tham số time nếu cần, hoặc tạm chấp nhận logic hiện tại)
                        
                        # Upload Ảnh từ file cục bộ
                        evidence_url = None
                        if os.path.exists(local_img_path):
                            try:
                                with open(local_img_path, "rb") as f:
                                    file_bytes = f.read()
                                file_path_cloud = f"evidence/{user_id}_{int(time.time())}_offline.jpg"
                                app.supabase_client.storage.from_("rollcall_thumbnails").upload(
                                    path=file_path_cloud, file=file_bytes, file_options={"content-type": "image/jpeg"}
                                )
                                evidence_url = app.supabase_client.storage.from_("rollcall_thumbnails").get_public_url(file_path_cloud)
                            except Exception as up_err:
                                print(f"Lỗi upload ảnh offline: {up_err}")
                        
                        # Ghi DB
                        attendance_data = {
                            "sinh_vien_id": user_id,
                            "thoi_khoa_bieu_id": tkb_id,
                            "status": status,
                            "is_manual": False,
                            "anh_nhan_dien": evidence_url,
                            "thiet_bi_id": CURRENT_DEVICE_UUID,
                            "thoi_gian_diem_danh": check_time.isoformat() # Quan trọng: Dùng giờ cũ
                        }
                        app.supabase_client.table("diem_danh").upsert(
                            attendance_data, on_conflict="sinh_vien_id, thoi_khoa_bieu_id"
                        ).execute()
                        
                        print(f"SYNC SUCCESS: {record['student_name']} (Time: {timestamp_str})")
                        
                        # Xóa ảnh cục bộ sau khi thành công
                        if os.path.exists(local_img_path):
                            os.remove(local_img_path)
                            
                    else:
                        print(f"SYNC IGNORE: {record['student_name']} đi nhầm giờ lúc {timestamp_str}. Xóa bỏ.")
                        if os.path.exists(local_img_path):
                            os.remove(local_img_path)

                except Exception as e:
                    print(f"SYNC FAILED (Mạng vẫn lỗi?): {e}")
                    failed_records.append(record) # Đưa vào danh sách thất bại
        
        # Nếu có bản ghi thất bại (do vẫn mất mạng), trả lại vào kho
        if failed_records:
            print(f"SYNC: Trả lại {len(failed_records)} bản ghi vào hàng đợi.")
            restore_queue(failed_records)




# 5. KHỞI ĐỘNG SERVER
@app.route('/')
def index():
    return "Edge Server (Optimized Logic) is running!"

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "success", "message": "Server is alive!"})

if __name__ == '__main__':
    app = create_app()
    
    # TẠO BIẾN CHIA SẺ (Quan trọng nhất)
    # Biến này sẽ được truyền đi khắp nơi
    shared_device_state = {
        "uuid": None,
        "is_alive": False
    }
    
    print("LOG: Starting Camera Thread...")
    # Truyền shared_device_state vào hàm camera
    socketio.start_background_task(camera_processing_thread, app, shared_device_state)

    print("LOG: Starting Device Heartbeat...")
    # Truyền shared_device_state vào hàm heartbeat
    socketio.start_background_task(heartbeat_thread, app, shared_device_state)
    
    print(f"LOG: Starting Flask-SocketIO server...")
    socketio.run(
        app, 
        host=server_config.SERVER_HOST, 
        port=server_config.SERVER_PORT, 
        debug=True, 
        use_reloader=False, 
        allow_unsafe_werkzeug=True 
    )