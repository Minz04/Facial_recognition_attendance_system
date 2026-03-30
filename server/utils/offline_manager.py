import json
import os
import threading

# Tên file lưu trữ dữ liệu tạm
OFFLINE_FILE = "offline_queue.json"
# Khóa an toàn để tránh xung đột luồng
FILE_LOCK = threading.Lock()

def save_offline_record(data):
    """
    Lưu một bản ghi điểm danh vào file JSON khi mất mạng.
    data gồm: {sinh_vien_id, timestamp, local_image_path, student_name}
    """
    with FILE_LOCK:
        queue = []
        # 1. Đọc dữ liệu cũ (nếu có)
        if os.path.exists(OFFLINE_FILE):
            try:
                with open(OFFLINE_FILE, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content:
                        queue = json.loads(content)
            except Exception as e:
                print(f"Lỗi đọc file offline: {e}")
                queue = [] 
        
        # 2. Thêm bản ghi mới
        queue.append(data)
        
        # 3. Ghi đè lại file
        with open(OFFLINE_FILE, 'w', encoding='utf-8') as f:
            json.dump(queue, f, ensure_ascii=False, indent=2)
            
def read_and_clear_queue():
    """
    Đọc toàn bộ hàng đợi để xử lý, sau đó xóa trắng file.
    Trả về: List các bản ghi.
    """
    with FILE_LOCK:
        if not os.path.exists(OFFLINE_FILE):
            return []
            
        try:
            with open(OFFLINE_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content: return []
                queue = json.loads(content)
            
            # Xóa nội dung file sau khi đã đọc vào RAM
            with open(OFFLINE_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)
                
            return queue
        except Exception as e:
            print(f"Lỗi đọc queue: {e}")
            return []

def restore_queue(queue):
    """
    Nếu xử lý thất bại (vẫn mất mạng), ghi lại dữ liệu vào file để lần sau xử lý tiếp.
    """
    with FILE_LOCK:
        with open(OFFLINE_FILE, 'w', encoding='utf-8') as f:
            json.dump(queue, f, ensure_ascii=False, indent=2)