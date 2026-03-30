import threading
from collections import deque

# --- BIẾN TOÀN CỤC BẮT BUỘC CHO VOTING & COOLDOWN ---
# {client_id: deque(['user_id_1', 'user_id_2', 'UNKNOWN', ...])}
CLIENT_VOTING_QUEUE = {} 

# KHÓA THREAD cho Voting/Queue/Cooldown
VOTING_LOCK = threading.Lock() 

# Lưu thời gian điểm danh cuối cùng cho mỗi người (phục vụ Cooldown)
last_attendance_records = {}

# UUID của thiết bị hiện tại 
CURRENT_DEVICE_UUID = None