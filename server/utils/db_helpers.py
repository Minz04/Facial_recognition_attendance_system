import pytz
from datetime import datetime, timedelta, time

# Định nghĩa Timezone (Dùng cho các hàm so sánh logic)
VN_TZ = pytz.timezone('Asia/Ho_Chi_Minh')

# --- HÀM TIỆN ÍCH THỜI GIAN (QUAN TRỌNG) ---
def get_vn_time_str():
    # 1. Lấy giờ UTC
    utc_now = datetime.utcnow()
    # 2. Cộng 7 tiếng
    vn_now = utc_now + timedelta(hours=7)
    # 3. Tạo chuỗi
    time_str = vn_now.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    # [DEBUG] In ra để xem Python tính đúng chưa
    print(f"DEBUG TIME: UTC={utc_now.strftime('%H:%M')} | VN_CALC={time_str}")
    return time_str

# --- 1. HÀM LOGIC XÁC ĐỊNH TRẠNG THÁI (ĐI TRỄ / ĐÚNG GIỜ) ---
def determine_status(start_time_input, threshold_minutes=15):
    """
    So sánh giờ hiện tại với giờ học.
    Logic thông minh: Tự động phát hiện nếu giờ DB là UTC và bù +7.
    """
    try:
        # 1. Lấy giờ hiện tại chính xác theo VN
        now = datetime.now(VN_TZ)
        
        # 2. Xử lý Input đầu vào (Parse chuỗi giờ)
        start_time_obj = None
        
        if isinstance(start_time_input, str):
            clean_str = start_time_input.split('+')[0].strip() # Bỏ đuôi timezone +00
            try:
                start_time_obj = datetime.strptime(clean_str, "%H:%M:%S").time()
            except ValueError:
                try:
                    start_time_obj = datetime.strptime(clean_str, "%H:%M").time()
                except ValueError:
                    print(f"LỖI FORMAT: Không đọc được giờ '{clean_str}'")
                    return "Đi trễ" # Input lỗi
                    
        elif isinstance(start_time_input, time):
            start_time_obj = start_time_input
        else:
            return "Đi trễ"

        # 3. Tạo DateTime đầy đủ cho buổi học
        start_dt = datetime.combine(now.date(), start_time_obj)
        start_dt = VN_TZ.localize(start_dt) 
        
        # --- [FIX LOGIC QUAN TRỌNG] TỰ ĐỘNG BÙ GIỜ NẾU LỆCH ---
        diff_hours = (now - start_dt).total_seconds() / 3600
        
        if diff_hours > 5: 
            # Ví dụ: Bây giờ 17h, DB lưu 10h -> Lệch 7 tiếng -> Đây là UTC!
            print(f"WARN: Phát hiện lệch múi giờ (Lệch {diff_hours:.1f}h). Tự động cộng thêm 7h.")
            start_dt = start_dt + timedelta(hours=7)
        # -----------------------------------------------------

        # 4. Tính ngưỡng (Giờ học + 15 phút)
        late_threshold = start_dt + timedelta(minutes=threshold_minutes)
        
        # 5. So sánh
        # print(f"LOGIC TIME: Giờ học={start_dt.strftime('%H:%M')} | Hạn chót={late_threshold.strftime('%H:%M:%S')} | Hiện tại={now.strftime('%H:%M:%S')}")
        
        if now <= late_threshold:
            return "Đúng giờ"
        else:
            return "Đi trễ"

    except Exception as e:
        print(f"CRITICAL ERROR in determine_status: {e}")
        return "Đi trễ"
    
# --- 2. HÀM LẤY LỊCH HỌC ---
def get_active_schedule_for_student(supabase_client, student_id, check_time=None):
    """
    Tìm buổi học hiện tại cho sinh viên.
    (Đã sửa: So sánh dựa trên lop_id thay vì tên lớp để tránh lỗi)
    """
    try:
        # Xử lý múi giờ
        if check_time:
            now = check_time
            if now.tzinfo is None:
                now = VN_TZ.localize(now)
        else:
            now = datetime.now(VN_TZ)
            
        current_date = now.strftime('%Y-%m-%d')
        current_time = now.strftime('%H:%M:%S')

        # BƯỚC 1: Lấy thông tin sinh viên (Lấy lop_id thay vì tên lớp)
        # .select('lop_id') chắc chắn đúng vì đây là khóa ngoại
        student_res = supabase_client.table('sinh_vien')\
            .select('nguoi_dung_id, lop_id')\
            .eq('id', student_id)\
            .execute()
        
        if not student_res.data:
            return None
        
        student_info = student_res.data[0]
        nguoi_dung_id = student_info.get('nguoi_dung_id')
        student_lop_id = student_info.get('lop_id') # Lấy UUID của lớp

        # BƯỚC 2: Tìm tất cả các Lịch học đang diễn ra
        # Cũng lấy lop_id từ lịch học
        schedules_res = supabase_client.table('thoi_khoa_bieu')\
            .select('id, gio_bat_dau, gio_ket_thuc, lop_id')\
            .eq('ngay_hoc', current_date)\
            .lte('gio_bat_dau', current_time)\
            .gte('gio_ket_thuc', current_time)\
            .execute()

        active_schedules = schedules_res.data
        if not active_schedules:
            return None 

        # BƯỚC 3: Duyệt qua các lịch học để tìm lịch phù hợp
        for schedule in active_schedules:
            tkb_id = schedule['id']
            
            # --- Ưu tiên 1: Kiểm tra xem có trong danh sách Tín chỉ (tkb_thanh_vien) không ---
            if nguoi_dung_id:
                member_check = supabase_client.table('tkb_thanh_vien')\
                    .select('id')\
                    .eq('tkb_id', tkb_id)\
                    .eq('nguoi_dung_id', nguoi_dung_id)\
                    .execute()
                
                if member_check.data:
                    return schedule

            # --- Ưu tiên 2: Kiểm tra Lớp hành chính (Dựa trên ID) ---
            schedule_lop_id = schedule.get('lop_id')
            
            # So sánh 2 UUID (Chuyển sang string cho chắc chắn)
            if student_lop_id and schedule_lop_id:
                if str(student_lop_id) == str(schedule_lop_id):
                    return schedule

        return None 

    except Exception as e:
        print(f"Lỗi lấy lịch học: {e}")
        return None

# --- 3. HÀM QUẢN LÝ THIẾT BỊ ---
def get_or_create_device(supabase, device_name):
    try:
        res = supabase.table('thiet_bi').select('id').eq('ten_thiet_bi', device_name).execute()
        
        if res.data:
            print(f"LOG: Tìm thấy thiết bị cũ: {device_name} (ID: {res.data[0]['id']})")
            return res.data[0]['id']
        else:
            print(f"LOG: Tạo thiết bị mới: {device_name}")
            now_str = get_vn_time_str()
            
            new_device = {
                "ten_thiet_bi": device_name,
                "trang_thai": "Hoạt động",       
                "vi_tri": "Chưa cập nhật",
                "ngay_cap_nhat": now_str
            }
            res_ins = supabase.table('thiet_bi').insert(new_device).execute()
            if res_ins.data:
                return res_ins.data[0]['id']
            
    except Exception as e:
        print(f"Lỗi tạo thiết bị: {e}")
        return None

# 4. HÀM CẬP NHẬT HEARTBEAT (DEBUG MODE)
def update_device_heartbeat(supabase, device_uuid):
    if not device_uuid: return
    try:
        # Lấy giờ
        now_str = get_vn_time_str()
        
        # In ra UUID đang được update để chắc chắn không update nhầm thằng khác
        print(f"DEBUG DB: Đang update cho UUID={device_uuid} với giờ={now_str}")

        data = supabase.table('thiet_bi').update({
            'ngay_cap_nhat': now_str, 
            'trang_thai': 'Hoạt động' 
        }).eq('id', device_uuid).execute()
        
        # Kiểm tra xem Database có phản hồi gì không
        if not data.data:
            print("CẢNH BÁO: Lệnh update chạy nhưng KHÔNG CÓ dòng nào thay đổi (Sai ID?)")
            
    except Exception as e:
        print(f"Lỗi Heartbeat: {e}")