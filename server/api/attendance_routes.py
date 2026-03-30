from flask import Blueprint, jsonify, request, current_app
from flasgger import swag_from
import time 
from datetime import datetime, timedelta
from urllib.parse import unquote
import pytz 

# Giả định các helper và state đã được định nghĩa đúng trong project của bạn
from server.utils.helpers import extract_path_from_url
from server.app_state import (
    CLIENT_VOTING_QUEUE,
    VOTING_LOCK,
    last_attendance_records
)

attendance_bp = Blueprint('attendance', __name__)

# ĐỊNH NGHĨA THAM SỐ SWAGGER
SCHEDULE_ID_PARAM = {'name': 'schedule_id', 'in': 'path', 'type': 'string', 'required': True, 'description': 'ID của Buổi học (Thoi_khoa_bieu_id)'}

# 1. API DEBUG (Xem trạng thái Server AI)
@attendance_bp.route('/current_state', methods=['GET'])
@swag_from({
    'tags': ['Attendance (Debug)'],
    'description': 'Lấy trạng thái Voting hiện tại trong bộ nhớ Server.',
    'responses': {200: {'description': 'OK'}}
})
def get_current_state():
    with VOTING_LOCK:
        voting_queues_list = {cid: list(q) for cid, q in CLIENT_VOTING_QUEUE.items()}
        
        cooldown_records_str = {}
        for uid, t in last_attendance_records.items():
            if isinstance(t, datetime):
                cooldown_records_str[uid] = t.strftime('%Y-%m-%d %H:%M:%S')
            else:
                cooldown_records_str[uid] = str(t)

    return jsonify({
        "voting_queues": voting_queues_list,
        "cooldown_records": cooldown_records_str
    }), 200


# 2. API BÁO CÁO CHI TIẾT (GET /report) 
@attendance_bp.route('/schedule/<string:schedule_id>/report', methods=['GET'])
@swag_from({
    'tags': ['Attendance Business (Web Backend)'],
    'summary': 'Lấy báo cáo điểm danh (Tự động tính Vắng dựa trên sĩ số lớp)',
    'parameters': [SCHEDULE_ID_PARAM], 
    'responses': {200: {'description': 'Danh sách sinh viên và trạng thái'}}
})
def get_attendance_report(schedule_id):
    try:
        supabase = current_app.supabase_client

        # 1. Lấy thông tin buổi học
        # Lấy thêm lop_id để query sinh viên chính xác
        schedule_res = supabase.table('thoi_khoa_bieu')\
            .select('*, lop_hoc(id, ten_lop)')\
            .eq('id', schedule_id)\
            .execute()
            
        if not schedule_res.data:
            return jsonify({"message": "Không tìm thấy buổi học"}), 404
        
        schedule_info = schedule_res.data[0]
        end_time_str = schedule_info.get('gio_ket_thuc', '23:59:59')
        
        # [QUAN TRỌNG] Lấy ID Lớp để tìm sinh viên
        lop_id = schedule_info.get('lop_id')
        if not lop_id:
             return jsonify({"message": "Lịch học này chưa được gán Lớp (lop_id is NULL)"}), 400

        # 2. Xác định trạng thái mặc định (Vắng hay Chưa điểm danh)
        vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        now_time = datetime.now(vn_tz).time()
        
        try:
            end_time = datetime.strptime(end_time_str, "%H:%M:%S").time()
        except:
            end_time = datetime.strptime("23:59:59", "%H:%M:%S").time()
        
        # Nếu giờ hiện tại đã qua giờ kết thúc -> Mặc định là Vắng. Ngược lại là Chưa ĐD.
        default_status = "Vắng" if now_time > end_time else "Chưa điểm danh"

        # 3. Lấy danh sách sinh viên THUỘC LỚP ĐÓ (SỬA LỖI TẠI ĐÂY)
        # Sử dụng .eq('lop_id', lop_id) thay vì lọc theo tên lớp
        # Join lop_hoc(ten_lop) để hiển thị tên lớp cho đẹp
        all_students_res = supabase.table('sinh_vien')\
            .select('id, ma_sv, ho_ten, lop_hoc(ten_lop)')\
            .eq('lop_id', lop_id)\
            .execute()
            
        all_students = all_students_res.data

        # 4. Lấy danh sách ĐÃ ĐIỂM DANH của buổi học này
        checked_in = supabase.table('diem_danh')\
            .select('sinh_vien_id, status, thoi_gian_diem_danh, anh_nhan_dien')\
            .eq('thoi_khoa_bieu_id', schedule_id)\
            .execute()
        
        attendance_map = {rec['sinh_vien_id']: rec for rec in checked_in.data}

        # 5. GHÉP DỮ LIỆU (Merge)
        report_data = []
        for sv in all_students:
            sv_id = sv['id']
            
            # [XỬ LÝ TÊN LỚP] Lấy tên lớp từ object lồng nhau
            ten_lop = "Chưa xếp lớp"
            if sv.get('lop_hoc'):
                ten_lop = sv['lop_hoc'].get('ten_lop')

            if sv_id in attendance_map:
                # Có đi học
                record = attendance_map[sv_id]
                status = record['status']
                check_in = record['thoi_gian_diem_danh']
                img = record['anh_nhan_dien']
            else:
                # Không đi học -> Gán trạng thái mặc định (Vắng/Chưa ĐD)
                status = default_status
                check_in = None
                img = None

            report_data.append({
                "sinh_vien_id": sv_id,
                "ma_sv": sv['ma_sv'],
                "ho_ten": sv['ho_ten'],
                "lop": ten_lop, # Trả về tên lớp dạng chuỗi
                "status": status,
                "thoi_gian": check_in,
                "anh_minh_chung": img
            })

        return jsonify({"report": report_data}), 200
    except Exception as e:
        print(f"Lỗi Report: {e}")
        return jsonify({"message": f"Lỗi Report: {str(e)}"}), 500


# 3. API ĐIỂM DANH THỦ CÔNG (POST /manual)
@attendance_bp.route('/manual', methods=['POST'])
@swag_from({
    'tags': ['Attendance Business (Web Backend)'],
    'description': 'Giảng viên sửa trạng thái điểm danh thủ công (UPSERT).',
    'parameters': [
        {
            'name': 'body', 'in': 'body', 'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'sinh_vien_id': {'type': 'string'},
                    'thoi_khoa_bieu_id': {'type': 'string'},
                    'status': {'type': 'string', 'enum': ['Đúng giờ', 'Đi trễ', 'Vắng']}
                }
            }
        }
    ],
    'responses': {200: {'description': 'Cập nhật thành công.'}}
})
def manual_attendance():
    try:
        data = request.get_json()
        sinh_vien_id = data.get('sinh_vien_id')
        thoi_khoa_bieu_id = data.get('thoi_khoa_bieu_id')
        status = data.get('status') 

        if not all([sinh_vien_id, thoi_khoa_bieu_id, status]):
            return jsonify({"message": "Thiếu thông tin bắt buộc."}), 400

        valid_statuses = ['Đúng giờ', 'Đi trễ', 'Vắng']
        if status not in valid_statuses:
            return jsonify({"message": f"Trạng thái sai. Chỉ nhận: {valid_statuses}"}), 400

        vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        now = datetime.now(vn_tz).replace(tzinfo=None)

        attendance_record = {
            "sinh_vien_id": sinh_vien_id,
            "thoi_khoa_bieu_id": thoi_khoa_bieu_id,
            "status": status,            
            "is_manual": True,            
            "thoi_gian_diem_danh": now.isoformat(),
            "thiet_bi_id": None 
        }

        current_app.supabase_client.table("diem_danh").upsert(
            attendance_record, on_conflict="sinh_vien_id, thoi_khoa_bieu_id"
        ).execute()

        return jsonify({"message": f"Đã cập nhật thủ công: {status}", "data": attendance_record}), 200

    except Exception as e:
        return jsonify({"message": f"Lỗi Server: {str(e)}"}), 500


# 4. API THỐNG KÊ (GET /stats)
@attendance_bp.route('/schedule/<string:schedule_id>/stats', methods=['GET'])
@swag_from({
    'tags': ['Attendance Business (Web Backend)'],
    'summary': 'Lấy số liệu thống kê của buổi học',
    'parameters': [SCHEDULE_ID_PARAM],
    'responses': {200: {'description': 'Thống kê số lượng.'}}
})
def get_attendance_stats(schedule_id):
    try:
        report_response = get_attendance_report(schedule_id)
        
        if report_response[1] != 200: 
            return report_response

        report_data = report_response[0].json['report']
        
        stats = {"tong_so": len(report_data), "dung_gio": 0, "di_tre": 0, "vang": 0, "chua_diem_danh": 0}

        for item in report_data:
            st = item['status']
            if st == 'Đúng giờ': stats['dung_gio'] += 1
            elif st == 'Đi trễ': stats['di_tre'] += 1
            elif st == 'Vắng': stats['vang'] += 1
            elif st == 'Chưa điểm danh': stats['chua_diem_danh'] += 1

        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500


# 5. API RESET BUỔI HỌC (DELETE /reset)
@attendance_bp.route('/schedule/<string:schedule_id>/reset', methods=['DELETE'])
@swag_from({
    'tags': ['Attendance Business (Web Backend)'],
    'summary': 'Xóa toàn bộ dữ liệu điểm danh VÀ ẢNH MINH CHỨNG của buổi học này',
    'parameters': [SCHEDULE_ID_PARAM],
    'responses': {200: {'description': 'Reset thành công.'}}
})
def reset_schedule_attendance(schedule_id):
    try:
        supabase = current_app.supabase_client
        bucket_name = "rollcall_thumbnails"

        # 1. Lấy danh sách ảnh cần xóa TRƯỚC KHI xóa dữ liệu
        records = supabase.table('diem_danh')\
            .select('anh_nhan_dien')\
            .eq('thoi_khoa_bieu_id', schedule_id)\
            .execute()
        
        files_to_delete = []
        for row in records.data:
            path = extract_path_from_url(row.get('anh_nhan_dien'), bucket_name)
            if path:
                files_to_delete.append(path)

        # 2. Xóa ảnh trong Storage (nếu có)
        if files_to_delete:
            supabase.storage.from_(bucket_name).remove(files_to_delete)
            print(f"LOG: Đã xóa {len(files_to_delete)} ảnh minh chứng khỏi Storage.")

        # 3. Xóa dữ liệu trong DB
        supabase.table('diem_danh')\
            .delete().eq('thoi_khoa_bieu_id', schedule_id).execute()
            
        return jsonify({"message": f"Đã reset dữ liệu và xóa {len(files_to_delete)} ảnh minh chứng."}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500


# 6. API DỌN DẸP HỆ THỐNG (TOÀN BỘ) (DELETE /cleanup)
@attendance_bp.route('/cleanup', methods=['DELETE'])
@swag_from({
    'tags': ['System Maintenance'],
    'summary': 'Xóa dữ liệu điểm danh và ảnh trong khoảng thời gian cụ thể.',
    'description': 'Xóa dữ liệu từ start_date đến end_date. Nếu không nhập start_date, sẽ xóa tất cả dữ liệu trước end_date.',
    'parameters': [
        {
            'name': 'start_date',
            'in': 'query',
            'type': 'string',
            'format': 'date',
            'description': 'Ngày bắt đầu (YYYY-MM-DD). Để trống nếu muốn xóa từ đầu.',
            'required': False
        },
        {
            'name': 'end_date',
            'in': 'query',
            'type': 'string',
            'format': 'date',
            'description': 'Ngày kết thúc (YYYY-MM-DD). Dữ liệu sẽ bị xóa tính đến cuối ngày này.',
            'required': True,
            'default': datetime.now().strftime('%Y-%m-%d')
        }
    ],
    'responses': {
        200: {'description': 'Báo cáo số lượng đã xóa.'},
        400: {'description': 'Lỗi định dạng ngày.'}
    }
})
def cleanup_system_data():
    try:
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')

        if not end_date_str:
            return jsonify({"message": "Vui lòng nhập ngày kết thúc (end_date)."}), 400

        # Validate và chuẩn hóa format ngày
        try:
            datetime.strptime(end_date_str, '%Y-%m-%d')
            if start_date_str:
                datetime.strptime(start_date_str, '%Y-%m-%d')
                
            end_date_full = f"{end_date_str}T23:59:59"
            if start_date_str:
                start_date_full = f"{start_date_str}T00:00:00"
            
        except ValueError:
            return jsonify({"message": "Sai định dạng ngày. Vui lòng dùng YYYY-MM-DD."}), 400

        supabase = current_app.supabase_client
        bucket_name = "rollcall_thumbnails"
        
        print(f"LOG: Bắt đầu dọn dẹp dữ liệu từ {start_date_str if start_date_str else 'Đầu'} đến {end_date_str}...")

        # BƯỚC 1: Lấy danh sách cần xóa để lấy ảnh
        query = supabase.table('diem_danh').select('id, anh_nhan_dien')
        
        # Điều kiện thời gian
        query = query.lte('thoi_gian_diem_danh', end_date_full)
        if start_date_str:
            query = query.gte('thoi_gian_diem_danh', start_date_full)
            
        records = query.execute()
        
        if not records.data:
            return jsonify({"message": "Không tìm thấy dữ liệu trong khoảng thời gian này."}), 200

        files_to_delete = []
        for row in records.data:
            path = extract_path_from_url(row.get('anh_nhan_dien'), bucket_name)
            if path:
                files_to_delete.append(path)

        # BƯỚC 2: Xóa ảnh Storage
        if files_to_delete:
            remove_res = supabase.storage.from_(bucket_name).remove(files_to_delete)
            print(f"LOG: Đã xóa {len(remove_res)} file ảnh.")

        # BƯỚC 3: Xóa dữ liệu trong Database
        delete_query = supabase.table('diem_danh').delete()
        delete_query = delete_query.lte('thoi_gian_diem_danh', end_date_full)
        if start_date_str:
            delete_query = delete_query.gte('thoi_gian_diem_danh', start_date_full)
            
        delete_res = delete_query.execute()
        count = len(delete_res.data)

        return jsonify({
            "message": "Dọn dẹp hoàn tất.",
            "deleted_records": count,
            "deleted_images": len(files_to_delete),
            "range": f"{start_date_str if start_date_str else 'All'} -> {end_date_str}"
        }), 200

    except Exception as e:
        return jsonify({"message": f"Lỗi dọn dẹp: {str(e)}"}), 500