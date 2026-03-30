import time
import cv2
import numpy as np
import json
import requests
from flask import Blueprint, request, jsonify, current_app
from flasgger import swag_from
from sklearn.preprocessing import normalize
import config as server_config

# [CẤU HÌNH] Không dùng Data Augmentation nữa vì đã có 3 ảnh thật
FaceAugmentor = None 

face_management_bp = Blueprint('face_management', __name__)

# ==============================================================================
# HÀM BỔ TRỢ (HELPER)
# ==============================================================================
def parse_image_urls(source_data):
    """
    Hàm xử lý thông minh:
    - Nếu input là List (do DB là JSONB): Trả về ngay.
    - Nếu input là String (do DB là Text): Parse JSON string.
    """
    if not source_data:
        return None, []
    
    # TRƯỜNG HỢP 1: Dữ liệu đã là List (do Supabase tự convert JSONB)
    if isinstance(source_data, list):
        if len(source_data) > 0:
            return source_data[0], source_data # (Ảnh đầu, Cả danh sách)
        return None, []

    # TRƯỜNG HỢP 2: Dữ liệu là String (Legacy hoặc Text column)
    if isinstance(source_data, str):
        try:
            if source_data.strip().startswith('['):
                url_list = json.loads(source_data)
                if isinstance(url_list, list) and len(url_list) > 0:
                    return url_list[0], url_list
                return None, []
            else:
                return source_data, [source_data]
        except:
            return source_data, [source_data]
            
    return None, []

# ==============================================================================
# 1. ĐĂNG KÝ KHUÔN MẶT (MULTI-IMAGE)
# ==============================================================================
@face_management_bp.route('/register_face', methods=['POST'])
@swag_from({
    'tags': ['Face Management (Web Backend)'],
    'description': 'Upload 3 ảnh (Thẳng, Trái, Phải) kèm Mã Sinh Viên. Có kiểm tra trùng lặp.',
    'consumes': ['multipart/form-data'],
    'parameters': [
        {'name': 'sinh_vien_id', 'in': 'formData', 'type': 'string', 'required': True, 'description': 'Mã Sinh Viên'},
        {'name': 'face_images', 'in': 'formData', 'type': 'file', 'required': True, 'description': 'Chọn 3 file ảnh'}
    ],
    'responses': {
        200: {'description': 'Thành công'}, 
        400: {'description': 'Lỗi dữ liệu'},
        409: {'description': 'Xung đột: Ảnh đã thuộc về người khác'}
    }
})
def register_face():
    try:
        # --- KHỞI TẠO ---
        ma_sv_input = request.form.get('sinh_vien_id')
        files = request.files.getlist('face_images') 

        if not ma_sv_input:
            return jsonify({"message": "Thiếu mã sinh viên"}), 400
            
        # Fallback: Nếu client gửi nhầm key 'face_image' (số ít)
        if not files or len(files) == 0:
            single = request.files.get('face_image')
            if single: files = [single]
            else: return jsonify({"message": "Chưa chọn ảnh nào"}), 400

        supabase = current_app.supabase_client
        BUCKET_NAME = "face_registration" 

        # Lấy các module AI từ app context
        face_detector = current_app.face_detector
        face_encoder = current_app.face_encoder
        face_matcher = current_app.face_matcher

        print("LOG: Reloading faces from DB before registration check...")
        face_matcher.load_known_faces()

        # --- BƯỚC 1: CHECK SINH VIÊN ---
        student_res = supabase.table('sinh_vien').select('id, ho_ten').eq('ma_sv', ma_sv_input).execute()
        if not student_res.data:
            return jsonify({"message": f"Không tìm thấy SV: {ma_sv_input}"}), 404
        
        real_student_uuid = student_res.data[0]['id']
        student_name = student_res.data[0]['ho_ten']

        # --- BƯỚC 2: UPLOAD TỪNG ẢNH & CHECK TRÙNG ---
        uploaded_urls = []
        storage_paths = []
        
        print(f"LOG: Nhận {len(files)} ảnh từ {student_name}. Đang xử lý...")

        for idx, file in enumerate(files):
            if file.filename == '': continue
            
            # Đọc bytes
            file_bytes = file.read()
            
            # Convert sang ảnh OpenCV để xử lý AI
            nparr = np.frombuffer(file_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None: 
                print(f"WARN: Ảnh {idx} bị lỗi format.")
                continue

            # ==================================================================
            # [TÍNH NĂNG QUAN TRỌNG]: KIỂM TRA TRÙNG LẶP KHUÔN MẶT
            # ==================================================================
            try:
                # 1. Detect khuôn mặt trong ảnh upload
                # (Lưu ý: Dùng detector của server để tìm box)
                face_boxes = face_detector.detect_faces(frame)
                
                if face_boxes:
                    # Lấy mặt to nhất
                    bbox = face_boxes[0]
                    x, y, x2, y2 = bbox
                    face_crop = frame[y:y2, x:x2]

                    # 2. Encode ra vector
                    vector = face_encoder.encode_face(face_crop)

                    if vector is not None:
                        # 3. So khớp với toàn bộ DB
                        matched_person, confidence = face_matcher.match_face(vector)

                        # 4. Nếu khớp ai đó
                        if matched_person:
                            existing_uid = matched_person['user_id']
                            existing_name = matched_person['full_name']

                            # Nếu khớp với NGƯỜI KHÁC (Không phải chính mình) -> CHẶN
                            if existing_uid != real_student_uuid:
                                print(f"ALERT: Phát hiện trùng lặp! Ảnh upload giống {existing_name}")
                                return jsonify({
                                    "message": f"Ảnh thứ {idx+1} đã được đăng ký bởi sinh viên: {existing_name}!",
                                    "conflict_with": existing_name
                                }), 409
                            else:
                                print(f"LOG: Ảnh khớp với chính chủ ({student_name}) -> OK (Update/Re-register)")
                else:
                    # Tùy chọn: Nếu không tìm thấy mặt thì có cho upload không? 
                    # Ở đây ta tạm cho qua, hoặc có thể return lỗi "Ảnh không rõ mặt"
                    print(f"WARN: Không tìm thấy mặt trong ảnh {idx}")

            except Exception as ai_err:
                print(f"Lỗi kiểm tra AI ảnh {idx}: {ai_err}")
                # Không return lỗi ở đây để tránh việc lỗi AI làm chết luồng upload, 
                # nhưng nếu bạn muốn chặt chẽ thì có thể return.
            
            # ==================================================================
            # KẾT THÚC KIỂM TRA -> TIẾN HÀNH UPLOAD
            # ==================================================================

            file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
            file_path_storage = f"pending/{real_student_uuid}_{int(time.time())}_{idx}.{file_ext}"
            
            # Reset pointer về đầu file sau khi đọc (Bắt buộc)
            file.seek(0) 
            
            try:
                supabase.storage.from_(BUCKET_NAME).upload(
                    path=file_path_storage, 
                    file=file_bytes, 
                    file_options={"content-type": "image/jpeg"}
                )
                public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_path_storage)
                
                uploaded_urls.append(public_url)
                storage_paths.append(file_path_storage)
            except Exception as up_err:
                print(f"Lỗi upload ảnh {idx}: {up_err}")
                continue

        if len(uploaded_urls) == 0:
            return jsonify({"message": "Lỗi upload ảnh (Không ảnh nào hợp lệ hoặc lỗi Storage)"}), 500

        # --- BƯỚC 3: LƯU DB (JSON LIST) ---
        request_data = {
            "sinh_vien_id": real_student_uuid,
            "anh_nguon": json.dumps(uploaded_urls),          
            "image_path_storage": json.dumps(storage_paths), 
            "trang_thai_duyet": "pending",
            "ngay_tao": "now()"
        }

        res = supabase.table('face_register_requests').insert(request_data).execute()

        if res.data:
            return jsonify({
                "message": f"Đã gửi {len(uploaded_urls)} ảnh chờ duyệt.",
                "preview_url": uploaded_urls[0]
            }), 200
        else:
            return jsonify({"message": "Lỗi lưu Database"}), 500

    except Exception as e:
        print(f"REGISTER ERROR: {e}")
        return jsonify({"message": f"Lỗi Server: {str(e)}"}), 500


# ==============================================================================
# 2. LẤY DANH SÁCH CHỜ DUYỆT (PENDING)
# ==============================================================================
@face_management_bp.route('/pending', methods=['GET'])
@swag_from({
    'tags': ['Face Management (Web Backend)'],
    'description': 'Lấy danh sách SV chờ duyệt.',
    'parameters': [
        {'name': 'class_id', 'in': 'query', 'type': 'string', 'required': False}
    ],
    'responses': {200: {'description': 'Thành công'}}
})
def get_pending_faces():
    try:
        class_id_filter = request.args.get('class_id')
        supabase = current_app.supabase_client
        
        # 1. LẤY DỮ LIỆU (Đã sửa query)
        response = supabase.table('face_register_requests')\
            .select('id, anh_nguon, ngay_tao, sinh_vien(id, ma_sv, ho_ten, lop_hoc(ten_lop))')\
            .eq('trang_thai_duyet', 'pending')\
            .order('ngay_tao', desc=True)\
            .execute()
            
        if not response.data:
            return jsonify([]), 200

        # 2. GOM NHÓM THEO SINH VIÊN
        unique_students = {}
        
        for item in response.data:
            sv_info = item.get('sinh_vien', {})
            if not sv_info: continue 
            
            # Lấy thông tin lớp
            lop_obj = sv_info.get('lop_hoc')
            ten_lop = lop_obj.get('ten_lop') if lop_obj else "Chưa xếp lớp"
            
            # QUAN TRỌNG: Lấy UUID của sinh viên
            student_uuid = sv_info.get('id') 
            if not student_uuid: continue # Bỏ qua nếu không có ID

            ma_sv = sv_info.get('ma_sv')
            
            # Filter lớp
            if class_id_filter and str(ten_lop) != str(class_id_filter):
                continue
            
            # Logic gom nhóm: 1 Sinh viên có thể có nhiều request
            if ma_sv not in unique_students:
                unique_students[ma_sv] = {
                    "student_id": student_uuid,  # <--- TRẢ VỀ CHÍNH XÁC KEY NÀY
                    "ma_sv": ma_sv,
                    "ho_ten": sv_info.get('ho_ten', 'Unknown'),
                    "lop": ten_lop,
                    "image_url": item['anh_nguon'], # Ảnh đại diện
                    "ngay_tao": item.get('ngay_tao', 'N/A'),
                    "request_ids": [item['id']],    # Danh sách các request con
                    "so_luong_anh": 1
                }
            else:
                unique_students[ma_sv]["so_luong_anh"] += 1
                unique_students[ma_sv]["request_ids"].append(item['id'])

        final_list = list(unique_students.values())
        return jsonify(final_list), 200

    except Exception as e:
        print(f"Lỗi pending: {e}")
        return jsonify({"message": f"Lỗi Server: {str(e)}"}), 500


# 3. LẤY DANH SÁCH ĐÃ DUYỆT (BẢN FIX LIST/DICT & DEBUG DATA)
@face_management_bp.route('/faces', methods=['GET'])
@swag_from({
    'tags': ['Face Management (Web Backend)'],
    'description': 'Lấy danh sách đã duyệt.',
    'parameters': [{'name': 'class_name', 'in': 'query', 'type': 'string', 'required': False}],
    'responses': {200: {'description': 'OK'}}
})
def get_approved_faces():
    try:
        class_name = request.args.get('class_name')
        
        fk_name = "face_encodings_sinh_vien_id_fkey"
        
        select_query = f'id, ngay_tao, anh_nguon, trang_thai_duyet, sinh_vien!{fk_name}!inner(ho_ten, ma_sv, lop_hoc(ten_lop))'
        
        query = current_app.supabase_client.table('face_encodings')\
            .select(select_query)\
            .eq('trang_thai_duyet', True)

        if class_name:
            query = query.eq('sinh_vien.lop_hoc.ten_lop', class_name)
            
        res = query.order('ngay_tao', desc=True).execute()
        
        # [DEBUG] In ra nội dung DATA thực tế của dòng đầu tiên
        if len(res.data) > 0:
            print(f"LOG REAL DATA (Row 0): {res.data[0]}") 
        
        result_list = []
        for item in res.data:
            # 1. Lấy dữ liệu sinh viên
            sv = item.get('sinh_vien')
            
            # [FIX QUAN TRỌNG]: Kiểm tra nếu sv là LIST (mảng) thay vì Dict
            if isinstance(sv, list):
                if len(sv) > 0:
                    sv = sv[0] # Lấy phần tử đầu tiên
                else:
                    sv = {}
            elif sv is None:
                sv = {}
            
            # 2. Lấy thông tin lớp (Cũng kiểm tra List/Dict cho chắc ăn)
            lop_data = sv.get('lop_hoc')
            if isinstance(lop_data, list):
                lop_data = lop_data[0] if len(lop_data) > 0 else {}
                
            if lop_data and isinstance(lop_data, dict):
                ten_lop = lop_data.get('ten_lop', "Chưa xếp lớp")
            else:
                ten_lop = "Chưa xếp lớp"
            
            # 3. Xử lý URL ảnh
            main_img, _ = parse_image_urls(item.get('anh_nguon'))

            result_list.append({
                "id": item['id'],
                "ma_sv": sv.get('ma_sv', 'Chưa xác định'),
                "ho_ten": sv.get('ho_ten', 'Unknown'),
                "lop": ten_lop,
                "image_url": main_img,
                "ngay_tao": item['ngay_tao']
            })
        
        return jsonify({
            "filter_class": class_name,
            "count": len(result_list),
            "faces": result_list
        }), 200

    except Exception as e: 
        print(f"DB Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"message": f"Lỗi DB: {str(e)}"}), 500


# ==============================================================================
# 4. DUYỆT & TẠO CENTROID 
# ==============================================================================
@face_management_bp.route('/<student_id>/approve', methods=['POST'])
@swag_from({
    'tags': ['Face Management (Web Backend)'],
    'description': 'Duyệt 3 ảnh -> Tính Vector Trung Bình -> Lưu 1 dòng DB.',
    'parameters': [
        {'name': 'student_id', 'in': 'path', 'type': 'string', 'required': True}
    ],
    'responses': {200: {'description': 'Thành công'}}
})
def approve_face(student_id):
    try:
        supabase = current_app.supabase_client
        face_encoder = current_app.face_encoder
        
        # 1. Tìm Request Pending
        pending_res = supabase.table('face_register_requests')\
            .select('*')\
            .eq('sinh_vien_id', student_id)\
            .eq('trang_thai_duyet', 'pending')\
            .execute()
            
        if not pending_res.data:
            return jsonify({"message": "Không tìm thấy yêu cầu chờ duyệt."}), 404
            
        pending_item = pending_res.data[0]
        
        # Lấy danh sách URL ảnh
        _, image_urls = parse_image_urls(pending_item['anh_nguon'])
        
        if not image_urls:
            return jsonify({"message": "Dữ liệu ảnh bị lỗi."}), 400

        print(f"LOG: Đang duyệt {len(image_urls)} ảnh cho SV {student_id}...")
        
        vectors = []
        
        # 2. Tải và Encode từng ảnh
        for url in image_urls:
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        # Encode (đã bao gồm L2 norm cho từng vector)
                        vec = face_encoder.encode_face(img)
                        if vec is not None:
                            vectors.append(vec)
            except Exception as dl_err:
                print(f"WARN: Không tải được ảnh {url}: {dl_err}")

        if len(vectors) == 0:
            return jsonify({"message": "Lỗi: Không tìm thấy khuôn mặt trong ảnh nào cả."}), 400

        # 3. TÍNH CENTROID (Trung bình cộng)
        mean_vector = np.mean(vectors, axis=0)
        
        # 4. CHUẨN HÓA L2 LẦN CUỐI (Cho vector trung bình)
        norm = np.linalg.norm(mean_vector)
        if norm > 0:
            final_embedding = mean_vector / norm
        else:
            final_embedding = mean_vector
            
        # 5. CẬP NHẬT DATABASE
        # 5.1 Xóa dữ liệu cũ (Reset)
        supabase.table('face_encodings').delete().eq('sinh_vien_id', student_id).execute()
        
        # 5.2 Insert Vector Mới
        data_to_insert = {
            "sinh_vien_id": student_id,
            "encoding": final_embedding.tolist(),
            "anh_nguon": image_urls, 
            "trang_thai_duyet": True,
            "ngay_tao": "now()"
        }
        
        supabase.table('face_encodings').insert(data_to_insert).execute()
        
        # 5.3 Xóa request tạm
        supabase.table('face_register_requests').delete().eq('id', pending_item['id']).execute()
        student_uuid = pending_item.get('sinh_vien_id')
        if student_uuid:
            supabase.table('sinh_vien').update({'ly_do_tu_choi_anh': None}).eq('id', student_uuid).execute()

        # 5.4 Reload Model RAM
        try:
            if hasattr(current_app, 'face_matcher'):
                current_app.face_matcher.load_known_faces()
        except: pass
        
        return jsonify({"message": f"Duyệt thành công! Đã tạo vector trung bình từ {len(vectors)} ảnh."}), 200

    except Exception as e:
        print(f"APPROVE ERROR: {e}")
        return jsonify({"message": f"Lỗi Server: {str(e)}"}), 500


# ==============================================================================
# 5. TỪ CHỐI & XÓA (CLEANUP JSON PATHS)
# ==============================================================================
@face_management_bp.route('/<student_id>/reject', methods=['POST'])
@swag_from({
    'tags': ['Face Management (Web Backend)'],
    'description': 'Từ chối ảnh.',
    'parameters': [
        {'name': 'student_id', 'in': 'path', 'type': 'string', 'required': True},
        {'name': 'reason', 'in': 'formData', 'type': 'string'}
    ],
    'responses': {200: {'description': 'Thành công'}}
})
def reject_face(student_id):
    try:
        reason = request.form.get('reason', 'Ảnh không đạt yêu cầu').strip()
        supabase = current_app.supabase_client
        BUCKET_NAME = "face_registration"

        pending_res = supabase.table('face_register_requests')\
            .select('*')\
            .eq('sinh_vien_id', student_id)\
            .execute()
            
        if pending_res.data:
            for item in pending_res.data:
                # Parse danh sách đường dẫn Storage để xóa
                raw_paths = item.get('image_path_storage')
                paths_to_delete = []
                
                if raw_paths:
                    try:
                        if raw_paths.strip().startswith('['):
                            paths_to_delete = json.loads(raw_paths)
                        else:
                            paths_to_delete = [raw_paths]
                    except:
                        paths_to_delete = [raw_paths]
                
                if paths_to_delete:
                    try:
                        supabase.storage.from_(BUCKET_NAME).remove(paths_to_delete)
                    except: pass
            
            # Xóa record
            supabase.table('face_register_requests').delete().eq('sinh_vien_id', student_id).execute()

        supabase.table('sinh_vien').update({'ly_do_tu_choi_anh': reason}).eq('id', student_id).execute()

        return jsonify({"message": "Đã từ chối và xóa ảnh."}), 200

    except Exception as e:
        return jsonify({"message": f"Lỗi: {str(e)}"}), 500


# ==============================================================================
# 6. XÓA VECTOR (DELETE FACE)
# ==============================================================================
@face_management_bp.route('/<face_encoding_id>', methods=['DELETE'])
@swag_from({
    'tags': ['Face Management (Web Backend)'],
    'description': 'Xóa vector và ảnh gốc.',
    'parameters': [
        {'name': 'face_encoding_id', 'in': 'path', 'type': 'string', 'required': True}
    ],
    'responses': {200: {'description': 'Thành công'}}
})
def delete_face(face_encoding_id):
    try:
        supabase = current_app.supabase_client
        BUCKET_NAME = "face_registration"
        
        target = supabase.table('face_encodings').select('anh_nguon').eq('id', face_encoding_id).execute()
        
        if not target.data: 
            return jsonify({"message": "Dữ liệu không tồn tại."}), 404
            
        source_data = target.data[0].get('anh_nguon')
        _, url_list = parse_image_urls(source_data) 
        
        # Xóa file trong Storage
        files_to_remove = []
        for url in url_list:
            if BUCKET_NAME in url:
                try:
                    file_path = url.split(f"/{BUCKET_NAME}/")[-1]
                    from urllib.parse import unquote
                    file_path = unquote(file_path)
                    files_to_remove.append(file_path)
                except: pass
        
        if files_to_remove:
            supabase.storage.from_(BUCKET_NAME).remove(files_to_remove)

        # Xóa DB
        supabase.table('face_encodings').delete().eq('id', face_encoding_id).execute()

        # Reload RAM
        try:
            if hasattr(current_app, 'face_matcher'):
                current_app.face_matcher.load_known_faces()
        except: pass

        return jsonify({"message": "Đã xóa dữ liệu thành công."}), 200

    except Exception as e: 
        return jsonify({"message": f"Lỗi Server: {str(e)}"}), 500
    
# ... (các import ở đầu file giữ nguyên) ...

@face_management_bp.route('/identify', methods=['POST'])
def identify_face_api():
    try:
        # 1. Nhận file ảnh
        if 'image' not in request.files:
            return jsonify({"message": "Thiếu file ảnh"}), 400
        
        file = request.files['image']
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"message": "Ảnh lỗi"}), 400

        # 2. Lấy các module AI từ biến toàn cục (đã khai báo trong main_server.py)
        face_detector = current_app.face_detector
        face_encoder = current_app.face_encoder
        face_matcher = current_app.face_matcher
        
        # 3. Phát hiện khuôn mặt (Detect)
        # Lưu ý: Hàm detect trả về list các bounding box [x, y, x2, y2]
        faces = face_detector.detect_faces(img)
        
        if not faces:
            return jsonify({"match": False, "status": "Không tìm thấy mặt"}), 200
            
        # Lấy khuôn mặt lớn nhất
        x, y, x2, y2 = faces[0]
        face_crop = img[y:y2, x:x2]
        
        # 4. Mã hóa (Encode 512d)
        vector = face_encoder.encode_face(face_crop)
        
        if vector is None:
             return jsonify({"match": False, "status": "Không thể mã hóa"}), 200
             
        # 5. So khớp (Match)
        match_info, conf = face_matcher.match_face(vector)
        
        if match_info:
            return jsonify({
                "match": True,
                "student_id": match_info['user_id'], # Trả về UUID sinh viên
                "ma_sv": match_info['ma_sv'],
                "ho_ten": match_info['full_name'],
                "confidence": conf
            }), 200
        else:
             return jsonify({"match": False, "status": "Không khớp ai"}), 200

    except Exception as e:
        print(f"IDENTIFY API ERROR: {e}")
        return jsonify({"message": str(e)}), 500