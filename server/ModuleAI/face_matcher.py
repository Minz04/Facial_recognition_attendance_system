import numpy as np
from scipy.spatial.distance import cosine
import config as server_config
from supabase import Client 
import json

class FaceMatcher:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.threshold = getattr(server_config, 'DEEPFACE_RECOGNITION_THRESHOLD', 0.40)
        self.metric = getattr(server_config, 'DEEPFACE_DISTANCE_METRIC', 'euclidean_l2')
        print(f"LOG: FaceMatcher initialized. Metric: {self.metric} | Threshold: {self.threshold}")
        
        self.known_encodings = [] 
        self.load_known_faces() 

    def load_known_faces(self):
        try:
            print("LOG: Đang tải dữ liệu khuôn mặt từ DB...")
            
            fk_relationship = "face_encodings_sinh_vien_id_fkey"
            query = f"*, sinh_vien!{fk_relationship}(id, ma_sv, ho_ten)"
            
            response = (
                self.supabase.from_('face_encodings')
                .select(query)
                .eq('trang_thai_duyet', True)
                .execute()
            )
            
            self.known_encodings = []
            
            for record in response.data:
                try:
                    encoding_raw = record.get('encoding')
                    if not encoding_raw: continue
                    
                    # [XỬ LÝ ĐA DẠNG INPUT]
                    if isinstance(encoding_raw, str):
                        # Nếu là string JSON: "[-0.1, 0.2...]"
                        encoding_np = np.array(json.loads(encoding_raw), dtype=np.float64)
                    elif isinstance(encoding_raw, list):
                        # Nếu Supabase đã tự convert sang list float
                        encoding_np = np.array(encoding_raw, dtype=np.float64)
                    else:
                        continue
                    
                    sv_info = record.get('sinh_vien') or {}
                    
                    self.known_encodings.append({
                        "user_id": record.get('sinh_vien_id'),
                        "full_name": sv_info.get('ho_ten', 'Unknown'),
                        "ma_sv": sv_info.get('ma_sv', 'Unknown'),
                        "encoding": encoding_np
                    })
                except Exception as parse_err:
                    print(f"WARN: Lỗi parse data: {parse_err}")
                    continue
            
            loaded_names = [item['full_name'] for item in self.known_encodings]
            print(f"LOG: Danh sách khuôn mặt trong RAM: {loaded_names}")
            
            print(f"LOG: Đã tải xong {len(self.known_encodings)} khuôn mặt vào RAM.")
            
        except Exception as e:
            print(f"CRITICAL ERROR loading encodings: {e}")
        
    def match_face(self, unknown_encoding):
        if not self.known_encodings or unknown_encoding is None:
            return None, 0.0

        min_distance = float('inf')
        best_match = None

        for known in self.known_encodings:
            distance = 0.0
            if self.metric == 'cosine':
                distance = cosine(known['encoding'], unknown_encoding)
            else:
                diff = known['encoding'] - unknown_encoding
                distance = np.linalg.norm(diff)
            
            if distance < min_distance:
                min_distance = distance
                best_match = known

        confidence = max(0.0, (self.threshold - min_distance) / self.threshold)
        
        # [DEBUG LOG] Để bạn biết hệ thống đang so sánh
        # if min_distance < 1.0: 
        #    print(f"DEBUG: Distance {min_distance:.4f} vs Threshold {self.threshold}")

        if min_distance < self.threshold:
            print(f"LOG: MATCH FOUND! -> {best_match['full_name']} (Dist: {min_distance:.4f})")
            return best_match, confidence
        else:
            print(f"LOG: UNKNOW, nhưng khoảng cách {min_distance:.4f} > {self.threshold} (Chưa đạt)")
            return None, confidence