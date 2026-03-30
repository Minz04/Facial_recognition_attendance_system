import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from deep_face_pipeline import FaceRecognitionPipeline


# --- CẤU HÌNH ---
CONFIG = {
    "model": "Facenet512",
    "metric": "euclidean_l2",
    "backend": "mtcnn",
    "padding": 0.15 
}

DB_PATH = "mini_database_test"
TEST_PATH = "mini_dataset_test"
YOLO_WEIGHTS = 'yolov5/runs/train/train_face_detection_v3/weights/best.pt'

# --- HÀM CHUẨN HÓA TÊN (QUAN TRỌNG NHẤT) ---
def normalize_name(name):
    """
    Biến đổi: "Brad_Pitt", "Brad Pitt", "brad pitt" -> "bradpitt"
    Để so sánh chính xác tuyệt đối.
    """
    if not name: return ""
    return name.lower().replace("_", "").replace(" ", "").strip()

def get_true_label(filename, db_names_normalized):
    # Filename: 001_Brad_Pitt.jpg
    base = os.path.splitext(filename)[0]
    if "unknown" in base.lower(): return "unknown"
    
    parts = base.split('_')
    # Lấy tên: Brad_Pitt
    raw_name = "_".join(parts[1:]).strip() if len(parts) >= 2 and parts[0].isdigit() else base.strip()
    
    # Chuẩn hóa để check trong DB
    norm_name = normalize_name(raw_name)
    
    # Kiểm tra xem tên này có trong DB không (so sánh dạng chuẩn hóa)
    if norm_name in db_names_normalized:
        return norm_name # Trả về tên đã chuẩn hóa
    return "unknown"

def run_auto_tune():
    print(f"--- BẮT ĐẦU DÒ TÌM NGƯỠNG TỐI ƯU (ĐÃ FIX LỖI SO SÁNH) ---")
    
    pipeline = FaceRecognitionPipeline(
        yolo_model_weights_path=YOLO_WEIGHTS,
        db_path=DB_PATH,
        df_model_name=CONFIG['model'],
        df_distance_metric=CONFIG['metric'],
        alignment_detector_backend=CONFIG['backend'],
        custom_threshold=100.0, # Mẹo: Threshold cực cao để luôn lấy kết quả
        use_alignment=True
    )

    # Lấy danh sách tên folder trong DB và chuẩn hóa trước
    raw_db_names = [f for f in os.listdir(DB_PATH) if os.path.isdir(os.path.join(DB_PATH, f))]
    db_names_normalized = set([normalize_name(n) for n in raw_db_names])
    
    test_files = [f for f in os.listdir(TEST_PATH) if f.lower().endswith(('.jpg', '.png'))]
    print(f"-> Đang xử lý {len(test_files)} ảnh...")
    
    data_points = []

    # 1. THU THẬP DỮ LIỆU
    for filename in tqdm(test_files):
        # Lấy nhãn thực tế (đã chuẩn hóa: bradpitt)
        true_lbl_norm = get_true_label(filename, db_names_normalized)
        
        try:
            _, res = pipeline.process_image(os.path.join(TEST_PATH, filename))
            
            if len(res) > 0:
                pred_name_raw = res[0]['name']
                distance = res[0]['distance']
                
                # Chuẩn hóa tên dự đoán (Brad_Pitt -> bradpitt)
                pred_name_norm = normalize_name(pred_name_raw)
                
                # Logic xác định danh tính:
                # Nếu tên thật == tên dự đoán (đã chuẩn hóa) -> Cùng 1 người (Identity Match)
                is_same_identity = (true_lbl_norm != "unknown" and true_lbl_norm == pred_name_norm)
                
                data_points.append({
                    "true_label": true_lbl_norm,
                    "distance": distance,
                    "is_same_identity": is_same_identity,
                    "found_face": True
                })
            else:
                # Không tìm thấy mặt -> Tính là False Negative nếu là người quen
                data_points.append({
                    "true_label": true_lbl_norm,
                    "distance": 999.0, # Gán distance cực lớn
                    "is_same_identity": False,
                    "found_face": False
                })
                
        except Exception: continue

    if len(data_points) == 0:
        print("Lỗi: Không có dữ liệu.")
        return

    # 2. QUÉT NGƯỠNG
    print("\n-> Đang tính toán ma trận...")
    df = pd.DataFrame(data_points)
    
    # Metric L2 thường nằm trong khoảng 0.5 - 1.5
    thresholds = np.arange(0.5, 1.6, 0.01)
    
    best_f1 = 0
    best_thresh = 0
    best_acc = 0
    history = []

    for t in thresholds:
        # LOGIC ĐÁNH GIÁ:
        
        # 1. Trường hợp máy đoán là "Known" (Distance <= t)
        # - TP: Máy đoán đúng người (Identity Match)
        # - FP: Máy đoán nhầm người (Identity Mismatch) HOẶC đoán Unknown thành Known
        
        # 2. Trường hợp máy đoán là "Unknown" (Distance > t)
        # - TN: Nhãn thật là Unknown -> Máy chặn đúng
        # - FN: Nhãn thật là Known -> Máy bỏ sót
        
        # Lọc dataframe
        predicted_match = df[df['distance'] <= t]
        predicted_unknown = df[df['distance'] > t]
        
        # --- TÍNH TOÁN TP, FP ---
        # TP: Khoảng cách nhỏ VÀ đúng là người đó
        tp = len(predicted_match[predicted_match['is_same_identity'] == True])
        
        # FP: Khoảng cách nhỏ NHƯNG sai người (hoặc nhận vơ Unknown)
        fp = len(predicted_match[predicted_match['is_same_identity'] == False])
        
        # --- TÍNH TOÁN TN, FN ---
        # TN: Khoảng cách lớn VÀ thực tế là Unknown
        tn = len(predicted_unknown[predicted_unknown['true_label'] == "unknown"])
        
        # FN: Khoảng cách lớn NHƯNG thực tế là người quen (Bị đẩy sang Unknown)
        fn = len(predicted_unknown[predicted_unknown['true_label'] != "unknown"])
        
        # Các trường hợp NoFace (distance=999) sẽ tự động rơi vào FN (nếu là người quen) hoặc TN (nếu là Unknown)
        
        try:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            acc = (tp + tn) / len(df)
        except: continue

        history.append({"Threshold": t, "F1": f1, "Accuracy": acc})

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_acc = acc

    # 3. KẾT QUẢ
    print("\n" + "="*50)
    print(f"KẾT QUẢ CUỐI CÙNG (SAU KHI FIX LỖI TÊN)")
    print("="*50)
    print(f"NGƯỠNG VÀNG: {best_thresh:.2f}")
    print(f"-> Accuracy: {best_acc*100:.2f}%")
    print(f"-> F1-Score: {best_f1:.3f}")
    print("="*50)
    
    # Vẽ biểu đồ
    res_df = pd.DataFrame(history)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=res_df, x="Threshold", y="F1", label="F1 Score", color="blue")
    sns.lineplot(data=res_df, x="Threshold", y="Accuracy", label="Accuracy", color="green", linestyle="--")
    plt.axvline(best_thresh, color='red', linestyle=':')
    plt.title(f"Tối ưu ngưỡng ({CONFIG['model']})")
    plt.grid(True)
    plt.savefig("Threshold_Optimization_Graph_Fixed.png")
    print("Đã lưu biểu đồ: Threshold_Optimization_Graph_Fixed.png")

if __name__ == "__main__":
    run_auto_tune()