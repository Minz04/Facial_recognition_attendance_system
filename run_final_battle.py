import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from deep_face_pipeline import FaceRecognitionPipeline

# --- CẤU HÌNH HỆ THỐNG ---
PROJECT_ROOT = os.getcwd()
YOLO_REPO = 'yolov5'
YOLO_WEIGHTS = os.path.join(YOLO_REPO, 'runs', 'train', 'train_face_detection_v3', 'weights', 'best.pt')
DB_PATH = "mini_database_test"
TEST_PATH = "mini_dataset_test" 

# --- CẤU HÌNH 4 TỔ HỢP ĐÃ TỐI ƯU ---
# (Đã điền sẵn ngưỡng bạn tìm được)
BATTLE_CONFIGS = [
    {
        "name": "Facenet512_L2_MTCNN",
        "model": "Facenet512",
        "metric": "euclidean_l2",
        "backend": "mtcnn",
        "threshold": 1.12
    }
]  

# Thư mục tổng chứa kết quả
MASTER_OUTPUT_DIR = "FINAL_BATTLE_RESULTS_FULL_v1"
if not os.path.exists(MASTER_OUTPUT_DIR): os.makedirs(MASTER_OUTPUT_DIR)

# Hàm lấy nhãn chuẩn
def get_true_label(filename, db_names):
    base = os.path.splitext(filename)[0]
    if "unknown" in base.lower(): return "Unknown"
    parts = base.split('_')
    name = "_".join(parts[1:]).strip() if len(parts) >= 2 and parts[0].isdigit() else base.strip()
    return name if name.lower() in db_names else "Unknown"

def run_final_battle():
    print(f"--- BẮT ĐẦU CHẠY THỰC NGHIỆM TRÊN TOÀN BỘ DATASET ({TEST_PATH}) ---")
    
    # 1. Chuẩn bị
    db_names = [f.lower() for f in os.listdir(DB_PATH) if os.path.isdir(os.path.join(DB_PATH, f))]
    test_files = [f for f in os.listdir(TEST_PATH) if f.lower().endswith(('.jpg', '.png'))]
    print(f"Số lượng ảnh cần xử lý: {len(test_files)} ảnh/tổ hợp")
    print(f"Tổng số tổ hợp: {len(BATTLE_CONFIGS)}")
    print("-" * 60)

    summary_data = []

    # 2. Vòng lặp qua từng cấu hình
    for config in BATTLE_CONFIGS:
        combo_name = config['name']
        print(f"\n>>> Đang chạy: {combo_name} (Threshold: {config['threshold']}) ...")
        
        # Tạo folder riêng cho tổ hợp này
        SUB_DIR = os.path.join(MASTER_OUTPUT_DIR, combo_name)
        if not os.path.exists(SUB_DIR): os.makedirs(SUB_DIR)

        try:
            # Khởi tạo Pipeline
            pipeline = FaceRecognitionPipeline(
                yolo_model_weights_path=YOLO_WEIGHTS,
                db_path=DB_PATH,
                yolo_repo_path=YOLO_REPO,
                df_model_name=config['model'],
                df_distance_metric=config['metric'],
                alignment_detector_backend=config['backend'],
                custom_threshold=config['threshold'], # Truyền ngưỡng tối ưu vào đây
                use_alignment=True
            )
            
            # Warm-up
            if len(test_files) > 0:
                pipeline.process_image(os.path.join(TEST_PATH, test_files[0]))

            y_true = []
            y_pred = []
            latencies = []

            # Chạy Loop
            for filename in tqdm(test_files, desc=combo_name):
                true_lbl = get_true_label(filename, db_names)
                
                start = time.time()
                try:
                    # Xử lý ảnh
                    _, res = pipeline.process_image(os.path.join(TEST_PATH, filename))
                    latencies.append(time.time() - start)
                    pred_lbl = res[0]['name'] if len(res) > 0 else "NoFace"
                except: continue

                # Normalize
                clean_true = true_lbl.lower().replace(" ","").replace("_","")
                clean_pred = pred_lbl.lower().replace(" ","").replace("_","")
                final_pred = true_lbl if clean_true == clean_pred else pred_lbl
                
                y_true.append(true_lbl)
                y_pred.append(final_pred)

            # --- TÍNH TOÁN & LƯU KẾT QUẢ ---
            if len(y_true) > 0:
                # 1. Metrics cơ bản
                acc = accuracy_score(y_true, y_pred)
                prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
                fps = 1.0 / (sum(latencies) / len(latencies))
                
                print(f"   [DONE] Acc: {acc*100:.2f}% | F1: {f1:.3f} | FPS: {fps:.2f}")

                # 2. Lưu Report chi tiết (Từng class) vào file text trong folder con
                report_path = os.path.join(SUB_DIR, "detailed_classification_report.txt")
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(f"MODEL CONFIGURATION: {combo_name}\n")
                    f.write(f"Threshold: {config['threshold']}\n")
                    f.write(f"Accuracy: {acc*100:.2f}%\n")
                    f.write(f"Avg FPS: {fps:.2f}\n")
                    f.write("-" * 50 + "\n")
                    f.write(classification_report(y_true, y_pred, zero_division=0))
                
                # 3. Vẽ Confusion Matrix lưu vào folder con
                try:
                    labels = sorted(list(set(y_true + y_pred)))
                    if len(labels) < 50: # Chỉ vẽ nếu không quá nhiều class để tránh rối
                        cm = confusion_matrix(y_true, y_pred, labels=labels)
                        plt.figure(figsize=(12, 10))
                        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
                        plt.title(f"Confusion Matrix: {combo_name}\n(Acc: {acc*100:.2f}%)")
                        plt.ylabel('True Label')
                        plt.xlabel('Predicted Label')
                        plt.tight_layout()
                        plt.savefig(os.path.join(SUB_DIR, "confusion_matrix.png"))
                        plt.close()
                except: pass

                # 4. Thêm vào dữ liệu tổng hợp
                summary_data.append({
                    "Combo Name": combo_name,
                    "Model": config['model'],
                    "Backend": config['backend'],
                    "Threshold": config['threshold'],
                    "Accuracy (%)": round(acc * 100, 2),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1-Score": round(f1, 4),
                    "FPS": round(fps, 2),
                    "Avg Latency (s)": round(1.0/fps, 4)
                })

        except Exception as e:
            print(f"   [ERROR] Lỗi khi chạy {combo_name}: {e}")

    # 3. TỔNG HỢP CUỐI CÙNG (MASTER REPORT)
    if len(summary_data) > 0:
        df = pd.DataFrame(summary_data)
        df = df.sort_values(by="F1-Score", ascending=False)
        
        # Lưu file Excel tổng ở thư mục gốc
        master_csv = os.path.join(MASTER_OUTPUT_DIR, "Master_Summary_Report.csv")
        df.to_csv(master_csv, index=False)
        print(f"\n[XONG] Đã lưu bảng tổng hợp tại: {master_csv}")

        # Vẽ biểu đồ so sánh tổng quan
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df, x="FPS", y="Accuracy (%)", hue="Combo Name", style="Backend", s=300, alpha=0.9)
        
        # Gắn nhãn
        for i in range(len(df)):
            row = df.iloc[i]
            plt.text(row["FPS"], row["Accuracy (%)"] + 0.5, 
                     f"{row['Combo Name']}\n(F1: {row['F1-Score']})", 
                     fontsize=9, fontweight='bold', ha='center')
            
        plt.title(f"KẾT QUẢ CUỐI CÙNG TRÊN {len(test_files)} ẢNH")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(MASTER_OUTPUT_DIR, "Final_Comparison_Chart.png"))
        
        print("\n" + "="*50)
        print("ĐÃ HOÀN TẤT TOÀN BỘ QUÁ TRÌNH TEST")
        print(f"Vui lòng kiểm tra thư mục: {MASTER_OUTPUT_DIR}")
        print("="*50)

if __name__ == "__main__":
    run_final_battle()