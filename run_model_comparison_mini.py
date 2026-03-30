import os
import time
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from deep_face_pipeline import FaceRecognitionPipeline

# --- CẤU HÌNH HỆ THỐNG ---
PROJECT_ROOT = os.getcwd()
YOLO_REPO = 'yolov5'
YOLO_WEIGHTS = os.path.join(YOLO_REPO, 'runs', 'train', 'train_face_detection_v3', 'weights', 'best.pt')

# Đường dẫn dataset MINI của bạn (400 ảnh)
DB_PATH = "mini_database_test"            
TEST_PATH = "mini_dataset_test" 

# --- [KHU VỰC CHỌN TỔ HỢP TEST] ---
# Bạn có thể thêm/bớt tùy thích
MODELS = ["ArcFace", "Facenet512", "Facenet", "VGG-Face", "OpenFace", "DeepFace"] 
METRICS = ["cosine", "euclidean_l2"] 
BACKENDS = ["opencv", "mtcnn", "retinaface"] # RetinaFace rất chính xác nhưng chậm, thêm vào nếu máy khỏe

OUTPUT_DIR = "GRID_SEARCH_RESULTS"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# Hàm lấy nhãn chuẩn
def get_true_label(filename, db_names):
    base = os.path.splitext(filename)[0]
    if "unknown" in base.lower(): return "Unknown"
    parts = base.split('_')
    name = "_".join(parts[1:]).strip() if len(parts) >= 2 and parts[0].isdigit() else base.strip()
    return name if name.lower() in db_names else "Unknown"

def run_grid_search():
    print(f"--- BẮT ĐẦU GRID SEARCH TRÊN {TEST_PATH} ---")
    
    # Load danh sách người quen trong DB
    db_names = [f.lower() for f in os.listdir(DB_PATH) if os.path.isdir(os.path.join(DB_PATH, f))]
    test_files = [f for f in os.listdir(TEST_PATH) if f.lower().endswith(('.jpg', '.png'))]
    
    print(f"Số lượng ảnh: {len(test_files)}")
    print(f"Số tổ hợp: {len(MODELS)} Models x {len(METRICS)} Metrics x {len(BACKENDS)} Alignments")
    print("="*60)

    results = []
    
    # 3 VÒNG LẶP (Model -> Metric -> Backend)
    for model in MODELS:
        for metric in METRICS:
            for backend in BACKENDS:
                combo_name = f"{model}_{metric}_{backend}"
                print(f"\n>>> Testing: {combo_name} ...")
                
                try:
                    # Khởi tạo Pipeline
                    pipeline = FaceRecognitionPipeline(
                        yolo_model_weights_path=YOLO_WEIGHTS,
                        db_path=DB_PATH,
                        yolo_repo_path=YOLO_REPO,
                        df_model_name=model,
                        df_distance_metric=metric,
                        alignment_detector_backend=backend,
                        use_alignment=True
                    )

                    y_true = []
                    y_pred = []
                    latencies = []

                    # Chạy từng ảnh
                    for filename in tqdm(test_files, leave=False):
                        true_lbl = get_true_label(filename, db_names)
                        
                        start = time.time()
                        try:
                            # Chỉ xử lý, không lưu ảnh output để tối ưu tốc độ
                            _, res = pipeline.process_image(os.path.join(TEST_PATH, filename))
                            latencies.append(time.time() - start)
                            
                            pred_lbl = res[0]['name'] if len(res) > 0 else "NoFace"
                        except: continue

                        # Chuẩn hóa tên để so sánh
                        clean_true = true_lbl.lower().replace(" ","").replace("_","")
                        clean_pred = pred_lbl.lower().replace(" ","").replace("_","")
                        
                        final_pred = true_lbl if clean_true == clean_pred else pred_lbl
                        
                        y_true.append(true_lbl)
                        y_pred.append(final_pred)

                    # Tính chỉ số
                    if len(y_true) > 0:
                        acc = accuracy_score(y_true, y_pred)
                        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
                        fps = 1.0 / (sum(latencies) / len(latencies))
                        
                        print(f"    [KQ] Acc: {acc*100:.1f}% | F1: {f1:.2f} | FPS: {fps:.1f}")
                        
                        results.append({
                            "Combo": combo_name,
                            "Model": model,
                            "Metric": metric,
                            "Backend": backend,
                            "Accuracy": acc * 100,
                            "Precision": prec,
                            "Recall": rec,
                            "F1-Score": f1,
                            "FPS": fps
                        })

                except Exception as e:
                    print(f"    [LỖI] {combo_name}: {e}")

    # Xuất báo cáo Excel
    if len(results) > 0:
        df = pd.DataFrame(results)
        # Sắp xếp theo F1-Score (thường quan trọng hơn Accuracy)
        df = df.sort_values(by=["F1-Score", "FPS"], ascending=False)
        
        csv_path = os.path.join(OUTPUT_DIR, "Final_Grid_Report.csv")
        df.to_csv(csv_path, index=False)
        
        print("\n" + "="*50)
        print("TOP 5 CẤU HÌNH TỐT NHẤT (Sắp xếp theo F1-Score)")
        print("="*50)
        print(df[["Combo", "F1-Score", "FPS"]].head(5))
        print(f"\n[DONE] Báo cáo chi tiết đã lưu tại: {csv_path}")

if __name__ == "__main__":
    run_grid_search()