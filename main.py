# main.py
from deep_face_pipeline import FaceRecognitionPipeline
import os
import urllib.request
import time

# --- Cấu hình chung ---
YOLO_MODEL_WEIGHTS = 'yolov5/runs/train/train_face_detection_v2/weights/best.pt'
YOLO_REPO_PATH = 'yolov5'
YOLO_CONFIDENCE = 0.4

DEEPFACE_DB_PATH = 'database'
DEEPFACE_MODEL_NAME = 'Facenet'
DEEPFACE_DISTANCE_METRIC = 'cosine'

USE_FACE_ALIGNMENT = True  # Đặt True hoặc False tùy theo nhu cầu test tốc độ
ALIGNMENT_DETECTOR = 'mtcnn' # 'opencv' nhẹ hơn 'retinaface' hoặc 'mtcnn', dùng để test tốc độ

# --- Cấu hình nguồn xử lý & đầu ra ---
IMAGE_TO_PROCESS = "database/01_Bill_Gates/Bill_Gates_0003.jpg"  # Đường dẫn đến ảnh cần xử lý, hoặc None nếu không có ảnh
VIDEO_TO_PROCESS = 0

# KÍCH THƯỚC CAPTURE MONG MUỐN CỦA CAMERA
CAMERA_CAPTURE_WIDTH = 640
CAMERA_CAPTURE_HEIGHT = 480

# KÍCH THƯỚC CỬA SỔ HIỂN THỊ VIDEO MONG MUỐN (có thể khác với capture resolution)
DISPLAY_WINDOW_WIDTH = 800  # Ví dụ, hiển thị to hơn chút
DISPLAY_WINDOW_HEIGHT = 600

OUTPUT_BASE_DIR = 'output'
OUTPUT_IMAGE_NAME = 'processed_image_WITH_Face_ALIGNMENT.jpg'
# OUTPUT_VIDEO_NAME_PREFIX = 'processed_video'

def main():
    if not os.path.exists(OUTPUT_BASE_DIR):
        os.makedirs(OUTPUT_BASE_DIR)

    print("Initializing Face Recognition Pipeline...")
    pipeline = FaceRecognitionPipeline(
        yolo_model_weights_path=YOLO_MODEL_WEIGHTS,
        yolo_repo_path=YOLO_REPO_PATH,
        db_path=DEEPFACE_DB_PATH,
        df_model_name=DEEPFACE_MODEL_NAME,
        df_distance_metric=DEEPFACE_DISTANCE_METRIC,
        yolo_confidence=YOLO_CONFIDENCE,
        use_alignment=USE_FACE_ALIGNMENT, # Sử dụng giá trị đã cấu hình
        alignment_detector_backend=ALIGNMENT_DETECTOR # Sử dụng giá trị đã cấu hình
    )

    if pipeline.yolo_detector.model is None:
        print("CRITICAL: Failed to initialize YOLO model. Exiting.")
        return

    # Xử lý ảnh (nếu có)
    if IMAGE_TO_PROCESS:
        if not os.path.exists(IMAGE_TO_PROCESS):
            print(f"Error: Image file not found at '{IMAGE_TO_PROCESS}'. Skipping image processing.")
        else:
            print(f"Processing image: {IMAGE_TO_PROCESS}...")
            output_image_path = os.path.join(OUTPUT_BASE_DIR, OUTPUT_IMAGE_NAME)
            processed_image, results = pipeline.process_image(IMAGE_TO_PROCESS, output_image_path)
            if processed_image is not None:
                print(f"Image processing complete. Output: {output_image_path}")
                for res in results:
                    name_str = f"{res.get('name', 'N/A')}"
                    dist_str = f"{res.get('distance', -1):.4f}" if res.get('distance', -1) != -1 else "N/A"
                    print(f"  - Found: {name_str:<15} (Dist: {dist_str})")
            else:
                print(f"Image processing failed for {IMAGE_TO_PROCESS}.")


    # Xử lý video
    if VIDEO_TO_PROCESS is not None:
        print(f"\nProcessing video source: {VIDEO_TO_PROCESS}...")
        output_video_path = None
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if isinstance(VIDEO_TO_PROCESS, str) and os.path.isfile(VIDEO_TO_PROCESS):
            base_name = os.path.basename(VIDEO_TO_PROCESS)
            name, ext = os.path.splitext(base_name)
            output_video_path = os.path.join(OUTPUT_BASE_DIR, f"{name}_processed_{timestamp}{ext}")
        elif isinstance(VIDEO_TO_PROCESS, int):
            output_video_path = os.path.join(OUTPUT_BASE_DIR, f"webcam_capture_{timestamp}.mp4")

        # TRUYỀN CÁC THAM SỐ ĐỘ PHÂN GIẢI VÀO ĐÂY
        pipeline.process_video(VIDEO_TO_PROCESS, output_video_path,
                               display_width=DISPLAY_WINDOW_WIDTH,
                               display_height=DISPLAY_WINDOW_HEIGHT,
                               camera_capture_width=CAMERA_CAPTURE_WIDTH,
                               camera_capture_height=CAMERA_CAPTURE_HEIGHT)
        
        if output_video_path and os.path.exists(output_video_path):
            print(f"Video processing session ended. Output video: {output_video_path}")
        elif output_video_path:
             print(f"Video processing session ended. Attempted to save to {output_video_path}, but file was not created.")
        else:
            print("Video processing session (webcam/stream without specified output file) ended.")

if __name__ == '__main__':
    print("Starting Face Recognition Application...")
    main()
    print("\nApplication finished.")