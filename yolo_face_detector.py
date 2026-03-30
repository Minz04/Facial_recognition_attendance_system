import torch
import cv2
import numpy as np

class YOLOFaceDetector:
    def __init__(self, yolo_repo_path='yolov5', model_weights_path='yolov5/runs/train/train_face_detection_v3/weights/best.pt', confidence_threshold=0.32):
        """ 
        Khởi tạo YOLOv5 face detector.
        :param yolo_repo_path: Đường dẫn đến thư mục repository YOLOv5 (đã clone).
        :param model_weights_path: Đường dẫn đến file weights .pt đã train cho việc phát hiện khuôn mặt.
        :param confidence_threshold: Ngưỡng tin cậy để xem xét một phát hiện.
        """
        self.weights_path_used = model_weights_path # Lưu lại để tham khảo nếu cần
        try:
            # Load mô hình YOLOv5 từ local repo
            print(f"Attempting to load YOLOv5 model from: {self.weights_path_used}") 
            self.model = torch.hub.load(
                yolo_repo_path, 
                'custom', # Tải mô hình tùy chỉnh
                path=self.weights_path_used,  # Đường dẫn đến weights
                source='local',  # Tải từ nguồn local
                force_reload=False # Không cần tải lại nếu đã có mô hình
            )
            self.model.conf = confidence_threshold  # Ngưỡng tin cậy
            print(f"YOLOv5 model loaded successfully from {self.weights_path_used}")
        except Exception as e:
            print(f"Error loading YOLOv5 model from {self.weights_path_used}: {e}")
            print("Make sure you have cloned YOLOv5 into the 'yolov5' directory and specified the correct weights path relative to your project root.")
            print("Also ensure the YOLOv5 repository in 'yolov5/' is functional and has hubconf.py.")
            self.model = None

    def detect_faces(self, image):
        """
        Phát hiện khuôn mặt trong một ảnh.
        :param image: Ảnh đầu vào (dạng mảng NumPy, đọc bằng OpenCV).
        :return: Danh sách các bounding box (x1, y1, x2, y2) của các khuôn mặt được phát hiện.
                 Và danh sách các ảnh khuôn mặt đã được cắt (cropped faces).
        """
        if self.model is None:
            print("YOLO model is not loaded. Cannot detect faces.")
            return [], []

        results = self.model(image) 
        detections = results.xyxy[0].cpu().numpy() 
        
        bboxes = []
        cropped_faces = []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            bboxes.append((int(x1), int(y1), int(x2), int(y2)))
            
            h, w = image.shape[:2] # Lấy height, width của ảnh
            crop_x1 = max(0, int(x1))  # Tránh cắt ra ngoài biên trái
            crop_y1 = max(0, int(y1))  # Tránh cắt ra ngoài biên trên
            crop_x2 = min(w, int(x2))  # Tránh cắt ra ngoài biên phải
            crop_y2 = min(h, int(y2))  # Tránh cắt ra ngoài biên dưới
            
            if crop_y2 > crop_y1 and crop_x2 > crop_x1: 
                cropped_face = image[crop_y1:crop_y2, crop_x1:crop_x2]
                cropped_faces.append(cropped_face)
            else:
                # print(f"Invalid crop dimensions for bbox: {(x1,y1,x2,y2)}. Original image shape: {image.shape}")
                cropped_faces.append(None) 

        return bboxes, cropped_faces

# if __name__ == '__main__':
#     custom_weights_path = 'yolov5/runs/train/train_face_detection_v2/weights/best.pt'
    
#     print(f"Initializing detector with weights: {custom_weights_path}")
#     # Khi gọi constructor, giá trị này sẽ được truyền vào model_weights_path
#     detector = YOLOFaceDetector(model_weights_path=custom_weights_path) 
    
#     if detector.model:
#         test_image_path = 'test_image.jpg' # Đường dẫn đến ảnh muốn kiểm tra

#         print(f"Testing with image: {test_image_path}") 
#         try:
#             img = cv2.imread(test_image_path) 
#             if img is None:
#                 print(f"Could not read test image: {test_image_path}. Please ensure the file exists and is a valid image.")
#             else:
#                 print(f"Image '{test_image_path}' loaded successfully, shape: {img.shape}")
#                 bboxes, cropped_faces = detector.detect_faces(img)
#                 print(f"Detected {len(bboxes)} faces.")

#                 for i, bbox in enumerate(bboxes):
#                     x1, y1, x2, y2 = bbox
#                     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     if cropped_faces[i] is not None and cropped_faces[i].size > 0 :
#                         cv2.imshow(f"Cropped Face {i+1}", cropped_faces[i])
#                     else:
#                         print(f"Cropped face {i+1} is None or empty.")
                
#                 cv2.imshow("Detected Faces", img)
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()
#         except Exception as e:
#             print(f"Error during test execution: {e}")
#             import traceback
#             traceback.print_exc()
#     else:
#         print("Detector model not initialized. Exiting test.")
