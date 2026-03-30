import cv2
import numpy as np
import base64
import io

from urllib.parse import unquote

def extract_path_from_url(url, bucket_name):
    """
    Chuyển Public URL thành Path để xóa trong Storage.
    VD: https://.../rollcall_thumbnails/evidence/img.jpg -> evidence/img.jpg
    """
    if not url: return None
    try:
        # Tìm vị trí tên bucket trong URL
        if f"/{bucket_name}/" in url:
            # Lấy phần sau tên bucket
            path_raw = url.split(f"/{bucket_name}/")[-1]
            # Giải mã ký tự đặc biệt (%20 -> space)
            return unquote(path_raw)
    except:
        return None
    return None

def crop_face_from_image(image, face_location_xyxy, padding=10):
    """
    Cắt khuôn mặt từ ảnh dựa trên bounding box (x1, y1, x2, y2) từ YOLO.
    Args:
        image (np.array): Ảnh gốc (BGR).
        face_location_xyxy (tuple): Bounding box theo định dạng (x1, y1, x2, y2).
        padding (int): Số pixel thêm vào xung quanh khuôn mặt.
    Returns:
        np.array: Ảnh khuôn mặt đã cắt. None nếu input không hợp lệ.
    """
    if image is None or image.size == 0:
        return None
    if not isinstance(face_location_xyxy, (list, tuple)) or len(face_location_xyxy) != 4:
        print(f"Invalid face_location_xyxy format: {face_location_xyxy}. Expected (x1, y1, x2, y2).")
        return None

    x1, y1, x2, y2 = [int(val) for val in face_location_xyxy]
    
    h, w = image.shape[:2]

    # Áp dụng padding
    x1_padded = max(0, x1 - padding)
    y1_padded = max(0, y1 - padding)
    x2_padded = min(w, x2 + padding)
    y2_padded = min(h, y2 + padding)
    
    # Đảm bảo crop không bị đảo ngược hoặc có kích thước âm
    if y2_padded > y1_padded and x2_padded > x1_padded:
        face_image = image[y1_padded:y2_padded, x1_padded:x2_padded]
        return face_image
    else:
        # print(f"Warning: Invalid crop dimensions after padding: ({x1_padded},{y1_padded},{x2_padded},{y2_padded}) for original bbox ({x1},{y1},{x2},{y2}). Image shape: {image.shape}")
        return None

def image_to_jpeg_bytes(image, quality=80):
    """
    Chuyển đổi ảnh OpenCV (np.array) sang định dạng JPEG bytes.
    Args:
        image (np.array): Ảnh OpenCV BGR.
        quality (int): Chất lượng JPEG (0-100).
    Returns:
        bytes: Dữ liệu ảnh JPEG. None nếu ảnh rỗng hoặc lỗi.
    """
    if image is None or image.size == 0:
        return None
    is_success, buffer = cv2.imencode(".jpeg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if is_success:
        return buffer.tobytes()
    return None

def img_bytes_to_cv2(img_bytes):
    """
    Chuyển đổi JPEG bytes sang ảnh OpenCV (np.array).
    """
    if img_bytes is None or len(img_bytes) == 0:
        return None
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img