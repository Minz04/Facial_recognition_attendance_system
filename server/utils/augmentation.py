import cv2
import numpy as np

class FaceAugmentor:
    """
    Class hỗ trợ tạo ra các biến thể của khuôn mặt để tăng cường dữ liệu training.
    """
    
    @staticmethod
    def adjust_brightness(image, factor):
        """Tăng/Giảm độ sáng. factor > 1 là sáng, < 1 là tối."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.multiply(v, factor)
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def add_noise(image):
        """Thêm nhiễu hạt (Noise) để giả lập camera chất lượng thấp."""
        row, col, ch = image.shape
        mean = 0
        var = 10
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    @staticmethod
    def flip_horizontal(image):
        """Lật ngang ảnh (Gương)."""
        return cv2.flip(image, 1)

    @staticmethod
    def rotate_image(image, angle):
        """Xoay ảnh một góc nhỏ (góc nghiêng của đầu)."""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    @staticmethod
    def scale_and_crop(image, scale_factor=0.9):
        """
        Crop/Scale nhỏ ảnh để mô phỏng khoảng cách xa hơn một chút.
        """
        (h, w) = image.shape[:2]
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Cắt và dán vào giữa ảnh gốc (tạo nền đen nếu cần) hoặc chỉ resize nhỏ lại
        # Ở đây ta chỉ resize nhỏ và sau đó cắt lại về kích thước ban đầu (h, w)
        # Để giữ nguyên kích thước đầu vào (Input size) cho FaceNet/ArcFace
        
        # Để đơn giản, ta chỉ thu nhỏ ảnh và để đó, mô hình sẽ tự căn chỉnh lại (alignment)
        # nếu kích thước khuôn mặt nhỏ hơn trong input size (112x112).
        
        # Vì ảnh đã được cắt sát mặt, chúng ta chỉ cần thêm padding viền đen (tối ưu hơn)
        delta_w = w - new_w
        delta_h = h - new_h
        
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left
        
        # Thêm padding màu đen
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded


    @staticmethod
    def generate_augmentations(original_face_img):
        """
        Nhận vào 1 ảnh mặt gốc (đã cắt), trả về danh sách 8 ảnh biến thể tối ưu.
        """
        augmented_images = []
        
        # 1. Ảnh gốc 
        augmented_images.append(original_face_img)
        
        # 2. Lật ngang (1 vector)
        augmented_images.append(FaceAugmentor.flip_horizontal(original_face_img))
        
        # 3. Ánh sáng (2 vector)
        augmented_images.append(FaceAugmentor.adjust_brightness(original_face_img, 1.3)) # Sáng
        augmented_images.append(FaceAugmentor.adjust_brightness(original_face_img, 0.7)) # Tối
        
        # 4. Góc nghiêng (2 vector)
        augmented_images.append(FaceAugmentor.rotate_image(original_face_img, 10)) # Xoay phải 10 độ
        augmented_images.append(FaceAugmentor.rotate_image(original_face_img, -10)) # Xoay trái 10 độ

        # 5. Chất lượng/Nhiễu (1 vector)
        augmented_images.append(FaceAugmentor.add_noise(original_face_img))
        
        # 6. Khoảng cách/Thu nhỏ (1 vector)
        augmented_images.append(FaceAugmentor.scale_and_crop(original_face_img, 0.95)) # Giả lập zoom out
        
        return augmented_images