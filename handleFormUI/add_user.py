import cv2
import sys
import os
import numpy as np
import traceback
import re
import json
import time
from PyQt5.QtWidgets import QDialog, QMessageBox, QApplication, QLabel, QProgressBar, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from ui.ui_form_ChupAnh import Ui_Form

try:
    from yolo_face_detector import YOLOFaceDetector as AddUserYOLOFaceDetector
    ADD_USER_YOLO_AVAILABLE = True
except ImportError:
    ADD_USER_YOLO_AVAILABLE = False
    class AddUserYOLOFaceDetector:
        def __init__(self, *args, **kwargs): self.model = None
        def detect_faces(self, *args, **kwargs): return [], []

MAX_IMAGES_PER_USER = 8
USER_FACE_QUALITY_WARN_THRESHOLD = 45
USER_FACE_QUALITY_ACCEPT_THRESHOLD = 60

class FaceQualityChecker:
    @staticmethod
    def calculate_overall_quality(image_bgr, face_bbox_coords=None):
        if image_bgr is None or image_bgr.size == 0: return 0, 0, 0, 0
        try:
            face_crop_bgr, face_size = image_bgr, min(image_bgr.shape[0], image_bgr.shape[1])
            if face_bbox_coords:
                x1, y1, x2, y2 = map(int, face_bbox_coords)
                h_img, w_img = image_bgr.shape[:2]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)
                if x1 < x2 and y1 < y2: face_crop_bgr = image_bgr[y1:y2, x1:x2]
                face_size = min(x2 - x1, y2 - y1)
            if face_crop_bgr.size == 0: return 0,0,0,0
            gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
            blur_score_raw = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_quality = 0
            if blur_score_raw > 200: blur_quality = 50
            elif blur_score_raw > 100: blur_quality = 35
            elif blur_score_raw > 50: blur_quality = 20
            elif blur_score_raw > 20: blur_quality = 10
            brightness_score_raw = np.mean(gray)
            brightness_quality = 0
            if 80 <= brightness_score_raw <= 180: brightness_quality = 50
            elif 60 <= brightness_score_raw <= 200: brightness_quality = 35
            elif 40 <= brightness_score_raw <= 220: brightness_quality = 20
            elif 20 <= brightness_score_raw <= 240: brightness_quality = 10
            return blur_quality + brightness_quality, blur_score_raw, brightness_score_raw, face_size
        except Exception: return 0, 0, 0, 0

class SaveUserThread(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    def __init__(self, user_name_processed, folder_name_final, images_list, db_path, parent=None):
        super().__init__(parent)
        self.user_name_processed = user_name_processed
        self.folder_name_final = folder_name_final
        self.images_list = images_list
        self.db_path = db_path
    def run(self):
        try:
            user_folder_path = os.path.join(self.db_path, self.folder_name_final)
            if not os.path.exists(user_folder_path): os.makedirs(user_folder_path)
            total_images = len(self.images_list)
            for i, img_data in enumerate(self.images_list):
                self.status_updated.emit(f"Đang lưu ảnh {i+1}/{total_images}...")
                image_filename = f"{self.user_name_processed}_{i + 1:02d}.jpg"
                save_path = os.path.join(user_folder_path, image_filename)
                if not cv2.imwrite(save_path, img_data):
                    raise IOError(f"Lưu ảnh {save_path} thất bại.")
                self.progress_updated.emit(int(((i + 1) / total_images) * 100))
            self.finished_signal.emit(True, f"Đã thêm '{self.user_name_processed.replace('_',' ')}' thành công!")
        except Exception as e:
            self.finished_signal.emit(False, f"Lỗi khi lưu người dùng: {str(e)}")

class AddUserDialog(QDialog, Ui_Form):
    user_added_completed = pyqtSignal()
    request_return_to_main = pyqtSignal()
    def __init__(self, database_main_folder_path: str, 
                 yolo_weights_path: str, yolo_repo_path: str,
                 yolo_confidence: float, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Thêm Người Dùng Mới (Nâng cao)")
        self.db_path = database_main_folder_path
        self.capture = None
        self.timer_camera_preview = QTimer(self)
        self.timer_camera_preview.timeout.connect(self.update_camera_preview_feed)
        self.captured_frame_for_review = None
        self.temp_images_list = []
        self.quality_scores_list = []
        self.current_image_count = 0
        self.save_thread = None
        self.progress_dialog = None
        self.quality_checker = FaceQualityChecker()
        self.yolo_face_detector = None
        if ADD_USER_YOLO_AVAILABLE:
            self.yolo_face_detector = AddUserYOLOFaceDetector(yolo_repo_path=yolo_repo_path, model_weights_path=yolo_weights_path, confidence_threshold=yolo_confidence)
            if self.yolo_face_detector.model is None: self.yolo_face_detector = None
        if self.yolo_face_detector is None:
            self.cv_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.btnChupAnh.clicked.connect(self.action_capture_image)
        self.btnDongY.clicked.connect(self.action_accept_image)
        self.btnChupLai.clicked.connect(self.action_recapture_image)
        self.btnHoanTat.clicked.connect(self.action_complete_registration)
        self.btnReset.clicked.connect(self.action_reset_form)
        self.btnQuayLai.clicked.connect(self.action_go_back)
        self.txtTenNguoiMoi.textChanged.connect(self.check_name_and_activate_capture)
        self._improved_initialize_camera()
        self.reset_form_to_initial_state()
        if not os.path.exists(self.db_path):
            try: os.makedirs(self.db_path)
            except Exception as e: QMessageBox.critical(self, "Lỗi", f"Không thể tạo thư mục '{self.db_path}': {e}")

    def _improved_initialize_camera(self):
        try:
            for cam_id in [0, 1, -1]:
                self.capture = cv2.VideoCapture(cam_id)
                if self.capture and self.capture.isOpened(): break
            if not self.capture or not self.capture.isOpened(): raise ValueError("Không thể mở camera.")
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.timer_camera_preview.start(33)
        except Exception as e:
            QMessageBox.warning(self, "Lỗi Camera", f"Không thể mở webcam: {e}")
            self.capture = None

    def update_camera_preview_feed(self):
        if self.capture and self.capture.isOpened() and self.timer_camera_preview.isActive():
            ret, frame_bgr = self.capture.read()
            if ret:
                display_frame = frame_bgr.copy()
                face_rects = []
                if self.yolo_face_detector and self.yolo_face_detector.model:
                    bboxes_yolo, _ = self.yolo_face_detector.detect_faces(frame_bgr)
                    for x1,y1,x2,y2 in bboxes_yolo: face_rects.append((x1,y1,x2-x1,y2-y1))
                elif hasattr(self, 'cv_cascade'):
                    gray_for_cascade = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    face_rects = self.cv_cascade.detectMultiScale(gray_for_cascade, 1.1, 4)
                for (x, y, w, h) in face_rects:
                    quality_score, _, _, _ = self.quality_checker.calculate_overall_quality(frame_bgr, (x, y, x + w, y + h))
                    color, quality_text = ((0,0,255), f"Kem ({quality_score})")
                    if quality_score >= USER_FACE_QUALITY_ACCEPT_THRESHOLD: color, quality_text = ((0,255,0), f"Tot ({quality_score})")
                    elif quality_score >= USER_FACE_QUALITY_WARN_THRESHOLD: color, quality_text = ((0,255,255), f"Kha ({quality_score})")
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(display_frame, quality_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                try:
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    h_f, w_f, ch_f = frame_rgb.shape
                    qt_image = QImage(frame_rgb.data, w_f, h_f, ch_f * w_f, QImage.Format_RGB888)
                    self.labelCamera.setPixmap(QPixmap.fromImage(qt_image).scaled(self.labelCamera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                except Exception as e_qt_preview: print(f"Lỗi update preview Qt: {e_qt_preview}")

    def display_captured_image(self, frame_bgr):
        if frame_bgr is None: return
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qt_image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.labelCamera.setPixmap(QPixmap.fromImage(qt_image).scaled(self.labelCamera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e: QMessageBox.warning(self, "Lỗi", f"Lỗi hiển thị ảnh chụp: {e}")
    
    def _update_status_label(self):
        status_text = f"Ảnh đã chấp nhận: {self.current_image_count}/{MAX_IMAGES_PER_USER}"
        if self.quality_scores_list:
            avg_quality = np.mean(self.quality_scores_list)
            status_text += f" | CL TB: {avg_quality:.0f}"
        self.labelTookPhoto.setText(status_text)
        self.label_2.setText(f"Yêu cầu chụp: {MAX_IMAGES_PER_USER} ảnh")

    def reset_form_to_initial_state(self):
        self.temp_images_list.clear()
        self.quality_scores_list.clear()
        self.current_image_count = 0
        self.captured_frame_for_review = None
        self.txtTenNguoiMoi.clear()
        self.txtTenNguoiMoi.setEnabled(True)
        self._update_status_label()
        self.btnChupAnh.show()
        self.btnChupAnh.setEnabled(False)
        self.btnDongY.hide()
        self.btnChupLai.hide()
        self.btnHoanTat.hide()
        self.btnReset.show()
        self.btnQuayLai.show()
        if self.capture and self.capture.isOpened() and not self.timer_camera_preview.isActive():
            self.timer_camera_preview.start(33)
        self.labelCamera.clear()
        self.labelCamera.setText("Đang mở camera..." if self.capture and self.capture.isOpened() else "Lỗi Camera!")

    def check_name_and_activate_capture(self):
        user_name = self.txtTenNguoiMoi.text().strip()
        can_capture = bool(user_name) and (self.capture is not None and self.capture.isOpened())
        if self.btnChupAnh.isVisible(): self.btnChupAnh.setEnabled(can_capture)

    def action_capture_image(self):
        if not (self.capture and self.capture.isOpened()): QMessageBox.warning(self, "Lỗi Camera", "Camera chưa sẵn sàng."); return
        if not self.txtTenNguoiMoi.text().strip(): QMessageBox.warning(self, "Thiếu Thông Tin", "Nhập tên người dùng."); return
        ret, frame_bgr = self.capture.read()
        if ret and frame_bgr is not None:
            face_bbox_for_quality = None
            if self.yolo_face_detector and self.yolo_face_detector.model:
                bboxes_yolo, _ = self.yolo_face_detector.detect_faces(frame_bgr)
                if bboxes_yolo: face_bbox_for_quality = max(bboxes_yolo, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            elif hasattr(self, 'cv_cascade'):
                 gray_for_cascade = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                 face_rects = self.cv_cascade.detectMultiScale(gray_for_cascade, 1.1, 4)
                 if len(face_rects) > 0:
                    x,y,w,h = max(face_rects, key=lambda r: r[2]*r[3])
                    face_bbox_for_quality = (x,y,x+w,y+h)
            if face_bbox_for_quality is None: QMessageBox.warning(self, "Không Tìm Thấy Khuôn Mặt", "Không tìm thấy khuôn mặt. Thử lại."); return
            overall_score, blur_raw, bright_raw, face_size = self.quality_checker.calculate_overall_quality(frame_bgr, face_bbox_for_quality)
            if overall_score < USER_FACE_QUALITY_WARN_THRESHOLD:
                if QMessageBox.question(self, "Chất Lượng Ảnh Thấp", f"Chất lượng ảnh thấp (Điểm: {overall_score}).\nĐộ nét: {blur_raw:.0f}, Độ sáng: {bright_raw:.0f}\nDùng ảnh này hay chụp lại?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.No: return
            self.timer_camera_preview.stop()
            self.captured_frame_for_review = frame_bgr.copy()
            self.captured_frame_quality_score = overall_score
            self.display_captured_image(self.captured_frame_for_review)
            self.btnChupAnh.hide(); self.btnDongY.show(); self.btnChupLai.show()
            self.txtTenNguoiMoi.setEnabled(False)
        else: QMessageBox.warning(self, "Lỗi Chụp Ảnh", "Không thể chụp ảnh.")

    def action_accept_image(self):
        if self.captured_frame_for_review is None: return
        self.temp_images_list.append(self.captured_frame_for_review.copy())
        self.quality_scores_list.append(getattr(self, 'captured_frame_quality_score', 0))
        self.current_image_count = len(self.temp_images_list)
        self._update_status_label()
        self.captured_frame_for_review = None
        if hasattr(self, 'captured_frame_quality_score'): delattr(self, 'captured_frame_quality_score')
        if self.current_image_count < MAX_IMAGES_PER_USER:
            self.timer_camera_preview.start(33)
            self.btnChupAnh.show(); self.btnChupAnh.setEnabled(True)
            self.btnDongY.hide(); self.btnChupLai.hide()
            self.txtTenNguoiMoi.setEnabled(False)
        else: self._action_finish_capture_session()

    def _action_finish_capture_session(self):
        self.timer_camera_preview.stop()
        self.labelCamera.setText(f"Đã đủ {MAX_IMAGES_PER_USER} ảnh.\nNhấn 'Hoàn Tất' để lưu.")
        self.btnChupAnh.hide(); self.btnDongY.hide(); self.btnChupLai.hide()
        self.btnHoanTat.show(); self.btnHoanTat.setEnabled(True)
        self.txtTenNguoiMoi.setEnabled(False)
        avg_quality_final = np.mean(self.quality_scores_list) if self.quality_scores_list else 0
        QMessageBox.information(self, "Hoàn Tất Chụp Ảnh", f"Đã chụp {self.current_image_count} ảnh.\nChất lượng TB: {avg_quality_final:.0f}/100.\nNhấn 'Hoàn Tất' để lưu.")

    def action_recapture_image(self):
        self.captured_frame_for_review = None
        if hasattr(self, 'captured_frame_quality_score'): delattr(self, 'captured_frame_quality_score')
        self.timer_camera_preview.start(33)
        self.btnChupAnh.show(); self.btnChupAnh.setEnabled(True)
        self.btnDongY.hide(); self.btnChupLai.hide()
        self.labelCamera.clear(); self.labelCamera.setText("Đang mở camera...")

    def find_next_available_id(self):
        if not os.path.exists(self.db_path) or not os.path.isdir(self.db_path): return 1
        existing_ids = set()
        try:
            for item_name in os.listdir(self.db_path):
                if os.path.isdir(os.path.join(self.db_path, item_name)):
                    match = re.match(r"^(\d+)_.*", item_name)
                    if match: existing_ids.add(int(match.group(1)))
        except Exception as e: print(f"Lỗi quét ID: {e}"); return None
        current_id = 1
        while current_id in existing_ids: current_id += 1
        return current_id

    def action_complete_registration(self):
        user_name_raw = self.txtTenNguoiMoi.text().strip()
        user_name_processed = re.sub(r'_+', '_', ''.join(c if c.isalnum() or c.isspace() else '' for c in user_name_raw).strip()).replace(' ', '_')
        if not user_name_processed: QMessageBox.warning(self, "Tên Không Hợp Lệ", "Tên không hợp lệ."); return
        if len(self.temp_images_list) != MAX_IMAGES_PER_USER: QMessageBox.warning(self, "Thiếu Ảnh", f"Cần {MAX_IMAGES_PER_USER} ảnh."); return
        next_id = self.find_next_available_id()
        if next_id is None: QMessageBox.critical(self, "Lỗi ID", "Không thể tạo ID."); return
        folder_id_str = f"{next_id:02d}"
        folder_name_final = f"{folder_id_str}_{user_name_processed}"
        user_folder_path_check = os.path.join(self.db_path, folder_name_final)
        if os.path.exists(user_folder_path_check):
            if QMessageBox.question(self, "Người Dùng Tồn Tại", f"Thư mục '{folder_name_final}' đã tồn tại. Ghi đè?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.No: return
            try:
                import shutil; shutil.rmtree(user_folder_path_check)
            except Exception as e_del: QMessageBox.critical(self, "Lỗi Xóa", f"Không thể xóa thư mục cũ: {e_del}"); return
        self.btnHoanTat.setEnabled(False); self.btnReset.setEnabled(False); self.btnQuayLai.setEnabled(False)
        self.progress_dialog = QDialog(self); self.progress_dialog.setWindowTitle("Đang Lưu..."); self.progress_dialog.setFixedSize(350,120); self.progress_dialog.setModal(True)
        layout = QVBoxLayout(self.progress_dialog); self.progress_label_status = QLabel("Đang chuẩn bị...", self.progress_dialog); self.progress_bar_saving = QProgressBar(self.progress_dialog)
        self.progress_bar_saving.setRange(0,100); layout.addWidget(self.progress_label_status); layout.addWidget(self.progress_bar_saving); self.progress_dialog.setLayout(layout)
        self.save_thread = SaveUserThread(user_name_processed, folder_name_final, self.temp_images_list, self.db_path, self)
        self.save_thread.progress_updated.connect(self.progress_bar_saving.setValue)
        self.save_thread.status_updated.connect(self.progress_label_status.setText)
        self.save_thread.finished_signal.connect(self._on_save_completed)
        self.save_thread.start(); self.progress_dialog.exec_()

    def _on_save_completed(self, success, message):
        if self.progress_dialog: self.progress_dialog.accept(); self.progress_dialog = None
        self.btnHoanTat.setEnabled(True); self.btnReset.setEnabled(True); self.btnQuayLai.setEnabled(True)
        if success:
            QMessageBox.information(self, "Thành Công", message)
            self.user_added_completed.emit()
            self.action_go_back() 
        else: QMessageBox.critical(self, "Lỗi Lưu Trữ", message)

    def action_reset_form(self):
        if self.temp_images_list:
            if QMessageBox.question(self, 'Reset Form', "Xóa ảnh và thông tin đã nhập?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
                self.reset_form_to_initial_state()
        else: self.reset_form_to_initial_state()

    def action_go_back(self):
        if self.save_thread and self.save_thread.isRunning(): QMessageBox.warning(self, "Đang Lưu", "Chờ lưu xong."); return
        if self.temp_images_list and self.current_image_count > 0 and not (self.save_thread and self.save_thread.isRunning()):
             if self.sender() == self.btnQuayLai: # Chỉ hỏi nếu nhấn nút Quay Lại thủ công
                if QMessageBox.question(self, "Xác Nhận Thoát", "Dữ liệu chưa lưu sẽ mất. Thoát?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.No: return
        if self.timer_camera_preview.isActive(): self.timer_camera_preview.stop()
        if self.capture: self.capture.release(); self.capture = None
        self.temp_images_list.clear(); self.quality_scores_list.clear()
        self.request_return_to_main.emit(); self.reject()

    def closeEvent(self, event):
        if self.save_thread and self.save_thread.isRunning():
            if QMessageBox.question(self, "Đang Lưu", "Đang lưu. Đóng sẽ gây lỗi. Chắc chắn đóng?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.No:
                event.ignore(); return
            else: self.save_thread.terminate(); self.save_thread.wait()
        if self.timer_camera_preview.isActive(): self.timer_camera_preview.stop()
        if self.capture: self.capture.release(); self.capture = None
        super().closeEvent(event)