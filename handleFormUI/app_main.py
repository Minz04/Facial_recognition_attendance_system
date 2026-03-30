import os
import sys

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import cấu hình từ config.py
import config 

# Các import cần thiết khác
import re
import traceback
import numpy as np
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from ui.ui_form_FaceRecognition import Ui_MainWindow # ui vẫn nằm ngang hàng với handleFormUI


try:
    # Các import này vẫn giữ nguyên vì worker.py và add_user.py nằm cùng cấp
    from handleFormUI.worker import RecognitionWorker
    from handleFormUI.add_user import AddUserDialog 
except ImportError as e_fallback:
    error_msg = f"Không thể tải module:\n{e_fallback}\n\nTraceback:\n{traceback.format_exc()}"
    QMessageBox.critical(None, "Lỗi Import Module", error_msg); sys.exit(1)

# Thay thế các định nghĩa đường dẫn cứng bằng các biến từ config
DATABASE_FOLDER_APP = config.DATABASE_FOLDER
YOLO_MODEL_WEIGHTS_PATH = config.YOLO_MODEL_WEIGHTS_PATH
YOLO_REPO_PATH = config.YOLO_REPO_PATH
YOLO_CONFIDENCE_THRESHOLD = config.YOLO_CONFIDENCE_THRESHOLD

os.makedirs(DATABASE_FOLDER_APP, exist_ok=True)

class FaceRecognitionApp(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None): # Khởi tạo Main Window
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Hệ Thống Nhận Diện Khuôn Mặt") 
        self.recognition_thread = None
        self.add_user_window = None
        self.last_recognized_person_folder = None
        self.initialize_recognition_worker()
        self.btnAddPerson.clicked.connect(self.action_open_add_user_form)
        self.labelPicturePerson.setAlignment(Qt.AlignCenter)
        self.clear_recognized_person_info()
        self.statusBar().showMessage("Hệ thống sẵn sàng.", 3000)

    def initialize_recognition_worker(self): # Khởi tạo RecognitionWorker
        if self.recognition_thread and self.recognition_thread.isRunning():
            self.recognition_thread.stop()
            if not self.recognition_thread.wait(3000):
                self.recognition_thread.terminate(); self.recognition_thread.wait()
        self.recognition_thread = None
        try:
            self.recognition_thread = RecognitionWorker(
                yolo_weights_path=YOLO_MODEL_WEIGHTS_PATH, # Sử dụng biến đã được gán từ config
                yolo_repo_path=YOLO_REPO_PATH,           # Sử dụng biến đã được gán từ config
                db_path=DATABASE_FOLDER_APP,             # Sử dụng biến đã được gán từ config
                yolo_confidence=YOLO_CONFIDENCE_THRESHOLD, # Sử dụng biến đã được gán từ config
                parent=self
            )
            self.recognition_thread.signals.frame_ready.connect(self.update_camera_display)
            self.recognition_thread.signals.recognition_result.connect(self.display_recognition_result)
            self.recognition_thread.signals.no_recognition.connect(self.clear_recognized_person_info)
            self.recognition_thread.signals.error.connect(self.handle_worker_error)
            if hasattr(self.recognition_thread, '_prevent_run') and self.recognition_thread._prevent_run:
                QMessageBox.critical(self, "Lỗi Khởi Tạo Worker", "Worker không thể khởi tạo. Kiểm tra console.")
                self.btnAddPerson.setEnabled(False); self.labelCamera.setText("❌ LỖI KHỞI TẠO"); self.statusBar().showMessage("LỖI: Worker không thể khởi tạo!")
            else:
                self.recognition_thread.start(); self.statusBar().showMessage("Đang khởi động hệ thống...", 5000)
        except Exception as e_init_worker:
            QMessageBox.critical(self, "Lỗi Worker", f"Không thể tạo RecognitionWorker:\n{e_init_worker}\n\n{traceback.format_exc()}"); self.labelCamera.setText("❌ LỖI WORKER")

    @pyqtSlot(QImage)
    def update_camera_display(self, qt_image): # Cập nhật khung hình camera
        if hasattr(self, 'labelCamera') and qt_image is not None:
            try:
                pixmap = QPixmap.fromImage(qt_image).scaled(self.labelCamera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.labelCamera.setPixmap(pixmap)
            except Exception as e: print(f"[LỖI MAIN_APP] update_camera_display: {e}")

    def _load_person_avatar(self, person_folder_name): # Tải ảnh đại diện người dùng
        try:
            person_folder_path = os.path.join(DATABASE_FOLDER_APP, person_folder_name)
            if not os.path.isdir(person_folder_path): self.labelPicturePerson.setText("📁 Thư mục lỗi"); self.labelPicturePerson.setPixmap(QPixmap()); return
            photo_path_to_display = None
            for fname in sorted(os.listdir(person_folder_path)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')): photo_path_to_display = os.path.join(person_folder_path, fname); break
            if photo_path_to_display and os.path.exists(photo_path_to_display):
                pixmap = QPixmap(photo_path_to_display)
                self.labelPicturePerson.setPixmap(pixmap.scaled(self.labelPicturePerson.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation) if not pixmap.isNull() else QPixmap())
                if pixmap.isNull(): self.labelPicturePerson.setText("🖼️ Ảnh lỗi")
            else: self.labelPicturePerson.setText("📷 Không có ảnh"); self.labelPicturePerson.setPixmap(QPixmap())
        except Exception as e_avatar: print(f"[LỖI MAIN_APP] _load_person_avatar: {e_avatar}"); self.labelPicturePerson.setText("⚠️ Lỗi tải ảnh")

    @pyqtSlot(np.ndarray, str, str)
    def display_recognition_result(self, face_crop_bgr, recognized_name, recognized_folder_name):
        try:
            self.txt_name_person.setText(recognized_name)
            match_id = re.match(r"^(\d+)_?(.*)", recognized_folder_name)
            id_str = match_id.group(1) if match_id else "N/A"
            self.txt_id_person.setText(id_str)
            if self.last_recognized_person_folder != recognized_folder_name:
                self._load_person_avatar(recognized_folder_name)
                self.statusBar().showMessage(f"✅ Nhận diện: {recognized_name} (ID: {id_str})", 3000)
                self.last_recognized_person_folder = recognized_folder_name
        except Exception as e_display: print(f"[LỖI MAIN_APP] display_recognition_result: {e_display}")

    @pyqtSlot()
    def clear_recognized_person_info(self):
        if hasattr(self, 'txt_id_person') and hasattr(self, 'txt_name_person'):
            if self.txt_id_person.toPlainText() != "---" or self.txt_name_person.toPlainText() != "---" or self.last_recognized_person_folder is not None:
                self.txt_name_person.setText("---"); self.txt_id_person.setText("---")
                self.labelPicturePerson.setText("👤 (Chưa nhận diện)"); self.labelPicturePerson.setPixmap(QPixmap())
                self.last_recognized_person_folder = None

    @pyqtSlot(str)
    def handle_worker_error(self, error_message):
        self.statusBar().showMessage(f"⚠️ Lỗi Worker: {error_message}", 7000)
        critical_keywords = ["camera", "model", "yolo", "deepface", "không thể mở", "không tải được"]
        if any(keyword in error_message.lower() for keyword in critical_keywords):
            QMessageBox.warning(self, "Lỗi Worker Nghiêm Trọng", f"Lỗi worker:\n{error_message}\nHệ thống có thể không ổn định.")

    def action_open_add_user_form(self):
        if self.recognition_thread and self.recognition_thread.isRunning():
            self.recognition_thread.stop()
            self.labelCamera.setText("📹 Camera tạm dừng..."); self.statusBar().showMessage("⏸️ Tạm dừng nhận diện.", 3000)
        try:
            self.add_user_window = AddUserDialog(
                database_main_folder_path=DATABASE_FOLDER_APP, # Sử dụng biến đã được gán từ config
                yolo_weights_path=YOLO_MODEL_WEIGHTS_PATH, # Sử dụng biến đã được gán từ config
                yolo_repo_path=YOLO_REPO_PATH,           # Sử dụng biến đã được gán từ config
                yolo_confidence=YOLO_CONFIDENCE_THRESHOLD, # Sử dụng biến đã được gán từ config
                parent=self
            )
            self.add_user_window.user_added_completed.connect(self.handle_new_user_completed)
            self.add_user_window.request_return_to_main.connect(self.handle_return_from_add_user)
            self.hide(); self.add_user_window.show()
        except Exception as e_open_add:
            QMessageBox.critical(self, "Lỗi Mở Form", f"Không thể mở form thêm người dùng:\n{e_open_add}\n{traceback.format_exc()}"); self.show_main_window_and_restart_worker()

    @pyqtSlot()
    def handle_new_user_completed(self):
        if self.recognition_thread: self.recognition_thread.reload_embeddings(); self.statusBar().showMessage("🔄 DB cập nhật.", 3000)
        if self.add_user_window: self.add_user_window = None
        self.show_main_window_and_restart_worker()

    @pyqtSlot()
    def handle_return_from_add_user(self):
        if self.add_user_window: self.add_user_window = None
        self.show_main_window_and_restart_worker()

    def show_main_window_and_restart_worker(self):
        self.show(); self.clear_recognized_person_info(); self.labelCamera.setText("Đang kết nối camera...")
        if self.recognition_thread:
            if not self.recognition_thread.isRunning():
                if hasattr(self.recognition_thread, '_prevent_run') and self.recognition_thread._prevent_run:
                    QMessageBox.warning(self, "Lỗi Worker", "Worker lỗi. Kiểm tra console."); self.labelCamera.setText("❌ LỖI WORKER")
                else: self.recognition_thread.start(); self.statusBar().showMessage("🔄 Đang khởi động lại nhận diện...", 3000)
        else: self.initialize_recognition_worker()

    def closeEvent(self, event):
        self.statusBar().showMessage("Đang đóng ứng dụng...", 0)
        QApplication.processEvents()
        if self.add_user_window: self.add_user_window.close(); self.add_user_window = None
        if self.recognition_thread and self.recognition_thread.isRunning():
            self.recognition_thread.stop()
            if not self.recognition_thread.wait(5000): self.recognition_thread.terminate(); self.recognition_thread.wait()
        event.accept()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("Face Recognition System"); app.setApplicationVersion("1.1 Compact")
        app.setStyleSheet("""
            QMainWindow { background-color: #e8e8e8; }
            QLabel#labelCamera { background-color: black; border: 1px solid #cccccc; }
            QLabel#labelPicturePerson { background-color: #f0f0f0; border: 1px solid #cccccc; }
            QPushButton { background-color: #d0d0d0; border: 1px solid #b0b0b0; padding: 5px; min-width: 70px; }
            QPushButton:hover { background-color: #c0c0c0; } QPushButton:pressed { background-color: #b0b0b0; }
            QLineEdit, QTextEdit { border: 1px solid #cccccc; padding: 3px; background-color: white; }
            QStatusBar { background-color: #d8d8d8; color: #333333; }
        """)
        main_app_window = FaceRecognitionApp(); main_app_window.show()
        sys.exit(app.exec_())
    except Exception as e_global:
        print(f"[CRITICAL ERROR - MAIN] {e_global}"); traceback.print_exc()
        try: QMessageBox.critical(None, "Lỗi Khởi Động", f"Lỗi nghiêm trọng:\n{e_global}\n\n{traceback.format_exc()}")
        except: pass
        if 'app' in locals(): app.quit()
        sys.exit(1)