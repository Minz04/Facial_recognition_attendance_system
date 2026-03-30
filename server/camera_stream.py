import cv2
import threading
import time
import os
import logging

# Cấu hình RTSP TCP để ổn định
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CameraStream:
    def __init__(self, src=0, shared_state=None):
        self.src = src
        self.shared_state = shared_state
        self.cap = None 
        
        self.grabbed = False
        self.frame = None
        self.last_read_time = time.time()
        
        self.running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def set_alive_status(self, status):
        if self.shared_state is not None:
            self.shared_state["is_alive"] = status

    def connect(self):
        if self.cap is not None:
            self.cap.release()
            
        logging.info(f"Đang kết nối Camera: {self.src}...")
        try:
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            # Cố gắng giảm bộ đệm xuống mức thấp nhất
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.cap.isOpened():
                logging.info(">>> KẾT NỐI CAMERA THÀNH CÔNG! <<<")
                self.set_alive_status(True)
                return True
            else:
                logging.error("Kết nối thất bại!")
                self.set_alive_status(False)
                return False
        except Exception as e:
            logging.error(f"Lỗi khởi tạo Camera: {e}")
            self.set_alive_status(False)
            return False

    def update(self):
        self.connect()
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.set_alive_status(False)
                time.sleep(2) 
                self.connect()
                continue

            # Đọc frame liên tục
            grabbed, frame = self.cap.read()

            if grabbed and frame is not None and frame.size > 0:
                with self.lock:
                    self.grabbed = True
                    self.frame = frame 
                    self.last_read_time = time.time()
                
                if self.shared_state and not self.shared_state["is_alive"]:
                    self.set_alive_status(True)
            else:
                with self.lock:
                    self.grabbed = False
                
                # Watchdog: Mất tín hiệu quá 5s
                if time.time() - self.last_read_time > 5.0:
                    logging.warning("Mất tín hiệu quá 5s -> Reconnect...")
                    self.set_alive_status(False)
                    self.connect()

            time.sleep(0.005) 

    def read(self):
        if (time.time() - self.last_read_time) > 5.0:
            return False, None
        with self.lock: 
            if self.grabbed and self.frame is not None:
                return True, self.frame.copy()
            else:
                return False, None
    
    def stop(self):
        self.running = False
        self.thread.join()
        if self.cap: self.cap.release()