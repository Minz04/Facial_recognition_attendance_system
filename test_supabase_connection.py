from database.supabase_client import (
    get_all_face_encodings,
    insert_new_student_and_encoding,
    record_attendance,
    get_pending_students_for_approval
)
import numpy as np
import io

if __name__ == "__main__":
    print("Testing Supabase connection and functions...\n")

    # Kiểm tra lấy danh sách face encodings
    encodings = get_all_face_encodings()
    print(f"Fetched {len(encodings)} face encodings.")
    if encodings:
        print(f"Example encoding data:\n{encodings[0]}\n")

    # Kiểm tra lấy sinh viên đang chờ duyệt (nếu có cột 'duyet')
    pending_students = get_pending_students_for_approval()
    print(f"Fetched {len(pending_students)} students pending approval.")
    if pending_students:
        print(f"Example pending student:\n{pending_students[0]}\n")

    # có thể test thử thêm insert mới
    """
    fake_sv_data = {
        "ma_sv": "SV001",
        "ho_ten": "Nguyen Van A",
        "lop": "KTPM01",
        "email": "vana@example.com"
    }

    fake_encoding = np.random.rand(128).tolist()  # random face vector 128 chiều
    fake_image = io.BytesIO(b"fake_image_data")   # test dummy image (bytes)

    sv_id, img_url = insert_new_student_and_encoding(
        fake_sv_data, fake_encoding, fake_image, "test_face.jpg"
    )
    print(f"✅ Inserted new student {sv_id}, image URL: {img_url}")
    """
