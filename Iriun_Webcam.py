import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is available")
    cap.release()
def get_iriun_camera():
    cap = cv2.VideoCapture(1)  # ถ้า Iriun อยู่ที่ index 1
    if not cap.isOpened():
        raise RuntimeError("ไม่สามารถเชื่อมต่อ Iriun Webcam ได้")
    return cap