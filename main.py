import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from Iriun_Webcam import get_iriun_camera

from lstm_model import build_lstm_model, predict_sequence
import numpy as np

# ===== กำหนดจำนวน feature และ class =====
num_features = 150   # 75 จุด (pose+hand) × 2 (x,y)
num_classes = 4      # Fall / No Fall / Pre-Fall / Falling
model = build_lstm_model(num_features, num_classes)

sequence_buffer = []

# ===== Mapping Class → ชื่อ Condition =====
GESTURE_LABELS = {
    0: "Fall",
    1: "No Fall",
    2: "Pre-Fall",
    3: "Falling"
}

# ===== โหลด Pose Landmarker =====
pose_model_path = 'models/pose_landmarker_lite.task'
pose_base = python.BaseOptions(model_asset_path=pose_model_path)
pose_options = vision.PoseLandmarkerOptions(base_options=pose_base,
                                            output_segmentation_masks=False)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

# ===== โหลด Hand Landmarker =====
hand_model_path = 'models/hand_landmarker.task'
hand_base = python.BaseOptions(model_asset_path=hand_model_path)
hand_options = vision.HandLandmarkerOptions(base_options=hand_base, num_hands=2)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# ===== Connections สำหรับ Pose Skeleton =====
POSE_CONNECTIONS = [
    (11, 13), (13, 15),   # Left arm
    (12, 14), (14, 16),   # Right arm
    (11, 12),             # Shoulders
    (23, 24),             # Hips
    (11, 23), (12, 24),   # Torso
    (23, 25), (25, 27),   # Left leg
    (24, 26), (26, 28),   # Right leg
    (27, 29), (29, 31),   # Left foot
    (28, 30), (30, 32)    # Right foot
]

# ===== Connections สำหรับ Hand Skeleton =====
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # นิ้วโป้ง
    (0, 5), (5, 6), (6, 7), (7, 8),       # นิ้วชี้
    (0, 9), (9, 10), (10, 11), (11, 12),  # นิ้วกลาง
    (0, 13), (13, 14), (14, 15), (15, 16),# นิ้วนาง
    (0, 17), (17, 18), (18, 19), (19, 20) # นิ้วก้อย
]

# ===== กำหนดสีเดียวกันสำหรับเส้นและจุด =====
LINE_COLOR = (0, 255, 0)   # เขียว
POINT_COLOR = (0, 255, 0)  # เขียว

# ===== เปิดกล้อง Iriun =====
cap = get_iriun_camera()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # ===== ตรวจจับ Pose =====
    pose_result = pose_detector.detect(mp_image)
    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]

        # วาดจุด
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, POINT_COLOR, -1)

        # วาดเส้นเชื่อม
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            x1, y1 = int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h)
            x2, y2 = int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), LINE_COLOR, 2)

    # ===== ตรวจจับ Hand =====
    hand_result = hand_detector.detect(mp_image)
    if hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            # วาดจุด
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, POINT_COLOR, -1)

            # วาดเส้นเชื่อม
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                x1, y1 = int(hand[start_idx].x * w), int(hand[start_idx].y * h)
                x2, y2 = int(hand[end_idx].x * w), int(hand[end_idx].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), LINE_COLOR, 2)

    # ===== เก็บ landmark ลง buffer (เติมค่า default ถ้าไม่เจอ) =====
    frame_features = []

    # Pose
    if pose_result.pose_landmarks:
        for lm in pose_result.pose_landmarks[0]:
            frame_features.extend([lm.x, lm.y])
    else:
        frame_features.extend([0.0] * (33*2))  # เติม 0 ถ้าไม่เจอ pose

    # Hand
    if hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            for lm in hand:
                frame_features.extend([lm.x, lm.y])
        # ถ้าเจอแค่มือเดียว → เติมอีกมือเป็น 0
        if len(hand_result.hand_landmarks) == 1:
            frame_features.extend([0.0] * (21*2))
    else:
        frame_features.extend([0.0] * (21*2*2))  # เติม 0 ถ้าไม่เจอมือเลย

    sequence_buffer.append(frame_features)

    # ===== ถ้า buffer ยาวถึง 30 frame → predict =====
    if len(sequence_buffer) == 30:
        seq = np.array(sequence_buffer)
        if seq.shape[1] == num_features:   # ต้องครบ 150 feature
            gesture_id = predict_sequence(model, seq)
            gesture_name = GESTURE_LABELS[gesture_id]
            print("Condition:", gesture_name)
        else:
            print("Invalid sequence shape:", seq.shape)
        sequence_buffer = []  # reset buffer

    cv2.imshow("Pose + Hand Skeleton + Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()