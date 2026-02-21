import argparse

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

from Iriun_Webcam import get_iriun_camera
from lstm_model import predict_sequence


def open_camera(camera_mode, source):
    if camera_mode == "iriun":
        return get_iriun_camera()

    if camera_mode == "pi":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open Pi camera at index 0")
        return cap

    if camera_mode == "rtsp":
        if not source:
            raise ValueError("RTSP mode requires --source <rtsp_url>")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open RTSP stream: {source}")
        return cap

    if camera_mode == "index":
        if source is None:
            cam_index = 0
        else:
            cam_index = int(source)
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index: {cam_index}")
        return cap

    raise ValueError(f"Unsupported camera mode: {camera_mode}")


def main():
    parser = argparse.ArgumentParser(description="Pose+Hand fall detection runtime")
    parser.add_argument(
        "--camera",
        default="iriun",
        choices=["iriun", "pi", "rtsp", "index"],
        help="Camera source mode",
    )
    parser.add_argument(
        "--source",
        help="RTSP URL for --camera rtsp, or camera index for --camera index",
    )
    parser.add_argument(
        "--model",
        default="models/lstm_fall_model.h5",
        help="Path to trained model (.h5/.keras)",
    )
    args = parser.parse_args()

    # Feature and class config
    num_features = 150  # (33 pose + 21*2 hands) * 2 coords(x,y)
    model = load_model(args.model)
    sequence_buffer = []

    gesture_labels = {
        0: "Fall",
        1: "No Fall",
        2: "Pre-Fall",
        3: "Falling",
    }

    pose_model_path = "models/pose_landmarker_lite.task"
    hand_model_path = "models/hand_landmarker.task"

    pose_base = python.BaseOptions(model_asset_path=pose_model_path)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base, output_segmentation_masks=False
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    hand_base = python.BaseOptions(model_asset_path=hand_model_path)
    hand_options = vision.HandLandmarkerOptions(base_options=hand_base, num_hands=2)
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    pose_connections = [
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 12),
        (23, 24),
        (11, 23),
        (12, 24),
        (23, 25),
        (25, 27),
        (24, 26),
        (26, 28),
        (27, 29),
        (29, 31),
        (28, 30),
        (30, 32),
    ]

    hand_connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]

    line_color = (0, 255, 0)
    point_color = (0, 255, 0)

    cap = open_camera(args.camera, args.source)
    print(f"Using camera mode: {args.camera}, source: {args.source}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        pose_result = pose_detector.detect(mp_image)
        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks[0]
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, point_color, -1)
            for start_idx, end_idx in pose_connections:
                x1 = int(landmarks[start_idx].x * w)
                y1 = int(landmarks[start_idx].y * h)
                x2 = int(landmarks[end_idx].x * w)
                y2 = int(landmarks[end_idx].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)

        hand_result = hand_detector.detect(mp_image)
        if hand_result.hand_landmarks:
            for hand in hand_result.hand_landmarks:
                for lm in hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, point_color, -1)
                for start_idx, end_idx in hand_connections:
                    x1 = int(hand[start_idx].x * w)
                    y1 = int(hand[start_idx].y * h)
                    x2 = int(hand[end_idx].x * w)
                    y2 = int(hand[end_idx].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)

        frame_features = []
        if pose_result.pose_landmarks:
            for lm in pose_result.pose_landmarks[0]:
                frame_features.extend([lm.x, lm.y])
        else:
            frame_features.extend([0.0] * (33 * 2))

        if hand_result.hand_landmarks:
            for hand in hand_result.hand_landmarks:
                for lm in hand:
                    frame_features.extend([lm.x, lm.y])
            if len(hand_result.hand_landmarks) == 1:
                frame_features.extend([0.0] * (21 * 2))
        else:
            frame_features.extend([0.0] * (21 * 2 * 2))

        sequence_buffer.append(frame_features)

        if len(sequence_buffer) == 30:
            seq = np.array(sequence_buffer)
            if seq.shape[1] == num_features:
                gesture_id = predict_sequence(model, seq)
                gesture_name = gesture_labels.get(gesture_id, str(gesture_id))
                print("Condition:", gesture_name)
                cv2.putText(
                    frame,
                    f"Condition: {gesture_name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
            else:
                print("Invalid sequence shape:", seq.shape)
            sequence_buffer = []

        cv2.imshow("Pose + Hand Skeleton + Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
