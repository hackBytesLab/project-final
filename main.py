import argparse
import json
import os
import time
import urllib.error
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

from lstm_model import predict_sequence

try:
    from Iriun_Webcam import get_iriun_camera
except ImportError:
    get_iriun_camera = None


DEFAULT_LABELS = "Fall,No_Fall,Pre-Fall,Falling"


def parse_labels(raw_labels):
    labels = [x.strip() for x in (raw_labels or "").split(",") if x.strip()]
    if not labels:
        raise ValueError("At least one class label is required")
    return labels


def parse_labels_or_empty(raw_labels):
    return [x.strip() for x in (raw_labels or "").split(",") if x.strip()]


def parse_env_int(name, default):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def load_env_defaults():
    # Support both standard .env and existing .evnv files.
    for path in (".env", ".evnv"):
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                if key:
                    os.environ.setdefault(key, value)


def send_line_alert(channel_access_token, message, to_user_id=None, timeout=10):
    if not channel_access_token:
        raise ValueError("LINE channel access token is required")

    if to_user_id:
        endpoint = "https://api.line.me/v2/bot/message/push"
        payload = {
            "to": to_user_id,
            "messages": [{"type": "text", "text": message}],
        }
    else:
        endpoint = "https://api.line.me/v2/bot/message/broadcast"
        payload = {"messages": [{"type": "text", "text": message}]}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {channel_access_token}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LINE API error {e.code}: {err_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"LINE API connection error: {e}") from e


def open_camera(camera_mode, source):
    if camera_mode == "iriun":
        if get_iriun_camera is None:
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                raise RuntimeError(
                    "Iriun mode selected but Iriun_Webcam.py is missing and camera index 1 cannot be opened."
                )
            return cap
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
    load_env_defaults()

    parser = argparse.ArgumentParser(description="Pose+Hand fall detection runtime")
    parser.add_argument(
        "--camera",
        default=os.getenv("CAMERA_MODE", "pi"),
        choices=["iriun", "pi", "rtsp", "index"],
        help="Camera source mode",
    )
    parser.add_argument(
        "--source",
        default=os.getenv("CAMERA_SOURCE"),
        help="RTSP URL for --camera rtsp, or camera index for --camera index",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_PATH", "models/lstm_fall_model.h5"),
        help="Path to trained model (.h5/.keras)",
    )
    parser.add_argument(
        "--line-token",
        default=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"),
        help="LINE Messaging API channel access token. If omitted, no LINE alert is sent.",
    )
    parser.add_argument(
        "--line-user-id",
        default=os.getenv("LINE_USER_ID"),
        help="Target LINE user/group/room ID for push API. If omitted, broadcast API is used.",
    )
    parser.add_argument(
        "--line-cooldown-seconds",
        type=int,
        default=parse_env_int("LINE_COOLDOWN_SECONDS", 60),
        help="Minimum seconds between LINE alerts.",
    )
    parser.add_argument(
        "--alert-classes",
        default=os.getenv("ALERT_CLASSES", "Fall"),
        help="Comma-separated class names that should trigger LINE alerts.",
    )
    parser.add_argument(
        "--labels",
        default=os.getenv("LABELS", DEFAULT_LABELS),
        help="Comma-separated class names in model output order.",
    )
    args = parser.parse_args()

    # Feature and class config
    num_features = 150  # (33 pose + 21*2 hands) * 2 coords(x,y)
    model = load_model(args.model)
    sequence_buffer = []

    label_names = parse_labels(args.labels)
    gesture_labels = {i: name for i, name in enumerate(label_names)}

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
    alert_classes = set(parse_labels_or_empty(args.alert_classes))
    last_alert_time = 0.0

    cap = open_camera(args.camera, args.source)
    print(f"Using camera mode: {args.camera}, source: {args.source}")
    if args.line_token:
        if args.line_user_id:
            print("[LINE] Alert enabled via push API")
        else:
            print("[LINE] Alert enabled via broadcast API")
    else:
        print("[LINE] Alert disabled (missing token)")

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

                if args.line_token and gesture_name in alert_classes:
                    now = time.time()
                    if now - last_alert_time >= args.line_cooldown_seconds:
                        alert_message = (
                            f"ALERT: Detected {gesture_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        try:
                            send_line_alert(
                                channel_access_token=args.line_token,
                                message=alert_message,
                                to_user_id=args.line_user_id,
                            )
                            last_alert_time = now
                            print("[LINE] Alert sent")
                        except Exception as e:
                            print(f"[LINE] Alert failed: {e}")

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
