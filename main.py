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

from feature_layout import build_frame_features_with_options, resolve_feature_layout
from lstm_model import predict_sequence

try:
    from Iriun_Webcam import get_iriun_camera
except ImportError:
    get_iriun_camera = None


DEFAULT_LABELS = "Fall,No_Fall,Pre-Fall,Falling"
TIMESTEPS = 30
SINGLE_PERSON_FEATURES = 150


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


def parse_env_float(name, default):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_env_bool(name, default=False):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
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


def predict_sequence_with_score(model, sequence):
    sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
    prediction = model.predict(sequence, verbose=0)[0]
    return int(np.argmax(prediction)), float(np.max(prediction))


def sort_pose_landmarks_by_x(pose_landmarks):
    if not pose_landmarks:
        return []

    def center_x(landmarks):
        return float(np.mean([lm.x for lm in landmarks]))

    return sorted(pose_landmarks, key=center_x)


def assign_hands_to_poses(pose_landmarks, hand_landmarks, max_hands_per_pose=2):
    assignments = [[] for _ in range(len(pose_landmarks))]
    if not pose_landmarks or not hand_landmarks:
        return assignments

    pose_refs = []
    for pose in pose_landmarks:
        left_wrist = pose[15]
        right_wrist = pose[16]
        pose_refs.append(
            (
                (left_wrist.x + right_wrist.x) / 2.0,
                (left_wrist.y + right_wrist.y) / 2.0,
            )
        )

    for hand in hand_landmarks:
        wrist = hand[0]
        hx, hy = wrist.x, wrist.y
        best_idx = None
        best_dist = None
        for idx, (px, py) in enumerate(pose_refs):
            dist = (hx - px) ** 2 + (hy - py) ** 2
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None:
            assignments[best_idx].append((best_dist, hand))

    result = []
    for group in assignments:
        group.sort(key=lambda x: x[0])
        result.append([hand for _, hand in group[:max_hands_per_pose]])
    return result


def pose_center(landmarks):
    if not landmarks:
        return (0.0, 0.0)
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (float(np.mean(xs)), float(np.mean(ys)))


def match_detections_to_tracks(detections, track_centers, max_distance):
    assignments = {}
    used_tracks = set()
    used_detections = set()
    candidates = []

    for track_idx, center in enumerate(track_centers):
        if center is None:
            continue
        for det_idx, det in enumerate(detections):
            dx = center[0] - det["center"][0]
            dy = center[1] - det["center"][1]
            dist = float(np.sqrt(dx * dx + dy * dy))
            candidates.append((dist, track_idx, det_idx))

    candidates.sort(key=lambda x: x[0])
    for dist, track_idx, det_idx in candidates:
        if dist > max_distance:
            continue
        if track_idx in used_tracks or det_idx in used_detections:
            continue
        assignments[track_idx] = det_idx
        used_tracks.add(track_idx)
        used_detections.add(det_idx)

    free_tracks = [i for i, center in enumerate(track_centers) if center is None and i not in used_tracks]
    for det_idx in range(len(detections)):
        if det_idx in used_detections:
            continue
        if not free_tracks:
            break
        track_idx = free_tracks.pop(0)
        assignments[track_idx] = det_idx
        used_tracks.add(track_idx)
        used_detections.add(det_idx)

    return assignments


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
    parser.add_argument(
        "--max-people",
        type=int,
        default=parse_env_int("MAX_PEOPLE", 0),
        help="People slots in feature vector (0=auto from model input shape).",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=parse_env_int("MAX_HANDS", 0),
        help="Hand slots in feature vector (0=2*max-people).",
    )
    parser.add_argument(
        "--detect-people",
        type=int,
        default=parse_env_int("DETECT_PEOPLE", 4),
        help="Maximum people to visualize per frame.",
    )
    parser.add_argument(
        "--inference-mode",
        default=os.getenv("INFERENCE_MODE", "auto"),
        choices=["auto", "frame", "per-person"],
        help="auto=per-person for 150-feature model when detect-people>1, otherwise frame mode.",
    )
    parser.add_argument(
        "--normalize-geometry",
        default=parse_env_bool("NORMALIZE_GEOMETRY", False),
        action="store_true",
        help="Normalize pose/hand geometry per entity before inference.",
    )
    parser.add_argument(
        "--track-max-distance",
        type=float,
        default=parse_env_float("TRACK_MAX_DISTANCE", 0.20),
        help="Max normalized center distance for track association in per-person mode.",
    )
    parser.add_argument(
        "--track-max-missed",
        type=int,
        default=parse_env_int("TRACK_MAX_MISSED", 15),
        help="Frames to keep an unmatched track before reset in per-person mode.",
    )
    args = parser.parse_args()

    # Feature and class config
    model = load_model(args.model)
    model_input_shape = model.input_shape
    if isinstance(model_input_shape, list):
        model_input_shape = model_input_shape[0]
    if not model_input_shape or model_input_shape[-1] is None:
        raise ValueError(f"Unsupported model input shape: {model_input_shape}")
    num_features = int(model_input_shape[-1])
    feature_max_people, feature_max_hands = resolve_feature_layout(
        num_features=num_features,
        max_people_arg=args.max_people,
        max_hands_arg=args.max_hands,
    )
    detect_people = max(feature_max_people, max(1, args.detect_people))
    supports_single_person_model = (
        num_features == SINGLE_PERSON_FEATURES
        and feature_max_people == 1
        and feature_max_hands == 2
    )
    if args.inference_mode == "auto":
        inference_mode = "per-person" if (supports_single_person_model and detect_people > 1) else "frame"
    else:
        inference_mode = args.inference_mode

    if inference_mode == "per-person" and not supports_single_person_model:
        raise ValueError(
            "per-person mode requires a single-person model (150 features, 1 pose slot, 2 hand slots). "
            "Use --inference-mode frame, or use a 150-feature model."
        )

    detect_hands = feature_max_hands
    if inference_mode == "per-person":
        detect_hands = max(detect_hands, detect_people * 2)

    sequence_buffer = []
    track_centers = [None] * detect_people
    track_missed = [0] * detect_people
    track_buffers = [[] for _ in range(detect_people)]
    track_labels = [""] * detect_people

    label_names = parse_labels(args.labels)
    gesture_labels = {i: name for i, name in enumerate(label_names)}

    pose_model_path = "models/pose_landmarker_lite.task"
    hand_model_path = "models/hand_landmarker.task"

    pose_base = python.BaseOptions(model_asset_path=pose_model_path)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base,
        output_segmentation_masks=False,
        num_poses=detect_people,
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    hand_base = python.BaseOptions(model_asset_path=hand_model_path)
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base,
        num_hands=detect_hands,
    )
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
    print(
        f"[Feature layout] num_features={num_features}, "
        f"max_people={feature_max_people}, max_hands={feature_max_hands}, "
        f"detect_people={detect_people}, detect_hands={detect_hands}, "
        f"inference_mode={inference_mode}, normalize_geometry={args.normalize_geometry}, "
        f"track_max_distance={args.track_max_distance}, track_max_missed={args.track_max_missed}"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        pose_result = pose_detector.detect(mp_image)
        pose_landmarks = sort_pose_landmarks_by_x(
            list((pose_result.pose_landmarks or [])[:detect_people])
        )
        if pose_landmarks:
            for landmarks in pose_landmarks:
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
        hand_landmarks = hand_result.hand_landmarks or []
        if hand_landmarks:
            for hand in hand_landmarks:
                for lm in hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, point_color, -1)
                for start_idx, end_idx in hand_connections:
                    x1 = int(hand[start_idx].x * w)
                    y1 = int(hand[start_idx].y * h)
                    x2 = int(hand[end_idx].x * w)
                    y2 = int(hand[end_idx].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)

        if inference_mode == "per-person":
            hand_assignments = assign_hands_to_poses(
                pose_landmarks,
                hand_landmarks,
                max_hands_per_pose=2,
            )
            detections = []
            for person_idx, person_pose in enumerate(pose_landmarks):
                person_hands = hand_assignments[person_idx] if person_idx < len(hand_assignments) else []
                detections.append(
                    {
                        "pose": person_pose,
                        "hands": person_hands,
                        "center": pose_center(person_pose),
                    }
                )

            track_assignments = match_detections_to_tracks(
                detections=detections,
                track_centers=track_centers,
                max_distance=args.track_max_distance,
            )
            det_to_track = {det_idx: track_idx for track_idx, det_idx in track_assignments.items()}

            for track_idx in range(detect_people):
                if track_idx not in track_assignments:
                    track_missed[track_idx] += 1
                    if track_missed[track_idx] > args.track_max_missed:
                        track_centers[track_idx] = None
                        track_buffers[track_idx] = []
                        track_labels[track_idx] = ""
                        track_missed[track_idx] = 0
                    continue

                det_idx = track_assignments[track_idx]
                det = detections[det_idx]
                track_centers[track_idx] = det["center"]
                track_missed[track_idx] = 0
                person_features = build_frame_features_with_options(
                    [det["pose"]],
                    det["hands"],
                    max_people=1,
                    max_hands=2,
                    normalize_geometry=args.normalize_geometry,
                )
                track_buffers[track_idx].append(person_features)

                if len(track_buffers[track_idx]) == TIMESTEPS:
                    seq = np.array(track_buffers[track_idx], dtype=np.float32)
                    if seq.shape[1] == num_features:
                        gesture_id, gesture_score = predict_sequence_with_score(model, seq)
                        gesture_name = gesture_labels.get(gesture_id, str(gesture_id))
                        track_labels[track_idx] = f"{gesture_name} {gesture_score:.2f}"
                        print(f"Condition[P{track_idx + 1}]: {gesture_name} ({gesture_score:.2f})")

                        if args.line_token and gesture_name in alert_classes:
                            now = time.time()
                            if now - last_alert_time >= args.line_cooldown_seconds:
                                alert_message = (
                                    f"ALERT: Person {track_idx + 1} detected {gesture_name} at "
                                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
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
                    else:
                        print(f"Invalid sequence shape for person {track_idx + 1}:", seq.shape)
                    track_buffers[track_idx] = []

            for det_idx, det in enumerate(detections):
                if det_idx not in det_to_track:
                    continue
                track_idx = det_to_track[det_idx]
                label_text = track_labels[track_idx] if track_labels[track_idx] else "..."
                anchor_x = int(det["pose"][0].x * w)
                anchor_y = max(20, int(det["pose"][0].y * h) - 10)
                cv2.putText(
                    frame,
                    f"P{track_idx + 1}: {label_text}",
                    (anchor_x, anchor_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
        else:
            frame_features = build_frame_features_with_options(
                pose_landmarks,
                hand_landmarks,
                max_people=feature_max_people,
                max_hands=feature_max_hands,
                normalize_geometry=args.normalize_geometry,
            )

            sequence_buffer.append(frame_features)

            if len(sequence_buffer) == TIMESTEPS:
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
