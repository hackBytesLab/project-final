import argparse
from contextlib import contextmanager
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from feature_layout import (
    build_frame_features_with_options,
    enhance_sequence_features,
    infer_enhancement_variant,
)
from rule_fusion import DEFAULT_THRESHOLDS, fuse_rule_with_lstm, separate_prefall_falling


DEFAULT_MODEL_PATH = "models/lstm_all_combined_216_sub8k_prefallboost_pi5.tflite"
DEFAULT_POSE_MODEL = "models/pose_landmarker_lite.task"
DEFAULT_HAND_MODEL = "models/hand_landmarker.task"
DEFAULT_LABELS = "Fall,No_Fall,Pre-Fall,Falling"
DEFAULT_TIMESTEPS = 30
BASE_FEATURES = 150


@contextmanager
def silence_native_stderr():
    try:
        stderr_fd = sys.stderr.fileno()
        saved_fd = os.dup(stderr_fd)
    except (AttributeError, OSError, ValueError):
        yield
        return

    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        try:
            os.dup2(saved_fd, stderr_fd)
        finally:
            os.close(saved_fd)


def parse_labels(raw):
    labels = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one label is required")
    return labels


def load_tflite_interpreter(model_path, num_threads):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter

    with silence_native_stderr():
        interpreter = Interpreter(model_path=str(model_path), num_threads=int(num_threads))
        interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_shape = input_details.get("shape_signature", input_details.get("shape"))
    if len(input_shape) != 3:
        raise ValueError(f"Expected 3D input shape, got {input_shape}")
    timesteps = int(input_shape[1]) if int(input_shape[1]) > 0 else DEFAULT_TIMESTEPS
    num_features = int(input_shape[2])
    return {
        "interpreter": interpreter,
        "input_details": input_details,
        "output_details": output_details,
        "timesteps": timesteps,
        "num_features": num_features,
    }


def predict_tflite(model_info, batch):
    interpreter = model_info["interpreter"]
    input_details = model_info["input_details"]
    output_details = model_info["output_details"]
    input_index = int(input_details["index"])

    input_dtype = input_details["dtype"]
    input_scale, input_zero = input_details.get("quantization", (0.0, 0))
    tensor = batch.astype(np.float32, copy=False)
    if input_dtype in (np.int8, np.uint8) and input_scale:
        tensor = np.round(tensor / input_scale + input_zero).astype(input_dtype)
    else:
        tensor = tensor.astype(input_dtype, copy=False)

    interpreter.set_tensor(input_index, tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(int(output_details["index"]))

    output_dtype = output_details["dtype"]
    output_scale, output_zero = output_details.get("quantization", (0.0, 0))
    if output_dtype in (np.int8, np.uint8) and output_scale:
        output = (output.astype(np.float32) - float(output_zero)) * float(output_scale)
    return output


def create_detectors(pose_model_path, hand_model_path):
    pose_base = python.BaseOptions(model_asset_path=str(pose_model_path))
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base,
        output_segmentation_masks=False,
        num_poses=1,
    )
    with silence_native_stderr():
        pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    hand_base = python.BaseOptions(model_asset_path=str(hand_model_path))
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base,
        num_hands=2,
    )
    with silence_native_stderr():
        hand_detector = vision.HandLandmarker.create_from_options(hand_options)
    return pose_detector, hand_detector


def open_camera(args):
    if args.camera == "picamera2":
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise RuntimeError(
                "picamera2 is not installed. Use --camera index or install picamera2 on the Pi."
            ) from e

        camera = Picamera2()
        config = camera.create_preview_configuration(
            main={"format": "RGB888", "size": (args.width, args.height)}
        )
        camera.configure(config)
        camera.start()
        time.sleep(0.5)
        return {"backend": "picamera2", "camera": camera}

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera_index}")
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    return {"backend": "opencv", "camera": cap}


def read_frame(camera_info):
    if camera_info["backend"] == "picamera2":
        frame_rgb = camera_info["camera"].capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return True, frame_bgr, frame_rgb

    ok, frame_bgr = camera_info["camera"].read()
    if not ok:
        return False, None, None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return True, frame_bgr, frame_rgb


def close_camera(camera_info):
    if camera_info["backend"] == "picamera2":
        camera_info["camera"].stop()
        return
    camera_info["camera"].release()


def label_color(label_name):
    normalized = str(label_name).strip().lower()
    if normalized in ("fall", "falling"):
        return (0, 0, 255)
    if normalized in ("pre-fall", "pre_fall"):
        return (0, 165, 255)
    return (0, 200, 0)


def main():
    parser = argparse.ArgumentParser(description="Realtime Pi 5 TFLite runtime for fall detection")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to Pi 5 friendly .tflite model")
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL, help="Path to pose landmarker task file")
    parser.add_argument("--hand-model", default=DEFAULT_HAND_MODEL, help="Path to hand landmarker task file")
    parser.add_argument("--camera", choices=["picamera2", "index"], default="index", help="Camera backend")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index when --camera index")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS")
    parser.add_argument("--model-threads", type=int, default=4, help="TFLite interpreter threads")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Comma-separated class labels")
    parser.add_argument("--smooth-window", type=int, default=5, help="Mean smoothing over recent predictions")
    parser.add_argument("--infer-every", type=int, default=1, help="Run inference every N frames once buffer is full")
    parser.add_argument(
        "--normalize-geometry",
        action="store_true",
        help="Normalize landmarks before feature extraction",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV preview window for headless runs",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    pose_model_path = Path(args.pose_model)
    hand_model_path = Path(args.hand_model)
    for path in (model_path, pose_model_path, hand_model_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    labels = parse_labels(args.labels)
    model_info = load_tflite_interpreter(model_path, num_threads=args.model_threads)
    if model_info["timesteps"] != DEFAULT_TIMESTEPS:
        raise ValueError(f"Expected timesteps={DEFAULT_TIMESTEPS}, got {model_info['timesteps']}")
    enhancement_variant = infer_enhancement_variant(model_info["num_features"], BASE_FEATURES)
    if enhancement_variant not in ("velocity", "full"):
        raise ValueError(
            f"Expected enhanced feature count derived from base={BASE_FEATURES}, got {model_info['num_features']}"
        )
    use_summary_features = enhancement_variant == "full"

    pose_detector, hand_detector = create_detectors(pose_model_path, hand_model_path)
    camera_info = open_camera(args)

    sequence_buffer = deque(maxlen=model_info["timesteps"])
    prob_history = deque(maxlen=max(1, int(args.smooth_window)))
    current_label = "warming up"
    current_score = 0.0
    frame_count = 0
    fps_window_start = time.time()
    fps_counter = 0
    current_fps = 0.0

    print(
        f"[Pi5 runtime] model={model_path} camera={args.camera} "
        f"timesteps={model_info['timesteps']} features={model_info['num_features']} "
        f"enhancement_variant={enhancement_variant}"
    )

    try:
        while True:
            ok, frame_bgr, frame_rgb = read_frame(camera_info)
            if not ok:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            pose_result = pose_detector.detect(mp_image)
            hand_result = hand_detector.detect(mp_image)

            pose_landmarks = pose_result.pose_landmarks or []
            hand_landmarks = hand_result.hand_landmarks or []
            frame_features = build_frame_features_with_options(
                pose_landmarks=pose_landmarks[:1],
                hand_landmarks=hand_landmarks[:2],
                max_people=1,
                max_hands=2,
                normalize_geometry=args.normalize_geometry,
            )
            if len(frame_features) != BASE_FEATURES:
                raise ValueError(f"Expected {BASE_FEATURES} base features, got {len(frame_features)}")

            sequence_buffer.append(frame_features)
            frame_count += 1

            if len(sequence_buffer) == model_info["timesteps"] and (frame_count % max(1, args.infer_every) == 0):
                sequence = np.asarray(sequence_buffer, dtype=np.float32)
                enhanced_sequence = enhance_sequence_features(
                    sequence,
                    include_summary_features=use_summary_features,
                )
                if enhanced_sequence.shape != (DEFAULT_TIMESTEPS, model_info["num_features"]):
                    raise ValueError(
                        f"Expected enhanced sequence shape {(DEFAULT_TIMESTEPS, model_info['num_features'])}, "
                        f"got {enhanced_sequence.shape}"
                    )

                probs = predict_tflite(model_info, enhanced_sequence[None, ...])[0]
                prob_history.append(probs)
                smoothed_probs = np.mean(np.stack(prob_history, axis=0), axis=0)
                fused_id, fused_name, _ = fuse_rule_with_lstm(
                    smoothed_probs,
                    labels,
                    enhanced_sequence,
                    thresholds=DEFAULT_THRESHOLDS,
                )
                override_id = None
                if fused_name.strip().lower() in ("pre-fall", "pre_fall", "falling"):
                    override_id = separate_prefall_falling(enhanced_sequence, labels)
                final_id = override_id if override_id is not None else fused_id
                current_label = labels[final_id] if final_id < len(labels) else str(final_id)
                current_score = float(smoothed_probs[final_id])
                print(f"Condition: {current_label} ({current_score:.2f})")

            fps_counter += 1
            now = time.time()
            elapsed = now - fps_window_start
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_window_start = now

            if not args.no_display:
                color = label_color(current_label)
                cv2.putText(
                    frame_bgr,
                    f"Condition: {current_label} ({current_score:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    f"FPS: {current_fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Pi 5 Fall Detection", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        close_camera(camera_info)
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
