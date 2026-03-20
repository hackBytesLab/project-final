"""
Auto-label a folder of sequential PNG/JPG frames into LSTM-ready X.npy / y.npy using an existing 4-class model.

Usage:
python auto_label_images.py --frames-dir tmp_subjects \
    --model models/lstm_aug2000.h5 \
    --out data_autolabel_tmp_subjects \
    --labels Fall,No_Fall,Pre-Fall,Falling \
    --sequence-length 30
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from feature_layout import (
    build_frame_features_with_options,
    enhance_sequence_features,
    resolve_feature_layout,
    compute_num_features,
    POSE_FEATURES_PER_PERSON,
    ENHANCED_EXTRA_FEATURES,
)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, help="Folder containing ordered PNG/JPG frames")
    ap.add_argument("--model", required=True, help="Trained LSTM model (.h5)")
    ap.add_argument("--out", default="data_autolabel", help="Output folder for X.npy / y.npy")
    ap.add_argument("--labels", default="Fall,No_Fall,Pre-Fall,Falling", help="Comma labels in class-id order")
    ap.add_argument("--sequence-length", type=int, default=30, help="Frames per sequence")
    ap.add_argument("--pose-model", default="models/pose_landmarker_lite.task", help="MediaPipe pose model path")
    return ap.parse_args()


def load_model_info(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    num_features = int(input_shape[-1])
    return model, num_features


def create_pose_detector(model_path, max_people=1):
    base = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base,
        output_segmentation_masks=False,
        num_poses=max_people,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_pose_features(detector, image_bgr, max_people, max_hands):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_bgr)
    result = detector.detect(mp_image)
    pose_landmarks = result.pose_landmarks or []
    hand_landmarks = []  # not using hands for now
    feats = build_frame_features_with_options(
        pose_landmarks=pose_landmarks,
        hand_landmarks=hand_landmarks,
        max_people=max_people,
        max_hands=max_hands,
        normalize_geometry=False,
    )
    return np.array(feats, dtype=np.float32)


def main():
    args = parse_args()
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    model, num_features = load_model_info(args.model)
    max_people, max_hands = resolve_feature_layout(num_features, max_people_arg=1, max_hands_arg=0)

    detector = create_pose_detector(args.pose_model, max_people=max_people)

    frame_paths = sorted(
        [p for p in Path(args.frames_dir).glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )
    if len(frame_paths) < args.sequence_length:
        raise SystemExit("Not enough frames for one sequence")

    sequences = []
    for i in range(0, len(frame_paths) - args.sequence_length + 1, args.sequence_length):
        seq_paths = frame_paths[i : i + args.sequence_length]
        frames = []
        for p in seq_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            feats = extract_pose_features(detector, img, max_people=max_people, max_hands=max_hands)
            frames.append(feats)
        if len(frames) == args.sequence_length:
            sequences.append(np.stack(frames, axis=0))

    if not sequences:
        raise SystemExit("No sequences produced.")

    X150 = np.stack(sequences, axis=0)  # (N, T, 150 or more)
    # enhance if model expects enhanced
    base_features = compute_num_features(max_people, max_hands)
    enhanced_expected = base_features + ENHANCED_EXTRA_FEATURES
    if num_features == enhanced_expected:
        X_model = np.stack([enhance_sequence_features(seq) for seq in X150], axis=0)
    else:
        X_model = X150

    probs = model.predict(X_model, verbose=0)
    preds = np.argmax(probs, axis=1).astype(np.int32)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X150)
    np.save(out_dir / "y.npy", preds)
    with (out_dir / "pred_labels.txt").open("w", encoding="utf-8") as f:
        for i, p in enumerate(preds):
            name = labels[p] if p < len(labels) else str(p)
            f.write(f"{i}\t{p}\t{name}\n")
    print("Saved", X150.shape, preds.shape, "to", out_dir)


if __name__ == "__main__":
    main()
