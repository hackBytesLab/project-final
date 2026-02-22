import os
import argparse
import json
import csv
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from feature_layout import build_frame_features_with_options, compute_num_features

POSE_MODEL_PATH = 'models/pose_landmarker_lite.task'
HAND_MODEL_PATH = 'models/hand_landmarker.task'
DEFAULT_LABELS = 'Fall,No_Fall,Pre-Fall,Falling'


def create_detectors(max_people=1, max_hands=2):
    pose_base = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base, output_segmentation_masks=False, num_poses=max_people
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    hand_base = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    hand_options = vision.HandLandmarkerOptions(base_options=hand_base, num_hands=max_hands)
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)
    return pose_detector, hand_detector


def extract_frame_features(
    frame,
    pose_detector,
    hand_detector,
    max_people,
    max_hands,
    normalize_geometry=False,
):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    pose_result = pose_detector.detect(mp_image)
    hand_result = hand_detector.detect(mp_image)
    return build_frame_features_with_options(
        pose_result.pose_landmarks,
        hand_result.hand_landmarks,
        max_people=max_people,
        max_hands=max_hands,
        normalize_geometry=normalize_geometry,
    )


def parse_labels(raw_labels):
    labels = [x.strip() for x in (raw_labels or '').split(',') if x.strip()]
    if not labels:
        raise ValueError('At least one label is required in --labels')
    return labels


def infer_source_video_id(filename):
    stem = os.path.splitext(filename)[0]
    marker = "__seg_"
    if marker in stem:
        return stem.split(marker)[0]
    return stem


def process_videos(
    input_dir,
    output_dir,
    timesteps=30,
    step=15,
    labels=None,
    max_people=1,
    max_hands=0,
    normalize_geometry=False,
):
    effective_max_hands = max_hands if max_hands > 0 else max_people * 2
    num_features = compute_num_features(max_people=max_people, max_hands=effective_max_hands)
    pose_detector, hand_detector = create_detectors(
        max_people=max_people,
        max_hands=effective_max_hands,
    )
    classes = labels if labels else sorted(
        [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    )
    if not classes:
        raise ValueError('No class subfolders found in input directory')

    X = []
    y = []
    sample_meta = []
    class_map = {cls: idx for idx, cls in enumerate(classes)}
    sample_counts = {cls: 0 for cls in classes}
    sample_idx = 0

    for cls in classes:
        cls_dir = os.path.join(input_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"Warning: class folder not found, skipped: {cls_dir}")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                continue
            path = os.path.join(cls_dir, fname)
            cap = cv2.VideoCapture(path)
            frames_features = []
            success = True
            while success:
                success, frame = cap.read()
                if not success:
                    break
                try:
                    feats = extract_frame_features(
                        frame,
                        pose_detector,
                        hand_detector,
                        max_people=max_people,
                        max_hands=effective_max_hands,
                        normalize_geometry=normalize_geometry,
                    )
                except Exception as e:
                    print('Warning: detection failed on frame:', e)
                    feats = [0.0] * num_features
                frames_features.append(feats)
            cap.release()

            n_frames = len(frames_features)
            if n_frames < timesteps:
                print(f"Skipping {path} (too short: {n_frames} frames)")
                continue

            # sliding window
            for i in range(0, n_frames - timesteps + 1, step):
                seq = frames_features[i:i + timesteps]
                X.append(seq)
                y.append(class_map[cls])
                sample_counts[cls] += 1
                clip_rel = os.path.relpath(path, input_dir).replace("\\", "/")
                source_video = infer_source_video_id(fname)
                sample_meta.append(
                    {
                        "sample_idx": sample_idx,
                        "class_id": class_map[cls],
                        "class_name": cls,
                        "clip_path": clip_rel,
                        "source_video": source_video,
                        "window_start": i,
                        "window_end": i + timesteps - 1,
                        "group_id": clip_rel,
                    }
                )
                sample_idx += 1

            print(f"Processed {path}: frames={n_frames}, samples={(n_frames - timesteps + 1 + (step-1))//step}")

    if X:
        X = np.array(X, dtype=np.float32)
    else:
        X = np.empty((0, timesteps, num_features), dtype=np.float32)
    y = np.array(y, dtype=np.int32) if y else np.empty((0,), dtype=np.int32)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    with open(os.path.join(output_dir, 'class_map.json'), 'w', encoding='utf-8') as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)
    meta_path = os.path.join(output_dir, 'sample_meta.csv')
    with open(meta_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_idx",
                "class_id",
                "class_name",
                "clip_path",
                "source_video",
                "window_start",
                "window_end",
                "group_id",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_meta)

    print('Saved dataset ->', output_dir)
    print('Classes:', class_map)
    print('X shape:', X.shape)
    print('y shape:', y.shape)
    print('sample_meta:', meta_path)
    for cls, count in sample_counts.items():
        if count == 0:
            print(f"Warning: no training samples produced for class '{cls}'")


def main():
    parser = argparse.ArgumentParser(description='Convert labeled videos into dataset for LSTM')
    parser.add_argument('--input', required=True, help='Input folder with subfolders per class (videos)')
    parser.add_argument('--output', default='data', help='Output folder to save X.npy and y.npy')
    parser.add_argument('--timesteps', type=int, default=30)
    parser.add_argument('--step', type=int, default=15, help='Sliding window step (default 50%% overlap)')
    parser.add_argument('--labels', default=DEFAULT_LABELS, help='Comma-separated class names in fixed output order')
    parser.add_argument('--max-people', type=int, default=1, help='Pose slots used in features')
    parser.add_argument('--max-hands', type=int, default=0, help='Hand slots used in features (0=2*max-people)')
    parser.add_argument(
        '--normalize-geometry',
        action='store_true',
        help='Normalize pose/hand geometry per entity before saving features.',
    )

    args = parser.parse_args()
    labels = parse_labels(args.labels) if args.labels else None
    process_videos(
        args.input,
        args.output,
        timesteps=args.timesteps,
        step=args.step,
        labels=labels,
        max_people=args.max_people,
        max_hands=args.max_hands,
        normalize_geometry=args.normalize_geometry,
    )


if __name__ == '__main__':
    main()
