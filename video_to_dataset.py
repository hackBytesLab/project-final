import os
import argparse
import json
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

POSE_MODEL_PATH = 'models/pose_landmarker_lite.task'
HAND_MODEL_PATH = 'models/hand_landmarker.task'
DEFAULT_LABELS = 'Fall,No_Fall,Pre-Fall,Falling'


def create_detectors():
    pose_base = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base, output_segmentation_masks=False
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    hand_base = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    hand_options = vision.HandLandmarkerOptions(base_options=hand_base, num_hands=2)
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)
    return pose_detector, hand_detector


def extract_frame_features(frame, pose_detector, hand_detector):
    h, w, _ = frame.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    pose_result = pose_detector.detect(mp_image)
    hand_result = hand_detector.detect(mp_image)

    features = []

    # Pose (33 points x,y)
    if pose_result.pose_landmarks:
        for lm in pose_result.pose_landmarks[0]:
            features.extend([lm.x, lm.y])
    else:
        features.extend([0.0] * (33 * 2))

    # Hands (2 hands x 21 points x,y)
    if hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            for lm in hand:
                features.extend([lm.x, lm.y])
        if len(hand_result.hand_landmarks) == 1:
            features.extend([0.0] * (21 * 2))
    else:
        features.extend([0.0] * (21 * 2 * 2))

    return features


def parse_labels(raw_labels):
    labels = [x.strip() for x in (raw_labels or '').split(',') if x.strip()]
    if not labels:
        raise ValueError('At least one label is required in --labels')
    return labels


def process_videos(input_dir, output_dir, timesteps=30, step=15, labels=None):
    pose_detector, hand_detector = create_detectors()
    classes = labels if labels else sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    if not classes:
        raise ValueError('No class subfolders found in input directory')

    X = []
    y = []
    class_map = {cls: idx for idx, cls in enumerate(classes)}
    sample_counts = {cls: 0 for cls in classes}

    for cls in classes:
        cls_dir = os.path.join(input_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"Warning: class folder not found, skipped: {cls_dir}")
            continue
        for fname in os.listdir(cls_dir):
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
                    feats = extract_frame_features(frame, pose_detector, hand_detector)
                except Exception as e:
                    print('Warning: detection failed on frame:', e)
                    feats = [0.0] * 150
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

            print(f"Processed {path}: frames={n_frames}, samples={(n_frames - timesteps + 1 + (step-1))//step}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    with open(os.path.join(output_dir, 'class_map.json'), 'w', encoding='utf-8') as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    print('Saved dataset ->', output_dir)
    print('Classes:', class_map)
    print('X shape:', X.shape)
    print('y shape:', y.shape)
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

    args = parser.parse_args()
    labels = parse_labels(args.labels) if args.labels else None
    process_videos(args.input, args.output, timesteps=args.timesteps, step=args.step, labels=labels)


if __name__ == '__main__':
    main()
