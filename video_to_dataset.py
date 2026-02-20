import os
import argparse
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Uses the same feature extraction logic as main.py
pose_model_path = 'models/pose_landmarker_lite.task'
hand_model_path = 'models/hand_landmarker.task'

pose_base = python.BaseOptions(model_asset_path=pose_model_path)
pose_options = vision.PoseLandmarkerOptions(base_options=pose_base,
                                            output_segmentation_masks=False)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

hand_base = python.BaseOptions(model_asset_path=hand_model_path)
hand_options = vision.HandLandmarkerOptions(base_options=hand_base, num_hands=2)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)


def extract_frame_features(frame):
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


def process_videos(input_dir, output_dir, timesteps=30, step=15):
    classes = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    if not classes:
        raise ValueError('No class subfolders found in input directory')

    X = []
    y = []
    class_map = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        cls_dir = os.path.join(input_dir, cls)
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
                    feats = extract_frame_features(frame)
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

            print(f"Processed {path}: frames={n_frames}, samples={(n_frames - timesteps + 1 + (step-1))//step}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)

    print('Saved dataset ->', output_dir)
    print('Classes:', class_map)
    print('X shape:', X.shape)
    print('y shape:', y.shape)


def main():
    parser = argparse.ArgumentParser(description='Convert labeled videos into dataset for LSTM')
    parser.add_argument('--input', required=True, help='Input folder with subfolders per class (videos)')
    parser.add_argument('--output', default='data', help='Output folder to save X.npy and y.npy')
    parser.add_argument('--timesteps', type=int, default=30)
    parser.add_argument('--step', type=int, default=15, help='Sliding window step (default 50% overlap)')

    args = parser.parse_args()
    process_videos(args.input, args.output, timesteps=args.timesteps, step=args.step)


if __name__ == '__main__':
    main()
