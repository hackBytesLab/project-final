import os
import argparse
import numpy as np
import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

POSE_MODEL_PATH = 'models/pose_landmarker_lite.task'
HAND_MODEL_PATH = 'models/hand_landmarker.task'


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


def infer_on_video(video_path, model_path, out_csv, timesteps=30, step=1, batch_size=64, labels_map=None, out_video=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError('Model file not found: ' + model_path)

    model = load_model(model_path)
    pose_detector, hand_detector = create_detectors()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video: ' + video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Read all frames and extract features
    features = []
    frames = []
    print('Extracting features from video...')
    success = True
    while success:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)
        try:
            feats = extract_frame_features(frame, pose_detector, hand_detector)
        except Exception:
            feats = [0.0] * 150
        features.append(feats)

    n_frames = len(features)
    if n_frames == 0:
        raise RuntimeError('No frames extracted')

    # Build sliding windows
    windows = []
    centers = []
    for i in range(0, n_frames - timesteps + 1, step):
        seq = features[i:i + timesteps]
        windows.append(seq)
        centers.append(i + timesteps // 2)

    if len(windows) == 0:
        print('Video too short for given timesteps:', n_frames)
        cap.release()
        return

    X = np.array(windows, dtype=np.float32)
    print('Prepared', X.shape[0], 'windows, running inference...')

    preds = []
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:i + batch_size]
        p = model.predict(batch)
        preds.append(p)
    preds = np.vstack(preds)
    pred_ids = preds.argmax(axis=1)
    pred_scores = preds.max(axis=1)

    # Map predictions to per-frame labels using centers
    frame_labels = [None] * n_frames
    frame_scores = [0.0] * n_frames
    for cid, pid, score in zip(centers, pred_ids, pred_scores):
        frame_labels[cid] = int(pid)
        frame_scores[cid] = float(score)

    # Fill missing frame labels by nearest assigned label
    last_label = None
    for i in range(n_frames):
        if frame_labels[i] is None:
            frame_labels[i] = last_label
        else:
            last_label = frame_labels[i]
    # forward-fill then backward-fill if needed
    for i in range(n_frames - 1, -1, -1):
        if frame_labels[i] is None:
            frame_labels[i] = frame_labels[i + 1] if i + 1 < n_frames else 0

    # Convert frame labels to segments (merge consecutive equal labels)
    segments = []
    cur_label = frame_labels[0]
    start = 0
    for i in range(1, n_frames):
        if frame_labels[i] != cur_label:
            segments.append((cur_label, start, i - 1))
            cur_label = frame_labels[i]
            start = i
    segments.append((cur_label, start, n_frames - 1))

    # Write CSV: class_id, class_name, start_time, end_time, avg_score
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'class_name', 'start_time_s', 'end_time_s', 'avg_score'])
        for seg in segments:
            cid, sfrm, efrm = seg
            scores = frame_scores[sfrm:efrm + 1]
            avg_score = float(np.mean([sc for sc in scores if sc > 0])) if any(sc > 0 for sc in scores) else 0.0
            cls_name = labels_map[cid] if labels_map and cid in labels_map else str(cid)
            writer.writerow([cid, cls_name, sfrm / fps, efrm / fps, avg_score])

    print('Saved segments to', out_csv)

    # Optionally write visualization video with overlay labels
    if out_video:
        print('Rendering labeled video ->', out_video)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(out_video, fourcc, fps, (w, h))
        for i, frame in enumerate(frames):
            lbl = frame_labels[i]
            txt = labels_map[lbl] if labels_map and lbl in labels_map else str(lbl)
            cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            out.write(frame)
        out.release()
        print('Saved labeled video')

    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer long video and output labeled segments using trained model')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--model', required=True, help='Trained Keras model (.h5)')
    parser.add_argument('--out-csv', default='segments.csv', help='Output CSV with segments')
    parser.add_argument('--timesteps', type=int, default=30)
    parser.add_argument('--step', type=int, default=1, help='Step when creating windows (1 => every frame center)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--labels', help='Optional comma-separated class names in order (e.g. Fall,No_Fall,Pre-Fall,Falling)')
    parser.add_argument('--out-video', help='Optional path to save visualization video')

    args = parser.parse_args()

    label_map = None
    if args.labels:
        parts = [p.strip() for p in args.labels.split(',')]
        label_map = {i: parts[i] for i in range(len(parts))}

    infer_on_video(args.video, args.model, args.out_csv, timesteps=args.timesteps, step=args.step, batch_size=args.batch_size, labels_map=label_map, out_video=args.out_video)
