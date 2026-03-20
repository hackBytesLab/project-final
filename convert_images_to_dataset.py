import argparse
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def parse_args():
    ap = argparse.ArgumentParser(description="Convert image dataset (png/jpg + YOLO txt) to X.npy/y.npy for LSTM")
    ap.add_argument("--root", required=True, help="Root folder containing images/train, images/val and labels/train, labels/val")
    ap.add_argument("--out", default="data_from_images", help="Output folder to write X.npy and y.npy")
    ap.add_argument("--frames", type=int, default=30, help="Number of timesteps to duplicate per image")
    ap.add_argument("--label-map", default="", help="Optional mapping '0:Fall,1:No_Fall,2:No_Fall'")
    return ap.parse_args()


def load_label_map(arg):
    default = {0: "Fall", 1: "No_Fall", 2: "No_Fall"}
    if not arg:
        return default
    mapping = {}
    for part in arg.split(","):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        try:
            mapping[int(k.strip())] = v.strip()
        except ValueError:
            continue
    return mapping or default


def image_paths(root):
    root = Path(root)
    for split in ["train", "val"]:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in img_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            yield split, img_path, lbl_path


def load_label(lbl_path, label_map):
    if not lbl_path.exists():
        return None
    with lbl_path.open() as f:
        line = f.readline().strip()
    if not line:
        return None
    cls_id = int(line.split()[0])
    return label_map.get(cls_id)


class PoseExtractor:
    def __init__(self, model_path="models/pose_landmarker_lite.task"):
        base = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base,
            output_segmentation_masks=False,
            num_poses=1,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def extract(self, image_bgr):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_bgr)
        result = self.detector.detect(mp_image)
        if not result.pose_landmarks:
            return None
        lms = result.pose_landmarks[0]
        feats = []
        for lm in lms[:33]:
            feats.extend([lm.x, lm.y])
        # pad to 150 (hands absent)
        if len(feats) < 150:
            feats.extend([0.0] * (150 - len(feats)))
        return np.array(feats, dtype=np.float32)


def main():
    args = parse_args()
    label_map = load_label_map(args.label_map)
    extractor = PoseExtractor()
    X_list = []
    y_list = []

    for split, img_path, lbl_path in image_paths(args.root):
        label = load_label(lbl_path, label_map)
        if label not in ["Fall", "No_Fall"]:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        pose_feat = extractor.extract(img)
        if pose_feat is None:
            continue
        seq = np.tile(pose_feat, (args.frames, 1))
        X_list.append(seq)
        y_list.append(0 if label == "Fall" else 1)

    if not X_list:
        raise SystemExit("No samples parsed.")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    print("Saved", X.shape, y.shape, "to", out_dir)


if __name__ == "__main__":
    main()
