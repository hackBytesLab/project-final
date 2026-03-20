import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from feature_layout import compute_geometric_features


FEATURE_NAMES = [
    "trunk_angle",
    "aspect_ratio",
    "hip_height_ratio",
    "angular_velocity",
]


def load_arrays(data_path, labels_path):
    data = np.load(data_path)
    labels = np.load(labels_path)
    if data.ndim != 3:
        raise ValueError("Expected data array of shape (samples, timesteps, features)")
    if labels.shape[0] != data.shape[0]:
        raise ValueError("Label count does not match number of samples")
    return data, labels.astype(np.int32)


def parse_label_names(arg, class_map_path, num_classes):
    names = []
    if arg:
        names = [s.strip() for s in arg.split(",") if s.strip()]
    elif class_map_path:
        with Path(class_map_path).open("r", encoding="utf-8") as f:
            raw = json.load(f)
        # Accept {"fall": 0, "no_fall":1} or { "0": "fall", "1": "no_fall" }
        for k, v in raw.items():
            try:
                idx = int(v)
                name = str(k)
            except Exception:
                idx = int(k)
                name = str(v)
            if idx >= len(names):
                names.extend([""] * (idx - len(names) + 1))
            names[idx] = name
    if not names:
        names = [f"class_{i}" for i in range(num_classes)]
    if len(names) < num_classes:
        names.extend([f"class_{i}" for i in range(len(names), num_classes)])
    return names[:num_classes]


def parse_negative_labels(arg, label_names):
    tokens = [s.strip() for s in (arg or "").split(",") if s.strip()]
    negatives = set()
    for tok in tokens:
        if tok.isdigit():
            negatives.add(int(tok))
            continue
        low = tok.lower()
        for idx, name in enumerate(label_names):
            if name.lower() == low:
                negatives.add(idx)
                break
    if not negatives:
        negatives.add(1)  # Default: class id 1 is NO_FALL in current pipeline
    return negatives


def extract_sequence_features(sequence):
    seq = np.asarray(sequence, dtype=np.float32)
    geo = np.stack([compute_geometric_features(frame) for frame in seq], axis=0)
    trunk_angles = geo[:, 0]
    aspect_ratios = geo[:, 1]
    hip_height_ratios = geo[:, 2]

    ang_vel = np.zeros_like(trunk_angles)
    if len(trunk_angles) > 1:
        ang_vel[1:] = np.abs(np.diff(trunk_angles))

    return {
        "trunk_angle": float(trunk_angles[-1]),
        "aspect_ratio": float(aspect_ratios[-1]),
        "hip_height_ratio": float(hip_height_ratios[-1]),
        "angular_velocity": float(ang_vel.max()),
    }


def derive_threshold(values, positive_mask, higher_is_fall=True):
    vals = np.asarray(values, dtype=np.float32)
    y_true = np.asarray(positive_mask, dtype=np.int32)
    if not higher_is_fall:
        vals = -vals
    fpr, tpr, thr = roc_curve(y_true, vals)
    youden = tpr - fpr
    idx = int(np.argmax(youden))
    best_thr = float(thr[idx])
    if not higher_is_fall:
        best_thr = -best_thr
    return best_thr, float(tpr[idx]), float(fpr[idx])


def summarize_feature(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def plot_histograms(per_class, out_path):
    plt.figure(figsize=(14, 10))
    for i, feat in enumerate(FEATURE_NAMES):
        plt.subplot(3, 2, i + 1)
        for cls, values in per_class.items():
            plt.hist(values[feat], bins=40, alpha=0.5, label=str(cls))
        plt.title(feat)
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Calibrate geometric thresholds from labeled sequences")
    parser.add_argument("--data", required=True, help="Path to sequences.npy (N, T, F)")
    parser.add_argument("--labels", required=True, help="Path to labels.npy (N,)")
    parser.add_argument(
        "--label-names",
        default="",
        help="Comma-separated label names in class-id order (e.g., FALL,NO_FALL,FALLING,PRE_FALL)",
    )
    parser.add_argument("--class-map", default="", help="Optional class_map.json (name->id or id->name)")
    parser.add_argument(
        "--negative-labels",
        default="1",
        help="IDs or names treated as NO_FALL/normal (comma-separated). Default: 1",
    )
    parser.add_argument("--out", default="thresholds.json", help="Where to save suggested thresholds")
    parser.add_argument("--plot", action="store_true", help="Save histograms alongside thresholds.json")
    parser.add_argument("--plot-path", default="", help="Custom path for histogram PNG")
    args = parser.parse_args()

    data, labels = load_arrays(args.data, args.labels)
    num_classes = int(labels.max()) + 1
    label_names = parse_label_names(args.label_names, args.class_map, num_classes)
    negative_ids = parse_negative_labels(args.negative_labels, label_names)
    positive_mask = [0 if lbl in negative_ids else 1 for lbl in labels.tolist()]

    per_class = {name: {feat: [] for feat in FEATURE_NAMES} for name in label_names}
    pooled = {feat: [] for feat in FEATURE_NAMES}

    for seq, lbl in zip(data, labels):
        feats = extract_sequence_features(seq)
        class_key = label_names[lbl] if lbl < len(label_names) else str(lbl)
        for k, v in feats.items():
            pooled[k].append(v)
            per_class[class_key][k].append(v)

    results = {"thresholds": {}, "per_class": {}, "label_names": label_names}
    higher_fall = {
        "trunk_angle": True,
        "aspect_ratio": False,
        "hip_height_ratio": True,
        "angular_velocity": True,
    }

    for feat in FEATURE_NAMES:
        thr, tpr, fpr = derive_threshold(
            pooled[feat],
            positive_mask,
            higher_is_fall=higher_fall.get(feat, True),
        )
        results["thresholds"][feat] = {"value": thr, "tpr": tpr, "fpr": fpr}

    for cls, values in per_class.items():
        results["per_class"][cls] = {feat: summarize_feature(vals) for feat, vals in values.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if args.plot:
        plot_path = Path(args.plot_path) if args.plot_path else out_path.with_suffix(".png")
        plot_histograms(per_class, plot_path)
        print(f"[PLOT] saved -> {plot_path}")

    print(f"[DONE] thresholds saved to {out_path}")
    for feat, info in results["thresholds"].items():
        print(f"  {feat}: {info['value']:.3f} (TPR={info['tpr']:.2f}, FPR={info['fpr']:.2f})")


if __name__ == "__main__":
    main()
