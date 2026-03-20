import argparse
import csv
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from lstm_model import build_lstm_model
from feature_layout import enhance_sequence_features, POSE_FEATURES_PER_PERSON, ENHANCED_EXTRA_FEATURES


def generate_sample(
    data_dir, samples=200, timesteps=30, num_features=150, num_classes=4, random_state=42
):
    rng = np.random.default_rng(random_state)
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = rng.random((samples, timesteps, num_features), dtype=np.float32)
    y = rng.integers(0, num_classes, size=(samples,), dtype=np.int32)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)

    class_map = {f"class_{i}": i for i in range(num_classes)}
    with (out_dir / "class_map.json").open("w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    with (out_dir / "sample_meta.csv").open("w", newline="", encoding="utf-8") as f:
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
        for idx in range(samples):
            writer.writerow(
                {
                    "sample_idx": idx,
                    "class_id": int(y[idx]),
                    "class_name": f"class_{int(y[idx])}",
                    "clip_path": f"class_{int(y[idx])}/sample_{idx:06d}.mp4",
                    "source_video": "generated",
                    "window_start": 0,
                    "window_end": timesteps - 1,
                    "group_id": f"generated_clip_{idx:06d}",
                }
            )
    print(f"Generated sample data -> {out_dir / 'X.npy'}, {out_dir / 'y.npy'}")


def load_data(data_dir):
    data_dir = Path(data_dir)
    x_path = data_dir / "X.npy"
    y_path = data_dir / "y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("Expected X.npy and y.npy in data directory")
    return np.load(x_path), np.load(y_path)


def parse_labels(raw_labels):
    return [x.strip() for x in (raw_labels or "").split(",") if x.strip()]


def load_labels_from_class_map(data_dir):
    class_map_path = Path(data_dir) / "class_map.json"
    if not class_map_path.exists():
        return None
    with class_map_path.open("r", encoding="utf-8") as f:
        class_map = json.load(f)
    id_to_name = {}
    for name, idx in class_map.items():
        try:
            id_to_name[int(idx)] = str(name)
        except (TypeError, ValueError):
            continue
    if not id_to_name:
        return None
    max_id = max(id_to_name.keys())
    return [id_to_name.get(i, f"class_{i}") for i in range(max_id + 1)]


def resolve_label_names(data_dir, labels_arg, num_classes):
    labels = parse_labels(labels_arg)
    if labels:
        if len(labels) != num_classes:
            raise ValueError(
                f"--labels count ({len(labels)}) does not match num_classes ({num_classes})"
            )
        return labels

    map_labels = load_labels_from_class_map(data_dir)
    if map_labels and len(map_labels) == num_classes:
        return map_labels
    return [f"class_{i}" for i in range(num_classes)]


def resolve_reports_dir(args):
    reports_dir = Path(args.reports_dir or args.eval_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "cv").mkdir(parents=True, exist_ok=True)
    (reports_dir / "holdout").mkdir(parents=True, exist_ok=True)
    (reports_dir / "summary").mkdir(parents=True, exist_ok=True)
    return reports_dir


def load_meta_rows(meta_csv_path, expected_len):
    meta_path = Path(meta_csv_path)
    if not meta_path.exists():
        return None
    with meta_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None

    if "sample_idx" in (reader.fieldnames or []):
        rows.sort(key=lambda r: int(r.get("sample_idx", 0)))

    if len(rows) != expected_len:
        raise ValueError(
            f"Meta rows ({len(rows)}) do not match sample count in X/y ({expected_len})"
        )

    normalized = []
    for i, row in enumerate(rows):
        clip_path = row.get("clip_path", "")
        group_id = row.get("group_id", "") or clip_path or f"sample_{i}"
        source_video = row.get("source_video", "") or "unknown"
        normalized.append(
            {
                "sample_idx": i,
                "group_id": group_id,
                "clip_path": clip_path,
                "source_video": source_video,
            }
        )
    return normalized


def save_history_artifacts(history, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = history.history
    epochs = list(range(1, len(hist.get("loss", [])) + 1))

    csv_path = out_dir / "train_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "val_loss", "accuracy", "val_accuracy"])
        for e in epochs:
            i = e - 1
            writer.writerow(
                [
                    e,
                    hist.get("loss", [None] * len(epochs))[i],
                    hist.get("val_loss", [None] * len(epochs))[i],
                    hist.get("accuracy", [None] * len(epochs))[i],
                    hist.get("val_accuracy", [None] * len(epochs))[i],
                ]
            )

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist.get("loss", []), label="train_loss")
    plt.plot(epochs, hist.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist.get("accuracy", []), label="train_acc")
    plt.plot(epochs, hist.get("val_accuracy", []), label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "learning_curves.png", dpi=150)
    plt.close()


def save_confusion_matrix_csv(path, label_names, matrix):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + label_names)
        for i, row in enumerate(matrix):
            writer.writerow([label_names[i]] + list(row))


def plot_confusion_matrix(path, matrix, label_names, title, normalize=False):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(label_names))
    plt.xticks(ticks, label_names, rotation=45, ha="right")
    plt.yticks(ticks, label_names)

    thresh = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text = f"{val:.2f}" if normalize else str(int(val))
            plt.text(j, i, text, ha="center", va="center", color="white" if val > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_confusion_artifacts(out_dir, label_names, y_true, y_pred):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_idx = list(range(len(label_names)))
    cm_raw = confusion_matrix(y_true, y_pred, labels=labels_idx)
    row_sums = cm_raw.sum(axis=1, keepdims=True).astype(np.float64)
    cm_norm = np.zeros_like(cm_raw, dtype=np.float64)
    np.divide(cm_raw.astype(np.float64), row_sums, out=cm_norm, where=row_sums != 0)

    save_confusion_matrix_csv(out_dir / "confusion_matrix_raw.csv", label_names, cm_raw)
    save_confusion_matrix_csv(out_dir / "confusion_matrix_norm.csv", label_names, np.round(cm_norm, 6))
    plot_confusion_matrix(
        out_dir / "confusion_matrix_raw.png", cm_raw, label_names, "Confusion Matrix (Raw)", normalize=False
    )
    plot_confusion_matrix(
        out_dir / "confusion_matrix_norm.png", cm_norm, label_names, "Confusion Matrix (Normalized)", normalize=True
    )
    return cm_raw, cm_norm


def save_roc_artifacts(out_dir, label_names, y_true, y_prob):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(label_names)
    class_ids = np.arange(n_classes)

    y_true_bin = label_binarize(y_true, classes=class_ids)
    if y_true_bin.shape[1] == 1 and n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fpr = {}
    tpr = {}
    thresholds = {}
    roc_auc = {}
    valid_class_ids = []

    for i in range(n_classes):
        positives = int(np.sum(y_true_bin[:, i] == 1))
        negatives = int(np.sum(y_true_bin[:, i] == 0))
        if positives == 0 or negatives == 0:
            fpr[i] = np.array([])
            tpr[i] = np.array([])
            thresholds[i] = np.array([])
            roc_auc[i] = float("nan")
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            fpr[i], tpr[i], thresholds[i] = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        valid_class_ids.append(i)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        if len(np.unique(y_true_bin.ravel())) >= 2:
            fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(
                y_true_bin.ravel(), y_prob.ravel()
            )
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        else:
            fpr["micro"] = np.array([])
            tpr["micro"] = np.array([])
            thresholds["micro"] = np.array([])
            roc_auc["micro"] = float("nan")

    if valid_class_ids:
        all_fpr = np.unique(np.concatenate([fpr[i] for i in valid_class_ids]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in valid_class_ids:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(valid_class_ids)
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        thresholds["macro"] = np.array([])
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    else:
        fpr["macro"] = np.array([])
        tpr["macro"] = np.array([])
        thresholds["macro"] = np.array([])
        roc_auc["macro"] = float("nan")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            if len(np.unique(y_true)) < 2:
                weighted_auc = float("nan")
            elif n_classes > 2:
                weighted_auc = float(
                    roc_auc_score(
                        y_true,
                        y_prob,
                        labels=class_ids,
                        multi_class="ovr",
                        average="weighted",
                    )
                )
            else:
                weighted_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
    except ValueError:
        # Some folds may miss one or more classes in y_true.
        weighted_auc = float("nan")
    roc_auc["weighted"] = weighted_auc

    curves_csv = out_dir / "roc_curves.csv"
    with curves_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["curve_type", "class_id", "class_name", "fpr", "tpr", "threshold"])
        for i, name in enumerate(label_names):
            if len(fpr[i]) == 0:
                continue
            for j, (x, yv) in enumerate(zip(fpr[i], tpr[i])):
                thr = thresholds[i][j] if j < len(thresholds[i]) else ""
                writer.writerow(["class", i, name, float(x), float(yv), thr])
        for curve_key in ("micro", "macro"):
            thr_values = thresholds[curve_key]
            if len(fpr[curve_key]) == 0:
                continue
            for j, (x, yv) in enumerate(zip(fpr[curve_key], tpr[curve_key])):
                thr = thr_values[j] if j < len(thr_values) else ""
                writer.writerow([curve_key, "", "", float(x), float(yv), thr])

    auc_summary = {
        "per_class": {label_names[i]: float(roc_auc[i]) for i in range(n_classes)},
        "micro_auc": float(roc_auc["micro"]),
        "macro_auc": float(roc_auc["macro"]),
        "weighted_auc": float(roc_auc["weighted"]),
    }
    with (out_dir / "auc_summary.json").open("w", encoding="utf-8") as f:
        json.dump(auc_summary, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(8, 6))
    for i, name in enumerate(label_names):
        if len(fpr[i]) == 0 or np.isnan(roc_auc[i]):
            continue
        plt.plot(fpr[i], tpr[i], label=f"{name} (AUC={roc_auc[i]:.3f})")
    if len(fpr["micro"]) > 0 and not np.isnan(roc_auc["micro"]):
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            linestyle="--",
            label=f"micro (AUC={roc_auc['micro']:.3f})",
        )
    if len(fpr["macro"]) > 0 and not np.isnan(roc_auc["macro"]):
        plt.plot(
            fpr["macro"],
            tpr["macro"],
            linestyle="--",
            label=f"macro (AUC={roc_auc['macro']:.3f})",
        )
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (OvR)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close()
    return auc_summary


def evaluate_and_save(out_dir, label_names, y_true, y_prob):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = np.argmax(y_prob, axis=1)
    labels_idx = list(range(len(label_names)))

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_idx,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    with (out_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
    with (out_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(
            classification_report(
                y_true,
                y_pred,
                labels=labels_idx,
                target_names=label_names,
                zero_division=0,
            )
        )

    save_confusion_artifacts(out_dir, label_names, y_true, y_pred)
    auc_summary = save_roc_artifacts(out_dir, label_names, y_true, y_prob)

    summary = {
        "accuracy": report_dict.get("accuracy", 0.0),
        "macro_precision": report_dict.get("macro avg", {}).get("precision", 0.0),
        "macro_recall": report_dict.get("macro avg", {}).get("recall", 0.0),
        "macro_f1": report_dict.get("macro avg", {}).get("f1-score", 0.0),
        "weighted_precision": report_dict.get("weighted avg", {}).get("precision", 0.0),
        "weighted_recall": report_dict.get("weighted avg", {}).get("recall", 0.0),
        "weighted_f1": report_dict.get("weighted avg", {}).get("f1-score", 0.0),
        "macro_auc": auc_summary["macro_auc"],
        "micro_auc": auc_summary["micro_auc"],
        "weighted_auc": auc_summary["weighted_auc"],
    }
    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def compute_balanced_class_weights(y, classes):
    counts = np.bincount(y, minlength=classes).astype(np.float64)
    present = counts > 0
    present_classes = int(np.sum(present))
    if present_classes == 0:
        return {i: 1.0 for i in range(classes)}

    total = float(np.sum(counts))
    weights = {}
    for i in range(classes):
        if counts[i] > 0:
            weights[i] = total / (present_classes * counts[i])
    return weights


def compute_balanced_focal_alpha(y, classes, cap=4.0):
    counts = np.bincount(y, minlength=classes).astype(np.float64)
    present = counts > 0
    present_classes = int(np.sum(present))
    if present_classes == 0:
        return [1.0] * classes

    total = float(np.sum(counts))
    alpha = np.zeros(classes, dtype=np.float64)
    for i in range(classes):
        if counts[i] > 0:
            alpha[i] = total / (present_classes * counts[i])
        else:
            alpha[i] = 0.0

    mean_present = float(np.mean(alpha[present])) if np.any(present) else 1.0
    if mean_present <= 0:
        mean_present = 1.0
    alpha = alpha / mean_present
    if cap and cap > 0:
        alpha = np.clip(alpha, 0.0, float(cap))
    return [float(x) for x in alpha]


def build_sample_weights(y, class_weights):
    return np.array([class_weights.get(int(cls), 1.0) for cls in y], dtype=np.float32)


def parse_class_weight_override(raw, classes):
    """
    Parse manual class weights string "0:1,1:0.5,2:2,3:8".
    Returns dict[int, float]; ignores malformed entries; clamps to class count.
    """
    weights = {}
    if not raw:
        return weights
    for part in raw.split(","):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        try:
            idx = int(k.strip())
            val = float(v.strip())
        except ValueError:
            continue
        if idx < 0 or idx >= classes:
            continue
        weights[idx] = val
    return weights


def oversample_training_data(X, y, classes, random_state):
    rng = np.random.default_rng(random_state)
    counts = np.bincount(y, minlength=classes).astype(np.int64)
    target = int(np.max(counts)) if counts.size else 0
    if target <= 0:
        return X, y, counts, counts

    sampled = []
    for cls in range(classes):
        cls_idx = np.where(y == cls)[0]
        if cls_idx.size == 0:
            continue
        if cls_idx.size < target:
            chosen = rng.choice(cls_idx, size=target, replace=True)
        else:
            chosen = cls_idx
        sampled.append(chosen)

    if not sampled:
        return X, y, counts, counts

    sampled_idx = np.concatenate(sampled)
    rng.shuffle(sampled_idx)
    y_bal = y[sampled_idx]
    after_counts = np.bincount(y_bal, minlength=classes).astype(np.int64)
    return X[sampled_idx], y_bal, counts, after_counts


def _shift_sequence(seq, shift):
    if shift == 0:
        return seq
    out = np.empty_like(seq)
    if shift > 0:
        out[:shift] = seq[0:1]
        out[shift:] = seq[:-shift]
    else:
        s = -shift
        out[-s:] = seq[-1:]
        out[:-s] = seq[s:]
    return out


def apply_sequence_augmentation(
    seq,
    rng,
    noise_std=0.01,
    scale_range=0.05,
    max_time_shift=2,
    feature_dropout=0.01,
    time_mask_ratio=0.10,
):
    aug = np.array(seq, dtype=np.float32, copy=True)
    timesteps = aug.shape[0]

    if max_time_shift > 0:
        shift = int(rng.integers(-max_time_shift, max_time_shift + 1))
        aug = _shift_sequence(aug, shift)

    if scale_range > 0:
        scale = 1.0 + float(rng.uniform(-scale_range, scale_range))
        aug *= scale

    if noise_std > 0:
        aug += rng.normal(0.0, noise_std, size=aug.shape).astype(np.float32)

    if feature_dropout > 0:
        drop_mask = rng.random(aug.shape, dtype=np.float32) < float(feature_dropout)
        aug[drop_mask] = 0.0

    if time_mask_ratio > 0 and timesteps > 1:
        max_mask = max(1, int(round(timesteps * float(time_mask_ratio))))
        mask_len = int(rng.integers(0, max_mask + 1))
        if mask_len > 0 and mask_len < timesteps:
            start = int(rng.integers(0, timesteps - mask_len + 1))
            aug[start:start + mask_len] = 0.0

    return aug


def augment_training_data(X, y, classes, args, random_state):
    if args.augment_mode == "none" or args.augment_factor <= 0:
        return X, y, {"augment_mode": "none", "augmented_samples": 0}

    rng = np.random.default_rng(random_state)
    counts = np.bincount(y, minlength=classes).astype(np.int64)
    max_count = int(np.max(counts)) if counts.size else 0
    if max_count <= 0:
        return X, y, {"augment_mode": args.augment_mode, "augmented_samples": 0}

    if args.augment_mode == "minority":
        minority_classes = [
            cls for cls in range(classes)
            if counts[cls] > 0 and counts[cls] < max_count * float(args.augment_minority_ratio)
        ]
        if not minority_classes:
            return X, y, {"augment_mode": "minority", "augmented_samples": 0}
        source_idx = np.where(np.isin(y, minority_classes))[0]
    else:
        source_idx = np.arange(len(y))

    if source_idx.size == 0:
        return X, y, {"augment_mode": args.augment_mode, "augmented_samples": 0}

    n_aug = int(round(source_idx.size * float(args.augment_factor)))
    if n_aug <= 0:
        return X, y, {"augment_mode": args.augment_mode, "augmented_samples": 0}

    chosen = rng.choice(source_idx, size=n_aug, replace=True)
    X_aug = np.empty((n_aug, X.shape[1], X.shape[2]), dtype=np.float32)
    y_aug = y[chosen].copy()

    for i, idx in enumerate(chosen):
        X_aug[i] = apply_sequence_augmentation(
            X[idx],
            rng=rng,
            noise_std=args.augment_noise_std,
            scale_range=args.augment_scale_range,
            max_time_shift=args.augment_time_shift,
            feature_dropout=args.augment_feature_dropout,
            time_mask_ratio=args.augment_time_mask_ratio,
        )

    X_out = np.concatenate([X, X_aug], axis=0)
    y_out = np.concatenate([y, y_aug], axis=0)
    shuffle_idx = np.arange(len(y_out))
    rng.shuffle(shuffle_idx)
    X_out = X_out[shuffle_idx]
    y_out = y_out[shuffle_idx]

    info = {
        "augment_mode": args.augment_mode,
        "augment_factor": float(args.augment_factor),
        "augment_minority_ratio": float(args.augment_minority_ratio),
        "augmented_samples": int(n_aug),
        "train_samples_before_aug": int(len(y)),
        "train_samples_after_aug": int(len(y_out)),
    }
    return X_out, y_out, info


def train_single_run(X_train, y_train, X_val, y_val, num_features, classes, args, run_seed=42):
    train_X = X_train
    train_y = y_train
    sample_weight = None
    balance_info = {"mode": args.balance_mode}
    focal_alpha_value = float(args.focal_alpha)

    if args.loss_function == "focal" and args.focal_alpha_mode == "balanced":
        focal_alpha_value = compute_balanced_focal_alpha(
            y_train,
            classes,
            cap=args.focal_alpha_cap,
        )
        balance_info["focal_alpha_mode"] = "balanced"
        balance_info["focal_alpha_vector"] = focal_alpha_value
    elif args.loss_function == "focal":
        balance_info["focal_alpha_mode"] = "fixed"
        balance_info["focal_alpha"] = float(args.focal_alpha)

    model = build_lstm_model(
        num_features,
        classes,
        loss_name=args.loss_function,
        focal_gamma=args.focal_gamma,
        focal_alpha=focal_alpha_value,
        dropout_rate=args.dropout_rate,
        use_batchnorm=not args.no_batchnorm,
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
    ]

    if args.balance_mode == "oversample":
        train_X, train_y, before_counts, after_counts = oversample_training_data(
            X_train, y_train, classes, run_seed
        )
        balance_info["before_counts"] = [int(x) for x in before_counts]
        balance_info["after_counts"] = [int(x) for x in after_counts]
        balance_info["train_samples_before"] = int(len(y_train))
        balance_info["train_samples_after"] = int(len(train_y))

    train_X, train_y, aug_info = augment_training_data(
        train_X,
        train_y,
        classes,
        args,
        random_state=run_seed + 123,
    )
    balance_info.update(aug_info)

    override_weights = parse_class_weight_override(args.class_weight_override, classes)
    if override_weights:
        sample_weight = build_sample_weights(train_y, override_weights)
        balance_info["class_weights"] = {str(k): float(v) for k, v in override_weights.items()}
        balance_info["class_weight_mode"] = "override"
    elif args.balance_mode == "class_weight":
        class_weights = compute_balanced_class_weights(train_y, classes)
        sample_weight = build_sample_weights(train_y, class_weights)
        balance_info["class_weights"] = {str(k): float(v) for k, v in class_weights.items()}
        balance_info["class_weight_mode"] = "balanced"

    history = model.fit(
        train_X,
        to_categorical(train_y, num_classes=classes),
        validation_data=(X_val, to_categorical(y_val, num_classes=classes)),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        sample_weight=sample_weight,
        verbose=2,
    )
    probs = model.predict(X_val, verbose=0)
    return model, history, probs, balance_info


def class_counts(y, label_names):
    counts = {}
    for i, name in enumerate(label_names):
        counts[name] = int(np.sum(y == i))
    return counts


def select_holdout_indices(y, groups, test_size, random_state):
    all_idx = np.arange(len(y))
    if groups is None:
        train_idx, test_idx = train_test_split(
            all_idx, test_size=test_size, random_state=random_state, stratify=y
        )
        return train_idx, test_idx

    n_splits = max(2, int(round(1.0 / test_size)))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    try:
        classes = np.unique(y)
        expected_classes = set(classes.tolist())
        candidates = []
        for train_idx, test_idx in splitter.split(np.zeros(len(y)), y, groups=groups):
            train_classes = set(np.unique(y[train_idx]).tolist())
            test_classes = set(np.unique(y[test_idx]).tolist())
            train_full_coverage = train_classes == expected_classes
            test_full_coverage = test_classes == expected_classes
            full_coverage = (
                train_full_coverage and test_full_coverage
            )
            ratio_delta = abs((len(test_idx) / len(y)) - test_size)
            test_counts = np.bincount(y[test_idx], minlength=len(classes))
            min_test_count = int(np.min(test_counts)) if len(test_counts) > 0 else 0
            candidates.append(
                {
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "full_coverage": full_coverage,
                    "train_full_coverage": train_full_coverage,
                    "test_full_coverage": test_full_coverage,
                    "min_test_count": min_test_count,
                    "test_cov": len(test_classes),
                    "train_cov": len(train_classes),
                    "ratio_delta": ratio_delta,
                    "test_missing": sorted(expected_classes - test_classes),
                    "train_missing": sorted(expected_classes - train_classes),
                }
            )

        if not candidates:
            raise ValueError("No holdout candidate splits were generated.")

        candidates.sort(
            key=lambda c: (
                not c["train_full_coverage"],
                not c["full_coverage"],
                -c["min_test_count"],
                -c["test_cov"],
                -c["train_cov"],
                c["ratio_delta"],
            )
        )
        chosen = candidates[0]
        if not chosen["full_coverage"]:
            print(
                "[WARN] No group-aware holdout split had full class coverage. "
                f"Selected best candidate with missing test classes={chosen['test_missing']} "
                f"missing train classes={chosen['train_missing']}."
            )
        return chosen["train_idx"], chosen["test_idx"]
    except ValueError as e:
        raise ValueError(
            f"Unable to create group-aware holdout split ({e}). "
            "Try reducing --num-folds or use --split-unit sample."
        ) from e


def build_cv_splits(y, groups, num_folds, random_state):
    min_class = int(np.min(np.bincount(y)))
    if min_class < num_folds:
        raise ValueError(
            f"Smallest class has {min_class} samples, but --num-folds={num_folds}. "
            f"Please reduce --num-folds to <= {min_class}."
        )

    if groups is None:
        splitter = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=random_state
        )
        return list(splitter.split(np.zeros(len(y)), y))

    splitter = StratifiedGroupKFold(
        n_splits=num_folds, shuffle=True, random_state=random_state
    )
    try:
        return list(splitter.split(np.zeros(len(y)), y, groups=groups))
    except ValueError as e:
        raise ValueError(
            f"Unable to create group-aware folds ({e}). "
            "Try reducing --num-folds or use --split-unit sample."
        ) from e


def aggregate_cv_metrics(fold_summaries):
    if not fold_summaries:
        return {}
    metric_keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "macro_auc",
        "micro_auc",
        "weighted_auc",
    ]
    aggregate = {}
    for key in metric_keys:
        vals = np.array([f[key] for f in fold_summaries], dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    return aggregate


def save_overview(summary_dir, overview):
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    with (summary_dir / "overview.json").open("w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# Training Overview")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- validation_mode: {overview.get('validation_mode')}")
    lines.append(f"- split_unit_requested: {overview.get('split_unit_requested')}")
    lines.append(f"- split_unit_effective: {overview.get('split_unit_effective')}")
    lines.append(f"- augment_mode: {overview.get('augment_mode')}")
    lines.append(f"- augment_factor: {overview.get('augment_factor')}")
    lines.append(f"- loss_function: {overview.get('loss_function')}")
    lines.append(f"- focal_alpha_mode: {overview.get('focal_alpha_mode')}")
    lines.append(f"- focal_alpha_cap: {overview.get('focal_alpha_cap')}")
    lines.append(f"- focal_gamma: {overview.get('focal_gamma')}")
    lines.append(f"- focal_alpha: {overview.get('focal_alpha')}")
    lines.append(f"- num_folds: {overview.get('num_folds')}")
    lines.append(f"- test_size: {overview.get('test_size')}")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- total_samples: {overview.get('total_samples')}")
    lines.append(f"- class_distribution: {overview.get('class_distribution')}")
    lines.append("")
    if overview.get("cv_aggregate"):
        lines.append("## CV Aggregate")
        for k, v in overview["cv_aggregate"].items():
            lines.append(f"- {k}: mean={v['mean']:.4f}, std={v['std']:.4f}")
        lines.append("")
    if overview.get("holdout_metrics"):
        lines.append("## Holdout Metrics")
        for k, v in overview["holdout_metrics"].items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("")

    with (summary_dir / "overview.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Train LSTM fall-detection model")
    parser.add_argument("--data-dir", default="data", help="Folder containing X.npy and y.npy")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--dropout-rate", type=float, default=0.3, help="Dropout rate for LSTM stack")
    parser.add_argument(
        "--no-batchnorm", action="store_true", help="Disable batch normalization layers in the LSTM stack"
    )
    parser.add_argument("--out", default="models/lstm_fall_model.h5", help="Output model path")
    parser.add_argument(
        "--eval-dir",
        default="work_csv",
        help="Legacy report directory. Used when --reports-dir is omitted.",
    )
    parser.add_argument("--reports-dir", default="", help="Base folder for evaluation reports")
    parser.add_argument(
        "--plots-dir",
        default="",
        help="Optional folder for high-level plots/overview (default: <reports-dir>/summary)",
    )
    parser.add_argument("--labels", default="", help="Optional comma-separated labels in class-id order")
    parser.add_argument(
        "--enhance-features",
        action="store_true",
        help="Add pose velocity (66 coords diff) -> 216 dims for single-person layout.",
    )
    parser.add_argument(
        "--validation-mode",
        choices=["split", "kfold", "holdout-kfold"],
        default="holdout-kfold",
        help="Validation strategy",
    )
    parser.add_argument("--num-folds", type=int, default=5, help="Number of folds for kfold modes")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout/split fraction")
    parser.add_argument("--split-unit", choices=["sample", "clip"], default="clip")
    parser.add_argument("--meta-csv", default="", help="Sample metadata CSV (default: <data-dir>/sample_meta.csv)")
    parser.add_argument(
        "--balance-mode",
        choices=["none", "class_weight", "oversample"],
        default="none",
        help="Imbalance handling for training folds/final model",
    )
    parser.add_argument(
        "--class-weight-override",
        default="",
        help="Optional manual class weights, e.g. '0:1,1:0.5,2:2,3:8'. If set, overrides computed class weights.",
    )
    parser.add_argument(
        "--augment-mode",
        choices=["none", "minority", "all"],
        default="none",
        help="Data augmentation mode for train split",
    )
    parser.add_argument(
        "--augment-factor",
        type=float,
        default=0.0,
        help="Augmented sample ratio relative to selected source samples",
    )
    parser.add_argument(
        "--augment-minority-ratio",
        type=float,
        default=0.8,
        help="Class is minority when count < max_count * ratio",
    )
    parser.add_argument("--augment-noise-std", type=float, default=0.01)
    parser.add_argument("--augment-scale-range", type=float, default=0.05)
    parser.add_argument("--augment-time-shift", type=int, default=2)
    parser.add_argument("--augment-feature-dropout", type=float, default=0.01)
    parser.add_argument("--augment-time-mask-ratio", type=float, default=0.10)
    parser.add_argument(
        "--loss-function",
        choices=["categorical_crossentropy", "focal"],
        default="categorical_crossentropy",
        help="Training loss function",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma (used when --loss-function focal)",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="Focal loss alpha (used when --loss-function focal)",
    )
    parser.add_argument(
        "--focal-alpha-mode",
        choices=["fixed", "balanced"],
        default="fixed",
        help="How to set focal alpha when --loss-function focal",
    )
    parser.add_argument(
        "--focal-alpha-cap",
        type=float,
        default=4.0,
        help="Upper cap for balanced focal alpha vector",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--generate-sample", action="store_true", help="Generate small random dataset and exit")
    args = parser.parse_args()

    if args.generate_sample:
        generate_sample(args.data_dir, random_state=args.random_state)
        return

    X, y = load_data(args.data_dir)
    if X.ndim != 3:
        raise ValueError("X must be 3D: (n_samples, timesteps, num_features)")
    if y.ndim > 1:
        y = y.flatten()
    if y.size == 0:
        raise ValueError("y is empty")
    if np.min(y) < 0:
        raise ValueError("y must contain non-negative class ids")

    if args.enhance_features:
        X = np.stack([enhance_sequence_features(seq) for seq in X], axis=0)
        print(f"[INFO] Enhanced features enabled -> new shape {X.shape}")

    classes = int(np.max(y)) + 1
    label_names = resolve_label_names(args.data_dir, args.labels, classes)
    _, _, num_features = X.shape

    reports_dir = resolve_reports_dir(args)
    summary_dir = reports_dir / "summary"
    plots_dir = Path(args.plots_dir) if args.plots_dir else summary_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    meta_path = Path(args.meta_csv) if args.meta_csv else Path(args.data_dir) / "sample_meta.csv"
    meta_rows = load_meta_rows(meta_path, len(y)) if meta_path.exists() else None
    split_unit_effective = args.split_unit
    if args.split_unit == "clip" and meta_rows is None:
        split_unit_effective = "sample"
        print(
            f"[WARN] split-unit=clip requested but meta not found at '{meta_path}'. "
            "Fallback to split-unit=sample."
        )
    groups_all = None
    if split_unit_effective == "clip" and meta_rows is not None:
        groups_all = np.array([r["group_id"] for r in meta_rows], dtype=object)

    overview = {
        "validation_mode": args.validation_mode,
        "split_unit_requested": args.split_unit,
        "split_unit_effective": split_unit_effective,
        "balance_mode": args.balance_mode,
        "augment_mode": args.augment_mode,
        "augment_factor": float(args.augment_factor),
        "loss_function": args.loss_function,
        "enhance_features": bool(args.enhance_features),
        "focal_alpha_mode": args.focal_alpha_mode,
        "focal_alpha_cap": float(args.focal_alpha_cap),
        "focal_gamma": float(args.focal_gamma),
        "focal_alpha": float(args.focal_alpha),
        "dropout_rate": float(args.dropout_rate),
        "use_batchnorm": bool(not args.no_batchnorm),
        "num_folds": int(args.num_folds),
        "test_size": float(args.test_size),
        "total_samples": int(len(y)),
        "class_distribution": class_counts(y, label_names),
    }

    cv_fold_summaries = []
    holdout_metrics = None
    split_info = {}

    if args.validation_mode in ("kfold", "holdout-kfold"):
        if args.validation_mode == "holdout-kfold":
            dev_idx, hold_idx = select_holdout_indices(
                y, groups_all, args.test_size, args.random_state
            )
            split_info["dev_samples"] = int(len(dev_idx))
            split_info["holdout_samples"] = int(len(hold_idx))
            if groups_all is not None:
                dev_groups = set(groups_all[dev_idx].tolist())
                hold_groups = set(groups_all[hold_idx].tolist())
                split_info["group_overlap_dev_holdout"] = int(len(dev_groups & hold_groups))
            else:
                dev_groups = None
                hold_groups = None
        else:
            dev_idx = np.arange(len(y))
            hold_idx = None
            dev_groups = set(groups_all.tolist()) if groups_all is not None else None

        y_dev = y[dev_idx]
        X_dev = X[dev_idx]
        groups_dev = groups_all[dev_idx] if groups_all is not None else None
        splits = build_cv_splits(y_dev, groups_dev, args.num_folds, args.random_state)

        for fold_i, (tr_sub, va_sub) in enumerate(splits, start=1):
            fold_dir = reports_dir / "cv" / f"fold_{fold_i}"
            X_tr, y_tr = X_dev[tr_sub], y_dev[tr_sub]
            X_va, y_va = X_dev[va_sub], y_dev[va_sub]
            model, history, probs, balance_info = train_single_run(
                X_tr, y_tr, X_va, y_va, num_features, classes, args, run_seed=args.random_state + fold_i
            )
            del model
            save_history_artifacts(history, fold_dir)
            fold_metrics = evaluate_and_save(fold_dir, label_names, y_va, probs)
            fold_metrics["fold"] = fold_i
            fold_metrics["train_samples"] = int(len(tr_sub))
            fold_metrics["val_samples"] = int(len(va_sub))
            if groups_dev is not None:
                tr_groups = set(groups_dev[tr_sub].tolist())
                va_groups = set(groups_dev[va_sub].tolist())
                fold_metrics["group_overlap_train_val"] = int(len(tr_groups & va_groups))
            fold_metrics["balance_mode"] = args.balance_mode
            if "class_weights" in balance_info:
                fold_metrics["class_weights"] = balance_info["class_weights"]
            if "train_samples_after" in balance_info:
                fold_metrics["train_samples_after"] = balance_info["train_samples_after"]
            cv_fold_summaries.append(fold_metrics)

        cv_aggregate = aggregate_cv_metrics(cv_fold_summaries)
        with (summary_dir / "cv_summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"folds": cv_fold_summaries, "aggregate": cv_aggregate},
                f,
                ensure_ascii=False,
                indent=2,
            )
        overview["cv_aggregate"] = cv_aggregate

        final_train_idx, final_val_idx = select_holdout_indices(
            y_dev, groups_dev, args.test_size, args.random_state + 1
        )
        final_model, final_history, _, final_balance = train_single_run(
            X_dev[final_train_idx],
            y_dev[final_train_idx],
            X_dev[final_val_idx],
            y_dev[final_val_idx],
            num_features,
            classes,
            args,
            run_seed=args.random_state + 1000,
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_model.save(out_path)
        save_history_artifacts(final_history, plots_dir)
        overview["final_balance"] = final_balance

        if args.validation_mode == "holdout-kfold":
            holdout_probs = final_model.predict(X[hold_idx], verbose=0)
            holdout_metrics = evaluate_and_save(
                reports_dir / "holdout", label_names, y[hold_idx], holdout_probs
            )
        else:
            split_probs = final_model.predict(X_dev[final_val_idx], verbose=0)
            holdout_metrics = evaluate_and_save(
                reports_dir / "holdout", label_names, y_dev[final_val_idx], split_probs
            )
            split_info["final_validation_samples"] = int(len(final_val_idx))

    else:
        train_idx, val_idx = select_holdout_indices(y, groups_all, args.test_size, args.random_state)
        split_info["train_samples"] = int(len(train_idx))
        split_info["val_samples"] = int(len(val_idx))
        if groups_all is not None:
            tr_groups = set(groups_all[train_idx].tolist())
            va_groups = set(groups_all[val_idx].tolist())
            split_info["group_overlap_train_val"] = int(len(tr_groups & va_groups))

        model, history, probs, final_balance = train_single_run(
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx],
            num_features,
            classes,
            args,
            run_seed=args.random_state + 1000,
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(out_path)
        save_history_artifacts(history, plots_dir)
        holdout_metrics = evaluate_and_save(reports_dir / "holdout", label_names, y[val_idx], probs)
        overview["final_balance"] = final_balance

    overview["split_info"] = split_info
    overview["holdout_metrics"] = holdout_metrics
    overview["model_output"] = str(Path(args.out))
    save_overview(summary_dir, overview)

    print("Training finished.")
    print("Model:", args.out)
    print("Reports:", reports_dir)
    print("Summary:", summary_dir)


if __name__ == "__main__":
    main()
