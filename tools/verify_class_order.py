import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def read_manifest(manifest_path):
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"class_name", "video_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest is missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            class_name = (row.get("class_name") or "").strip()
            video_path = (row.get("video_path") or "").strip()
            if not class_name or not video_path:
                continue
            rows.append({"class_name": class_name, "video_path": video_path})
    if not rows:
        raise ValueError("No valid rows found in manifest.")
    return rows


def dominant_class_from_segments(segments_csv):
    per_class_duration = defaultdict(float)
    total_duration = 0.0

    with open(segments_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                class_id = int(row["class_id"])
                start = float(row["start_time_s"])
                end = float(row["end_time_s"])
            except (KeyError, ValueError, TypeError):
                continue

            duration = max(0.0, end - start)
            per_class_duration[class_id] += duration
            total_duration += duration

    if not per_class_duration:
        return None, 0.0, 0.0

    dominant_id = max(per_class_duration, key=per_class_duration.get)
    dominant_duration = per_class_duration[dominant_id]
    dominance_ratio = dominant_duration / total_duration if total_duration > 0 else 0.0
    return dominant_id, dominant_duration, dominance_ratio


def build_suggested_labels(id_to_class):
    if not id_to_class:
        return ""
    sorted_ids = sorted(id_to_class.keys())
    labels = [id_to_class[i] for i in sorted_ids]
    return ",".join(labels)


def run_infer_video(video_path, model_path, out_csv, timesteps, step, batch_size):
    cmd = [
        sys.executable,
        str(ROOT / "infer_video.py"),
        "--video",
        str(video_path),
        "--model",
        str(model_path),
        "--out-csv",
        str(out_csv),
        "--timesteps",
        str(timesteps),
        "--step",
        str(step),
        "--batch-size",
        str(batch_size),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Verify model class-id order using reference videos with known classes."
    )
    parser.add_argument(
        "--manifest",
        default="refs/refs_manifest.csv",
        help="CSV file with columns: class_name,video_path",
    )
    parser.add_argument(
        "--model",
        default="models/lstm_fall_model.h5",
        help="Path to trained model file",
    )
    parser.add_argument("--out-dir", default="work_csv", help="Output directory")
    parser.add_argument("--timesteps", type=int, default=30)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--fail-on-conflict",
        action="store_true",
        help="Exit with non-zero code if multiple classes map to the same dominant class_id.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    manifest_rows = read_manifest(manifest_path)
    report_rows = []
    id_to_classes = defaultdict(list)

    for item in manifest_rows:
        class_name = item["class_name"]
        video_path = Path(item["video_path"])
        if not video_path.exists():
            print(f"[WARN] Missing video file for {class_name}: {video_path}")
            continue

        out_csv = out_dir / f"{class_name}_ref_segments.csv"
        print(f"[INFO] Running inference for reference class '{class_name}'...")
        run_infer_video(
            video_path=video_path,
            model_path=model_path,
            out_csv=out_csv,
            timesteps=args.timesteps,
            step=args.step,
            batch_size=args.batch_size,
        )

        dominant_id, dominant_duration, ratio = dominant_class_from_segments(out_csv)
        report_rows.append(
            {
                "class_name": class_name,
                "video_path": str(video_path),
                "dominant_class_id": dominant_id if dominant_id is not None else "",
                "dominant_duration_s": f"{dominant_duration:.3f}",
                "dominance_ratio": f"{ratio:.4f}",
                "segments_csv": str(out_csv),
            }
        )
        if dominant_id is not None:
            id_to_classes[dominant_id].append(class_name)

    report_csv = out_dir / "class_order_report.csv"
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_name",
                "video_path",
                "dominant_class_id",
                "dominant_duration_s",
                "dominance_ratio",
                "segments_csv",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    conflict = any(len(v) > 1 for v in id_to_classes.values())
    id_to_class = {k: v[0] for k, v in id_to_classes.items() if len(v) == 1}

    class_order_json = out_dir / "class_order.json"
    with open(class_order_json, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(id_to_class.items())}, f, indent=2)

    labels_txt = out_dir / "suggested_labels.txt"
    labels_string = build_suggested_labels(id_to_class)
    with open(labels_txt, "w", encoding="utf-8") as f:
        f.write(labels_string + "\n")

    print(f"[OK] Wrote report: {report_csv}")
    print(f"[OK] Wrote class map: {class_order_json}")
    print(f"[OK] Suggested --labels: {labels_string}")

    if conflict:
        print("[WARN] Conflict detected: multiple class names mapped to same class_id.")
        for cid, classes in sorted(id_to_classes.items()):
            if len(classes) > 1:
                print(f" - class_id {cid}: {classes}")
        if args.fail_on_conflict:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
