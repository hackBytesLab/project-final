import argparse
import csv
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2


def sanitize_name(name):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return sanitized.strip("._") or "unknown"


def has_ffmpeg(ffmpeg_bin):
    try:
        subprocess.run(
            [ffmpeg_bin, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def parse_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def normalize_field_name(name):
    return (name or "").strip().lstrip("\ufeff")


def normalize_row_keys(row):
    return {normalize_field_name(k): v for k, v in row.items()}


def cut_with_ffmpeg(
    ffmpeg_bin,
    video_path,
    out_file,
    start,
    end,
    copy_codec=False,
    dry_run=False,
):
    if copy_codec:
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(video_path),
            "-c",
            "copy",
            str(out_file),
        ]
    else:
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-an",
            str(out_file),
        ]
    if dry_run:
        print("[DRY-RUN][ffmpeg]", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def cut_with_opencv(video_path, out_file, start, end, dry_run=False):
    if dry_run:
        print(
            "[DRY-RUN][opencv]",
            f"video={video_path} start={start:.3f}s end={end:.3f}s out={out_file}",
        )
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid frame size for video: {video_path}")

    start_frame = max(int(round(start * fps)), 0)
    end_frame = max(int(round(end * fps)), start_frame + 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output writer: {out_file}")

    frame_idx = start_frame
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Cut video segments into class subfolders (ffmpeg/OpenCV backend)."
    )
    parser.add_argument("--video", required=True, help="Source long video path")
    parser.add_argument("--segments-csv", required=True, help="Segments CSV path")
    parser.add_argument(
        "--output-dir",
        default="data_videos",
        help="Output base directory (class subfolders created automatically).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Skip segments shorter than this duration (seconds).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "ffmpeg", "opencv"],
        default="auto",
        help="Cut backend. auto=ffmpeg when available, otherwise opencv.",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable name")
    parser.add_argument(
        "--copy-codec",
        action="store_true",
        help="Use stream copy instead of re-encode (faster but less accurate cuts).",
    )
    parser.add_argument(
        "--filename-prefix",
        default="",
        help="Prefix for output clip filenames (used to avoid collisions across multiple source videos).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    args = parser.parse_args()

    video_path = Path(args.video)
    csv_path = Path(args.segments_csv)
    output_dir = Path(args.output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Segments CSV not found: {csv_path}")

    ffmpeg_available = has_ffmpeg(args.ffmpeg_bin)
    if args.backend == "ffmpeg" and not ffmpeg_available:
        raise RuntimeError(f"ffmpeg not found: {args.ffmpeg_bin}")
    if args.backend == "auto":
        resolved_backend = "ffmpeg" if ffmpeg_available else "opencv"
    else:
        resolved_backend = args.backend
    if resolved_backend == "opencv" and args.copy_codec:
        print("[WARN] --copy-codec is ignored when backend=opencv")

    prefix = sanitize_name(args.filename_prefix) if args.filename_prefix else ""
    if prefix:
        prefix += "__"
    print(f"[INFO] backend={resolved_backend}")

    output_dir.mkdir(parents=True, exist_ok=True)
    class_counters = defaultdict(int)
    produced = 0
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"class_id", "class_name", "start_time_s", "end_time_s"}
        fieldnames = {normalize_field_name(x) for x in (reader.fieldnames or [])}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"Segments CSV missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            row = normalize_row_keys(row)
            start = parse_float(row.get("start_time_s"))
            end = parse_float(row.get("end_time_s"))
            if start is None or end is None:
                skipped += 1
                continue

            duration = end - start
            if duration < args.min_duration:
                skipped += 1
                continue

            class_name = (row.get("class_name") or "").strip()
            class_id = (row.get("class_id") or "").strip()
            if not class_name:
                class_name = f"class_{class_id or 'unknown'}"
            class_name = sanitize_name(class_name)

            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            class_counters[class_name] += 1
            out_file = class_dir / f"{prefix}seg_{class_counters[class_name]:06d}.mp4"

            if resolved_backend == "ffmpeg":
                cut_with_ffmpeg(
                    ffmpeg_bin=args.ffmpeg_bin,
                    video_path=video_path,
                    out_file=out_file,
                    start=start,
                    end=end,
                    copy_codec=args.copy_codec,
                    dry_run=args.dry_run,
                )
            else:
                cut_with_opencv(
                    video_path=video_path,
                    out_file=out_file,
                    start=start,
                    end=end,
                    dry_run=args.dry_run,
                )
            produced += 1

    print(f"[OK] Produced clips: {produced}")
    print(f"[OK] Skipped rows  : {skipped}")
    print(f"[OK] Output folder : {output_dir}")
    print("[INFO] Per-class clips:")
    for class_name, count in sorted(class_counters.items(), key=lambda x: x[0]):
        print(f" - {class_name}: {count}")


if __name__ == "__main__":
    main()
