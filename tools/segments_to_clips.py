import argparse
import csv
import re
import subprocess
from collections import defaultdict
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser(
        description="Cut video segments into class subfolders using ffmpeg."
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
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable name")
    parser.add_argument(
        "--copy-codec",
        action="store_true",
        help="Use stream copy instead of re-encode (faster but less accurate cuts).",
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
    if not has_ffmpeg(args.ffmpeg_bin):
        raise RuntimeError(f"ffmpeg not found: {args.ffmpeg_bin}")

    output_dir.mkdir(parents=True, exist_ok=True)
    class_counters = defaultdict(int)
    produced = 0
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"class_id", "class_name", "start_time_s", "end_time_s"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Segments CSV missing columns: {', '.join(sorted(missing))}")

        for row in reader:
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
            out_file = class_dir / f"seg_{class_counters[class_name]:06d}.mp4"

            if args.copy_codec:
                cmd = [
                    args.ffmpeg_bin,
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
                    args.ffmpeg_bin,
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

            if args.dry_run:
                print("[DRY-RUN]", " ".join(cmd))
                produced += 1
                continue

            subprocess.run(cmd, check=True)
            produced += 1

    print(f"[OK] Produced clips: {produced}")
    print(f"[OK] Skipped rows  : {skipped}")
    print(f"[OK] Output folder : {output_dir}")
    print("[INFO] Per-class clips:")
    for class_name, count in sorted(class_counters.items(), key=lambda x: x[0]):
        print(f" - {class_name}: {count}")


if __name__ == "__main__":
    main()
