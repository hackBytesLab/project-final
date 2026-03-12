import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def sanitize_name(name):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return sanitized.strip("._") or "video"


def parse_videos(raw):
    parts = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
    if not parts:
        raise ValueError("--videos is required")
    return parts


def run_cmd(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build dataset from multiple long videos via infer -> filter -> cut -> video_to_dataset."
    )
    parser.add_argument(
        "--videos",
        required=True,
        help="Comma-separated video paths, e.g. Data/train.mp4,Data/train2.mp4",
    )
    parser.add_argument(
        "--model",
        default="models/lstm_fall_model.h5",
        help="Model used for auto-label inference",
    )
    parser.add_argument(
        "--labels",
        default="Fall,No_Fall,Pre-Fall,Falling",
        help="Comma-separated label order",
    )
    parser.add_argument("--timesteps", type=int, default=30)
    parser.add_argument("--infer-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-score", type=float, default=0.60)
    parser.add_argument("--min-duration", type=float, default=0.50)
    parser.add_argument(
        "--backend",
        choices=["auto", "ffmpeg", "opencv"],
        default="auto",
        help="Clip cutting backend",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--copy-codec", action="store_true")
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="Project root where data_videos/work_csv/data are created",
    )
    parser.add_argument("--work-csv-dir", default="work_csv")
    parser.add_argument("--clips-dir", default="data_videos")
    parser.add_argument("--dataset-dir", default="data")
    parser.add_argument(
        "--dataset-step",
        type=int,
        default=15,
        help="Sliding window step for video_to_dataset.py",
    )
    parser.add_argument(
        "--max-people",
        type=int,
        default=1,
        help="People slots used in dataset feature extraction.",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=0,
        help="Hand slots used in dataset feature extraction (0=2*max-people).",
    )
    parser.add_argument(
        "--normalize-geometry",
        action="store_true",
        help="Normalize pose/hand geometry in infer and dataset feature extraction.",
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Save labeled preview videos from infer_video.py",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing clips_dir, dataset_dir, and work-csv generated CSVs before running.",
    )
    args = parser.parse_args()

    py = sys.executable
    workspace = Path(args.workspace_root).resolve()
    work_csv_dir = (workspace / args.work_csv_dir).resolve()
    clips_dir = (workspace / args.clips_dir).resolve()
    dataset_dir = (workspace / args.dataset_dir).resolve()

    if args.clean_output:
        for target in (clips_dir, dataset_dir):
            if target.exists():
                print("[INFO] Removing:", target)
                shutil.rmtree(target)
        if work_csv_dir.exists():
            for p in work_csv_dir.glob("segments_*"):
                if p.is_file():
                    p.unlink(missing_ok=True)

    run_cmd([py, "tools/prepare_workspace.py", "--root", str(workspace)])

    videos = parse_videos(args.videos)
    for video in videos:
        video_path = Path(video)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        video_id = sanitize_name(video_path.stem)
        segments_named = work_csv_dir / f"segments_named_{video_id}.csv"
        segments_filtered = work_csv_dir / f"segments_filtered_{video_id}.csv"

        infer_cmd = [
            py,
            "infer_video.py",
            "--video",
            str(video_path),
            "--model",
            args.model,
            "--out-csv",
            str(segments_named),
            "--timesteps",
            str(args.timesteps),
            "--step",
            str(args.infer_step),
            "--batch-size",
            str(args.batch_size),
            "--labels",
            args.labels,
        ]
        if args.save_preview:
            infer_cmd.extend(
                ["--out-video", str(work_csv_dir / f"labeled_preview_{video_id}.mp4")]
            )
        if args.normalize_geometry:
            infer_cmd.append("--normalize-geometry")
        run_cmd(infer_cmd)

        run_cmd(
            [
                py,
                "tools/filter_segments.py",
                "--input-csv",
                str(segments_named),
                "--output-csv",
                str(segments_filtered),
                "--min-score",
                str(args.min_score),
                "--expected-labels",
                args.labels,
            ]
        )

        cut_cmd = [
            py,
            "tools/segments_to_clips.py",
            "--video",
            str(video_path),
            "--segments-csv",
            str(segments_filtered),
            "--output-dir",
            str(clips_dir),
            "--min-duration",
            str(args.min_duration),
            "--backend",
            args.backend,
            "--ffmpeg-bin",
            args.ffmpeg_bin,
            "--filename-prefix",
            video_id,
        ]
        if args.copy_codec:
            cut_cmd.append("--copy-codec")
        run_cmd(cut_cmd)

    dataset_cmd = [
        py,
        "video_to_dataset.py",
        "--input",
        str(clips_dir),
        "--output",
        str(dataset_dir),
        "--timesteps",
        str(args.timesteps),
        "--step",
        str(args.dataset_step),
        "--labels",
        args.labels,
        "--max-people",
        str(args.max_people),
        "--max-hands",
        str(args.max_hands),
    ]
    if args.normalize_geometry:
        dataset_cmd.append("--normalize-geometry")
    run_cmd(dataset_cmd)

    print("[OK] Dataset build complete.")
    print("[OK] Dataset dir:", dataset_dir)
    print("[OK] Work CSV dir:", work_csv_dir)


if __name__ == "__main__":
    main()
