import argparse
from pathlib import Path


DEFAULT_DIRS = [
    "data_long",
    "work_csv",
    "data_videos/Fall",
    "data_videos/No_Fall",
    "data_videos/Pre-Fall",
    "data_videos/Falling",
    "refs",
]


def main():
    parser = argparse.ArgumentParser(
        description="Create standard workspace folders for auto-label -> retrain pipeline."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory where folders will be created.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    created = []
    existed = []

    for rel_dir in DEFAULT_DIRS:
        target = root / rel_dir
        if target.exists():
            existed.append(str(target))
        else:
            target.mkdir(parents=True, exist_ok=True)
            created.append(str(target))

    print("Workspace root:", root)
    print("Created:")
    for p in created:
        print(" -", p)

    print("Already existed:")
    for p in existed:
        print(" -", p)


if __name__ == "__main__":
    main()
