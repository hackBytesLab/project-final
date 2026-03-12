import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser(
        description="Filter segments CSV by confidence score (avg_score)."
    )
    parser.add_argument("--input-csv", required=True, help="Input segments CSV")
    parser.add_argument("--output-csv", required=True, help="Output filtered CSV")
    parser.add_argument(
        "--min-score", type=float, default=0.60, help="Minimum avg_score threshold"
    )
    parser.add_argument(
        "--expected-labels",
        default="",
        help="Optional comma-separated labels for missing-class warning.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    kept_rows = []
    skipped = 0
    class_counts = Counter()

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("Input CSV has no header.")

        required = {"class_id", "class_name", "start_time_s", "end_time_s", "avg_score"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(f"Input CSV missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            score = parse_float(row.get("avg_score"), default=-1.0)
            if score is None or score < args.min_score:
                skipped += 1
                continue
            kept_rows.append(row)
            class_name = (row.get("class_name") or "").strip()
            class_counts[class_name] += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"[OK] Input rows: {len(kept_rows) + skipped}")
    print(f"[OK] Kept rows : {len(kept_rows)}")
    print(f"[OK] Skipped   : {skipped}")
    print(f"[OK] Output    : {output_csv}")

    print("[INFO] Class distribution after filtering:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[0]):
        print(f" - {class_name}: {count}")

    if args.expected_labels.strip():
        expected = [x.strip() for x in args.expected_labels.split(",") if x.strip()]
        missing_labels = [label for label in expected if class_counts[label] == 0]
        if missing_labels:
            print("[WARN] Missing classes after filtering:", ", ".join(missing_labels))


if __name__ == "__main__":
    main()
