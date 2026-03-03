import argparse
import csv
import json
import math
from pathlib import Path


def parse_labels(raw):
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def resolve_roc_path(eval_dir, roc_csv):
    if roc_csv:
        path = Path(roc_csv)
        if not path.exists():
            raise FileNotFoundError(f"roc csv not found: {path}")
        return path
    if not eval_dir:
        raise ValueError("Provide either --roc-csv or --eval-dir")
    candidate = Path(eval_dir) / "holdout" / "roc_curves.csv"
    if not candidate.exists():
        raise FileNotFoundError(f"roc csv not found: {candidate}")
    return candidate


def read_class_rows(roc_csv):
    by_class = {}
    with roc_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("curve_type") or "").strip().lower() != "class":
                continue
            class_name = (row.get("class_name") or "").strip()
            class_id_raw = (row.get("class_id") or "").strip()
            if not class_name:
                class_name = f"class_{class_id_raw}" if class_id_raw else "unknown"
            try:
                fpr = float(row.get("fpr", "nan"))
                tpr = float(row.get("tpr", "nan"))
                thr = float(row.get("threshold", "nan"))
            except ValueError:
                continue
            if not (math.isfinite(fpr) and math.isfinite(tpr) and math.isfinite(thr)):
                continue
            if class_name not in by_class:
                by_class[class_name] = []
            by_class[class_name].append(
                {
                    "fpr": fpr,
                    "tpr": tpr,
                    "threshold": thr,
                    "class_id": class_id_raw,
                }
            )
    return by_class


def recommend_threshold(points):
    best = None
    for p in points:
        youden_j = p["tpr"] - p["fpr"]
        dist_topleft = math.sqrt((1.0 - p["tpr"]) ** 2 + (p["fpr"] ** 2))
        candidate = {
            "threshold": p["threshold"],
            "youden_j": youden_j,
            "fpr": p["fpr"],
            "tpr": p["tpr"],
            "distance_to_topleft": dist_topleft,
        }
        if best is None:
            best = candidate
            continue
        if candidate["youden_j"] > best["youden_j"]:
            best = candidate
            continue
        if candidate["youden_j"] == best["youden_j"] and candidate["distance_to_topleft"] < best["distance_to_topleft"]:
            best = candidate
            continue
    return best


def build_outputs(roc_csv, by_class, labels=None):
    thresholds = {}
    details = {}

    class_names = sorted(by_class.keys())

    def resolve_output_name(class_name, points):
        if not labels:
            return class_name
        class_id_raw = ""
        if points:
            class_id_raw = str(points[0].get("class_id", "")).strip()
        if class_id_raw.isdigit():
            idx = int(class_id_raw)
            if 0 <= idx < len(labels):
                return labels[idx]
        if class_name in labels:
            return class_name
        return class_name

    if labels:
        def class_sort_key(name):
            points = by_class.get(name, [])
            class_id_raw = str(points[0].get("class_id", "")).strip() if points else ""
            if class_id_raw.isdigit():
                idx = int(class_id_raw)
                return (0 if 0 <= idx < len(labels) else 1, idx)
            return (2, name)
        class_names = sorted(class_names, key=class_sort_key)

    for class_name in class_names:
        points = by_class.get(class_name, [])
        if not points:
            continue
        output_name = resolve_output_name(class_name, points)
        best = recommend_threshold(points)
        thresholds[output_name] = float(best["threshold"])
        details[output_name] = {
            "recommended_threshold": float(best["threshold"]),
            "youden_j": float(best["youden_j"]),
            "tpr": float(best["tpr"]),
            "fpr": float(best["fpr"]),
            "distance_to_topleft": float(best["distance_to_topleft"]),
            "num_points": int(len(points)),
        }

    payload = {
        "method": "youden_j",
        "source_roc_csv": str(roc_csv.as_posix()),
        "thresholds": thresholds,
        "details": details,
    }
    return payload


def save_markdown(path, payload):
    lines = []
    lines.append("# Recommended Class Thresholds")
    lines.append("")
    lines.append(f"- method: `{payload.get('method')}`")
    lines.append(f"- source_roc_csv: `{payload.get('source_roc_csv')}`")
    lines.append("")
    lines.append("| class | threshold | youden_j | tpr | fpr | points |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    details = payload.get("details", {})
    for cls, v in details.items():
        lines.append(
            f"| {cls} | {v['recommended_threshold']:.6f} | {v['youden_j']:.6f} | {v['tpr']:.6f} | {v['fpr']:.6f} | {v['num_points']} |"
        )
    lines.append("")
    lines.append("Runtime example:")
    lines.append("```bash")
    lines.append("python main.py --model <MODEL_PATH> --labels Fall,No_Fall,Pre-Fall,Falling --thresholds-json <THRESHOLDS_JSON>")
    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Recommend class-specific thresholds from roc_curves.csv")
    parser.add_argument("--eval-dir", default="", help="Eval directory containing holdout/roc_curves.csv")
    parser.add_argument("--roc-csv", default="", help="Direct path to roc_curves.csv")
    parser.add_argument("--labels", default="", help="Optional comma-separated labels for ordering")
    parser.add_argument("--output-json", default="", help="Output JSON path")
    parser.add_argument("--output-md", default="", help="Output markdown summary path")
    args = parser.parse_args()

    roc_csv = resolve_roc_path(args.eval_dir, args.roc_csv)
    labels = parse_labels(args.labels)

    by_class = read_class_rows(roc_csv)
    if not by_class:
        raise RuntimeError(f"No class curves found in {roc_csv}")

    payload = build_outputs(roc_csv, by_class, labels=labels)

    out_json = Path(args.output_json) if args.output_json else roc_csv.parent / "recommended_thresholds.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] threshold json: {out_json}")

    if args.output_md:
        out_md = Path(args.output_md)
    else:
        out_md = out_json.with_suffix(".md")
    save_markdown(out_md, payload)
    print(f"[OK] threshold summary: {out_md}")


if __name__ == "__main__":
    main()
