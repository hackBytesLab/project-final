import argparse
import csv
import json
import os
from pathlib import Path
from typing import Optional


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel(from_dir: Path, to_path: Path) -> str:
    return os.path.relpath(to_path.resolve(), start=from_dir.resolve()).replace("\\", "/")


def parse_confusion_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    headers = rows[0][1:]
    matrix = []
    labels = []
    for r in rows[1:]:
        labels.append(r[0])
        matrix.append([int(x) for x in r[1:]])
    return headers, labels, matrix


def top_confusions(labels, matrix, top_k=3):
    items = []
    for i, true_name in enumerate(labels):
        for j, pred_name in enumerate(labels):
            if i == j:
                continue
            count = matrix[i][j]
            if count > 0:
                items.append((count, true_name, pred_name))
    items.sort(reverse=True)
    return items[:top_k]


def fmt(x, digits=4):
    return f"{float(x):.{digits}f}"


def get_texts(lang: str):
    if lang == "th":
        return {
            "title": "RESULT & DISCUSSION",
            "dataset_setup": "## 1) ชุดข้อมูลและการตั้งค่า",
            "validation": "Validation",
            "split_unit": "Split unit",
            "samples": "จำนวนตัวอย่าง",
            "holdout": "Holdout",
            "balance_mode": "Balance mode",
            "class_distribution": "การกระจายคลาส",
            "key_metrics": "## 2) ค่าชี้วัดหลัก (Holdout)",
            "metric_col": "Metric",
            "value_col": "Value",
            "accuracy": "Accuracy",
            "macro_precision": "Macro Precision",
            "macro_recall": "Macro Recall",
            "macro_f1": "Macro F1",
            "micro_auc": "Micro AUC",
            "macro_auc": "Macro AUC",
            "per_class": "## 3) ผลลัพธ์รายคลาส",
            "class_col": "Class",
            "precision_col": "Precision",
            "recall_col": "Recall",
            "f1_col": "F1",
            "support_col": "Support",
            "auc_col": "AUC",
            "viz": "## 4) Visualization",
            "discussion": "## 5) Discussion",
            "strength": "จุดแข็ง: โมเดลตรวจจับ `Pre-Fall` ได้ดีใน holdout (recall สูง)",
            "weakness": "จุดที่ต้องปรับ: คลาส `Falling` ยังพลาดสูง (recall ต่ำ/ศูนย์ในหลายรอบ)",
            "main_confusions": "ความสับสนหลักจาก confusion matrix:",
            "confusion_line": "`{true_name}` ถูกทำนายเป็น `{pred_name}` จำนวน **{count}** ครั้ง",
            "imbalance_note": "ข้อมูลยังไม่สมดุล ทำให้โมเดลเอนเอียงไปคลาสหลัก (`No_Fall`)",
            "next_step": "แนวทางถัดไป: เพิ่มตัวอย่าง `Falling`/`Pre-Fall` และปรับ loss (เช่น focal loss + class weight)",
            "source": "Source run",
            "html_lang": "th",
            "html_eval": "สรุปผลการประเมิน",
            "html_per_class": "สรุปรายคลาส",
            "html_plots": "กราฟประกอบ",
            "html_discussion": "อภิปรายผล",
            "html_confusion": "ความสับสน",
            "html_imbalanced": "ข้อมูลยัง imbalance สูง ทำให้โมเดลเอนเอียงไปคลาสใหญ่ (`No_Fall`)",
        }
    return {
        "title": "RESULT & DISCUSSION",
        "dataset_setup": "## 1) Dataset & Setup",
        "validation": "Validation",
        "split_unit": "Split unit",
        "samples": "Samples",
        "holdout": "Holdout",
        "balance_mode": "Balance mode",
        "class_distribution": "Class distribution",
        "key_metrics": "## 2) Key Metrics (Holdout)",
        "metric_col": "Metric",
        "value_col": "Value",
        "accuracy": "Accuracy",
        "macro_precision": "Macro Precision",
        "macro_recall": "Macro Recall",
        "macro_f1": "Macro F1",
        "micro_auc": "Micro AUC",
        "macro_auc": "Macro AUC",
        "per_class": "## 3) Per-Class Performance",
        "class_col": "Class",
        "precision_col": "Precision",
        "recall_col": "Recall",
        "f1_col": "F1",
        "support_col": "Support",
        "auc_col": "AUC",
        "viz": "## 4) Visualization",
        "discussion": "## 5) Discussion",
        "strength": "Strength: `Pre-Fall` is detected well in holdout (high recall).",
        "weakness": "Weakness: `Falling` still has high miss rate (low/zero recall in some runs).",
        "main_confusions": "Main confusions from the confusion matrix:",
        "confusion_line": "`{true_name}` predicted as `{pred_name}` **{count}** times",
        "imbalance_note": "Data is still highly imbalanced, biasing the model toward major classes (`No_Fall`).",
        "next_step": "Next step: add more `Falling`/`Pre-Fall` samples and tune loss (e.g., focal loss + class weight).",
        "source": "Source run",
        "html_lang": "en",
        "html_eval": "Evaluation Snapshot",
        "html_per_class": "Per-Class Summary",
        "html_plots": "Plots",
        "html_discussion": "Discussion",
        "html_confusion": "Confusion",
        "html_imbalanced": "Data remains imbalanced, so predictions are biased toward major classes (`No_Fall`).",
    }


def build_markdown_and_html(
    run_dir: Path, out_md: Path, out_html: Optional[Path] = None, lang: str = "en"
):
    t = get_texts(lang)
    holdout_dir = run_dir / "holdout"
    summary_dir = run_dir / "summary"

    metrics = load_json(holdout_dir / "metrics_summary.json")
    report = load_json(holdout_dir / "classification_report.json")
    auc_summary = load_json(holdout_dir / "auc_summary.json")
    overview = load_json(summary_dir / "overview.json")
    _, labels, cm_raw = parse_confusion_csv(holdout_dir / "confusion_matrix_raw.csv")
    confusions = top_confusions(labels, cm_raw, top_k=3)

    class_rows = []
    for label in labels:
        v = report.get(label, {})
        class_rows.append(
            {
                "label": label,
                "precision": float(v.get("precision", 0.0)),
                "recall": float(v.get("recall", 0.0)),
                "f1": float(v.get("f1-score", 0.0)),
                "support": int(v.get("support", 0)),
            }
        )

    out_dir = out_md.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    img_cm_norm = rel(out_dir, holdout_dir / "confusion_matrix_norm.png")
    img_cm_raw = rel(out_dir, holdout_dir / "confusion_matrix_raw.png")
    img_roc = rel(out_dir, holdout_dir / "roc_curve.png")
    img_learning = rel(out_dir, summary_dir / "learning_curves.png")

    lines = []
    lines.append(f"# {t['title']}")
    lines.append("")
    lines.append(t["dataset_setup"])
    lines.append("")
    lines.append(
        f"- {t['validation']}: `{overview.get('validation_mode')}` | {t['split_unit']}: `{overview.get('split_unit_effective')}`"
    )
    lines.append(
        f"- {t['samples']}: **{overview.get('total_samples')}** | {t['holdout']}: **{overview.get('split_info', {}).get('holdout_samples', 'N/A')}**"
    )
    lines.append(f"- {t['balance_mode']}: `{overview.get('balance_mode', 'none')}`")
    lines.append(
        f"- {t['class_distribution']}: {overview.get('class_distribution')}"
    )
    lines.append("")
    lines.append(t["key_metrics"])
    lines.append("")
    lines.append(f"| {t['metric_col']} | {t['value_col']} |")
    lines.append("|---|---:|")
    lines.append(f"| {t['accuracy']} | {fmt(metrics['accuracy'])} |")
    lines.append(f"| {t['macro_precision']} | {fmt(metrics['macro_precision'])} |")
    lines.append(f"| {t['macro_recall']} | {fmt(metrics['macro_recall'])} |")
    lines.append(f"| {t['macro_f1']} | {fmt(metrics['macro_f1'])} |")
    lines.append(f"| {t['micro_auc']} | {fmt(metrics['micro_auc'])} |")
    lines.append(f"| {t['macro_auc']} | {fmt(metrics['macro_auc'])} |")
    lines.append("")
    lines.append(t["per_class"])
    lines.append("")
    lines.append(
        f"| {t['class_col']} | {t['precision_col']} | {t['recall_col']} | {t['f1_col']} | {t['support_col']} | {t['auc_col']} |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in class_rows:
        auc_v = auc_summary.get("per_class", {}).get(r["label"], 0.0)
        lines.append(
            f"| {r['label']} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | {r['support']} | {fmt(auc_v)} |"
        )
    lines.append("")
    lines.append(t["viz"])
    lines.append("")
    lines.append(f"![Confusion Matrix (Norm)]({img_cm_norm})")
    lines.append("")
    lines.append(f"![Confusion Matrix (Raw)]({img_cm_raw})")
    lines.append("")
    lines.append(f"![ROC Curve]({img_roc})")
    lines.append("")
    lines.append(f"![Learning Curves]({img_learning})")
    lines.append("")
    lines.append(t["discussion"])
    lines.append("")
    lines.append(f"- {t['strength']}")
    lines.append(f"- {t['weakness']}")
    if confusions:
        lines.append(f"- {t['main_confusions']}")
        for count, true_name, pred_name in confusions:
            lines.append(
                "  - "
                + t["confusion_line"].format(
                    true_name=true_name, pred_name=pred_name, count=count
                )
            )
    lines.append(f"- {t['imbalance_note']}")
    lines.append(f"- {t['next_step']}")
    lines.append("")
    lines.append(f"_{t['source']}: `{run_dir.as_posix()}`_")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    if out_html:
        html = f"""<!doctype html>
<html lang="{t['html_lang']}">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{t['title']}</title>
  <style>
    :root {{
      --bg: #eef2ff;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #4b5563;
      --accent: #1d4ed8;
      --line: #dbe2f5;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at 20% 20%, #f8faff 0, var(--bg) 45%, #e5ebff 100%);
    }}
    .wrap {{
      max-width: 1180px;
      margin: 24px auto;
      padding: 20px;
    }}
    h1 {{
      margin: 0 0 16px;
      font-size: 44px;
      letter-spacing: 1px;
      color: #0f172a;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 16px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
    }}
    .card h2 {{
      margin: 0 0 10px;
      font-size: 22px;
      color: #111827;
    }}
    .metric-table, .class-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .metric-table th, .metric-table td, .class-table th, .class-table td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      text-align: right;
    }}
    .metric-table th:first-child, .metric-table td:first-child,
    .class-table th:first-child, .class-table td:first-child {{
      text-align: left;
    }}
    .img-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .img-grid img {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    .note {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}
    ul {{
      margin: 8px 0 0;
      padding-left: 18px;
    }}
    li {{
      margin: 6px 0;
    }}
    code {{
      background: #f3f6ff;
      padding: 2px 6px;
      border-radius: 6px;
    }}
    .footer {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 960px) {{
      .grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 34px; }}
      .img-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{t['title']}</h1>
    <div class="grid">
      <section class="card">
        <h2>{t['html_eval']}</h2>
        <p class="note">
          {t['validation']}: <code>{overview.get('validation_mode')}</code> |
          {t['split_unit']}: <code>{overview.get('split_unit_effective')}</code> |
          {t['samples']}: <b>{overview.get('total_samples')}</b> |
          {t['holdout']}: <b>{overview.get('split_info', {}).get('holdout_samples', 'N/A')}</b>
        </p>
        <table class="metric-table">
          <tr><th>{t['metric_col']}</th><th>{t['value_col']}</th></tr>
          <tr><td>{t['accuracy']}</td><td>{fmt(metrics['accuracy'])}</td></tr>
          <tr><td>{t['macro_precision']}</td><td>{fmt(metrics['macro_precision'])}</td></tr>
          <tr><td>{t['macro_recall']}</td><td>{fmt(metrics['macro_recall'])}</td></tr>
          <tr><td>{t['macro_f1']}</td><td>{fmt(metrics['macro_f1'])}</td></tr>
          <tr><td>{t['micro_auc']}</td><td>{fmt(metrics['micro_auc'])}</td></tr>
          <tr><td>{t['macro_auc']}</td><td>{fmt(metrics['macro_auc'])}</td></tr>
        </table>

        <h2 style="margin-top:14px;">{t['html_per_class']}</h2>
        <table class="class-table">
          <tr><th>{t['class_col']}</th><th>{t['precision_col']}</th><th>{t['recall_col']}</th><th>{t['f1_col']}</th><th>{t['support_col']}</th></tr>
          {''.join(
              f"<tr><td>{r['label']}</td><td>{fmt(r['precision'])}</td><td>{fmt(r['recall'])}</td><td>{fmt(r['f1'])}</td><td>{r['support']}</td></tr>"
              for r in class_rows
          )}
        </table>
      </section>

      <section class="card">
        <h2>{t['html_plots']}</h2>
        <div class="img-grid">
          <img src="{img_cm_norm}" alt="Confusion matrix normalized" />
          <img src="{img_cm_raw}" alt="Confusion matrix raw" />
          <img src="{img_roc}" alt="ROC curve" />
          <img src="{img_learning}" alt="Learning curves" />
        </div>
      </section>
    </div>

    <section class="card" style="margin-top:16px;">
      <h2>{t['html_discussion']}</h2>
      <ul>
        <li>{t['strength']}</li>
        <li>{t['weakness']}</li>
        {''.join(f"<li>{t['html_confusion']}: <code>{true_name}</code> -> <code>{pred_name}</code> <b>{count}</b></li>" for count, true_name, pred_name in confusions)}
        <li>{t['html_imbalanced']}</li>
      </ul>
      <div class="footer">{t['source']}: <code>{run_dir.as_posix()}</code></div>
    </section>
  </div>
</body>
</html>
"""
        out_html.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build RESULT & DISCUSSION report from training outputs.")
    parser.add_argument("--run-dir", required=True, help="Path to train reports directory containing holdout/ and summary/")
    parser.add_argument("--out-md", default="", help="Output markdown path")
    parser.add_argument("--out-html", default="", help="Output html path (optional)")
    parser.add_argument("--lang", choices=["en", "th"], default="en", help="Report language")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run-dir not found: {run_dir}")

    out_md = Path(args.out_md) if args.out_md else run_dir / "summary" / "result_discussion.md"
    out_html = Path(args.out_html) if args.out_html else None
    build_markdown_and_html(run_dir, out_md, out_html, lang=args.lang)
    print(f"[OK] Report generated: {out_md}")
    if out_html:
        print(f"[OK] HTML generated: {out_html}")


if __name__ == "__main__":
    main()
