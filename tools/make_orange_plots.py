import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _set_orange_theme():
    plt.rcParams.update(
        {
            "figure.facecolor": "#fffaf3",
            "axes.facecolor": "#fff7eb",
            "axes.edgecolor": "#b45309",
            "axes.labelcolor": "#7c2d12",
            "xtick.color": "#7c2d12",
            "ytick.color": "#7c2d12",
            "text.color": "#7c2d12",
            "grid.color": "#fdba74",
            "grid.alpha": 0.35,
            "axes.grid": True,
        }
    )


def plot_learning_curves(summary_dir: Path, dpi: int):
    history_path = summary_dir / "train_history.csv"
    if not history_path.exists():
        return None
    df = pd.read_csv(history_path)
    if df.empty or "epoch" not in df.columns:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1, ax2 = axes
    ax1.plot(df["epoch"], df["loss"], label="train loss", color="#f97316", linewidth=2.2)
    ax1.plot(df["epoch"], df["val_loss"], label="val loss", color="#ea580c", linewidth=2.2)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(frameon=True)

    if "accuracy" in df.columns and "val_accuracy" in df.columns:
        ax2.plot(df["epoch"], df["accuracy"], label="train acc", color="#fb923c", linewidth=2.2)
        ax2.plot(df["epoch"], df["val_accuracy"], label="val acc", color="#c2410c", linewidth=2.2)
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend(frameon=True)

    fig.suptitle("Learning Curves (Orange Theme)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path = summary_dir / "learning_curves_orange.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_roc_curve(holdout_dir: Path, dpi: int):
    roc_csv = holdout_dir / "roc_curves.csv"
    if not roc_csv.exists():
        return None
    df = pd.read_csv(roc_csv)
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    class_df = df[df["curve_type"] == "class"]
    class_names = [c for c in class_df["class_name"].dropna().unique()]
    class_colors = ["#fed7aa", "#fdba74", "#fb923c", "#f97316", "#ea580c"]

    for i, class_name in enumerate(class_names):
        part = class_df[class_df["class_name"] == class_name]
        color = class_colors[i % len(class_colors)]
        ax.plot(part["fpr"], part["tpr"], color=color, linewidth=2, label=f"{class_name}")

    micro = df[df["curve_type"] == "micro"]
    if not micro.empty:
        ax.plot(micro["fpr"], micro["tpr"], color="#9a3412", linestyle="--", linewidth=2, label="micro")

    macro = df[df["curve_type"] == "macro"]
    if not macro.empty:
        ax.plot(macro["fpr"], macro["tpr"], color="#7c2d12", linestyle="-.", linewidth=2, label="macro")

    ax.plot([0, 1], [0, 1], color="#c2410c", linestyle=":", linewidth=1.5, label="chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Orange Theme)")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    out_path = holdout_dir / "roc_curve_orange.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_confusion(csv_path: Path, out_path: Path, title: str, is_normalized: bool, dpi: int):
    df = pd.read_csv(csv_path, index_col=0)
    mat = df.values.astype(float)
    labels = df.index.to_list()

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(mat, cmap="Oranges", vmin=0.0 if is_normalized else None, vmax=1.0 if is_normalized else None)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = (mat.max() if mat.size else 0.0) * 0.55
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            text = f"{value:.3f}" if is_normalized else f"{int(round(value))}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "#7c2d12",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_confusions(holdout_dir: Path, dpi: int):
    results = []
    norm_csv = holdout_dir / "confusion_matrix_norm.csv"
    raw_csv = holdout_dir / "confusion_matrix_raw.csv"
    if norm_csv.exists():
        results.append(
            _plot_confusion(
                norm_csv,
                holdout_dir / "confusion_matrix_norm_orange.png",
                "Confusion Matrix (Normalized, Orange Theme)",
                is_normalized=True,
                dpi=dpi,
            )
        )
    if raw_csv.exists():
        results.append(
            _plot_confusion(
                raw_csv,
                holdout_dir / "confusion_matrix_raw_orange.png",
                "Confusion Matrix (Raw, Orange Theme)",
                is_normalized=False,
                dpi=dpi,
            )
        )
    return results


def plot_auc_bar(holdout_dir: Path, dpi: int):
    auc_path = holdout_dir / "auc_summary.json"
    if not auc_path.exists():
        return None
    with auc_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    per_class = data.get("per_class", {})
    labels = list(per_class.keys()) + ["micro", "macro", "weighted"]
    values = list(per_class.values()) + [
        data.get("micro_auc", np.nan),
        data.get("macro_auc", np.nan),
        data.get("weighted_auc", np.nan),
    ]
    colors = ["#fdba74"] * len(per_class) + ["#fb923c", "#f97316", "#c2410c"]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, values, color=colors, edgecolor="#7c2d12", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUC")
    ax.set_title("AUC Summary (Orange Theme)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, v in zip(bars, values):
        if np.isnan(v):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, min(v + 0.02, 1.02), f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path = holdout_dir / "auc_bar_orange.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate orange-themed report charts from a train.py report folder.")
    parser.add_argument("--reports-dir", required=True, help="Path like work_csv/eval_xxx_timestamp")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    holdout_dir = reports_dir / "holdout"
    summary_dir = reports_dir / "summary"
    if not holdout_dir.exists() or not summary_dir.exists():
        raise FileNotFoundError(f"Invalid reports dir: {reports_dir}")

    _set_orange_theme()
    generated = []
    for path in [
        plot_learning_curves(summary_dir, args.dpi),
        plot_roc_curve(holdout_dir, args.dpi),
        plot_auc_bar(holdout_dir, args.dpi),
    ]:
        if path is not None:
            generated.append(path)
    generated.extend(plot_confusions(holdout_dir, args.dpi))

    for p in generated:
        print(str(p))


if __name__ == "__main__":
    main()
