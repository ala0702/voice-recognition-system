import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#dddddd",
    "grid.linewidth":     0.8,
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "figure.dpi":         100,
    "savefig.dpi":        150,
    "savefig.bbox":       "tight",
})

_C_GENUINE  = "#4C72B0"   # muted steel blue
_C_IMPOSTOR = "#C44E52"   # muted brick red
_C_EER      = "#DD8452"   # muted amber


# ── Core metrics ─────────────────────────────────────────────────────────────

def compute_roc(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_val = auc(fpr, tpr)
    return fpr, tpr, thresholds, auc_val


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    eer_threshold = float(interp1d(fnr, thresholds)(eer))
    return eer, eer_threshold


def compute_far_frr(labels, scores, thresholds):
    labels = np.array(labels)
    scores = np.array(scores)
    n_genuine  = labels.sum()
    n_impostor = (labels == 0).sum()
    far_vals, frr_vals = [], []
    for t in thresholds:
        decisions = scores >= t
        far_vals.append(np.sum((decisions == 1) & (labels == 0)) / n_impostor)
        frr_vals.append(np.sum((decisions == 0) & (labels == 1)) / n_genuine)
    return np.array(far_vals), np.array(frr_vals)


# ── Console + file output ─────────────────────────────────────────────────────

def print_metrics(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)

    fpr, tpr, _, auc_val = compute_roc(labels, scores)
    eer, eer_threshold    = compute_eer(labels, scores)
    far_at_eer, frr_at_eer = compute_far_frr(labels, scores, [eer_threshold])

    genuine_scores  = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    print(f"Trials          : {len(labels)}  ({labels.sum()} genuine / {(labels == 0).sum()} impostor)")
    print(f"AUC             : {auc_val:.4f}")
    print(f"EER             : {eer * 100:.2f}%")
    print(f"EER threshold   : {eer_threshold:.4f}")
    print(f"FAR @ EER       : {far_at_eer[0] * 100:.2f}%")
    print(f"FRR @ EER       : {frr_at_eer[0] * 100:.2f}%")
    print(f"Genuine  score  : {genuine_scores.mean():.4f} ± {genuine_scores.std():.4f}")
    print(f"Impostor score  : {impostor_scores.mean():.4f} ± {impostor_scores.std():.4f}")

    return {
        "auc": auc_val, "eer": eer, "eer_threshold": eer_threshold,
        "far_at_eer": far_at_eer[0], "frr_at_eer": frr_at_eer[0],
    }


def save_metrics(labels, scores, results_dir, prefix="", fixed_threshold=None):
    labels = np.array(labels)
    scores = np.array(scores)

    _, _, _, auc_val      = compute_roc(labels, scores)
    eer, eer_threshold    = compute_eer(labels, scores)
    far_at_eer, frr_at_eer = compute_far_frr(labels, scores, [eer_threshold])
    genuine_scores  = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    rows = [
        ("Trials (genuine / impostor)", f"{len(labels)}  ({labels.sum()} / {(labels == 0).sum()})"),
        ("AUC",                         f"{auc_val:.4f}"),
        ("EER",                         f"{eer * 100:.2f}%"),
        ("EER threshold",               f"{eer_threshold:.4f}"),
        ("FAR @ EER",                   f"{far_at_eer[0] * 100:.2f}%"),
        ("FRR @ EER",                   f"{frr_at_eer[0] * 100:.2f}%"),
        ("Genuine score (mean ± std)",  f"{genuine_scores.mean():.4f} ± {genuine_scores.std():.4f}"),
        ("Impostor score (mean ± std)", f"{impostor_scores.mean():.4f} ± {impostor_scores.std():.4f}"),
    ]
    if fixed_threshold is not None:
        far_f, frr_f = compute_far_frr(labels, scores, [fixed_threshold])
        rows.append((f"FAR @ {fixed_threshold:.4f} (EXP1)", f"{far_f[0] * 100:.2f}%"))
        rows.append((f"FRR @ {fixed_threshold:.4f} (EXP1)", f"{frr_f[0] * 100:.2f}%"))

    col1 = max(len(r[0]) for r in rows)
    col2 = max(len(r[1]) for r in rows)
    sep  = f"+{'-' * (col1 + 2)}+{'-' * (col2 + 2)}+"
    lines = [sep]
    for label, value in rows:
        lines.append(f"| {label:<{col1}} | {value:<{col2}} |")
        lines.append(sep)

    path = os.path.join(results_dir, f"{prefix}metrics.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Metrics saved to {path}")


# ── Per-condition plots (called by save_all_plots) ────────────────────────────

def plot_score_distribution(genuine_scores, impostor_scores, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(genuine_scores.min(), impostor_scores.min()),
        max(genuine_scores.max(), impostor_scores.max()),
        60,
    )
    ax.hist(impostor_scores, bins=bins, alpha=0.65, label="Impostor", color=_C_IMPOSTOR, density=True)
    ax.hist(genuine_scores,  bins=bins, alpha=0.65, label="Genuine",  color=_C_GENUINE,  density=True)
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution: Genuine vs Impostor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc(labels, scores, auc_val, eer, eer_threshold, path):
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=_C_GENUINE, lw=2.5, label=f"ROC  (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=1, linestyle="--", label="Random classifier")
    ax.scatter([eer], [1 - eer], color=_C_EER, zorder=5, s=80,
               label=f"EER = {eer * 100:.2f}%  (thr = {eer_threshold:.4f})")
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Acceptance Rate  (1 − FRR)")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_det(labels, scores, eer, path):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr * 100, fnr * 100, color=_C_GENUINE, lw=2.5)
    ax.scatter([eer * 100], [eer * 100], color=_C_EER, zorder=5, s=80,
               label=f"EER = {eer * 100:.2f}%")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("FAR (%)"); ax.set_ylabel("FRR (%)")
    ax.set_title("DET Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_far_frr_vs_threshold(labels, scores, eer_threshold, path):
    thresholds = np.linspace(min(scores), max(scores), 500)
    far, frr = compute_far_frr(labels, scores, thresholds)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, far * 100, label="FAR", color=_C_IMPOSTOR, lw=2)
    ax.plot(thresholds, frr * 100, label="FRR", color=_C_GENUINE,  lw=2)
    ax.axvline(eer_threshold, color=_C_EER, linestyle="--", lw=1.5,
               label=f"EER threshold = {eer_threshold:.4f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("FAR and FRR vs Threshold")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_all_plots(labels, scores, results_dir, prefix=""):
    labels = np.array(labels)
    scores = np.array(scores)
    _, _, _, auc_val = compute_roc(labels, scores)
    eer, eer_threshold = compute_eer(labels, scores)
    genuine_scores  = scores[labels == 1]
    impostor_scores = scores[labels == 0]
    p = lambda name: os.path.join(results_dir, f"{prefix}{name}")
    plot_score_distribution(genuine_scores, impostor_scores, p("score_distribution.png"))
    plot_roc(labels, scores, auc_val, eer, eer_threshold, p("roc_curve.png"))
    plot_det(labels, scores, eer, p("det_curve.png"))
    plot_far_frr_vs_threshold(labels, scores, eer_threshold, p("far_frr_threshold.png"))


# ── Comparison bar chart (groups) ─────────────────────────────────────────────

def plot_far_frr_by_group(groups, threshold, xlabel, path):
    names = [str(g[0]) for g in groups]
    far_vals, frr_vals = [], []
    for _, labels, scores in groups:
        labels, scores = np.array(labels), np.array(scores)
        far, frr = compute_far_frr(labels, scores, [threshold])
        far_vals.append(far[0] * 100)
        frr_vals.append(frr[0] * 100)

    x  = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, len(names) * 1.4), 5))

    def _bars(ax, vals, color, title, ylabel):
        bars = ax.bar(x, vals, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        top = max(vals) if vals else 1
        ax.set_ylim(0, top * 1.28)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.02,
                    f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    _bars(ax1, far_vals, _C_IMPOSTOR, "False Acceptance Rate", "FAR (%)")
    _bars(ax2, frr_vals, _C_GENUINE,  "False Rejection Rate",  "FRR (%)")

    fig.suptitle(f"FAR & FRR at threshold = {threshold:.4f}", fontsize=11, color="#555555")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── Noise experiment — trend charts ──────────────────────────────────────────

def plot_noise_progression(ratios_pct, conditions, path, ylabel="FRR (%)",
                           title="FRR vs Noise Level — Baseline vs Denoising"):
    """
    Line chart across noise levels for multiple conditions.

    conditions : list of (label, color, values_list)
                 values may be None for skipped conditions
    ratios_pct : e.g. [70, 85, 100, 110, 120]
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x        = np.arange(len(ratios_pct))
    x_labels = [f"{r}%" for r in ratios_pct]

    all_vals = [v for _, _, vals in conditions for v in vals if v is not None]
    y_max    = max(all_vals) * 1.3 if all_vals else 100

    for name, color, vals in conditions:
        xi = [i for i, v in enumerate(vals) if v is not None]
        yi = [v for v in vals if v is not None]
        if not xi:
            continue
        ax.plot(xi, yi, "o-", color=color, label=name, lw=2.5,
                markersize=9, markeredgecolor="white", markeredgewidth=1.5)
        for i, v in zip(xi, yi):
            ax.annotate(f"{v:.1f}%", (i, v),
                        textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Noise amplitude ratio  (relative to signal RMS)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n(lower = better)")
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_noise_grouped_bars(ratios_pct, conditions, path, ylabel="FRR (%)",
                            title="FRR Comparison: Baseline vs Denoising Methods Across Noise Levels"):
    """
    Grouped bar chart: one cluster per noise ratio, one bar per condition.

    conditions : list of (label, color, values_list)
    """
    n_ratios    = len(ratios_pct)
    valid_conds = [(n, c, v) for n, c, v in conditions
                   if any(x is not None for x in v)]
    n_conds = len(valid_conds)
    if n_conds == 0:
        return

    bar_w = 0.75 / n_conds
    x     = np.arange(n_ratios)

    all_vals = [v for _, _, vals in valid_conds for v in vals if v is not None]
    y_max    = max(all_vals) * 1.25 if all_vals else 100

    fig, ax = plt.subplots(figsize=(max(11, n_ratios * 2.2), 6))

    for i, (name, color, vals) in enumerate(valid_conds):
        offset = x + (i - n_conds / 2 + 0.5) * bar_w
        heights = [v if v is not None else 0 for v in vals]
        bars    = ax.bar(offset, heights, width=bar_w, color=color, label=name,
                         edgecolor="white", linewidth=0.6, alpha=0.85)
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + y_max * 0.01,
                        f"{v:.1f}%", ha="center", va="bottom",
                        fontsize=8.5, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}%" for r in ratios_pct], fontsize=11)
    ax.set_xlabel("Noise amplitude ratio  (relative to signal RMS)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n(lower = better)")
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.35, color="#cccccc")
    ax.grid(False, axis="x")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
