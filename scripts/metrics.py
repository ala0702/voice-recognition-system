import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d


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
    n_genuine = labels.sum()
    n_impostor = (labels == 0).sum()
    far_vals, frr_vals = [], []
    for t in thresholds:
        decisions = scores >= t
        far_vals.append(np.sum((decisions == 1) & (labels == 0)) / n_impostor)
        frr_vals.append(np.sum((decisions == 0) & (labels == 1)) / n_genuine)
    return np.array(far_vals), np.array(frr_vals)


def print_metrics(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)

    fpr, tpr, _, auc_val = compute_roc(labels, scores)
    eer, eer_threshold = compute_eer(labels, scores)
    far_at_eer, frr_at_eer = compute_far_frr(labels, scores, [eer_threshold])

    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    print(f"Trials          : {len(labels)}  ({labels.sum()} genuine / {(labels == 0).sum()} impostor)")
    print(f"AUC             : {auc_val:.4f}")
    print(f"EER             : {eer * 100:.2f}%")
    print(f"EER threshold   : {eer_threshold:.4f}")
    print(f"FAR @ EER       : {far_at_eer[0] * 100:.2f}%")
    print(f"FRR @ EER       : {frr_at_eer[0] * 100:.2f}%")
    print(f"Genuine  score  : {genuine_scores.mean():.4f} ± {genuine_scores.std():.4f}")
    print(f"Impostor score  : {impostor_scores.mean():.4f} ± {impostor_scores.std():.4f}")

    result = {
        "auc": auc_val,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "far_at_eer": far_at_eer[0],
        "frr_at_eer": frr_at_eer[0],
    }
    return result


def save_metrics(labels, scores, results_dir, prefix=""):
    labels = np.array(labels)
    scores = np.array(scores)

    _, _, _, auc_val = compute_roc(labels, scores)
    eer, eer_threshold = compute_eer(labels, scores)
    far_at_eer, frr_at_eer = compute_far_frr(labels, scores, [eer_threshold])

    genuine_scores = scores[labels == 1]
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

    col1 = max(len(r[0]) for r in rows)
    col2 = max(len(r[1]) for r in rows)
    sep = f"+{'-' * (col1 + 2)}+{'-' * (col2 + 2)}+"
    lines = [sep]
    for label, value in rows:
        lines.append(f"| {label:<{col1}} | {value:<{col2}} |")
        lines.append(sep)

    path = os.path.join(results_dir, f"{prefix}metrics.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Metrics saved to {path}")


def plot_score_distribution(genuine_scores, impostor_scores, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(impostor_scores, bins=50, alpha=0.6, label="Impostor", color="crimson", density=True)
    ax.hist(genuine_scores, bins=50, alpha=0.6, label="Genuine", color="steelblue", density=True)
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution: Genuine vs Impostor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(labels, scores, auc_val, eer, eer_threshold, path):
    fpr, tpr, _ = roc_curve(labels, scores)
    eer_far = eer
    eer_tar = 1 - eer
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.scatter([eer_far], [eer_tar], color="crimson", zorder=5,
               label=f"EER = {eer * 100:.2f}%  (threshold = {eer_threshold:.4f})")
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Acceptance Rate (1 - FRR)")
    ax.set_title("ROC Curve — Threshold Selection at EER")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_det(labels, scores, eer, path):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr * 100, fnr * 100, color="steelblue", lw=2)
    ax.scatter([eer * 100], [eer * 100], color="crimson", zorder=5, label=f"EER = {eer * 100:.2f}%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FAR (%)")
    ax.set_ylabel("FRR (%)")
    ax.set_title("DET Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_far_frr_vs_threshold(labels, scores, eer_threshold, path):
    thresholds = np.linspace(min(scores), max(scores), 500)
    far, frr = compute_far_frr(labels, scores, thresholds)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, far * 100, label="FAR", color="crimson")
    ax.plot(thresholds, frr * 100, label="FRR", color="steelblue")
    ax.axvline(eer_threshold, color="gray", linestyle="--", label=f"EER threshold = {eer_threshold:.4f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("FAR and FRR vs Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _bar_with_labels(ax, x, vals, color, ylabel, title):
    ax.bar(x, vals, color=color)
    ax.set_xticks(x)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    top = max(vals) if vals else 1
    ax.set_ylim(0, top * 1.2 + 1)
    for i, v in enumerate(vals):
        ax.text(i, v + top * 0.03, f"{v:.1f}%", ha="center", fontsize=9)


def plot_far_frr_by_group(groups, threshold, xlabel, path):
    names = [str(g[0]) for g in groups]
    far_vals, frr_vals = [], []
    for _, labels, scores in groups:
        labels, scores = np.array(labels), np.array(scores)
        far, frr = compute_far_frr(labels, scores, [threshold])
        far_vals.append(far[0] * 100)
        frr_vals.append(frr[0] * 100)

    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    _bar_with_labels(ax1, x, far_vals, "crimson", "FAR (%)", f"FAR per {xlabel}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_xlabel(xlabel)

    _bar_with_labels(ax2, x, frr_vals, "steelblue", "FRR (%)", f"FRR per {xlabel}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_xlabel(xlabel)

    fig.suptitle(f"FAR & FRR at threshold = {threshold:.4f}", fontsize=10, color="gray")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_all_plots(labels, scores, results_dir, prefix=""):
    labels = np.array(labels)
    scores = np.array(scores)

    _, _, _, auc_val = compute_roc(labels, scores)
    eer, eer_threshold = compute_eer(labels, scores)

    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    p = lambda name: f"{results_dir}/{prefix}{name}"
    plot_score_distribution(genuine_scores, impostor_scores, p("score_distribution.png"))
    plot_roc(labels, scores, auc_val, eer, eer_threshold, p("roc_curve.png"))
    plot_det(labels, scores, eer, p("det_curve.png"))
    plot_far_frr_vs_threshold(labels, scores, eer_threshold, p("far_frr_threshold.png"))
