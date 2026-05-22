import os
import random
import numpy as np
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, "../pretrained_models/spkrec-ecapa-voxceleb"))
TEST_DIR = os.path.normpath(os.path.join(BASE_DIR, "../data/test_genuine"))
PROFILES_PATH = os.path.normpath(os.path.join(BASE_DIR, "../voice_profiles.npy"))
RESULTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "../results"))

GENUINE_FILES = [f"test_{i}.wav" for i in range(1, 6)]
IMPOSTOR_FILES = [f"test_{i}.wav" for i in range(6, 11)]
RANDOM_SEED = 42


def load_model():
    model = ECAPA_TDNN(
        input_size=80,
        channels=[1024, 1024, 1024, 1024, 3072],
        lin_neurons=192,
    )
    state_dict = torch.load(
        os.path.join(MODEL_DIR, "embedding_model.ckpt"), map_location="cpu"
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def build_mel_extractor():
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=80,
    )


def extract_embedding(audio_path, model, mel_extractor):
    signal, _ = sf.read(audio_path)
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    log_mel = torch.log(mel_extractor(signal_tensor) + 1e-6).transpose(1, 2)
    with torch.no_grad():
        return model(log_mel).squeeze()


def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def build_trials(speakers, voice_database, rng):
    trials = []
    enrolled = set(voice_database.keys())
    for speaker in speakers:
        if speaker not in enrolled:
            continue
        for fname in GENUINE_FILES:
            path = os.path.join(TEST_DIR, speaker, fname)
            if os.path.exists(path):
                trials.append((speaker, path, 1))

        candidates = [s for s in speakers if s != speaker and s in enrolled]
        for fname in IMPOSTOR_FILES:
            path = os.path.join(TEST_DIR, speaker, fname)
            if os.path.exists(path):
                claimed = rng.choice(candidates)
                trials.append((claimed, path, 0))

    return trials


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


def plot_roc(labels, scores, auc_val, path):
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("1 - FRR (TAR)")
    ax.set_title("ROC Curve")
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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    voice_database = np.load(PROFILES_PATH, allow_pickle=True).item()
    speakers = sorted(
        s for s in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, s))
    )

    model = load_model()
    mel_extractor = build_mel_extractor()

    trials = build_trials(speakers, voice_database, rng)
    print(f"Total trials: {len(trials)}")

    labels, scores = [], []
    for i, (claimed_id, audio_path, label) in enumerate(trials):
        try:
            test_emb = extract_embedding(audio_path, model, mel_extractor)
            ref = torch.tensor(voice_database[claimed_id])
            scores.append(cosine_similarity(test_emb, ref))
            labels.append(label)
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(trials)}")

    labels = np.array(labels)
    scores = np.array(scores)

    fpr, tpr, _ = roc_curve(labels, scores)
    auc_val = auc(fpr, tpr)
    eer, eer_threshold = compute_eer(labels, scores)
    far_at_eer, frr_at_eer = compute_far_frr(labels, scores, [eer_threshold])

    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    print(f"\n=== Experiment 1 — Verification Metrics ===")
    print(f"Trials          : {len(labels)}  ({labels.sum()} genuine / {(labels == 0).sum()} impostor)")
    print(f"AUC             : {auc_val:.4f}")
    print(f"EER             : {eer * 100:.2f}%")
    print(f"EER threshold   : {eer_threshold:.4f}")
    print(f"FAR @ EER       : {far_at_eer[0] * 100:.2f}%")
    print(f"FRR @ EER       : {frr_at_eer[0] * 100:.2f}%")
    print(f"Genuine  score  : {genuine_scores.mean():.4f} ± {genuine_scores.std():.4f}")
    print(f"Impostor score  : {impostor_scores.mean():.4f} ± {impostor_scores.std():.4f}")

    plot_score_distribution(genuine_scores, impostor_scores, os.path.join(RESULTS_DIR, "score_distribution.png"))
    plot_roc(labels, scores, auc_val, os.path.join(RESULTS_DIR, "roc_curve.png"))
    plot_det(labels, scores, eer, os.path.join(RESULTS_DIR, "det_curve.png"))
    plot_far_frr_vs_threshold(labels, scores, eer_threshold, os.path.join(RESULTS_DIR, "far_frr_threshold.png"))

    print(f"\nPlots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
