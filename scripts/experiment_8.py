"""
Experiment 8: High-noise verification (P3 Challenge 1).

P3 specification: "nagrania z duzym poziomem halasu, tzn. takie, w ktorych
amplituda halasu wynosi od 70% do 120% amplitudy sygnalu" (~0-3 dB SNR).

Section 1 — Baseline   : show degradation across amplitude ratios 70-120 %.
Section 2 — Enhanced   : same noisy signals + noisereduce preprocessing.

Both sections process the EXACT SAME noisy signals so the only variable is
whether spectral denoising is applied — a clean ablation design.
"""
import os
import sys
import random
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verification import voice_database, model, mel_extractor
from metrics import (print_metrics, save_metrics, save_all_plots,
                     plot_far_frr_by_group, plot_noise_progression,
                     plot_noise_grouped_bars)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.normpath(os.path.join(BASE_DIR, "../data/test_genuine"))
RESULTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "../results/experiment_8"))

GENUINE_FILES = [f"test_{i}.wav" for i in range(1, 6)]
IMPOSTOR_FILES = [f"test_{i}.wav" for i in range(6, 11)]
AMPLITUDE_RATIOS = [0.70, 0.85, 1.00, 1.10, 1.20]
RANDOM_SEED = 42
EXP1_THRESHOLD = 0.7301
SAMPLE_RATE = 16000


def _read_eer(prefix):
    """Read EER (%) from a saved metrics.txt file."""
    path = os.path.join(RESULTS_DIR, f"{prefix}metrics.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "| EER" in line and "threshold" not in line.lower():
                parts = line.split("|")
                if len(parts) >= 3:
                    try:
                        return float(parts[2].strip().rstrip("%"))
                    except ValueError:
                        pass
    return None


def add_amplitude_noise(signal, amplitude_ratio, rng):
    """Add Gaussian noise: noise_RMS = amplitude_ratio * signal_RMS."""
    signal_rms = np.sqrt(np.mean(signal ** 2))
    if signal_rms == 0:
        return signal
    noise = rng.normal(0, amplitude_ratio * signal_rms, len(signal))
    return (signal + noise).astype(np.float32)


def denoise(signal, stationary=False):
    import noisereduce as nr
    return nr.reduce_noise(y=signal, sr=SAMPLE_RATE, stationary=stationary).astype(np.float32)


def extract_embedding(signal):
    t = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    log_mel = torch.log(mel_extractor(t) + 1e-6).transpose(1, 2)
    with torch.no_grad():
        return model(log_mel).squeeze()


def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def build_trials(speakers, rng):
    enrolled = set(voice_database.keys())
    all_trials = []
    for speaker in speakers:
        if speaker not in enrolled:
            continue
        for fname in GENUINE_FILES:
            path = os.path.join(TEST_DIR, speaker, fname)
            if os.path.exists(path):
                all_trials.append((speaker, path, 1))
        candidates = [s for s in speakers if s != speaker and s in enrolled]
        if not candidates:
            continue
        for fname in IMPOSTOR_FILES:
            path = os.path.join(TEST_DIR, speaker, fname)
            if os.path.exists(path):
                all_trials.append((rng.choice(candidates), path, 0))
    rng.shuffle(all_trials)
    return all_trials


def precompute_noisy(trials, amplitude_ratio):
    """Pre-generate all noisy signals for a given ratio (deterministic seed)."""
    rng = np.random.default_rng(RANDOM_SEED + int(amplitude_ratio * 100))
    noisy_map = {}
    for _, audio_path, _ in trials:
        if audio_path in noisy_map:
            continue
        try:
            signal, _ = sf.read(audio_path)
            noisy_map[audio_path] = add_amplitude_noise(signal.astype(np.float32),
                                                        amplitude_ratio, rng)
        except Exception:
            noisy_map[audio_path] = None
    return noisy_map


def run_trials(trials, noisy_map, use_denoise=False, stationary=False):
    labels, scores = [], []
    for claimed_id, audio_path, label in trials:
        noisy = noisy_map.get(audio_path)
        if noisy is None:
            continue
        try:
            processed = denoise(noisy, stationary=stationary) if use_denoise else noisy
            emb = extract_embedding(processed)
            ref = torch.tensor(voice_database[claimed_id])
            scores.append(cosine_similarity(emb, ref))
            labels.append(label)
        except Exception:
            continue
    return np.array(labels), np.array(scores)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng_trials = random.Random(RANDOM_SEED)

    speakers = sorted(
        s for s in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, s))
    )
    trials = build_trials(speakers, rng_trials)
    n_genuine = sum(l for _, _, l in trials)
    print(f"Total trials: {len(trials)} ({n_genuine} genuine / {len(trials) - n_genuine} impostor)")

    try:
        import noisereduce  # noqa: F401
    except ImportError:
        print("\nWARNING: noisereduce not installed — skipping Sections 2 and 3.")
        print("Install with:  pip install noisereduce")
        return

    # ---------- Section 1: Baseline (no denoising) ----------
    s1_done = os.path.exists(os.path.join(RESULTS_DIR, "baseline_120pct_metrics.txt"))
    if s1_done:
        print("\n=== Section 1: SKIPPED (results already exist) ===")
    else:
        print("\n=== Section 1: Baseline — no denoising ===")
        groups_baseline = []
        for ratio in AMPLITUDE_RATIOS:
            ratio_pct = int(ratio * 100)
            print(f"\n--- Amplitude ratio = {ratio} ({ratio_pct}% of signal RMS) ---")
            noisy_map = precompute_noisy(trials, ratio)
            labels, scores = run_trials(trials, noisy_map, use_denoise=False)
            print_metrics(labels, scores)
            save_metrics(labels, scores, RESULTS_DIR, prefix=f"baseline_{ratio_pct}pct_")
            save_all_plots(labels, scores, RESULTS_DIR, prefix=f"baseline_{ratio_pct}pct_")
            groups_baseline.append((f"{ratio_pct}%", labels, scores))
        plot_far_frr_by_group(
            groups_baseline, EXP1_THRESHOLD, "Noise amplitude ratio",
            os.path.join(RESULTS_DIR, "baseline_far_frr_by_ratio.png"),
        )

    # ---------- Section 2: Enhanced — stationary=False ----------
    s2_done = os.path.exists(os.path.join(RESULTS_DIR, "enhanced_120pct_metrics.txt"))
    if s2_done:
        print("\n=== Section 2: SKIPPED (results already exist) ===")
    else:
        print("\n=== Section 2: Enhanced — noisereduce (stationary=False) ===")
        groups_enhanced = []
        for ratio in AMPLITUDE_RATIOS:
            ratio_pct = int(ratio * 100)
            print(f"\n--- Amplitude ratio = {ratio} ({ratio_pct}%) + noisereduce (non-stationary) ---")
            noisy_map = precompute_noisy(trials, ratio)
            labels, scores = run_trials(trials, noisy_map, use_denoise=True, stationary=False)
            print_metrics(labels, scores)
            save_metrics(labels, scores, RESULTS_DIR, prefix=f"enhanced_{ratio_pct}pct_")
            save_all_plots(labels, scores, RESULTS_DIR, prefix=f"enhanced_{ratio_pct}pct_")
            groups_enhanced.append((f"{ratio_pct}%+NR", labels, scores))
        plot_far_frr_by_group(
            groups_enhanced, EXP1_THRESHOLD, "Noise amplitude ratio",
            os.path.join(RESULTS_DIR, "enhanced_far_frr_by_ratio.png"),
        )

    # ---------- Section 3: Enhanced — stationary=True ----------
    s3_done = os.path.exists(os.path.join(RESULTS_DIR, "stationary_120pct_metrics.txt"))
    if s3_done:
        print("\n=== Section 3: SKIPPED (results already exist) ===")
    else:
        print("\n=== Section 3: Enhanced — noisereduce (stationary=True) ===")
        groups_stationary = []
        for ratio in AMPLITUDE_RATIOS:
            ratio_pct = int(ratio * 100)
            print(f"\n--- Amplitude ratio = {ratio} ({ratio_pct}%) + noisereduce (stationary) ---")
            noisy_map = precompute_noisy(trials, ratio)
            labels, scores = run_trials(trials, noisy_map, use_denoise=True, stationary=True)
            print_metrics(labels, scores)
            save_metrics(labels, scores, RESULTS_DIR, prefix=f"stationary_{ratio_pct}pct_")
            save_all_plots(labels, scores, RESULTS_DIR, prefix=f"stationary_{ratio_pct}pct_")
            groups_stationary.append((f"{ratio_pct}%+NR-S", labels, scores))
        plot_far_frr_by_group(
            groups_stationary, EXP1_THRESHOLD, "Noise amplitude ratio",
            os.path.join(RESULTS_DIR, "stationary_far_frr_by_ratio.png"),
        )

    # ---------- FRR trend charts: all ratios × all conditions ----------
    print("\n=== Computing FRR @ EXP1 threshold for trend charts ===")
    from metrics import compute_far_frr

    ratios_pct = [int(r * 100) for r in AMPLITUDE_RATIOS]
    COND_SPEC = [
        ("Baseline",        "#C44E52", False, False),  # muted brick red
        ("NR (non-stat.)",  "#4C72B0", True,  False),  # muted steel blue
        ("NR (stationary)", "#55A868", True,  True),   # muted sage green
    ]

    # Only include conditions whose result files exist
    def _has_results(prefix_template):
        return os.path.exists(
            os.path.join(RESULTS_DIR, f"{prefix_template.format(120)}metrics.txt")
        )

    prefix_templates = {
        "Baseline":        "baseline_{}pct_",
        "NR (non-stat.)":  "enhanced_{}pct_",
        "NR (stationary)": "stationary_{}pct_",
    }

    conditions = []
    for name, color, use_dn, stat in COND_SPEC:
        tmpl = prefix_templates[name]
        if not _has_results(tmpl):
            print(f"  Skipping {name} — no result files found")
            continue
        print(f"  Computing FRR for: {name}")
        frr_vals = []
        for ratio in AMPLITUDE_RATIOS:
            noisy_map = precompute_noisy(trials, ratio)
            labs, scrs = run_trials(trials, noisy_map,
                                    use_denoise=use_dn, stationary=stat)
            _, frr = compute_far_frr(labs, scrs, [EXP1_THRESHOLD])
            frr_vals.append(round(frr[0] * 100, 2))
            print(f"    {int(ratio*100)}%: FRR = {frr_vals[-1]:.1f}%")
        conditions.append((name, color, frr_vals))

    ylabel = f"FRR @ threshold {EXP1_THRESHOLD} (%)"
    plot_noise_progression(
        ratios_pct, conditions,
        os.path.join(RESULTS_DIR, "noise_frr_progression.png"),
        ylabel=ylabel,
        title="FRR vs Noise Level — Baseline vs Denoising",
    )
    plot_noise_grouped_bars(
        ratios_pct, conditions,
        os.path.join(RESULTS_DIR, "noise_frr_grouped_bars.png"),
        ylabel=ylabel,
        title="FRR Comparison: Baseline vs Denoising Methods Across Noise Levels",
    )

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
