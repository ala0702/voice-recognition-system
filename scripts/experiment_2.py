import os
import sys
import random
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verification import voice_database, model, mel_extractor
from metrics import print_metrics, save_metrics, save_all_plots, plot_far_frr_by_group

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.normpath(os.path.join(BASE_DIR, "../data/test_genuine"))
RESULTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "../results/experiment_2"))

GENUINE_FILES = [f"test_{i}.wav" for i in range(1, 6)]
IMPOSTOR_FILES = [f"test_{i}.wav" for i in range(6, 11)]
AMPLITUDE_FACTORS = [25, 1, 0.04]
RANDOM_SEED = 42
EXP1_THRESHOLD = 0.7301


def build_trials(speakers, rng):
    enrolled = set(voice_database.keys())
    trials = []
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


def extract_embedding(signal):
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    log_mel = torch.log(mel_extractor(signal_tensor) + 1e-6).transpose(1, 2)
    with torch.no_grad():
        return model(log_mel).squeeze()


def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = random.Random(RANDOM_SEED)
    amp_rng = random.Random(RANDOM_SEED + 1)

    speakers = sorted(
        s for s in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, s))
    )
    trials = build_trials(speakers, rng)
    print(f"Running {len(trials)} trials with random amplitude scaling from {AMPLITUDE_FACTORS}")

    labels, scores, factors = [], [], []
    for i, (claimed_id, audio_path, label) in enumerate(trials):
        factor = amp_rng.choice(AMPLITUDE_FACTORS)
        try:
            signal, _ = sf.read(audio_path)
            test_emb = extract_embedding(signal * factor)
            ref = torch.tensor(voice_database[claimed_id])
            scores.append(cosine_similarity(test_emb, ref))
            labels.append(label)
            factors.append(factor)
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(trials)}")

    labels = np.array(labels)
    scores = np.array(scores)
    factors = np.array(factors)

    print("\n=== Experiment 2 — Amplitude Scaling (Overall) ===")
    print_metrics(labels, scores)
    save_metrics(labels, scores, RESULTS_DIR)
    save_all_plots(labels, scores, RESULTS_DIR)

    print("\n=== Per Amplitude Factor ===")
    for factor in AMPLITUDE_FACTORS:
        mask = factors == factor
        if mask.sum() < 2:
            continue
        print(f"\n-- Factor x{factor} ({mask.sum()} trials) --")
        print_metrics(labels[mask], scores[mask])
        save_metrics(labels[mask], scores[mask], RESULTS_DIR, prefix=f"factor_{factor}_")
        save_all_plots(labels[mask], scores[mask], RESULTS_DIR, prefix=f"factor_{factor}_")

    groups = [
        (f"x{factor}", labels[factors == factor], scores[factors == factor])
        for factor in AMPLITUDE_FACTORS
        if (factors == factor).sum() >= 2
    ]
    plot_far_frr_by_group(
        groups, EXP1_THRESHOLD, "Amplitude Factor",
        os.path.join(RESULTS_DIR, "far_frr_by_factor.png")
    )

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
