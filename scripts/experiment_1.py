import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verification import verify_speaker, voice_database
from metrics import print_metrics, save_metrics, save_all_plots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.normpath(os.path.join(BASE_DIR, "../data/test_genuine"))
RESULTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "../results/experiment_1"))

GENUINE_FILES = [f"test_{i}.wav" for i in range(1, 6)]
IMPOSTOR_FILES = [f"test_{i}.wav" for i in range(6, 11)]
RANDOM_SEED = 42
THRESHOLD = 0.25


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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    speakers = sorted(
        s for s in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, s))
    )
    trials = build_trials(speakers, rng)
    print(f"Running {len(trials)} trials ({sum(l for _, _, l in trials)} genuine / {sum(1-l for _, _, l in trials)} impostor)")

    labels, scores = [], []
    for i, (claimed_id, audio_path, label) in enumerate(trials):
        _, score = verify_speaker(claimed_id, audio_path, threshold=THRESHOLD)
        labels.append(label)
        scores.append(score)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(trials)}")

    print("\n=== Experiment 1 — Baseline Verification Metrics ===")
    print_metrics(labels, scores)
    save_metrics(labels, scores, RESULTS_DIR)
    save_all_plots(labels, scores, RESULTS_DIR)
    print(f"\nPlots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
