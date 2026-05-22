import os
import sys
import random
import numpy as np
import torch
import torchaudio
import soundfile as sf
import scipy.signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verification import voice_database, model
from metrics import print_metrics, save_metrics, save_all_plots, plot_far_frr_by_group

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.normpath(os.path.join(BASE_DIR, "../data/test_genuine"))
RESULTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "../results/experiment_3"))

GENUINE_FILES = [f"test_{i}.wav" for i in range(1, 6)]
IMPOSTOR_FILES = [f"test_{i}.wav" for i in range(6, 11)]
DOWNSAMPLE_FACTORS = [2, 5, 10]
ORIGINAL_SR = 16000
N_SAMPLES = 200
RANDOM_SEED = 42
EXP1_THRESHOLD = 0.7301


def build_mel_extractor(sample_rate):
    win_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)
    n_fft = max(win_length, 2 * 80)
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=80,
    )


def extract_embedding(signal, mel_extractor):
    signal_tensor = torch.tensor(signal.astype(np.float32)).unsqueeze(0)
    log_mel = torch.log(mel_extractor(signal_tensor) + 1e-6).transpose(1, 2)
    with torch.no_grad():
        return model(log_mel).squeeze()


def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def naive_downsample(signal, factor):
    return signal[::factor]


def proper_downsample(signal, factor):
    return scipy.signal.resample_poly(signal, 1, factor).astype(np.float32)


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
        for fname in IMPOSTOR_FILES:
            path = os.path.join(TEST_DIR, speaker, fname)
            if os.path.exists(path):
                all_trials.append((rng.choice(candidates), path, 0))
    rng.shuffle(all_trials)
    return all_trials[:N_SAMPLES]


def run_trials(trials, downsample_fn, factor):
    new_sr = ORIGINAL_SR // factor
    mel_extractor = build_mel_extractor(new_sr)
    labels, scores = [], []
    for claimed_id, audio_path, label in trials:
        try:
            signal, _ = sf.read(audio_path)
            test_emb = extract_embedding(downsample_fn(signal, factor), mel_extractor)
            ref = torch.tensor(voice_database[claimed_id])
            scores.append(cosine_similarity(test_emb, ref))
            labels.append(label)
        except Exception:
            continue
    return np.array(labels), np.array(scores)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    speakers = sorted(
        s for s in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, s))
    )
    trials = build_trials(speakers, rng)
    n_genuine = sum(l for _, _, l in trials)
    print(f"Selected {len(trials)} trials ({n_genuine} genuine / {len(trials) - n_genuine} impostor)")

    naive_groups = []
    proper_groups = []

    for factor in DOWNSAMPLE_FACTORS:
        new_sr = ORIGINAL_SR // factor
        label_name = f"{new_sr} Hz"

        print(f"\n--- Naive x{factor}  ({ORIGINAL_SR} → {new_sr} Hz) ---")
        labels, scores = run_trials(trials, naive_downsample, factor)
        print_metrics(labels, scores)
        save_metrics(labels, scores, RESULTS_DIR, prefix=f"naive_x{factor}_")
        save_all_plots(labels, scores, RESULTS_DIR, prefix=f"naive_x{factor}_")
        naive_groups.append((label_name, labels, scores))

        print(f"\n--- Proper x{factor}  ({ORIGINAL_SR} → {new_sr} Hz) ---")
        labels, scores = run_trials(trials, proper_downsample, factor)
        print_metrics(labels, scores)
        save_metrics(labels, scores, RESULTS_DIR, prefix=f"proper_x{factor}_")
        save_all_plots(labels, scores, RESULTS_DIR, prefix=f"proper_x{factor}_")
        proper_groups.append((label_name, labels, scores))

    plot_far_frr_by_group(
        naive_groups, EXP1_THRESHOLD, "Sample Rate (Hz) — Naive Downsampling",
        os.path.join(RESULTS_DIR, "naive_far_frr_by_sr.png"),
    )
    plot_far_frr_by_group(
        proper_groups, EXP1_THRESHOLD, "Sample Rate (Hz) — Proper Downsampling",
        os.path.join(RESULTS_DIR, "proper_far_frr_by_sr.png"),
    )

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
