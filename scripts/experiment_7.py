import os
import sys
import random
import numpy as np
import torch
import soundfile as sf
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verification import voice_database, model, mel_extractor
from metrics import print_metrics, save_metrics, save_all_plots, plot_far_frr_by_group

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.normpath(os.path.join(BASE_DIR, "../data/test_genuine"))
RESULTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "../results/experiment_7"))

GENUINE_FILES = [f"test_{i}.wav" for i in range(1, 6)]
IMPOSTOR_FILES = [f"test_{i}.wav" for i in range(6, 11)]
T60_VALUES = [0.2, 0.5, 1.0, 2.0, 5.0, 8.0]
SAMPLE_RATE = 16000
N_SAMPLES = 100
RANDOM_SEED = 42
EXP1_THRESHOLD = 0.7301


def generate_rir(t60, sample_rate, rng):
    duration = t60 * 2
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    decay = np.exp(-6.91 * t / t60)
    rir = rng.standard_normal(n) * decay
    rir /= np.max(np.abs(rir))
    return rir.astype(np.float32)


def apply_reverb(signal, rir):
    reverbed = fftconvolve(signal, rir, mode="full")[: len(signal)]
    peak = np.max(np.abs(reverbed))
    if peak > 0:
        reverbed = reverbed / peak * np.max(np.abs(signal))
    return reverbed.astype(np.float32)


def extract_embedding(signal):
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    log_mel = torch.log(mel_extractor(signal_tensor) + 1e-6).transpose(1, 2)
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
        for fname in IMPOSTOR_FILES:
            path = os.path.join(TEST_DIR, speaker, fname)
            if os.path.exists(path):
                all_trials.append((rng.choice(candidates), path, 0))
    rng.shuffle(all_trials)
    return all_trials[:N_SAMPLES]


def run_trials(trials, t60, rng):
    rir = generate_rir(t60, SAMPLE_RATE, rng)
    labels, scores = [], []
    for claimed_id, audio_path, label in trials:
        try:
            signal, _ = sf.read(audio_path)
            reverbed = apply_reverb(signal.astype(np.float32), rir)
            test_emb = extract_embedding(reverbed)
            ref = torch.tensor(voice_database[claimed_id])
            scores.append(cosine_similarity(test_emb, ref))
            labels.append(label)
        except Exception:
            continue
    return np.array(labels), np.array(scores)


def save_reverb_samples(t60_values, rng):
    samples_dir = os.path.join(RESULTS_DIR, "audio_samples")
    os.makedirs(samples_dir, exist_ok=True)
    for speaker in ["id000ala", "id000kuba"]:
        src = os.path.join(TEST_DIR, speaker, "test_1.wav")
        if not os.path.exists(src):
            continue
        signal, _ = sf.read(src)
        signal = signal.astype(np.float32)
        sf.write(os.path.join(samples_dir, f"{speaker}_original.wav"), signal, SAMPLE_RATE)
        for t60 in t60_values:
            rir = generate_rir(t60, SAMPLE_RATE, rng)
            reverbed = apply_reverb(signal, rir)
            fname = f"{speaker}_t60_{str(t60).replace('.', '')}s.wav"
            sf.write(os.path.join(samples_dir, fname), reverbed, SAMPLE_RATE)
    print(f"Audio samples saved to {samples_dir}/")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng_trials = random.Random(RANDOM_SEED)
    rng_rir = np.random.default_rng(RANDOM_SEED + 1)

    save_reverb_samples(T60_VALUES, np.random.default_rng(RANDOM_SEED + 2))

    speakers = sorted(
        s for s in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, s))
    )
    trials = build_trials(speakers, rng_trials)
    n_genuine = sum(l for _, _, l in trials)
    print(f"Selected {len(trials)} trials ({n_genuine} genuine / {len(trials) - n_genuine} impostor)")

    groups = []
    for t60 in T60_VALUES:
        print(f"\n--- T60 = {t60}s ---")
        labels, scores = run_trials(trials, t60, rng_rir)
        print_metrics(labels, scores)
        prefix = f"t60_{str(t60).replace('.', '')}s_"
        save_metrics(labels, scores, RESULTS_DIR, prefix=prefix)
        save_all_plots(labels, scores, RESULTS_DIR, prefix=prefix)
        groups.append((f"T60={t60}s", labels, scores))

    plot_far_frr_by_group(
        groups, EXP1_THRESHOLD, "Czas pogłosu T60",
        os.path.join(RESULTS_DIR, "far_frr_by_t60.png"),
    )

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
