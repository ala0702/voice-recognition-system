import os
import sys
import random
import numpy as np
import torch
import soundfile as sf
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verification import voice_database, model, mel_extractor
from metrics import print_metrics, save_metrics, save_all_plots, plot_far_frr_by_group

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.normpath(os.path.join(BASE_DIR, "../data/test_genuine"))
BACKGROUND_DIR = os.path.normpath(os.path.join(BASE_DIR, "../data/background_sounds"))
RESULTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "../results/experiment_5"))

GENUINE_FILES = [f"test_{i}.wav" for i in range(1, 6)]
IMPOSTOR_FILES = [f"test_{i}.wav" for i in range(6, 11)]
SNR_LEVELS_DB = [20, 10, 0]
ORIGINAL_SR = 16000
N_SAMPLES = 100
RANDOM_SEED = 42
EXP1_THRESHOLD = 0.7301


def load_background(path, target_len):
    signal, sr = sf.read(path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sr != ORIGINAL_SR:
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        signal = torchaudio.functional.resample(signal_tensor, sr, ORIGINAL_SR).squeeze().numpy()
    if len(signal) < target_len:
        repeats = int(np.ceil(target_len / len(signal)))
        signal = np.tile(signal, repeats)
    return signal[:target_len].astype(np.float32)


def mix_at_snr(speech, background, snr_db):
    speech_power = np.mean(speech ** 2)
    bg_power = np.mean(background ** 2)
    if bg_power == 0:
        return speech
    scale = np.sqrt(speech_power / (bg_power * 10 ** (snr_db / 10)))
    return (speech + scale * background).astype(np.float32)


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


def run_trials(trials, snr_db, background_files, rng):
    labels, scores = [], []
    for claimed_id, audio_path, label in trials:
        try:
            speech, _ = sf.read(audio_path)
            speech = speech.astype(np.float32)
            bg_path = rng.choice(background_files)
            background = load_background(bg_path, len(speech))
            mixed = mix_at_snr(speech, background, snr_db)
            test_emb = extract_embedding(mixed)
            ref = torch.tensor(voice_database[claimed_id])
            scores.append(cosine_similarity(test_emb, ref))
            labels.append(label)
        except Exception:
            continue
    return np.array(labels), np.array(scores)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    background_files = [
        os.path.join(BACKGROUND_DIR, f)
        for f in os.listdir(BACKGROUND_DIR)
        if f.lower().endswith(".wav")
    ]
    print(f"Found {len(background_files)} background sound files")

    speakers = sorted(
        s for s in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, s))
    )
    trials = build_trials(speakers, rng)
    n_genuine = sum(l for _, _, l in trials)
    print(f"Selected {len(trials)} trials ({n_genuine} genuine / {len(trials) - n_genuine} impostor)")

    groups = []
    for snr_db in SNR_LEVELS_DB:
        print(f"\n--- SNR = {snr_db} dB ---")
        labels, scores = run_trials(trials, snr_db, background_files, rng)
        print_metrics(labels, scores)
        save_metrics(labels, scores, RESULTS_DIR, prefix=f"snr_{snr_db}db_")
        save_all_plots(labels, scores, RESULTS_DIR, prefix=f"snr_{snr_db}db_")
        groups.append((f"{snr_db} dB", labels, scores))

    plot_far_frr_by_group(
        groups, EXP1_THRESHOLD, "SNR (dB)",
        os.path.join(RESULTS_DIR, "far_frr_by_snr.png"),
    )

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
