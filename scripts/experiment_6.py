import os
import sys
import random
import subprocess
import tempfile
import numpy as np
import torch
import soundfile as sf
import imageio_ffmpeg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verification import voice_database, model, mel_extractor
from metrics import print_metrics, save_metrics, save_all_plots, plot_far_frr_by_group

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.normpath(os.path.join(BASE_DIR, "../data/test_genuine"))
RESULTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "../results/experiment_6"))

GENUINE_FILES = [f"test_{i}.wav" for i in range(1, 6)]
IMPOSTOR_FILES = [f"test_{i}.wav" for i in range(6, 11)]

CODECS = {
    "mp3":  {"ext": "mp3",  "ffmpeg_args": ["-c:a", "libmp3lame"],  "bitrates": [8, 32, 128]},
    "aac":  {"ext": "aac",  "ffmpeg_args": ["-c:a", "aac"],         "bitrates": [8, 32, 128]},
    "opus": {"ext": "opus", "ffmpeg_args": ["-c:a", "libopus"],     "bitrates": [6, 16, 64]},
}

N_SAMPLES = 100
RANDOM_SEED = 42
EXP1_THRESHOLD = 0.7301
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def compress_decompress(signal, codec_ext, ffmpeg_args, bitrate_kbps):
    with tempfile.TemporaryDirectory() as tmp:
        src_wav = os.path.join(tmp, "input.wav")
        compressed = os.path.join(tmp, f"output.{codec_ext}")
        decoded_wav = os.path.join(tmp, "decoded.wav")

        sf.write(src_wav, signal, 16000)

        subprocess.run(
            [FFMPEG, "-y", "-i", src_wav] + ffmpeg_args +
            ["-b:a", f"{bitrate_kbps}k", compressed],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        subprocess.run(
            [FFMPEG, "-y", "-i", compressed, "-ar", "16000", "-ac", "1", decoded_wav],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )

        decoded, _ = sf.read(decoded_wav)
        return decoded.astype(np.float32)


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


def run_trials(trials, codec_ext, ffmpeg_args, bitrate_kbps):
    labels, scores = [], []
    for claimed_id, audio_path, label in trials:
        try:
            signal, _ = sf.read(audio_path)
            decoded = compress_decompress(signal.astype(np.float32), codec_ext, ffmpeg_args, bitrate_kbps)
            test_emb = extract_embedding(decoded)
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

    for codec_name, cfg in CODECS.items():
        print(f"\n{'='*40}")
        print(f"Codec: {codec_name.upper()}")
        groups = []
        for bitrate in cfg["bitrates"]:
            print(f"\n--- {codec_name.upper()} @ {bitrate} kbps ---")
            labels, scores = run_trials(trials, cfg["ext"], cfg["ffmpeg_args"], bitrate)
            print_metrics(labels, scores)
            prefix = f"{codec_name}_{bitrate}kbps_"
            save_metrics(labels, scores, RESULTS_DIR, prefix=prefix)
            save_all_plots(labels, scores, RESULTS_DIR, prefix=prefix)
            groups.append((f"{bitrate} kbps", labels, scores))

        plot_far_frr_by_group(
            groups, EXP1_THRESHOLD, f"Bitrate — {codec_name.upper()}",
            os.path.join(RESULTS_DIR, f"{codec_name}_far_frr_by_bitrate.png"),
        )

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
