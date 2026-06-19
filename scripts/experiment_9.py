"""
Experiment 9: Cross-language verification (P3 Challenge 2).

P3 specification: "Wypowiedzi do uwierzytelniania sa w innym jezyku niz probki
uzyte przy wdrazaniu systemu."

Enrollment: English (VoxCeleb — voice_profiles.npy).
Test genuine: Spanish (XTTS-synthesized via generate_cross_language.py).
Test impostor: English impostor files (test_6-10) claimed as enrolled speaker.

Section 1 — Baseline       : English-only profiles vs Spanish test samples.
Section 2 — Multi-enrollment: profiles averaged from English + Spanish enroll
                              samples (enroll_lang_1-3), same test samples.
                              Profiles saved to voice_profiles_multilang.npy.

Run generate_cross_language.py before this script.
"""
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
CROSS_LANG_DIR = os.path.normpath(os.path.join(BASE_DIR, "../data/test_cross_language"))
PROFILES_PATH = os.path.normpath(os.path.join(BASE_DIR, "../voice_profiles.npy"))
MULTILANG_PROFILES_PATH = os.path.normpath(os.path.join(BASE_DIR, "../voice_profiles_multilang.npy"))
RESULTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "../results/experiment_9"))

GENUINE_FILES = [f"test_{i}.wav" for i in range(1, 6)]
IMPOSTOR_FILES = [f"test_{i}.wav" for i in range(6, 11)]
TEST_LANG_FILES = [f"test_lang_{i}.wav" for i in range(1, 6)]
ENROLL_LANG_FILES = [f"enroll_lang_{i}.wav" for i in range(1, 4)]
RANDOM_SEED = 42
EXP1_THRESHOLD = 0.7301


def extract_embedding(signal):
    t = torch.tensor(signal.astype(np.float32), dtype=torch.float32).unsqueeze(0)
    log_mel = torch.log(mel_extractor(t) + 1e-6).transpose(1, 2)
    with torch.no_grad():
        return model(log_mel).squeeze()


def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def build_trials(speakers, rng, profiles):
    """Cross-lang genuine + English impostor trials for given profile set."""
    enrolled = set(profiles.keys())
    all_trials = []
    for speaker in speakers:
        if speaker not in enrolled:
            continue
        # Genuine: Spanish XTTS test utterances
        for fname in TEST_LANG_FILES:
            path = os.path.join(CROSS_LANG_DIR, speaker, fname)
            if os.path.exists(path):
                all_trials.append((speaker, path, 1))
        # Impostor: English audio claimed as a different enrolled speaker
        candidates = [s for s in speakers if s != speaker and s in enrolled]
        if not candidates:
            continue
        for fname in IMPOSTOR_FILES:
            path = os.path.join(TEST_DIR, speaker, fname)
            if os.path.exists(path):
                all_trials.append((rng.choice(candidates), path, 0))
    rng.shuffle(all_trials)
    return all_trials


def build_english_trials(speakers, rng, profiles):
    """Standard English genuine + impostor trials (same as Experiment 1)."""
    enrolled = set(profiles.keys())
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


def run_trials(trials, profiles):
    labels, scores = [], []
    for claimed_id, audio_path, label in trials:
        if claimed_id not in profiles:
            continue
        try:
            signal, _ = sf.read(audio_path)
            emb = extract_embedding(signal)
            ref = torch.tensor(profiles[claimed_id])
            scores.append(cosine_similarity(emb, ref))
            labels.append(label)
        except Exception:
            continue
    return np.array(labels), np.array(scores)


def build_multilang_profiles():
    """
    Average English profiles with Spanish enrollment embeddings.
    Saved to voice_profiles_multilang.npy for future reference.
    """
    if os.path.exists(MULTILANG_PROFILES_PATH):
        print(f"Loading cached multilang profiles from {MULTILANG_PROFILES_PATH}")
        return np.load(MULTILANG_PROFILES_PATH, allow_pickle=True).item()

    print("Building multilang profiles (English + Spanish enroll embeddings)...")
    multilang = {}
    for speaker_id, eng_profile in voice_database.items():
        lang_embs = []
        for fname in ENROLL_LANG_FILES:
            path = os.path.join(CROSS_LANG_DIR, speaker_id, fname)
            if not os.path.exists(path):
                continue
            try:
                signal, _ = sf.read(path)
                lang_embs.append(extract_embedding(signal).numpy())
            except Exception:
                continue

        if lang_embs:
            # Equal weight: average of (English profile + all Spanish enroll embeddings)
            all_vecs = [eng_profile] + lang_embs
            multilang[speaker_id] = np.mean(all_vecs, axis=0).astype(np.float32)
        else:
            multilang[speaker_id] = eng_profile  # fallback: English only

    np.save(MULTILANG_PROFILES_PATH, multilang)
    updated = sum(1 for s in multilang if not np.array_equal(multilang[s], voice_database[s]))
    print(f"Multilang profiles built: {updated}/{len(multilang)} speakers updated with Spanish data")
    return multilang


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.isdir(CROSS_LANG_DIR):
        print(f"ERROR: {CROSS_LANG_DIR} not found.")
        print("Run:  python scripts/generate_cross_language.py")
        sys.exit(1)

    speakers = sorted(
        s for s in os.listdir(CROSS_LANG_DIR)
        if os.path.isdir(os.path.join(CROSS_LANG_DIR, s))
    )
    if not speakers:
        print("No cross-language data found. Run generate_cross_language.py first.")
        sys.exit(1)
    print(f"Cross-language data available for {len(speakers)} speakers")

    rng = random.Random(RANDOM_SEED)

    # ---------- Section 1: Baseline (English-only profiles) ----------
    print("\n=== Section 1: Baseline — English-only enrollment ===")
    trials = build_trials(speakers, rng, voice_database)
    n_genuine = sum(l for _, _, l in trials)
    print(f"Trials: {len(trials)} ({n_genuine} genuine cross-lang / {len(trials)-n_genuine} impostor)")
    labels_base, scores_base = run_trials(trials, voice_database)
    print_metrics(labels_base, scores_base)
    save_metrics(labels_base, scores_base, RESULTS_DIR, prefix="baseline_")
    save_all_plots(labels_base, scores_base, RESULTS_DIR, prefix="baseline_")

    # ---------- Section 2: Multi-enrollment (English + Spanish profiles) ----------
    print("\n=== Section 2: Multi-enrollment — English + Spanish profiles ===")
    multilang_profiles = build_multilang_profiles()
    rng2 = random.Random(RANDOM_SEED)
    trials_ml = build_trials(speakers, rng2, multilang_profiles)
    labels_ml, scores_ml = run_trials(trials_ml, multilang_profiles)
    print_metrics(labels_ml, scores_ml)
    save_metrics(labels_ml, scores_ml, RESULTS_DIR, prefix="multilang_")
    save_all_plots(labels_ml, scores_ml, RESULTS_DIR, prefix="multilang_")

    # ---------- Section 3: Multilang profiles vs English-only test ----------
    print("\n=== Section 3: Multilang profiles — English-only test (sanity check) ===")
    eng_speakers = sorted(
        s for s in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, s))
    )
    rng3 = random.Random(RANDOM_SEED)
    trials_eng = build_english_trials(eng_speakers, rng3, multilang_profiles)
    n_genuine3 = sum(l for _, _, l in trials_eng)
    print(f"Trials: {len(trials_eng)} ({n_genuine3} genuine / {len(trials_eng)-n_genuine3} impostor)")
    labels_eng, scores_eng = run_trials(trials_eng, multilang_profiles)
    print_metrics(labels_eng, scores_eng)
    save_metrics(labels_eng, scores_eng, RESULTS_DIR, prefix="multilang_eng_test_")
    save_all_plots(labels_eng, scores_eng, RESULTS_DIR, prefix="multilang_eng_test_")

    # ---------- Comparison ----------
    plot_far_frr_by_group(
        [("English only", labels_base, scores_base),
         ("Multi-enrollment (ES test)", labels_ml, scores_ml),
         ("Multi-enrollment (EN test)", labels_eng, scores_eng)],
        EXP1_THRESHOLD, "Enrollment strategy",
        os.path.join(RESULTS_DIR, "comparison_enrollment_strategy.png"),
    )

    print(f"\nAll results saved to {RESULTS_DIR}/")
    print(f"Multilang profiles saved to {MULTILANG_PROFILES_PATH}")


if __name__ == "__main__":
    main()
