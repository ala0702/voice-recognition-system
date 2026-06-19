"""
Experiment 10: Multi-speaker diarization (P3 Challenge 3).

P3 specification: "Zamiast pojedynczego mowcy mamy do czynienia z dialogami
lub rozmowa grupy osob."

Dialogue construction: [Speaker_A | Intruder_B | Speaker_A]  (A-B-A pattern).
Genuine trial : A is the claimed speaker, A IS present in the dialogue.
Impostor trial: X is the audio source (impostor), Y is the claimed speaker,
                neither X nor Y is the other speaker in the dialogue.

Section 1 — No diarization   : verify on raw mixed dialogue (shows problem).
Section 2 — K-means diarization: split into 2-s segments, embed with ECAPA-TDNN,
                                  k-means (k=2), pick cluster closest to claimed
                                  speaker's profile, concatenate and verify.
Section 3 — pyannote.audio   : professional diarization pipeline (optional).
                                Requires HF_TOKEN env var and:
                                    pip install pyannote.audio
                                Access request: hf.co/pyannote/speaker-diarization-3.1
"""


import inspect as _inspect
_orig_getmodule = _inspect.getmodule
def _safe_getmodule(*args, **kwargs):
    try:
        return _orig_getmodule(*args, **kwargs)
    except Exception:
        return None
_inspect.getmodule = _safe_getmodule


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
RESULTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "../results/experiment_10"))

GENUINE_FILES = [f"test_{i}.wav" for i in range(1, 6)]
IMPOSTOR_FILES = [f"test_{i}.wav" for i in range(6, 11)]
SAMPLE_RATE = 16000
SEGMENT_DURATION = 2.0   # seconds per segment for k-means diarization
MIN_SEGMENT_LEN = 3200   # 0.2 s — skip segments shorter than this after diarization
RANDOM_SEED = 42
EXP1_THRESHOLD = 0.7301


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_signal(path):
    signal, sr = sf.read(path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return signal.astype(np.float32)


def build_dialogue(seg_a1, seg_b, seg_a2):
    """Concatenate A-B-A segments into a single dialogue signal."""
    return np.concatenate([seg_a1, seg_b, seg_a2])


def extract_embedding(signal):
    t = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    log_mel = torch.log(mel_extractor(t) + 1e-6).transpose(1, 2)
    with torch.no_grad():
        return model(log_mel).squeeze()


def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ---------------------------------------------------------------------------
# Trial building
# ---------------------------------------------------------------------------

def build_trials(speakers, rng):
    """
    Returns list of:
      (claimed_id, main_wav, intruder_wav, label)
    where main_wav is the audio of the genuine/impostor speaker.
    """
    enrolled = set(voice_database.keys())
    genuine_trials, impostor_trials = [], []

    for speaker in speakers:
        if speaker not in enrolled:
            continue
        others = [s for s in speakers if s != speaker and s in enrolled]
        if not others:
            continue

        for fname in GENUINE_FILES:
            main_path = os.path.join(TEST_DIR, speaker, fname)
            intruder = rng.choice(others)
            intruder_path = os.path.join(TEST_DIR, intruder,
                                         rng.choice(GENUINE_FILES))
            if os.path.exists(main_path) and os.path.exists(intruder_path):
                genuine_trials.append((speaker, main_path, intruder_path, 1))

        for fname in IMPOSTOR_FILES:
            impost_path = os.path.join(TEST_DIR, speaker, fname)
            # intruder = third speaker (neither impostor nor claimed)
            intruder = rng.choice(others)
            intruder_path = os.path.join(TEST_DIR, intruder,
                                         rng.choice(GENUINE_FILES))
            # claimed = enrolled speaker who is NOT present in the dialogue
            other_claimed = [s for s in others if s != intruder]
            if not other_claimed:
                continue
            claimed = rng.choice(other_claimed)
            if os.path.exists(impost_path) and os.path.exists(intruder_path):
                impostor_trials.append((claimed, impost_path, intruder_path, 0))

    all_trials = genuine_trials + impostor_trials
    rng.shuffle(all_trials)
    return all_trials


# ---------------------------------------------------------------------------
# Section 1: no diarization
# ---------------------------------------------------------------------------

def run_no_diarization(trials):
    labels, scores = [], []
    for claimed_id, main_wav, intruder_wav, label in trials:
        try:
            sig_main = load_signal(main_wav)
            sig_int = load_signal(intruder_wav)
            # Split each into two halves to form the A-B-A pattern
            half_main = len(sig_main) // 2
            dialogue = build_dialogue(sig_main[:half_main], sig_int,
                                      sig_main[half_main:])
            emb = extract_embedding(dialogue)
            ref = torch.tensor(voice_database[claimed_id])
            scores.append(cosine_similarity(emb, ref))
            labels.append(label)
        except Exception:
            continue
    return np.array(labels), np.array(scores)


# ---------------------------------------------------------------------------
# Section 2: k-means embedding diarization
# ---------------------------------------------------------------------------

def kmeans_diarize(dialogue, claimed_ref_emb, n_speakers=2):
    """
    Split dialogue into fixed-length segments, embed each with ECAPA-TDNN,
    cluster with k-means (k=n_speakers), return segments from the cluster
    whose centroid is most similar to the claimed speaker's profile.
    """
    from sklearn.cluster import KMeans

    seg_len = int(SEGMENT_DURATION * SAMPLE_RATE)
    n_segs = len(dialogue) // seg_len
    if n_segs < n_speakers + 1:
        return None

    embeddings = []
    for i in range(n_segs):
        seg = dialogue[i * seg_len:(i + 1) * seg_len]
        emb = extract_embedding(seg)
        embeddings.append(emb.numpy())
    embeddings = np.array(embeddings)

    km = KMeans(n_clusters=n_speakers, n_init=10, random_state=42)
    cluster_labels = km.fit_predict(embeddings)

    ref_np = claimed_ref_emb.numpy()
    best_cluster, best_sim = -1, -np.inf
    for c in range(n_speakers):
        mask = cluster_labels == c
        if not mask.any():
            continue
        centroid = embeddings[mask].mean(axis=0)
        sim = np.dot(centroid, ref_np) / (
            np.linalg.norm(centroid) * np.linalg.norm(ref_np) + 1e-8
        )
        if sim > best_sim:
            best_sim, best_cluster = sim, c

    segs = [
        dialogue[i * seg_len:(i + 1) * seg_len]
        for i in range(n_segs) if cluster_labels[i] == best_cluster
    ]
    return np.concatenate(segs) if segs else None


def run_kmeans_diarization(trials):
    labels, scores = [], []
    for claimed_id, main_wav, intruder_wav, label in trials:
        try:
            sig_main = load_signal(main_wav)
            sig_int = load_signal(intruder_wav)
            half_main = len(sig_main) // 2
            dialogue = build_dialogue(sig_main[:half_main], sig_int,
                                      sig_main[half_main:])
            ref = torch.tensor(voice_database[claimed_id])
            extracted = kmeans_diarize(dialogue, ref)
            if extracted is None or len(extracted) < MIN_SEGMENT_LEN:
                continue
            emb = extract_embedding(extracted)
            scores.append(cosine_similarity(emb, ref))
            labels.append(label)
        except Exception:
            continue
    return np.array(labels), np.array(scores)


# ---------------------------------------------------------------------------
# Section 3: pyannote.audio diarization (optional)
# ---------------------------------------------------------------------------
def pyannote_diarize(pipeline, dialogue, claimed_ref_emb):
    """
    Run pyannote speaker diarization, pick the speaker label whose average
    embedding is closest to the claimed profile, return their audio.
    """
    dialogue_tensor = torch.tensor(dialogue).unsqueeze(0)
    diarization = pipeline({"waveform": dialogue_tensor,
                            "sample_rate": SAMPLE_RATE})

    speaker_segments = {}
    for turn, _, spk in diarization.speaker_diarization.itertracks(yield_label=True):
        start = int(turn.start * SAMPLE_RATE)
        end = int(turn.end * SAMPLE_RATE)
        if end - start < MIN_SEGMENT_LEN:
            continue
        speaker_segments.setdefault(spk, []).append(dialogue[start:end])

    if not speaker_segments:
        return None

    ref_np = claimed_ref_emb.numpy()
    best_spk, best_sim = None, -np.inf
    for spk, segs in speaker_segments.items():
        concat = np.concatenate(segs)
        emb = extract_embedding(concat).numpy()
        sim = np.dot(emb, ref_np) / (np.linalg.norm(emb) * np.linalg.norm(ref_np) + 1e-8)
        if sim > best_sim:
            best_sim, best_spk = sim, spk

    return np.concatenate(speaker_segments[best_spk]) if best_spk else None


def run_pyannote_diarization(trials, pipeline):
    labels, scores = [], []
    for claimed_id, main_wav, intruder_wav, label in trials:
        try:
            sig_main = load_signal(main_wav)
            sig_int = load_signal(intruder_wav)
            half_main = len(sig_main) // 2
            dialogue = build_dialogue(sig_main[:half_main], sig_int,
                                      sig_main[half_main:])
            ref = torch.tensor(voice_database[claimed_id])
            extracted = pyannote_diarize(pipeline, dialogue, ref)
            if extracted is None or len(extracted) < MIN_SEGMENT_LEN:
                continue
            emb = extract_embedding(extracted)
            scores.append(cosine_similarity(emb, ref))
            labels.append(label)
        except Exception as e:
            print(f"  [WARN] {e}")
            continue
    return np.array(labels), np.array(scores)
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    enrolled = set(voice_database.keys())
    speakers = sorted(
        s for s in os.listdir(TEST_DIR)
        if os.path.isdir(os.path.join(TEST_DIR, s)) and s in enrolled
    )
    trials = build_trials(speakers, rng)
    n_genuine = sum(l for *_, l in trials)
    print(f"Dialogue trials: {len(trials)} ({n_genuine} genuine / {len(trials)-n_genuine} impostor)")

    comparison_groups = []

    # ---------- Section 1: no diarization ----------
    if os.path.exists(os.path.join(RESULTS_DIR, "no_diarization_metrics.txt")):
        print("\n=== Section 1: SKIPPED (results already exist) ===")
        labels1, scores1 = None, None
    else:
        print("\n=== Section 1: No diarization (raw mixed dialogue) ===")
        labels1, scores1 = run_no_diarization(trials)
        print_metrics(labels1, scores1)
        save_metrics(labels1, scores1, RESULTS_DIR, prefix="no_diarization_")
        save_all_plots(labels1, scores1, RESULTS_DIR, prefix="no_diarization_")
    if labels1 is not None:
        comparison_groups.append(("No diarization", labels1, scores1))

    # ---------- Section 2: k-means diarization ----------
    if os.path.exists(os.path.join(RESULTS_DIR, "kmeans_diarization_metrics.txt")):
        print("\n=== Section 2: SKIPPED (results already exist) ===")
        labels2, scores2 = None, None
    else:
        print("\n=== Section 2: K-means embedding diarization ===")
        try:
            from sklearn.cluster import KMeans  # noqa: F401
        except ImportError:
            print("WARNING: scikit-learn not installed — skipping Section 2.")
            print("Install with:  pip install scikit-learn")
            labels2, scores2 = None, None
        else:
            labels2, scores2 = run_kmeans_diarization(trials)
            print_metrics(labels2, scores2)
            save_metrics(labels2, scores2, RESULTS_DIR, prefix="kmeans_diarization_")
            save_all_plots(labels2, scores2, RESULTS_DIR, prefix="kmeans_diarization_")
    if labels2 is not None:
        comparison_groups.append(("K-means", labels2, scores2))

    # ---------- Section 3: pyannote.audio ----------
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("\n=== Section 3: pyannote.audio — SKIPPED (HF_TOKEN not set) ===")
        print("To enable: set HF_TOKEN=<your_token> and pip install pyannote.audio")
        print("Request model access at: https://hf.co/pyannote/speaker-diarization-3.1")
    else:
        try:
            import sys as _sys
            from unittest.mock import MagicMock as _MM
            # stub missing optional deps that pyannote/speechbrain try to lazy-import
            for _mod in ["speechbrain.integrations.k2_fsa", "k2", "lhotse"]:
                if _mod not in _sys.modules:
                    _sys.modules[_mod] = _MM()
            # torch 2.12.0 double-registers _c10d_functional::wait_tensor — suppress silently
            import torch as _torch
            _orig_reg = _torch.library.register_autograd
            def _safe_register_autograd(*a, **kw):
                try:
                    return _orig_reg(*a, **kw)
                except RuntimeError:
                    pass
            _torch.library.register_autograd = _safe_register_autograd
            from pyannote.audio import Pipeline
            _torch.library.register_autograd = _orig_reg  # restore
        except Exception as _e:
            print(f"\n=== Section 3: pyannote.audio — SKIPPED (import failed: {_e}) ===")
            Pipeline = None
        if Pipeline is not None:
            print("\n=== Section 3: pyannote.audio diarization ===")
            print("Loading pyannote/speaker-diarization-3.1 ...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token,
            )
            labels3, scores3 = run_pyannote_diarization(trials, pipeline)
            print_metrics(labels3, scores3)
            save_metrics(labels3, scores3, RESULTS_DIR, prefix="pyannote_diarization_")
            save_all_plots(labels3, scores3, RESULTS_DIR, prefix="pyannote_diarization_")
            comparison_groups.append(("pyannote", labels3, scores3))

    # ---------- Comparison chart ----------
    if len(comparison_groups) > 1:
        plot_far_frr_by_group(
            comparison_groups, EXP1_THRESHOLD, "Diarization method",
            os.path.join(RESULTS_DIR, "comparison_diarization_methods.png"),
        )

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
