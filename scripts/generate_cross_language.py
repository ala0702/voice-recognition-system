"""
One-time data preparation for Experiment 9 (cross-language verification).

Uses XTTS v2 to clone each enrolled speaker's voice and synthesize Spanish
utterances. Two separate sets are generated to avoid data leakage:
  - enroll_lang_1..3  : used to UPDATE the speaker profile in experiment_9.py
  - test_lang_1..5    : used as test samples in experiment_9.py

Run once before experiment_9.py:
    python scripts/generate_cross_language.py

Requirements:
    pip install TTS

Output: data/test_cross_language/<speaker_id>/*.wav
"""
import os
import sys
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENROLL_DIR = os.path.normpath(os.path.join(BASE_DIR, "../data/enrollment"))
CROSS_LANG_DIR = os.path.normpath(os.path.join(BASE_DIR, "../data/test_cross_language"))

LANGUAGE = "es"  # Spanish
RANDOM_SEED = 42

# Kept separate so enrollment and test phrases never overlap
ENROLL_PHRASES = [
    "Buenos días, este es mi registro biométrico de voz.",
    "Mi voz es única y sirve como identificación personal.",
    "Estoy proporcionando una muestra de voz para el sistema.",
]

TEST_PHRASES = [
    "Hola, quiero verificar mi identidad en el sistema.",
    "Por favor, confirma que soy el usuario autorizado.",
    "La biometría de voz es un método seguro de autenticación.",
    "Mi huella vocal puede verificar quién soy en cualquier idioma.",
    "Estoy hablando en español para demostrar el reconocimiento de voz.",
]


def main():
    try:
        from TTS.api import TTS
    except ImportError:
        print("ERROR: TTS library not installed.")
        print("Install with:  pip install TTS")
        sys.exit(1)

    print("Loading XTTS v2 model (first run downloads ~1.8 GB)...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    os.makedirs(CROSS_LANG_DIR, exist_ok=True)

    speakers = sorted(
        s for s in os.listdir(ENROLL_DIR)
        if os.path.isdir(os.path.join(ENROLL_DIR, s))
    )
    print(f"Found {len(speakers)} enrolled speakers\n")

    all_needed = (
        [f"enroll_lang_{j}.wav" for j in range(1, len(ENROLL_PHRASES) + 1)] +
        [f"test_lang_{j}.wav" for j in range(1, len(TEST_PHRASES) + 1)]
    )

    for i, speaker_id in enumerate(speakers):
        speaker_out = os.path.join(CROSS_LANG_DIR, speaker_id)
        os.makedirs(speaker_out, exist_ok=True)

        existing = set(os.listdir(speaker_out))
        if all(f in existing for f in all_needed):
            print(f"[{i+1}/{len(speakers)}] {speaker_id}: already complete")
            continue

        enroll_wav_dir = os.path.join(ENROLL_DIR, speaker_id)
        wav_files = sorted(f for f in os.listdir(enroll_wav_dir) if f.endswith(".wav"))
        if not wav_files:
            print(f"[{i+1}/{len(speakers)}] {speaker_id}: no enrollment audio, skipping")
            continue
        reference_wav = os.path.join(enroll_wav_dir, wav_files[0])

        errors = 0
        for j, phrase in enumerate(ENROLL_PHRASES, start=1):
            out = os.path.join(speaker_out, f"enroll_lang_{j}.wav")
            if os.path.exists(out):
                continue
            try:
                tts.tts_to_file(text=phrase, speaker_wav=reference_wav,
                                language=LANGUAGE, file_path=out)
            except Exception as e:
                print(f"  [ERROR] enroll_lang_{j}: {e}")
                errors += 1

        for j, phrase in enumerate(TEST_PHRASES, start=1):
            out = os.path.join(speaker_out, f"test_lang_{j}.wav")
            if os.path.exists(out):
                continue
            try:
                tts.tts_to_file(text=phrase, speaker_wav=reference_wav,
                                language=LANGUAGE, file_path=out)
            except Exception as e:
                print(f"  [ERROR] test_lang_{j}: {e}")
                errors += 1

        status = "done" if errors == 0 else f"done with {errors} error(s)"
        print(f"[{i+1}/{len(speakers)}] {speaker_id}: {status}")

    print(f"\nCross-language data written to: {CROSS_LANG_DIR}")
    print("Next: python scripts/experiment_9.py")


if __name__ == "__main__":
    main()
