import os
import random
import shutil
import kagglehub

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "../data/background_sounds"))
MAX_FILES = 200
RANDOM_SEED = 42

os.environ["KAGGLEHUB_CACHE"] = r"D:\thesis_alicja\biometria\kaggle_cache"

# Twoja oryginalna reszta kodu:
path = kagglehub.dataset_download("chrisfilo/urbansound8k")
print("Pobrano do:", path)

print("Downloading UrbanSound8K...")
path = kagglehub.dataset_download("chrisfilo/urbansound8k")
print(f"Downloaded to: {path}")

all_wavs = []
for root, dirs, files in os.walk(path):
    for fname in files:
        if fname.lower().endswith(".wav"):
            all_wavs.append(os.path.join(root, fname))

random.seed(RANDOM_SEED)
selected = random.sample(all_wavs, min(MAX_FILES, len(all_wavs)))

os.makedirs(OUTPUT_DIR, exist_ok=True)

for src in selected:
    dst = os.path.join(OUTPUT_DIR, os.path.basename(src))
    shutil.copy2(src, dst)

print(f"Copied {len(selected)} WAV files to {OUTPUT_DIR}")
