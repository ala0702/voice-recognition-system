import os
import soundfile as sf
import librosa
from datasets import load_dataset
import os
import imageio_ffmpeg
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())


TARGET_USERS = 100
ENROLL_SAMPLES = 5   # samples for setting profile
TEST_SAMPLES = 10    # for future tests (minimum 500, we will have 10 per user so 1000)
TOTAL_SAMPLES = ENROLL_SAMPLES + TEST_SAMPLES

ENROLL_DIR = "../data/enrollment"
TEST_DIR = "../data/test_genuine"

os.makedirs(ENROLL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


dataset = load_dataset("acul3/voxceleb2", split="train", streaming=True)

users_completed = 0
samples_per_user = {}

for item in dataset:
    speaker_id = item['speaker_id']
    
    if users_completed >= TARGET_USERS:
        break
        
    # initilize new user
    if speaker_id not in samples_per_user:
        if len(samples_per_user) >= TARGET_USERS:
            continue
            
        samples_per_user[speaker_id] = 0
        os.makedirs(f"{ENROLL_DIR}/{speaker_id}", exist_ok=True)
        os.makedirs(f"{TEST_DIR}/{speaker_id}", exist_ok=True)

    current_count = samples_per_user[speaker_id]
    if current_count < TOTAL_SAMPLES:
        # resampling 16 kHz
        audio_array = item['audio_path']['array']
        original_sr = item['audio_path']['sampling_rate']
        audio_16k = librosa.resample(y=audio_array, orig_sr=original_sr, target_sr=16000)
        
        # first 5 enrollment, next 10 tests
        if current_count < ENROLL_SAMPLES:
            file_path = f"{ENROLL_DIR}/{speaker_id}/enroll_{current_count + 1}.wav"
        else:
            test_idx = current_count - ENROLL_SAMPLES + 1
            file_path = f"{TEST_DIR}/{speaker_id}/test_{test_idx}.wav"
            
        sf.write(file_path, audio_16k, 16000)
        samples_per_user[speaker_id] += 1
        
        if samples_per_user[speaker_id] == TOTAL_SAMPLES:
            users_completed += 1
            print(f"[{users_completed}/{TARGET_USERS}] profile: {speaker_id}")