import os
import io
import soundfile as sf
import librosa
from datasets import load_dataset, Features, Value
import imageio_ffmpeg

# Fix for FFmpeg path on Windows environment
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

TARGET_USERS = 100
ENROLL_SAMPLES = 5    # Number of samples used to build the voice profile
TEST_SAMPLES = 10     # Number of test samples per user
TOTAL_SAMPLES = ENROLL_SAMPLES + TEST_SAMPLES

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
ENROLL_DIR = os.path.join(BASE_DIR, "../data/enrollment")
TEST_DIR = os.path.join(BASE_DIR, "../data/test_genuine")

os.makedirs(ENROLL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

custom_features = Features({
    "audio_path": {"bytes": Value("binary"), "path": Value("string")},
    "transcription": Value("string"),
    "language": Value("string"),
    "speaker_id": Value("string"),
    "gender": Value("string")
})

print("Connecting to Hugging Face and initializing stream...")
dataset = load_dataset("acul3/voxceleb2", split="train", streaming=True, features=custom_features)

users_completed = 0
samples_per_user = {}

print("Starting data downloading and processing...")

for item in dataset:
    # 1. STOP CONDITION: Exit instantly when we have 100 fully completed profiles
    if users_completed >= TARGET_USERS:
        break
        
    speaker_id = item['speaker_id']
    
    # 2. SKIP NEW USERS ONLY IF we already have 100 profiles actively inside our dictionary
    # AND this speaker is not one of them.
    if speaker_id not in samples_per_user and len(samples_per_user) >= TARGET_USERS:
        # If some users in our dict are stuck, we remove the oldest uncompleted user 
        # to free up a slot for this new active speaker.
        uncompleted_speakers = [spk for spk, count in samples_per_user.items() if count < TOTAL_SAMPLES]
        if uncompleted_speakers:
            stuck_speaker = uncompleted_speakers[0]
            del samples_per_user[stuck_speaker]
            # Now len(samples_per_user) is 99, so the code below will accept the new speaker!
        else:
            continue
        
    # Initialize directory structure for a new user
    if speaker_id not in samples_per_user:
        samples_per_user[speaker_id] = 0
        os.makedirs(os.path.join(ENROLL_DIR, speaker_id), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, speaker_id), exist_ok=True)

    current_count = samples_per_user[speaker_id]
    
    # Process audio if the user still needs more samples
    if current_count < TOTAL_SAMPLES:
        try:
            audio_bytes = item['audio_path']['bytes']
            audio_stream = io.BytesIO(audio_bytes)
            
            audio_array, original_sr = librosa.load(audio_stream, sr=None)
            audio_16k = librosa.resample(y=audio_array, orig_sr=original_sr, target_sr=16000)
            
            if current_count < ENROLL_SAMPLES:
                file_path = os.path.join(ENROLL_DIR, speaker_id, f"enroll_{current_count + 1}.wav")
            else:
                test_idx = current_count - ENROLL_SAMPLES + 1
                file_path = os.path.join(TEST_DIR, speaker_id, f"test_{test_idx}.wav")
                
            sf.write(file_path, audio_16k, 16000)
            samples_per_user[speaker_id] += 1
            
            if samples_per_user[speaker_id] == TOTAL_SAMPLES:
                users_completed += 1
                print(f"[{users_completed}/{TARGET_USERS}] Completed profile for speaker: {speaker_id}")
                
        except Exception as e:
            print(f"Skipped sample for {speaker_id} due to error: {e}")
            continue

print("Data downloading and processing finished successfully!")