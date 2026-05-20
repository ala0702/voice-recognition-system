import os
import subprocess
import numpy as np
import librosa
import soundfile as sf
import imageio_ffmpeg

# Get the clean path to the FFmpeg executable
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

TARGET_SR = 16000
ENROLL_SAMPLES = 5

# BASE_DIR targets 'voice-recognition-system/scripts'
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."

# Correct relative routing based on your folder structure
RAW_DIR = os.path.normpath(os.path.join(BASE_DIR, "../temp"))
ENROLL_DIR = os.path.normpath(os.path.join(BASE_DIR, "../data/enrollment"))
TEST_DIR = os.path.normpath(os.path.join(BASE_DIR, "../data/test_genuine"))

members = {
    "id000ala": "ala",
    "id000kuba": "kuba"
}

for speaker_id, raw_folder_name in members.items():
    raw_member_path = os.path.join(RAW_DIR, raw_folder_name)
    
    if not os.path.exists(raw_member_path):
        print(f"Warning: Folder {raw_member_path} not found. Skipping.")
        continue
        
    os.makedirs(os.path.join(ENROLL_DIR, speaker_id), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, speaker_id), exist_ok=True)
    
    # Track only our renamed clean .m4a files
    files = [f for f in os.listdir(raw_member_path) if f.lower().endswith('.m4a')]
    
    # Custom sort to handle numbers correctly (ala1, ala2 ... ala10)
    def extract_number(filename):
        just_name = os.path.splitext(filename)[0]
        digits = "".join([c for c in just_name if c.isdigit()])
        return int(digits) if digits else filename

    files.sort(key=extract_number)
    
    print(f"\nProcessing speaker: {speaker_id}...")
    print(f"Source path: {raw_member_path}")
    print(f"Found files count: {len(files)}")
    
    for idx, file_name in enumerate(files):
        if idx >= 15:
            break
            
        full_raw_path = os.path.join(raw_member_path, file_name)
        
        try:
            # 1. Use FFmpeg directly via subprocess to extract raw 16-bit PCM mono stream.
            # We target a high sample rate temporarily (44100) to keep high definition before resampling.
            command = [
                FFMPEG_EXE,
                "-y",               # Overwrite output files if they exist
                "-i", full_raw_path,# Input file path
                "-f", "s16le",      # Force raw 16-bit signed PCM little-endian format
                "-acodec", "pcm_s16le",
                "-ac", "1",         # Force 1 audio channel (mono)
                "-ar", "44100",     # Intermediate sampling rate
                "-"                 # Output to stdout stream (pipe directly to RAM)
            ]
            
            # Run FFmpeg process silently in the background
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout_data, stderr_data = process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {stderr_data.decode('utf-8', errors='ignore')}")
            
            # 2. Convert the raw binary buffer into a floating-point numpy array
            audio_array = np.frombuffer(stdout_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            if len(audio_array) == 0:
                raise ValueError("Extracted audio data stream is empty.")

            # 3. Perform accurate resampling from 44.1 kHz to 16 kHz using librosa
            audio_16k = librosa.resample(y=audio_array, orig_sr=44100, target_sr=TARGET_SR)
            
            # Assign pathing structures
            if idx < ENROLL_SAMPLES:
                output_path = os.path.join(ENROLL_DIR, speaker_id, f"enroll_{idx + 1}.wav")
            else:
                test_idx = idx - ENROLL_SAMPLES + 1
                output_path = os.path.join(TEST_DIR, speaker_id, f"test_{test_idx}.wav")
                
            # 4. Save the finalized array as a proper 16kHz WAV file
            sf.write(output_path, audio_16k, TARGET_SR)
            print(f"  -> Saved: {output_path}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {file_name}: {e}")

print("\nTeam members added successfully to the biometric dataset!")