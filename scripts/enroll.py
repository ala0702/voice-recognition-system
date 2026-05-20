import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, "../pretrained_models/spkrec-ecapa-voxceleb"))
ENROLL_DIR = os.path.normpath(os.path.join(BASE_DIR, "../data/enrollment"))
PROFILES_PATH = os.path.normpath(os.path.join(BASE_DIR, "../voice_profiles.npy"))

print("Loading ECAPA-TDNN model weights directly via PyTorch state_dict...")
# Initialize the model with correct official channel sizes
model = ECAPA_TDNN(
    input_size=80, 
    channels=[1024, 1024, 1024, 1024, 3072], 
    lin_neurons=192
)

# Load the weights
weights_path = os.path.join(MODEL_DIR, "embedding_model.ckpt")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Missing weights file at {weights_path}.")

state_dict = torch.load(weights_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

# Native PyTorch audio feature extractor (80 Mel-filterbanks, matching VoxCeleb specs)
mel_extractor = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=80
)

voice_database = {}
print("\nStarting enrollment process...")

# Iterate through each speaker folder
for speaker_id in os.listdir(ENROLL_DIR):
    speaker_folder = os.path.join(ENROLL_DIR, speaker_id)
    if not os.path.isdir(speaker_folder):
        continue
        
    print(f"Enrolling speaker: {speaker_id}")
    speaker_embeddings = []
    
    wav_files = [f for f in os.listdir(speaker_folder) if f.endswith('.wav')]
    
    for wav_file in wav_files:
        wav_path = os.path.join(speaker_folder, wav_file)
        
        try:
            # Load 16kHz WAV sample
            signal, fs = sf.read(wav_path)
            signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
            
            # Extract Mel Spectrogram & apply log-scale (Log-Mel Filterbank Features)
            mel_spec = mel_extractor(signal_tensor)
            log_mel_spec = torch.log(mel_spec + 1e-6)
            
            # Transpose to match SpeechBrain's input format: [batch, time, features]
            features = log_mel_spec.transpose(1, 2)
            
            # Compute speaker embedding
            with torch.no_grad():
                embedding = model(features).squeeze().cpu().numpy()
                # Ensure it's a 1D vector of shape (192,)
                if embedding.ndim == 1:
                    speaker_embeddings.append(embedding)
                
        except Exception as e:
            print(f"  [ERROR] Could not process {wav_file}: {e}")
            
    if speaker_embeddings:
        # Save averaged 192-dimensional vector as speaker profile
        average_profile = np.mean(speaker_embeddings, axis=0)
        voice_database[speaker_id] = average_profile
        print(f"  -> Profile created successfully using {len(speaker_embeddings)} samples.")

# Save database to disk
np.save(PROFILES_PATH, voice_database)
print(f"\nEnrollment complete! Saved {len(voice_database)} users to '{PROFILES_PATH}'.")