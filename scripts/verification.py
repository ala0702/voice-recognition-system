import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, "../pretrained_models/spkrec-ecapa-voxceleb"))
PROFILES_PATH = os.path.normpath(os.path.join(BASE_DIR, "../voice_profiles.npy"))

# 1. Load the stored biometric database
if not os.path.exists(PROFILES_PATH):
    raise FileNotFoundError(f"Biometric profiles not found at {PROFILES_PATH}. Please run enroll.py first!")

voice_database = np.load(PROFILES_PATH, allow_pickle=True).item()
print(f"Loaded biometric database containing {len(voice_database)} user profiles.")

# 2. Initialize and load the exact same ECAPA-TDNN model architecture
print("Loading ECAPA-TDNN model weights directly via PyTorch state_dict...")
model = ECAPA_TDNN(
    input_size=80, 
    channels=[1024, 1024, 1024, 1024, 3072], 
    lin_neurons=192
)

weights_path = os.path.join(MODEL_DIR, "embedding_model.ckpt")
state_dict = torch.load(weights_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

# 3. Setup the identical Mel-filterbank extractor used during enrollment
mel_extractor = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=80
)

# 4. Core biometric verification function
def verify_speaker(claimed_identity, test_audio_path, threshold=0.25):
    """
    Verifies if the test audio sample matches the template of the claimed identity.
    Returns: (bool access_granted, float similarity_score)
    """
    if claimed_identity not in voice_database:
        print(f"Verification Failed: Identity '{claimed_identity}' does not exist in the database.")
        return False, 0.0
        
    try:
        # Load and preprocess the test audio file
        signal, fs = sf.read(test_audio_path)
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        
        # Extract features
        mel_spec = mel_extractor(signal_tensor)
        log_mel_spec = torch.log(mel_spec + 1e-6)
        features = log_mel_spec.transpose(1, 2)
        
        # Generate test embedding vector
        with torch.no_grad():
            test_embedding = model(features).squeeze() # Shape: (192,)
            
        # Retrieve the saved reference template from enrollment database
        reference_profile = torch.tensor(voice_database[claimed_identity])
        
        # Compute Cosine Similarity between the vectors
        # Formula: (A dot B) / (||A|| * ||B||)
        similarity = torch.nn.functional.cosine_similarity(
            test_embedding.unsqueeze(0), 
            reference_profile.unsqueeze(0)
        ).item()
        
        # Decision boundary check
        access_granted = similarity >= threshold
        return access_granted, similarity
        
    except Exception as e:
        print(f"[ERROR] Failed to process verification sample: {e}")
        return False, 0.0

# --- QUICK BIOMETRIC DEMO / VERIFICATION TEST ---
if __name__ == "__main__":
    print("\n--- Running Biometric System Demo ---")
    
    # Let's test using your profile (id000kuba) and someone else's (id000ala)
    # Adjust paths if your test directories have slightly different names
    test_user = "id000kuba"
    imposter_user = "id000ala"
    real_sample_file = os.path.normpath(os.path.join(BASE_DIR, f"../data/test_genuine/{test_user}/test_1.wav"))
    
    # Default experimental threshold (to be optimized later in task 1)
    SYSTEM_THRESHOLD = 0.75
    
    if os.path.exists(real_sample_file):
        print(f"Loading actual unseen test file from: {test_user}")
        
        # TEST 1: Genuine Attempt (Kuba says he is Kuba and gives his real test audio)
        print("\n[Scenario 1: Genuine User Access Attempt]")
        granted, score = verify_speaker(test_user, real_sample_file, threshold=SYSTEM_THRESHOLD)
        print(f"Claimed ID: {test_user} | Score: {score:.4f} | Threshold: {SYSTEM_THRESHOLD}")
        print(f"System Decision: {'>>> ACCESS GRANTED <<<' if granted else '>>> ACCESS DENIED <<<'}")
        
        # TEST 2: Imposter Attempt (Kuba tells the system he is Ala, using his own voice)
        print("\n[Scenario 2: Imposter Spoofing Attack Attempt]")
        granted, score = verify_speaker(imposter_user, real_sample_file, threshold=SYSTEM_THRESHOLD)
        print(f"Claimed ID: {imposter_user} | Score: {score:.4f} | Threshold: {SYSTEM_THRESHOLD}")
        print(f"System Decision: {'>>> ACCESS GRANTED <<<' if granted else '>>> ACCESS DENIED <<<'}")
    else:
        print(f"\n[Notice] Could not find demo file at '{real_sample_file}'.")
        print("Please verify your file paths under 'data/test_genuine/' to run the instant demo.")