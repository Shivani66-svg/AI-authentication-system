"""
=============================================================================
  TIER 2: VOICE AUTHENTICATION MODULE
  Uses sounddevice for audio recording and librosa for MFCC feature extraction.
  Enrollment: Records voice passphrase and stores MFCC features.
  Authentication: Records voice again, compares using DTW distance.
=============================================================================
"""

import numpy as np
import time
import sys

try:
    import sounddevice as sd
except ImportError:
    print("[ERROR] sounddevice not installed. Run: pip install sounddevice")
    sd = None

try:
    import librosa
except ImportError:
    print("[ERROR] librosa not installed. Run: pip install librosa")
    librosa = None

try:
    import soundfile as sf
except ImportError:
    sf = None


# Audio parameters — use system default sample rate for compatibility
SAMPLE_RATE = 44100   # Match the actual hardware sample rate
RECORD_DURATION = 4   # seconds per recording
N_MFCC = 13


def record_audio(duration=RECORD_DURATION, sample_rate=SAMPLE_RATE):
    """
    Record audio from the microphone.

    Args:
        duration: Recording duration in seconds
        sample_rate: Audio sample rate

    Returns:
        numpy array of audio samples, or None on error
    """
    if sd is None:
        print("  [VOICE ERROR] sounddevice not available!")
        return None

    try:
        print(f"  [VOICE] Recording for {duration} seconds — speak NOW!")

        # Start recording
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                       channels=1, dtype='float32')

        # Simple countdown (non-blocking display)
        for i in range(duration, 0, -1):
            print(f"  [VOICE]   {i}...", flush=True)
            time.sleep(1)

        sd.wait()  # Ensure recording is fully done
        audio = audio.flatten()

        # Check audio level
        max_amp = np.max(np.abs(audio))
        rms_energy = np.sqrt(np.mean(audio ** 2))
        print(f"  [VOICE] Recording done. (max amplitude: {max_amp:.4f}, RMS: {rms_energy:.4f})")

        if max_amp < 0.02:
            print("  [VOICE WARNING] Very low audio — almost silent! Check microphone.")
            return None

        if rms_energy < 0.005:
            print("  [VOICE WARNING] Audio energy too low — no speech detected.")
            return None

        return audio

    except Exception as e:
        print(f"  [VOICE ERROR] Recording failed: {e}")
        return None


def extract_mfcc_features(audio, sample_rate=SAMPLE_RATE):
    """
    Extract MFCC features from audio data.

    Args:
        audio: numpy array of audio samples
        sample_rate: audio sample rate

    Returns:
        numpy array of MFCC features (shape: n_mfcc x time_frames)
    """
    if librosa is None:
        print("  [VOICE ERROR] librosa not available!")
        return None

    try:
        # Trim silence from the beginning and end
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)

        # If trimmed audio is too short, no actual speech was detected
        if len(audio_trimmed) < sample_rate * 0.5:
            print("  [VOICE] No speech detected — audio is mostly silence/noise.")
            return None

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio_trimmed, sr=sample_rate, n_mfcc=N_MFCC)

        # Normalize MFCCs per coefficient (zero mean, unit variance)
        mfcc_mean = np.mean(mfccs, axis=1, keepdims=True)
        mfcc_std = np.std(mfccs, axis=1, keepdims=True)
        mfcc_std[mfcc_std < 1e-6] = 1.0  # Avoid division by zero
        mfccs_normalized = (mfccs - mfcc_mean) / mfcc_std

        return mfccs_normalized

    except Exception as e:
        print(f"  [VOICE ERROR] Feature extraction failed: {e}")
        return None


def enroll_voice():
    """
    Capture voice features for enrollment.
    Records one clear passphrase sample.

    Returns:
        numpy array of MFCC features, or None if cancelled.
    """
    if sd is None or librosa is None:
        print("  [VOICE ERROR] Required libraries not available!")
        print("  [VOICE] Install with: pip install sounddevice librosa soundfile")
        return None

    print("\n  [VOICE] Starting voice enrollment...")
    print("  [VOICE] Speak a clear passphrase when recording begins.")
    print("  [VOICE] Recording starts in 2 seconds...\n")
    time.sleep(2)

    audio = record_audio()
    if audio is None:
        print("  [VOICE] Recording failed!")
        return None

    features = extract_mfcc_features(audio)
    if features is None:
        print("  [VOICE] Feature extraction failed!")
        return None

    print(f"  [VOICE] Enrollment complete! Feature shape: {features.shape}")
    return features


def capture_voice():
    """
    Capture live voice features WITHOUT comparing to any stored template.
    Used during authentication to scan against all enrolled users.

    Returns:
        numpy array of MFCC features, or None if failed.
    """
    if sd is None or librosa is None:
        print("  [VOICE ERROR] Required libraries not available!")
        return None

    print("\n  [VOICE] Scanning voice...")
    print("  [VOICE] Speak your passphrase NOW!\n")

    audio = record_audio()
    if audio is None:
        return None

    features = extract_mfcc_features(audio)
    if features is None:
        return None

    print(f"  [VOICE] Voice scan complete! Feature shape: {features.shape}")
    return features


def verify_voice(stored_features, threshold=80.0):
    """
    Verify voice against stored template using DTW distance.

    Args:
        stored_features: numpy array of enrolled MFCC features
        threshold: maximum DTW distance for a match (lower = stricter)

    Returns:
        tuple: (passed: bool, distance: float)
    """
    from utils import dtw_distance

    if sd is None or librosa is None:
        print("  [VOICE ERROR] Required libraries not available!")
        return False, float('inf')

    print("\n  [VOICE] Starting voice verification...")
    print("  [VOICE] Speak the SAME passphrase you used during enrollment.")
    print("  [VOICE] Recording starts in 2 seconds...\n")

    time.sleep(2)

    audio = record_audio()
    if audio is None:
        return False, float('inf')

    features = extract_mfcc_features(audio)
    if features is None:
        return False, float('inf')

    # Compute DTW distance (transpose to get time-series format)
    distance = dtw_distance(features.T, stored_features.T)

    passed = distance <= threshold
    status = "PASSED" if passed else "FAILED"
    print(f"  [VOICE] Verification {status} (Distance: {distance:.2f}, Threshold: {threshold:.1f})")

    return passed, distance
