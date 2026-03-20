"""
=============================================================================
  SECURITY CONFIGURATION — Centralized settings for the security system
  All sensitive thresholds, keys, and timeouts are managed here.
=============================================================================
"""

import os
import hashlib
import platform
import uuid

# ── Directory Paths ─────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "user_data")
LOG_DIR = os.path.join(PROJECT_DIR, "security_logs")

# ── Encryption Key Derivation ──────────────────────────────────────────────
# Derives a unique key from machine identity so biometric data is
# non-transferable between machines. In production, use a hardware
# security module (HSM) or a proper key management service (KMS).
def _derive_machine_key():
    """Derive a Fernet-compatible key from machine-specific identifiers."""
    # Combine machine-specific data for key derivation
    machine_id_parts = [
        platform.node(),           # hostname
        platform.machine(),        # architecture
        platform.processor(),      # CPU identifier
    ]
    
    # Try to get a more unique ID on Windows
    try:
        machine_id_parts.append(str(uuid.getnode()))  # MAC address
    except Exception:
        pass
    
    combined = "|".join(machine_id_parts).encode("utf-8")
    
    # Use PBKDF2 to derive a strong key from machine identity
    # Salt is fixed per-project to be deterministic
    salt = b"securitix_biometric_v1_salt_2024"
    key_bytes = hashlib.pbkdf2_hmac("sha256", combined, salt, iterations=100_000)
    
    # Fernet requires a 32-byte URL-safe base64-encoded key
    import base64
    return base64.urlsafe_b64encode(key_bytes[:32])


ENCRYPTION_KEY = _derive_machine_key()

# ── Flask Secret Key ───────────────────────────────────────────────────────
# For session management. Override via environment variable in production.
FLASK_SECRET_KEY = os.environ.get(
    "SECURITIX_SECRET_KEY",
    hashlib.sha256(ENCRYPTION_KEY + b"flask_session").hexdigest()
)

# ── Authentication Thresholds ──────────────────────────────────────────────
IRIS_THRESHOLD = 0.82           # Minimum cosine similarity for iris match
VOICE_THRESHOLD = 55.0          # Maximum DTW distance for voice match
GESTURE_THRESHOLD = 0.85        # Minimum cosine similarity for gesture match

# ── Brute-Force Protection ─────────────────────────────────────────────────
MAX_FAILED_ATTEMPTS = 5         # Lock after this many failures
LOCKOUT_DURATION_SECONDS = 60   # How long the lockout lasts
LOCKOUT_WINDOW_SECONDS = 300    # Time window to count failures (5 min)

# ── Session & Timeout ──────────────────────────────────────────────────────
OPERATION_TIMEOUT_SECONDS = 900 # 15 min max for any single operation
SESSION_TIMEOUT_SECONDS = 1800  # 30 min session timeout

# ── Username Validation ────────────────────────────────────────────────────
USERNAME_MIN_LENGTH = 2
USERNAME_MAX_LENGTH = 32
# Only alphanumeric + underscore allowed (prevents path traversal)
USERNAME_PATTERN = r"^[a-zA-Z0-9_]+$"

# ── Liveness Detection ─────────────────────────────────────────────────────
LIVENESS_EAR_VARIANCE_THRESHOLD = 0.001  # Min eye aspect ratio variance (blink detection)
LIVENESS_CHECK_FRAMES = 50              # Number of frames to observe for liveness
