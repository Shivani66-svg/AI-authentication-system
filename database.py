"""
=============================================================================
  DATABASE MODULE — User Biometric Data Storage (SECURED)
  Stores enrolled user data as encrypted binary files.
  
  Security improvements:
    - Biometric data encrypted at rest (Fernet AES-128-CBC)
    - Username validation (alphanumeric only, prevents path traversal)
    - No allow_pickle=True (prevents code execution attacks)
    - Restrictive file permissions
    - Security event logging
=============================================================================
"""

import os
import re
import json
import stat
import numpy as np

from config import DATA_DIR, USERNAME_PATTERN, USERNAME_MIN_LENGTH, USERNAME_MAX_LENGTH
from crypto_utils import encrypt_and_save, load_and_decrypt
from security_logger import log_security_event, log_error


def ensure_data_dir():
    """Create the data directory if it doesn't exist, with restrictive permissions."""
    os.makedirs(DATA_DIR, exist_ok=True)
    # Restrict directory permissions (owner only) on non-Windows
    try:
        if os.name != 'nt':
            os.chmod(DATA_DIR, stat.S_IRWXU)  # 700: owner read/write/execute only
    except Exception:
        pass


def validate_username(username):
    """
    Validate a username for safety.
    
    Returns:
        str: sanitized username, or None if invalid
    """
    if not username or not isinstance(username, str):
        return None
    
    username = username.lower().strip()
    
    # Length check
    if len(username) < USERNAME_MIN_LENGTH or len(username) > USERNAME_MAX_LENGTH:
        return None
    
    # Pattern check (alphanumeric + underscore only — no path traversal)
    if not re.match(USERNAME_PATTERN, username):
        return None
    
    # Block dangerous names
    blocked = {"con", "prn", "aux", "nul", "com1", "lpt1", "..", ".", "admin", "root"}
    if username in blocked:
        return None
    
    return username


def get_user_dir(username):
    """Get the directory path for a specific user (with validation)."""
    safe_name = validate_username(username)
    if safe_name is None:
        raise ValueError(f"Invalid username: '{username}'")
    
    user_dir = os.path.join(DATA_DIR, safe_name)
    
    # Double-check the resolved path is within DATA_DIR (defense in depth)
    real_data = os.path.realpath(DATA_DIR)
    real_user = os.path.realpath(user_dir)
    if not real_user.startswith(real_data):
        log_error("get_user_dir", f"Path traversal attempt: {username}")
        raise ValueError("Invalid username")
    
    return user_dir


def user_exists(username):
    """Check if a user is already enrolled."""
    try:
        user_dir = get_user_dir(username)
    except ValueError:
        return False
    info_file = os.path.join(user_dir, "user_info.json")
    return os.path.exists(info_file)


def get_all_users():
    """Get a list of all enrolled usernames."""
    ensure_data_dir()
    users = []
    if os.path.exists(DATA_DIR):
        for name in os.listdir(DATA_DIR):
            # Validate each directory name as a username
            if validate_username(name) is None:
                continue
            info_file = os.path.join(DATA_DIR, name, "user_info.json")
            if os.path.isdir(os.path.join(DATA_DIR, name)) and os.path.exists(info_file):
                users.append(name)
    return users


def save_user_data(username, iris_features, voice_features, gesture_features):
    """
    Save all biometric data for a user (ENCRYPTED).

    Args:
        username: User's name/ID (will be validated)
        iris_features: numpy array of iris feature vector
        voice_features: numpy array of MFCC voice features
        gesture_features: numpy array of hand gesture landmarks
    
    Raises:
        ValueError: if username is invalid
    """
    ensure_data_dir()
    user_dir = get_user_dir(username)  # validates username
    os.makedirs(user_dir, exist_ok=True)

    # Save user info (non-sensitive metadata — not encrypted)
    safe_name = validate_username(username)
    info = {
        "username": safe_name,
        "enrolled": True,
        "tiers": {
            "iris": True,
            "voice": True,
            "gesture": True,
        }
    }
    with open(os.path.join(user_dir, "user_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # Save biometric data as ENCRYPTED files
    encrypt_and_save(os.path.join(user_dir, "iris_features.enc"), iris_features)
    encrypt_and_save(os.path.join(user_dir, "voice_features.enc"), voice_features)
    encrypt_and_save(os.path.join(user_dir, "gesture_features.enc"), gesture_features)

    log_security_event("DATA_SAVE", f"Biometric data saved for user '{safe_name}'")
    print(f"\n  [DATABASE] User '{safe_name}' data saved successfully (encrypted).")


def load_user_data(username):
    """
    Load all biometric data for a user (DECRYPTED).

    Returns:
        tuple: (iris_features, voice_features, gesture_features) or None if not found
    """
    try:
        user_dir = get_user_dir(username)
    except ValueError:
        return None

    if not user_exists(username):
        print(f"\n  [DATABASE] User '{username}' not found.")
        return None

    try:
        # Try loading encrypted files first (new format)
        iris_enc = os.path.join(user_dir, "iris_features.enc")
        voice_enc = os.path.join(user_dir, "voice_features.enc")
        gesture_enc = os.path.join(user_dir, "gesture_features.enc")
        
        if os.path.exists(iris_enc):
            # New encrypted format
            iris_features = load_and_decrypt(iris_enc)
            voice_features = load_and_decrypt(voice_enc)
            gesture_features = load_and_decrypt(gesture_enc)
        else:
            # Legacy unencrypted format — load and re-encrypt
            iris_features = np.load(
                os.path.join(user_dir, "iris_features.npy"), allow_pickle=False
            )
            voice_features = np.load(
                os.path.join(user_dir, "voice_features.npy"), allow_pickle=False
            )
            gesture_features = np.load(
                os.path.join(user_dir, "gesture_features.npy"), allow_pickle=False
            )
            
            # Migrate: re-save as encrypted and remove old files
            log_security_event("MIGRATION", f"Migrating user '{username}' to encrypted storage")
            encrypt_and_save(iris_enc, iris_features)
            encrypt_and_save(voice_enc, voice_features)
            encrypt_and_save(gesture_enc, gesture_features)
            
            # Remove old unencrypted files
            for old_file in ["iris_features.npy", "voice_features.npy", "gesture_features.npy"]:
                old_path = os.path.join(user_dir, old_file)
                if os.path.exists(old_path):
                    os.remove(old_path)
        
        if iris_features is None or voice_features is None or gesture_features is None:
            log_error("load_user_data", f"Decryption failed for user '{username}'")
            return None
        
        return iris_features, voice_features, gesture_features
    except Exception as e:
        log_error("load_user_data", f"Error loading data for '{username}': {e}")
        print(f"\n  [DATABASE] Error loading data for '{username}'.")
        return None


def delete_user(username):
    """Delete a user's enrolled data."""
    import shutil
    try:
        user_dir = get_user_dir(username)
    except ValueError:
        return False
    
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
        log_security_event("USER_DELETE", f"User '{username}' data deleted")
        print(f"\n  [DATABASE] User '{username}' deleted.")
        return True
    return False
