"""
=============================================================================
  DATABASE MODULE — User Biometric Data Storage
  Stores enrolled user data as JSON + numpy binary files
=============================================================================
"""

import os
import json
import numpy as np

# Directory to store all user data
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data")


def ensure_data_dir():
    """Create the data directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def get_user_dir(username):
    """Get the directory path for a specific user."""
    return os.path.join(DATA_DIR, username.lower().strip())


def user_exists(username):
    """Check if a user is already enrolled."""
    user_dir = get_user_dir(username)
    info_file = os.path.join(user_dir, "user_info.json")
    return os.path.exists(info_file)


def get_all_users():
    """Get a list of all enrolled usernames."""
    ensure_data_dir()
    users = []
    if os.path.exists(DATA_DIR):
        for name in os.listdir(DATA_DIR):
            info_file = os.path.join(DATA_DIR, name, "user_info.json")
            if os.path.isdir(os.path.join(DATA_DIR, name)) and os.path.exists(info_file):
                users.append(name)
    return users


def save_user_data(username, iris_features, voice_features, gesture_features):
    """
    Save all biometric data for a user.

    Args:
        username: User's name/ID
        iris_features: numpy array of iris feature vector
        voice_features: numpy array of MFCC voice features
        gesture_features: numpy array of hand gesture landmarks
    """
    ensure_data_dir()
    user_dir = get_user_dir(username)
    os.makedirs(user_dir, exist_ok=True)

    # Save user info
    info = {
        "username": username.lower().strip(),
        "enrolled": True,
        "tiers": {
            "iris": True,
            "voice": True,
            "gesture": True,
        }
    }
    with open(os.path.join(user_dir, "user_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # Save biometric data as numpy files
    np.save(os.path.join(user_dir, "iris_features.npy"), iris_features)
    np.save(os.path.join(user_dir, "voice_features.npy"), voice_features)
    np.save(os.path.join(user_dir, "gesture_features.npy"), gesture_features)

    print(f"\n  [DATABASE] User '{username}' data saved successfully.")


def load_user_data(username):
    """
    Load all biometric data for a user.

    Returns:
        tuple: (iris_features, voice_features, gesture_features) or None if not found
    """
    user_dir = get_user_dir(username)

    if not user_exists(username):
        print(f"\n  [DATABASE] User '{username}' not found.")
        return None

    try:
        iris_features = np.load(os.path.join(user_dir, "iris_features.npy"), allow_pickle=True)
        voice_features = np.load(os.path.join(user_dir, "voice_features.npy"), allow_pickle=True)
        gesture_features = np.load(os.path.join(user_dir, "gesture_features.npy"), allow_pickle=True)
        return iris_features, voice_features, gesture_features
    except Exception as e:
        print(f"\n  [DATABASE] Error loading data for '{username}': {e}")
        return None


def delete_user(username):
    """Delete a user's enrolled data."""
    import shutil
    user_dir = get_user_dir(username)
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
        print(f"\n  [DATABASE] User '{username}' deleted.")
        return True
    return False
