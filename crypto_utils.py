"""
=============================================================================
  CRYPTO UTILITIES — Encryption/Decryption for biometric data
  Uses Fernet (AES-128-CBC with HMAC-SHA256) for symmetric encryption.
  Biometric data is encrypted at rest so raw templates cannot be stolen.
=============================================================================
"""

import numpy as np
import io
from cryptography.fernet import Fernet

from config import ENCRYPTION_KEY


def _get_cipher():
    """Get a Fernet cipher instance using the machine-derived key."""
    return Fernet(ENCRYPTION_KEY)


def encrypt_array(arr):
    """
    Encrypt a numpy array.
    
    Args:
        arr: numpy array to encrypt
    
    Returns:
        bytes: encrypted data
    """
    cipher = _get_cipher()
    
    # Serialize numpy array to bytes
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    raw_bytes = buffer.getvalue()
    
    # Encrypt
    return cipher.encrypt(raw_bytes)


def decrypt_array(encrypted_data):
    """
    Decrypt an encrypted numpy array.
    
    Args:
        encrypted_data: encrypted bytes
    
    Returns:
        numpy array, or None if decryption fails
    """
    try:
        cipher = _get_cipher()
        decrypted = cipher.decrypt(encrypted_data)
        
        # Deserialize numpy array
        buffer = io.BytesIO(decrypted)
        return np.load(buffer, allow_pickle=False)
    except Exception:
        return None


def encrypt_and_save(filepath, arr):
    """
    Encrypt a numpy array and save to file.
    
    Args:
        filepath: path to save encrypted data
        arr: numpy array to encrypt and save
    """
    encrypted = encrypt_array(arr)
    with open(filepath, "wb") as f:
        f.write(encrypted)


def load_and_decrypt(filepath):
    """
    Load and decrypt a numpy array from file.
    
    Args:
        filepath: path to encrypted data file
    
    Returns:
        numpy array, or None if file doesn't exist or decryption fails
    """
    try:
        with open(filepath, "rb") as f:
            encrypted = f.read()
        return decrypt_array(encrypted)
    except FileNotFoundError:
        return None
    except Exception:
        return None
