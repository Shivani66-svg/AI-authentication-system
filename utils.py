"""
=============================================================================
  UTILITIES MODULE — Shared helper functions
=============================================================================
"""

import numpy as np
from scipy.spatial.distance import cosine


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    Returns a value between 0 (no match) and 1 (perfect match).
    """
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return 1.0 - cosine(vec1, vec2)


def euclidean_distance(vec1, vec2):
    """Compute Euclidean distance between two vectors."""
    return np.linalg.norm(np.array(vec1) - np.array(vec2))


def normalize_vector(vec):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def dtw_distance(seq1, seq2):
    """
    Dynamic Time Warping distance between two sequences.
    Used for comparing voice MFCC features of different lengths.
    """
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return float('inf')

    # Ensure both are 2D
    if seq1.ndim == 1:
        seq1 = seq1.reshape(-1, 1)
    if seq2.ndim == 1:
        seq2 = seq2.reshape(-1, 1)

    # DTW cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    return dtw_matrix[n, m] / (n + m)  # Normalize by path length


def print_banner(text, char="=", width=60):
    """Print a formatted banner."""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_status(label, status, success=True):
    """Print a formatted status line."""
    icon = "✓" if success else "✗"
    color_code = "" # Terminal colors not reliable on Windows
    print(f"  [{icon}] {label}: {status}")


def print_tier_header(tier_num, tier_name):
    """Print a formatted tier header."""
    print(f"\n{'─' * 50}")
    print(f"  TIER {tier_num}: {tier_name}")
    print(f"{'─' * 50}")
