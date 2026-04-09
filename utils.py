"""
=============================================================================
  UTILITIES MODULE — Shared helper functions
=============================================================================
"""

import numpy as np
from scipy.spatial.distance import cosine


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two NORMALIZED vectors.
    Returns a value between 0 (no match) and 1 (perfect match).
    Normalizes both vectors first for consistent comparison.
    """
    v1 = np.array(vec1, dtype=np.float64).flatten()
    v2 = np.array(vec2, dtype=np.float64).flatten()

    # Handle length mismatch — truncate to shorter
    min_len = min(len(v1), len(v2))
    if min_len == 0:
        return 0.0
    v1 = v1[:min_len]
    v2 = v2[:min_len]

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0

    return float(1.0 - cosine(v1, v2))


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


def find_best_match(scores, threshold, margin, higher_is_better=True):
    """
    Find the best matching user with discrimination margin enforcement.
    
    The best match must:
      1. Pass the threshold
      2. Beat the 2nd best by at least 'margin'
    
    This prevents picking the wrong user when two users have very similar scores.
    
    Args:
        scores: dict of {username: score}
        threshold: minimum score for 'higher_is_better=True', max for False
        margin: minimum gap required between best and 2nd best
        higher_is_better: True for similarity (higher=better), False for distance (lower=better)
    
    Returns:
        tuple: (best_user, best_score, passed, is_discriminated)
            - passed: True if threshold is met
            - is_discriminated: True if margin check passed (best is clearly better than 2nd)
    """
    if not scores:
        return None, 0.0, False, False

    if higher_is_better:
        sorted_users = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    else:
        sorted_users = sorted(scores.items(), key=lambda x: x[1], reverse=False)

    best_user, best_score = sorted_users[0]

    # Check threshold
    if higher_is_better:
        passed = best_score >= threshold
    else:
        passed = best_score <= threshold

    # Check discrimination margin (only matters with 2+ users)
    if len(sorted_users) >= 2:
        second_score = sorted_users[1][1]
        if higher_is_better:
            gap = best_score - second_score
        else:
            gap = second_score - best_score  # for distance, 2nd should be higher
        is_discriminated = gap >= margin
    else:
        # Only 1 user enrolled — no margin needed
        is_discriminated = True

    return best_user, best_score, passed, is_discriminated


def print_banner(text, char="=", width=60):
    """Print a formatted banner."""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_status(label, status, success=True):
    """Print a formatted status line."""
    icon = "✓" if success else "✗"
    print(f"  [{icon}] {label}: {status}")


def print_tier_header(tier_num, tier_name):
    """Print a formatted tier header."""
    print(f"\n{'─' * 50}")
    print(f"  TIER {tier_num}: {tier_name}")
    print(f"{'─' * 50}")
