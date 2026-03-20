"""
=============================================================================
  THREE-TIER BIOMETRIC SECURITY SYSTEM
  =====================================

  Tier 1: Iris Detection    (MediaPipe Face Mesh)
  Tier 2: Voice Detection   (MFCC + DTW)
  Tier 3: Hand Gesture      (MediaPipe Hands)

  Modes:
    1. ENROLL  — Register a new user's biometric data
    2. AUTHENTICATE — Verify a user against stored data
    3. LIST USERS — Show all enrolled users
    4. DELETE USER — Remove a user's data
    5. EXIT

  Author: Security System v1.0
=============================================================================
"""

import sys
import os
import time
import subprocess

# Ensure the project directory is in PATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import user_exists, save_user_data, load_user_data, get_all_users, delete_user
from database import validate_username
from iris_auth import enroll_iris, verify_iris, capture_iris
from voice_auth import enroll_voice, verify_voice, capture_voice
from gesture_auth import enroll_gesture, verify_gesture, capture_gesture
from utils import print_banner, print_status, print_tier_header, cosine_similarity, dtw_distance
from config import (
    MAX_FAILED_ATTEMPTS, LOCKOUT_DURATION_SECONDS,
    IRIS_THRESHOLD, VOICE_THRESHOLD, GESTURE_THRESHOLD
)
from security_logger import (
    log_enrollment, log_authentication, log_auth_attempt,
    log_lockout, log_user_deleted, log_error
)

# ── Brute-Force Protection ──────────────────────────────────────────────────
failed_auth_timestamps = []


def is_locked_out():
    """Check if too many failed auth attempts have occurred."""
    now = time.time()
    # Clean old entries
    failed_auth_timestamps[:] = [t for t in failed_auth_timestamps if now - t < 300]
    if len(failed_auth_timestamps) >= MAX_FAILED_ATTEMPTS:
        if failed_auth_timestamps and (now - failed_auth_timestamps[-1]) < LOCKOUT_DURATION_SECONDS:
            return True
    return False


def record_failure():
    """Record a failed authentication."""
    failed_auth_timestamps.append(time.time())


def clear_screen():
    """Clear the console screen safely (no shell injection)."""
    if os.name == 'nt':
        subprocess.run(['cmd', '/c', 'cls'], shell=False)
    else:
        subprocess.run(['clear'], shell=False)


def show_main_menu():
    """Display the main menu."""
    clear_screen()
    print()
    print("=" * 60)
    print("   ____  _____ ____ _   _ ____  ___ _______   __")
    print("  / ___|| ____/ ___| | | |  _ \\|_ _|_   _\\ \\ / /")
    print("  \\___ \\|  _|| |   | | | | |_) || |  | |  \\ V / ")
    print("   ___) | |__| |___| |_| |  _ < | |  | |   | |  ")
    print("  |____/|_____\\____|\\___/|_| \\_\\___|_|_|   |_|  ")
    print("                                                  ")
    print("     THREE-TIER BIOMETRIC SECURITY SYSTEM         ")
    print("=" * 60)
    print()
    print("  [1]  Enroll New User")
    print("  [2]  Authenticate User")
    print("  [3]  List Enrolled Users")
    print("  [4]  Delete User")
    print("  [5]  Exit")
    print()
    print("-" * 60)


def show_result_screen(passed, username, scores):
    """Display the final authentication result."""
    clear_screen()
    print()
    if passed:
        print("=" * 60)
        print("      █████╗  ██████╗ ██████╗███████╗███████╗███████╗")
        print("     ██╔══██╗██╔════╝██╔════╝██╔════╝██╔════╝██╔════╝")
        print("     ███████║██║     ██║     █████╗  ███████╗███████╗")
        print("     ██╔══██║██║     ██║     ██╔══╝  ╚════██║╚════██║")
        print("     ██║  ██║╚██████╗╚██████╗███████╗███████║███████║")
        print("     ╚═╝  ╚═╝ ╚═════╝ ╚═════╝╚══════╝╚══════╝╚══════╝")
        print()
        print("          ██████╗ ██████╗  █████╗ ███╗   ██╗████████╗")
        print("         ██╔════╝ ██╔══██╗██╔══██╗████╗  ██║╚══██╔══╝")
        print("         ██║  ███╗██████╔╝███████║██╔██╗ ██║   ██║   ")
        print("         ██║   ██║██╔══██╗██╔══██║██║╚██╗██║   ██║   ")
        print("         ╚██████╔╝██║  ██║██║  ██║██║ ╚████║   ██║   ")
        print("          ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ")
        print("=" * 60)
        print(f"\n  Identified as: {username}")
        print(f"\n  All three security tiers passed successfully.")
    else:
        print("=" * 60)
        print("     ██████╗ ███████╗███╗   ██╗██╗███████╗██████╗ ")
        print("     ██╔══██╗██╔════╝████╗  ██║██║██╔════╝██╔══██╗")
        print("     ██║  ██║█████╗  ██╔██╗ ██║██║█████╗  ██║  ██║")
        print("     ██║  ██║██╔══╝  ██║╚██╗██║██║██╔══╝  ██║  ██║")
        print("     ██████╔╝███████╗██║ ╚████║██║███████╗██████╔╝")
        print("     ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝╚══════╝╚═════╝ ")
        print()
        print("      █████╗  ██████╗ ██████╗███████╗███████╗███████╗")
        print("     ██╔══██╗██╔════╝██╔════╝██╔════╝██╔════╝██╔════╝")
        print("     ███████║██║     ██║     █████╗  ███████╗███████╗")
        print("     ██╔══██║██║     ██║     ██╔══╝  ╚════██║╚════██║")
        print("     ██║  ██║╚██████╗╚██████╗███████╗███████║███████║")
        print("     ╚═╝  ╚═╝ ╚═════╝ ╚═════╝╚══════╝╚══════╝╚══════╝")
        print("=" * 60)
        print(f"\n  No matching user found in the system.")
        print(f"\n  Biometric data does not match any enrolled user.")

    # Show detailed scores
    print(f"\n  {'─' * 45}")
    print(f"  AUTHENTICATION SUMMARY")
    print(f"  {'─' * 45}")
    for tier, (status, detail) in scores.items():
        icon = "[PASS]" if status else "[FAIL]"
        print(f"  {icon} {tier}: {detail}")
    print(f"  {'─' * 45}")
    print()


def enroll_user():
    """Full enrollment flow for a new user."""
    clear_screen()
    print_banner("USER ENROLLMENT")
    print()

    username = input("  Enter username: ").strip()
    if not username:
        print("  [ERROR] Username cannot be empty.")
        input("\n  Press ENTER to continue...")
        return

    # Validate username (security: prevent path traversal)
    safe_name = validate_username(username)
    if safe_name is None:
        print("  [ERROR] Invalid username. Use only letters, numbers, and underscores (2-32 chars).")
        input("\n  Press ENTER to continue...")
        return
    username = safe_name

    if user_exists(username):
        print(f"  [WARNING] User '{username}' already exists!")
        overwrite = input("  Overwrite existing data? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("  [INFO] Enrollment cancelled.")
            input("\n  Press ENTER to continue...")
            return

    print(f"\n  Enrolling user: {username}")
    print("  You will go through 3 biometric enrollment steps.\n")

    # ── TIER 1: Iris Enrollment ──────────────────────────────────────
    print_tier_header(1, "IRIS ENROLLMENT")
    input("  Press ENTER when ready...")

    iris_features = enroll_iris()
    if iris_features is None:
        print("\n  [ERROR] Iris enrollment failed. Aborting.")
        input("\n  Press ENTER to continue...")
        return

    print_status("Iris Enrollment", "Complete", True)
    time.sleep(1)

    # ── TIER 2: Voice Enrollment ─────────────────────────────────────
    print_tier_header(2, "VOICE ENROLLMENT")

    voice_features = enroll_voice()
    if voice_features is None:
        print("\n  [ERROR] Voice enrollment failed. Aborting.")
        input("\n  Press ENTER to continue...")
        return

    print_status("Voice Enrollment", "Complete", True)
    time.sleep(1)

    # ── TIER 3: Gesture Enrollment ───────────────────────────────────
    print_tier_header(3, "HAND GESTURE ENROLLMENT")
    input("  Press ENTER when ready...")

    gesture_features = enroll_gesture()
    if gesture_features is None:
        print("\n  [ERROR] Gesture enrollment failed. Aborting.")
        input("\n  Press ENTER to continue...")
        return

    print_status("Gesture Enrollment", "Complete", True)
    time.sleep(1)

    # ── Save all data ────────────────────────────────────────────────
    save_user_data(username, iris_features, voice_features, gesture_features)
    log_enrollment(username, True)

    print_banner(f"ENROLLMENT COMPLETE FOR '{username.upper()}'", char="*")
    print("\n  All three biometric tiers enrolled successfully!")
    print("  You can now authenticate using the main menu.\n")
    input("  Press ENTER to continue...")


def authenticate_user():
    """
    Full authentication flow — fully automatic, no username required.
    Captures all 3 biometrics and scans against ALL enrolled users
    to identify and verify the person.
    """
    clear_screen()
    print_banner("USER AUTHENTICATION")
    print()

    # Brute-force lockout check
    if is_locked_out():
        remaining = int(LOCKOUT_DURATION_SECONDS - (time.time() - failed_auth_timestamps[-1]))
        print(f"  [SECURITY] System locked due to too many failed attempts.")
        print(f"  [SECURITY] Try again in {max(remaining, 1)} seconds.")
        log_lockout("console", "Authentication blocked by lockout")
        input("\n  Press ENTER to continue...")
        return

    log_auth_attempt("Console authentication started")
    print("  The system will automatically identify you.")
    print("  No username required — just present your biometrics.\n")

    # Check if there are any enrolled users
    users = get_all_users()
    if not users:
        print("  [ERROR] No users enrolled in the system!")
        print("  Please enroll at least one user first.")
        input("\n  Press ENTER to continue...")
        return

    print(f"  [{len(users)} user(s) enrolled in database]\n")
    print("  You will go through 3 biometric scans.")
    print("  The system will match your data against all enrolled users.\n")

    # ── TIER 1: Iris Scan ────────────────────────────────────────────
    print_tier_header(1, "IRIS SCAN")
    input("  Press ENTER when ready to scan your iris...")

    live_iris = capture_iris()
    if live_iris is None:
        print("\n  [ERROR] Iris scan failed. Authentication aborted.")
        input("\n  Press ENTER to continue...")
        return

    # Match iris against all enrolled users
    print("\n  [SYSTEM] Matching iris against enrolled users...")
    iris_scores = {}
    for user in users:
        data = load_user_data(user)
        if data is not None:
            stored_iris, _, _ = data
            score = cosine_similarity(stored_iris, live_iris)
            iris_scores[user] = score
            print(f"    - {user}: {score:.2%}")

    if not iris_scores:
        print("  [ERROR] Could not load any user data.")
        input("\n  Press ENTER to continue...")
        return

    # Find best iris match
    best_iris_user = max(iris_scores, key=iris_scores.get)
    best_iris_score = iris_scores[best_iris_user]
    iris_passed = best_iris_score >= IRIS_THRESHOLD

    if iris_passed:
        print(f"\n  [TIER 1 PASSED] Best match: '{best_iris_user}' (Score: {best_iris_score:.2%})")
    else:
        print(f"\n  [TIER 1 FAILED] Best match: '{best_iris_user}' (Score: {best_iris_score:.2%})")
        print(f"  [SECURITY] No iris match found above threshold ({IRIS_THRESHOLD:.0%}).")

    time.sleep(1)

    # ── TIER 2: Voice Scan ───────────────────────────────────────────
    print_tier_header(2, "VOICE SCAN")
    input("  Press ENTER when ready to speak your passphrase...")

    live_voice = capture_voice()
    if live_voice is None:
        print("\n  [ERROR] Voice scan failed. Authentication aborted.")
        input("\n  Press ENTER to continue...")
        return

    # Match voice against all enrolled users
    print("\n  [SYSTEM] Matching voice against enrolled users...")
    voice_scores = {}
    for user in users:
        data = load_user_data(user)
        if data is not None:
            _, stored_voice, _ = data
            distance = dtw_distance(live_voice.T, stored_voice.T)
            voice_scores[user] = distance
            print(f"    - {user}: DTW distance = {distance:.2f}")

    if not voice_scores:
        print("  [ERROR] Could not load any user data.")
        input("\n  Press ENTER to continue...")
        return

    # Find best voice match (lowest distance)
    best_voice_user = min(voice_scores, key=voice_scores.get)
    best_voice_distance = voice_scores[best_voice_user]
    voice_passed = best_voice_distance <= VOICE_THRESHOLD

    if voice_passed:
        print(f"\n  [TIER 2 PASSED] Best match: '{best_voice_user}' (Distance: {best_voice_distance:.2f})")
    else:
        print(f"\n  [TIER 2 FAILED] Best match: '{best_voice_user}' (Distance: {best_voice_distance:.2f})")
        print(f"  [SECURITY] No voice match found below threshold ({VOICE_THRESHOLD}).")

    time.sleep(1)

    # ── TIER 3: Gesture Scan ─────────────────────────────────────────
    print_tier_header(3, "HAND GESTURE SCAN")
    input("  Press ENTER when ready to show your gesture...")

    live_gesture = capture_gesture()
    if live_gesture is None:
        print("\n  [ERROR] Gesture scan failed. Authentication aborted.")
        input("\n  Press ENTER to continue...")
        return

    # Match gesture against all enrolled users
    print("\n  [SYSTEM] Matching gesture against enrolled users...")
    gesture_scores = {}
    for user in users:
        data = load_user_data(user)
        if data is not None:
            _, _, stored_gesture = data
            score = cosine_similarity(stored_gesture, live_gesture)
            gesture_scores[user] = score
            print(f"    - {user}: {score:.2%}")

    if not gesture_scores:
        print("  [ERROR] Could not load any user data.")
        input("\n  Press ENTER to continue...")
        return

    # Find best gesture match
    best_gesture_user = max(gesture_scores, key=gesture_scores.get)
    best_gesture_score = gesture_scores[best_gesture_user]
    gesture_passed = best_gesture_score >= GESTURE_THRESHOLD

    if gesture_passed:
        print(f"\n  [TIER 3 PASSED] Best match: '{best_gesture_user}' (Score: {best_gesture_score:.2%})")
    else:
        print(f"\n  [TIER 3 FAILED] Best match: '{best_gesture_user}' (Score: {best_gesture_score:.2%})")
        print(f"  [SECURITY] No gesture match found above threshold ({GESTURE_THRESHOLD:.0%}).")

    time.sleep(1)

    # ── Final Decision ───────────────────────────────────────────────
    # All 3 tiers must pass AND the best match must be the SAME user
    all_passed = iris_passed and voice_passed and gesture_passed

    # Check if all tiers agree on the same user
    if all_passed:
        if best_iris_user == best_voice_user == best_gesture_user:
            identified_user = best_iris_user
        else:
            # Tiers identified different users — security conflict
            print("\n  [SECURITY WARNING] Biometric tiers matched DIFFERENT users!")
            print(f"    Iris  → {best_iris_user}")
            print(f"    Voice → {best_voice_user}")
            print(f"    Gesture → {best_gesture_user}")
            all_passed = False
            identified_user = "CONFLICT"
    else:
        # Use the most common match or best iris match for display
        identified_user = best_iris_user

    # Build scores summary
    scores = {
        "Tier 1 (Iris)": (iris_passed, f"Best: '{best_iris_user}' — Score: {best_iris_score:.2%}"),
        "Tier 2 (Voice)": (voice_passed, f"Best: '{best_voice_user}' — Distance: {best_voice_distance:.2f}"),
        "Tier 3 (Gesture)": (gesture_passed, f"Best: '{best_gesture_user}' — Score: {best_gesture_score:.2%}"),
    }

    # Log the authentication result
    auth_scores = {
        "iris": f"{best_iris_score:.2%}",
        "voice_dist": f"{best_voice_distance:.2f}",
        "gesture": f"{best_gesture_score:.2%}",
    }
    if all_passed:
        log_authentication(identified_user, True, auth_scores)
    else:
        record_failure()
        log_authentication(identified_user, False, auth_scores)

    show_result_screen(all_passed, identified_user, scores)
    input("  Press ENTER to continue...")


def list_users():
    """List all enrolled users."""
    clear_screen()
    print_banner("ENROLLED USERS")

    users = get_all_users()
    if not users:
        print("\n  No users enrolled yet.")
    else:
        print(f"\n  Total enrolled users: {len(users)}\n")
        for i, user in enumerate(users, 1):
            print(f"  {i}. {user}")

    print()
    input("  Press ENTER to continue...")


def delete_user_flow():
    """Delete an enrolled user."""
    clear_screen()
    print_banner("DELETE USER")
    print()

    username = input("  Enter username to delete: ").strip()
    if not username:
        return

    # Validate username
    safe_name = validate_username(username)
    if safe_name is None:
        print("\n  [ERROR] Invalid username.")
        input("\n  Press ENTER to continue...")
        return

    if not user_exists(safe_name):
        print(f"\n  [ERROR] User '{safe_name}' not found.")
        input("\n  Press ENTER to continue...")
        return

    confirm = input(f"  Are you sure you want to delete '{safe_name}'? (y/n): ").strip().lower()
    if confirm == 'y':
        delete_user(safe_name)
        log_user_deleted(safe_name, deleted_by="console")
        print(f"  User '{safe_name}' has been deleted.")
    else:
        print("  Deletion cancelled.")

    input("\n  Press ENTER to continue...")


def main():
    """Main application loop."""
    while True:
        show_main_menu()
        choice = input("  Select option [1-5]: ").strip()

        if choice == '1':
            enroll_user()
        elif choice == '2':
            authenticate_user()
        elif choice == '3':
            list_users()
        elif choice == '4':
            delete_user_flow()
        elif choice == '5':
            clear_screen()
            print_banner("GOODBYE!", char="*")
            print("\n  Security System shutting down...\n")
            sys.exit(0)
        else:
            print("\n  [ERROR] Invalid option. Please select 1-5.")
            time.sleep(1)


if __name__ == "__main__":
    main()
