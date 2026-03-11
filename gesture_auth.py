"""
=============================================================================
  TIER 3: HAND GESTURE AUTHENTICATION MODULE
  Uses MediaPipe Hands for hand landmark detection.
  Enrollment: User holds a specific hand gesture, landmarks are recorded.
  Authentication: User recreates the gesture, landmarks are compared.
=============================================================================
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Colors (BGR)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
DARK_BG = (30, 30, 30)
ORANGE = (0, 165, 255)

# Finger tip landmark indices
FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
FINGER_PIPS = [3, 6, 10, 14, 18]  # PIP joints (for finger up/down detection)
FINGER_MCPS = [2, 5, 9, 13, 17]   # MCP joints

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def get_landmark_array(hand_landmarks):
    """
    Convert hand landmarks to a normalized numpy array.
    Normalizes relative to the wrist (landmark 0) and hand scale.
    """
    landmarks = hand_landmarks.landmark
    coords = []
    for lm in landmarks:
        coords.append([lm.x, lm.y, lm.z])
    coords = np.array(coords)

    # Normalize: translate so wrist is at origin
    wrist = coords[0].copy()
    coords -= wrist

    # Scale normalization: divide by distance from wrist to middle finger MCP
    scale = np.linalg.norm(coords[9])  # Middle finger MCP
    if scale > 0:
        coords /= scale

    return coords.flatten()


def get_finger_states(hand_landmarks):
    """
    Detect which fingers are extended (up) or folded (down).
    Returns a list of 5 booleans [thumb, index, middle, ring, pinky].
    """
    landmarks = hand_landmarks.landmark

    states = []

    # Thumb: compare x position of tip vs IP joint
    # (works for right hand; flip for left)
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    # Use x-direction comparison (simplified)
    states.append(abs(thumb_tip.x - thumb_mcp.x) > abs(thumb_ip.x - thumb_mcp.x))

    # Other fingers: tip.y < pip.y means finger is extended (y increases downward)
    for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        states.append(landmarks[tip_idx].y < landmarks[pip_idx].y)

    return states


def get_gesture_name(finger_states):
    """Get a human-readable name for common gestures."""
    pattern = tuple(finger_states)

    gestures = {
        (False, False, False, False, False): "Fist",
        (True, True, True, True, True): "Open Hand",
        (False, True, False, False, False): "Pointing",
        (False, True, True, False, False): "Peace / Victory",
        (True, True, False, False, True): "Rock On",
        (True, False, False, False, True): "Hang Loose",
        (False, False, False, False, True): "Pinky",
        (True, True, True, False, False): "Three",
        (False, True, True, True, True): "Four",
        (True, False, False, False, False): "Thumbs Up",
        (False, True, False, False, True): "Spider-Man",
        (False, True, True, True, False): "Three Middle",
    }

    return gestures.get(pattern, "Custom Gesture")


def extract_gesture_features(hand_landmarks):
    """
    Extract a comprehensive feature vector from hand landmarks.
    Includes:
      - Normalized landmark coordinates (63 values: 21 landmarks × 3)
      - Finger states (5 values)
      - Inter-finger distances (10 values: pairwise distances between fingertips)
      - Finger angles (5 values)

    Returns:
        numpy array of gesture features
    """
    features = []

    # 1. Normalized landmark positions
    landmark_array = get_landmark_array(hand_landmarks)
    features.extend(landmark_array)

    # 2. Finger extension states
    finger_states = get_finger_states(hand_landmarks)
    features.extend([1.0 if s else 0.0 for s in finger_states])

    # 3. Pairwise distances between fingertips
    landmarks = hand_landmarks.landmark
    for i in range(len(FINGER_TIPS)):
        for j in range(i + 1, len(FINGER_TIPS)):
            t1 = landmarks[FINGER_TIPS[i]]
            t2 = landmarks[FINGER_TIPS[j]]
            dist = math.sqrt((t1.x - t2.x) ** 2 + (t1.y - t2.y) ** 2 + (t1.z - t2.z) ** 2)
            features.append(dist)

    # 4. Finger curl angles (angle at PIP joint)
    for i in range(5):
        tip = landmarks[FINGER_TIPS[i]]
        pip = landmarks[FINGER_PIPS[i]]
        mcp = landmarks[FINGER_MCPS[i]]

        # Vectors
        v1 = np.array([mcp.x - pip.x, mcp.y - pip.y, mcp.z - pip.z])
        v2 = np.array([tip.x - pip.x, tip.y - pip.y, tip.z - pip.z])

        # Angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        features.append(angle)

    return np.array(features, dtype=np.float64)


def draw_gesture_ui(frame, hand_landmarks, fw, fh, mode, progress, status_text,
                    finger_states=None, gesture_name="", match_score=None):
    """Draw the hand gesture detection UI overlay."""
    # Draw hand landmarks using MediaPipe's drawing utility
    mp_drawing.draw_landmarks(
        frame, hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )

    # Top panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, 120), DARK_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    title = f"TIER 3: GESTURE {'ENROLLMENT' if mode == 'enroll' else 'VERIFICATION'}"
    cv2.putText(frame, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2, cv2.LINE_AA)
    cv2.putText(frame, status_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    # Show finger states
    if finger_states:
        fingers_text = "Fingers: " + " ".join(
            [f"{FINGER_NAMES[i]}:{'UP' if s else 'DN'}" for i, s in enumerate(finger_states)]
        )
        cv2.putText(frame, fingers_text, (15, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1, cv2.LINE_AA)

    if gesture_name:
        cv2.putText(frame, f"Gesture: {gesture_name}", (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = fw - 320, 75, 300, 15
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill = int(bar_w * min(progress, 1.0))
    bar_color = GREEN if mode == 'enroll' else CYAN
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    cv2.putText(frame, f"{int(progress * 100)}%", (bar_x - 40, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)

    if match_score is not None:
        score_text = f"Match: {match_score:.2%}"
        score_color = GREEN if match_score > 0.80 else RED
        cv2.putText(frame, score_text, (fw - 200, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, score_color, 2, cv2.LINE_AA)

    # Bottom bar
    cv2.rectangle(frame, (0, fh - 30), (fw, fh), (20, 20, 20), -1)
    cv2.putText(frame, "Hold your gesture steady | Q: Cancel",
                (15, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)


def enroll_gesture(frame_callback=None):
    """
    Capture hand gesture features for enrollment.
    The user holds a gesture and features are averaged over multiple frames.

    Args:
        frame_callback: Optional function(frame) to display frames in external GUI.

    Returns:
        numpy array of averaged gesture features, or None if cancelled.
    """
    print("\n  [GESTURE] Starting hand gesture enrollment...")
    print("  [GESTURE] Hold up your hand with a UNIQUE gesture.")
    print("  [GESTURE] Suggestions: Peace sign, Rock on, Three fingers, etc.")
    print("  [GESTURE] Keep your hand STEADY during capture.")
    print("  [GESTURE] Press 'Q' to cancel.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [GESTURE ERROR] Cannot open webcam!")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    collected_features = []
    target_samples = 30
    start_time = time.time()
    timeout = 30
    stable_gesture = None
    stability_count = 0

    while len(collected_features) < target_samples:
        ret, frame = cap.read()
        if not ret:
            break

        if time.time() - start_time > timeout:
            print("  [GESTURE] Timeout reached.")
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        progress = len(collected_features) / target_samples
        status = f"Capturing... ({len(collected_features)}/{target_samples} frames)"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            finger_states = get_finger_states(hand_landmarks)
            gesture_name = get_gesture_name(finger_states)

            # Check gesture stability (same gesture for multiple frames)
            current_pattern = tuple(finger_states)
            if current_pattern == stable_gesture:
                stability_count += 1
            else:
                stable_gesture = current_pattern
                stability_count = 1

            # Only capture when gesture is stable (held for at least 5 frames)
            if stability_count >= 5:
                features = extract_gesture_features(hand_landmarks)
                collected_features.append(features)
                status = f"Capturing '{gesture_name}'... ({len(collected_features)}/{target_samples})"
            else:
                status = f"Hold steady... (stabilizing {stability_count}/5)"

            draw_gesture_ui(frame, hand_landmarks, fw, fh, 'enroll',
                          progress, status, finger_states, gesture_name)
        else:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (fw, 80), DARK_BG, -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.putText(frame, "TIER 3: GESTURE ENROLLMENT", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2, cv2.LINE_AA)
            cv2.putText(frame, "No hand detected - show your gesture", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)

        if frame_callback:
            frame_callback(frame)
            time.sleep(0.03)
        else:
            cv2.imshow("Security System - Gesture Enrollment", frame)
            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
                print("  [GESTURE] Enrollment cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                hands.close()
                return None

    cap.release()
    if not frame_callback:
        cv2.destroyAllWindows()
    hands.close()

    if len(collected_features) < 10:
        print("  [GESTURE] Not enough samples. Try again.")
        return None

    avg_features = np.mean(collected_features, axis=0)
    gesture_name = get_gesture_name(
        [bool(f) for f in avg_features[63:68]]  # finger states from feature vector
    )
    print(f"  [GESTURE] Enrollment complete! Gesture: '{gesture_name}'")
    print(f"  [GESTURE] ({len(collected_features)} frames captured)")
    return avg_features


def capture_gesture(frame_callback=None):
    """
    Capture live hand gesture features WITHOUT comparing to any stored template.
    Used during authentication to scan against all enrolled users.

    Args:
        frame_callback: Optional function(frame) to display frames in external GUI.

    Returns:
        numpy array of averaged gesture features, or None if failed/cancelled.
    """
    print("\n  [GESTURE] Scanning hand gesture...")
    print("  [GESTURE] Show your hand gesture to the camera.")
    print("  [GESTURE] Press 'Q' to cancel.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [GESTURE ERROR] Cannot open webcam!")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    collected_features = []
    target_samples = 20
    start_time = time.time()
    timeout = 20

    while len(collected_features) < target_samples:
        ret, frame = cap.read()
        if not ret:
            break

        if time.time() - start_time > timeout:
            print("  [GESTURE] Timeout reached.")
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        progress = len(collected_features) / target_samples
        status = f"Scanning... ({len(collected_features)}/{target_samples} frames)"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            finger_states = get_finger_states(hand_landmarks)
            gesture_name = get_gesture_name(finger_states)

            features = extract_gesture_features(hand_landmarks)
            collected_features.append(features)

            status = f"Scanning '{gesture_name}'... ({len(collected_features)}/{target_samples})"
            draw_gesture_ui(frame, hand_landmarks, fw, fh, 'verify',
                          progress, status, finger_states, gesture_name)
        else:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (fw, 80), DARK_BG, -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.putText(frame, "TIER 3: GESTURE SCAN", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2, cv2.LINE_AA)
            cv2.putText(frame, "No hand detected - show your gesture", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)

        if frame_callback:
            frame_callback(frame)
            time.sleep(0.03)
        else:
            cv2.imshow("Security System - Gesture Scan", frame)
            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
                print("  [GESTURE] Scan cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                hands.close()
                return None

    cap.release()
    if not frame_callback:
        cv2.destroyAllWindows()
    hands.close()

    if len(collected_features) < 5:
        print("  [GESTURE] Not enough samples captured.")
        return None

    avg_features = np.mean(collected_features, axis=0)
    print(f"  [GESTURE] Scan complete! ({len(collected_features)} frames captured)")
    return avg_features


def verify_gesture(stored_features, threshold=0.80):
    """
    Verify hand gesture against stored template.

    Args:
        stored_features: numpy array of enrolled gesture features
        threshold: minimum cosine similarity for a match

    Returns:
        tuple: (passed: bool, score: float)
    """
    from utils import cosine_similarity

    print("\n  [GESTURE] Starting gesture verification...")
    print("  [GESTURE] Show the SAME gesture you used during enrollment.")
    print("  [GESTURE] Press 'Q' to cancel.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [GESTURE ERROR] Cannot open webcam!")
        return False, 0.0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    collected_features = []
    target_samples = 20
    best_score = 0.0
    start_time = time.time()
    timeout = 20

    while len(collected_features) < target_samples:
        ret, frame = cap.read()
        if not ret:
            break

        if time.time() - start_time > timeout:
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        progress = len(collected_features) / target_samples

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            finger_states = get_finger_states(hand_landmarks)
            gesture_name = get_gesture_name(finger_states)

            features = extract_gesture_features(hand_landmarks)
            collected_features.append(features)

            # Real-time score
            current_avg = np.mean(collected_features, axis=0)
            score = cosine_similarity(stored_features, current_avg)
            best_score = max(best_score, score)

            status = f"Verifying '{gesture_name}'... ({len(collected_features)}/{target_samples})"
            draw_gesture_ui(frame, hand_landmarks, fw, fh, 'verify',
                          progress, status, finger_states, gesture_name, score)
        else:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (fw, 80), DARK_BG, -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.putText(frame, "TIER 3: GESTURE VERIFICATION", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2, cv2.LINE_AA)
            cv2.putText(frame, "No hand detected - show your gesture", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)

        cv2.imshow("Security System - Gesture Verification", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            print("  [GESTURE] Verification cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            return False, 0.0

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    if len(collected_features) < 5:
        print("  [GESTURE] Not enough samples. Verification failed.")
        return False, 0.0

    avg_features = np.mean(collected_features, axis=0)
    final_score = cosine_similarity(stored_features, avg_features)

    passed = final_score >= threshold
    status = "PASSED" if passed else "FAILED"
    print(f"  [GESTURE] Verification {status} (Score: {final_score:.2%}, Threshold: {threshold:.0%})")

    return passed, final_score
