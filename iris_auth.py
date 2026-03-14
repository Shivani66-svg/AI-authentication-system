"""
=============================================================================
  TIER 1: IRIS AUTHENTICATION MODULE
  Uses MediaPipe Face Mesh with refined landmarks for iris detection.
  Enrollment: Captures iris feature vector from multiple frames.
  Authentication: Compares live iris features with stored template.
=============================================================================
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import time

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh

# Iris landmark indices (with refine_landmarks=True)
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

# Eye contour indices
LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                    173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249,
                     263, 466, 388, 387, 386, 385, 384, 398]

# Colors (BGR)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
DARK_BG = (30, 30, 30)
ORANGE = (0, 165, 255)


def get_landmark_coords(landmarks, index, fw, fh):
    """Convert normalized landmark to pixel coordinates."""
    lm = landmarks[index]
    return int(lm.x * fw), int(lm.y * fh)


def compute_iris_radius(landmarks, iris_indices, fw, fh):
    """Compute average iris radius."""
    cx, cy = get_landmark_coords(landmarks, iris_indices[0], fw, fh)
    radii = []
    for idx in iris_indices[1:]:
        px, py = get_landmark_coords(landmarks, idx, fw, fh)
        r = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        radii.append(r)
    return np.mean(radii) if radii else 0


def extract_iris_features(landmarks, fw, fh):
    """
    Extract a feature vector that characterizes a person's iris geometry.
    Features include:
      - Relative iris positions within the eye region
      - Iris radius ratios (left vs right)
      - Inter-iris distance ratio relative to face width
      - Eye aspect ratios
      - Iris-to-eye-width ratios
    Returns a numpy array of features.
    """
    features = []

    # Left iris center
    l_cx, l_cy = get_landmark_coords(landmarks, LEFT_IRIS_INDICES[0], fw, fh)
    # Right iris center
    r_cx, r_cy = get_landmark_coords(landmarks, RIGHT_IRIS_INDICES[0], fw, fh)

    # Iris radii
    l_radius = compute_iris_radius(landmarks, LEFT_IRIS_INDICES, fw, fh)
    r_radius = compute_iris_radius(landmarks, RIGHT_IRIS_INDICES, fw, fh)

    # Eye corners for left eye
    le_left = get_landmark_coords(landmarks, 33, fw, fh)
    le_right = get_landmark_coords(landmarks, 133, fw, fh)
    le_top = get_landmark_coords(landmarks, 159, fw, fh)
    le_bottom = get_landmark_coords(landmarks, 145, fw, fh)

    # Eye corners for right eye
    re_left = get_landmark_coords(landmarks, 362, fw, fh)
    re_right = get_landmark_coords(landmarks, 263, fw, fh)
    re_top = get_landmark_coords(landmarks, 386, fw, fh)
    re_bottom = get_landmark_coords(landmarks, 374, fw, fh)

    # Face reference points
    nose_tip = get_landmark_coords(landmarks, 1, fw, fh)
    face_left = get_landmark_coords(landmarks, 234, fw, fh)
    face_right = get_landmark_coords(landmarks, 454, fw, fh)
    face_width = max(math.dist(face_left, face_right), 1)

    # Left eye dimensions
    le_width = max(math.dist(le_left, le_right), 1)
    le_height = max(math.dist(le_top, le_bottom), 1)

    # Right eye dimensions
    re_width = max(math.dist(re_left, re_right), 1)
    re_height = max(math.dist(re_top, re_bottom), 1)

    # --- Feature extraction ---

    # 1. Iris radius ratio (left / right) — structural uniqueness
    radius_ratio = l_radius / max(r_radius, 0.001)
    features.append(radius_ratio)

    # 2. Left iris radius relative to left eye width
    features.append(l_radius / le_width)

    # 3. Right iris radius relative to right eye width
    features.append(r_radius / re_width)

    # 4. Left eye aspect ratio (height / width)
    features.append(le_height / le_width)

    # 5. Right eye aspect ratio
    features.append(re_height / re_width)

    # 6. Inter-iris distance relative to face width
    inter_iris = math.dist((l_cx, l_cy), (r_cx, r_cy))
    features.append(inter_iris / face_width)

    # 7. Left iris horizontal position ratio in eye
    features.append((l_cx - le_left[0]) / le_width)

    # 8. Right iris horizontal position ratio in eye
    features.append((r_cx - re_left[0]) / re_width)

    # 9. Left iris vertical position ratio in eye
    features.append((l_cy - le_top[1]) / max(le_height, 1))

    # 10. Right iris vertical position ratio in eye
    features.append((r_cy - re_top[1]) / max(re_height, 1))

    # 11. Left eye width / face width ratio
    features.append(le_width / face_width)

    # 12. Right eye width / face width ratio
    features.append(re_width / face_width)

    # 13-16. Relative iris perimeter landmark positions (left iris)
    for idx in LEFT_IRIS_INDICES[1:]:
        px, py = get_landmark_coords(landmarks, idx, fw, fh)
        features.append((px - l_cx) / max(l_radius, 1))
        features.append((py - l_cy) / max(l_radius, 1))

    # 17-24. Relative iris perimeter landmark positions (right iris)
    for idx in RIGHT_IRIS_INDICES[1:]:
        px, py = get_landmark_coords(landmarks, idx, fw, fh)
        features.append((px - r_cx) / max(r_radius, 1))
        features.append((py - r_cy) / max(r_radius, 1))

    return np.array(features, dtype=np.float64)


def draw_ui(frame, landmarks, fw, fh, mode, progress, status_text, match_score=None):
    """Draw the iris detection UI overlay."""
    # Draw eye contours
    for eye_indices, color in [(LEFT_EYE_INDICES, YELLOW), (RIGHT_EYE_INDICES, CYAN)]:
        pts = []
        for idx in eye_indices:
            x, y = get_landmark_coords(landmarks, idx, fw, fh)
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)

    # Draw iris circles
    for iris_indices in [LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES]:
        cx, cy = get_landmark_coords(landmarks, iris_indices[0], fw, fh)
        radius = int(compute_iris_radius(landmarks, iris_indices, fw, fh))
        cv2.circle(frame, (cx, cy), radius, MAGENTA, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 2, GREEN, -1, cv2.LINE_AA)

    # Top panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, 100), DARK_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    title = f"TIER 1: IRIS {'ENROLLMENT' if mode == 'enroll' else 'VERIFICATION'}"
    cv2.putText(frame, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2, cv2.LINE_AA)
    cv2.putText(frame, status_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = 15, 75, 300, 15
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill = int(bar_w * min(progress, 1.0))
    bar_color = GREEN if mode == 'enroll' else CYAN
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    cv2.putText(frame, f"{int(progress * 100)}%", (bar_x + bar_w + 10, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)

    if match_score is not None:
        score_text = f"Match Score: {match_score:.2%}"
        score_color = GREEN if match_score > 0.82 else RED
        cv2.putText(frame, score_text, (fw - 250, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, score_color, 2, cv2.LINE_AA)

    # Bottom bar
    cv2.rectangle(frame, (0, fh - 30), (fw, fh), (20, 20, 20), -1)
    cv2.putText(frame, "Look straight at the camera | Q: Cancel",
                (15, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)


def enroll_iris(frame_callback=None, cap=None):
    """
    Capture iris features for enrollment.
    Collects features over multiple frames and averages them for a stable template.

    Args:
        frame_callback: Optional function(frame) to display frames in external GUI.
                        If None, uses cv2.imshow.

    Returns:
        numpy array of averaged iris features, or None if cancelled.
    """
    print("\n  [IRIS] Starting iris enrollment...")
    print("  [IRIS] Please look straight at the camera.")
    print("  [IRIS] Keep your eyes open and hold still.")
    print("  [IRIS] Press 'Q' to cancel.\n")

    own_cap = cap is None
    if own_cap:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  [IRIS ERROR] Cannot open webcam!")
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    collected_features = []
    target_samples = 50  # Number of frames to average (more = more stable template)
    start_time = time.time()
    timeout = 30  # seconds

    while len(collected_features) < target_samples:
        ret, frame = cap.read()
        if not ret:
            break

        if time.time() - start_time > timeout:
            print("  [IRIS] Timeout reached.")
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        progress = len(collected_features) / target_samples
        status = f"Capturing... ({len(collected_features)}/{target_samples} frames)"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            features = extract_iris_features(landmarks, fw, fh)
            collected_features.append(features)
            draw_ui(frame, landmarks, fw, fh, 'enroll', progress, status)
        else:
            # No face panel
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (fw, 80), DARK_BG, -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.putText(frame, "TIER 1: IRIS ENROLLMENT", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2, cv2.LINE_AA)
            cv2.putText(frame, "No face detected - please look at camera", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)

        if frame_callback:
            frame_callback(frame)
            # Small delay to keep UI responsive
            time.sleep(0.03)
        else:
            cv2.imshow("Security System - Iris Enrollment", frame)
            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
                print("  [IRIS] Enrollment cancelled.")
                if own_cap:
                    cap.release()
                cv2.destroyAllWindows()
                face_mesh.close()
                return None

    if own_cap:
        cap.release()
    if not frame_callback:
        cv2.destroyAllWindows()
    face_mesh.close()

    if len(collected_features) < 10:
        print("  [IRIS] Not enough samples collected. Try again.")
        return None

    # Average the features for a stable template
    avg_features = np.mean(collected_features, axis=0)
    print(f"  [IRIS] Enrollment complete! ({len(collected_features)} frames captured)")
    return avg_features


def capture_iris(frame_callback=None, cap=None):
    """
    Capture live iris features WITHOUT comparing to any stored template.
    Used during authentication to scan against all enrolled users.

    Args:
        frame_callback: Optional function(frame) to display frames in external GUI.

    Returns:
        numpy array of averaged iris features, or None if failed/cancelled.
    """
    print("\n  [IRIS] Scanning iris...")
    print("  [IRIS] Please look straight at the camera.")
    print("  [IRIS] Press 'Q' to cancel.\n")

    own_cap = cap is None
    if own_cap:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  [IRIS ERROR] Cannot open webcam!")
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
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
            print("  [IRIS] Timeout reached.")
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        progress = len(collected_features) / target_samples
        status = f"Scanning... ({len(collected_features)}/{target_samples} frames)"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            features = extract_iris_features(landmarks, fw, fh)
            collected_features.append(features)
            draw_ui(frame, landmarks, fw, fh, 'verify', progress, status)
        else:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (fw, 80), DARK_BG, -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.putText(frame, "TIER 1: IRIS SCAN", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2, cv2.LINE_AA)
            cv2.putText(frame, "No face detected - please look at camera", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)

        if frame_callback:
            frame_callback(frame)
            time.sleep(0.03)
        else:
            cv2.imshow("Security System - Iris Scan", frame)
            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
                print("  [IRIS] Scan cancelled.")
                if own_cap:
                    cap.release()
                cv2.destroyAllWindows()
                face_mesh.close()
                return None

    if own_cap:
        cap.release()
    if not frame_callback:
        cv2.destroyAllWindows()
    face_mesh.close()

    if len(collected_features) < 5:
        print("  [IRIS] Not enough samples captured.")
        return None

    avg_features = np.mean(collected_features, axis=0)
    print(f"  [IRIS] Scan complete! ({len(collected_features)} frames captured)")
    return avg_features


def verify_iris(stored_features, threshold=0.82):
    """
    Verify iris against stored template.

    Args:
        stored_features: numpy array of enrolled iris features
        threshold: minimum cosine similarity for a match

    Returns:
        tuple: (passed: bool, score: float)
    """
    from utils import cosine_similarity

    print("\n  [IRIS] Starting iris verification...")
    print("  [IRIS] Please look straight at the camera.")
    print("  [IRIS] Press 'Q' to cancel.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [IRIS ERROR] Cannot open webcam!")
        return False, 0.0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
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
        results = face_mesh.process(rgb)

        progress = len(collected_features) / target_samples

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            features = extract_iris_features(landmarks, fw, fh)
            collected_features.append(features)

            # Real-time score
            current_avg = np.mean(collected_features, axis=0)
            score = cosine_similarity(stored_features, current_avg)
            best_score = max(best_score, score)

            status = f"Verifying... ({len(collected_features)}/{target_samples})"
            draw_ui(frame, landmarks, fw, fh, 'verify', progress, status, score)
        else:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (fw, 80), DARK_BG, -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.putText(frame, "TIER 1: IRIS VERIFICATION", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2, cv2.LINE_AA)
            cv2.putText(frame, "No face detected - please look at camera", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)

        cv2.imshow("Security System - Iris Verification", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            print("  [IRIS] Verification cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            face_mesh.close()
            return False, 0.0

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    if len(collected_features) < 5:
        print("  [IRIS] Not enough samples. Verification failed.")
        return False, 0.0

    avg_features = np.mean(collected_features, axis=0)
    final_score = cosine_similarity(stored_features, avg_features)

    passed = final_score >= threshold
    status = "PASSED" if passed else "FAILED"
    print(f"  [IRIS] Verification {status} (Score: {final_score:.2%}, Threshold: {threshold:.0%})")

    return passed, final_score
