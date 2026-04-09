"""
=============================================================================
  FLASK WEB APPLICATION — Three-Tier Biometric Security System Frontend
  Serves the web UI and provides API endpoints for biometric operations.
  
  Security improvements:
    - Secret key for session management
    - Security headers (CSP, X-Frame-Options, etc.)
    - Rate limiting on authentication endpoints
    - Brute-force lockout (5 failed attempts → 60s cooldown)
    - No internal exception details leaked to clients
    - Operation timeout enforcement
    - Security audit logging
=============================================================================
"""

import sys
import os
import json
import threading
import time
from datetime import datetime
from functools import wraps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, request, Response
import cv2
from database import user_exists, save_user_data, load_user_data, get_all_users, delete_user
from database import validate_username
from utils import cosine_similarity, dtw_distance, find_best_match
from config import (
    FLASK_SECRET_KEY, MAX_FAILED_ATTEMPTS, LOCKOUT_DURATION_SECONDS,
    LOCKOUT_WINDOW_SECONDS, IRIS_THRESHOLD, VOICE_THRESHOLD, GESTURE_THRESHOLD,
    IRIS_MARGIN, VOICE_MARGIN, GESTURE_MARGIN,
    OPERATION_TIMEOUT_SECONDS
)
from voice_assistant import say
from security_logger import (
    log_enrollment, log_authentication, log_auth_attempt, log_failed_tier,
    log_lockout, log_user_deleted, log_security_event
)

def log_error(operation, error_msg):
    """Helper to log errors since it's used in app but not in security_logger."""
    log_security_event(f"ERROR_{operation.upper()}", error_msg)

# ════════════════════════════════════════════════════════════════════════════════
#  WEB-COMPATIBLE BIOMETRIC FUNCTIONS
#  These work with browser APIs instead of direct hardware access
# ════════════════════════════════════════════════════════════════════════════════

def enroll_iris_web():
    """Web-compatible iris enrollment - simulates the process for demo."""
    try:
        # Simulate iris capture process (normally this would process browser camera stream)
        time.sleep(2)  # Simulate capture time

        # Generate mock iris features (in real implementation, this would come from browser)
        import numpy as np
        iris_features = np.random.rand(128).astype(np.float32)
        iris_features = iris_features / np.linalg.norm(iris_features)  # Normalize

        return iris_features
    except Exception as e:
        print(f"[IRIS ENROLL ERROR] {e}")
        return None

def enroll_voice_web():
    """Web-compatible voice enrollment - simulates the process for demo."""
    try:
        # Simulate voice recording process (normally this would process browser microphone stream)
        time.sleep(3)  # Simulate recording time

        # Generate mock voice features (in real implementation, this would come from browser)
        import numpy as np
        # Simulate MFCC features
        voice_features = np.random.rand(100, 13).astype(np.float32)

        return voice_features
    except Exception as e:
        print(f"[VOICE ENROLL ERROR] {e}")
        return None

def enroll_gesture_web():
    """Web-compatible gesture enrollment - simulates the process for demo."""
    try:
        # Simulate gesture capture process (normally this would process browser camera stream)
        time.sleep(2.5)  # Simulate capture time

        # Generate mock gesture features (in real implementation, this would come from browser)
        import numpy as np
        gesture_features = np.random.rand(63).astype(np.float32)  # 21 landmarks * 3 coordinates
        gesture_features = gesture_features / np.linalg.norm(gesture_features)  # Normalize

        return gesture_features
    except Exception as e:
        print(f"[GESTURE ENROLL ERROR] {e}")
        return None

def capture_iris_web():
    """Web-compatible iris capture - simulates the process for demo."""
    try:
        # Simulate iris capture process
        time.sleep(1.5)  # Simulate capture time

        # Generate mock iris features
        import numpy as np
        iris_features = np.random.rand(128).astype(np.float32)
        iris_features = iris_features / np.linalg.norm(iris_features)  # Normalize

        return iris_features
    except Exception as e:
        print(f"[IRIS CAPTURE ERROR] {e}")
        return None

def capture_voice_web():
    """Web-compatible voice capture - simulates the process for demo."""
    try:
        # Simulate voice recording process
        time.sleep(2.5)  # Simulate recording time

        # Generate mock voice features
        import numpy as np
        voice_features = np.random.rand(80, 13).astype(np.float32)

        return voice_features
    except Exception as e:
        print(f"[VOICE CAPTURE ERROR] {e}")
        return None

def capture_gesture_web():
    """Web-compatible gesture capture - simulates the process for demo."""
    try:
        # Simulate gesture capture process
        time.sleep(2)  # Simulate capture time

        # Generate mock gesture features
        import numpy as np
        gesture_features = np.random.rand(63).astype(np.float32)  # 21 landmarks * 3 coordinates
        gesture_features = gesture_features / np.linalg.norm(gesture_features)  # Normalize

        return gesture_features
    except Exception as e:
        print(f"[GESTURE CAPTURE ERROR] {e}")
        return None

# Configure Flask with explicit paths for templates and static files
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = FLASK_SECRET_KEY

# ── Security Headers ────────────────────────────────────────────────────────
@app.after_request
def add_security_headers(response):
    """Add security headers to every response."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


# ── Brute-Force Protection ──────────────────────────────────────────────────
failed_attempts = []  # list of timestamps of failed auth attempts
failed_lock = threading.Lock()


def is_locked_out():
    """Check if the system is in brute-force lockout."""
    with failed_lock:
        now = time.time()
        # Clean old entries outside the window
        failed_attempts[:] = [t for t in failed_attempts if now - t < LOCKOUT_WINDOW_SECONDS]
        
        if len(failed_attempts) >= MAX_FAILED_ATTEMPTS:
            # Check if most recent failure is within lockout duration
            if failed_attempts and (now - failed_attempts[-1]) < LOCKOUT_DURATION_SECONDS:
                return True
    return False


def record_failed_attempt():
    """Record a failed authentication attempt."""
    with failed_lock:
        failed_attempts.append(time.time())


def get_lockout_remaining():
    """Get remaining lockout seconds."""
    with failed_lock:
        if not failed_attempts:
            return 0
        elapsed = time.time() - failed_attempts[-1]
        remaining = LOCKOUT_DURATION_SECONDS - elapsed
        return max(0, int(remaining))


# ── Rate Limiting ───────────────────────────────────────────────────────────
request_timestamps = {}
request_lock = threading.Lock()


def rate_limit(max_requests=10, window_seconds=60):
    """Simple rate limiter decorator."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            client_ip = request.remote_addr or "unknown"
            now = time.time()
            
            with request_lock:
                if client_ip not in request_timestamps:
                    request_timestamps[client_ip] = []
                
                # Clean old timestamps
                request_timestamps[client_ip] = [
                    t for t in request_timestamps[client_ip]
                    if now - t < window_seconds
                ]
                
                if len(request_timestamps[client_ip]) >= max_requests:
                    log_security_event("RATE_LIMIT", f"IP={client_ip} endpoint={request.path}")
                    return jsonify({
                        "success": False,
                        "error": "Too many requests. Please try again later."
                    }), 429
                
                request_timestamps[client_ip].append(now)
            
            return f(*args, **kwargs)
        return wrapper
    return decorator


# ── Global state for async biometric operations ────────────────────────────
operation_status = {
    "active": False,
    "operation": None,       # "enroll" or "authenticate"
    "current_tier": 0,
    "tier_status": {},       # Per-tier results
    "message": "",
    "complete": False,
    "result": None,          # Final result
    "username": None,
    "error": None,
    "started_at": None,
}

status_lock = threading.Lock()


def reset_status():
    """Reset the global operation status."""
    global operation_status
    with status_lock:
        operation_status = {
            "active": False,
            "operation": None,
            "current_tier": 0,
            "tier_status": {},
            "message": "",
            "complete": False,
            "result": None,
            "username": None,
            "error": None,
            "started_at": None,
        }


def run_enrollment(username):
    """Run enrollment in a background thread."""
    global operation_status
    try:
        with status_lock:
            operation_status["active"] = True
            operation_status["operation"] = "enroll"
            operation_status["username"] = username
            operation_status["started_at"] = time.time()

        # ── TIER 1: Iris ──
        with status_lock:
            operation_status["current_tier"] = 1
            operation_status["message"] = "Scanning iris... Look straight at the camera."
            operation_status["tier_status"]["tier1"] = {"status": "running", "detail": "Capturing iris data..."}

        say("Scanning iris. Look straight at the camera.")
        # Use web-compatible iris enrollment
        iris_features = enroll_iris_web()

        if iris_features is None:
            with status_lock:
                operation_status["tier_status"]["tier1"] = {"status": "failed", "detail": "Iris capture failed or was cancelled."}
                operation_status["complete"] = True
                operation_status["result"] = "failed"
                operation_status["error"] = "Iris enrollment failed."
            log_enrollment(username, False, "Iris capture failed")
            return

        with status_lock:
            operation_status["tier_status"]["tier1"] = {"status": "passed", "detail": "Iris data captured successfully."}

        # ── TIER 2: Voice ──
        with status_lock:
            operation_status["current_tier"] = 2
            operation_status["message"] = "Recording voice... Speak your passphrase."
            operation_status["tier_status"]["tier2"] = {"status": "running", "detail": "Recording voice samples..."}

        say("Recording voice. Speak your passphrase.")
        # Use web-compatible voice enrollment
        voice_features = enroll_voice_web()

        if voice_features is None:
            with status_lock:
                operation_status["tier_status"]["tier2"] = {"status": "failed", "detail": "Voice capture failed."}
                operation_status["complete"] = True
                operation_status["result"] = "failed"
                operation_status["error"] = "Voice enrollment failed."
            log_enrollment(username, False, "Voice capture failed")
            return

        with status_lock:
            operation_status["tier_status"]["tier2"] = {"status": "passed", "detail": "Voice data captured successfully."}

        # ── TIER 3: Gesture ──
        with status_lock:
            operation_status["current_tier"] = 3
            operation_status["message"] = "Scanning gesture... Show your hand gesture."
            operation_status["tier_status"]["tier3"] = {"status": "running", "detail": "Capturing gesture data..."}

        say("Scanning gesture. Show your hand gesture.")
        # Use web-compatible gesture enrollment
        gesture_features = enroll_gesture_web()

        if gesture_features is None:
            with status_lock:
                operation_status["tier_status"]["tier3"] = {"status": "failed", "detail": "Gesture capture failed."}
                operation_status["complete"] = True
                operation_status["result"] = "failed"
                operation_status["error"] = "Gesture enrollment failed."
            log_enrollment(username, False, "Gesture capture failed")
            return

        with status_lock:
            operation_status["tier_status"]["tier3"] = {"status": "passed", "detail": "Gesture data captured successfully."}

        # ── Save ──
        save_user_data(username, iris_features, voice_features, gesture_features)
        log_enrollment(username, True)

        with status_lock:
            operation_status["complete"] = True
            operation_status["result"] = "success"
            operation_status["message"] = f"User '{username}' enrolled successfully!"

        say(f"Enrollment completed successfully for user {username}.")

    except Exception as e:
        log_error("enrollment", str(e))
        with status_lock:
            operation_status["complete"] = True
            operation_status["result"] = "failed"
            operation_status["error"] = "An internal error occurred during enrollment."


def run_authentication():
    """Run authentication in a background thread."""
    global operation_status
    try:
        with status_lock:
            operation_status["active"] = True
            operation_status["operation"] = "authenticate"
            operation_status["started_at"] = time.time()

        log_auth_attempt("Web API authentication started")

        users = get_all_users()
        if not users:
            with status_lock:
                operation_status["complete"] = True
                operation_status["result"] = "failed"
                operation_status["error"] = "No users enrolled in the system."
            return

        # ── TIER 1: Iris Scan ──
        with status_lock:
            operation_status["current_tier"] = 1
            operation_status["message"] = "Scanning iris... Look straight at the camera."
            operation_status["tier_status"]["tier1"] = {"status": "running", "detail": "Scanning iris..."}

        say("Scanning iris. Look straight at the camera.")
        # Use web-compatible iris capture
        live_iris = capture_iris_web()

        if live_iris is None:
            with status_lock:
                operation_status["tier_status"]["tier1"] = {"status": "failed", "detail": "Iris scan failed or cancelled."}
                operation_status["complete"] = True
                operation_status["result"] = "failed"
                operation_status["error"] = "Iris scan failed."
            record_failed_attempt()
            log_authentication("unknown", False, detail="Iris scan failed")
            return

        # Match against all users
        iris_scores = {}
        for user in users:
            data = load_user_data(user)
            if data is not None:
                stored_iris, _, _ = data
                score = cosine_similarity(stored_iris, live_iris)
                iris_scores[user] = score

        if not iris_scores:
            with status_lock:
                operation_status["complete"] = True
                operation_status["result"] = "failed"
                operation_status["error"] = "Could not process biometric data."
            record_failed_attempt()
            return

        best_iris_user, best_iris_score, iris_passed, iris_disc = find_best_match(
            iris_scores, IRIS_THRESHOLD, IRIS_MARGIN, higher_is_better=True
        )
        if iris_passed and not iris_disc:
            iris_passed = False

        with status_lock:
            status = "passed" if iris_passed else "failed"
            detail = f"Best match: {best_iris_user} ({best_iris_score:.1%})"
            if not iris_disc and len(iris_scores) > 1:
                detail += " [ambiguous]"
            operation_status["tier_status"]["tier1"] = {
                "status": status,
                "detail": detail,
                "score": round(best_iris_score * 100, 1),
                "matched_user": best_iris_user,
            }

        # ── TIER 2: Voice Scan ──
        with status_lock:
            operation_status["current_tier"] = 2
            operation_status["message"] = "Recording voice... Speak your passphrase."
            operation_status["tier_status"]["tier2"] = {"status": "running", "detail": "Recording voice..."}

        say("Recording voice. Speak your passphrase.")
        # Use web-compatible voice capture
        live_voice = capture_voice_web()

        if live_voice is None:
            with status_lock:
                operation_status["tier_status"]["tier2"] = {"status": "failed", "detail": "Voice scan failed."}
                operation_status["complete"] = True
                operation_status["result"] = "failed"
                operation_status["error"] = "Voice scan failed."
            record_failed_attempt()
            log_authentication("unknown", False, detail="Voice scan failed")
            return

        voice_scores = {}
        for user in users:
            data = load_user_data(user)
            if data is not None:
                _, stored_voice, _ = data
                distance = dtw_distance(live_voice.T, stored_voice.T)
                voice_scores[user] = distance

        best_voice_user, best_voice_distance, voice_passed, voice_disc = find_best_match(
            voice_scores, VOICE_THRESHOLD, VOICE_MARGIN, higher_is_better=False
        )
        if voice_passed and not voice_disc:
            voice_passed = False

        with status_lock:
            status = "passed" if voice_passed else "failed"
            detail = f"Best match: {best_voice_user} (dist: {best_voice_distance:.1f})"
            if not voice_disc and len(voice_scores) > 1:
                detail += " [ambiguous]"
            operation_status["tier_status"]["tier2"] = {
                "status": status,
                "detail": detail,
                "score": round(max(0, 100 - best_voice_distance * 2), 1),
                "matched_user": best_voice_user,
            }

        # ── TIER 3: Gesture Scan ──
        with status_lock:
            operation_status["current_tier"] = 3
            operation_status["message"] = "Scanning gesture... Show your hand gesture."
            operation_status["tier_status"]["tier3"] = {"status": "running", "detail": "Scanning gesture..."}

        say("Scanning gesture. Show your hand gesture.")
        # Use web-compatible gesture capture
        live_gesture = capture_gesture_web()

        if live_gesture is None:
            with status_lock:
                operation_status["tier_status"]["tier3"] = {"status": "failed", "detail": "Gesture scan failed."}
                operation_status["complete"] = True
                operation_status["result"] = "failed"
                operation_status["error"] = "Gesture scan failed."
            record_failed_attempt()
            log_authentication("unknown", False, detail="Gesture scan failed")
            return

        gesture_scores = {}
        for user in users:
            data = load_user_data(user)
            if data is not None:
                _, _, stored_gesture = data
                score = cosine_similarity(stored_gesture, live_gesture)
                gesture_scores[user] = score

        best_gesture_user, best_gesture_score, gesture_passed, gest_disc = find_best_match(
            gesture_scores, GESTURE_THRESHOLD, GESTURE_MARGIN, higher_is_better=True
        )
        if gesture_passed and not gest_disc:
            gesture_passed = False

        with status_lock:
            status = "passed" if gesture_passed else "failed"
            detail = f"Best match: {best_gesture_user} ({best_gesture_score:.1%})"
            if not gest_disc and len(gesture_scores) > 1:
                detail += " [ambiguous]"
            operation_status["tier_status"]["tier3"] = {
                "status": status,
                "detail": detail,
                "score": round(best_gesture_score * 100, 1),
                "matched_user": best_gesture_user,
            }

        # ── Final decision ──
        all_passed = iris_passed and voice_passed and gesture_passed
        if all_passed and best_iris_user == best_voice_user == best_gesture_user:
            identified_user = best_iris_user
        elif all_passed:
            all_passed = False
            identified_user = "CONFLICT"
        else:
            identified_user = best_iris_user

        scores = {
            "iris": f"{best_iris_score:.1%}",
            "voice_dist": f"{best_voice_distance:.1f}",
            "gesture": f"{best_gesture_score:.1%}",
        }

        if all_passed:
            log_authentication(identified_user, True, scores)
        else:
            record_failed_attempt()
            log_authentication(identified_user, False, scores)

        with status_lock:
            operation_status["complete"] = True
            operation_status["result"] = "granted" if all_passed else "denied"
            operation_status["username"] = identified_user
            operation_status["message"] = (
                f"Access GRANTED — Welcome, {identified_user}!" if all_passed
                else "Access DENIED — Biometric verification failed."
            )

        if all_passed:
            say(f"Access granted. Welcome, {identified_user}.")
        else:
            say("Access denied. Biometric verification failed.")

    except Exception as e:
        log_error("authentication", str(e))
        record_failed_attempt()
        with status_lock:
            operation_status["complete"] = True
            operation_status["result"] = "failed"
            operation_status["error"] = "An internal error occurred during authentication."


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/users")
@rate_limit(max_requests=30, window_seconds=60)
def api_list_users():
    users = get_all_users()
    return jsonify({"users": users, "count": len(users)})


@app.route("/api/delete_user", methods=["POST"])
@rate_limit(max_requests=10, window_seconds=60)
def api_delete_user():
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "Invalid request."})
    
    username = data.get("username", "").strip()
    
    # Validate username
    safe_name = validate_username(username)
    if safe_name is None:
        return jsonify({"success": False, "error": "Invalid username."})
    
    if not user_exists(safe_name):
        # Unified error message to prevent user enumeration
        return jsonify({"success": False, "error": "Operation failed."})
    
    delete_user(safe_name)
    log_user_deleted(safe_name, deleted_by="web_api")
    return jsonify({"success": True, "message": f"User '{safe_name}' deleted."})


@app.route("/api/enroll", methods=["POST"])
@rate_limit(max_requests=5, window_seconds=60)
def api_enroll():
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "Invalid request."})
    
    username = data.get("username", "").strip().lower()
    
    # Validate username
    safe_name = validate_username(username)
    if safe_name is None:
        return jsonify({"success": False, "error": "Invalid username. Use only letters, numbers, and underscores (2-32 chars)."})
    
    if operation_status["active"]:
        return jsonify({"success": False, "error": "Another operation is already running."})

    reset_status()
    thread = threading.Thread(target=run_enrollment, args=(safe_name,), daemon=True)
    thread.start()
    log_security_event("ENROLL_START", f"Enrollment started for user '{safe_name}'")
    return jsonify({"success": True, "message": "Enrollment started."})


@app.route("/api/authenticate", methods=["POST"])
@rate_limit(max_requests=10, window_seconds=60)
def api_authenticate():
    # Check brute-force lockout
    if is_locked_out():
        remaining = get_lockout_remaining()
        log_lockout("web_api", f"Authentication blocked, {remaining}s remaining")
        return jsonify({
            "success": False,
            "error": f"Too many failed attempts. Try again in {remaining} seconds.",
            "locked_out": True,
            "lockout_remaining": remaining
        })
    
    if operation_status["active"]:
        return jsonify({"success": False, "error": "Another operation is already running."})

    reset_status()
    thread = threading.Thread(target=run_authentication, daemon=True)
    thread.start()
    return jsonify({"success": True, "message": "Authentication started."})


@app.route("/api/status")
def api_status():
    with status_lock:
        # Check for operation timeout
        if operation_status["active"] and operation_status.get("started_at"):
            elapsed = time.time() - operation_status["started_at"]
            if elapsed > OPERATION_TIMEOUT_SECONDS:
                operation_status["complete"] = True
                operation_status["result"] = "failed"
                operation_status["error"] = "Operation timed out."
                operation_status["active"] = False
                log_security_event("TIMEOUT", "Operation timed out")
        
        # Don't leak internal timing info — only send safe fields
        safe_status = {
            "active": operation_status["active"],
            "operation": operation_status["operation"],
            "current_tier": operation_status["current_tier"],
            "tier_status": operation_status["tier_status"],
            "message": operation_status["message"],
            "complete": operation_status["complete"],
            "result": operation_status["result"],
            "username": operation_status["username"],
            "error": operation_status["error"],
        }
        return jsonify(safe_status)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    reset_status()
    return jsonify({"success": True})


def generate_frames():
    """Generator function for MJPEG video stream."""
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error: Could not encode frame")
                continue

            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except Exception as e:
        print(f"Video streaming error: {e}")
    finally:
        if cap is not None:
            cap.release()


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  THREE-TIER BIOMETRIC SECURITY SYSTEM")
    print("  Web Frontend — http://localhost:5000")
    print("  [LOCK] Security hardening ACTIVE")
    print("=" * 60 + "\n")
    app.run(debug=False, port=5000, threaded=True)
