"""
=============================================================================
  FLASK WEB APPLICATION — Three-Tier Biometric Security System Frontend
  Serves the web UI and provides API endpoints for biometric operations.
=============================================================================
"""

import sys
import os
import json
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, request
from database import user_exists, save_user_data, load_user_data, get_all_users, delete_user
from utils import cosine_similarity, dtw_distance

app = Flask(__name__)

# Global state for async biometric operations
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
        }


def run_enrollment(username):
    """Run enrollment in a background thread."""
    global operation_status
    try:
        with status_lock:
            operation_status["active"] = True
            operation_status["operation"] = "enroll"
            operation_status["username"] = username

        # ── TIER 1: Iris ──
        with status_lock:
            operation_status["current_tier"] = 1
            operation_status["message"] = "Scanning iris... Look straight at the camera."
            operation_status["tier_status"]["tier1"] = {"status": "running", "detail": "Capturing iris data..."}

        from iris_auth import enroll_iris
        iris_features = enroll_iris()

        if iris_features is None:
            with status_lock:
                operation_status["tier_status"]["tier1"] = {"status": "failed", "detail": "Iris capture failed or was cancelled."}
                operation_status["complete"] = True
                operation_status["result"] = "failed"
                operation_status["error"] = "Iris enrollment failed."
            return

        with status_lock:
            operation_status["tier_status"]["tier1"] = {"status": "passed", "detail": "Iris data captured successfully."}

        # ── TIER 2: Voice ──
        with status_lock:
            operation_status["current_tier"] = 2
            operation_status["message"] = "Recording voice... Speak your passphrase."
            operation_status["tier_status"]["tier2"] = {"status": "running", "detail": "Recording voice samples..."}

        from voice_auth import enroll_voice
        voice_features = enroll_voice()

        if voice_features is None:
            with status_lock:
                operation_status["tier_status"]["tier2"] = {"status": "failed", "detail": "Voice capture failed."}
                operation_status["complete"] = True
                operation_status["result"] = "failed"
                operation_status["error"] = "Voice enrollment failed."
            return

        with status_lock:
            operation_status["tier_status"]["tier2"] = {"status": "passed", "detail": "Voice data captured successfully."}

        # ── TIER 3: Gesture ──
        with status_lock:
            operation_status["current_tier"] = 3
            operation_status["message"] = "Scanning gesture... Show your hand gesture."
            operation_status["tier_status"]["tier3"] = {"status": "running", "detail": "Capturing gesture data..."}

        from gesture_auth import enroll_gesture
        gesture_features = enroll_gesture()

        if gesture_features is None:
            with status_lock:
                operation_status["tier_status"]["tier3"] = {"status": "failed", "detail": "Gesture capture failed."}
                operation_status["complete"] = True
                operation_status["result"] = "failed"
                operation_status["error"] = "Gesture enrollment failed."
            return

        with status_lock:
            operation_status["tier_status"]["tier3"] = {"status": "passed", "detail": "Gesture data captured successfully."}

        # ── Save ──
        save_user_data(username, iris_features, voice_features, gesture_features)

        with status_lock:
            operation_status["complete"] = True
            operation_status["result"] = "success"
            operation_status["message"] = f"User '{username}' enrolled successfully!"

    except Exception as e:
        with status_lock:
            operation_status["complete"] = True
            operation_status["result"] = "failed"
            operation_status["error"] = str(e)


def run_authentication():
    """Run authentication in a background thread."""
    global operation_status
    try:
        with status_lock:
            operation_status["active"] = True
            operation_status["operation"] = "authenticate"

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

        from iris_auth import capture_iris
        live_iris = capture_iris()

        if live_iris is None:
            with status_lock:
                operation_status["tier_status"]["tier1"] = {"status": "failed", "detail": "Iris scan failed or cancelled."}
                operation_status["complete"] = True
                operation_status["result"] = "failed"
                operation_status["error"] = "Iris scan failed."
            return

        # Match against all users
        iris_scores = {}
        for user in users:
            data = load_user_data(user)
            if data is not None:
                stored_iris, _, _ = data
                score = cosine_similarity(stored_iris, live_iris)
                iris_scores[user] = score

        best_iris_user = max(iris_scores, key=iris_scores.get)
        best_iris_score = iris_scores[best_iris_user]
        iris_passed = best_iris_score >= 0.75

        with status_lock:
            status = "passed" if iris_passed else "failed"
            operation_status["tier_status"]["tier1"] = {
                "status": status,
                "detail": f"Best match: {best_iris_user} ({best_iris_score:.1%})",
                "score": round(best_iris_score * 100, 1),
                "matched_user": best_iris_user,
            }

        # ── TIER 2: Voice Scan ──
        with status_lock:
            operation_status["current_tier"] = 2
            operation_status["message"] = "Recording voice... Speak your passphrase."
            operation_status["tier_status"]["tier2"] = {"status": "running", "detail": "Recording voice..."}

        from voice_auth import capture_voice
        live_voice = capture_voice()

        if live_voice is None:
            with status_lock:
                operation_status["tier_status"]["tier2"] = {"status": "failed", "detail": "Voice scan failed."}
                operation_status["complete"] = True
                operation_status["result"] = "failed"
                operation_status["error"] = "Voice scan failed."
            return

        voice_scores = {}
        for user in users:
            data = load_user_data(user)
            if data is not None:
                _, stored_voice, _ = data
                distance = dtw_distance(live_voice.T, stored_voice.T)
                voice_scores[user] = distance

        best_voice_user = min(voice_scores, key=voice_scores.get)
        best_voice_distance = voice_scores[best_voice_user]
        voice_passed = best_voice_distance <= 80.0

        with status_lock:
            status = "passed" if voice_passed else "failed"
            operation_status["tier_status"]["tier2"] = {
                "status": status,
                "detail": f"Best match: {best_voice_user} (dist: {best_voice_distance:.1f})",
                "score": round(max(0, 100 - best_voice_distance * 2), 1),
                "matched_user": best_voice_user,
            }

        # ── TIER 3: Gesture Scan ──
        with status_lock:
            operation_status["current_tier"] = 3
            operation_status["message"] = "Scanning gesture... Show your hand gesture."
            operation_status["tier_status"]["tier3"] = {"status": "running", "detail": "Scanning gesture..."}

        from gesture_auth import capture_gesture
        live_gesture = capture_gesture()

        if live_gesture is None:
            with status_lock:
                operation_status["tier_status"]["tier3"] = {"status": "failed", "detail": "Gesture scan failed."}
                operation_status["complete"] = True
                operation_status["result"] = "failed"
                operation_status["error"] = "Gesture scan failed."
            return

        gesture_scores = {}
        for user in users:
            data = load_user_data(user)
            if data is not None:
                _, _, stored_gesture = data
                score = cosine_similarity(stored_gesture, live_gesture)
                gesture_scores[user] = score

        best_gesture_user = max(gesture_scores, key=gesture_scores.get)
        best_gesture_score = gesture_scores[best_gesture_user]
        gesture_passed = best_gesture_score >= 0.80

        with status_lock:
            status = "passed" if gesture_passed else "failed"
            operation_status["tier_status"]["tier3"] = {
                "status": status,
                "detail": f"Best match: {best_gesture_user} ({best_gesture_score:.1%})",
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

        with status_lock:
            operation_status["complete"] = True
            operation_status["result"] = "granted" if all_passed else "denied"
            operation_status["username"] = identified_user
            operation_status["message"] = (
                f"Access GRANTED — Welcome, {identified_user}!" if all_passed
                else "Access DENIED — Biometric verification failed."
            )

    except Exception as e:
        with status_lock:
            operation_status["complete"] = True
            operation_status["result"] = "failed"
            operation_status["error"] = str(e)


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/users")
def api_list_users():
    users = get_all_users()
    return jsonify({"users": users, "count": len(users)})


@app.route("/api/delete_user", methods=["POST"])
def api_delete_user():
    data = request.json
    username = data.get("username", "").strip()
    if not username:
        return jsonify({"success": False, "error": "Username required."})
    if not user_exists(username):
        return jsonify({"success": False, "error": f"User '{username}' not found."})
    delete_user(username)
    return jsonify({"success": True, "message": f"User '{username}' deleted."})


@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    data = request.json
    username = data.get("username", "").strip().lower()
    if not username:
        return jsonify({"success": False, "error": "Username is required."})
    if operation_status["active"]:
        return jsonify({"success": False, "error": "Another operation is already running."})

    reset_status()
    thread = threading.Thread(target=run_enrollment, args=(username,), daemon=True)
    thread.start()
    return jsonify({"success": True, "message": "Enrollment started."})


@app.route("/api/authenticate", methods=["POST"])
def api_authenticate():
    if operation_status["active"]:
        return jsonify({"success": False, "error": "Another operation is already running."})

    reset_status()
    thread = threading.Thread(target=run_authentication, daemon=True)
    thread.start()
    return jsonify({"success": True, "message": "Authentication started."})


@app.route("/api/status")
def api_status():
    with status_lock:
        return jsonify(operation_status)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    reset_status()
    return jsonify({"success": True})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  THREE-TIER BIOMETRIC SECURITY SYSTEM")
    print("  Web Frontend — http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=False, port=5000, threaded=True)
