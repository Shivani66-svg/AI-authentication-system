# 🔒 SECURITIX — Three-Tier AI Biometric Authentication System

A Python-based multi-layered biometric security system that uses **Iris Detection**, **Voice Recognition**, and **Hand Gesture Recognition** to authenticate users. The system requires all three biometric tiers to match the same user for access to be granted.

---

## 🎯 Key Features

- **Three-Tier Biometric Authentication** — Iris + Voice + Hand Gesture
- **Automatic User Identification** — No username required during authentication
- **Liveness Detection** — Blink-based anti-spoofing prevents photo/video attacks
- **Encrypted Biometric Storage** — AES-128-CBC encryption for all biometric data at rest
- **Brute-Force Protection** — 5 failed attempts trigger a 60-second lockout
- **Security Audit Logging** — Every event is timestamped and logged
- **Discrimination Margin** — Prevents misidentification when multiple users have similar biometrics
- **Desktop GUI** — Full Tkinter interface with live camera feed
- **Web Interface** — Flask-based web API with rate limiting
- **Console Interface** — Terminal-based menu for quick use
- **Voice Assistant** — Text-to-speech feedback during operations

---

## 🏗️ Architecture

### Security Tiers

| Tier | Biometric | Technology | Comparison Method | Threshold |
|------|-----------|------------|-------------------|-----------|
| 1 | **Iris Detection** | MediaPipe Face Mesh | Cosine similarity on iris geometry | ≥ 88% |
| 2 | **Voice Recognition** | librosa MFCC + DTW | Dynamic Time Warping distance | ≤ 45.0 |
| 3 | **Hand Gesture** | MediaPipe Hands | Cosine similarity on landmarks | ≥ 90% |

### Security Layers

```
┌──────────────────────────────────────────────────────────────┐
│  USER INPUT                                                  │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────┐   Username validation (alphanumeric)   │
│  │  Input Validation │──────────────────────────────────────▶│
│  └──────────────────┘                                        │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────┐   Blink detection via EAR variance     │
│  │ Liveness Check   │──────────────────────────────────────▶│
│  └──────────────────┘                                        │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────┐   3 tiers × threshold + margin check   │
│  │ Biometric Match  │──────────────────────────────────────▶│
│  └──────────────────┘                                        │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────┐   All 3 tiers must match SAME user     │
│  │  Identity Fusion │──────────────────────────────────────▶│
│  └──────────────────┘                                        │
│       │                                                      │
│       ▼                                                      │
│  ACCESS GRANTED / DENIED                                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- **Python 3.8+**
- **Webcam** (for iris & gesture detection)
- **Microphone** (for voice detection)
- **Windows 10/11** (recommended) / macOS / Linux

### Setup

```bash
# Clone the repository
git clone https://github.com/Shivani66-svg/AI-authentication-system.git
cd AI-authentication-system

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Camera capture & image processing |
| `mediapipe` | Iris & hand landmark detection |
| `numpy` | Numerical operations |
| `scipy` | Signal processing & distance metrics |
| `sounddevice` | Audio recording from microphone |
| `librosa` | MFCC voice feature extraction |
| `soundfile` | Audio file I/O |
| `flask` | Web application framework |
| `Pillow` | Image processing for GUI |
| `pyttsx3` | Text-to-speech voice assistant |
| `cryptography` | AES encryption for biometric data |

---

## 🚀 Usage

### Option 1: Desktop GUI (Recommended)

```bash
python gui_app.py
```

The full GUI application opens with an embedded camera feed, enrollment, authentication, and user management — all in one window.

### Option 2: Console Application

```bash
python security_system.py
```

A terminal-based menu interface:

```
============================================================
     THREE-TIER BIOMETRIC SECURITY SYSTEM
============================================================

  [1]  Enroll New User
  [2]  Authenticate User
  [3]  List Enrolled Users
  [4]  Delete User
  [5]  Exit
```

### Option 3: Web Application

```bash
python app.py
```

Opens a Flask web server at `http://localhost:5000` with API endpoints for all operations.

---

## 📂 How It Works

### Enrollment

1. Enter a username (letters, numbers, underscores only)
2. **Iris** — Look at the camera; iris geometry features are captured over 50 frames with blink-based liveness check
3. **Voice** — Speak your passphrase **twice**; MFCC features are extracted and averaged for a stable template
4. **Gesture** — Hold a unique hand gesture; landmark features are captured over 30 frames
5. All biometric data is **encrypted** (AES-128-CBC) and stored locally

### Authentication

1. No username needed — the system **automatically identifies** you
2. **Tier 1 (Iris)** — Live iris scan with liveness detection; compared against all enrolled users
3. **Tier 2 (Voice)** — Speak your passphrase; voice features compared using Dynamic Time Warping
4. **Tier 3 (Gesture)** — Show your hand gesture; landmark features compared via cosine similarity
5. **All 3 tiers must pass** and match the **same user** for access to be granted
6. If scores between users are too close, authentication is denied (discrimination margin)

---

## 🔐 Security Features

| Feature | Description |
|---------|-------------|
| **Encrypted Storage** | Biometric data stored as `.enc` files using Fernet (AES-128-CBC + HMAC-SHA256) |
| **Liveness Detection** | Eye Aspect Ratio (EAR) variance check detects photos/videos (no blinks = rejected) |
| **Brute-Force Protection** | 5 failed attempts → 60-second lockout across GUI, console, and web |
| **Discrimination Margin** | Best match must be significantly better than 2nd best to prevent misidentification |
| **Path Traversal Prevention** | Usernames strictly validated — only `[a-zA-Z0-9_]` allowed |
| **Security Audit Logging** | All events logged with timestamps to `security_logs/` |
| **Flask Hardening** | Secret key, X-Frame-Options, CSP, no-cache headers, rate limiting |
| **No `allow_pickle`** | Prevents arbitrary code execution via numpy deserialization attacks |
| **Safe Screen Clearing** | Uses `subprocess.run()` instead of `os.system()` (no shell injection) |
| **Generic Error Messages** | API responses never leak internal details or user existence |
| **Machine-Bound Keys** | Encryption key derived from machine identity — data non-transferable |
| **Auto Migration** | Legacy unencrypted `.npy` files automatically encrypted on first load |

---

## 📁 Project Structure

```
AI-authentication-system/
├── gui_app.py              # Desktop GUI application (Tkinter)
├── security_system.py      # Console application (terminal menu)
├── app.py                  # Web application (Flask)
│
├── iris_auth.py            # Tier 1: Iris detection & verification
├── voice_auth.py           # Tier 2: Voice recording & verification
├── gesture_auth.py         # Tier 3: Hand gesture recognition & verification
│
├── database.py             # Encrypted user data storage & retrieval
├── config.py               # Centralized security configuration
├── crypto_utils.py         # AES encryption/decryption utilities
├── security_logger.py      # Security audit event logging
├── utils.py                # Shared utilities (DTW, cosine similarity, matching)
├── voice_assistant.py      # Text-to-speech feedback
│
├── templates/              # Flask HTML templates
├── static/                 # Flask static assets
├── user_data/              # Enrolled user biometric data (encrypted)
│   └── <username>/
│       ├── user_info.json
│       ├── iris_features.enc
│       ├── voice_features.enc
│       └── gesture_features.enc
├── security_logs/          # Timestamped security audit logs
│
├── requirements.txt        # Python dependencies
├── Securitix.spec          # PyInstaller build specification
├── .gitignore              # Excludes sensitive data from Git
└── README.md               # This file
```

---

## 🏭 Building the Standalone Executable

To create a portable `.exe` that runs without Python installed:

```bash
pip install pyinstaller
pyinstaller Securitix.spec --noconfirm
```

The output will be in `dist/Securitix/`. Double-click `Securitix.exe` to run.

---

## ⚙️ Configuration

All security settings are centralized in `config.py`:

```python
# Authentication Thresholds
IRIS_THRESHOLD = 0.88           # Minimum cosine similarity for iris
VOICE_THRESHOLD = 45.0          # Maximum DTW distance for voice
GESTURE_THRESHOLD = 0.90        # Minimum cosine similarity for gesture

# Discrimination Margin (prevents misidentification)
IRIS_MARGIN = 0.03              # Best must beat 2nd best by 3%
VOICE_MARGIN = 5.0              # Best must be 5 units below 2nd best
GESTURE_MARGIN = 0.03           # Best must beat 2nd best by 3%

# Brute-Force Protection
MAX_FAILED_ATTEMPTS = 5         # Lock after 5 failures
LOCKOUT_DURATION_SECONDS = 60   # 60-second cooldown
```

---

## 📝 Tips for Best Results

- **Iris** — Look straight at the camera with both eyes open. Blink naturally (the system checks for blinks as liveness proof).
- **Voice** — Speak clearly and consistently. Use the same passphrase for both enrollment samples and authentication.
- **Gesture** — Use a unique gesture (peace sign, rock on, etc.) and hold it steady. Avoid common gestures like open hand.
- **Lighting** — Use consistent lighting for enrollment and authentication.
- **Distance** — Maintain similar distance from the camera (~50-70 cm for iris, ~30-50 cm for gesture).
- **Re-enroll** — If authentication frequently fails, delete and re-enroll with steady, controlled captures.


