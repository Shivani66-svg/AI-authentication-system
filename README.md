# Three-Tier Biometric Security System

A Python-based multi-layered biometric security system that uses **Iris Detection**, **Voice Recognition**, and **Hand Gesture Recognition** to authenticate users.

## Security Tiers

| Tier | Biometric | Technology | Method |
|------|-----------|------------|--------|
| 1 | **Iris Detection** | MediaPipe Face Mesh | Cosine similarity on iris geometry features |
| 2 | **Voice Detection** | librosa MFCC + DTW | Dynamic Time Warping on MFCC voice features |
| 3 | **Hand Gesture** | MediaPipe Hands | Cosine similarity on hand landmark features |

## How It Works

### Enrollment
1. Enter a username
2. **Iris**: Look at the camera — iris geometry features are captured over 30 frames
3. **Voice**: Speak a passphrase 3 times — MFCC features are extracted and averaged
4. **Gesture**: Hold a unique hand gesture — landmark features are captured over 30 frames
5. All biometric data is stored locally in `user_data/<username>/`

### Authentication
1. Enter your username
2. **Tier 1 (Iris)**: Look at the camera — iris features are compared with stored template
3. **Tier 2 (Voice)**: Speak the same passphrase — voice features are compared using DTW
4. **Tier 3 (Gesture)**: Show the same hand gesture — gesture features are compared
5. **All 3 tiers must pass** for access to be granted

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- `opencv-python` — Camera & display
- `mediapipe` — Iris & hand landmark detection
- `numpy` — Numerical operations
- `scipy` — Signal processing
- `sounddevice` — Audio recording
- `librosa` — MFCC feature extraction
- `soundfile` — Audio file I/O

## Usage

```bash
python security_system.py
```

The main menu will appear:
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

## Project Structure

```
iris/
├── security_system.py    # Main application (menu & orchestration)
├── iris_auth.py          # Tier 1: Iris enrollment & verification
├── voice_auth.py         # Tier 2: Voice enrollment & verification
├── gesture_auth.py       # Tier 3: Hand gesture enrollment & verification
├── database.py           # User data storage (JSON + numpy)
├── utils.py              # Shared utilities (DTW, cosine similarity)
├── requirements.txt      # Python dependencies
├── user_data/            # Enrolled user biometric data (auto-created)
│   └── <username>/
│       ├── user_info.json
│       ├── iris_features.npy
│       ├── voice_features.npy
│       └── gesture_features.npy
└── README.md             # This file
```

## Requirements

- Python 3.8+
- Webcam (for iris & gesture detection)
- Microphone (for voice detection)
- Windows / macOS / Linux

## Notes

- During **iris enrollment**, look straight at the camera with both eyes open
- During **voice enrollment**, speak clearly and consistently
- During **gesture enrollment**, hold your chosen gesture steady
- Use the **same conditions** (lighting, distance) for enrollment and authentication
- The system stores data locally — no internet required
