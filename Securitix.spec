# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Securitix — Three-Tier Biometric Security System
Bundles the GUI app with all dependencies into a single folder distribution.
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect data files from key packages
mediapipe_datas = collect_data_files('mediapipe')
librosa_datas = collect_data_files('librosa')

# Collect all submodules that might be lazily imported
mediapipe_hiddens = collect_submodules('mediapipe')
librosa_hiddens = collect_submodules('librosa')

block_cipher = None

a = Analysis(
    ['gui_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        *mediapipe_datas,
        *librosa_datas,
        # Include our project modules as data (they're imported at runtime)
        ('config.py', '.'),
        ('crypto_utils.py', '.'),
        ('security_logger.py', '.'),
        ('database.py', '.'),
        ('utils.py', '.'),
        ('iris_auth.py', '.'),
        ('voice_auth.py', '.'),
        ('gesture_auth.py', '.'),
        ('voice_assistant.py', '.'),
    ],
    hiddenimports=[
        # Our modules
        'config',
        'crypto_utils',
        'security_logger',
        'database',
        'utils',
        'iris_auth',
        'voice_auth',
        'gesture_auth',
        'voice_assistant',
        # Dependencies that may not be auto-detected
        'mediapipe',
        'cv2',
        'numpy',
        'scipy',
        'scipy.spatial',
        'scipy.spatial.distance',
        'sounddevice',
        'soundfile',
        'librosa',
        'librosa.feature',
        'librosa.effects',
        'pyttsx3',
        'pyttsx3.drivers',
        'pyttsx3.drivers.sapi5',
        'cryptography',
        'cryptography.fernet',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        *mediapipe_hiddens,
        *librosa_hiddens,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Securitix',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # No console window (windowed mode)
    icon=None,       # Add your .ico file path here if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Securitix',
)
