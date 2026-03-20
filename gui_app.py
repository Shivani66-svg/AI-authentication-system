"""
=============================================================================
  SECURITIX — Three-Tier Biometric Security System (Desktop GUI)
  Built with Tkinter — runs directly in IDE, no browser needed.
  Camera feeds are embedded directly in the GUI window.
=============================================================================
"""

import sys
import os
import threading
import time
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import user_exists, save_user_data, load_user_data, get_all_users, delete_user
from database import validate_username
from utils import cosine_similarity, dtw_distance
from voice_assistant import say, say_wait
from config import (
    MAX_FAILED_ATTEMPTS, LOCKOUT_DURATION_SECONDS,
    IRIS_THRESHOLD, VOICE_THRESHOLD, GESTURE_THRESHOLD
)
from security_logger import (
    log_enrollment, log_authentication, log_auth_attempt,
    log_lockout, log_user_deleted, log_error
)

# ── Brute-Force Protection ──────────────────────────────────────────────────
_failed_ts = []

def _is_locked():
    now = time.time()
    _failed_ts[:] = [t for t in _failed_ts if now - t < 300]
    if len(_failed_ts) >= MAX_FAILED_ATTEMPTS:
        if _failed_ts and (now - _failed_ts[-1]) < LOCKOUT_DURATION_SECONDS:
            return True
    return False

def _record_fail():
    _failed_ts.append(time.time())

# ══════════════════════════════════════════════════════
#  COLOR THEME
# ══════════════════════════════════════════════════════
C = {
    "bg":       "#0a0e1a",
    "card":     "#111827",
    "input":    "#1a2035",
    "cyan":     "#00e5ff",
    "purple":   "#a855f7",
    "green":    "#22c55e",
    "red":      "#ef4444",
    "orange":   "#f59e0b",
    "txt":      "#e2e8f0",
    "txt2":     "#7a8599",
    "border":   "#1e293b",
    "glow":     "#0e4a5c",
    "btn1":     "#0891b2",
    "btn_red":  "#dc2626",
    "btn_auth": "#7c3aed",
    "pass_bg":  "#064e3b",
    "fail_bg":  "#450a0a",
    "run_bg":   "#0c2d48",
}

FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_HEAD  = ("Segoe UI", 12, "bold")
FONT_BODY  = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO  = ("Consolas", 9)
FONT_MONO_S = ("Consolas", 8)
FONT_BTN   = ("Segoe UI", 11, "bold")

# Camera display size
CAM_W = 640
CAM_H = 360


def make_tier_row(parent, icon, name):
    """Create a consistently aligned tier status row using grid layout."""
    row = tk.Frame(parent, bg=C["input"], highlightbackground=C["border"],
                   highlightthickness=1, padx=12, pady=8)
    row.pack(fill="x", pady=2)
    row.columnconfigure(1, weight=1)

    tk.Label(row, text=icon, font=("Segoe UI", 14), bg=C["input"],
             width=3).grid(row=0, column=0, rowspan=2, padx=(0, 8), sticky="w")

    tk.Label(row, text=name, font=("Segoe UI", 10, "bold"),
             fg=C["txt"], bg=C["input"], anchor="w").grid(row=0, column=1, sticky="ew")

    detail = tk.Label(row, text="Waiting...", font=FONT_MONO_S,
                      fg=C["txt2"], bg=C["input"], anchor="w")
    detail.grid(row=1, column=1, sticky="ew")

    badge = tk.Label(row, text=" WAITING ", font=("Consolas", 8, "bold"),
                     fg=C["txt2"], bg=C["bg"], padx=10, pady=3, width=12, anchor="center")
    badge.grid(row=0, column=2, rowspan=2, padx=(10, 0), sticky="e")

    return {"row": row, "detail": detail, "badge": badge}


class SecurityApp(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("SECURITIX — Three-Tier Biometric Security")
        self.geometry("960x750")
        self.minsize(900, 700)
        self.configure(bg=C["bg"])

        self._cam_photo = None  # Keep reference to avoid GC

        self._build_ui()
        self.show_page("dashboard")

    def _build_ui(self):
        # ── Header ──
        hdr = tk.Frame(self, bg=C["bg"])
        hdr.pack(fill="x", padx=25, pady=(12, 0))

        left = tk.Frame(hdr, bg=C["bg"])
        left.pack(side="left")
        tk.Label(left, text="🛡️  SECURITIX", font=("Segoe UI", 18, "bold"),
                 fg=C["cyan"], bg=C["bg"]).pack(side="left")
        tk.Label(left, text="  THREE-TIER BIOMETRIC SYSTEM", font=FONT_MONO,
                 fg=C["txt2"], bg=C["bg"]).pack(side="left", pady=(5, 0))

        right = tk.Frame(hdr, bg=C["bg"])
        right.pack(side="right")
        dot = tk.Canvas(right, width=10, height=10, bg=C["bg"], highlightthickness=0)
        dot.pack(side="left", padx=(0, 6))
        dot.create_oval(1, 1, 9, 9, fill=C["green"], outline="")
        tk.Label(right, text="SYSTEM ONLINE", font=FONT_MONO,
                 fg=C["green"], bg=C["bg"]).pack(side="left")

        # ── Separator ──
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x", padx=25, pady=(8, 0))

        # ── Nav Bar ──
        nav = tk.Frame(self, bg=C["card"])
        nav.pack(fill="x", padx=25, pady=(10, 0))

        self.nav_btns = {}
        for pid, label in [("dashboard", "📊 DASHBOARD"), ("enroll", "📝 ENROLL"),
                            ("authenticate", "🔐 AUTHENTICATE"), ("manage", "⚙️ MANAGE")]:
            b = tk.Button(nav, text=label, font=("Segoe UI", 10, "bold"),
                          bg=C["card"], fg=C["txt2"], activebackground=C["input"],
                          activeforeground=C["cyan"], border=0, cursor="hand2",
                          padx=15, pady=10, command=lambda p=pid: self.show_page(p))
            b.pack(side="left", fill="both", expand=True)
            self.nav_btns[pid] = b

        # ── Content (scrollable) ──
        container = tk.Frame(self, bg=C["bg"])
        container.pack(fill="both", expand=True, padx=25, pady=8)

        canvas = tk.Canvas(container, bg=C["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg=C["bg"])

        self.scroll_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw",
                            tags="frame")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Make canvas frame expand to full width
        def _on_canvas_configure(e):
            canvas.itemconfig("frame", width=e.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.content = self.scroll_frame

        self.pages = {}
        self._build_dashboard()
        self._build_enroll()
        self._build_authenticate()
        self._build_manage()

    def show_page(self, pid):
        for p in self.pages.values():
            p.pack_forget()
        self.pages[pid].pack(fill="both", expand=True)
        for k, b in self.nav_btns.items():
            b.configure(bg=C["glow"] if k == pid else C["card"],
                        fg=C["cyan"] if k == pid else C["txt2"])
        if pid == "dashboard": self._refresh_dashboard()
        elif pid == "manage": self._refresh_users()

    # ══════════════════════════════════════════════════
    #  CAMERA FEED HELPER
    # ══════════════════════════════════════════════════
    def _update_cam(self, frame):
        """Thread-safe: schedule frame display on the main thread."""
        try:
            # Resize frame for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            scale = min(CAM_W / w, CAM_H / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Add letterbox padding to fill exact CAM_W x CAM_H
            canvas = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
            canvas[:] = (10, 14, 26)  # match bg color
            y_off = (CAM_H - new_h) // 2
            x_off = (CAM_W - new_w) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = frame_resized

            img = Image.fromarray(canvas)
            self.after(0, lambda: self._display_cam(img))
        except Exception:
            pass  # GUI might be closing

    def _display_cam(self, img):
        """Display a PIL image on the camera label (must be called from main thread)."""
        try:
            self._cam_photo = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=self._cam_photo)
        except Exception:
            pass

    def _show_cam_panel(self, parent):
        """Show the camera feed panel inside a parent frame."""
        self.cam_frame = tk.Frame(parent, bg="#000", highlightbackground=C["cyan"],
                                  highlightthickness=2)
        self.cam_frame.pack(fill="x", pady=(8, 0))

        # Camera header
        cam_hdr = tk.Frame(self.cam_frame, bg=C["card"], padx=12, pady=6)
        cam_hdr.pack(fill="x")
        tk.Label(cam_hdr, text="📷  LIVE CAMERA FEED", font=("Consolas", 9, "bold"),
                 fg=C["cyan"], bg=C["card"]).pack(side="left")
        self.cam_status_lbl = tk.Label(cam_hdr, text="● ACTIVE", font=("Consolas", 8, "bold"),
                                       fg=C["green"], bg=C["card"])
        self.cam_status_lbl.pack(side="right")

        # Camera image label
        self.cam_label = tk.Label(self.cam_frame, bg="#000",
                                  width=CAM_W, height=CAM_H)
        self.cam_label.pack(padx=2, pady=(0, 2))

        # Show a placeholder
        placeholder = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        placeholder[:] = (10, 14, 26)
        cv2.putText(placeholder, "Camera initializing...", (CAM_W // 2 - 120, CAM_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1, cv2.LINE_AA)
        img = Image.fromarray(placeholder)
        self._cam_photo = ImageTk.PhotoImage(img)
        self.cam_label.configure(image=self._cam_photo)

    def _hide_cam_panel(self):
        """Hide and destroy the camera panel."""
        try:
            if hasattr(self, 'cam_frame') and self.cam_frame.winfo_exists():
                self.cam_frame.destroy()
        except Exception:
            pass

    # ══════════════════════════════════════════════════
    #  DASHBOARD
    # ══════════════════════════════════════════════════
    def _build_dashboard(self):
        pg = tk.Frame(self.content, bg=C["bg"])
        self.pages["dashboard"] = pg

        tk.Label(pg, text="📊  System Dashboard", font=FONT_TITLE,
                 fg=C["txt"], bg=C["bg"], anchor="w").pack(fill="x")
        tk.Label(pg, text="> security_status: active | monitoring...", font=FONT_MONO,
                 fg=C["txt2"], bg=C["bg"], anchor="w").pack(fill="x", pady=(0, 12))

        # Stats
        sf = tk.Frame(pg, bg=C["bg"])
        sf.pack(fill="x", pady=(0, 12))
        for emoji, val, lbl, attr in [("👥", "0", "ENROLLED USERS", "stat_users"),
                                       ("🔒", "3", "SECURITY TIERS", None),
                                       ("🟢", "ACTIVE", "SYSTEM STATUS", None)]:
            card = tk.Frame(sf, bg=C["card"], highlightbackground=C["border"],
                           highlightthickness=1, padx=25, pady=16)
            card.pack(side="left", fill="both", expand=True, padx=(0, 8))
            tk.Label(card, text=emoji, font=("Segoe UI", 20), bg=C["card"]).pack()
            vl = tk.Label(card, text=val, font=("Segoe UI", 22, "bold"),
                          fg=C["cyan"], bg=C["card"])
            vl.pack()
            tk.Label(card, text=lbl, font=FONT_MONO_S, fg=C["txt2"], bg=C["card"]).pack()
            if attr: setattr(self, attr, vl)

        # Tiers overview
        tf = tk.Frame(pg, bg=C["card"], highlightbackground=C["border"],
                      highlightthickness=1, padx=25, pady=16)
        tf.pack(fill="x")
        tk.Label(tf, text="SECURITY TIERS OVERVIEW", font=FONT_HEAD,
                 fg=C["cyan"], bg=C["card"], anchor="w").pack(fill="x", pady=(0, 12))

        tr = tk.Frame(tf, bg=C["card"])
        tr.pack(fill="x")
        for num, ico, nm, desc in [("TIER 01", "👁️", "Iris Detection",
                                     "MediaPipe Face Mesh\niris geometry analysis"),
                                    ("TIER 02", "🎤", "Voice Detection",
                                     "MFCC extraction +\nDynamic Time Warping"),
                                    ("TIER 03", "🤚", "Hand Gesture",
                                     "MediaPipe Hands\nlandmark matching")]:
            tc = tk.Frame(tr, bg=C["input"], highlightbackground=C["border"],
                          highlightthickness=1, padx=18, pady=12)
            tc.pack(side="left", fill="both", expand=True, padx=(0, 6))
            tk.Label(tc, text=num, font=FONT_MONO_S, fg=C["purple"], bg=C["input"]).pack()
            tk.Label(tc, text=ico, font=("Segoe UI", 26), bg=C["input"]).pack(pady=4)
            tk.Label(tc, text=nm, font=("Segoe UI", 11, "bold"),
                     fg=C["txt"], bg=C["input"]).pack()
            tk.Label(tc, text=desc, font=("Segoe UI", 8), fg=C["txt2"],
                     bg=C["input"], justify="center").pack(pady=(4, 0))

    def _refresh_dashboard(self):
        self.stat_users.configure(text=str(len(get_all_users())))

    # ══════════════════════════════════════════════════
    #  ENROLL
    # ══════════════════════════════════════════════════
    def _build_enroll(self):
        pg = tk.Frame(self.content, bg=C["bg"])
        self.pages["enroll"] = pg

        tk.Label(pg, text="📝  User Enrollment", font=FONT_TITLE,
                 fg=C["txt"], bg=C["bg"], anchor="w").pack(fill="x")
        tk.Label(pg, text="> register biometric data across all security tiers",
                 font=FONT_MONO, fg=C["txt2"], bg=C["bg"], anchor="w").pack(fill="x", pady=(0, 12))

        # Form
        self.enroll_form = tk.Frame(pg, bg=C["card"], highlightbackground=C["border"],
                                    highlightthickness=1, padx=28, pady=20)
        self.enroll_form.pack(fill="x")

        tk.Label(self.enroll_form, text="USERNAME", font=("Consolas", 9, "bold"),
                 fg=C["cyan"], bg=C["card"], anchor="w").pack(fill="x", pady=(0, 6))
        self.enroll_name = tk.Entry(self.enroll_form, font=("Segoe UI", 13), bg=C["input"],
                                    fg=C["txt"], insertbackground=C["cyan"],
                                    highlightbackground=C["border"], highlightthickness=1,
                                    border=0, relief="flat")
        self.enroll_name.pack(fill="x", ipady=10, pady=(0, 12))

        tk.Label(self.enroll_form,
                 text="⚡ The system will capture iris → voice → gesture.\n"
                      "   Camera feed will appear below during capture.",
                 font=FONT_SMALL, fg=C["txt2"], bg=C["card"], anchor="w",
                 justify="left").pack(fill="x", pady=(0, 14))

        self.enroll_btn = tk.Button(self.enroll_form, text="🚀  START ENROLLMENT",
                                    font=FONT_BTN, bg=C["btn1"], fg="white",
                                    activebackground="#0e7490", border=0,
                                    cursor="hand2", padx=28, pady=11,
                                    command=self._start_enroll)
        self.enroll_btn.pack(anchor="w")

        # Progress
        self.enroll_prog = tk.Frame(pg, bg=C["card"], highlightbackground=C["border"],
                                    highlightthickness=1, padx=28, pady=18)
        tk.Label(self.enroll_prog, text="ENROLLMENT IN PROGRESS", font=FONT_HEAD,
                 fg=C["txt"], bg=C["card"], anchor="w").pack(fill="x", pady=(0, 4))
        self.enroll_msg = tk.Label(self.enroll_prog, text="Starting...", font=FONT_MONO,
                                   fg=C["txt2"], bg=C["card"], anchor="w")
        self.enroll_msg.pack(fill="x", pady=(0, 10))

        self.et = {}
        for tid, ico, nm in [("t1", "👁️", "Tier 1 — Iris Scan"),
                              ("t2", "🎤", "Tier 2 — Voice Recording"),
                              ("t3", "🤚", "Tier 3 — Gesture Scan")]:
            self.et[tid] = make_tier_row(self.enroll_prog, ico, nm)

        # Camera goes inside enroll_prog (after tier rows)
        self.enroll_cam_slot = tk.Frame(self.enroll_prog, bg=C["card"])
        self.enroll_cam_slot.pack(fill="x")

        # Result
        self.enroll_res = tk.Frame(pg, bg=C["card"], highlightbackground=C["border"],
                                   highlightthickness=1, padx=28, pady=25)
        self.er_icon = tk.Label(self.enroll_res, text="", font=("Segoe UI", 42), bg=C["card"])
        self.er_icon.pack()
        self.er_title = tk.Label(self.enroll_res, text="", font=("Segoe UI", 18, "bold"), bg=C["card"])
        self.er_title.pack(pady=(5, 3))
        self.er_sub = tk.Label(self.enroll_res, text="", font=FONT_BODY, fg=C["txt2"], bg=C["card"])
        self.er_sub.pack()
        tk.Button(self.enroll_res, text="↩  ENROLL ANOTHER", font=("Segoe UI", 10, "bold"),
                  bg=C["btn1"], fg="white", border=0, cursor="hand2", padx=20, pady=8,
                  command=self._reset_enroll).pack(pady=(16, 0))

    def _set_tier(self, tiers, tid, status, text):
        t = tiers[tid]
        cfg = {"waiting": (C["txt2"], C["bg"],     " WAITING "),
               "running": (C["cyan"],  C["run_bg"], " SCANNING "),
               "passed":  (C["green"], C["pass_bg"]," PASSED "),
               "failed":  (C["red"],   C["fail_bg"]," FAILED ")}
        fg, bg, badge = cfg.get(status, cfg["waiting"])
        t["row"].configure(highlightbackground=fg)
        t["detail"].configure(text=text)
        t["badge"].configure(text=badge, fg=fg, bg=bg)

    def _start_enroll(self):
        name = self.enroll_name.get().strip().lower()
        if not name:
            messagebox.showwarning("Required", "Please enter a username.")
            return
        # Validate username (security: prevent path traversal)
        safe_name = validate_username(name)
        if safe_name is None:
            messagebox.showerror("Invalid", "Invalid username. Use only letters, numbers, and underscores (2-32 chars).")
            return
        name = safe_name
        if user_exists(name):
            if not messagebox.askyesno("Exists", f"'{name}' exists. Overwrite?"):
                return

        self.enroll_form.pack_forget()
        self.enroll_res.pack_forget()
        self.enroll_prog.pack(fill="x", pady=(8, 0))
        for tid in self.et:
            self._set_tier(self.et, tid, "waiting", "Waiting...")
        self.enroll_btn.configure(state="disabled")

        # Show camera in enrollment
        self._show_cam_panel(self.enroll_cam_slot)

        say("Enrollment started.")
        threading.Thread(target=self._run_enroll, args=(name,), daemon=True).start()

    def _run_enroll(self, name):
        cam = None
        try:
            # Pre-open camera once for instant startup across all tiers
            cam = cv2.VideoCapture(0)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # Warm up: read frames to let auto-exposure settle
            for _ in range(10):
                cam.read()

            # Tier 1: Iris
            self.after(0, lambda: self._set_tier(self.et, "t1", "running", "Look at the camera..."))
            self.after(0, lambda: self.enroll_msg.configure(text="Scanning iris — look at camera..."))
            say_wait("Iris enrollment. Look at the camera.")
            from iris_auth import enroll_iris
            iris = enroll_iris(frame_callback=self._update_cam, cap=cam)
            if iris is None:
                say("Enrollment failed.")
                cam.release()
                cam = None
                self.after(0, lambda: self._enroll_done(False, "Iris enrollment failed."))
                return
            self.after(0, lambda: self._set_tier(self.et, "t1", "passed", "Iris captured!"))

            time.sleep(3)  # 3-second pause between tiers

            # Tier 2: Voice
            self.after(0, lambda: self._set_tier(self.et, "t2", "running", "Recording voice..."))
            self.after(0, lambda: self.enroll_msg.configure(text="Recording voice — speak your passphrase..."))
            self._show_mic_graphic()
            say_wait("Voice enrollment. Speak your phrase.")
            from voice_auth import enroll_voice
            voice = enroll_voice()
            if voice is None:
                say("Enrollment failed.")
                cam.release()
                cam = None
                self.after(0, lambda: self._enroll_done(False, "Voice enrollment failed."))
                return
            self.after(0, lambda: self._set_tier(self.et, "t2", "passed", "Voice captured!"))

            time.sleep(3)  # 3-second pause between tiers

            # Flush stale frames before gesture capture
            for _ in range(5):
                cam.read()

            # Tier 3: Gesture
            self.after(0, lambda: self._set_tier(self.et, "t3", "running", "Show your gesture..."))
            self.after(0, lambda: self.enroll_msg.configure(text="Scanning gesture — show hand to camera..."))
            say_wait("Gesture enrollment. Show your hand gesture.")
            from gesture_auth import enroll_gesture
            gesture = enroll_gesture(frame_callback=self._update_cam, cap=cam)
            if gesture is None:
                say("Enrollment failed.")
                cam.release()
                cam = None
                self.after(0, lambda: self._enroll_done(False, "Gesture enrollment failed."))
                return
            self.after(0, lambda: self._set_tier(self.et, "t3", "passed", "Gesture captured!"))

            cam.release()
            cam = None
            save_user_data(name, iris, voice, gesture)
            log_enrollment(name, True)
            say("Enrollment completed.")
            self.after(0, lambda: self._enroll_done(True, f"User '{name}' enrolled successfully!"))

        except Exception as e:
            if cam is not None:
                try:
                    cam.release()
                except Exception:
                    pass
            say("Enrollment failed.")
            self.after(0, lambda: self._enroll_done(False, str(e)))

    def _show_mic_graphic(self):
        """Show a microphone graphic on the camera panel during voice recording."""
        mic = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        mic[:] = (10, 14, 26)
        # Draw microphone icon
        cx, cy = CAM_W // 2, CAM_H // 2 - 30
        cv2.ellipse(mic, (cx, cy - 20), (25, 40), 0, 0, 360, (0, 229, 255), 3)
        cv2.line(mic, (cx, cy + 20), (cx, cy + 50), (0, 229, 255), 3)
        cv2.line(mic, (cx - 20, cy + 50), (cx + 20, cy + 50), (0, 229, 255), 3)
        cv2.ellipse(mic, (cx, cy + 5), (35, 50), 0, 0, 180, (0, 229, 255), 2)
        # Text
        cv2.putText(mic, "VOICE RECORDING", (cx - 110, cy + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 229, 255), 2, cv2.LINE_AA)
        cv2.putText(mic, "Speak your passphrase clearly...", (cx - 155, cy + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (122, 133, 153), 1, cv2.LINE_AA)
        # Pulsing rings
        for r in [60, 80, 100]:
            cv2.circle(mic, (cx, cy), r, (20, 40, 60), 1, cv2.LINE_AA)

        img = Image.fromarray(cv2.cvtColor(mic, cv2.COLOR_BGR2RGB))
        self.after(0, lambda: self._display_cam(img))

    def _enroll_done(self, success, msg):
        self._hide_cam_panel()
        self.enroll_msg.configure(text="Complete!" if success else "Failed.")
        self.er_icon.configure(text="✅" if success else "❌")
        self.er_title.configure(text="ENROLLMENT COMPLETE" if success else "ENROLLMENT FAILED",
                                fg=C["green"] if success else C["red"])
        self.er_sub.configure(text=msg)
        self.enroll_res.configure(highlightbackground=C["green"] if success else C["red"])
        self.enroll_res.pack(fill="x", pady=(8, 0))

    def _reset_enroll(self):
        self._hide_cam_panel()
        self.enroll_prog.pack_forget()
        self.enroll_res.pack_forget()
        self.enroll_form.pack(fill="x")
        self.enroll_btn.configure(state="normal")
        self.enroll_name.delete(0, "end")

    # ══════════════════════════════════════════════════
    #  AUTHENTICATE
    # ══════════════════════════════════════════════════
    def _build_authenticate(self):
        pg = tk.Frame(self.content, bg=C["bg"])
        self.pages["authenticate"] = pg

        tk.Label(pg, text="🔐  Authentication", font=FONT_TITLE,
                 fg=C["txt"], bg=C["bg"], anchor="w").pack(fill="x")
        tk.Label(pg, text="> automatic identity verification — no username required",
                 font=FONT_MONO, fg=C["txt2"], bg=C["bg"], anchor="w").pack(fill="x", pady=(0, 12))

        # Start card
        self.auth_start = tk.Frame(pg, bg=C["card"], highlightbackground=C["border"],
                                   highlightthickness=1, padx=28, pady=30)
        self.auth_start.pack(fill="x")
        tk.Label(self.auth_start, text="🔐", font=("Segoe UI", 36), bg=C["card"]).pack()
        tk.Label(self.auth_start, text="BIOMETRIC VERIFICATION", font=("Segoe UI", 14, "bold"),
                 fg=C["txt"], bg=C["card"]).pack(pady=(6, 6))
        tk.Label(self.auth_start,
                 text="The system will scan your iris, voice, and gesture\nto identify you. No username needed.",
                 font=FONT_SMALL, fg=C["txt2"], bg=C["card"], justify="center").pack(pady=(0, 14))
        self.auth_btn = tk.Button(self.auth_start, text="⚡  BEGIN AUTHENTICATION",
                                  font=("Segoe UI", 12, "bold"), bg=C["btn_auth"], fg="white",
                                  activebackground="#6d28d9", border=0, cursor="hand2",
                                  padx=35, pady=12, command=self._start_auth)
        self.auth_btn.pack()

        # Progress
        self.auth_prog = tk.Frame(pg, bg=C["card"], highlightbackground=C["border"],
                                  highlightthickness=1, padx=28, pady=18)
        tk.Label(self.auth_prog, text="AUTHENTICATION IN PROGRESS", font=FONT_HEAD,
                 fg=C["txt"], bg=C["card"], anchor="w").pack(fill="x", pady=(0, 4))
        self.auth_msg = tk.Label(self.auth_prog, text="Starting...", font=FONT_MONO,
                                 fg=C["txt2"], bg=C["card"], anchor="w")
        self.auth_msg.pack(fill="x", pady=(0, 10))

        self.at = {}
        for tid, ico, nm in [("t1", "👁️", "Tier 1 — Iris Verification"),
                              ("t2", "🎤", "Tier 2 — Voice Verification"),
                              ("t3", "🤚", "Tier 3 — Gesture Verification")]:
            self.at[tid] = make_tier_row(self.auth_prog, ico, nm)

        # Camera slot inside auth progress
        self.auth_cam_slot = tk.Frame(self.auth_prog, bg=C["card"])
        self.auth_cam_slot.pack(fill="x")

        # Result
        self.auth_res = tk.Frame(pg, bg=C["card"], highlightbackground=C["border"],
                                 highlightthickness=1, padx=28, pady=25)
        self.ar_icon = tk.Label(self.auth_res, text="", font=("Segoe UI", 42), bg=C["card"])
        self.ar_icon.pack()
        self.ar_title = tk.Label(self.auth_res, text="", font=("Segoe UI", 18, "bold"), bg=C["card"])
        self.ar_title.pack(pady=(5, 3))
        self.ar_sub = tk.Label(self.auth_res, text="", font=FONT_BODY, fg=C["txt2"], bg=C["card"])
        self.ar_sub.pack()
        tk.Button(self.auth_res, text="↩  TRY AGAIN", font=("Segoe UI", 10, "bold"),
                  bg=C["btn_auth"], fg="white", border=0, cursor="hand2",
                  padx=20, pady=8, command=self._reset_auth).pack(pady=(16, 0))

    def _start_auth(self):
        if not get_all_users():
            messagebox.showwarning("No Users", "No users enrolled! Enroll first.")
            return
        # Brute-force lockout check
        if _is_locked():
            remaining = int(LOCKOUT_DURATION_SECONDS - (time.time() - _failed_ts[-1]))
            messagebox.showerror("Locked Out",
                f"Too many failed attempts.\nTry again in {max(remaining, 1)} seconds.")
            log_lockout("gui", "Authentication blocked by lockout")
            return
        log_auth_attempt("GUI authentication started")
        self.auth_start.pack_forget()
        self.auth_res.pack_forget()
        self.auth_prog.pack(fill="x", pady=(8, 0))
        for tid in self.at:
            self._set_tier(self.at, tid, "waiting", "Waiting...")
        self.auth_btn.configure(state="disabled")

        # Show camera in authentication
        self._show_cam_panel(self.auth_cam_slot)

        say("Authentication started.")
        threading.Thread(target=self._run_auth, daemon=True).start()

    def _run_auth(self):
        cam = None
        try:
            users = get_all_users()

            # Pre-open camera once for instant startup across all tiers
            cam = cv2.VideoCapture(0)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            for _ in range(10):
                cam.read()

            # Tier 1: Iris
            self.after(0, lambda: self._set_tier(self.at, "t1", "running", "Scanning iris..."))
            self.after(0, lambda: self.auth_msg.configure(text="Scanning iris — look at camera..."))
            say_wait("Iris authentication. Look at the camera.")
            from iris_auth import capture_iris
            live_iris = capture_iris(frame_callback=self._update_cam, cap=cam)
            if live_iris is None:
                say("Authentication failed.")
                cam.release()
                cam = None
                self.after(0, lambda: self._auth_done(False, "Iris scan failed."))
                return

            iris_scores = {}
            for u in users:
                d = load_user_data(u)
                if d: iris_scores[u] = cosine_similarity(d[0], live_iris)
            best_iris = max(iris_scores, key=iris_scores.get)
            iris_sc = iris_scores[best_iris]
            iris_ok = iris_sc >= IRIS_THRESHOLD
            s = "passed" if iris_ok else "failed"
            self.after(0, lambda: self._set_tier(self.at, "t1", s, f"Best: {best_iris} ({iris_sc:.0%})"))

            time.sleep(3)  # 3-second pause between tiers

            # Tier 2: Voice
            self.after(0, lambda: self._set_tier(self.at, "t2", "running", "Recording voice..."))
            self.after(0, lambda: self.auth_msg.configure(text="Recording voice — speak your passphrase..."))
            self._show_mic_graphic()
            say_wait("Voice authentication. Speak your passphrase.")
            from voice_auth import capture_voice
            live_voice = capture_voice()
            if live_voice is None:
                say("Authentication failed.")
                cam.release()
                cam = None
                self.after(0, lambda: self._auth_done(False, "Voice scan failed."))
                return

            voice_scores = {}
            for u in users:
                d = load_user_data(u)
                if d: voice_scores[u] = dtw_distance(live_voice.T, d[1].T)
            best_voice = min(voice_scores, key=voice_scores.get)
            voice_d = voice_scores[best_voice]
            voice_ok = voice_d <= VOICE_THRESHOLD
            s = "passed" if voice_ok else "failed"
            self.after(0, lambda: self._set_tier(self.at, "t2", s, f"Best: {best_voice} (dist: {voice_d:.1f})"))

            time.sleep(3)  # 3-second pause between tiers

            # Flush stale frames before gesture capture
            for _ in range(5):
                cam.read()

            # Tier 3: Gesture
            self.after(0, lambda: self._set_tier(self.at, "t3", "running", "Scanning gesture..."))
            self.after(0, lambda: self.auth_msg.configure(text="Scanning gesture — show your hand..."))
            say_wait("Gesture authentication. Show your hand gesture.")
            from gesture_auth import capture_gesture
            live_gesture = capture_gesture(frame_callback=self._update_cam, cap=cam)
            if live_gesture is None:
                say("Authentication failed.")
                cam.release()
                cam = None
                self.after(0, lambda: self._auth_done(False, "Gesture scan failed."))
                return

            gest_scores = {}
            for u in users:
                d = load_user_data(u)
                if d: gest_scores[u] = cosine_similarity(d[2], live_gesture)
            best_gest = max(gest_scores, key=gest_scores.get)
            gest_sc = gest_scores[best_gest]
            gest_ok = gest_sc >= GESTURE_THRESHOLD
            s = "passed" if gest_ok else "failed"
            self.after(0, lambda: self._set_tier(self.at, "t3", s, f"Best: {best_gest} ({gest_sc:.0%})"))

            cam.release()
            cam = None

            # Decision
            all_ok = iris_ok and voice_ok and gest_ok
            same = (best_iris == best_voice == best_gest)

            # Logging
            auth_scores = {
                "iris": f"{iris_sc:.2%}",
                "voice_dist": f"{voice_d:.1f}",
                "gesture": f"{gest_sc:.2%}",
            }

            if all_ok and same:
                log_authentication(best_iris, True, auth_scores)
                say(f"Access granted to {best_iris}.")
                self.after(0, lambda: self._auth_done(True, f"Identified as: {best_iris}"))
            elif all_ok:
                _record_fail()
                log_authentication("CONFLICT", False, auth_scores)
                say("Authentication failed.")
                self.after(0, lambda: self._auth_done(False, "Conflict — tiers matched different users."))
            else:
                _record_fail()
                log_authentication(best_iris, False, auth_scores)
                say("Authentication failed.")
                self.after(0, lambda: self._auth_done(False, "Biometric verification failed."))

        except Exception as e:
            if cam is not None:
                try:
                    cam.release()
                except Exception:
                    pass
            say("Authentication failed.")
            self.after(0, lambda: self._auth_done(False, str(e)))

    def _auth_done(self, granted, msg):
        self._hide_cam_panel()
        self.auth_msg.configure(text="Complete!" if granted else "Failed.")
        self.ar_icon.configure(text="🔓" if granted else "🔒")
        self.ar_title.configure(text="ACCESS GRANTED" if granted else "ACCESS DENIED",
                                fg=C["green"] if granted else C["red"])
        self.ar_sub.configure(text=msg)
        self.auth_res.configure(highlightbackground=C["green"] if granted else C["red"])
        self.auth_res.pack(fill="x", pady=(8, 0))

    def _reset_auth(self):
        self._hide_cam_panel()
        self.auth_prog.pack_forget()
        self.auth_res.pack_forget()
        self.auth_start.pack(fill="x")
        self.auth_btn.configure(state="normal")

    # ══════════════════════════════════════════════════
    #  MANAGE
    # ══════════════════════════════════════════════════
    def _build_manage(self):
        pg = tk.Frame(self.content, bg=C["bg"])
        self.pages["manage"] = pg

        tk.Label(pg, text="⚙️  Manage Users", font=FONT_TITLE,
                 fg=C["txt"], bg=C["bg"], anchor="w").pack(fill="x")
        tk.Label(pg, text="> view and manage enrolled biometric profiles",
                 font=FONT_MONO, fg=C["txt2"], bg=C["bg"], anchor="w").pack(fill="x", pady=(0, 12))

        self.manage_card = tk.Frame(pg, bg=C["card"], highlightbackground=C["border"],
                                    highlightthickness=1, padx=22, pady=16)
        self.manage_card.pack(fill="both", expand=True)
        self.users_box = tk.Frame(self.manage_card, bg=C["card"])
        self.users_box.pack(fill="both", expand=True)

    def _refresh_users(self):
        for w in self.users_box.winfo_children():
            w.destroy()
        users = get_all_users()
        if not users:
            tk.Label(self.users_box, text="👤\n\nNo users enrolled yet.", font=FONT_BODY,
                     fg=C["txt2"], bg=C["card"], justify="center").pack(pady=40)
            return
        for user in users:
            row = tk.Frame(self.users_box, bg=C["input"], highlightbackground=C["border"],
                           highlightthickness=1, padx=16, pady=10)
            row.pack(fill="x", pady=3)
            row.columnconfigure(1, weight=1)
            tk.Label(row, text="👤", font=("Segoe UI", 14), bg=C["input"]).grid(
                row=0, column=0, padx=(0, 12))
            tk.Label(row, text=user.capitalize(), font=("Segoe UI", 12, "bold"),
                     fg=C["txt"], bg=C["input"], anchor="w").grid(row=0, column=1, sticky="w")
            tk.Button(row, text="🗑 DELETE", font=("Segoe UI", 9, "bold"),
                      bg=C["btn_red"], fg="white", border=0, cursor="hand2", padx=14, pady=5,
                      command=lambda u=user: self._del_user(u)).grid(row=0, column=2, padx=(10, 0))

    def _del_user(self, name):
        if messagebox.askyesno("Confirm", f"Delete '{name}'? Cannot be undone."):
            delete_user(name)
            log_user_deleted(name, deleted_by="gui")
            self._refresh_users()


if __name__ == "__main__":
    app = SecurityApp()
    app.mainloop()
