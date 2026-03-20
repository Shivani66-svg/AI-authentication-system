"""
=============================================================================
  SECURITY LOGGER — Audit trail for all security-relevant events
  Logs authentication attempts, enrollments, deletions, and lockouts.
=============================================================================
"""

import os
import logging
import time
from datetime import datetime

from config import LOG_DIR


def _setup_logger():
    """Set up the security event logger with file and console handlers."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Restrict directory permissions (owner only) on non-Windows
    try:
        if os.name != 'nt':
            os.chmod(LOG_DIR, 0o700)
    except Exception:
        pass
    
    logger = logging.getLogger("securitix.security")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers on reimport
    if logger.handlers:
        return logger
    
    # File handler — one log file per day
    log_file = os.path.join(LOG_DIR, f"security_{datetime.now().strftime('%Y-%m-%d')}.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    
    # Format with timestamp, level, and message
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler (minimal)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


_logger = _setup_logger()


# ── Public API ──────────────────────────────────────────────────────────────

def log_enrollment(username, success, detail=""):
    """Log an enrollment event."""
    status = "SUCCESS" if success else "FAILED"
    _logger.info(f"ENROLLMENT | user={username} | status={status} | {detail}")


def log_authentication(username, success, scores=None, detail=""):
    """Log an authentication attempt."""
    status = "GRANTED" if success else "DENIED"
    score_str = ""
    if scores:
        parts = [f"{k}={v}" for k, v in scores.items()]
        score_str = " | " + " | ".join(parts)
    _logger.info(f"AUTH | user={username} | status={status}{score_str} | {detail}")


def log_auth_attempt(detail=""):
    """Log a generic authentication attempt start."""
    _logger.info(f"AUTH_ATTEMPT | {detail}")


def log_failed_tier(tier_name, detail=""):
    """Log a failed biometric tier."""
    _logger.info(f"TIER_FAIL | tier={tier_name} | {detail}")


def log_lockout(source="", detail=""):
    """Log a brute-force lockout event."""
    _logger.warning(f"LOCKOUT | source={source} | {detail}")


def log_user_deleted(username, deleted_by="system"):
    """Log a user deletion."""
    _logger.info(f"DELETE | user={username} | by={deleted_by}")


def log_security_event(event_type, detail=""):
    """Log a generic security event."""
    _logger.info(f"{event_type} | {detail}")


def log_error(context, error_msg):
    """Log a security-relevant error (without leaking to user)."""
    _logger.error(f"ERROR | context={context} | {error_msg}")


def log_liveness_check(passed, detail=""):
    """Log a liveness detection result."""
    status = "PASSED" if passed else "FAILED"
    _logger.info(f"LIVENESS | status={status} | {detail}")
