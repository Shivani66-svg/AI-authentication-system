"""
=============================================================================
  VOICE ASSISTANT MODULE — Text-to-Speech announcements
  Uses pyttsx3 for offline speech synthesis on Windows.
=============================================================================
"""

import threading
import pyttsx3


class VoiceAssistant:
    """Thread-safe voice assistant for announcements."""

    def __init__(self, rate=170, volume=1.0):
        self.rate = rate
        self.volume = volume
        self._lock = threading.Lock()

    def _speak_sync(self, text):
        """Speak text synchronously (blocks until done)."""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)

            # Try to use a female voice if available
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)

            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"  [VOICE ASSISTANT] Speech error: {e}")

    def speak(self, text):
        """Speak text in a background thread (non-blocking)."""
        thread = threading.Thread(target=self._speak_sync, args=(text,), daemon=True)
        thread.start()


# Global assistant instance
assistant = VoiceAssistant()


def say(text):
    """Convenience function to speak text (non-blocking)."""
    assistant.speak(text)


def say_wait(text):
    """Speak text and WAIT until speech is completely finished (blocking)."""
    assistant._speak_sync(text)
