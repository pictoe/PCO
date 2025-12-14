"""
FOLLOWUP listening window manager for Wyzer AI Assistant.

After every completed TTS response, Wyzer enters a FOLLOWUP state where hotword
detection is temporarily disabled. The user can say a follow-up WITHOUT hotword
detection for approximately FOLLOWUP_TIMEOUT_SEC (default 3.0 seconds).

The timer is extended (reset) on each detected speech. FOLLOWUP ends when:
1. Silence/no valid speech occurs for FOLLOWUP_TIMEOUT_SEC, OR
2. User says an explicit end phrase like "no", "nope", "that's all", etc.

If user says any other text, treat it as a normal query and allow chaining
(re-enter FOLLOWUP after responding).
"""
import re
import time
from typing import Optional, List
from wyzer.core.logger import get_logger
from wyzer.core.config import Config


class FollowupManager:
    """Manager for FOLLOWUP listening window behavior"""
    
    # Exit phrases: if transcript matches any of these (substring, case-insensitive),
    # exit FOLLOWUP and return to IDLE
    EXIT_PHRASES: List[str] = [
        "no",
        "nope",
        "that's all",
        "thats all",
        "stop",
        "cancel",
        "never mind",
        "nevermind",
        "nothing else",
        "all good",
    ]
    
    def __init__(self):
        """Initialize FollowupManager"""
        self.logger = get_logger()
        self._followup_active: bool = False
        self._window_start_time: float = 0.0
        self._last_speech_time: float = 0.0
        self._chain_count: int = 0
        self._grace_period_duration: float = 1.5  # 1.5 seconds to let TTS prompt finish
    
    def start_followup_window(self) -> None:
        """
        Start a new FOLLOWUP listening window.
        Called after TTS completes and we want to listen for follow-ups.
        """
        if not Config.FOLLOWUP_ENABLED:
            return
        
        self._followup_active = True
        self._window_start_time = time.time()
        self._last_speech_time = self._window_start_time
        self._chain_count = 0
        
        self.logger.info("[STATE] FOLLOWUP: listening (hotword disabled)")
    
    def is_followup_active(self) -> bool:
        """Check if FOLLOWUP window is currently active"""
        return self._followup_active and Config.FOLLOWUP_ENABLED
    
    def reset_speech_timer(self) -> None:
        """Reset the silence timer (called when speech is detected)"""
        if not self.is_followup_active():
            return
        self._last_speech_time = time.time()
    
    def check_timeout(self) -> bool:
        """
        Check if FOLLOWUP window has timed out (silence for too long).
        
        Returns:
            True if timed out (should exit FOLLOWUP), False otherwise
        """
        if not self.is_followup_active():
            return False
        
        current_time = time.time()
        time_since_speech = current_time - self._last_speech_time
        
        if time_since_speech >= Config.FOLLOWUP_TIMEOUT_SEC:
            self.logger.info(
                f"[STATE] FOLLOWUP: timeout after {time_since_speech:.2f}s silence"
            )
            self._followup_active = False
            return True
        
        return False
    
    def is_in_grace_period(self) -> bool:
        """
        Check if FOLLOWUP is in grace period (TTS prompt still playing).
        During grace period, ignore VAD speech detection to avoid picking up TTS audio.
        
        Returns:
            True if still in grace period, False if listening is active
        """
        if not self.is_followup_active():
            return False
        
        current_time = time.time()
        time_since_start = current_time - self._window_start_time
        return time_since_start < self._grace_period_duration
    
    def is_exit_phrase(self, text: str) -> bool:
        """
        Check if transcript matches an exit phrase.
        
        Normalizes text (lowercase, strip punctuation, strip whitespace)
        and checks if text is PRIMARILY an exit phrase (not just contains one).
        
        Uses word-boundary matching to avoid false positives like:
        - "how much snow have we got in the past seven days" should NOT match "nothing else"
        - "no thanks" should match "no"
        - "stop" should match "stop"
        
        Args:
            text: User's transcript
            
        Returns:
            True if text matches exit phrase, False otherwise
        """
        if not text:
            return False
        
        # Normalize: lowercase, remove punctuation, strip whitespace
        normalized = self._normalize_text(text)
        words = normalized.split()
        
        # Check for exact phrase matches or leading phrase matches
        for phrase in self.EXIT_PHRASES:
            phrase_normalized = self._normalize_text(phrase)
            phrase_words = phrase_normalized.split()
            
            # Exact match
            if normalized == phrase_normalized:
                self.logger.info(f"[FOLLOWUP] Exit phrase detected: '{text}' -> '{normalized}'")
                return True
            
            # If text starts with the phrase, it's probably an exit + extra words
            # e.g., "no thanks" starts with "no"
            if len(phrase_words) > 0 and len(words) >= len(phrase_words):
                if words[:len(phrase_words)] == phrase_words:
                    self.logger.info(f"[FOLLOWUP] Exit phrase detected: '{text}' -> '{normalized}'")
                    return True
        
        return False
    
    def end_followup_window(self) -> None:
        """Explicitly end the FOLLOWUP window (e.g., when user says exit phrase)"""
        if self._followup_active:
            self.logger.info("[STATE] FOLLOWUP: ended")
            self._followup_active = False
            self._chain_count = 0
    
    def increment_chain(self) -> bool:
        """
        Increment the follow-up chain counter.
        
        Returns:
            True if chain limit not exceeded, False if max chain depth reached
        """
        self._chain_count += 1
        if self._chain_count > Config.FOLLOWUP_MAX_CHAIN:
            self.logger.info(
                f"[FOLLOWUP] Max chain depth ({Config.FOLLOWUP_MAX_CHAIN}) reached"
            )
            self._followup_active = False
            return False
        return True
    
    def get_chain_count(self) -> int:
        """Get current chain count"""
        return self._chain_count
    
    def get_remaining_time(self) -> float:
        """
        Get remaining time in FOLLOWUP window (seconds).
        
        Returns:
            Remaining time in seconds, or 0 if window is expired/inactive
        """
        if not self.is_followup_active():
            return 0.0
        
        current_time = time.time()
        time_since_speech = current_time - self._last_speech_time
        remaining = Config.FOLLOWUP_TIMEOUT_SEC - time_since_speech
        
        return max(0.0, remaining)
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text for exit phrase matching.
        
        - Lowercase
        - Remove punctuation (keep spaces)
        - Strip leading/trailing whitespace
        - Collapse multiple spaces to single space
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Lowercase
        normalized = text.lower()
        
        # Remove punctuation (keep alphanumeric, spaces)
        # This regex keeps a-z, 0-9, and spaces only
        normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
        
        # Collapse multiple spaces to single space
        normalized = re.sub(r"\s+", " ", normalized)
        
        # Strip leading/trailing whitespace
        normalized = normalized.strip()
        
        return normalized
