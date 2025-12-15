"""
FOLLOWUP Listening Window - Implementation Summary

This document summarizes the professional FOLLOWUP feature implementation for Wyzer AI Assistant.
"""

# ============================================================================
# SUMMARY
# ============================================================================

## GOAL
After every completed TTS response, Wyzer enters a FOLLOWUP state where hotword detection is
temporarily disabled. The user can say a follow-up WITHOUT hotword detection for ~3 seconds 
(default, configurable). Timer extends on speech. Ends on silence or explicit exit phrase.
Fully deterministic (no LLM needed to decide follow-up).

## FILES MODIFIED/CREATED

### 1. wyzer/core/config.py
   - ADDED: FOLLOWUP_ENABLED (default: True)
   - ADDED: FOLLOWUP_TIMEOUT_SEC (default: 3.0)
   - ADDED: FOLLOWUP_MAX_CHAIN (default: 3)

### 2. wyzer/core/state.py
   - ADDED: AssistantState.FOLLOWUP enum value
   - State machine now includes: IDLE, HOTWORD_DETECTED, LISTENING, TRANSCRIBING, THINKING, SPEAKING, FOLLOWUP

### 3. wyzer/core/followup_manager.py [NEW FILE]
   - FollowupManager class encapsulates FOLLOWUP behavior
   - Key methods:
     * start_followup_window() - Initialize FOLLOWUP listening
     * is_followup_active() - Check if FOLLOWUP window is active
     * reset_speech_timer() - Extend timeout on detected speech
     * check_timeout() - Check if silence timeout exceeded
     * is_exit_phrase(text) - Detect exit phrases (normalized, substring match)
     * end_followup_window() - Explicitly end FOLLOWUP
     * increment_chain() - Track follow-up chain depth
     * get_chain_count() - Get current chain count
     * get_remaining_time() - Get remaining time in window
   - Exit phrases (case-insensitive, punctuation-insensitive):
     "no", "nope", "stop", "cancel", "that's all"/"thats all", 
     "never mind"/"nevermind", "nothing else", "all good"

### 4. wyzer/stt/stt_router.py
   - UPDATED: transcribe() method now accepts optional mode parameter
   - mode="followup" can be used for future mode-specific STT processing

### 5. wyzer/core/assistant.py (WyzerAssistant - single process)
   - ADDED: Import FollowupManager
   - ADDED: self.followup_manager instance variable
   - ADDED: self._is_followup_response flag to track if response is from FOLLOWUP
   - ADDED: _process_followup() method to handle FOLLOWUP audio frames
   - ADDED: _transcribe_followup() and _background_transcribe_followup() for FOLLOWUP STT
   - ADDED: _start_followup_window() to initiate FOLLOWUP listening
   - ADDED: _re_enter_followup() for follow-up chain re-entry
   - MODIFIED: _main_loop() to handle FOLLOWUP state and timeout checks
   - MODIFIED: _process_listening() to call _process_followup() for FOLLOWUP state
   - MODIFIED: _check_transcription_complete() to:
     * Detect exit phrases during FOLLOWUP
     * Pass is_followup flag to _think_and_respond()
     * Re-enter FOLLOWUP if transcript is non-empty and not exit phrase
   - MODIFIED: _think_and_respond() to accept is_followup parameter
   - MODIFIED: _background_think() to store is_followup in result dict
   - MODIFIED: _check_thinking_complete() to handle FOLLOWUP responses
   - MODIFIED: _speak_and_reset() to accept is_followup parameter
   - MODIFIED: _check_speaking_complete() to:
     * Call _start_followup_window() after TTS instead of _reset_to_idle()
     * Handle FOLLOWUP response re-entry

### 6. wyzer/core/assistant.py (WyzerAssistantMultiprocess)
   - ADDED: Import FollowupManager
   - ADDED: self.followup_manager instance variable
   - ADDED: _process_followup() method for FOLLOWUP audio handling
   - ADDED: _send_audio_to_brain_followup() to mark audio as followup for orchestrator
   - ADDED: _start_followup_window() to enter FOLLOWUP state
   - MODIFIED: _main_loop() to process FOLLOWUP state and check timeouts
   - MODIFIED: _poll_brain_messages() to:
     * Detect exit phrases in FOLLOWUP responses
     * Trigger _start_followup_window() when TTS finishes
     * Track is_followup flag in brain messages

### 7. scripts/test_followup_manager.py [NEW FILE]
   - Comprehensive unit test suite for FollowupManager
   - Test classes:
     * TestExitPhraseDetection - 10+ tests for phrase matching
     * TestTimeoutBehavior - Tests for timeout and timer reset
     * TestChainBehavior - Tests for chain counting and max limits
     * TestNormalization - Tests for text normalization
     * TestIntegration - Full integration tests
   - ~200 lines, pytest-compatible

### 8. README.md
   - UPDATED: Features list to mention FOLLOWUP
   - UPDATED: State machine description to include FOLLOWUP
   - ADDED: Comprehensive FOLLOWUP section covering:
     * How It Works
     * Example Interaction
     * Configuration Options
     * Exit Phrases
     * Deterministic Design
     * How to Disable
   - UPDATED: Project Structure section with followup_manager.py
   - UPDATED: Configuration section with FOLLOWUP environment variables

## BEHAVIOR FLOW

### Normal Flow (no FOLLOWUP):
1. User says hotword
2. Assistant listens, transcribes, thinks, speaks response
3. Return to IDLE
4. Back to step 1 (wait for hotword)

### With FOLLOWUP (new):
1. User says hotword
2. Assistant listens, transcribes, thinks, speaks response
3. **ENTER FOLLOWUP** (3 seconds, hotword disabled)
4. User says follow-up WITHOUT hotword
5. Assistant listens (no hotword needed), transcribes, thinks, speaks
6. **RE-ENTER FOLLOWUP** (chain count: 1)
7. User says another follow-up
8. Assistant responds
9. **RE-ENTER FOLLOWUP** (chain count: 2)
10. User says "all good" (exit phrase)
11. **EXIT FOLLOWUP**, return to IDLE
12. Wait for hotword

### Exit Conditions:
- Silence for FOLLOWUP_TIMEOUT_SEC (3.0s)
- User says exit phrase
- Max chain depth reached (3 consecutive follow-ups)

## DESIGN PRINCIPLES

✓ **Deterministic**: No LLM needed to decide if follow-up; timer + phrase matching only
✓ **Interrupt-Safe**: Uses existing mic/VAD pipeline; no device re-opening
✓ **Thread-Safe**: Integrates cleanly with both single-process and multiprocess implementations
✓ **Hotword Respect**: Hotword loop only runs in IDLE state; skipped during FOLLOWUP/LISTENING/RESPONDING
✓ **Extensible**: Config options (timeout, max chain) and exit phrases easily customizable
✓ **Logged**: Non-spammy [STATE] log entries for transparency
✓ **Windows Compatible**: Pure Python; no OS-specific assumptions beyond existing code

## CONFIGURATION

Environment variables (or edit config.py):
```
WYZER_FOLLOWUP_ENABLED=true         # Enable/disable feature
WYZER_FOLLOWUP_TIMEOUT_SEC=3.0      # Silence timeout (seconds)
WYZER_FOLLOWUP_MAX_CHAIN=3          # Max follow-up depth
```

## TESTING

Run unit tests:
```bash
pytest scripts/test_followup_manager.py -v
```

Tests cover:
- Exit phrase detection (exact, case-insensitive, with punctuation, substrings)
- Timeout behavior and timer reset
- Chain counting and max limits
- Text normalization
- Full integration scenarios

## CODE QUALITY

✓ Type hints added where practical
✓ Docstrings for all public methods
✓ No breaking changes to existing entrypoints
✓ Windows compatibility maintained
✓ ~300 lines of new feature code
✓ ~200 lines of test code
✓ ~100 lines of documentation

## EXAMPLE USAGE

```bash
# Run with default FOLLOWUP (enabled, 3 second timeout)
python run.py

# Disable FOLLOWUP (original behavior)
python run.py  # (after setting WYZER_FOLLOWUP_ENABLED=false)

# Custom timeout and chain limit
set WYZER_FOLLOWUP_TIMEOUT_SEC=5.0
set WYZER_FOLLOWUP_MAX_CHAIN=2
python run.py
```

## NEXT STEPS (FUTURE)

Possible enhancements:
- Add confidence scoring to follow-up detection
- Implement contextual exit phrase filtering
- Add metrics/telemetry for FOLLOWUP success rate
- Integration with orchestrator for context-aware responses
- Custom voice cues or earcons for FOLLOWUP state transitions
