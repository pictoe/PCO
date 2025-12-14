"""wyzer.core.assistant

Contains the original single-process `WyzerAssistant` and the new multiprocess
split implementation `WyzerAssistantMultiprocess`.

Multiprocess architecture:
- Process A (realtime core): mic stream + VAD + hotword + state machine
- Process B (brain worker): STT + LLM + tools + TTS

The original class is kept for `--single-process` debugging.
"""
import time
import os
import numpy as np
import threading
import tempfile
import wave
from queue import Queue, Empty
from typing import Optional, List, Any, Dict
from wyzer.core.config import Config
from wyzer.core.logger import get_logger
from wyzer.core.state import AssistantState, RuntimeState
from wyzer.audio.mic_stream import MicStream
from wyzer.audio.vad import VadDetector
from wyzer.audio.hotword import HotwordDetector
from wyzer.audio.audio_utils import concat_audio_frames
from wyzer.audio.audio_utils import audio_to_int16
from wyzer.stt.stt_router import STTRouter
from wyzer.brain.llm_engine import LLMEngine
from wyzer.tts.tts_router import TTSRouter
from wyzer.core.ipc import new_id, safe_put
from wyzer.core.process_manager import start_brain_process, stop_brain_process


class WyzerAssistant:
    """Main Wyzer Assistant coordinator"""
    
    def __init__(
        self,
        enable_hotword: bool = True,
        whisper_model: str = "small",
        whisper_device: str = "cpu",
        audio_device: Optional[int] = None,
        llm_mode: str = "ollama",
        ollama_model: str = "llama3.1:latest",
        ollama_url: str = "http://127.0.0.1:11434",
        llm_timeout: int = 30,
        tts_enabled: bool = True,
        tts_engine: str = "piper",
        piper_exe_path: str = "./wyzer/assets/piper/piper.exe",
        piper_model_path: str = "./wyzer/assets/piper/en_US-lessac-medium.onnx",
        piper_speaker_id: Optional[int] = None,
        tts_output_device: Optional[int] = None,
        speak_hotword_interrupt: bool = True
    ):
        """
        Initialize Wyzer Assistant
        
        Args:
            enable_hotword: Enable hotword detection
            whisper_model: Whisper model size
            whisper_device: Device for Whisper
            audio_device: Optional audio device index
            llm_mode: LLM mode ("ollama" or "off")
            ollama_model: Ollama model name
            ollama_url: Ollama API base URL
            llm_timeout: LLM request timeout in seconds
            tts_enabled: Enable TTS
            tts_engine: TTS engine ("piper")
            piper_exe_path: Path to Piper executable
            piper_model_path: Path to Piper model
            piper_speaker_id: Optional Piper speaker ID
            tts_output_device: Optional TTS output device
            speak_hotword_interrupt: Enable barge-in during speaking
        """
        self.logger = get_logger()
        self.enable_hotword = enable_hotword
        self.speak_hotword_interrupt = speak_hotword_interrupt
        
        # Initialize state
        self.state = RuntimeState()
        self.running = False
        
        # Hotword cooldown tracking
        self.last_hotword_time: float = 0.0
        
        # Post-barge-in speech gating
        self._bargein_pending_speech: bool = False
        self._bargein_wait_speech_deadline_ts: float = 0.0
        self._speaking_start_ts: float = 0.0
        
        # Background transcription
        self.stt_thread: Optional[threading.Thread] = None
        self.stt_result: Optional[str] = None
        
        # Background thinking (Phase 4)
        self.thinking_thread: Optional[threading.Thread] = None
        self.thinking_result: Optional[dict] = None
        
        # Background speaking (Phase 5)
        self.speaking_thread: Optional[threading.Thread] = None
        self.tts_stop_event: threading.Event = threading.Event()
        
        # Audio buffer for recording
        self.audio_buffer: List[np.ndarray] = []
        
        # Initialize components
        self.logger.info("Initializing Wyzer Assistant...")
        
        # Audio stream
        self.audio_queue: Queue = Queue(maxsize=Config.AUDIO_QUEUE_MAX_SIZE)
        self.mic_stream = MicStream(
            audio_queue=self.audio_queue,
            device=audio_device
        )
        
        # VAD
        self.vad = VadDetector()
        
        # Hotword detector (if enabled)
        self.hotword: Optional[HotwordDetector] = None
        if self.enable_hotword:
            try:
                self.hotword = HotwordDetector()
                self.logger.info(f"Hotword detection enabled for: {Config.HOTWORD_KEYWORDS}")
            except Exception as e:
                self.logger.error(f"Failed to initialize hotword detector: {e}")
                self.logger.warning("Continuing without hotword detection")
                self.enable_hotword = False
        
        # STT router
        self.stt = STTRouter(
            whisper_model=whisper_model,
            whisper_device=whisper_device
        )
        
        # LLM Brain (Phase 4)
        # LLM Brain (Phase 4)
        self.brain: Optional[LLMEngine] = None
        if llm_mode == "ollama":
            self.brain = LLMEngine(
                base_url=ollama_url,
                model=ollama_model,
                timeout=llm_timeout,
                enabled=True
            )
        else:
            self.logger.info("LLM brain disabled (STT-only mode)")
        
        # TTS (Phase 5)
        self.tts: Optional[TTSRouter] = None
        if tts_enabled:
            try:
                self.tts = TTSRouter(
                    engine=tts_engine,
                    piper_exe_path=piper_exe_path,
                    piper_model_path=piper_model_path,
                    piper_speaker_id=piper_speaker_id,
                    output_device=tts_output_device,
                    enabled=True
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize TTS: {e}")
                self.logger.warning("TTS will be disabled")
                self.tts = None
        else:
            self.logger.info("TTS disabled")
        
        self.logger.info("Wyzer Assistant initialized successfully")
    
    def start(self) -> None:
        """Start the assistant"""
        if self.running:
            self.logger.warning("Assistant already running")
            return
        
        self.logger.info("Starting Wyzer Assistant...")
        self.running = True
        
        # Start audio stream
        self.mic_stream.start()
        
        # If hotword disabled, immediately start listening
        if not self.enable_hotword:
            self.logger.info("No-hotword mode: Starting immediate listening")
            self.state.transition_to(AssistantState.LISTENING)
            self.logger.info("Listening... (speak now)")
        else:
            self.logger.info(f"Listening for hotword: {Config.HOTWORD_KEYWORDS}")
        
        # Run main loop
        try:
            self._main_loop()
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the assistant"""
        if not self.running:
            return
        
        self.logger.info("Stopping Wyzer Assistant...")
        self.running = False
        
        # Stop TTS if speaking
        if self.tts_stop_event:
            self.tts_stop_event.set()
        
        # Stop audio stream
        self.mic_stream.stop()
        
        # Wait briefly for threads to finish
        if self.speaking_thread and self.speaking_thread.is_alive():
            self.speaking_thread.join(timeout=0.5)
        
        self.logger.info("Wyzer Assistant stopped")
    
    def _main_loop(self) -> None:
        """Main processing loop"""
        while self.running:
            # Get audio frame from queue (with timeout)
            try:
                audio_frame = self.audio_queue.get(timeout=0.1)
            except Empty:
                # Check if transcription thread finished
                if self.state.is_in_state(AssistantState.TRANSCRIBING):
                    self._check_transcription_complete()
                # Check if thinking thread finished
                elif self.state.is_in_state(AssistantState.THINKING):
                    self._check_thinking_complete()
                # Check if speaking thread finished
                elif self.state.is_in_state(AssistantState.SPEAKING):
                    self._check_speaking_complete()
                continue
            
            # Process frame based on current state
            if self.state.is_in_state(AssistantState.IDLE):
                self._process_idle(audio_frame)
            
            elif self.state.is_in_state(AssistantState.LISTENING):
                self._process_listening(audio_frame)
            
            elif self.state.is_in_state(AssistantState.TRANSCRIBING):
                # In transcribing state, drain frames to prevent queue overflow
                # Don't process them, just discard
                # Also check if transcription is complete
                self._check_transcription_complete()
            
            elif self.state.is_in_state(AssistantState.THINKING):
                # In thinking state, drain frames and check if thinking is complete
                self._check_thinking_complete()
            
            elif self.state.is_in_state(AssistantState.SPEAKING):
                # In speaking state, drain frames BUT check for hotword interrupt
                self._process_speaking(audio_frame)
                self._check_speaking_complete()
    
    def _process_idle(self, audio_frame: np.ndarray) -> None:
        """
        Process audio frame in IDLE state
        
        Args:
            audio_frame: Audio frame to process
        """
        # In IDLE, check for hotword (if enabled)
        if self.enable_hotword and self.hotword:
            current_time = time.time()
            detected_keyword, score = self.hotword.detect(audio_frame)
            
            if detected_keyword:
                # Check cooldown period
                time_since_last = current_time - self.last_hotword_time
                
                if time_since_last < Config.HOTWORD_COOLDOWN_SEC:
                    self.logger.debug(
                        f"Hotword cooldown active: {time_since_last:.2f}s < {Config.HOTWORD_COOLDOWN_SEC}s (score: {score:.3f})"
                    )
                    return
                
                self.logger.info(f"Hotword '{detected_keyword}' accepted after cooldown")
                self.last_hotword_time = current_time
                
                # Drain a bit more to remove residual hotword audio
                self._drain_audio_queue(Config.POST_IDLE_DRAIN_SEC)
                
                # Transition to LISTENING
                self.state.transition_to(AssistantState.LISTENING)
                self.audio_buffer = []
                self.logger.info("Listening... (speak now)")
    
    def _process_listening(self, audio_frame: np.ndarray) -> None:
        """
        Process audio frame in LISTENING state
        
        Args:
            audio_frame: Audio frame to process
        """
        # Check if we're in post-barge-in waiting for speech mode
        if self._bargein_pending_speech:
            current_time = time.time()
            
            # Check if wait deadline has expired
            if current_time > self._bargein_wait_speech_deadline_ts:
                self.logger.info("Post-barge-in speech wait timeout - returning to IDLE")
                self._clear_bargein_flags()
                self._reset_to_idle()
                return
        
        # Add frame to buffer
        self.audio_buffer.append(audio_frame)
        self.state.total_frames_recorded += 1
        
        # Check for speech using VAD
        is_speech = self.vad.is_speech(audio_frame)
        
        if is_speech:
            # Speech detected
            if not self.state.speech_detected:
                self.logger.debug("Speech started")
                self.state.speech_detected = True
                
                # Clear barge-in pending flag when speech actually starts
                if self._bargein_pending_speech:
                    self.logger.debug("Speech started after barge-in - clearing pending flag")
                    self._bargein_pending_speech = False
            
            self.state.speech_frames_count += 1
            self.state.silence_frames = 0
        else:
            # No speech in this frame
            if self.state.speech_detected:
                # We've detected speech before, so this is silence after speech
                self.state.silence_frames += 1
        
        # Check stop conditions
        should_stop = False
        stop_reason = ""
        
        # 1. Silence timeout after speech
        if (self.state.speech_detected and 
            self.state.silence_frames >= Config.get_silence_timeout_frames()):
            should_stop = True
            stop_reason = "silence timeout"
        
        # 2. Maximum recording duration
        if self.state.total_frames_recorded >= Config.get_max_record_frames():
            should_stop = True
            stop_reason = "max duration"
        
        # 3. In no-hotword mode, stop after one utterance
        if (not self.enable_hotword and 
            self.state.speech_detected and 
            self.state.silence_frames >= Config.get_silence_timeout_frames()):
            should_stop = True
            stop_reason = "utterance complete (no-hotword mode)"
        
        if should_stop:
            self.logger.info(f"Recording stopped: {stop_reason}")
            
            # Only transcribe if we detected some speech
            if self.state.speech_frames_count > 0:
                self._transcribe_and_reset()
            else:
                self.logger.warning("No speech detected in recording")
                self._reset_to_idle()
    
    def _transcribe_and_reset(self) -> None:
        """Start background transcription and transition to TRANSCRIBING"""
        # Clear barge-in flags when completing listening
        self._clear_bargein_flags()
        
        # Transition to TRANSCRIBING
        self.state.transition_to(AssistantState.TRANSCRIBING)
        
        # Concatenate audio buffer
        audio_data = concat_audio_frames(self.audio_buffer)
        
        self.logger.info(
            f"Starting transcription of {len(audio_data)/Config.SAMPLE_RATE:.2f}s audio in background..."
        )
        
        # Start transcription in background thread
        self.stt_result = None
        self.stt_thread = threading.Thread(
            target=self._background_transcribe,
            args=(audio_data,),
            daemon=True
        )
        self.stt_thread.start()
    
    def _background_transcribe(self, audio_data: np.ndarray) -> None:
        """Background thread for STT processing"""
        try:
            transcript = self.stt.transcribe(audio_data)
            self.stt_result = transcript
        except Exception as e:
            self.logger.error(f"Error in background transcription: {e}")
            self.stt_result = None
    
    def _check_transcription_complete(self) -> None:
        """Check if background transcription is complete and handle result"""
        if self.stt_thread and not self.stt_thread.is_alive():
            # Thread finished
            transcript = self.stt_result
            
            # Display transcript
            if transcript:
                self.logger.info(f"Transcript: {transcript}")
                print(f"\nYou: {transcript}")
                
                # Pass to LLM brain if enabled
                if self.brain:
                    self._think_and_respond(transcript)
                else:
                    # No brain, just show transcript and return to idle
                    if not self.enable_hotword:
                        self.logger.info("No-hotword mode: Exiting after transcription")
                        self.running = False
                    else:
                        self._reset_to_idle()
            else:
                self.logger.warning("No valid transcript (empty or filtered as garbage)")
                
                # In no-hotword mode, exit even if no transcript
                if not self.enable_hotword:
                    self.logger.info("No-hotword mode: Exiting")
                    self.running = False
                else:
                    self._reset_to_idle()
    
    def _reset_to_idle(self) -> None:
        """Reset state to IDLE with queue draining"""
        # Clear barge-in flags when returning to idle
        self._clear_bargein_flags()
        
        self.audio_buffer = []
        
        # Drain audio queue for POST_IDLE_DRAIN_SEC to remove residual wake audio
        drain_frames = int(Config.POST_IDLE_DRAIN_SEC * Config.SAMPLE_RATE / Config.CHUNK_SAMPLES)
        drained_count = 0
        
        self.logger.debug(f"Draining audio queue for {Config.POST_IDLE_DRAIN_SEC}s ({drain_frames} frames)...")
        
        for _ in range(drain_frames):
            try:
                frame = self.audio_queue.get_nowait()
                drained_count += 1
                
                # Process frame through hotword detector to update prev_scores
                # but ignore any detections during drain period
                if self.hotword:
                    self.hotword.detect(frame)
                    
            except Empty:
                break
        
        if drained_count > 0:
            self.logger.debug(f"Drained {drained_count} frames from queue")
        
        self.state.transition_to(AssistantState.IDLE)
        
        if self.enable_hotword:
            self.logger.info(f"Ready. Listening for hotword: {Config.HOTWORD_KEYWORDS}")
    
    def _think_and_respond(self, transcript: str) -> None:
        """Start background LLM thinking and transition to THINKING state"""
        # Transition to THINKING
        self.state.transition_to(AssistantState.THINKING)
        
        self.logger.info("Thinking...")
        
        # Start thinking in background thread
        self.thinking_result = None
        self.thinking_thread = threading.Thread(
            target=self._background_think,
            args=(transcript,),
            daemon=True
        )
        self.thinking_thread.start()
    
    def _background_think(self, transcript: str) -> None:
        """Background thread for LLM processing"""
        try:
            # Use orchestrator for Phase 6 tool support
            from wyzer.core.orchestrator import handle_user_text
            result_dict = handle_user_text(transcript)
            
            # Convert orchestrator result to expected format
            self.thinking_result = {
                "reply": result_dict.get("reply", ""),
                "confidence": 0.8,
                "model": Config.OLLAMA_MODEL,
                "latency_ms": result_dict.get("latency_ms", 0)
            }
        except Exception as e:
            self.logger.error(f"Error in background thinking: {e}")
            self.thinking_result = {
                "reply": "I encountered an error processing your request.",
                "confidence": 0.3,
                "model": "error",
                "latency_ms": 0
            }
    
    def _check_thinking_complete(self) -> None:
        """Check if background thinking is complete and handle result"""
        if self.thinking_thread and not self.thinking_thread.is_alive():
            # Thread finished
            result = self.thinking_result
            
            if result:
                # Display response
                reply = result.get("reply", "")
                latency = result.get("latency_ms", 0)
                
                self.logger.info(f"Response generated in {latency}ms")
                print(f"\nWyzer: {reply}\n")
                
                # Speak response if TTS enabled
                if self.tts:
                    self._speak_and_reset(reply)
                else:
                    # No TTS, just go to idle or exit
                    if not self.enable_hotword:
                        self.logger.info("No-hotword mode: Exiting after response")
                        self.running = False
                    else:
                        self._reset_to_idle()
            else:
                self.logger.warning("No response from LLM brain")
                
                # In no-hotword mode, exit even if no response
                if not self.enable_hotword:
                    self.logger.info("No-hotword mode: Exiting")
                    self.running = False
                else:
                    self._reset_to_idle()
    
    def _speak_and_reset(self, text: str) -> None:
        """Start background speaking and transition to SPEAKING state"""
        # Transition to SPEAKING
        self.state.transition_to(AssistantState.SPEAKING)
        
        self.logger.info("Speaking...")
        
        # Clear stop event
        self.tts_stop_event.clear()
        
        # Record speaking start time for cooldown check
        self._speaking_start_ts = time.time()
        
        # Start speaking in background thread
        self.speaking_thread = threading.Thread(
            target=self._background_speak,
            args=(text,),
            daemon=True
        )
        self.speaking_thread.start()
    
    def _background_speak(self, text: str) -> None:
        """Background thread for TTS processing"""
        try:
            self.tts.speak(text, self.tts_stop_event)
        except Exception as e:
            self.logger.error(f"Error in background speaking: {e}")
    
    def _process_speaking(self, audio_frame: np.ndarray) -> None:
        """
        Process audio frame in SPEAKING state
        Check for hotword interrupt (barge-in)
        
        Args:
            audio_frame: Audio frame to process
        """
        # Only check for hotword interrupt if enabled
        if not self.speak_hotword_interrupt or not self.enable_hotword or not self.hotword:
            return
        
        current_time = time.time()

        # Always run hotword detection to keep detector state up-to-date,
        # then decide whether to act on a detection.
        detected_keyword, score = self.hotword.detect(audio_frame)
        
        # Check if speaking just started (prevent immediate interrupt from residual hotword)
        time_since_speak_start = current_time - self._speaking_start_ts
        if time_since_speak_start <= Config.SPEAK_START_COOLDOWN_SEC:
            # Too soon after speaking started, ignore hotword triggers
            return
        
        # Additionally check if waiting for speech after barge-in
        if self._bargein_pending_speech:
            # Still waiting for speech start after previous barge-in
            return
        
        if detected_keyword:
            # Check cooldown period
            time_since_last = current_time - self.last_hotword_time
            
            if time_since_last < Config.HOTWORD_COOLDOWN_SEC:
                self.logger.debug(
                    f"Hotword cooldown active during speaking: {time_since_last:.2f}s < {Config.HOTWORD_COOLDOWN_SEC}s"
                )
                return
            
            self.logger.info(f"Hotword '{detected_keyword}' detected - interrupting speech (barge-in)")
            self.last_hotword_time = current_time
            
            # Set barge-in pending speech flags
            if Config.POST_BARGEIN_REQUIRE_SPEECH_START:
                self._bargein_pending_speech = True
                self._bargein_wait_speech_deadline_ts = current_time + Config.POST_BARGEIN_WAIT_FOR_SPEECH_SEC
                self.logger.info(
                    f"Post-barge-in speech gating enabled; "
                    f"waiting for speech start up to {Config.POST_BARGEIN_WAIT_FOR_SPEECH_SEC}s"
                )
            
            # Stop TTS immediately
            self.tts_stop_event.set()
            
            # Wait briefly for speaking thread to stop
            if self.speaking_thread:
                self.speaking_thread.join(timeout=0.3)
            
            # Drain audio queue to remove stale frames
            self._drain_audio_queue(Config.POST_SPEAK_DRAIN_SEC)
            
            # Transition directly to LISTENING
            self.state.transition_to(AssistantState.LISTENING)
            self.audio_buffer = []
            
            # Clear speaking thread reference to prevent _check_speaking_complete from triggering
            self.speaking_thread = None
            
            self.logger.info("Listening... (speak now)")
    
    def _check_speaking_complete(self) -> None:
        """Check if background speaking is complete"""
        if self.speaking_thread and not self.speaking_thread.is_alive():
            # If we're in LISTENING state, barge-in already handled the transition
            if self.state.is_in_state(AssistantState.LISTENING):
                self.speaking_thread = None
                return
            
            # Speaking finished normally
            self.logger.debug("Speaking completed")
            
            # In no-hotword mode, exit after speaking
            if not self.enable_hotword:
                self.logger.info("No-hotword mode: Exiting after speaking")
                self.running = False
            else:
                # Drain queue and reset to IDLE
                self._drain_audio_queue(Config.POST_SPEAK_DRAIN_SEC)
                
                self.speaking_thread = None
                self.state.transition_to(AssistantState.IDLE)
                self.logger.info(f"Ready. Listening for hotword: {Config.HOTWORD_KEYWORDS}")
    
    def _drain_audio_queue(self, duration_sec: float) -> None:
        """Drain audio queue for specified duration"""
        drain_frames = int(duration_sec * Config.SAMPLE_RATE / Config.CHUNK_SAMPLES)
        drained_count = 0
        
        self.logger.debug(f"Draining audio queue for {duration_sec}s ({drain_frames} frames)...")
        
        for _ in range(drain_frames):
            try:
                frame = self.audio_queue.get_nowait()
                drained_count += 1
                
                # Process frame through hotword detector to update state
                # but ignore any detections during drain period
                if self.hotword:
                    self.hotword.detect(frame)
                    
            except Empty:
                break
        
        if drained_count > 0:
            self.logger.debug(f"Drained {drained_count} frames from queue")
    
    def _clear_bargein_flags(self) -> None:
        """Clear all barge-in related flags"""
        if self._bargein_pending_speech:
            self.logger.debug("Clearing barge-in pending speech flags")
        self._bargein_pending_speech = False
        self._bargein_wait_speech_deadline_ts = 0.0


class WyzerAssistantMultiprocess:
    """Realtime-core assistant that offloads heavy work to a Brain Worker process."""

    def __init__(
        self,
        enable_hotword: bool = True,
        whisper_model: str = "small",
        whisper_device: str = "cpu",
        whisper_compute_type: str = "int8",
        audio_device: Optional[int] = None,
        llm_mode: str = "ollama",
        ollama_model: str = "llama3.1:latest",
        ollama_url: str = "http://127.0.0.1:11434",
        llm_timeout: int = 30,
        tts_enabled: bool = True,
        tts_engine: str = "piper",
        piper_exe_path: str = "./wyzer/assets/piper/piper.exe",
        piper_model_path: str = "./wyzer/assets/piper/en_US-lessac-medium.onnx",
        piper_speaker_id: Optional[int] = None,
        tts_output_device: Optional[int] = None,
        speak_hotword_interrupt: bool = True,
        log_level: str = "INFO",
        ipc_queue_maxsize: int = 50,
    ):
        self.logger = get_logger()

        self.enable_hotword = enable_hotword
        self.speak_hotword_interrupt = speak_hotword_interrupt

        # State
        self.state = RuntimeState()
        self.running = False
        self.last_hotword_time: float = 0.0

        # Post-barge-in speech gating
        self._bargein_pending_speech: bool = False
        self._bargein_wait_speech_deadline_ts: float = 0.0
        self._speaking_start_ts: float = 0.0

        # Audio buffer
        self.audio_buffer: List[np.ndarray] = []

        # Audio stream
        self.audio_queue: Queue = Queue(maxsize=Config.AUDIO_QUEUE_MAX_SIZE)
        self.mic_stream = MicStream(audio_queue=self.audio_queue, device=audio_device)

        # VAD + hotword
        self.vad = VadDetector()
        self.hotword: Optional[HotwordDetector] = None
        if self.enable_hotword:
            try:
                self.hotword = HotwordDetector()
                self.logger.info(f"Hotword detection enabled for: {Config.HOTWORD_KEYWORDS}")
            except Exception as e:
                self.logger.error(f"Failed to initialize hotword detector: {e}")
                self.logger.warning("Continuing without hotword detection")
                self.enable_hotword = False

        # Brain worker process + IPC
        self._brain_proc = None
        self._core_to_brain_q = None
        self._brain_to_core_q = None

        # Brain speaking flag (driven by worker LOG events)
        self._brain_speaking: bool = False

        self._exit_after_tts: bool = False

        self._brain_config: Dict[str, Any] = {
            "log_level": log_level,
            "ipc_queue_maxsize": ipc_queue_maxsize,
            "whisper_model": whisper_model,
            "whisper_device": whisper_device,
            "whisper_compute_type": whisper_compute_type,
            "llm_mode": llm_mode,
            "ollama_model": ollama_model,
            "ollama_url": ollama_url,
            "llm_timeout": llm_timeout,
            "tts_enabled": bool(tts_enabled),
            "tts_engine": tts_engine,
            "piper_exe_path": piper_exe_path,
            "piper_model_path": piper_model_path,
            "piper_speaker_id": piper_speaker_id,
            "tts_output_device": tts_output_device,
        }

    def start(self) -> None:
        if self.running:
            self.logger.warning("Assistant already running")
            return

        self.logger.info("Starting Wyzer Assistant (multiprocess)...")
        self.running = True

        self._brain_proc, self._core_to_brain_q, self._brain_to_core_q = start_brain_process(self._brain_config)

        self.mic_stream.start()

        if not self.enable_hotword:
            self.logger.info("No-hotword mode: Starting immediate listening")
            self.state.transition_to(AssistantState.LISTENING)
            self.logger.info("Listening... (speak now)")
        else:
            self.logger.info(f"Listening for hotword: {Config.HOTWORD_KEYWORDS}")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            self.stop()

    def stop(self) -> None:
        if not self.running:
            return

        self.logger.info("Stopping Wyzer Assistant...")
        self.running = False

        try:
            self.mic_stream.stop()
        except Exception:
            pass

        if self._brain_proc and self._core_to_brain_q:
            stop_brain_process(self._brain_proc, self._core_to_brain_q)

        self.logger.info("Wyzer Assistant stopped")

    def _main_loop(self) -> None:
        while self.running:
            # Drain brain->core messages first to keep UI/logging snappy
            self._poll_brain_messages()

            try:
                audio_frame = self.audio_queue.get(timeout=0.05)
            except Empty:
                # Also allow hotword gating timeouts in LISTENING
                if self.state.is_in_state(AssistantState.LISTENING):
                    self._process_listening_timeout_tick()
                continue

            if self.state.is_in_state(AssistantState.IDLE):
                self._process_idle(audio_frame)
            elif self.state.is_in_state(AssistantState.LISTENING):
                self._process_listening(audio_frame)
            else:
                # In multiprocess mode, treat non-listening states like IDLE for hotword/barge-in.
                self._process_idle(audio_frame)

    def _process_idle(self, audio_frame: np.ndarray) -> None:
        if not (self.enable_hotword and self.hotword):
            return

        current_time = time.time()
        detected_keyword, score = self.hotword.detect(audio_frame)

        if not detected_keyword:
            return

        time_since_last = current_time - self.last_hotword_time
        if time_since_last < Config.HOTWORD_COOLDOWN_SEC:
            self.logger.debug(
                f"Hotword cooldown active: {time_since_last:.2f}s < {Config.HOTWORD_COOLDOWN_SEC}s (score: {score:.3f})"
            )
            return

        self.logger.info(f"Hotword '{detected_keyword}' accepted after cooldown")
        self.last_hotword_time = current_time

        # Hotword must barge-in instantly if TTS is speaking.
        # Only send INTERRUPT when the worker reports it's speaking.
        if self.speak_hotword_interrupt and self._brain_speaking and self._core_to_brain_q:
            safe_put(self._core_to_brain_q, {"type": "INTERRUPT", "reason": "hotword"})

        # Drain a bit more to remove residual hotword audio
        self._drain_audio_queue(Config.POST_IDLE_DRAIN_SEC)

        self.state.transition_to(AssistantState.LISTENING)
        self.audio_buffer = []
        self.logger.info("Listening... (speak now)")

    def _process_listening_timeout_tick(self) -> None:
        if not self._bargein_pending_speech:
            return
        current_time = time.time()
        if current_time > self._bargein_wait_speech_deadline_ts:
            self.logger.info("Post-barge-in speech wait timeout - returning to IDLE")
            self._clear_bargein_flags()
            self._reset_to_idle()

    def _process_listening(self, audio_frame: np.ndarray) -> None:
        # Post-barge-in speech gating
        if self._bargein_pending_speech:
            current_time = time.time()
            if current_time > self._bargein_wait_speech_deadline_ts:
                self.logger.info("Post-barge-in speech wait timeout - returning to IDLE")
                self._clear_bargein_flags()
                self._reset_to_idle()
                return

        self.audio_buffer.append(audio_frame)
        self.state.total_frames_recorded += 1

        is_speech = self.vad.is_speech(audio_frame)
        if is_speech:
            if not self.state.speech_detected:
                self.logger.debug("Speech started")
                self.state.speech_detected = True
                if self._bargein_pending_speech:
                    self.logger.debug("Speech started after barge-in - clearing pending flag")
                    self._bargein_pending_speech = False
            self.state.speech_frames_count += 1
            self.state.silence_frames = 0
        else:
            if self.state.speech_detected:
                self.state.silence_frames += 1

        should_stop = False
        stop_reason = ""
        if (
            self.state.speech_detected
            and self.state.silence_frames >= Config.get_silence_timeout_frames()
        ):
            should_stop = True
            stop_reason = "silence timeout"
        if self.state.total_frames_recorded >= Config.get_max_record_frames():
            should_stop = True
            stop_reason = "max duration"
        if (
            not self.enable_hotword
            and self.state.speech_detected
            and self.state.silence_frames >= Config.get_silence_timeout_frames()
        ):
            should_stop = True
            stop_reason = "utterance complete (no-hotword mode)"

        if not should_stop:
            return

        self.logger.info(f"Recording stopped: {stop_reason}")

        if self.state.speech_frames_count <= 0:
            self.logger.warning("No speech detected in recording")
            if not self.enable_hotword:
                self.logger.info("No-hotword mode: Exiting")
                self.running = False
            else:
                self._reset_to_idle()
            return

        self._send_audio_to_brain()

        # Return to IDLE immediately so hotword stays responsive.
        if self.enable_hotword:
            self._reset_to_idle()
        else:
            # In no-hotword mode, wait for RESULT then exit after TTS completes.
            self.state.transition_to(AssistantState.IDLE)

    def _send_audio_to_brain(self) -> None:
        if not self._core_to_brain_q:
            self.logger.error("Brain worker queue not available")
            return

        self._clear_bargein_flags()
        audio_data = concat_audio_frames(self.audio_buffer)
        self.audio_buffer = []

        # Save to temp WAV (keeps IPC lightweight on slow PCs)
        wav_path = None
        try:
            fd, wav_path = tempfile.mkstemp(prefix="wyzer_", suffix=".wav")
            try:
                os.close(fd)
            except Exception:
                pass
            int16_audio = audio_to_int16(audio_data)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(Config.SAMPLE_RATE)
                wf.writeframes(int16_audio.tobytes())
        except Exception as e:
            self.logger.error(f"Failed to write temp WAV: {e}")
            try:
                if wav_path:
                    os.unlink(wav_path)
            except Exception:
                pass
            return

        req_id = new_id()
        safe_put(
            self._core_to_brain_q,
            {
                "type": "AUDIO",
                "id": req_id,
                "wav_path": wav_path,
                "pcm_bytes": None,
                "sample_rate": Config.SAMPLE_RATE,
                "meta": {},
            },
        )

    def _poll_brain_messages(self) -> None:
        if not self._brain_to_core_q:
            return

        while True:
            try:
                msg = self._brain_to_core_q.get_nowait()
            except Exception:
                return

            mtype = (msg or {}).get("type")
            if mtype == "LOG":
                level = str(msg.get("level", "INFO")).upper()
                text = str(msg.get("msg", ""))

                if text == "tts_started":
                    self._brain_speaking = True
                elif text in {"tts_finished", "tts_interrupted"}:
                    self._brain_speaking = False

                if level == "DEBUG":
                    self.logger.debug(text)
                elif level == "WARNING":
                    self.logger.warning(text)
                elif level == "ERROR":
                    self.logger.error(text)
                else:
                    self.logger.info(text)

                # No-hotword mode: exit after speech finishes
                if not self.enable_hotword and self._exit_after_tts and text in {"tts_finished", "tts_interrupted"}:
                    self.running = False
                continue

            if mtype == "RESULT":
                meta = msg.get("meta") or {}
                user_text = str(meta.get("user_text") or "").strip()
                if user_text:
                    print(f"\nYou: {user_text}")

                reply = str(msg.get("reply", ""))
                print(f"\nWyzer: {reply}\n")

                tts_text = msg.get("tts_text")
                if not self.enable_hotword:
                    # Exit after TTS completes (if any)
                    self._exit_after_tts = bool(tts_text)
                    if not tts_text:
                        self.running = False
                continue

            # Unknown message types are ignored

    def _reset_to_idle(self) -> None:
        self._clear_bargein_flags()
        self.audio_buffer = []
        self._drain_audio_queue(Config.POST_IDLE_DRAIN_SEC)
        self.state.transition_to(AssistantState.IDLE)
        if self.enable_hotword:
            self.logger.info(f"Ready. Listening for hotword: {Config.HOTWORD_KEYWORDS}")

    def _drain_audio_queue(self, duration_sec: float) -> None:
        drain_frames = int(duration_sec * Config.SAMPLE_RATE / Config.CHUNK_SAMPLES)
        for _ in range(drain_frames):
            try:
                frame = self.audio_queue.get_nowait()
                if self.hotword:
                    self.hotword.detect(frame)
            except Empty:
                break

    def _clear_bargein_flags(self) -> None:
        if self._bargein_pending_speech:
            self.logger.debug("Clearing barge-in pending speech flags")
        self._bargein_pending_speech = False
        self._bargein_wait_speech_deadline_ts = 0.0


