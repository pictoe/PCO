# Wyzer AI Assistant - Phase 6

A voice-controlled AI assistant for Windows 10/11 with hotword detection, speech-to-text, local LLM brain (Ollama), local Piper TTS, and a Phase 6 tool-calling orchestrator for safe local actions.

## Features

- ✅ **Hotword Detection**: Wake the assistant with the default openWakeWord model ("hey jarvis")
- ✅ **Voice Activity Detection (VAD)**: Silero VAD with energy-based fallback
- ✅ **Speech-to-Text**: Fast, accurate transcription using faster-whisper
- ✅ **Local LLM Brain**: Conversational AI using Ollama (Phase 4)
- ✅ **Text-to-Speech**: Local, fast Piper TTS (Phase 5)
- ✅ **Barge-in Support**: Interrupt TTS by saying the hotword (Phase 5)
- ✅ **Tool Calling (Phase 6)**: LLM can call safe, allowlisted local tools
- ✅ **State Machine**: Clean state transitions (IDLE → LISTENING → TRANSCRIBING → THINKING → SPEAKING)
- ✅ **Cross-platform (core)**: Windows-first; some bundled binaries/tools are Windows-specific
- ✅ **Robust Audio**: 16kHz mono pipeline with proper buffering
- ✅ **Spam Filtering**: Automatically filters repetition spam and garbage output
- ✅ **Privacy-First**: Core voice + LLM run locally; some optional tools (e.g., weather/location) require internet

## System Requirements

- **OS**: Windows 10/11 (primary), Linux/macOS (compatible)
- **Python**: 3.10-3.12 recommended
- **Microphone**: Working audio input device
- **RAM**: 4GB+ recommended (2GB for Whisper + 2GB for LLM)
- **CPU**: Modern CPU with AVX support recommended
- **Ollama**: Required for Phase 4 LLM features (install separately)

**Note**: Some audio/ML dependencies may lag newest Python releases. If you hit install issues on 3.13+, try 3.10-3.12.

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended; matches run.bat)
python -m venv .venv
.venv\Scripts\activate.bat  # Windows (CMD)
# .venv\Scripts\Activate.ps1  # Windows (PowerShell)
# source .venv/bin/activate  # Linux/macOS

# Install requirements
pip install -r requirements.txt
```
### 2. Install Ollama (Phase 4)

For full conversational AI capabilities:

1. **Download and install Ollama**:
   - Visit: https://ollama.ai/download
   - Download the Windows/macOS/Linux installer
   - Run the installer

2. **Start Ollama** (if not auto-started):
   ```bash
   ollama serve
   ```

3. **Pull a model**:
   ```bash
   # Recommended: Fast, capable model
   ollama pull llama3.1:latest
   
   # Alternative: Smaller/faster model
   ollama pull llama3.2:3b
   
   # Alternative: Larger/more capable
   ollama pull llama3.1:70b
   ```

### 3. Piper TTS (Phase 5)

Piper (Windows) and a default voice model are already included in this repo under `wyzer/assets/piper/`.

**Quick test**:
```bash
.\wyzer\assets\piper\piper.exe --version
echo "Hello, this is a test" | .\wyzer\assets\piper\piper.exe -m .\wyzer\assets\piper\en_US-lessac-medium.onnx -f test.wav
```

**Recommended Voice Models**:
- **Fast & Good**: `en_US-lessac-medium.onnx` (~63 MB)
- **High Quality**: `en_US-lessac-high.onnx` (~147 MB)
- **Lightweight**: `en_US-lessac-low.onnx` (~13 MB)

### 4. Run Wyzer

```bash
# Windows convenience launcher (uses .venv\Scripts\python.exe)
run.bat

# Normal mode with hotword detection, LLM, and TTS
python run.py

# Specify Piper model path
python run.py --piper-model .\wyzer\assets\piper\en_US-lessac-medium.onnx

# Test mode (no hotword, immediate listening, with TTS)
python run.py --no-hotword

# Disable TTS (text-only responses)
python run.py --tts off

# Use different LLM model
python run.py --ollama-model llama3.2:3b

# STT-only mode (no LLM)
python run.py --llm off

# Enable debug logging
python run.py --log-level DEBUG

# Use GPU for Whisper (if CUDA available)
python run.py --whisper-device cuda

# Custom Ollama URL (e.g., remote server)
python run.py --ollama-url http://192.168.1.100:11434

# List available audio devices
python run.py --list-devices

# Use specific audio device (index from --list-devices)
python run.py --device 1

# Disable barge-in (no hotword interrupt during speaking)
python run.py --no-speak-interrupt
```

## Usage Examples

### Normal Mode (with Hotword, LLM, and TTS)
```bash
python run.py
```
1. Wait for "Ready. Listening for hotword..."
2. Say the wake word (default model: "hey jarvis")
3. Speak your request after acknowledgment
4. Assistant transcribes, thinks, and speaks the response
5. **Barge-in**: Say the hotword while speaking to interrupt immediately

### Test Mode (no Hotword)
```bash
python run.py --no-hotword
```
1. Assistant immediately starts listening
2. Speak your request
3. Pauses when you stop speaking (1.2s silence timeout)
4. Transcribes, thinks, speaks response (if enabled), and exits

### Custom Configuration
```bash
# Use medium model for better accuracy
python run.py --model medium

# Custom Piper voice model
python run.py --piper-model .\path\to\voice.onnx

# Use different TTS output device
python run.py --tts-device 2

# Combine options
python run.py --model medium --ollama-model llama3.2:3b --tts off
```

## Project Structure

```
wyzer/
├── core/
│   ├── config.py         # Central configuration
│   ├── logger.py         # Logging with rich formatting
│   ├── state.py          # State machine definitions (SPEAKING added)
│   └── assistant.py      # Main coordinator (TTS integration)
├── audio/
│   ├── mic_stream.py     # Microphone capture
│   ├── vad.py            # Voice activity detection
│   ├── hotword.py        # Wake word detection
│   └── audio_utils.py    # Audio utilities
├── stt/
│   ├── whisper_engine.py # Whisper STT engine
│   └── stt_router.py     # STT routing (extensible)
├── brain/                # Phase 4: LLM integration
│   ├── llm_engine.py     # Ollama LLM client
│   └── prompt.py         # System prompts
└── tts/                  # Phase 5: Text-to-speech
    ├── piper_engine.py   # Piper TTS engine
    ├── audio_player.py   # Interruptible audio playback
    └── tts_router.py     # TTS routing
run.py                    # Entry point
requirements.txt          # Dependencies
README.md                 # This file
```

## Configuration

Configuration can be customized via environment variables:

```bash
# Audio settings
set WYZER_SAMPLE_RATE=16000
set WYZER_CHUNK_MS=20

# Recording limits
set WYZER_MAX_RECORD_SECONDS=12.0
set WYZER_VAD_SILENCE_TIMEOUT=1.2

# VAD settings
set WYZER_VAD_THRESHOLD=0.5

# Hotword settings
set WYZER_HOTWORD_KEYWORDS=hey jarvis,jarvis
set WYZER_HOTWORD_THRESHOLD=0.5

# Whisper settings
set WYZER_WHISPER_MODEL=small
set WYZER_WHISPER_DEVICE=cpu

# Spam filter
set WYZER_MAX_TOKEN_REPEATS=6
set WYZER_MIN_TRANSCRIPT_LENGTH=2

# LLM settings (Phase 4)
set WYZER_LLM_MODE=ollama
set WYZER_OLLAMA_URL=http://127.0.0.1:11434
set WYZER_OLLAMA_MODEL=llama3.1:latest
set WYZER_LLM_TIMEOUT=30

# TTS settings (Phase 5)
set WYZER_TTS_ENABLED=true
set WYZER_TTS_ENGINE=piper
set WYZER_PIPER_EXE_PATH=.\wyzer\assets\piper\piper.exe
set WYZER_PIPER_MODEL_PATH=.\wyzer\assets\piper\en_US-lessac-medium.onnx
set WYZER_SPEAK_HOTWORD_INTERRUPT=true
set WYZER_SPEAK_START_COOLDOWN_SEC=1.8
set WYZER_POST_SPEAK_DRAIN_SEC=0.35
set WYZER_POST_BARGEIN_IGNORE_SEC=3.0
set WYZER_POST_BARGEIN_REQUIRE_SPEECH_START=true
set WYZER_POST_BARGEIN_WAIT_FOR_SPEECH_SEC=2.0
```

Or modify defaults in [wyzer/core/config.py](wyzer/core/config.py).

## Troubleshooting

### Piper TTS Issues

#### Piper Executable Not Found
```
Error: Piper executable not found: ./wyzer/assets/piper/piper.exe
```

**Solutions**:
1. Verify `wyzer/assets/piper/piper.exe` exists in this repo
2. Or point to a different Piper executable:
   ```bash
   python run.py --piper-exe C:\path\to\piper.exe
   ```

#### Piper Model Not Found
```
Error: Piper model not found at: ./wyzer/assets/piper/en_US-lessac-medium.onnx
```

**Solutions**:
1. Verify `wyzer/assets/piper/en_US-lessac-medium.onnx` exists in this repo
2. Or specify a different model path:
   ```bash
   python run.py --piper-model .\path\to\voice.onnx
   ```

#### No Audio Output from TTS
**Solutions**:
1. Check volume is not muted
2. List audio devices and select specific output:
   ```bash
   python run.py --list-devices
   python run.py --tts-device 1
   ```
3. Test Piper directly:
   ```bash
   echo "test" | .\wyzer\assets\piper\piper.exe -m .\wyzer\assets\piper\en_US-lessac-medium.onnx -f test.wav
   ```

#### Barge-in Not Working
**Solutions**:
1. Ensure hotword is enabled (not using `--no-hotword`)
2. Check if barge-in is disabled with `--no-speak-interrupt`
3. Verify hotword detector is working in IDLE state first
4. Increase hotword sensitivity:
   ```bash
   set WYZER_HOTWORD_THRESHOLD=0.3
   ```

#### TTS Interrupts Too Quickly After Starting
If TTS gets interrupted almost immediately after starting (within 200-500ms), this is caused by residual hotword audio in the mic buffer from the previous hotword activation.

**Solutions**:
1. Increase the speaking start cooldown (default 1.2s):
   ```bash
   set WYZER_SPEAK_START_COOLDOWN_SEC=1.5
   ```
   This prevents barge-in for the first X seconds after TTS starts speaking.
2. Increase post-idle drain time:
   ```bash
   set WYZER_POST_IDLE_DRAIN_SEC=0.5
   ```

#### Hotword Immediately Re-triggers After Barge-in
If the hotword is detected again immediately after interrupting TTS (barge-in), this is usually caused by residual audio in the mic buffer or echo from speakers.

**Solutions**:
1. Increase the post-barge-in ignore window (default 3.0s):
   ```bash
   set WYZER_POST_BARGEIN_IGNORE_SEC=4.0
   ```
2. Increase the wait-for-speech timeout after barge-in (default 2.0s):
   ```bash
   set WYZER_POST_BARGEIN_WAIT_FOR_SPEECH_SEC=3.0
   ```
   This keeps hotword detection disabled until speech actually starts or timeout expires.
3. Increase audio drain time after interrupt:
   ```bash
   set WYZER_POST_SPEAK_DRAIN_SEC=0.5
   ```
4. Use headphones instead of speakers to prevent echo feedback
5. Adjust microphone sensitivity/gain in Windows sound settings
6. Increase hotword cooldown period:
   ```bash
   set WYZER_HOTWORD_COOLDOWN_SEC=2.0
   ```
7. Disable speech-start requirement (less robust, but allows faster recovery):
   ```bash
   set WYZER_POST_BARGEIN_REQUIRE_SPEECH_START=false
   ```


### Ollama / LLM Issues

#### Ollama Not Running
```
Error: I couldn't reach the local model. Is Ollama running?
```

**Solutions**:
1. Start Ollama server:
   ```bash
   ollama serve
   ```

2. Verify Ollama is running:
   ```bash
   # Windows PowerShell
   Test-NetConnection -ComputerName localhost -Port 11434
   
   # Or check process
   Get-Process ollama
   ```

3. Test Ollama directly:
   ```bash
   ollama list
   curl http://localhost:11434/api/tags
   ```

#### Model Not Found
```
Error: Model 'llama3.1:latest' not found. Try: ollama pull llama3.1:latest
```

**Solutions**:
1. Pull the model:
   ```bash
   ollama pull llama3.1:latest
   ```

2. List available models:
   ```bash
   ollama list
   ```

3. Use a different model:
   ```bash
   python run.py --ollama-model llama3.2:3b
   ```

#### LLM Timeout
If responses are slow or timing out:

1. **Increase timeout**:
   ```bash
   python run.py --llm-timeout 60
   ```

2. **Use smaller model**:
   ```bash
   python run.py --ollama-model llama3.2:3b
   ```

3. **Check system resources**:
   - Close other applications
   - Monitor RAM usage
   - Check CPU usage during inference

#### Firewall Blocking Ollama
If Ollama is running but can't connect:

1. **Check Windows Firewall**:
   - Allow Ollama through firewall
   - Or disable temporarily for testing

2. **Test connection**:
   ```bash
   curl http://127.0.0.1:11434/api/tags
   ```

3. **Use alternative URL**:
   ```bash
   python run.py --ollama-url http://localhost:11434
   ```

#### Disable LLM (STT-only mode)
To test without LLM:
```bash
python run.py --llm off
```

### No Audio Input / Microphone Not Working

1. **Check device permissions** (Windows):
   - Go to Settings → Privacy → Microphone
   - Ensure "Allow apps to access your microphone" is ON
   - Ensure Python is allowed

2. **List and test devices**:
   ```bash
   python run.py --list-devices
   ```
   Find your microphone's index and use it:
   ```bash
   python run.py --device 1
   ```

3. **Test microphone separately**:
   ```bash
   python -c "import sounddevice as sd; print(sd.query_devices())"
   ```

### Hotword Not Detecting

1. **Check available models**:
   The assistant will log available openWakeWord models at startup with DEBUG level:
   ```bash
   python run.py --log-level DEBUG
   ```

2. **Adjust threshold**:
   ```bash
   set WYZER_HOTWORD_THRESHOLD=0.3
   python run.py
   ```

3. **Skip hotword for testing**:
   ```bash
   python run.py --no-hotword
   ```

### VAD/Silero Not Working

The assistant includes an energy-based VAD fallback. If you see:
```
Silero VAD not available. Using energy-based VAD fallback.
```

This is normal if `silero-vad` or `torch` failed to install. The fallback works but may be less accurate.

To install Silero VAD properly:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install silero-vad
```

### Audio Sample Rate Issues

If you see warnings about sample rate mismatches:
- The assistant expects 16kHz
- Most devices support this natively
- If not, basic resampling is attempted

To force device check:
```bash
python -c "import sounddevice as sd; print(sd.query_devices(1))"  # Replace 1 with your device
```

### Transcription is Empty or Filtered

The assistant filters:
1. Transcripts shorter than 2 characters
2. Repetition spam (token repeated >6 times)
3. Low alphabetic content (garbage)

If legitimate speech is filtered:
```bash
set WYZER_MAX_TOKEN_REPEATS=10
set WYZER_MIN_TRANSCRIPT_LENGTH=1
python run.py --log-level DEBUG
```

### Performance Issues / Slow Transcription

1. **Use smaller model**:
   ```bash
   python run.py --model tiny
   ```

2. **Use int8 compute** (default, fastest):
   Already enabled by default

3. **Upgrade to GPU** (if available):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   python run.py --whisper-device cuda
   ```

### Import Errors

If you see `ModuleNotFoundError`:
```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade

# Check Python version
python --version  # Must be 3.10+

# Check if packages are installed
pip list | findstr "sounddevice openwakeword faster-whisper"
```

### Windows-Specific Issues

1. **Long path errors**: Enable long paths in Windows
   ```
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```

2. **Antivirus blocking**: Add Python and the project folder to exclusions

3. **Virtual environment activation**:
   ```bash
   # PowerShell
   .venv\Scripts\Activate.ps1
   
   # CMD
   .venv\Scripts\activate.bat
   ```

## Development Notes

### State Machine

```
IDLE → (hotword detected) → LISTENING
LISTENING → (speech + silence timeout) → TRANSCRIBING
TRANSCRIBING → (done) → THINKING
THINKING → (LLM response) → SPEAKING
SPEAKING → (TTS complete) → IDLE

Barge-in (Phase 5):
SPEAKING → (hotword detected) → LISTENING (immediately)
```

In `--no-hotword` mode:
```
IDLE → (immediate) → LISTENING → TRANSCRIBING → THINKING → SPEAKING → EXIT
```

In `--llm off` mode:
```
IDLE → (hotword) → LISTENING → TRANSCRIBING → IDLE
```

In `--tts off` mode:
```
IDLE → (hotword) → LISTENING → TRANSCRIBING → THINKING → IDLE
```

### Audio Pipeline

1. **Capture**: sounddevice → 16kHz mono float32 → Queue
2. **Detection**: 
   - IDLE: hotword detector checks each frame
   - LISTENING: VAD checks each frame, buffers audio
   - SPEAKING: hotword detector checks for barge-in
3. **Transcription**: Concatenate buffer → Whisper → Filter → Display
4. **Thinking** (Phase 4): Transcript → Ollama LLM → Response → Display
5. **Speaking** (Phase 5): Response → Piper TTS → Audio playback (interruptible)

### Non-Blocking Processing

Phase 4-5 uses background threads to prevent audio queue overflow:
- **Transcription thread**: Processes audio while main loop drains mic queue
- **Thinking thread**: LLM processing while main loop continues draining audio
- **Speaking thread**: TTS synthesis + playback while main loop checks for interrupts
- This ensures real-time audio capture never blocks on slow STT/LLM/TTS operations

1. **Capture**: sounddevice → 16kHz mono float32 → Queue
2. **Detection**: 
   - IDLE: hotword detector checks each frame
   - LISTENING: VAD checks each frame, buffers audio
3. **Transcription**: Concatenate buffer → Whisper → Filter → Display

### Thread Safety

- One audio callback thread (sounddevice)
- One main loop thread (queue consumer)
- Queue with maxsize (drops frames if full, logs warning)
- No complex locking needed

### Adding Custom Hotwords

openWakeWord supports custom ONNX models. To add your own:

1. Train or download a custom `.onnx` model
2. Set the path via environment:
   ```bash
   set WYZER_HOTWORD_MODEL_PATH=path\\to\\model.onnx
   ```

## Current Limitations

1. **No conversation memory**: Each interaction is independent (stateless)
2. **Tools are allowlisted**: The assistant can only do what the local tool registry supports
3. **Basic resampling**: Uses linear interpolation (scipy would be better)
4. **Energy VAD fallback**: Less accurate than Silero
5. **Single-turn conversations**: No context from previous exchanges

## Future Phases (Not Implemented)

- Phase 7: Conversation memory / context
- Phase 8: GUI/system tray
- Phase 9: Multi-turn conversations

## License

[Your License Here]

## Support

For issues or questions:
1. Check this README's troubleshooting section
2. Enable debug logging: `python run.py --log-level DEBUG`
3. Check logs for specific error messages

## Dependencies

Core libraries:
- `sounddevice`: Microphone capture and TTS playback
- `numpy`: Audio processing
- `openwakeword`: Hotword detection
- `silero-vad`: Voice activity detection (with fallback)
- `faster-whisper`: Speech-to-text
- `onnxruntime`: ONNX model support
- `rich`: Pretty console output (optional)

External dependencies:
- **Ollama**: LLM inference (install separately)
- **Piper**: TTS synthesis (bundled in `wyzer/assets/piper/` for Windows)
