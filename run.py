#!/usr/bin/env python3
"""
Wyzer AI Assistant - Phase 6
Entry point for running the assistant.

Usage:
    python run.py                    # Run with hotword detection
    python run.py --no-hotword       # Run without hotword (immediate listening)
    python run.py --model medium     # Use different Whisper model
    python run.py --list-devices     # List audio devices
    
    # Test tools (Phase 6)
    set WYZER_TOOLS_TEST=1 & python run.py
"""
import sys
import os
import argparse
import multiprocessing as mp
from wyzer.core.logger import init_logger, get_logger
from wyzer.core.config import Config
from wyzer.audio.mic_stream import MicStream


def test_tools():
    """Test tool execution via orchestrator"""
    print("\n" + "=" * 60)
    print("  WYZER TOOLS TEST MODE - Phase 6")
    print("=" * 60 + "\n")
    
    from wyzer.core.orchestrator import handle_user_text
    from wyzer.tools.registry import build_default_registry
    
    # Build registry to test tool imports
    print("Building tool registry...")
    try:
        registry = build_default_registry()
        tools = registry.list_tools()
        print(f"Loaded {len(tools)} tools\n")
    except Exception as e:
        print(f"ERROR loading registry: {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    # Test queries
    test_queries = [
        "what time is it",
        "what's the system information",
        "show me monitor info",
        "open downloads",
        "pause music"
    ]
    
    print("Running test queries...\n")
    for query in test_queries:
        print(f"User: {query}")
        try:
            result = handle_user_text(query)
            print(f"Wyzer: {result.get('reply', '(no reply)')}")
            print(f"Latency: {result.get('latency_ms', 0)}ms")
        except Exception as e:
            print(f"ERROR: {e}")
        print()
    
    print("=" * 60)
    print("Test complete. Unset WYZER_TOOLS_TEST to run normally.")
    print("=" * 60 + "\n")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Wyzer AI Assistant - Voice Assistant with STT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                     # Normal mode with hotword
  python run.py --no-hotword        # Test mode: immediate listening
  python run.py --model medium      # Use Whisper medium model
  python run.py --device 1          # Use specific audio device
  python run.py --list-devices      # List available audio devices
        """
    )
    
    parser.add_argument(
        "--single-process",
        action="store_true",
        help="Run everything in one process (legacy path; easier debugging)"
    )

    parser.add_argument(
        "--profile",
        type=str,
        default="normal",
        choices=["low_end", "normal"],
        help="Performance profile (default: normal)"
    )

    parser.add_argument(
        "--no-hotword",
        action="store_true",
        help="Disable hotword detection (immediate listening mode)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: small)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Audio device index or name"
    )
    
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--whisper-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for Whisper inference (default: cpu)"
    )
    
    # LLM Brain arguments (Phase 4)
    parser.add_argument(
        "--llm",
        type=str,
        default="ollama",
        choices=["ollama", "off"],
        help="LLM mode: ollama or off (default: ollama)"
    )
    
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="llama3.1:latest",
        help="Ollama model name (default: llama3.1:latest)"
    )
    
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://127.0.0.1:11434",
        help="Ollama API base URL (default: http://127.0.0.1:11434)"
    )
    
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=30,
        help="LLM request timeout in seconds (default: 30)"
    )
    
    # TTS arguments (Phase 5)
    parser.add_argument(
        "--tts",
        type=str,
        default="on",
        choices=["on", "off"],
        help="Enable/disable TTS (default: on)"
    )
    
    parser.add_argument(
        "--tts-engine",
        type=str,
        default="piper",
        choices=["piper"],
        help="TTS engine to use (default: piper)"
    )
    
    parser.add_argument(
        "--piper-exe",
        type=str,
        default="./wyzer/assets/piper/piper.exe",
        help="Path to Piper executable (default: ./wyzer/assets/piper/piper.exe)"
    )
    
    parser.add_argument(
        "--piper-model",
        type=str,
        default="./wyzer/assets/piper/en_US-lessac-medium.onnx",
        help="Path to Piper voice model (default: ./wyzer/assets/piper/en_US-lessac-medium.onnx)"
    )
    
    parser.add_argument(
        "--tts-device",
        type=str,
        default=None,
        help="Audio output device index for TTS"
    )
    
    parser.add_argument(
        "--no-speak-interrupt",
        action="store_true",
        help="Disable barge-in (hotword interrupt during speaking)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    # Check for tools test mode
    if os.environ.get("WYZER_TOOLS_TEST", "0") == "1":
        test_tools()
        return 0
    
    args = parse_args()
    
    # Initialize logger
    init_logger(args.log_level)
    logger = get_logger()

    # Apply profile tweaks (keep behavior identical by default)
    whisper_compute_type = "int8"
    if args.profile == "low_end":
        whisper_compute_type = "int8"
        # Optional: allow user to provide a smaller Ollama model via env
        if args.ollama_model == "llama3.1:latest":
            low_end_model = os.environ.get("WYZER_OLLAMA_MODEL_LOW_END")
            if low_end_model:
                args.ollama_model = low_end_model
    
    # List devices if requested
    if args.list_devices:
        MicStream.list_devices()
        return 0
    
    # Parse audio device
    audio_device = None
    if args.device:
        try:
            audio_device = int(args.device)
        except ValueError:
            logger.error(f"Invalid device index: {args.device}")
            logger.info("Use --list-devices to see available devices")
            return 1
    
    # Parse TTS output device
    tts_output_device = None
    if args.tts_device:
        try:
            tts_output_device = int(args.tts_device)
        except ValueError:
            logger.error(f"Invalid TTS device index: {args.tts_device}")
            return 1
    
    # Print startup banner
    print("\n" + "=" * 60)
    print("  Wyzer AI Assistant - Phase 6")
    print("=" * 60)
    print(f"  Whisper Model: {args.model}")
    print(f"  Whisper Device: {args.whisper_device}")
    print(f"  Profile: {args.profile}")
    print(f"  Single Process: {args.single_process}")
    print(f"  Hotword Enabled: {not args.no_hotword}")
    if not args.no_hotword:
        print(f"  Hotword Keywords: {', '.join(Config.HOTWORD_KEYWORDS)}")
    print(f"  LLM Mode: {args.llm}")
    if args.llm == "ollama":
        print(f"  LLM Model: {args.ollama_model}")
        print(f"  LLM URL: {args.ollama_url}")
    print(f"  TTS Enabled: {args.tts == 'on'}")
    if args.tts == "on":
        print(f"  TTS Engine: {args.tts_engine}")
        print(f"  Piper Model: {args.piper_model}")
        print(f"  Barge-in Enabled: {not args.no_speak_interrupt}")
    print(f"  Sample Rate: {Config.SAMPLE_RATE}Hz")
    print(f"  Log Level: {args.log_level}")
    print("=" * 60 + "\n")
    
    # Import assistant (after logger is initialized)
    try:
        from wyzer.core.assistant import WyzerAssistant, WyzerAssistantMultiprocess
    except ImportError as e:
        logger.error(f"Failed to import assistant: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        return 1
    
    # Create and start assistant
    try:
        if args.single_process:
            assistant = WyzerAssistant(
                enable_hotword=not args.no_hotword,
                whisper_model=args.model,
                whisper_device=args.whisper_device,
                audio_device=audio_device,
                llm_mode=args.llm,
                ollama_model=args.ollama_model,
                ollama_url=args.ollama_url,
                llm_timeout=args.llm_timeout,
                tts_enabled=(args.tts == "on"),
                tts_engine=args.tts_engine,
                piper_exe_path=args.piper_exe,
                piper_model_path=args.piper_model,
                tts_output_device=tts_output_device,
                speak_hotword_interrupt=not args.no_speak_interrupt,
            )
        else:
            assistant = WyzerAssistantMultiprocess(
                enable_hotword=not args.no_hotword,
                whisper_model=args.model,
                whisper_device=args.whisper_device,
                whisper_compute_type=whisper_compute_type,
                audio_device=audio_device,
                llm_mode=args.llm,
                ollama_model=args.ollama_model,
                ollama_url=args.ollama_url,
                llm_timeout=args.llm_timeout,
                tts_enabled=(args.tts == "on"),
                tts_engine=args.tts_engine,
                piper_exe_path=args.piper_exe,
                piper_model_path=args.piper_model,
                tts_output_device=tts_output_device,
                speak_hotword_interrupt=not args.no_speak_interrupt,
                log_level=args.log_level,
            )
        
        logger.info("Starting assistant... (Press Ctrl+C to stop)")
        assistant.start()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
        return 0
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
