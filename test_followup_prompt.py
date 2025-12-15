#!/usr/bin/env python3
"""Test script to verify follow-up prompt handling."""

import sys
import os
import time


# This file is primarily a manual/interactive smoke test.
# When collected by pytest, it would block waiting for Ctrl+C.
# `PYTEST_CURRENT_TEST` isn't set during collection/import, so we detect pytest
# via the imported module instead.
if "pytest" in sys.modules:
    import pytest

    pytest.skip("Interactive smoke test (run this file directly)", allow_module_level=True)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wyzer.core.config import Config
from wyzer.core.assistant import WyzerAssistantMultiprocess
from wyzer.core.logger import init_logger

def run_followup_prompt_smoke_test() -> bool:
    """Test that follow-up prompt is handled silently."""
    init_logger("DEBUG")
    
    print("=" * 60)
    print("Testing Follow-Up Prompt Feature")
    print("=" * 60)
    
    # Create assistant with FOLLOWUP enabled
    Config.FOLLOWUP_ENABLED = True
    Config.FOLLOWUP_TIMEOUT_SEC = 3.0
    
    assistant = WyzerAssistantMultiprocess(
        hotword_keywords=["wyzer"],
        enable_hotword=True,
        log_level="DEBUG"
    )
    
    print("\n✓ Assistant initialized with FOLLOWUP enabled")
    print(f"  FOLLOWUP_ENABLED: {Config.FOLLOWUP_ENABLED}")
    print(f"  FOLLOWUP_TIMEOUT_SEC: {Config.FOLLOWUP_TIMEOUT_SEC}")
    
    # Verify followup_manager exists
    if hasattr(assistant, 'followup_manager'):
        print("✓ followup_manager initialized")
    else:
        print("✗ followup_manager NOT initialized!")
        return False
    
    # Verify brain worker is started
    print("\n✓ Starting assistant brain worker...")
    time.sleep(1)
    
    print("\nTest setup complete. The following behavior should occur:")
    print("1. If you say something, you should get a response")
    print("2. After the response, 'Is there anything else?' should play")
    print("3. This prompt should NOT appear in console output")
    print("4. You then have 3 seconds to respond without hotword")
    print("5. Say 'no' or 'stop' to exit the follow-up window")
    print("\nListening... (press Ctrl+C to exit)")
    
    try:
        assistant.run()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    finally:
        print("Shutting down...")
        assistant.stop()
        print("Done!")
    
    return True

if __name__ == "__main__":
    success = run_followup_prompt_smoke_test()
    sys.exit(0 if success else 1)
