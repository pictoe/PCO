"""Quick sanity checks for volume fast-path parsing.

This does NOT change system volume. It only exercises the deterministic parser.

Run:
  python scripts/test_volume_fastpath.py
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wyzer.core.orchestrator import get_registry
from wyzer.core.orchestrator import _try_fastpath_intents  # type: ignore


def main() -> None:
    reg = get_registry()

    samples = [
        "get volume",
        "what is the volume",
        "what's the volume",
        "what is spotify volume",
        "turn spotify down to 35%",
        "turn volume down to 35",
        "volume 35",
        "set volume to 15%",
        "sound down a bit",
        "turn it up",
        "mute",
        "unmute",
        "spotify volume 30",
        "set spotify volume to 20",
        "turn down spotify",
        "mute discord",
        "volume 50 for chrome",
    ]

    for s in samples:
        intents = _try_fastpath_intents(s, reg)
        print("\n==", s)
        if not intents:
            print("(no fastpath intents)")
            continue
        for i in intents:
            print({"tool": i.tool, "args": i.args, "continue_on_error": i.continue_on_error})


if __name__ == "__main__":
    main()
