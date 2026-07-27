#!/usr/bin/env python
"""Transcribe the provided sample answers and nothing else.

This was the first thing I ran. Egyptian Arabic mixed with English technical
terms is the highest-risk part of the stack, so it was worth proving Whisper
could handle it before building anything on top. It also produces the reference
transcripts used when reading the quality-gate output.

Usage:
    python scripts/transcribe_samples.py
    python scripts/transcribe_samples.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from voice_agent.config import AUDIO_DIR, CONFIG  # noqa: E402
from voice_agent.stt import Transcriber  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--language", help="force a decode language, e.g. 'ar'")
    args = parser.parse_args()

    files = sorted(AUDIO_DIR.glob("*.mp3"))
    if not files:
        print(f"No audio in {AUDIO_DIR}", file=sys.stderr)
        return 2

    print(f"Device: {CONFIG.resolved_device()} · Model: {CONFIG.stt.model_id}", file=sys.stderr)
    transcriber = Transcriber()
    rows = []

    for path in files:
        start = time.time()
        t = transcriber.transcribe(str(path), language=args.language)
        elapsed = time.time() - start
        rows.append(
            {
                "file": path.name,
                "language": t.language,
                "arabic_ratio": round(t.arabic_ratio, 3),
                "duration_s": round(t.duration_s or 0, 1),
                "transcribe_s": round(elapsed, 1),
                "text": t.text,
            }
        )
        if not args.json:
            print(f"\n{'=' * 70}\n{path.name}")
            print(
                f"  lang={t.language}  arabic={t.arabic_ratio:.0%}  "
                f"audio={t.duration_s:.0f}s  took={elapsed:.0f}s"
            )
            print(f"{'=' * 70}\n{t.text}\n")

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
