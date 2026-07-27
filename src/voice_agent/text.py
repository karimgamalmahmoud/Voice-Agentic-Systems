"""Script-level text utilities.

Arabic output from a 7B model is not reliably clean. In testing, Qwen produced
"استدعاءات синк داخل انتظارات أسكيد" - Cyrillic for "sync", and "async"
transliterated into Arabic letters. Both are unreadable to an Arabic speaker and
both are unspeakable by the TTS.

The prompts ask for Latin-script technical terms; this module is the safety net
for when the model ignores that.
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_ARABIC_RANGES = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)

# Scripts we ever legitimately emit. Anything else in a token means the model
# drifted alphabet mid-sentence.
_LATIN_RANGES = (
    (0x0041, 0x005A),
    (0x0061, 0x007A),
    (0x00C0, 0x024F),  # Latin-1 Supplement + Extended-A/B
)

# If sanitising would destroy this much of the text, the output is too damaged
# to silently repair - hand it back intact and let a human see the problem.
_MAX_DROP_RATIO = 0.30


def is_arabic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _ARABIC_RANGES)


def _is_latin_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _LATIN_RANGES)


def script_language(text: str) -> tuple[str, float]:
    """Classify language by script, returning (lang, arabic_ratio).

    Classified from the decoded text rather than from Whisper's own language
    token on purpose. Whisper detects per 30-second chunk, so a code-mixed
    answer can flip mid-file and produce a mixed-script transcript. Counting
    Arabic letters across the whole output is stable, and it gives exactly the
    binary we need: which language do we reply in.
    """
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return "en", 0.0
    arabic = sum(1 for c in letters if is_arabic_char(c))
    ratio = arabic / len(letters)
    # Threshold is low because code-mixed Egyptian Arabic carries a lot of
    # Latin-script technical vocabulary ("async", "EF Core", "thread pool").
    # An answer that is 25% Arabic script is an Arabic answer.
    return ("ar" if ratio >= 0.15 else "en"), ratio


def _token_has_foreign_script(token: str) -> bool:
    for ch in token:
        if not unicodedata.category(ch).startswith("L"):
            continue
        if is_arabic_char(ch) or _is_latin_char(ch):
            continue
        return True
    return False


def strip_foreign_scripts(text: str, context: str = "output") -> str:
    """Drop whole tokens containing neither-Arabic-nor-Latin letters.

    Token-level rather than character-level: deleting the Cyrillic letters out
    of "синк" leaves an empty string anyway, and half-deleting a mixed token
    leaves worse garbage than removing it. Whole-word removal at least yields a
    readable sentence.
    """
    if not text:
        return text
    tokens = text.split()
    kept = [t for t in tokens if not _token_has_foreign_script(t)]
    dropped = len(tokens) - len(kept)
    if not dropped:
        return text

    if dropped / max(len(tokens), 1) > _MAX_DROP_RATIO:
        logger.warning(
            "%s: %d/%d tokens use an unexpected script - too damaged to repair, "
            "returning as-is for inspection",
            context, dropped, len(tokens),
        )
        return text

    logger.warning(
        "%s: dropped %d token(s) in an unexpected script: %s",
        context, dropped, [t for t in tokens if _token_has_foreign_script(t)],
    )
    return re.sub(r"\s+", " ", " ".join(kept)).strip()
