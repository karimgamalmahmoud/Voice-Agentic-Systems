"""Tests for output script sanitising.

These exist because of a real failure. On the first full Colab run, Qwen
produced this Arabic follow-up question:

    "كيف هتتعامل مع السيناريوهات اللي فيها استدعاءات синк داخل انتظارات أسكيد؟"

"синк" is Cyrillic for "sync", and "أسكيد" is "async" transliterated into
Arabic letters. Both are unreadable to an Arabic speaker and unspeakable by the
TTS. The prompts now forbid this; this module is the safety net.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_agent.text import script_language, strip_foreign_scripts  # noqa: E402


def test_the_actual_regression_from_the_colab_run():
    bad = "كيف هتتعامل مع السيناريوهات اللي فيها استدعاءات синк داخل انتظارات أسكيد؟"
    cleaned = strip_foreign_scripts(bad)
    assert "синк" not in cleaned
    # The Arabic around it must survive.
    assert "كيف هتتعامل" in cleaned
    assert "السيناريوهات" in cleaned


def test_clean_arabic_is_untouched():
    text = "تقييمك 4 من 5. إجابة كويسة بس ناقصها تفاصيل."
    assert strip_foreign_scripts(text) == text


def test_latin_technical_terms_inside_arabic_survive():
    """The whole point of the script rule - these must never be stripped."""
    text = "لازم تستخدم async و EF Core وتتأكد من الـ thread pool"
    cleaned = strip_foreign_scripts(text)
    for term in ("async", "EF", "Core", "thread", "pool"):
        assert term in cleaned


def test_clean_english_is_untouched():
    text = "Good answer covering N+1 queries and AsNoTracking."
    assert strip_foreign_scripts(text) == text


def test_heavily_corrupted_text_is_returned_intact_for_inspection():
    """Past ~30% damage, silently 'repairing' would hide a broken model. Better
    to surface the mess than to emit a confident-looking fragment."""
    text = "привет мир это совсем не арабский текст здесь"
    assert strip_foreign_scripts(text) == text


def test_empty_and_whitespace_are_safe():
    assert strip_foreign_scripts("") == ""
    assert strip_foreign_scripts("   ").strip() == ""


def test_digits_and_punctuation_survive():
    text = "النتيجة 4/5 - كويس."
    cleaned = strip_foreign_scripts(text)
    assert "4/5" in cleaned


def test_script_language_still_exported_from_stt():
    """Callers and tests import it from voice_agent.stt; the move to text.py
    must not break that."""
    from voice_agent.stt import script_language as from_stt

    assert from_stt is script_language
