"""Tests for follow-up question validation and the templated fallback.

Driven by a real failure. A Colab run composed this Arabic probe for answer_2:

    "كيف تتعامل مع إبطال"

which is "How do you handle invalidat" - truncated mid-word. It is non-empty and
four words long, so every naive check passes it, and it would have been spoken
to the candidate.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_agent.agent import ScreeningAgent  # noqa: E402
from voice_agent.prompts import DEFAULT_FALLBACK_FOLLOW_UP, FALLBACK_FOLLOW_UPS  # noqa: E402

# The validator and fallback selector are pure - no models are loaded.
usable = ScreeningAgent.follow_up_is_usable
fallback = ScreeningAgent.fallback_follow_up


def test_the_actual_truncated_question_is_rejected():
    assert usable("كيف تتعامل مع إبطال") is False


def test_a_complete_arabic_question_is_accepted():
    assert usable("إزاي هتعرف الـ SQL اللي بيتولد فعلاً، وهتدور على إيه فيه؟") is True


def test_a_complete_english_question_is_accepted():
    assert usable("How would you spot a blocking call in the request path?") is True


def test_empty_and_whitespace_are_rejected():
    assert usable("") is False
    assert usable("   ") is False
    assert usable(None) is False


def test_too_short_is_rejected():
    assert usable("Why?") is False


def test_rambling_is_rejected():
    assert usable(" ".join(["word"] * 60) + "?") is False


def test_statement_without_question_mark_is_rejected():
    """Truncation almost never lands on a question mark, which is what makes it
    the useful signal here."""
    assert usable("Tell me more about how you would handle caching invalidation") is False


# -- language fidelity -----------------------------------------------------


def test_arabic_followup_to_an_english_answer_is_rejected():
    """The actual regression: answer_1 was English and got an Arabic probe."""
    arabic = "كيف ستتعامل مع سيناريو التعادل بين استخدام await وال Mutex أثناء التحقيق؟"
    assert usable(arabic, "en") is False
    assert usable(arabic, "ar") is True


def test_english_followup_to_an_arabic_answer_is_rejected():
    english = "How would you spot a blocking call in the request path?"
    assert usable(english, "ar") is False
    assert usable(english, "en") is True


def test_arabic_question_with_latin_technical_terms_still_counts_as_arabic():
    """Code-mixing must not trip the language check - this is how Egyptian
    engineers actually speak."""
    mixed = "الـ async/await بيساعد فين تحت الـ load، وإزاي تكتشف الـ blocking call؟"
    assert usable(mixed, "ar") is True


def test_language_check_is_skipped_when_no_language_given():
    assert usable("How would you measure that before changing anything?") is True


# -- fallback selection ----------------------------------------------------


def test_fallback_matches_competency_and_language():
    ar = fallback(None, "Async & concurrency", "ar")
    en = fallback(None, "Async & concurrency", "en")
    assert "async" in ar.lower()
    assert "async" in en.lower()
    assert ar != en


def test_every_fallback_is_itself_usable_and_in_the_right_language():
    """A fallback that fails our own validator would be worse than useless."""
    for competency, variants in FALLBACK_FOLLOW_UPS.items():
        for lang, text in variants.items():
            assert usable(text, lang), f"{competency}/{lang} is not a usable {lang} question"
    for lang, text in DEFAULT_FALLBACK_FOLLOW_UP.items():
        assert usable(text, lang), f"default/{lang} is not a usable {lang} question"


def test_unknown_competency_falls_back_to_the_generic_probe():
    assert fallback(None, "Something not in the rubric", "en") == DEFAULT_FALLBACK_FOLLOW_UP["en"]
    assert fallback(None, None, "ar") == DEFAULT_FALLBACK_FOLLOW_UP["ar"]


def test_all_five_competencies_have_a_targeted_fallback():
    for competency in [
        "Diagnostic method",
        "Data access (EF Core)",
        "Async & concurrency",
        "Caching & performance",
        "Tradeoff & communication",
    ]:
        for lang in ("en", "ar"):
            assert fallback(None, competency, lang) != DEFAULT_FALLBACK_FOLLOW_UP[lang], (
                f"{competency} fell through to the generic probe"
            )
