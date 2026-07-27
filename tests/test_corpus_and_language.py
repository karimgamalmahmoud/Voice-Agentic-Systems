"""Tests for the pieces that need no models: corpus chunking and script-based
language detection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_agent.retrieval import RUBRIC_DOC_ID, load_corpus  # noqa: E402
from voice_agent.stt import script_language, _collapse_repeats  # noqa: E402
from voice_agent.tts import clean_for_speech  # noqa: E402


# -- corpus ----------------------------------------------------------------


def test_corpus_loads_and_splits_into_sections():
    chunks = load_corpus()
    assert len(chunks) > 6
    assert {c.doc_id for c in chunks} >= {
        RUBRIC_DOC_ID,
        "01_async_concurrency",
        "03_data_access_ef_core",
    }


def test_all_five_rubric_competencies_are_found():
    names = {c.competency_name for c in load_corpus() if c.is_competency}
    assert len(names) == 5
    assert any("Diagnostic" in n for n in names)
    assert any("EF Core" in n for n in names)
    assert any("Async" in n for n in names)


def test_rubric_levels_section_is_present_and_separate():
    chunks = load_corpus()
    levels = [c for c in chunks if c.doc_id == RUBRIC_DOC_ID and c.section.lower() == "levels"]
    assert len(levels) == 1
    # The scale must survive intact - scoring is anchored on it.
    for n in ("1", "2", "3", "4", "5"):
        assert f"**{n}" in levels[0].text


def test_no_chunk_is_empty():
    assert all(c.text.strip() for c in load_corpus())


# -- language detection ----------------------------------------------------


def test_pure_english_is_english():
    lang, ratio = script_language("I would profile the endpoint and check the generated SQL.")
    assert lang == "en"
    assert ratio == 0.0


def test_pure_arabic_is_arabic():
    lang, ratio = script_language("هبدأ أقيس الأداء الأول قبل ما أغير أي حاجة")
    assert lang == "ar"
    assert ratio > 0.9


def test_code_mixed_egyptian_arabic_is_arabic():
    """The case that matters: heavy Latin-script technical vocabulary inside an
    Arabic answer must still route to Arabic output."""
    text = (
        "أنا هعمل profiling للـ endpoint الأول، وبعدين أشوف الـ EF Core queries "
        "وأتأكد إن مفيش N+1، وكمان الـ async await مش متعمل صح في الـ thread pool"
    )
    lang, ratio = script_language(text)
    assert lang == "ar"
    assert 0.15 < ratio < 0.9  # genuinely mixed, not one-sided


def test_english_with_one_stray_arabic_word_stays_english():
    lang, _ = script_language(
        "I would check the thread pool and the database indexes before anything else. نعم"
    )
    assert lang == "en"


def test_empty_text_does_not_crash():
    assert script_language("") == ("en", 0.0)


def test_repeated_sentences_are_collapsed():
    """Whisper loops on trailing silence; the loop must not reach the LLM."""
    looped = "Thanks for watching. Thanks for watching. Thanks for watching."
    assert _collapse_repeats(looped) == "Thanks for watching."


def test_distinct_sentences_survive_collapsing():
    text = "I would profile it. Then I would check the SQL."
    assert _collapse_repeats(text) == text


# -- speech cleanup --------------------------------------------------------


def test_markdown_is_stripped_before_speech():
    assert "*" not in clean_for_speech("**Score:** 4/5 — good `AsNoTracking` use")
    assert "`" not in clean_for_speech("**Score:** 4/5 — good `AsNoTracking` use")


def test_arabic_survives_speech_cleanup():
    assert "تقييمك" in clean_for_speech("**تقييمك** 4 من 5")
