"""Unit tests for the follow-up branch.

The whole reason the decision is plain Python over a typed coverage report is
that it can be tested like this - no GPU, no model, no network. These run in
about a second on any machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_agent.agent import decide_follow_up, priority_of  # noqa: E402
from voice_agent.config import AppConfig  # noqa: E402
from voice_agent.schemas import CompetencyAssessment, CoverageReport, CoverageStatus  # noqa: E402

CFG = AppConfig()

DIAGNOSTIC = "Diagnostic method"
EF_CORE = "Data access (EF Core)"
ASYNC = "Async & concurrency"
CACHING = "Caching & performance"
TRADEOFF = "Tradeoff & communication"


def report(*specs: tuple[str, str, float]) -> CoverageReport:
    return CoverageReport(
        assessments=[
            CompetencyAssessment(
                competency=name,
                status=CoverageStatus(status),
                confidence=conf,
                gap=f"gap in {name}",
            )
            for name, status, conf in specs
        ]
    )


def test_strong_answer_does_not_get_a_follow_up():
    """Nothing ambiguous left - a clarifying question would waste the turn."""
    decision = decide_follow_up(
        report(
            (DIAGNOSTIC, "covered", 0.9),
            (EF_CORE, "covered", 0.85),
            (ASYNC, "covered", 0.8),
            (CACHING, "covered", 0.7),
        ),
        CFG,
    )
    assert decision.should_follow_up is False
    assert "unambiguous" in decision.reason


def test_broadly_empty_answer_does_not_get_a_follow_up():
    """3 of 4 missing is 75%, over the 60% weak-answer threshold. One question
    cannot lift this into a different band, so we score and move on."""
    decision = decide_follow_up(
        report(
            (DIAGNOSTIC, "missing", 0.9),
            (EF_CORE, "missing", 0.9),
            (ASYNC, "partial", 0.8),
            (CACHING, "missing", 0.9),
        ),
        CFG,
    )
    assert decision.should_follow_up is False
    assert "thin across the board" in decision.reason


def test_mid_answer_with_one_soft_spot_gets_a_follow_up():
    """The case the branch exists for: they know the area but one competency is
    underdeveloped and a single question would resolve it."""
    decision = decide_follow_up(
        report(
            (DIAGNOSTIC, "covered", 0.9),
            (EF_CORE, "partial", 0.75),
            (ASYNC, "covered", 0.8),
            (CACHING, "missing", 0.6),
        ),
        CFG,
    )
    assert decision.should_follow_up is True
    assert decision.target_competency == EF_CORE


def test_highest_priority_partial_wins():
    """Two soft spots, one follow-up. Diagnostic method outranks caching because
    the rubric leads on measure-before-changing."""
    decision = decide_follow_up(
        report(
            (DIAGNOSTIC, "partial", 0.6),
            (CACHING, "partial", 0.9),
            (ASYNC, "covered", 0.8),
            (EF_CORE, "covered", 0.8),
        ),
        CFG,
    )
    assert decision.should_follow_up is True
    assert decision.target_competency == DIAGNOSTIC


def test_low_confidence_partial_is_not_probed():
    """A 'partial' the model is unsure about is not evidence of a real gap."""
    decision = decide_follow_up(
        report(
            (DIAGNOSTIC, "covered", 0.9),
            (EF_CORE, "partial", 0.20),  # below min_partial_confidence (0.45)
            (ASYNC, "covered", 0.8),
        ),
        CFG,
    )
    assert decision.should_follow_up is False


def test_follow_up_budget_is_spent_after_one():
    decision = decide_follow_up(
        report((DIAGNOSTIC, "partial", 0.9), (EF_CORE, "covered", 0.9)),
        CFG,
        follow_ups_so_far=1,
    )
    assert decision.should_follow_up is False
    assert "budget" in decision.reason.lower()


def test_empty_coverage_is_handled():
    decision = decide_follow_up(CoverageReport(), CFG)
    assert decision.should_follow_up is False


def test_every_decision_records_a_reason():
    """The reason is shown in the UI and snapshotted by the quality gate, so an
    unexplained branch is itself a bug."""
    cases = [
        report((DIAGNOSTIC, "covered", 0.9)),
        report((DIAGNOSTIC, "partial", 0.9), (EF_CORE, "covered", 0.9)),
        report((DIAGNOSTIC, "missing", 0.9), (EF_CORE, "missing", 0.9)),
        CoverageReport(),
    ]
    for r in cases:
        assert decide_follow_up(r, CFG).reason.strip()


@pytest.mark.parametrize(
    "name,expected_higher_than",
    [(DIAGNOSTIC, CACHING), (ASYNC, TRADEOFF), (EF_CORE, TRADEOFF)],
)
def test_priority_ordering(name, expected_higher_than):
    assert priority_of(name) > priority_of(expected_higher_than)


def test_unknown_competency_gets_default_priority():
    assert priority_of("Something the rubric never mentioned") == 0.5


def test_threshold_is_configurable():
    """Operators tune the branch from config, not by editing the policy."""
    lenient = AppConfig()
    lenient.decision.min_partial_confidence = 0.1
    coverage = report((DIAGNOSTIC, "covered", 0.9), (EF_CORE, "partial", 0.2))
    assert decide_follow_up(coverage, CFG).should_follow_up is False
    assert decide_follow_up(coverage, lenient).should_follow_up is True
