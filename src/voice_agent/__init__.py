"""Voice-driven technical screening agent."""

from .agent import ScreeningAgent, decide_follow_up
from .config import CONFIG, INTERVIEW_QUESTION
from .schemas import (
    CoverageReport,
    CoverageStatus,
    Evaluation,
    FollowUpDecision,
    Transcript,
    TurnResult,
)

__all__ = [
    "CONFIG",
    "INTERVIEW_QUESTION",
    "ScreeningAgent",
    "decide_follow_up",
    "CoverageReport",
    "CoverageStatus",
    "Evaluation",
    "FollowUpDecision",
    "Transcript",
    "TurnResult",
]

__version__ = "0.1.0"
