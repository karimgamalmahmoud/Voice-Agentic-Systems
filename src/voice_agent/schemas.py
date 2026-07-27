"""Structured types exchanged between pipeline stages.

The LLM is constrained to emit these shapes as JSON. Keeping the contract
explicit is what lets the follow-up decision be plain Python instead of another
model call.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class CoverageStatus(str, Enum):
    COVERED = "covered"
    PARTIAL = "partial"
    MISSING = "missing"


class Transcript(BaseModel):
    text: str
    language: Literal["ar", "en"]
    # Fraction of letters that are Arabic script - a code-mixing signal we
    # surface in the UI so the reviewer can see the mixing was detected.
    arabic_ratio: float = 0.0
    duration_s: Optional[float] = None


class RetrievedChunk(BaseModel):
    doc_id: str
    section: str
    text: str
    score: float


class CompetencyAssessment(BaseModel):
    competency: str
    status: CoverageStatus
    # Verbatim span from the transcript that justifies the status. Forces the
    # model to ground its judgement instead of free-associating.
    evidence: str = ""
    gap: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, v: object) -> float:
        try:
            return max(0.0, min(1.0, float(v)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.5


class CoverageReport(BaseModel):
    assessments: list[CompetencyAssessment] = Field(default_factory=list)

    def by_status(self, status: CoverageStatus) -> list[CompetencyAssessment]:
        return [a for a in self.assessments if a.status == status]


class FollowUpDecision(BaseModel):
    """Output of the deterministic branch. `reason` is always populated so the
    decision is auditable in the UI and in the quality-gate log."""

    should_follow_up: bool
    reason: str
    target_competency: Optional[str] = None
    question: Optional[str] = None


class Evaluation(BaseModel):
    score: int = Field(ge=1, le=5)
    justification: str
    covered: list[str] = Field(default_factory=list)
    missed: list[str] = Field(default_factory=list)

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score(cls, v: object) -> int:
        try:
            return max(1, min(5, int(round(float(v)))))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 3


class TurnResult(BaseModel):
    """Everything one screening run produced - the object the UI and the
    quality gate both render."""

    transcript: Transcript
    follow_up_transcript: Optional[Transcript] = None
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    coverage: CoverageReport = Field(default_factory=CoverageReport)
    decision: FollowUpDecision
    evaluation: Evaluation
    spoken_text: str = ""
    audio_path: Optional[str] = None
