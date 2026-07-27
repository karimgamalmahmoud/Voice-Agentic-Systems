"""The screening agent: transcribe -> retrieve -> assess -> BRANCH -> score.

The branch is the point of the exercise, so it is worth being explicit about
where the decision actually lives.

An LLM produces a *grounded, structured* coverage report: for each retrieved
rubric competency, is it covered, partial, or missing, with a verbatim quote and
a confidence. Plain Python then decides whether to follow up.

That split is deliberate. Asking a 7B model "should I ask a follow-up?" gives an
answer that drifts run to run and cannot be tested. A policy over typed fields
is inspectable, tunable from config, unit-testable without a GPU, and it can be
explained to a client in one sentence.

The policy itself:

  * "partial" is the only status worth probing. Someone who named EF Core but
    never said why it is slow can be resolved by one question. Someone who never
    mentioned data access at all will not learn it in fifteen seconds, and
    someone who explained N+1 queries properly has nothing left to clarify.
  * A broadly-empty answer is exempt. If most competencies are missing, the
    score band is already settled and a probe just wastes the candidate's time.
  * At most one follow-up, as the brief requires.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable, Optional

from .config import CONFIG, ARTIFACT_DIR, AppConfig, INTERVIEW_QUESTION, INTERVIEW_QUESTION_AR
from .llm import LLMClient
from .prompts import (
    COVERAGE_SYSTEM,
    COVERAGE_USER,
    DEFAULT_FALLBACK_FOLLOW_UP,
    EVALUATE_SYSTEM,
    EVALUATE_USER,
    FALLBACK_FOLLOW_UPS,
    FOLLOWUP_BLOCK,
    FOLLOWUP_SYSTEM,
    FOLLOWUP_USER,
    LANGUAGE_NAMES,
    SPOKEN_RESULT,
)
from .retrieval import (
    CorpusIndex,
    competency_names,
    format_competencies,
    format_references,
)
from .schemas import (
    CompetencyAssessment,
    CoverageReport,
    CoverageStatus,
    Evaluation,
    FollowUpDecision,
    RetrievedChunk,
    Transcript,
    TurnResult,
)
from .stt import Transcriber
from .text import strip_foreign_scripts
from .tts import Synthesizer

logger = logging.getLogger(__name__)

# Which gap is worth spending the single follow-up on. The question names
# async/await and EF Core explicitly, and the rubric leads on diagnostic method,
# so those outrank the softer competencies.
COMPETENCY_PRIORITY: dict[str, float] = {
    "diagnostic method": 1.0,
    "async": 0.95,
    "data access": 0.9,
    "caching": 0.6,
    "tradeoff": 0.4,
}
_DEFAULT_PRIORITY = 0.5


def priority_of(competency: str) -> float:
    name = competency.lower()
    for key, weight in COMPETENCY_PRIORITY.items():
        if key in name:
            return weight
    return _DEFAULT_PRIORITY


def decide_follow_up(
    coverage: CoverageReport,
    cfg: Optional[AppConfig] = None,
    follow_ups_so_far: int = 0,
) -> FollowUpDecision:
    """Pure function - no I/O, no model call. Unit-tested in tests/test_decision.py."""
    conf = (cfg or CONFIG).decision

    if follow_ups_so_far >= conf.max_follow_ups:
        return FollowUpDecision(
            should_follow_up=False,
            reason=f"Follow-up budget spent ({conf.max_follow_ups} max).",
        )

    total = len(coverage.assessments)
    if total == 0:
        return FollowUpDecision(
            should_follow_up=False,
            reason="No competencies were assessed; nothing to probe.",
        )

    missing = coverage.by_status(CoverageStatus.MISSING)
    covered = coverage.by_status(CoverageStatus.COVERED)
    partials = [
        a
        for a in coverage.by_status(CoverageStatus.PARTIAL)
        if a.confidence >= conf.min_partial_confidence
    ]

    if not partials:
        return FollowUpDecision(
            should_follow_up=False,
            reason=(
                f"No competency is partially covered above confidence "
                f"{conf.min_partial_confidence:.2f} "
                f"({len(covered)}/{total} covered, {len(missing)}/{total} missing). "
                "The answer is unambiguous, so a clarifying question would not move the score."
            ),
        )

    missing_ratio = len(missing) / total
    if missing_ratio >= conf.weak_answer_missing_ratio:
        return FollowUpDecision(
            should_follow_up=False,
            reason=(
                f"{len(missing)}/{total} competencies are absent "
                f"({missing_ratio:.0%} >= {conf.weak_answer_missing_ratio:.0%} threshold). "
                "The answer is thin across the board; one clarification cannot change the band."
            ),
        )

    target = max(partials, key=lambda a: (priority_of(a.competency), a.confidence))
    return FollowUpDecision(
        should_follow_up=True,
        target_competency=target.competency,
        reason=(
            f"'{target.competency}' is partially covered (confidence {target.confidence:.2f}, "
            f"priority {priority_of(target.competency):.2f}) and is the highest-value "
            f"resolvable gap of {len(partials)} candidate(s). Gap: {target.gap or 'underdeveloped'}."
        ),
    )


class ScreeningAgent:
    def __init__(self, cfg: Optional[AppConfig] = None, warm: bool = False):
        self.cfg = cfg or CONFIG
        self.transcriber = Transcriber(self.cfg.stt)
        self.index = CorpusIndex(self.cfg.retrieval)
        self.llm = LLMClient(self.cfg.llm)
        self.tts = Synthesizer(self.cfg.tts)
        if warm:
            self.warm_up()

    # -- setup -------------------------------------------------------------

    def warm_up(self) -> None:
        """Preload models so the first live turn is not a two-minute download."""
        self.transcriber.load()
        self.index.load()

    def question_text(self, language: str = "en") -> str:
        return INTERVIEW_QUESTION_AR if language == "ar" else INTERVIEW_QUESTION

    def speak_question(self, language: str = "en") -> Optional[tuple[int, "object"]]:
        return self.tts.synthesize(self.question_text(language), language)

    # -- stages ------------------------------------------------------------

    def transcribe(self, audio) -> Transcript:
        return self.transcriber.transcribe(audio)

    def retrieve(self, transcript: Transcript) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        """Retrieve against question + answer together.

        The answer alone is a noisy query - a weak answer retrieves almost
        nothing useful. Anchoring on the question keeps the scoring targets
        stable while still letting what the candidate actually said pull in the
        relevant reference notes.
        """
        query = f"{INTERVIEW_QUESTION}\n\nCandidate answer:\n{transcript.text}"
        competencies = self.index.retrieve_competencies(query)
        references = self.index.retrieve_references(query)
        logger.info(
            "Retrieved competencies=%s references=%s",
            [c.section for c in competencies],
            [f"{c.doc_id}" for c in references],
        )
        return competencies, references

    def assess_coverage(
        self,
        transcript: Transcript,
        competencies: list[RetrievedChunk],
        references: list[RetrievedChunk],
    ) -> CoverageReport:
        names = competency_names(competencies)
        payload = self.llm.complete_json(
            COVERAGE_SYSTEM,
            COVERAGE_USER.format(
                question=INTERVIEW_QUESTION,
                competencies=format_competencies(competencies),
                references=format_references(references),
                language=LANGUAGE_NAMES.get(transcript.language, transcript.language),
                transcript=transcript.text,
            ),
        )
        report = self._parse_coverage(payload, names)
        logger.info(
            "Coverage: %s",
            {a.competency: f"{a.status.value}@{a.confidence:.2f}" for a in report.assessments},
        )
        return report

    @staticmethod
    def _parse_coverage(payload: dict, expected: list[str]) -> CoverageReport:
        """Normalise the model's output back onto the competencies we asked about.

        Small models rename or drop entries; anything unmatched is recorded as
        missing at low confidence rather than silently vanishing from scoring.
        """
        raw = payload.get("assessments")
        if not isinstance(raw, list):
            raw = []

        seen: dict[str, CompetencyAssessment] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("competency", "")).strip()
            if not name:
                continue
            match = next(
                (
                    e
                    for e in expected
                    if e.lower() == name.lower()
                    or e.lower() in name.lower()
                    or name.lower() in e.lower()
                ),
                None,
            )
            canonical = match or name
            status = str(item.get("status", "missing")).strip().lower()
            if status not in {s.value for s in CoverageStatus}:
                status = "missing"
            seen[canonical] = CompetencyAssessment(
                competency=canonical,
                status=CoverageStatus(status),
                evidence=str(item.get("evidence", ""))[:400],
                gap=str(item.get("gap", ""))[:400],
                confidence=item.get("confidence", 0.5),
            )

        for name in expected:
            if name not in seen:
                logger.warning("Model omitted competency %r; recording as missing", name)
                seen[name] = CompetencyAssessment(
                    competency=name,
                    status=CoverageStatus.MISSING,
                    gap="Model did not return an assessment for this competency.",
                    confidence=0.2,
                )

        return CoverageReport(assessments=[seen[n] for n in expected if n in seen])

    @staticmethod
    def follow_up_is_usable(text: str) -> bool:
        """Reject questions that are truncated, empty, or rambling.

        A real run produced "كيف تتعامل مع إبطال" - "How do you handle
        invalidat", cut off mid-word. It passes a naive non-empty check and
        would have been read aloud to the candidate.

        The terminal question mark is the signal that actually catches this:
        truncation almost never lands on one. Word count alone would have let
        that four-word fragment through.
        """
        text = (text or "").strip()
        if not text:
            return False
        words = text.split()
        if not 4 <= len(words) <= 45:
            return False
        return text.endswith(("?", "؟"))

    def fallback_follow_up(self, competency: Optional[str], language: str) -> str:
        name = (competency or "").lower()
        for key, variants in FALLBACK_FOLLOW_UPS.items():
            if key in name:
                return variants.get(language, variants["en"])
        return DEFAULT_FALLBACK_FOLLOW_UP.get(language, DEFAULT_FALLBACK_FOLLOW_UP["en"])

    def compose_follow_up(
        self, transcript: Transcript, decision: FollowUpDecision, coverage: CoverageReport
    ) -> str:
        target = next(
            (a for a in coverage.assessments if a.competency == decision.target_competency),
            None,
        )
        system = FOLLOWUP_SYSTEM.format(
            language_name=LANGUAGE_NAMES.get(transcript.language, "English")
        )
        user = FOLLOWUP_USER.format(
            question=INTERVIEW_QUESTION,
            transcript=transcript.text,
            competency=decision.target_competency or "",
            gap=(target.gap if target else "") or "underdeveloped",
        )

        for attempt in (1, 2):
            raw = self.llm.complete(
                system,
                user if attempt == 1 else user + "\n\nReturn ONE complete question, "
                "ending in a question mark. Do not stop mid-sentence.",
                # Resampling at the same temperature tends to reproduce the same
                # truncation, so the retry is warmer.
                temperature=0.4 if attempt == 1 else 0.8,
                max_tokens=160,
            ).strip().strip('"')
            cleaned = strip_foreign_scripts(raw, context="follow-up question")
            if self.follow_up_is_usable(cleaned):
                return cleaned
            logger.warning("Unusable follow-up on attempt %d: %r", attempt, cleaned)

        fallback = self.fallback_follow_up(decision.target_competency, transcript.language)
        logger.warning("Falling back to a templated probe for %r", decision.target_competency)
        return fallback

    def evaluate(
        self,
        transcript: Transcript,
        competencies: list[RetrievedChunk],
        coverage: CoverageReport,
        follow_up_question: Optional[str] = None,
        follow_up_answer: Optional[Transcript] = None,
    ) -> Evaluation:
        coverage_lines = "\n".join(
            f"- {a.competency}: {a.status.value} (confidence {a.confidence:.2f})"
            + (f" | gap: {a.gap}" if a.gap else "")
            for a in coverage.assessments
        ) or "(none)"

        block = ""
        if follow_up_question and follow_up_answer:
            block = FOLLOWUP_BLOCK.format(
                question=follow_up_question, answer=follow_up_answer.text
            )

        payload = self.llm.complete_json(
            EVALUATE_SYSTEM.format(
                language_name=LANGUAGE_NAMES.get(transcript.language, "English")
            ),
            EVALUATE_USER.format(
                question=INTERVIEW_QUESTION,
                levels=self.index.rubric_levels(),
                competencies=format_competencies(competencies),
                coverage=coverage_lines,
                transcript=transcript.text,
                follow_up_block=block,
            ),
        )
        return Evaluation(
            score=payload.get("score", 3),
            justification=strip_foreign_scripts(
                str(payload.get("justification", "")).strip(), context="justification"
            ),
            covered=[strip_foreign_scripts(str(x)) for x in (payload.get("covered") or [])][:4],
            missed=[strip_foreign_scripts(str(x)) for x in (payload.get("missed") or [])][:4],
        )

    def speak_evaluation(
        self, evaluation: Evaluation, language: str, out_path: Optional[Path] = None
    ) -> tuple[str, Optional[str]]:
        template = SPOKEN_RESULT.get(language, SPOKEN_RESULT["en"])
        spoken = template.format(score=evaluation.score, justification=evaluation.justification)
        path = out_path or (ARTIFACT_DIR / "evaluation.wav")
        written = self.tts.synthesize_to_file(spoken, language, path)
        return spoken, (str(written) if written else None)

    # -- full loop ---------------------------------------------------------

    def run(
        self,
        audio,
        follow_up_provider: Optional[Callable[[str, str], Optional[object]]] = None,
        speak: bool = True,
        out_path: Optional[Path] = None,
    ) -> TurnResult:
        """Run one complete screening.

        `follow_up_provider(question, language)` supplies the candidate's answer
        to the clarifying question and returns audio, or None if unavailable.
        The Gradio UI wires this to a second mic capture; the batch quality gate
        passes None, so the decision is still made and logged but scoring falls
        back to the original answer.
        """
        transcript = self.transcribe(audio)
        competencies, references = self.retrieve(transcript)
        coverage = self.assess_coverage(transcript, competencies, references)
        decision = decide_follow_up(coverage, self.cfg)

        follow_up_answer: Optional[Transcript] = None
        if decision.should_follow_up:
            decision.question = self.compose_follow_up(transcript, decision, coverage)
            logger.info("Follow-up: %s", decision.question)
            if follow_up_provider is not None:
                reply_audio = follow_up_provider(decision.question, transcript.language)
                if reply_audio is not None:
                    follow_up_answer = self.transcribe(reply_audio)

        evaluation = self.evaluate(
            transcript,
            competencies,
            coverage,
            follow_up_question=decision.question,
            follow_up_answer=follow_up_answer,
        )

        spoken, audio_path = ("", None)
        if speak:
            spoken, audio_path = self.speak_evaluation(evaluation, transcript.language, out_path)

        return TurnResult(
            transcript=transcript,
            follow_up_transcript=follow_up_answer,
            retrieved=competencies + references,
            coverage=coverage,
            decision=decision,
            evaluation=evaluation,
            spoken_text=spoken,
            audio_path=audio_path,
        )
