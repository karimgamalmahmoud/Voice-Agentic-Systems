"""Gradio UI.

Runs the loop as explicit steps rather than one blocking call, because the
follow-up branch needs a second mic capture from the user in between. The
batch path (`ScreeningAgent.run`) keeps the single-call form for the quality
gate.

In Colab, launch with share=True: the tunnel gives a real browser microphone,
which is the whole reason this works without a local install.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Optional

import gradio as gr

from .agent import ScreeningAgent, decide_follow_up
from .config import ARTIFACT_DIR, AUDIO_DIR, CONFIG, INTERVIEW_QUESTION, INTERVIEW_QUESTION_AR
from .schemas import CoverageReport, CoverageStatus, Evaluation, FollowUpDecision, Transcript

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

AGENT: Optional[ScreeningAgent] = None

STATUS_ICON = {
    CoverageStatus.COVERED: "✅",
    CoverageStatus.PARTIAL: "🟡",
    CoverageStatus.MISSING: "❌",
}


def get_agent() -> ScreeningAgent:
    global AGENT
    if AGENT is None:
        AGENT = ScreeningAgent()
    return AGENT


# -- renderers -------------------------------------------------------------


def _render_transcript(t: Transcript, title: str = "Transcript") -> str:
    mixed = " · code-mixed" if 0.15 <= t.arabic_ratio <= 0.92 else ""
    dur = f" · {t.duration_s:.0f}s" if t.duration_s else ""
    return (
        f"### {title}\n"
        f"`{t.language}`{mixed} · Arabic script {t.arabic_ratio:.0%}{dur}\n\n"
        f"> {t.text or '_(empty)_'}"
    )


def _render_retrieval(competencies, references) -> str:
    lines = ["### Retrieved context", "", "**Scoring targets (rubric competencies)**", ""]
    for c in competencies:
        name = c.section.replace("Competency:", "").strip()
        lines.append(f"- `{c.score:.3f}` {name}")
    lines += ["", "**Supporting reference notes**", ""]
    if references:
        for r in references:
            lines.append(f"- `{r.score:.3f}` {r.doc_id} — {r.section}")
    else:
        lines.append("- _(none above threshold)_")
    return "\n".join(lines)


def _render_coverage(coverage: CoverageReport) -> str:
    lines = ["### Coverage analysis", "", "| | Competency | Status | Conf. | Evidence / gap |", "|---|---|---|---|---|"]
    for a in coverage.assessments:
        detail = a.evidence.strip() if a.status == CoverageStatus.COVERED else (a.gap or a.evidence)
        detail = (detail or "").replace("|", "\\|").replace("\n", " ")
        if len(detail) > 130:
            detail = detail[:130] + "…"
        lines.append(
            f"| {STATUS_ICON[a.status]} | {a.competency} | {a.status.value} | "
            f"{a.confidence:.2f} | {detail or '—'} |"
        )
    return "\n".join(lines)


def _render_decision(d: FollowUpDecision) -> str:
    verdict = "🔀 **Ask one clarifying follow-up**" if d.should_follow_up else "➡️ **Move on and score**"
    out = [f"### Agent decision\n\n{verdict}", "", f"_{d.reason}_"]
    if d.target_competency:
        out.append(f"\n**Probing:** {d.target_competency}")
    return "\n".join(out)


def _render_evaluation(e: Evaluation, spoken: str) -> str:
    stars = "★" * e.score + "☆" * (5 - e.score)
    lines = [
        f"### Evaluation — {e.score}/5  {stars}",
        "",
        f"**{e.justification}**",
        "",
    ]
    if e.covered:
        lines += ["**Covered**"] + [f"- {x}" for x in e.covered] + [""]
    if e.missed:
        lines += ["**Missed**"] + [f"- {x}" for x in e.missed] + [""]
    if spoken:
        lines += ["---", f"🔊 _Spoken back:_ {spoken}"]
    return "\n".join(lines)


# -- handlers --------------------------------------------------------------


def hear_question(language: str):
    agent = get_agent()
    lang = "ar" if language.startswith("Arabic") else "en"
    text = INTERVIEW_QUESTION_AR if lang == "ar" else INTERVIEW_QUESTION
    audio = agent.tts.synthesize(text, lang)
    if audio is None:
        return f"**Question ({lang}):** {text}\n\n_(TTS unavailable — text only)_", None
    return f"**Question ({lang}):** {text}", audio


def check_health():
    agent = get_agent()
    ok, msg = agent.llm.health()
    return ("✅ " if ok else "⚠️ ") + msg


def _finalize(agent: ScreeningAgent, state: dict, follow_up_answer: Optional[Transcript]):
    """Score and speak. Shared by the follow-up and no-follow-up paths."""
    evaluation = agent.evaluate(
        state["transcript"],
        state["competencies"],
        state["coverage"],
        follow_up_question=state["decision"].question,
        follow_up_answer=follow_up_answer,
    )
    spoken, path = agent.speak_evaluation(
        evaluation, state["transcript"].language, ARTIFACT_DIR / "evaluation.wav"
    )
    fu_md = _render_transcript(follow_up_answer, "Follow-up answer") if follow_up_answer else ""
    return _render_evaluation(evaluation, spoken), path, fu_md


def submit_answer(audio: Any, state: dict):
    if audio is None:
        return (
            "⚠️ Record or upload an answer first.",
            "", "", "", "", gr.update(visible=False), "", None, "", None, "", state,
        )

    agent = get_agent()
    transcript = agent.transcribe(audio)
    competencies, references = agent.retrieve(transcript)
    coverage = agent.assess_coverage(transcript, competencies, references)
    decision = decide_follow_up(coverage, agent.cfg)

    state = {
        "transcript": transcript,
        "competencies": competencies,
        "references": references,
        "coverage": coverage,
        "decision": decision,
    }

    base = (
        _render_transcript(transcript),
        _render_retrieval(competencies, references),
        _render_coverage(coverage),
        _render_decision(decision),
    )

    if decision.should_follow_up:
        decision.question = agent.compose_follow_up(transcript, decision, coverage)
        q_audio = agent.tts.synthesize(decision.question, transcript.language)
        return (
            "🔀 Agent chose to probe a gap — answer the follow-up below.",
            *base,
            gr.update(visible=True),
            f"### Clarifying follow-up\n\n> {decision.question}",
            q_audio,
            "",      # evaluation deferred
            None,
            "",      # follow-up transcript
            state,
        )

    eval_md, eval_path, fu_md = _finalize(agent, state, None)
    return (
        "✅ Complete.",
        *base,
        gr.update(visible=False),
        "",
        None,
        eval_md,
        eval_path,
        fu_md,
        state,
    )


def submit_follow_up(audio: Any, state: dict):
    if not state:
        return "⚠️ Submit an answer first.", "", None, "", state
    agent = get_agent()
    follow_up = agent.transcribe(audio) if audio is not None else None
    eval_md, eval_path, fu_md = _finalize(agent, state, follow_up)
    status = "✅ Complete." if follow_up else "✅ Scored without a follow-up answer."
    return status, eval_md, eval_path, fu_md, state


def skip_follow_up(state: dict):
    if not state:
        return "⚠️ Submit an answer first.", "", None, "", state
    agent = get_agent()
    eval_md, eval_path, fu_md = _finalize(agent, state, None)
    return "✅ Scored on the original answer (follow-up skipped).", eval_md, eval_path, fu_md, state


def load_sample(name: str):
    path = AUDIO_DIR / name
    return str(path) if path.exists() else None


# -- layout ----------------------------------------------------------------


def build_ui() -> gr.Blocks:
    samples = sorted(p.name for p in AUDIO_DIR.glob("*.mp3")) if AUDIO_DIR.exists() else []

    with gr.Blocks(title="Voice Screening Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎙️ Voice Screening Agent\n"
            "Senior .NET screening over a rubric corpus. Speak an answer in English or "
            "Egyptian Arabic — the agent transcribes it, retrieves the relevant rubric "
            "criteria, decides whether one clarifying follow-up is worth asking, then "
            "scores and speaks back **in your language**."
        )

        with gr.Row():
            health = gr.Markdown("_LLM status unknown_")
            health_btn = gr.Button("Check LLM", scale=0)
        health_btn.click(check_health, outputs=health)

        with gr.Accordion("Step 1 — The question", open=True):
            with gr.Row():
                lang_choice = gr.Radio(
                    ["English", "Arabic (العربية)"], value="English", label="Ask in", scale=2
                )
                ask_btn = gr.Button("🔊 Hear the question", scale=1)
            question_md = gr.Markdown(f"**Question:** {INTERVIEW_QUESTION}")
            question_audio = gr.Audio(label="Question audio", autoplay=False)
        ask_btn.click(hear_question, inputs=lang_choice, outputs=[question_md, question_audio])

        gr.Markdown("### Step 2 — Your answer")
        with gr.Row():
            with gr.Column(scale=3):
                answer_audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Record or upload your answer",
                )
            with gr.Column(scale=1):
                if samples:
                    sample_pick = gr.Dropdown(samples, label="…or load a provided sample")
                    sample_pick.change(load_sample, inputs=sample_pick, outputs=answer_audio)
                submit_btn = gr.Button("▶️ Submit answer", variant="primary")

        status = gr.Markdown()

        with gr.Row():
            transcript_md = gr.Markdown()
            decision_md = gr.Markdown()

        follow_up_group = gr.Group(visible=False)
        with follow_up_group:
            gr.Markdown("### Step 3 — Clarifying follow-up")
            follow_up_md = gr.Markdown()
            follow_up_audio_out = gr.Audio(label="Follow-up (spoken)", autoplay=False)
            follow_up_answer = gr.Audio(
                sources=["microphone", "upload"], type="numpy", label="Your follow-up answer"
            )
            with gr.Row():
                follow_up_btn = gr.Button("▶️ Submit follow-up", variant="primary")
                skip_btn = gr.Button("⏭️ Skip and score anyway")
        follow_up_transcript_md = gr.Markdown()

        gr.Markdown("### Result")
        evaluation_md = gr.Markdown()
        evaluation_audio = gr.Audio(label="Spoken evaluation", autoplay=True)

        with gr.Accordion("Retrieval + coverage detail", open=False):
            retrieval_md = gr.Markdown()
            coverage_md = gr.Markdown()

        state = gr.State({})

        submit_btn.click(
            submit_answer,
            inputs=[answer_audio, state],
            outputs=[
                status, transcript_md, retrieval_md, coverage_md, decision_md,
                follow_up_group, follow_up_md, follow_up_audio_out,
                evaluation_md, evaluation_audio, follow_up_transcript_md, state,
            ],
        )
        follow_up_btn.click(
            submit_follow_up,
            inputs=[follow_up_answer, state],
            outputs=[status, evaluation_md, evaluation_audio, follow_up_transcript_md, state],
        )
        skip_btn.click(
            skip_follow_up,
            inputs=[state],
            outputs=[status, evaluation_md, evaluation_audio, follow_up_transcript_md, state],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the voice screening agent UI")
    parser.add_argument("--share", action="store_true", help="public Gradio link (needed in Colab)")
    parser.add_argument("--port", type=int, default=int(os.getenv("VA_PORT", "7860")))
    parser.add_argument("--warm", action="store_true", help="preload STT + embedder before serving")
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Device: %s", CONFIG.resolved_device())

    if args.warm:
        logger.info("Warming models…")
        get_agent().warm_up()

    build_ui().launch(share=args.share, server_port=args.port, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
