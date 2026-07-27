# CLAUDE.md — context for the coding agent

Project context file used while building this repo with Claude Code.

## What this is

A voice agent that runs a senior .NET technical screening: ask a question by
voice → transcribe a spoken answer (Egyptian Arabic code-mixed with English
technical terms) → retrieve rubric criteria → **decide whether to ask one
clarifying follow-up** → score 1–5 → speak the result back in the input
language.

Built against a take-home brief with a 3–4 hour box. The grading order is
explicit and drives every decision below:

1. **It runs** from a clean checkout — non-negotiable
2. **Agentic design** — a real decision in the loop, not a linear script
3. **Coding-agent fluency**
4. **Quality gate** — sane retrieval, evidence of regression detection
5. **Voice** — Arabic / code-mixed handling, working speak-back

Visual polish, auth and deployment are explicitly **not** graded. Do not spend
effort there.

## Hard constraints

- **Open-source, self-hosted models only.** No OpenAI/Anthropic/Google/ElevenLabs
  API calls anywhere in the runtime path. No API keys.
- **Must run on a free Colab T4** (15 GB usable VRAM). Total resident model
  budget ~10 GB. If a change pushes past that, it is the wrong change.
- **Python only.**
- **The whole loop must survive a component failing.** TTS especially — losing
  the speak-back degrades to text, it never raises.

## Architecture rules

**The follow-up decision is deterministic Python, never an LLM call.**
This is the single most important design rule in the repo. The LLM produces a
structured `CoverageReport` (per competency: covered/partial/missing + verbatim
evidence + confidence). `decide_follow_up()` in `agent.py` applies a policy over
those typed fields. Rationale: testable without a GPU, inspectable (every
decision carries a `reason` string), tunable from config by a non-ML engineer,
and stable run to run. If you are tempted to add "ask the model whether to
follow up" — don't; that was considered and rejected.

**Layer boundaries.** Each stage takes and returns a pydantic model from
`schemas.py`. No dicts crossing module boundaries. No stage imports another
stage's internals; `agent.py` is the only orchestrator.

**Config over constants.** Every model id and threshold lives in `config.py`
and is environment-overridable. Nothing hardcoded at a call site.

**Lazy loading.** Models load on first use and cache on the instance. Importing
`voice_agent` must never pull 10 GB into VRAM — the CPU-only tests depend on
this.

## Style

- Python 3.10+, `from __future__ import annotations` at the top of every module.
- Type hints on every public function.
- Docstrings explain **why**, not what. If a choice has a non-obvious reason
  (a rejected alternative, a library footgun, a threshold that looks arbitrary),
  that reason goes in the docstring or an inline comment. Do not narrate what
  the code plainly does.
- Comments in the codebase should read like an engineer left them for a
  colleague, not like generated exposition.
- `logging`, never `print`, inside `src/`. Scripts may print.
- No emoji in `src/` except the UI rendering layer in `app.py`.

## Testing

- `tests/` must stay **CPU-only and model-free**. They exist so the branch logic
  is verifiable without a GPU. Never add a test that downloads weights.
- Every rule in the decision policy gets a test.
- Run: `python -m pytest tests/ -q`

## Things that will bite you

- **Do not use faster-whisper.** It is faster, but ctranslate2 pins a cuDNN
  major version that conflicts with Colab's CUDA stack. Plain `transformers`
  is slower and actually installs. This was a deliberate trade — do not
  "optimise" it back.
- **Do not pin `torch` in requirements.txt.** Colab's build is CUDA-matched;
  reinstalling breaks the runtime.
- **Do not force Whisper to `language="ar"`.** It transliterates English
  technical terms into Arabic script and destroys them. Auto-detect, then
  classify the language from the output script.
- **MMS-TTS needs romanized input** for Arabic (`tokenizer.is_uroman`). Check
  the flag; do not assume.
- **Ollama runs as a separate process** specifically to keep its dependencies
  away from the torch/transformers stack. Do not replace it with an in-process
  loader.
- The two irrelevant corpus files (`02_dependency_injection`,
  `04_api_design_security`) are **distractors on purpose**. They must stay in
  the corpus, and the quality gate asserts they stay out of retrieval results.

## Definition of done for any change

1. `python -m pytest tests/ -q` passes.
2. `python scripts/run_quality_gate.py` passes (needs GPU + Ollama).
3. The Colab notebook still runs top to bottom.
4. README reflects any architectural change.
