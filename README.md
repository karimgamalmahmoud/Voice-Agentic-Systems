# Voice Screening Agent

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karimgamalmahmoud/Voice-Agentic-Systems/blob/main/notebooks/run_on_colab.ipynb)

A voice agent that runs a short senior .NET technical screening end to end.

It asks a question out loud, listens to a spoken answer in **English or Egyptian
Arabic mixed with English technical terms**, retrieves the relevant criteria from
a rubric corpus, **decides whether one clarifying follow-up is worth asking**,
then scores the answer 1–5 and speaks the result back **in the language the
candidate used**.

Every model is open-source and self-hosted. **No API keys, no paid services.**

---

## Run it

### Colab (recommended — no local GPU needed)

Click the badge above → set the runtime to a **T4 GPU**
(`Runtime → Change runtime type → Hardware accelerator → T4 GPU`) →
`Runtime → Run all`.

The last cell prints a public `https://….gradio.live` link. Open it; the
microphone works through that tunnel.

**First run takes ~8–10 minutes**, almost entirely downloading ~10 GB of model
weights. I would rather state that honestly than claim two minutes — subsequent
runs in the same session are instant.

### Locally (needs an NVIDIA GPU, ~12 GB VRAM)

```bash
git clone https://github.com/karimgamalmahmoud/Voice-Agentic-Systems.git
cd Voice-Agentic-Systems

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# LLM server (separate terminal)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5:7b-instruct

python -m voice_agent.app --warm          # http://localhost:7860
```

`src/` needs to be importable — either `pip install -e .` or
`export PYTHONPATH=src`.

### No GPU at all?

The decision policy, corpus chunking and language detection are pure Python and
tested without any model:

```bash
pip install pydantic pytest numpy
python -m pytest tests/ -q          # 26 tests, ~4s
```

---

## Architecture

```
  mic / mp3
      │
      ▼
┌──────────────┐   Whisper large-v3 (transformers)
│     STT      │   auto-detect; language classified from output script
└──────┬───────┘
       │  Transcript{text, language, arabic_ratio}
       ▼
┌──────────────┐   BGE-M3 dense retrieval over assets/corpus
│  RETRIEVAL   │   Arabic query → English corpus, no translation hop
└──────┬───────┘   splits rubric competencies (scoring targets)
       │           from reference notes (supporting detail)
       ▼
┌──────────────┐   LLM → strict JSON, one entry per competency:
│   COVERAGE   │   covered | partial | missing + verbatim evidence + confidence
└──────┬───────┘
       │  CoverageReport
       ▼
┌──────────────┐   ◀── deterministic Python, NOT a model call
│   DECISION   │       decide_follow_up() in src/voice_agent/agent.py
└──┬────────┬──┘
   │        │
   │ ask    │ move on
   ▼        │
┌──────────┐│      LLM composes ONE probe, in the candidate's language
│ FOLLOW-UP││      → TTS → second mic capture → transcribe
└────┬─────┘│
     └───┬───┘
         ▼
┌──────────────┐   LLM scores against retrieved rubric + levels
│    SCORE     │   1–5 + one-line justification, forced to input language
└──────┬───────┘
       ▼
┌──────────────┐   MMS-TTS (VITS), Arabic or English
│     TTS      │
└──────────────┘
```

### The stack, and why

| Stage | Choice | Why this one |
|---|---|---|
| STT | `openai/whisper-large-v3` | Best open Arabic ASR. **Not** `large-v3-turbo` — turbo measurably degrades on non-English, and dialectal code-mixing is the hardest thing here. |
| LLM | `qwen2.5:7b-instruct` via Ollama | Strongest Arabic that fits a free T4 next to the other models, and dependable at structured JSON. Ollama keeps its deps in a separate process. |
| Embeddings | `BAAI/bge-m3` | Genuinely multilingual. Arabic transcripts retrieve English rubric sections directly. A monolingual embedder fails silently here. |
| TTS | `facebook/mms-tts-{ara,eng}` | Installs with plain `transformers`, no system packages. Honestly the weak link — see Tradeoffs. |
| UI | Gradio | `share=True` is what makes the Colab microphone work. |

Everything is repointable by environment variable — see
[`src/voice_agent/config.py`](src/voice_agent/config.py).

---

## How the agent decides to follow up

This is the part of the brief I spent the most care on, so it is worth being
precise about where the decision actually lives.

The LLM does **not** decide. It produces a grounded, structured coverage
report — for each retrieved rubric competency: `covered` / `partial` /
`missing`, a verbatim quote from the transcript, and a confidence. Plain Python
then applies a policy over those typed fields:
[`decide_follow_up()`](src/voice_agent/agent.py).

**The policy:**

1. **Only `partial` is worth probing.** This is the whole idea. Someone who
   said "yeah, EF Core can be slow" but never said *why* has a gap one question
   resolves. Someone who never mentioned data access will not learn it in
   fifteen seconds, and someone who explained N+1 queries properly has nothing
   left to clarify. `covered` and `missing` are both terminal; `partial` is the
   only recoverable state.
2. **Low-confidence partials are ignored** (< 0.45). A `partial` the model is
   unsure about is not evidence of a real gap.
3. **Broadly-empty answers are exempt.** If ≥ 60% of competencies are missing,
   the score band is already settled — one probe cannot move a 2 into a 4, and
   asking wastes the candidate's time.
4. **Highest-value gap wins.** Multiple soft spots, one question. Competencies
   are weighted: diagnostic method (1.0) and async (0.95) outrank caching (0.6)
   and communication (0.4), because the rubric leads on measure-before-changing
   and the question names async/await explicitly.
5. **One follow-up, hard cap**, per the brief.

**Why not just ask the LLM?** Three reasons. A 7B model asked "should I follow
up?" gives a different answer run to run, which makes the branch untestable. A
policy over typed fields is inspectable — every decision carries a `reason`
string that the UI renders and the quality gate snapshots. And it is tunable by
someone who is not an ML engineer: the thresholds are config, not prompt.

The tradeoff is real and I would defend it either way: a prompt-based decision
would pick up nuance my five rules miss. I took determinism because this is the
graded branch and "it does something explicable every time" beats "it is
occasionally cleverer".

All five rules are unit-tested in
[`tests/test_decision.py`](tests/test_decision.py) — no GPU required.

---

## Handling Arabic / code-mixed speech

Two of the three provided samples are Egyptian Arabic carrying English technical
vocabulary ("async", "EF Core", "thread pool").

- **Whisper auto-detects** rather than being forced to `ar` — forcing it
  transliterates English terms into Arabic script and mangles them.
- **Language is then classified from the output script, not from Whisper's
  language token.** Whisper detects per 30-second chunk, so a code-mixed answer
  can flip mid-file. Counting Arabic letters across the whole transcript is
  stable and gives exactly the binary needed: which language do we reply in.
  The threshold is 15% Arabic script, deliberately low, because a genuinely
  Arabic answer can be 40% Latin technical terms.
- **The scoring prompt is explicitly told** to judge technical substance and
  never fluency or transcription quality, and to keep English technical terms in
  Latin script when writing Arabic — the way an Egyptian engineer actually
  speaks.

---

## Quality gate

```bash
python scripts/run_quality_gate.py
```

Runs all three provided samples end to end and writes
[`docs/QUALITY_GATE_RESULTS.md`](docs/QUALITY_GATE_RESULTS.md) with each
transcript, what was retrieved, the branch decision and its reason, and the
score.

**It earned its keep.** On the first full run it failed, and everything it
caught was a real defect:

| What the gate caught | Root cause | Fix |
|---|---|---|
| `04_api_design_security` retrieved for all 3 samples | reference `top_k=4` against a pool of only 5 notes — not a filter | own `top_k=3` + a relative-score floor, tuned on the recorded scores |
| — (found by reading its output) | coverage marked EF Core `missing` for an answer that described N+1 and a missing index *in dialect* | prompt now credits the concept, not the English vocabulary |
| — (found by reading its output) | a follow-up truncated mid-word: `"كيف تتعامل مع إبطال"` | validate before use, retry warmer, fall back to a templated probe |
| An English answer answered with an **Arabic** follow-up | language fidelity was checked on the score but never on the question | check both ends; the gate now asserts the question's script too |

That last row is the useful one: the gate was passing while shipping a
wrong-language question, because it only guarded the output it knew about.
Two of the four were found by reading the report rather than by an assertion —
which is exactly why the report prints transcripts, retrieval scores and branch
reasoning instead of just a pass/fail.

**How I'd know if a change made the system worse.** The gate pins the things
that should never move and tolerates the one thing that legitimately does.
Retrieval precision is binary: the corpus contains two reference notes
(dependency injection, API security) that are irrelevant to a "slow endpoint
under load" question, and if a chunking or embedding change lets them into the
retrieved set, retrieval has stopped discriminating. Language fidelity is
likewise binary — an Arabic answer scored back in English is a regression no
score comparison would catch. Scores themselves I treat as noisy and compare
against a stored baseline with a ±1 tolerance, but two samples drifting together
in one run, or any sample moving two points, means the scoring band shifted and
a human needs to look. Branch decisions are snapshotted too: if a sample that
used to earn a follow-up stops earning one, the coverage pass has changed its
partial/missing calibration even when the final score lands in the same place.

In batch mode no human is present to answer a clarifying question, so when the
agent decides to follow up, the question it composed is recorded and scoring
falls back to the original answer. The branch is still exercised and logged.

---

## Tradeoffs and what I cut

**`transformers` instead of `faster-whisper`.** faster-whisper is roughly 3×
quicker, but its ctranslate2 backend pins a cuDNN major version that regularly
conflicts with Colab's preinstalled CUDA stack. "It runs from a clean checkout"
is the first grading criterion; throughput is not. I took the slower path that
does not break.

**TTS is the weakest component and I want to be upfront about it.**
Self-hostable Arabic TTS is genuinely thin. MMS-TTS is intelligible MSA, not
pleasant Egyptian, and it needs romanized input so it mangles some terms. Better
options exist (XTTS-v2, Chatterbox Multilingual) but cost setup time and license
clarity. Given the brief says polish is not graded, I spent the time on the
decision branch instead. Every TTS failure path degrades to "no audio, text
still shown" — losing the speak-back must never take down the loop.

**No vector database.** The corpus is ~6 KB. A numpy dot product is the entire
search engine; anything else would be ceremony.

**Cut deliberately:** streaming / barge-in (turn-based capture only),
diarization, multi-question interviews, session persistence, auth, any
containerization.

**Known limits.** Single-user, in-memory state. All four models sit resident in
VRAM (~10 GB of the T4's 15 GB) — fine for one session, wrong for a service.
The confidence numbers driving the branch are self-reported by a 7B model and
are not calibrated; they work as a relative ranking, not as probabilities.

## What I'd build next

1. **Calibrate the branch on real data.** The thresholds (0.45 confidence, 60%
   missing) are reasoned, not measured. Twenty labelled answers would turn them
   into something defensible.
2. **Streaming STT with barge-in**, so the exchange feels like a conversation
   rather than a walkie-talkie.
3. **Better Arabic TTS** — an Egyptian-dialect XTTS-v2 fine-tune is the obvious
   upgrade and would do more for perceived quality than anything else here.
4. **An LLM-judge regression harness** over a larger answer bank, so scoring
   changes are evaluated on distribution shift rather than three samples.
5. **Split the models behind separate services** so STT, embeddings and the LLM
   scale independently.

---

## Repo layout

```
src/voice_agent/
  config.py       all model ids + thresholds, env-overridable
  schemas.py      pydantic contracts between stages
  prompts.py      the three prompt templates
  stt.py          Whisper + script-based language detection
  retrieval.py    corpus chunking + BGE-M3 index
  llm.py          OpenAI-wire client, JSON repair + retry
  tts.py          MMS-TTS, degrades to silent on failure
  agent.py        orchestration + decide_follow_up()   ← the branch
  app.py          Gradio UI
scripts/
  transcribe_samples.py   STT spike over the provided audio
  run_quality_gate.py     full pipeline over all samples + invariants
tests/                    26 CPU-only tests
notebooks/run_on_colab.ipynb
assets/corpus/            rubric + reference notes (from the task package)
assets/audio/             provided sample answers
docs/PROMPTS.md           coding-agent prompt log
CLAUDE.md                 coding-agent context file
```

## Agent artifacts

[`CLAUDE.md`](CLAUDE.md) is the context file I gave the coding agent.
[`docs/PROMPTS.md`](docs/PROMPTS.md) is the log of the prompts that actually
drove the build, including the four places I rejected the obvious default and
why.
