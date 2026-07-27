# Coding-agent prompt log

Built with **Claude Code** (Opus). This is the log the brief asks for: the
prompts that actually drove the build, and the points where I pushed back on
what came out.

The context file the agent worked against is [`../CLAUDE.md`](../CLAUDE.md).

---

## How I drove it

Three things did most of the work, and they are the transferable part:

**1. I front-loaded the constraints instead of correcting after the fact.**
`CLAUDE.md` was written before any application code and states the grading order
verbatim from the brief, the VRAM ceiling, the "no LLM in the decision path"
rule, and a *Things that will bite you* section. Every one of those entries
exists because it is a trap an agent walks into by default — reaching for
faster-whisper, pinning torch, forcing Whisper to `ar`. Writing them down once
was cheaper than catching them in review three times.

**2. I made the agent state a recommendation before writing code.** For the
stack choice I asked for a decision with reasoning, not options. Surveys of
alternatives are cheap to generate and expensive to read.

**3. I forced the risky thing to the front.** Egyptian Arabic code-mixing was
the highest-uncertainty part of the stack, so the transcription spike
(`scripts/transcribe_samples.py`) was built before anything that depended on it.
Discovering an STT problem in hour three would have been fatal to the time box.

---

## Phase 1 — Understand the task

> this directory contain assignment for me "D:\…\AI Engineer Take Home
> Assignment" please check it and tell me about it please

Read the brief, rubric and corpus. The useful output was noticing that the
corpus contains **five reference notes of which only three are relevant** to the
question. That is not padding — it is the retrieval test. It became the primary
assertion in the quality gate.

## Phase 2 — Feasibility, before committing

> can we do all of this on google colab with gpu? and use python and opensource
> self hosted models only?

I asked for a straight yes/no with a VRAM budget rather than a menu. The answer
that mattered was the risk, not the stack: **an all-self-hosted repo cannot run
on a grader's laptop**, and "it runs from a clean checkout" is grading criterion
#1. That surfaced before any code was written and shaped the whole delivery —
the Colab notebook became the primary run path, and the LLM went behind an
OpenAI-compatible endpoint so a grader in a hurry has an escape hatch.

Catching that in phase 2 rather than at submission is the single highest-value
thing that happened in this build.

---

## Phase 3 — The decision branch

This is grading criterion #2, so it got the most deliberate prompting. The
instruction was roughly:

> The follow-up branch is the graded centrepiece. Do not implement it as an LLM
> call. Have the LLM produce a structured coverage report — per competency:
> covered/partial/missing, verbatim evidence, confidence — and then write the
> decision as plain Python over those fields. It has to be unit-testable with no
> GPU, and every decision must carry a human-readable reason.

**The insight worth keeping** came from arguing about what `partial` means. The
first cut treated any weak competency as probe-worthy. That is wrong, and
working out why produced the actual policy: `covered` and `missing` are both
*terminal* states — one has nothing left to ask, the other will not be fixed by
fifteen seconds of clarification. `partial` is the only **recoverable** state.
Once framed that way, the "broadly-empty answers are exempt" rule fell out on
its own: if most competencies are missing, the band is already settled.

## Phase 4 — Where I rejected the default

Four places the obvious choice was wrong. Each is now a comment in the code and
an entry in `CLAUDE.md`, because a future agent session would otherwise
"optimise" them straight back.

**faster-whisper → `transformers`.** faster-whisper is ~3× quicker and is what
any reasonable engineer reaches for. Its ctranslate2 backend pins a cuDNN major
version that conflicts with Colab's preinstalled CUDA stack. Criterion #1 beats
throughput. Rejected the fast option on install-reliability grounds.

**Whisper's language token → script-based detection.** The natural approach is
to trust Whisper's detected language. It detects per 30-second chunk, so a
code-mixed answer can flip mid-file and produce an incoherent result. Counting
Arabic letters across the finished transcript is cruder, stable, and gives
exactly the binary the system needs.

**`language="ar"` → auto-detect.** Forcing Arabic transliterates "EF Core" and
"thread pool" into Arabic script and destroys the technical content the rubric
is grading. This one is counter-intuitive enough that it is called out twice in
the repo.

**In-process LLM → Ollama subprocess.** Loading the LLM in the same process is
simpler and saves a moving part. But dependency conflicts between the LLM stack
and the torch/Whisper/TTS stack are the top cause of Colab notebooks failing to
install, and process isolation removes that class of failure entirely. It also
made the endpoint swappable for free.

## Phase 5 — Quality gate

> Write the gate to assert the things that must never change and tolerate the
> one thing that legitimately does. Scores from a 7B model move run to run —
> asserting exact scores gives a flaky gate that gets ignored.

Produced the split the README describes: retrieval precision and language
fidelity are binary assertions; scores get a ±1 tolerance against a stored
baseline, with a secondary rule that two samples drifting together in one run is
itself a signal even though each is individually within tolerance.

---

---

## Phase 6 — The debug loop, which is where most of the value was

Everything above is pre-flight. The interesting work started once it ran on a
real GPU, and the pattern that mattered was **feeding failures back as
constraints, not just as fixes**.

**Round 1 — install.** Ollama's installer needed `zstd`, absent from the Colab
image. Trivial fix. The non-trivial part was noticing the UI cell blocked
forever under `Run all`, so the tests and the quality gate below it had never
executed. Reordered so the blocking launch is last.

**Round 2 — the timeout that was not a timeout.** Every LLM call timed out. The
obvious read is "raise the timeout"; the actual cause was VRAM. The notebook
preloaded Whisper and BGE-M3 into the kernel, then ran the scripts via
`!python`, and each subprocess loaded a *second* copy. Ollama could not fit and
silently fell back to CPU. The health check passed throughout because it only
lists models and never runs inference, which is what made it look like a network
problem. Fix was structural — load the LLM onto the GPU first, run everything
in-kernel — plus an `ollama ps` check and a tok/s probe so the CPU fallback can
never again be invisible.

The prompt that got there was roughly: *"don't just raise the timeout — the
health check passes and generation doesn't, so tell me what differs between
those two paths."*

**Rounds 3-4 — defects the gate found.** Retrieval pulling a distractor,
coverage under-crediting dialect, a follow-up truncated mid-word, and an English
answer probed in Arabic. Detailed in the README table.

Two lessons worth stating:

*Thresholds guessed are thresholds wrong.* I first set the retrieval cutoff to
0.92 by feel, then realised I had no idea of the score distribution and loosened
it to 0.85 with a comment saying to tune it on real numbers. When the gate
printed the scores, the viable window turned out to be 0.858-0.902 — my original
guess would have silently dropped a needed document. The fix was making the gate
report its scores, not thinking harder about the number.

*A gate only guards what it knows about.* It passed while shipping an Arabic
follow-up to an English answer, because language fidelity was asserted on the
score and nobody had thought about the question. Every check added since covers
both ends of the pair.

---

## What I would do differently

I wrote the Gradio UI before running the pipeline end to end on a GPU. The
CPU-only tests cover the decision policy, corpus chunking and language
detection, but the first genuine end-to-end run happened well after the UI
existed. On a longer timeline I would have stood up the notebook and run the
transcription spike against real weights before building any UI — the brief's
own advice, and I half-followed it.

I would also have made the quality gate print its retrieval scores from the
start. Two of the four real defects were found by *reading* its output rather
than by an assertion, and the one threshold I guessed at was wrong. A report
that shows its working beats a report that says PASS.
