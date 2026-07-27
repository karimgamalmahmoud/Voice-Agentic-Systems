"""Central configuration.

Every model id and tunable lives here so the whole stack can be repointed from
environment variables without touching code. Defaults are sized for a free-tier
Colab T4 (16 GB VRAM).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = Path(os.getenv("VA_CORPUS_DIR", REPO_ROOT / "assets" / "corpus"))
AUDIO_DIR = Path(os.getenv("VA_AUDIO_DIR", REPO_ROOT / "assets" / "audio"))
ARTIFACT_DIR = Path(os.getenv("VA_ARTIFACT_DIR", REPO_ROOT / "artifacts"))

# The screening question, verbatim from the task package.
INTERVIEW_QUESTION = (
    "A REST endpoint in a .NET backend has become slow under load - it was fine "
    "in testing but times out when real traffic hits it. Walk me through how "
    "you'd investigate and fix it. Cover where async/await and EF Core fit in."
)

INTERVIEW_QUESTION_AR = (
    "عندنا REST endpoint في باك إند دوت نت بقى بطيء تحت الضغط. "
    "كان شغال تمام في الـ testing لكن بيعمل timeout لما الترافيك الحقيقي يزيد. "
    "اشرح لي إزاي هتحقق في المشكلة وتصلحها، "
    "وفين مكان الـ async/await والـ EF Core في الموضوع ده."
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


@dataclass
class STTConfig:
    # large-v3 rather than turbo: turbo measurably degrades on Arabic, and
    # dialectal code-mixing is the hardest part of this task.
    model_id: str = os.getenv("VA_STT_MODEL", "openai/whisper-large-v3")
    chunk_length_s: int = 30
    batch_size: int = int(os.getenv("VA_STT_BATCH", "8"))
    sample_rate: int = 16_000


@dataclass
class LLMConfig:
    # Ollama exposes an OpenAI-compatible API. Repointing base_url at any other
    # OpenAI-compatible endpoint is the only change needed to swap providers.
    base_url: str = os.getenv("VA_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    api_key: str = os.getenv("VA_LLM_API_KEY", "ollama")  # Ollama ignores this
    model: str = os.getenv("VA_LLM_MODEL", "qwen2.5:7b-instruct")
    temperature: float = _env_float("VA_LLM_TEMPERATURE", 0.2)
    max_tokens: int = int(os.getenv("VA_LLM_MAX_TOKENS", "1200"))
    # Generous on purpose. A T4 that has had to spill the model to CPU runs at
    # single-digit tokens/sec, and a slow answer is far more diagnosable than a
    # timeout stack trace. If you hit this ceiling, the real problem is almost
    # always VRAM pressure - check `ollama ps` shows 100% GPU.
    request_timeout: float = _env_float("VA_LLM_TIMEOUT", 300.0)
    max_retries: int = int(os.getenv("VA_LLM_RETRIES", "3"))


@dataclass
class RetrievalConfig:
    # BGE-M3 is genuinely multilingual: Arabic queries retrieve against an
    # English corpus without a translation hop.
    model_id: str = os.getenv("VA_EMBED_MODEL", "BAAI/bge-m3")
    top_k: int = int(os.getenv("VA_RETRIEVAL_TOP_K", "4"))
    min_score: float = _env_float("VA_RETRIEVAL_MIN_SCORE", 0.30)


@dataclass
class TTSConfig:
    arabic_model_id: str = os.getenv("VA_TTS_AR_MODEL", "facebook/mms-tts-ara")
    english_model_id: str = os.getenv("VA_TTS_EN_MODEL", "facebook/mms-tts-eng")
    enabled: bool = _env_bool("VA_TTS_ENABLED", True)


@dataclass
class DecisionConfig:
    """Thresholds for the follow-up branch. See agent.decide_follow_up."""

    # A competency must be judged 'partial' with at least this confidence
    # before it is considered worth probing.
    min_partial_confidence: float = _env_float("VA_MIN_PARTIAL_CONF", 0.45)
    # If this fraction or more of the assessed competencies are outright
    # missing, the answer is weak across the board and one probe cannot move
    # the score band - so we skip the follow-up and score what we have.
    weak_answer_missing_ratio: float = _env_float("VA_WEAK_MISSING_RATIO", 0.60)
    # Hard cap, mandated by the brief: at most one clarifying follow-up.
    max_follow_ups: int = 1


@dataclass
class AppConfig:
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    device: str = os.getenv("VA_DEVICE", "auto")

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover - torch is a hard dep at runtime
            return "cpu"


CONFIG = AppConfig()
