"""Speech-to-text on Whisper large-v3.

Deliberately uses plain `transformers` rather than faster-whisper. faster-whisper
is ~3x quicker, but its ctranslate2 backend pins a cuDNN major version that
regularly conflicts with Colab's preinstalled CUDA stack - and "it runs from a
clean checkout" outranks throughput here. See README (Tradeoffs).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .config import CONFIG, STTConfig
from .schemas import Transcript

logger = logging.getLogger(__name__)

_ARABIC_RANGES = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)


def _is_arabic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _ARABIC_RANGES)


def script_language(text: str) -> tuple[str, float]:
    """Classify language by script, returning (lang, arabic_ratio).

    We classify from the decoded text rather than from Whisper's own language
    token on purpose. Whisper detects per 30-second chunk, so a code-mixed
    answer can flip mid-file and produce a mixed-script transcript. Counting
    Arabic letters over the whole output is stable, and it gives exactly the
    binary we need: which language do we reply in.
    """
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return "en", 0.0
    arabic = sum(1 for c in letters if _is_arabic_char(c))
    ratio = arabic / len(letters)
    # Threshold is low because code-mixed Egyptian Arabic carries a lot of
    # Latin-script technical vocabulary ("async", "EF Core", "thread pool").
    # An answer that is 25% Arabic script is an Arabic answer.
    return ("ar" if ratio >= 0.15 else "en"), ratio


def _collapse_repeats(text: str) -> str:
    """Whisper occasionally loops a phrase on long silences. Collapse runs of an
    identical sentence down to one so the loop does not poison the LLM context."""
    parts = re.split(r"(?<=[.!?۔؟])\s+", text)
    out: list[str] = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        if out and out[-1].strip().lower() == stripped.lower():
            continue
        out.append(stripped)
    return " ".join(out)


class Transcriber:
    """Lazily-loaded Whisper wrapper. Load once, reuse across turns."""

    def __init__(self, cfg: Optional[STTConfig] = None, device: Optional[str] = None):
        self.cfg = cfg or CONFIG.stt
        self.device = device or CONFIG.resolved_device()
        self._pipe = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info("Loading STT %s on %s (%s)", self.cfg.model_id, self.device, dtype)
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.cfg.model_id,
            torch_dtype=dtype,
            device=0 if self.device == "cuda" else -1,
            chunk_length_s=self.cfg.chunk_length_s,
            batch_size=self.cfg.batch_size,
        )

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray, tuple],
        language: Optional[str] = None,
    ) -> Transcript:
        """Transcribe a file path, a numpy waveform, or a Gradio (sr, data) tuple.

        `language` forces a decode language; leave it None to let Whisper
        auto-detect, which is what we want for code-mixed input.
        """
        self.load()
        assert self._pipe is not None

        payload, duration = self._normalise(audio)
        generate_kwargs: dict = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language

        result = self._pipe(payload, generate_kwargs=generate_kwargs, return_timestamps=False)
        text = _collapse_repeats((result.get("text") or "").strip())
        lang, ratio = script_language(text)
        logger.info("Transcribed %.1fs -> %s (arabic_ratio=%.2f)", duration or -1, lang, ratio)
        return Transcript(text=text, language=lang, arabic_ratio=ratio, duration_s=duration)

    def _normalise(self, audio) -> tuple[object, Optional[float]]:
        """Return something the pipeline accepts, plus duration when known."""
        sr = self.cfg.sample_rate

        if isinstance(audio, (str, Path)):
            path = str(audio)
            data = self._load_file(path, sr)
            return {"raw": data, "sampling_rate": sr}, len(data) / sr

        # Gradio numpy audio components hand back (sample_rate, samples).
        if isinstance(audio, tuple) and len(audio) == 2:
            in_sr, data = audio
            data = self._to_mono_float32(np.asarray(data))
            if in_sr != sr:
                data = self._resample(data, int(in_sr), sr)
            return {"raw": data, "sampling_rate": sr}, len(data) / sr

        if isinstance(audio, np.ndarray):
            data = self._to_mono_float32(audio)
            return {"raw": data, "sampling_rate": sr}, len(data) / sr

        raise TypeError(f"Unsupported audio input: {type(audio)!r}")

    @staticmethod
    def _to_mono_float32(data: np.ndarray) -> np.ndarray:
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        # Gradio hands back int16 for mic capture; scale to [-1, 1].
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak > 1.0:
            data = data / 32768.0
        return data

    @staticmethod
    def _resample(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return data
        n = int(round(len(data) * dst_sr / src_sr))
        if n <= 0:
            return data
        # Linear interpolation is plenty: Whisper's own front end is far lossier
        # than the resampler, and this avoids a scipy/librosa dependency.
        return np.interp(
            np.linspace(0.0, len(data) - 1, n, dtype=np.float64),
            np.arange(len(data), dtype=np.float64),
            data,
        ).astype(np.float32)

    def _load_file(self, path: str, target_sr: int) -> np.ndarray:
        """Decode an audio file to mono float32 at target_sr.

        Tries soundfile first (fast, no subprocess) and falls back to ffmpeg via
        the helper transformers ships, which handles mp3 everywhere.
        """
        try:
            import soundfile as sf

            data, sr = sf.read(path, dtype="float32", always_2d=False)
            data = self._to_mono_float32(np.asarray(data))
            return self._resample(data, int(sr), target_sr)
        except Exception as exc:  # noqa: BLE001 - fall through to ffmpeg
            logger.debug("soundfile could not read %s (%s); using ffmpeg", path, exc)

        from transformers.pipelines.audio_utils import ffmpeg_read

        with open(path, "rb") as fh:
            raw = fh.read()
        return np.asarray(ffmpeg_read(raw, target_sr), dtype=np.float32)
