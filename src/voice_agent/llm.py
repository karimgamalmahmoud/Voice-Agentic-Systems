"""LLM client.

Talks OpenAI's wire format, pointed by default at a local Ollama server. That
choice buys two things: the LLM's dependencies stay isolated from the
torch/Whisper/TTS stack (the single biggest cause of "the notebook won't
install"), and swapping to any hosted OpenAI-compatible endpoint is one
environment variable.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from .config import CONFIG, LLMConfig

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    pass


def _extract_json(raw: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Small instruct models fence their JSON or prepend a sentence often enough
    that tolerating it is cheaper than burning a retry.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty response")

    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost balanced {...} span.
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"unbalanced JSON in response: {text[:200]!r}")


class LLMClient:
    def __init__(self, cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or CONFIG.llm
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.cfg.base_url,
                api_key=self.cfg.api_key,
                timeout=self.cfg.request_timeout,
                max_retries=0,  # we retry ourselves so we can also retry bad JSON
            )
        return self._client

    def health(self) -> tuple[bool, str]:
        """Cheap reachability probe, surfaced in the UI so a dead Ollama shows
        up as a clear message instead of a stack trace mid-run."""
        try:
            client = self._ensure()
            models = client.models.list()
            names = [m.id for m in models.data]
            if not names:
                return False, f"{self.cfg.base_url} reachable but serving no models"
            base = self.cfg.model.split(":")[0]
            if not any(n == self.cfg.model or n.split(":")[0] == base for n in names):
                return False, (
                    f"model '{self.cfg.model}' not found. Available: {', '.join(names[:6])}. "
                    f"Run: ollama pull {self.cfg.model}"
                )
            return True, f"{self.cfg.model} ready at {self.cfg.base_url}"
        except Exception as exc:  # noqa: BLE001 - any failure is a failed probe
            return False, f"cannot reach LLM at {self.cfg.base_url}: {exc}"

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        client = self._ensure()
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = client.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001
                last = exc
                # Older Ollama builds reject response_format; drop it and retry.
                if json_mode and "response_format" in kwargs and _is_bad_request(exc):
                    logger.warning("Server rejected json_mode; retrying without it")
                    kwargs.pop("response_format", None)
                    continue
                logger.warning("LLM call failed (attempt %d/%d): %s", attempt, self.cfg.max_retries, exc)
                if attempt < self.cfg.max_retries:
                    time.sleep(1.5 * attempt)
        raise LLMError(f"LLM request failed after {self.cfg.max_retries} attempts: {last}") from last

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Like `complete`, but retries when the payload will not parse.

        A malformed-JSON retry appends the parser error to the prompt, which
        recovers a 7B model far more reliably than resampling blind.
        """
        attempt_user = user
        last: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            raw = self.complete(
                system,
                attempt_user,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
            )
            try:
                return _extract_json(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                last = exc
                logger.warning("Unparseable JSON (attempt %d/%d): %s", attempt, self.cfg.max_retries, exc)
                attempt_user = (
                    f"{user}\n\n---\nYour previous reply could not be parsed as JSON "
                    f"({exc}). Return ONLY the JSON object, with no prose and no code fence."
                )
        raise LLMError(f"LLM did not return valid JSON after {self.cfg.max_retries} attempts: {last}")


def _is_bad_request(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in (400, 422):
        return True
    return "response_format" in str(exc).lower()
