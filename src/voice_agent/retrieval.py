"""Cross-lingual retrieval over the rubric corpus.

The corpus is English; answers arrive in Egyptian Arabic. BGE-M3 embeds both
into one space, so an Arabic transcript retrieves English rubric sections with
no translation hop.

The corpus is deliberately split into two roles:

  * the rubric's `## Competency:` sections are the *scoring targets* - the
    things the coverage pass must assess one by one;
  * files 01-05 are *supporting references* that give the model the technical
    detail behind each competency.

Two of the five reference notes (dependency injection, API security) are not
relevant to this question. Leaving them in and checking they stay out of the
top hits is the retrieval half of the quality gate.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import CONFIG, CORPUS_DIR, RetrievalConfig
from .schemas import RetrievedChunk

logger = logging.getLogger(__name__)

RUBRIC_DOC_ID = "00_scoring_rubric"
COMPETENCY_PREFIX = "Competency:"


@dataclass
class Chunk:
    doc_id: str
    section: str
    text: str

    @property
    def is_competency(self) -> bool:
        return self.doc_id == RUBRIC_DOC_ID and self.section.startswith(COMPETENCY_PREFIX)

    @property
    def competency_name(self) -> str:
        return self.section[len(COMPETENCY_PREFIX) :].strip()

    def embed_text(self) -> str:
        # Prepending the heading gives the embedder the topic even when the body
        # is a couple of terse sentences, which most of this corpus is.
        return f"{self.section}\n{self.text}"


def _split_markdown(path: Path) -> list[Chunk]:
    """Split a markdown file into `## `-delimited chunks.

    Content before the first `## ` is emitted as an Overview chunk so nothing in
    a short file is dropped.
    """
    doc_id = path.stem
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    chunks: list[Chunk] = []
    section = "Overview"
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            chunks.append(Chunk(doc_id=doc_id, section=section, text=body))

    for line in lines:
        if line.startswith("## "):
            flush()
            section = line[3:].strip()
            buffer = []
        elif line.startswith("# "):
            # Document title - keep as the Overview heading, do not open a chunk.
            if not buffer and section == "Overview":
                section = line[2:].strip()
        else:
            buffer.append(line)
    flush()
    return chunks


def load_corpus(corpus_dir: Optional[Path] = None) -> list[Chunk]:
    directory = Path(corpus_dir or CORPUS_DIR)
    files = sorted(directory.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No markdown files found in corpus dir: {directory}")
    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(_split_markdown(path))
    logger.info("Loaded %d chunks from %d corpus files", len(chunks), len(files))
    return chunks


class CorpusIndex:
    """In-memory dense index. The corpus is ~6 KB, so a numpy dot product is the
    entire search engine - a vector DB here would be pure ceremony."""

    def __init__(
        self,
        cfg: Optional[RetrievalConfig] = None,
        corpus_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.cfg = cfg or CONFIG.retrieval
        self.device = device or CONFIG.resolved_device()
        self.chunks = load_corpus(corpus_dir)
        self._model = None
        self._matrix: Optional[np.ndarray] = None

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        if self._matrix is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedder %s on %s", self.cfg.model_id, self.device)
        self._model = SentenceTransformer(self.cfg.model_id, device=self.device)
        self._matrix = self._encode([c.embed_text() for c in self.chunks])

    def _encode(self, texts: list[str]) -> np.ndarray:
        assert self._model is not None
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    # -- queries -----------------------------------------------------------

    def _search(self, query: str, candidates: list[int], top_k: int) -> list[RetrievedChunk]:
        self.load()
        assert self._matrix is not None
        if not candidates:
            return []
        q = self._encode([query])[0]
        sub = self._matrix[candidates]
        scores = sub @ q  # both sides are L2-normalised, so this is cosine
        order = np.argsort(-scores)[:top_k]
        return [
            RetrievedChunk(
                doc_id=self.chunks[candidates[i]].doc_id,
                section=self.chunks[candidates[i]].section,
                text=self.chunks[candidates[i]].text,
                score=float(scores[i]),
            )
            for i in order
        ]

    def rubric_levels(self) -> str:
        """The 1-5 level descriptions. Always injected, never retrieved - the
        scale must not depend on whether a similarity search happened to find
        it."""
        for chunk in self.chunks:
            if chunk.doc_id == RUBRIC_DOC_ID and chunk.section.lower().startswith("levels"):
                return chunk.text
        return ""

    def all_competencies(self) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(doc_id=c.doc_id, section=c.section, text=c.text, score=1.0)
            for c in self.chunks
            if c.is_competency
        ]

    def retrieve_competencies(self, query: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
        """Rank rubric competencies against the question + answer.

        The rubric says to score only the competencies the question touches, so
        this is a genuine filter: of the five competencies, the .NET latency
        question should surface diagnostics, EF Core, async and caching, and
        leave the rest below the cut.
        """
        k = top_k or self.cfg.top_k
        idx = [i for i, c in enumerate(self.chunks) if c.is_competency]
        hits = self._search(query, idx, len(idx))
        kept = [h for h in hits if h.score >= self.cfg.min_score][:k]
        # Never let a bad threshold strip the scoring targets entirely.
        if not kept:
            logger.warning("No competency cleared min_score=%.2f; keeping top %d", self.cfg.min_score, k)
            kept = hits[:k]
        return kept

    def retrieve_references(self, query: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
        """Supporting technical detail from the 01-05 reference notes."""
        k = top_k or self.cfg.top_k
        idx = [i for i, c in enumerate(self.chunks) if c.doc_id != RUBRIC_DOC_ID]
        hits = self._search(query, idx, len(idx))
        return [h for h in hits if h.score >= self.cfg.min_score][:k]


def format_competencies(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"### {c.section[len(COMPETENCY_PREFIX):].strip() if c.section.startswith(COMPETENCY_PREFIX) else c.section}\n{c.text}"
        for c in chunks
    )


def format_references(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(none retrieved)"
    return "\n\n".join(f"### {c.doc_id} - {c.section}\n{c.text}" for c in chunks)


def competency_names(chunks: list[RetrievedChunk]) -> list[str]:
    names = []
    for c in chunks:
        name = c.section
        if name.startswith(COMPETENCY_PREFIX):
            name = name[len(COMPETENCY_PREFIX) :].strip()
        names.append(name)
    return names
