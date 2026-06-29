"""Embedding utilities for representing journal content."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkingReport:
    """Summary of how often texts needed chunking before embedding."""

    text_label: str
    total_texts: int
    multi_chunk_texts: int
    max_chunks_for_one_text: int
    model_max_tokens: int
    effective_chunk_token_limit: int

    @property
    def single_chunk_texts(self) -> int:
        """Number of texts represented by one model input chunk."""

        return self.total_texts - self.multi_chunk_texts

    @property
    def multi_chunk_share(self) -> float:
        """Share of texts that required more than one chunk."""

        if self.total_texts == 0:
            return 0.0
        return self.multi_chunk_texts / self.total_texts

    def to_dataframe(self):
        """Return the report as a one-row pandas DataFrame."""

        import pandas as pd

        return pd.DataFrame(
            {
                "text_label": [self.text_label],
                "total_texts": [self.total_texts],
                "single_chunk_texts": [self.single_chunk_texts],
                "multi_chunk_texts": [self.multi_chunk_texts],
                "multi_chunk_share": [self.multi_chunk_share],
                "max_chunks_for_one_text": [self.max_chunks_for_one_text],
                "model_max_tokens": [self.model_max_tokens],
                "effective_chunk_token_limit": [self.effective_chunk_token_limit],
            }
        )


def chunk_text_by_tokens(
    text: str,
    model: Any,
    max_tokens: int | None = None,
) -> list[str]:
    """Split text into chunks that fit the model tokenizer limit."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not text.strip():
        return []

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise AttributeError("model must expose a tokenizer.")

    token_limit = max_tokens if max_tokens is not None else model.max_seq_length
    if token_limit < 1:
        raise ValueError("max_tokens must be at least 1.")

    # Reserve room for special tokens added by the model tokenizer.
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    chunk_token_limit = max(1, token_limit - special_tokens)
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) <= chunk_token_limit:
        return [text]

    chunks = []
    for start in range(0, len(token_ids), chunk_token_limit):
        chunk_ids = token_ids[start : start + chunk_token_limit]
        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def embed_long_text(text: str, model: Any) -> np.ndarray:
    """Embed text, averaging chunk embeddings when the text is too long."""

    chunks = chunk_text_by_tokens(text, model)
    if not chunks:
        raise ValueError("text must be a non-empty string.")
    if len(chunks) == 1:
        return model.encode(
            chunks[0],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    chunk_embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.mean(chunk_embeddings, axis=0)


class EmbeddingModel:
    """Small wrapper around a SentenceTransformer embedding model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
    ) -> None:
        """Initialize the sentence embedding model.

        Parameters
        ----------
        model_name:
            Name of the SentenceTransformer model to load.
        batch_size:
            Number of texts encoded in each batch.
        """

        if not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required to use EmbeddingModel. "
                "Install it with `pip install sentence-transformers`."
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.model: Any = SentenceTransformer(model_name)
        self.last_chunking_report: ChunkingReport | None = None

    def encode_texts(
        self,
        texts: list[str],
        text_label: str = "texts",
    ) -> np.ndarray:
        """Encode a non-empty list of texts into a NumPy array."""

        if not texts:
            raise ValueError("texts must contain at least one text.")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("All items in texts must be strings.")
        if any(not text.strip() for text in texts):
            raise ValueError("texts must not contain empty strings.")

        chunks_by_text = [chunk_text_by_tokens(text, self.model) for text in texts]
        self.last_chunking_report = self._build_chunking_report(
            chunks_by_text,
            text_label=text_label,
        )
        self._log_chunking_report(self.last_chunking_report)

        if any(len(chunks) > 1 for chunks in chunks_by_text):
            return np.vstack(
                [
                    self._encode_precomputed_chunks(chunks)
                    for chunks in chunks_by_text
                ]
            )

        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def encode_single(self, text: str) -> np.ndarray:
        """Encode one text into a single embedding vector."""

        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if not text.strip():
            raise ValueError("text must be a non-empty string.")

        return self.encode_texts([text], text_label="text")[0]

    def _build_chunking_report(
        self,
        chunks_by_text: list[list[str]],
        text_label: str,
    ) -> ChunkingReport:
        """Create a chunking report from precomputed text chunks."""

        tokenizer = getattr(self.model, "tokenizer", None)
        special_tokens = (
            tokenizer.num_special_tokens_to_add(pair=False)
            if tokenizer is not None
            else 0
        )
        model_max_tokens = int(self.model.max_seq_length)
        effective_limit = max(1, model_max_tokens - special_tokens)
        chunk_counts = [len(chunks) for chunks in chunks_by_text]

        return ChunkingReport(
            text_label=text_label,
            total_texts=len(chunks_by_text),
            multi_chunk_texts=sum(count > 1 for count in chunk_counts),
            max_chunks_for_one_text=max(chunk_counts),
            model_max_tokens=model_max_tokens,
            effective_chunk_token_limit=effective_limit,
        )

    def _encode_precomputed_chunks(self, chunks: list[str]) -> np.ndarray:
        """Encode one text from chunks that were already tokenizer-split."""

        if len(chunks) == 1:
            return self.model.encode(
                chunks[0],
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        chunk_embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.mean(chunk_embeddings, axis=0)

    @staticmethod
    def _log_chunking_report(report: ChunkingReport) -> None:
        """Log a compact summary of chunking behavior."""

        LOGGER.info(
            "Chunking report for %s: %s/%s required more than one chunk "
            "(%.2f%%; model limit: %s tokens; effective chunk limit: %s).",
            report.text_label,
            report.multi_chunk_texts,
            report.total_texts,
            report.multi_chunk_share * 100,
            report.model_max_tokens,
            report.effective_chunk_token_limit,
        )
