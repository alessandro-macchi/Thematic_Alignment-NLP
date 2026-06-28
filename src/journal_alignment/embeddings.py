"""Embedding utilities for representing journal content."""

from __future__ import annotations

from typing import Any

import numpy as np


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

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a non-empty list of texts into a NumPy array."""

        if not texts:
            raise ValueError("texts must contain at least one text.")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("All items in texts must be strings.")
        if any(not text.strip() for text in texts):
            raise ValueError("texts must not contain empty strings.")

        if any(len(chunk_text_by_tokens(text, self.model)) > 1 for text in texts):
            return np.vstack([embed_long_text(text, self.model) for text in texts])

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

        return self.encode_texts([text])[0]
