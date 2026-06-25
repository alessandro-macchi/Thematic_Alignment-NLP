"""Alignment metric utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _as_2d_array(values: np.ndarray, name: str) -> np.ndarray:
    """Return ``values`` as a non-empty two-dimensional float array."""

    array = np.asarray(values, dtype=float)

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array.")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature.")

    return array


def _validate_texts(texts: list[str], name: str) -> None:
    """Validate a non-empty list of non-empty strings."""

    if not texts:
        raise ValueError(f"{name} must contain at least one text.")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError(f"All items in {name} must be strings.")
    if any(not text.strip() for text in texts):
        raise ValueError(f"{name} must not contain empty strings.")


def _score_series(df: pd.DataFrame, score_column: str) -> pd.Series:
    """Return a numeric score series after validating the column exists."""

    if score_column not in df.columns:
        raise ValueError(f"Column '{score_column}' is missing from the DataFrame.")

    return pd.to_numeric(df[score_column], errors="coerce")


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarities between two embedding matrices."""

    a_matrix = _as_2d_array(a, "a")
    b_matrix = _as_2d_array(b, "b")

    if a_matrix.shape[1] != b_matrix.shape[1]:
        raise ValueError("a and b must have the same number of features.")

    a_norms = np.linalg.norm(a_matrix, axis=1)
    b_norms = np.linalg.norm(b_matrix, axis=1)

    if np.any(a_norms == 0):
        raise ValueError("a contains zero-vector embeddings.")
    if np.any(b_norms == 0):
        raise ValueError("b contains zero-vector embeddings.")

    return cosine_similarity(a_matrix, b_matrix)


def compute_alignment_scores(
    abstract_embeddings: np.ndarray,
    aims_scope_embedding: np.ndarray,
) -> np.ndarray:
    """Compute one alignment score per abstract against an Aims & Scope vector."""

    abstract_matrix = _as_2d_array(abstract_embeddings, "abstract_embeddings")
    reference_matrix = _as_2d_array(aims_scope_embedding, "aims_scope_embedding")

    if reference_matrix.shape[0] != 1:
        raise ValueError("aims_scope_embedding must contain exactly one embedding.")

    return cosine_similarity_matrix(abstract_matrix, reference_matrix).ravel()


def compute_tfidf_similarity(
    texts: list[str],
    reference_text: str,
) -> np.ndarray:
    """Compute TF-IDF cosine similarity between each text and a reference text."""

    _validate_texts(texts, "texts")
    if not isinstance(reference_text, str):
        raise TypeError("reference_text must be a string.")
    if not reference_text.strip():
        raise ValueError("reference_text must be a non-empty string.")

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([*texts, reference_text])
    similarities = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])

    return similarities.ravel()


def summarize_scores(
    df: pd.DataFrame,
    score_column: str = "alignment_score",
) -> pd.DataFrame:
    """Summarize a score column with basic descriptive statistics."""

    scores = _score_series(df, score_column)

    if scores.dropna().empty:
        raise ValueError(f"Column '{score_column}' does not contain numeric scores.")

    return pd.DataFrame(
        {
            "score_column": [score_column],
            "count": [scores.count()],
            "mean": [scores.mean()],
            "median": [scores.median()],
            "std": [scores.std()],
            "min": [scores.min()],
            "max": [scores.max()],
        }
    )


def aggregate_by_year(
    df: pd.DataFrame,
    score_column: str = "alignment_score",
) -> pd.DataFrame:
    """Aggregate score statistics by publication year."""

    if "year" not in df.columns:
        raise ValueError("Column 'year' is missing from the DataFrame.")

    scores = _score_series(df, score_column)

    if scores.dropna().empty:
        raise ValueError(f"Column '{score_column}' does not contain numeric scores.")

    working = df.copy()
    working[score_column] = scores

    return (
        working.groupby("year", as_index=False)
        .agg(
            count=(score_column, "count"),
            mean=(score_column, "mean"),
            median=(score_column, "median"),
            std=(score_column, "std"),
            min=(score_column, "min"),
            max=(score_column, "max"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )


def detect_outliers_zscore(
    df: pd.DataFrame,
    score_column: str = "alignment_score",
    threshold: float = 2.0,
) -> pd.DataFrame:
    """Add an ``is_outlier`` column using an absolute z-score threshold."""

    if threshold <= 0:
        raise ValueError("threshold must be greater than 0.")

    result = df.copy()
    scores = _score_series(result, score_column)
    std = scores.std(ddof=0)

    if pd.isna(std) or std == 0:
        result["is_outlier"] = False
        return result

    zscores = (scores - scores.mean()).abs() / std
    result["is_outlier"] = (zscores > threshold).fillna(False)
    return result


def detect_outliers_iqr(
    df: pd.DataFrame,
    score_column: str = "alignment_score",
) -> pd.DataFrame:
    """Add an ``is_outlier`` column using the 1.5 IQR rule."""

    result = df.copy()
    scores = _score_series(result, score_column)
    q1 = scores.quantile(0.25)
    q3 = scores.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        result["is_outlier"] = False
        return result

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    result["is_outlier"] = (
        (scores < lower_bound) | (scores > upper_bound)
    ).fillna(False)
    return result
