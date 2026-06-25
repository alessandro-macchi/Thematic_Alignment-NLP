"""Analysis utilities for alignment scores and thematic drift."""

from __future__ import annotations

import pandas as pd

from journal_alignment.embeddings import EmbeddingModel
from journal_alignment.metrics import (
    aggregate_by_year,
    compute_alignment_scores,
    compute_tfidf_similarity,
    detect_outliers_iqr,
    detect_outliers_zscore,
    summarize_scores,
)


class AlignmentAnalyzer:
    """Coordinate embedding models and metric functions for article analysis."""

    TEXT_COLUMN = "abstract"

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model

    def compute_embedding_alignment(
        self,
        df: pd.DataFrame,
        aims_scope: str,
    ) -> pd.DataFrame:
        """Add semantic alignment scores against the journal Aims & Scope text."""

        texts = self._article_texts(df)
        abstract_embeddings = self.embedding_model.encode_texts(texts)
        aims_scope_embedding = self.embedding_model.encode_single(aims_scope)
        scores = compute_alignment_scores(abstract_embeddings, aims_scope_embedding)

        result = df.copy()
        result["alignment_score"] = scores
        return result

    def compute_tfidf_baseline(
        self,
        df: pd.DataFrame,
        aims_scope: str,
    ) -> pd.DataFrame:
        """Add TF-IDF cosine similarity scores as a simple baseline."""

        scores = compute_tfidf_similarity(
            texts=self._article_texts(df),
            reference_text=aims_scope,
        )

        result = df.copy()
        result["tfidf_alignment_score"] = scores
        return result

    def get_top_articles(
        self,
        df: pd.DataFrame,
        score_column: str = "alignment_score",
        n: int = 5,
    ) -> pd.DataFrame:
        """Return the top ``n`` rows sorted by descending alignment score."""

        self._validate_score_request(df, score_column, n)
        return (
            df.sort_values(score_column, ascending=False)
            .head(n)
            .reset_index(drop=True)
            .copy()
        )

    def get_bottom_articles(
        self,
        df: pd.DataFrame,
        score_column: str = "alignment_score",
        n: int = 5,
    ) -> pd.DataFrame:
        """Return the bottom ``n`` rows sorted by ascending alignment score."""

        self._validate_score_request(df, score_column, n)
        return (
            df.sort_values(score_column, ascending=True)
            .head(n)
            .reset_index(drop=True)
            .copy()
        )

    def add_outlier_labels(
        self,
        df: pd.DataFrame,
        method: str = "zscore",
        score_column: str = "alignment_score",
        z_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """Add an ``is_outlier`` column using z-score or IQR detection."""

        normalized_method = method.strip().lower().replace("_", "").replace("-", "")
        if normalized_method == "zscore":
            return detect_outliers_zscore(
                df=df,
                score_column=score_column,
                threshold=z_threshold,
            )
        if normalized_method == "iqr":
            return detect_outliers_iqr(df=df, score_column=score_column)
        raise ValueError("method must be either 'zscore' or 'iqr'.")

    def build_summary_tables(
        self,
        df: pd.DataFrame,
        top_n: int = 5,
        outlier_method: str = "zscore",
        z_threshold: float = 2.0,
    ) -> dict[str, pd.DataFrame]:
        """Build reusable analysis DataFrames without saving them to disk."""

        labeled_df = self.add_outlier_labels(
            df,
            method=outlier_method,
            z_threshold=z_threshold,
        )
        outlier_articles = labeled_df[labeled_df["is_outlier"]].reset_index(drop=True)

        return {
            "alignment_scores": labeled_df.copy(),
            "summary_statistics": summarize_scores(labeled_df),
            "yearly_alignment": aggregate_by_year(labeled_df),
            "top_aligned_articles": self.get_top_articles(labeled_df, n=top_n),
            "least_aligned_articles": self.get_bottom_articles(labeled_df, n=top_n),
            "outlier_articles": outlier_articles.copy(),
        }

    def _article_texts(self, df: pd.DataFrame) -> list[str]:
        """Return valid article texts from the analysis text column."""

        if self.TEXT_COLUMN not in df.columns:
            raise ValueError(
                f"Column '{self.TEXT_COLUMN}' is missing from the DataFrame."
            )

        texts = df[self.TEXT_COLUMN].astype("string")
        if texts.isna().any():
            raise ValueError(
                f"Column '{self.TEXT_COLUMN}' must not contain missing values."
            )

        text_list = texts.tolist()
        if not text_list:
            raise ValueError("df must contain at least one article.")
        if any(not text.strip() for text in text_list):
            raise ValueError(
                f"Column '{self.TEXT_COLUMN}' must not contain empty strings."
            )

        return text_list

    @staticmethod
    def _validate_score_request(
        df: pd.DataFrame,
        score_column: str,
        n: int,
    ) -> None:
        """Validate arguments used for top and bottom article rankings."""

        if score_column not in df.columns:
            raise ValueError(f"Column '{score_column}' is missing from the DataFrame.")
        if n < 1:
            raise ValueError("n must be at least 1.")


__all__ = ["AlignmentAnalyzer"]
