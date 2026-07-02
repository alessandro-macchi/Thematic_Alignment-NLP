# Analyzing Thematic Alignment in Scientific Journals

This repository contains the NLP exam project **Analyzing Thematic Alignment in
Scientific Journals**. The project measures how closely the abstracts published
by a scientific journal align with the journal's stated Aims & Scope.

The current example uses PubMed records from the **World Journal of
Orthopedics**, but the same workflow can be reused for another PubMed-indexed
journal by changing the journal name and replacing the Aims & Scope text.

## Project Idea

The analysis treats the journal Aims & Scope as the reference text and each
article abstract as one document to compare against that reference. The pipeline
then computes:

- a semantic alignment score using SentenceTransformer embeddings and cosine
  similarity;
- descriptive statistics, yearly trends, top and least aligned articles, and
  score outliers;
- a BERTopic analysis to describe the main thematic clusters, their
  evolution over time, and the documents assigned to the noise cluster.

## Repository Structure

```text
nlp/
  data/
    raw/
      articles_pubmed.csv      # PubMed article metadata and abstracts
      aims_scope.txt           # Manually collected journal Aims & Scope text
    results/
      alignment_scores.csv     # Main article-level output
  notebooks/
    demo_analysis.ipynb        # Demonstration notebook using the package
  reports/
    figures/                   # Saved plots
    tables/                    # Saved summary CSV files
  scripts/
    fetch_pubmed_articles.py   # Download PubMed metadata for one journal
    run_pipeline.py            # Run the full local analysis pipeline
  src/
    journal_alignment/         # Project code
  requirements.txt
  README.md
  report.pdf                   
  presentation.pdf             
```

## Installation

Run all commands from the repository root:

```bash
pip install -r requirements.txt
```

The first run of the pipeline may download the selected SentenceTransformer
model. BERTopic and its clustering dependencies are installed through
`requirements.txt`. The default embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

## Data Inputs

The pipeline needs two local input files:

```text
data/raw/articles_pubmed.csv
data/raw/aims_scope.txt
```

The article CSV must contain at least these columns:

```text
title
abstract
year
```

The fetch script also stores useful metadata columns when available:

```text
pmid, journal, doi, publication_date, authors, keywords
```

The Aims & Scope file must be a plain text file copied manually from the
journal website or another official source.

## Small-Step Workflow

### 1. Fetch PubMed Articles

The stored results in this repository use articles dated **2015-2025**. To
reproduce that same time window, run:

```bash
python scripts/fetch_pubmed_articles.py --journal-name "World Journal of Orthopedics" --output-path data/raw/articles_pubmed.csv --start-year 2015 --end-year 2025 --max-results 5000
```

If no years are passed, the fetch script searches from the current calendar year
back 10 years. For example, if run in 2026, the default search window is
2016-2026.

```bash
python scripts/fetch_pubmed_articles.py --journal-name "World Journal of Orthopedics" --output-path data/raw/articles_pubmed.csv --max-results 5000
```

### 2. Add The Aims & Scope Text

Copy the journal Aims & Scope into:

```text
data/raw/aims_scope.txt
```

Keep it as plain text. No special formatting is required.

### 3. Run The Pipeline

After both input files exist, run:

```bash
python scripts/run_pipeline.py --articles-path data/raw/articles_pubmed.csv --aims-scope-path data/raw/aims_scope.txt
```

Optional arguments:

```bash
python scripts/run_pipeline.py --articles-path data/raw/articles_pubmed.csv --aims-scope-path data/raw/aims_scope.txt --model-name sentence-transformers/all-MiniLM-L6-v2 --top-n 10 --outlier-method zscore --z-threshold 2.0
```

To skip the optional BERTopic analysis and produce only the alignment outputs:

```bash
python scripts/run_pipeline.py --articles-path data/raw/articles_pubmed.csv --aims-scope-path data/raw/aims_scope.txt --skip-topic-model
```

### 4. Inspect The Results

Open the generated CSV files and figures in `data/results/` and `reports/`.
The notebook `notebooks/demo_analysis.ipynb` provides a guided demonstration,
but the implementation logic stays inside `src/journal_alignment/`.

## Generated Outputs

Main article-level file:

```text
data/results/alignment_scores.csv
```

Summary tables:

```text
reports/tables/summary_statistics.csv
reports/tables/yearly_alignment.csv
reports/tables/top_aligned_articles.csv
reports/tables/least_aligned_articles.csv
reports/tables/outlier_articles.csv
reports/tables/chunk_count_report.csv
reports/tables/bertopic_topic_summary.csv
reports/tables/bertopic_topic_diagnostics.csv
reports/tables/bertopic_noise_diagnostics.csv
reports/tables/bertopic_topics_over_time.csv
reports/tables/outlier_topic_assignments.csv
```

Figures:

```text
reports/figures/alignment_histogram.png
reports/figures/alignment_boxplot.png
reports/figures/alignment_by_year.png
reports/figures/article_scores_by_year.png
reports/figures/bertopic_topics_over_time.png
```

## Current Run Summary

The current stored outputs analyze 1100 validated article abstracts from the
World Journal of Orthopedics between 2015 and 2025.

- Mean semantic alignment score: 0.343.
- Median semantic alignment score: 0.342.
- Minimum and maximum semantic alignment scores: 0.035 and 0.747.
- BERTopic finds 9 non-noise topics plus one noise cluster.
- The largest non-noise topics are fractures/hip/femoral articles, knee/TKA
  articles, and spinal/lumbar pain articles.

## How To Read The Scores

`alignment_score` is the cosine similarity between an article abstract embedding
and the Aims & Scope embedding. Higher values indicate stronger semantic
similarity with the journal's stated thematic scope.

`is_outlier` marks unusually high or low semantic alignment scores according to
the configured outlier rule. The default rule is based on z-scores with a
threshold of 2.0.

The yearly table can be used to discuss whether the journal's published
articles become more or less aligned with its stated scope over time.

The BERTopic outputs should be interpreted as a complementary description of
corpus structure. Topic IDs describe groups of abstracts with similar vocabulary
and embeddings; they are not manual journal categories.

The BERTopic noise cluster is treated as a diagnostic group, not as irrelevant
content. In the current run, 267 out of 1100 articles are assigned to noise
(24.3%). Their abstracts are not shorter than clustered articles: the mean
abstract length is 247 words for noise articles versus 241 words for clustered
articles. This suggests the noise group is driven more by heterogeneous or
borderline themes than by very short abstracts.

## Main Report Artifacts

- `notebooks/demo_analysis.ipynb`: step-by-step notebook for inspecting the
  generated tables and figures.

## Main Modules

- `data.py`: PubMed fetching, CSV loading, Aims & Scope loading, and input
  validation.
- `preprocessing.py`: conservative whitespace normalization and text cleaning.
- `embeddings.py`: SentenceTransformer wrapper.
- `metrics.py`: cosine similarity, TF-IDF similarity, summaries, yearly
  aggregation, and outlier detection.
- `analysis.py`: article ranking and summary table construction.
- `topic_model.py`: BERTopic summaries, topic timing, outlier-topic
  assignments, and noise-cluster diagnostics.
- `pipeline.py`: end-to-end orchestration, output saving, and figure generation.
