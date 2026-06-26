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
- a TF-IDF cosine similarity score as a simpler baseline;
- descriptive statistics, yearly trends, top and least aligned articles, and
  score outliers.

This follows the NLP course material structure: text preprocessing, vector
representations, cosine similarity, and descriptive analysis of document-level
scores.

## Repository Structure

```text
nlp/
  course_material/             # NLP lecture notebooks and project instructions
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
    journal_alignment/         # Reusable project code
  requirements.txt
  README.md
```

## Installation

Run all commands from the repository root:

```bash
pip install -r requirements.txt
```

The first run of the pipeline may download the selected SentenceTransformer
model. The default model is:

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

To reproduce the current data collection step:

```bash
python scripts/fetch_pubmed_articles.py --journal-name "World Journal of Orthopedics" --output-path data/raw/articles_pubmed.csv --max-results 5000
```

By default, the fetch script searches the last 10 years up to the current year.
For a fixed time window, pass explicit years:

```bash
python scripts/fetch_pubmed_articles.py --journal-name "World Journal of Orthopedics" --output-path data/raw/articles_pubmed.csv --start-year 2016 --end-year 2026 --max-results 5000
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
```

Figures:

```text
reports/figures/alignment_histogram.png
reports/figures/alignment_boxplot.png
reports/figures/alignment_by_year.png
reports/figures/article_scores_by_year.png
```

## How To Read The Scores

`alignment_score` is the cosine similarity between an article abstract embedding
and the Aims & Scope embedding. Higher values indicate stronger semantic
similarity with the journal's stated thematic scope.

`tfidf_alignment_score` is a baseline score based on TF-IDF vectors. It is less
contextual than the embedding score, but it is useful as a simple comparison.

The yearly table can be used to discuss whether the journal's published
articles become more or less aligned with its stated scope over time.

## Main Modules

- `data.py`: PubMed fetching, CSV loading, Aims & Scope loading, and input
  validation.
- `preprocessing.py`: conservative whitespace normalization and text cleaning.
- `embeddings.py`: SentenceTransformer wrapper.
- `metrics.py`: cosine similarity, TF-IDF similarity, summaries, yearly
  aggregation, and outlier detection.
- `analysis.py`: article ranking and summary table construction.
- `pipeline.py`: end-to-end orchestration, output saving, and figure generation.
