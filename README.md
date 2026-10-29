# Evidence transparency in policy documents

An LLM pipeline that scores UK government policy documents against the
[Evidence Transparency Framework](https://senseaboutscience.org/transparency-of-evidence/transparency-assessments/),
and the data behind our report comparing it with the 2018 Evidence
Transparency Spot Check.

The report is in
[`annotations/spot_check_report/`](annotations/spot_check_report/Report%20on%20spot%20check%20(5).pdf).

## What the pipeline does

The pipeline takes a GOV.UK page or PDF and gives it four scores, one per
aspect of the framework: **diagnosis**, **proposal**, **implementation**,
and **testing and evaluation**. Each score runs from 0 (no evidence) to
3 (the strengths and weaknesses of the evidence are discussed).

1. **Preprocessing.** Extracts the text and its references, splits it into
   sentences and gives each sentence an id (`S0001`, `S0002`, …).
2. **Classification into aspects.** Claude Sonnet reads the whole document
   and assigns each sentence id to an aspect, or to none. Because the model
   returns ids rather than text, everything downstream uses the document's
   own wording.
3. **Classification into themes.** Sonnet identifies two to four core themes
   for diagnosis, proposal and implementation. Claude Haiku then assigns each
   sentence to a core theme or a supporting theme.
4. **Scoring.** Sonnet scores each aspect from its sentences, grouped by theme
   with their references, and explains its reasoning.

The prompts are in [`web_app/prompts/`](web_app/prompts/).

## Repository layout

| Path | Contents |
|---|---|
| `web_app/` | The pipeline and a Flask app for browsing results: `analyse.py` (LLM calls), `preprocess.py`, `pdf_text_extractor.py`, `web_parser.py`, `app.py` (viewer) |
| `tests/` | pytest tests for extraction and preprocessing |
| `scripts/` | Batch runs and the repeat runs used for the reliability analysis |
| `documents/<doc>/` | Each policy document as it was ingested: the source text, `input.txt` (one sentence per line, with ids) and `references.json` |
| `results/<doc>/` | Pipeline output for each document: the sentences assigned to each aspect, themes, scores and reasoning |
| `results/<doc>_rep1` … `_rep5` | Five complete reruns of the same document, used for the reliability figures |
| `annotations/spot_check_report/` | The report, its figures and `make_figures.py` |
| `annotations/spa_human_comparison/` | The State Pension age comparison against two human annotators |

Documents included: the 64 documents from the 2018 spot check that are still
available (listed in `annotations/scored_documents.csv`, with sources in
`annotations/2018_report_sources.csv`), plus the documents used for the
reliability runs.

## Reproducing the figures

Reproducing the figures doesn't need an API key: the scripts only read the
saved results.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd annotations/spot_check_report && python make_figures.py      # report figures
cd ../spa_human_comparison && python compare.py                # human comparison
```

## Running the pipeline

Running the pipeline calls the Anthropic API, which costs money. OCR of
scanned PDF pages uses Mistral.

```bash
cp .env.example .env          # then add your API keys
cd web_app && python app.py   # http://localhost:5001
```

The viewer at `/` browses existing results. Use `/upload-page` to ingest a new
document and `/analyse` to run the pipeline on it. Before using OCR on any
scanned pages, the upload asks you to approve them.

Run the tests with `pytest`. Development used Python 3.14.

**Known failure:** `TestNoReferenceGapsAcrossDocuments` currently fails. For
some of the included PDFs, reference extraction misses a footnote number or
reads one twice. Most of these are 2018 documents with irregular
footnote layouts.

## Licence

Code: MIT (see [LICENSE](LICENSE)).

The policy documents in `documents/` are UK Crown copyright, published under
the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
