#!/usr/bin/env python3
"""Regroup, weight and rescore existing results, without redoing sentence
extraction (see annotations/proposed_scoring_changes.md, Option B).

run_analysis in web_app/analyse.py regenerates everything from sentence
extraction onwards, which would change the themes and break the before/after
comparison.  This script reuses each aspect's classification.json and runs only
the two later steps, in separate invocations so the weights can be reviewed
(and corrected by hand) before any scoring is paid for:

    python scripts/rescore_weighted.py weight --dry-run
    python scripts/rescore_weighted.py weight
    python scripts/rescore_weighted.py score --dry-run
    python scripts/rescore_weighted.py score

Both steps call the Anthropic API (claude-sonnet-5-5) and cost credits; --dry-run
renders every prompt and prints a token estimate without calling anything.
An aspect that is already done is skipped, so a run that fails partway can be
restarted without paying twice: weighting skips aspects that have weights.json,
and scoring skips aspects whose scoring.json is newer than weights.json.
--force redoes them.  Before an aspect is first rescored, its scoring.json is
copied to scoring_<BACKUP_TAG>_before.json.  --keep-as TAG also copies the
weights.json or scoring.json about to be replaced to weights_TAG.json /
scoring_TAG.json, so one round's outputs survive the next:

    python scripts/rescore_weighted.py weight --force --keep-as round1

The classify step regroups each aspect's sentences into themes with the
current prompts/classify.txt (claude-haiku-5-5).  It needs --keep-as: the old
classification.json is kept as classification_TAG.json and, since weights are
matched to themes by name, the old weights.json is moved to weights_TAG.json so
the stale weights are never used.  An aspect whose classification_TAG.json
already exists is skipped.  Run weight and score afterwards:

    python scripts/rescore_weighted.py classify --keep-as round2 --dry-run
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from string import Template

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "web_app"))

import analyse  # noqa: E402
from llm_monitor import write_run_summary  # noqa: E402
from preprocess import construct_xml  # noqa: E402

# The documents in annotations/2018_report_comparison.ipynb.
COMPARISON_DOCS = [
    "civil_service_apprenticeship", "furniture_fire_safety", "modern_languages_a_level",
    "lobster_ia", "letting_agent_fees", "funeral_expenses", "prison_reform",
    "overseas_votes", "company_car_tax", "night_flight_combined",
]
BACKUP_TAG = "2026-10-01"


def rebuild_xml(aspect: str) -> None:
    construct_xml(analyse.DOCUMENTS_DIR, analyse.RESULTS_DIR, aspect)


def estimate_tokens(step: str, aspect: str) -> int:
    """Rough input size of the call that would be made (characters / 4)."""
    xml = analyse._read_summary_xml(aspect)
    intervention, problem = analyse._anchor_texts()
    should_explain = analyse._get_prompt(f"explain_{aspect}").rstrip("\n")
    if step == "weight":
        xml = analyse._weighting_input(xml)
        prompt = Template(analyse._get_prompt("weight_themes")).substitute(
            intervention=intervention, problem=problem,
            aspect=analyse.ASPECT_DESCRIPTIONS[aspect], should_explain=should_explain)
    else:
        prompt = Template(analyse._get_prompt(f"score_{aspect}") + "\n" + analyse._get_prompt("score_common")).substitute(
            intervention=intervention, problem=problem, should_explain=should_explain)
    return (len(xml) + len(prompt)) // 4


def keep_copy(path: Path, keep_as: str | None) -> None:
    """Copy an output about to be replaced to <stem>_<keep_as>.json, once."""
    if keep_as and path.exists():
        kept = path.with_name(f"{path.stem}_{keep_as}{path.suffix}")
        if not kept.exists():
            shutil.copy2(path, kept)


def _theme_count(aspect: str) -> int:
    with open(analyse.result_file(aspect, "classification.json")) as f:
        return len(json.load(f)["themes"])


def classify_aspect(aspect: str, dry_run: bool, force: bool, keep_as: str | None):
    if not keep_as:
        raise RuntimeError("classify needs --keep-as TAG, so the old grouping is kept")
    classification = analyse.result_file(aspect, "classification.json")
    kept = analyse.result_file(aspect, f"classification_{keep_as}.json")
    if kept.exists() and not force:
        return f"skipped (classification_{keep_as}.json exists)", None
    text = analyse.result_file(aspect, "text.txt").read_text()
    if dry_run:
        tokens = (len(text) + len(analyse._get_prompt("classify"))) // 4
        return f"would call, ~{tokens // 1000}k input tokens", None
    before = _theme_count(aspect)
    keep_copy(classification, keep_as)
    weights = analyse.result_file(aspect, "weights.json")
    if weights.exists():
        kept_weights = analyse.result_file(aspect, f"weights_{keep_as}.json")
        if kept_weights.exists():
            weights.unlink()
        else:
            weights.rename(kept_weights)
    record = analyse.call_llm_classify_themes(aspect)
    coverage = construct_xml(analyse.DOCUMENTS_DIR, analyse.RESULTS_DIR, aspect)
    return (f"regrouped: {before} -> {_theme_count(aspect)} themes, "
            f"{coverage['ungrouped']} of {coverage['total']} sentences ungrouped"), record


def weight_aspect(aspect: str, dry_run: bool, force: bool, keep_as: str | None):
    weights = analyse.result_file(aspect, "weights.json")
    if weights.exists() and not force:
        return "skipped (weights.json exists)", None
    if dry_run:
        # Built in memory only: the summary.xml on disk is left as it is.
        return f"would call, ~{estimate_tokens('weight', aspect) // 1000}k input tokens", None
    keep_copy(weights, keep_as)
    # Refresh summary.xml (sentence counts) before the weighting call reads it.
    weights.unlink(missing_ok=True)
    rebuild_xml(aspect)
    record = analyse.call_llm_weight_themes(aspect)
    rebuild_xml(aspect)
    return "weighted", record


def score_aspect(aspect: str, dry_run: bool, force: bool, keep_as: str | None):
    weights = analyse.result_file(aspect, "weights.json")
    scoring = analyse.result_file(aspect, "scoring.json")
    if not weights.exists():
        raise RuntimeError(f"{analyse.POLICY_NAME}/{aspect}: no weights.json; run the weight step first")
    if scoring.exists() and scoring.stat().st_mtime > weights.stat().st_mtime and not force:
        return "skipped (scoring.json newer than weights.json)", None
    if dry_run:
        # Scoring takes two turns (the compute_core_mean tool call, then the
        # answer); the second reads the document from cache.
        return (f"would call, ~{estimate_tokens('score', aspect) // 1000}k input tokens "
                "written to cache, then read back once"), None
    backup = analyse.result_file(aspect, f"scoring_{BACKUP_TAG}_before.json")
    if scoring.exists() and not backup.exists():
        shutil.copy2(scoring, backup)
    keep_copy(scoring, keep_as)
    # Folds in weights.json as it stands now, including any hand corrections.
    rebuild_xml(aspect)
    record = analyse.call_llm_score(aspect)
    return "scored", record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["classify", "weight", "score"])
    parser.add_argument("--docs", nargs="+", default=COMPARISON_DOCS, help="results/ directories to process")
    parser.add_argument("--aspects", nargs="+", default=list(analyse.ASPECTS), choices=analyse.ASPECTS)
    parser.add_argument("--dry-run", action="store_true", help="render prompts and estimate size; no API calls")
    parser.add_argument("--force", action="store_true", help="redo aspects that are already done")
    parser.add_argument("--keep-as", metavar="TAG", help="keep the outputs being replaced as <name>_TAG.json")
    args = parser.parse_args()

    if not args.dry_run:
        analyse.get_client()
        analyse._set_system_prompt()

    step = {"classify": classify_aspect, "weight": weight_aspect, "score": score_aspect}[args.step]
    total_calls = 0
    for doc in args.docs:
        analyse.set_policy_name(doc)
        records = []
        for aspect in args.aspects:
            outcome, record = step(aspect, args.dry_run, args.force, args.keep_as)
            print(f"{doc}/{aspect}: {outcome}", flush=True)
            if outcome.startswith("would call"):
                total_calls += 1
            if record is not None:
                records.append(record)
        if records:
            write_run_summary(doc, records)
    if args.dry_run:
        print(f"\n{total_calls} calls would be made.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
