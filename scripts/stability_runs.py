#!/usr/bin/env python3
"""Repeat runs that measure how stable the choice of core themes is, for two
designs, without touching the live results:

    grouped  theme grouping with claude-sonnet-5-5 (prompts/classify.txt), then
             the current weighting call (prompts/weight_themes.txt)
    core     one evidence-blind call that names the core lines of argument
             directly (prompts/weight_core.txt), with no grouping step
    find_core  one call per document that reads the whole text and names each
             aspect's core themes (prompts/find_core.txt); no sentence codes

Each run writes to results/<doc>/stability/<design>_run<N>/, laid out like a
results directory (it holds copies of intervention.txt, problem.txt and each
aspect's text.txt), so the usual analyse functions can be pointed at it.
Nothing is scored.  Compare the runs with scripts/stability_compare.py.

    python scripts/stability_runs.py grouped --runs 1 2 3 --dry-run
    python scripts/stability_runs.py core --runs 1 2 3

Every call goes to the Anthropic API (claude-sonnet-5-5) and costs credits;
--dry-run prints a token estimate without calling anything.  An aspect whose
output already exists in a run is skipped, so a failed run can be restarted
without paying twice.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "web_app"))

import analyse  # noqa: E402
from llm_monitor import write_run_summary  # noqa: E402
from preprocess import construct_xml  # noqa: E402
from rescore_weighted import COMPARISON_DOCS  # noqa: E402

GROUPING_MODEL = "claude-sonnet-5-5"


def run_dir(doc: str, design: str, run: int) -> Path:
    return PROJECT_ROOT / "results" / doc / "stability" / f"{design}_run{run}"


def prepare(doc: str, design: str, run: int) -> Path:
    """Create the run directory and point analyse at it."""
    live = PROJECT_ROOT / "results" / doc
    root = run_dir(doc, design, run)
    for name in ("intervention.txt", "problem.txt"):
        if (live / name).exists() and not (root / name).exists():
            root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(live / name, root / name)
    for aspect in analyse.ASPECTS:
        (root / aspect).mkdir(parents=True, exist_ok=True)
        if not (root / aspect / "text.txt").exists():
            shutil.copy2(live / aspect / "text.txt", root / aspect / "text.txt")
    analyse.RESULTS_DIR = root
    return root


def has_sentences(aspect: str) -> bool:
    return analyse.result_file(aspect, "text.txt").read_text().strip() != ""


def estimate(design: str, aspect: str) -> int:
    """Rough input tokens for the calls this aspect would need (chars / 4)."""
    text = analyse.result_file(aspect, "text.txt").read_text()
    if design == "core":
        return (len(text) + len(analyse._get_prompt("weight_core"))) // 4
    # Grouping reads the text; weighting reads it again, regrouped.
    return (2 * len(text) + len(analyse._get_prompt("classify")) + len(analyse._get_prompt("weight_themes"))) // 4


def run_grouped(aspect: str, dry_run: bool):
    if analyse.result_file(aspect, "weights.json").exists():
        return "skipped (done)", []
    if dry_run:
        return f"would call twice, ~{estimate('grouped', aspect) // 1000}k input tokens", []
    records = []
    # A grouping kept from a run that failed at weighting is reused, not paid
    # for again.
    if not analyse.result_file(aspect, "classification.json").exists():
        records.append(analyse.call_llm_classify_themes(aspect, model=GROUPING_MODEL))
    construct_xml(analyse.DOCUMENTS_DIR, analyse.RESULTS_DIR, aspect)
    records.append(analyse.call_llm_weight_themes(aspect))
    return "grouped and weighted", records


def run_core(aspect: str, dry_run: bool):
    if analyse.result_file(aspect, "core.json").exists():
        return "skipped (done)", []
    if dry_run:
        return f"would call, ~{estimate('core', aspect) // 1000}k input tokens", []
    return "core lines chosen", [analyse.call_llm_core_lines(aspect)]


def run_find_core(doc: str, dry_run: bool):
    """Whole-document call: done once per document, not per aspect."""
    if (analyse.RESULTS_DIR / "core_themes.json").exists():
        return "skipped (done)", [], 0
    tokens = (len(analyse._labelled_document()) + len(analyse._get_prompt("find_core")) + 3000) // 4
    if dry_run:
        return f"would call, ~{tokens // 1000}k input tokens", [], tokens
    return "core themes named", [analyse.call_llm_find_core()], tokens


def run_find_core_grouping(doc: str, dry_run: bool):
    """Group each core aspect on Haiku with this run's find_core themes."""
    core = json.loads((analyse.RESULTS_DIR / "core_themes.json").read_text())
    records, lines = [], []
    for aspect in analyse.CORE_ASPECTS:
        themes = core[aspect]["core_themes"]
        if not has_sentences(aspect) or not themes:
            lines.append(f"{aspect}: skipped (no sentences or no core themes)")
            continue
        if analyse.result_file(aspect, "weights.json").exists():
            lines.append(f"{aspect}: skipped (done)")
            continue
        if dry_run:
            text = analyse.result_file(aspect, "text.txt").read_text()
            lines.append(f"{aspect}: would call, ~{(len(text) + 4000) // 4 // 1000}k input tokens")
            continue
        record, problems = analyse.call_llm_classify_with_core(aspect, themes)
        records.append(record)
        coverage = construct_xml(analyse.DOCUMENTS_DIR, analyse.RESULTS_DIR, aspect)
        lines.append(f"{aspect}: grouped, {coverage['ungrouped']} of {coverage['total']} ungrouped"
                     + (f"; PROBLEMS: {problems}" if problems else ""))
    return lines, records


def run_find_core_score(doc: str, dry_run: bool):
    """Score each core aspect grouped by find_core_grouping, with the core
    claims folded into summary.xml."""
    records, lines = [], []
    for aspect in analyse.CORE_ASPECTS:
        if not analyse.result_file(aspect, "weights.json").exists():
            lines.append(f"{aspect}: skipped (not grouped)")
            continue
        if analyse.result_file(aspect, "scoring.json").exists():
            lines.append(f"{aspect}: skipped (done)")
            continue
        construct_xml(analyse.DOCUMENTS_DIR, analyse.RESULTS_DIR, aspect)
        if dry_run:
            size = len(analyse._read_summary_xml(aspect)) + len(analyse._get_prompt(f"score_{aspect}")) + len(analyse._get_prompt("score_common"))
            lines.append(f"{aspect}: would call, ~{size // 4 // 1000}k input tokens")
            continue
        records.append(analyse.call_llm_score(aspect))
        result = json.loads(analyse.result_file(aspect, "scoring.json").read_text())
        lines.append(f"{aspect}: scored {result['score']} (core mean {result['core_mean']})")
    # Testing is scored as one block, with no grouping or weights.
    for aspect in analyse.BLOCK_ASPECTS:
        if analyse.result_file(aspect, "scoring.json").exists():
            lines.append(f"{aspect}: skipped (done)")
            continue
        block = analyse._block_input(aspect)
        if dry_run:
            size = (len(block) if block else 0) + len(analyse._get_prompt(f"score_{aspect}")) + len(analyse._get_prompt("score_block"))
            lines.append(f"{aspect}: " + (f"would call, ~{size // 4 // 1000}k input tokens" if block else "would score 0 (no sentences), no call"))
            continue
        record = analyse.call_llm_score(aspect)
        if record is not None:
            records.append(record)
        result = json.loads(analyse.result_file(aspect, "scoring.json").read_text())
        lines.append(f"{aspect}: scored {result['score']} (block" + (")" if record else ", no sentences, no call)"))
    return lines, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("design", choices=["grouped", "core", "find_core", "find_core_grouping", "find_core_score"])
    parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--docs", nargs="+", default=COMPARISON_DOCS)
    parser.add_argument("--dry-run", action="store_true", help="estimate size; no API calls")
    args = parser.parse_args()

    if not args.dry_run:
        analyse.get_client()
        analyse._set_system_prompt()

    step = run_grouped if args.design == "grouped" else run_core
    calls = tokens = 0
    for run in args.runs:
        for doc in args.docs:
            analyse.set_policy_name(doc)
            prepare(doc, "find_core" if args.design.startswith("find_core") else args.design, run)
            records = []
            if args.design in ("find_core_grouping", "find_core_score"):
                step_fn = run_find_core_grouping if args.design == "find_core_grouping" else run_find_core_score
                lines, records = step_fn(doc, args.dry_run)
                for line in lines:
                    print(f"run {run} {doc}/{line}", flush=True)
                if records:
                    write_run_summary(doc, records)
                continue
            if args.design == "find_core":
                outcome, records, size = run_find_core(doc, args.dry_run)
                print(f"run {run} {doc}: {outcome}", flush=True)
                if outcome.startswith("would call"):
                    calls += 1
                    tokens += size
                if records:
                    write_run_summary(doc, records)
                continue
            for aspect in analyse.ASPECTS:
                if not has_sentences(aspect):
                    continue
                outcome, recs = step(aspect, args.dry_run)
                print(f"run {run} {doc}/{aspect}: {outcome}", flush=True)
                if outcome.startswith("would call"):
                    calls += 2 if args.design == "grouped" else 1
                    tokens += estimate(args.design, aspect)
                records.extend(recs)
            if records:
                write_run_summary(doc, records)
    if args.dry_run:
        print(f"\n{calls} calls would be made, ~{tokens // 1000}k input tokens in total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
