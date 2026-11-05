"""Repeat runs of the core-theme steps on a fixed set of sentences, without
touching the live results.  Three steps, run in this order:

    find_core  one call per document that reads the whole text and names each
               aspect's core themes (prompts/find_core.txt)
    group      groups each aspect's sentences under those themes
               (prompts/classify_with_core.txt)
    score      scores each aspect

Each run writes to results/<doc>/stability/find_core_run<N>/, laid out like a
results directory (it holds copies of intervention.txt, problem.txt and each
aspect's text.txt), so the usual analyse functions can be pointed at it.
Compare the runs with scripts/stability_compare.py.

    python scripts/stability_runs.py find_core --runs 1 2 3 --dry-run
    python scripts/stability_runs.py find_core --runs 1 2 3

Every call goes to the Anthropic API and costs credits; --dry-run prints a
token estimate without calling anything.  Work already done in a run is
skipped, so a failed run can be restarted without paying twice.
"""

#!/usr/bin/env python3
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

COMPARISON_DOCS = [
    "civil_service_apprenticeship", "furniture_fire_safety", "modern_languages_a_level",
    "lobster_ia", "letting_agent_fees", "prison_reform", "overseas_votes", "company_car_tax",
]


def run_dir(doc: str, run: int) -> Path:
    return PROJECT_ROOT / "results" / doc / "stability" / f"find_core_run{run}"


def prepare(doc: str, run: int) -> Path:
    """Create the run directory and point analyse at it."""
    live = PROJECT_ROOT / "results" / doc
    root = run_dir(doc, run)
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


def run_find_core(doc: str, dry_run: bool):
    """Whole-document call: done once per document, not per aspect."""
    if (analyse.RESULTS_DIR / "core_themes.json").exists():
        return ["skipped (done)"], []
    if dry_run:
        tokens = (len(analyse._labelled_document()) + len(analyse._get_prompt("find_core")) + 3000) // 4
        return [f"would call, ~{tokens // 1000}k input tokens"], []
    return ["core themes named"], [analyse.call_llm_find_core()]


def run_group(doc: str, dry_run: bool):
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


def run_score(doc: str, dry_run: bool):
    """Score each core aspect grouped by the group step, with the core claims
    folded into summary.xml."""
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
        record = analyse.call_llm_score_block(aspect)
        if record is not None:
            records.append(record)
        result = json.loads(analyse.result_file(aspect, "scoring.json").read_text())
        lines.append(f"{aspect}: scored {result['score']} (block" + (")" if record else ", no sentences, no call)"))
    return lines, records


STEPS = {"find_core": run_find_core, "group": run_group, "score": run_score}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=list(STEPS))
    parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--docs", nargs="+", default=COMPARISON_DOCS)
    parser.add_argument("--dry-run", action="store_true", help="estimate size; no API calls")
    args = parser.parse_args()

    if not args.dry_run:
        analyse.get_client()
        analyse._set_system_prompt()

    for run in args.runs:
        for doc in args.docs:
            analyse.set_policy_name(doc)
            prepare(doc, run)
            lines, records = STEPS[args.step](doc, args.dry_run)
            for line in lines:
                print(f"run {run} {doc}: {line}", flush=True)
            if records:
                write_run_summary(doc, records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
