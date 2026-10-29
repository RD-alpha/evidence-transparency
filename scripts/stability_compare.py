#!/usr/bin/env python3
"""Compare the repeat runs written by scripts/stability_runs.py.  No API calls.

For each design and aspect it reports, across every pair of runs:
- core overlap: the Jaccard overlap of the sets of sentences marked core
  (|A and B| / |A or B|), the measure that matters for scoring, since only
  the core sets the score;
- core share: the fraction of the aspect's sentences marked core;
- core units: how many core themes or core lines each run chose;
- grouping agreement (grouped design only): of the sentence pairs one run
  puts in the same theme, the share the other run also puts together,
  averaged over both directions.

    python scripts/stability_compare.py --csv annotations/stability_runs.csv
"""

import argparse
import csv
import json
import re
import statistics
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "web_app"))

from analyse import ASPECTS  # noqa: E402
from rescore_weighted import COMPARISON_DOCS  # noqa: E402
from stability_runs import run_dir  # noqa: E402

# Aspects shorter than this are listed but left out of the medians: with a
# handful of sentences, overlap is close to all or nothing.
MIN_SENTENCES = 20


def aspect_ids(doc: str, aspect: str) -> set[str]:
    text = (PROJECT_ROOT / "results" / doc / aspect / "text.txt").read_text()
    return set(re.findall(r"^(S\d{4,})", text, flags=re.M))


def load_grouped(root: Path, aspect: str, ids: set[str]):
    """(core sentence set, number of core themes, theme label per sentence)"""
    classification = root / aspect / "classification.json"
    weights = root / aspect / "weights.json"
    if not (classification.exists() and weights.exists()):
        return None
    themes = json.loads(classification.read_text())["themes"]
    weight = {}
    for entry in json.loads(weights.read_text())["themes"]:
        weight.setdefault(entry["theme"], []).append(entry["weight"])
    core, units, labels = set(), 0, {}
    for k, theme in enumerate(themes):
        members = [sid for sid in theme["ids"] if sid in ids]
        is_core = weight.get(theme["name"], ["not core"]).pop(0) == "core"
        if is_core and members:
            core.update(members)
            units += 1
        for sid in members:
            labels.setdefault(sid, k)
    return core, units, labels


def load_core(root: Path, aspect: str, ids: set[str]):
    path = root / aspect / "core.json"
    if not path.exists():
        return None
    lines = json.loads(path.read_text())["core_lines"]
    core = {sid for line in lines for sid in line["ids"] if sid in ids}
    return core, len([line for line in lines if line["ids"]]), None


def jaccard(a: set, b: set) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def pair_agreement(a: dict, b: dict) -> float | None:
    """Share of the pairs together in a that are also together in b."""
    shared = [sid for sid in a if sid in b]
    together = both = 0
    for x, y in combinations(shared, 2):
        if a[x] == a[y]:
            together += 1
            both += b[x] == b[y]
    return both / together if together else None


def compare(design: str, runs: list[int]):
    loader = load_core if design == "core" else load_grouped
    rows = []
    for doc in COMPARISON_DOCS:
        for aspect in ASPECTS:
            ids = aspect_ids(doc, aspect)
            if not ids:
                continue
            results = [loader(run_dir(doc, design, run), aspect, ids) for run in runs]
            if any(r is None for r in results):
                continue
            overlaps = [jaccard(a[0], b[0]) for a, b in combinations(results, 2)]
            row = {
                "design": design, "doc": doc, "aspect": aspect, "sentences": len(ids),
                "core_overlap": statistics.mean(overlaps),
                "core_overlap_min": min(overlaps),
                "core_share": "/".join(f"{len(r[0]) / len(ids):.0%}" for r in results),
                "core_units": "/".join(str(r[1]) for r in results),
                "grouping_agreement": "",
            }
            if design != "core":
                agreements = []
                for a, b in combinations(results, 2):
                    for x, y in ((a[2], b[2]), (b[2], a[2])):
                        value = pair_agreement(x, y)
                        if value is not None:
                            agreements.append(value)
                if agreements:
                    row["grouping_agreement"] = statistics.mean(agreements)
            rows.append(row)
    return rows


def print_scores(runs: list[int]):
    """Section scores across runs, for each design whose runs were scored."""
    for design in ("grouped", "find_core"):
        rows = []
        for doc in COMPARISON_DOCS:
            for aspect in ASPECTS:
                paths = [run_dir(doc, design, run) / aspect / "scoring.json" for run in runs]
                if all(p.exists() for p in paths):
                    results = [json.loads(p.read_text()) for p in paths]
                    rows.append((doc, aspect, [r["score"] for r in results], [r.get("core_mean") for r in results]))
        if not rows:
            continue
        same = sum(len(set(scores)) == 1 for _, _, scores, _ in rows)
        within_one = sum(max(scores) - min(scores) <= 1 for _, _, scores, _ in rows)
        print(f"\nSection scores, {design} ({len(rows)} aspects scored in every run)")
        for doc, aspect, scores, means in rows:
            mean_text = "/".join("-" if m is None else f"{m:.2f}" for m in means)
            print(f"{doc + '/' + aspect:45s} scores {'/'.join(map(str, scores)):7s} core means {mean_text}")
        print(f"same score in every run: {same} of {len(rows)}; within one level: {within_one} of {len(rows)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--csv", help="also write the per-aspect table to this file")
    args = parser.parse_args()

    all_rows = []
    for design in ("grouped", "core", "find_core"):
        rows = compare(design, args.runs)
        all_rows.extend(rows)
        if not rows:
            print(f"{design}: no complete set of runs yet")
            continue
        print(f"\n{design} ({len(rows)} aspects with every run complete)")
        print(f"{'document/aspect':45s} {'n':>4s} {'core overlap':>12s} {'min':>5s} {'core share':>14s} {'units':>8s} {'grouping':>9s}")
        for r in rows:
            grouping = f"{r['grouping_agreement']:.0%}" if r["grouping_agreement"] != "" else ""
            print(f"{r['doc'] + '/' + r['aspect']:45s} {r['sentences']:4d} {r['core_overlap']:12.0%} "
                  f"{r['core_overlap_min']:5.0%} {r['core_share']:>14s} {r['core_units']:>8s} {grouping:>9s}")
        big = [r for r in rows if r["sentences"] >= MIN_SENTENCES]
        if big:
            print(f"median over the {len(big)} aspects with {MIN_SENTENCES}+ sentences: "
                  f"core overlap {statistics.median(r['core_overlap'] for r in big):.0%}", end="")
            grouped = [r["grouping_agreement"] for r in big if r["grouping_agreement"] != ""]
            print(f", grouping agreement {statistics.median(grouped):.0%}" if grouped else "")

    print_scores(args.runs)

    if args.csv and all_rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
