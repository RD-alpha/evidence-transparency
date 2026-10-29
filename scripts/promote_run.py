#!/usr/bin/env python3
"""Copy one stability run of the core-theme design into the live results
folders, so the viewer and report display it.  No LLM calls.

    python scripts/promote_run.py --run 1
    python scripts/promote_run.py --run 1 --docs lobster_ia prison_reform

For each document, from results/<doc>/stability/find_core_run<N>/:
  - core_themes.json goes to results/<doc>/;
  - each of Diagnosis, Proposal and Implementation gets that run's
    classification.json, weights.json, summary.xml and scoring.json;
  - Testing, which the experiment runs score as one block, gets the files the
    live pipeline would write: a single-theme classification.json, a rebuilt
    summary.xml, and the run's scoring.json;
  - an aspect with no sentences gets an empty classification, an empty
    summary.xml and a score of 0 with the standard reasoning.

Whatever the live folders held is first copied to
results/<doc>/pre_promote_<date>/<aspect>/ (only on the first promotion of the
day, so a rerun never overwrites the backup).  results/<doc>/promoted_from.txt
records which run is live.
"""

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "web_app"))

import analyse  # noqa: E402
from preprocess import construct_xml  # noqa: E402
from rescore_weighted import COMPARISON_DOCS  # noqa: E402

PER_ASPECT_FILES = ("classification.json", "weights.json", "summary.xml", "scoring.json")


def backup(doc_dir: Path, tag: str) -> Path:
    dest = doc_dir / f"pre_promote_{tag}"
    if dest.exists():
        return dest
    for aspect in analyse.ASPECTS:
        for name in PER_ASPECT_FILES:
            src = doc_dir / aspect / name
            if src.exists():
                (dest / aspect).mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / aspect / name)
    if (doc_dir / "core_themes.json").exists():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(doc_dir / "core_themes.json", dest / "core_themes.json")
    return dest


def promote(doc: str, run: int, tag: str) -> list[str]:
    analyse.set_policy_name(doc)
    doc_dir = PROJECT_ROOT / "results" / doc
    src_root = doc_dir / "stability" / f"find_core_run{run}"
    if not (src_root / "core_themes.json").exists():
        return [f"{doc}: no find_core_run{run}, skipped"]
    backup(doc_dir, tag)
    lines = []

    shutil.copy2(src_root / "core_themes.json", doc_dir / "core_themes.json")

    for aspect in analyse.ASPECTS:
        live = doc_dir / aspect
        has_sentences = bool(analyse._aspect_sentence_ids(aspect))
        scoring_src = src_root / aspect / "scoring.json"

        if not has_sentences:
            (live / "weights.json").unlink(missing_ok=True)
            construct_xml(analyse.DOCUMENTS_DIR, analyse.RESULTS_DIR, aspect, classification={"themes": []})
            (live / "classification.json").write_text(json.dumps({"themes": []}, indent=2))
            analyse._write_empty_score(aspect)
            lines.append(f"{aspect}: no sentences, scored 0")
        elif aspect in analyse.BLOCK_ASPECTS:
            (live / "weights.json").unlink(missing_ok=True)
            analyse._write_single_theme(aspect)
            construct_xml(analyse.DOCUMENTS_DIR, analyse.RESULTS_DIR, aspect)
            shutil.copy2(scoring_src, live / "scoring.json")
            lines.append(f"{aspect}: block, scored {json.loads(scoring_src.read_text())['score']}")
        else:
            for name in PER_ASPECT_FILES:
                shutil.copy2(src_root / aspect / name, live / name)
            lines.append(f"{aspect}: scored {json.loads(scoring_src.read_text())['score']}")

    (doc_dir / "promoted_from.txt").write_text(
        f"results/{doc}/stability/find_core_run{run}, promoted {date.today().isoformat()}\n")
    return [f"{doc}:"] + [f"  {line}" for line in lines]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--docs", nargs="+", default=COMPARISON_DOCS)
    args = parser.parse_args()
    tag = date.today().isoformat()
    for doc in args.docs:
        print("\n".join(promote(doc, args.run, tag)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
