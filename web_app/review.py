"""Review workflow: diffs the human annotations against Claude's extraction
(results/<policy>/<aspect>/lines.txt) for the current document, and lets the
reviewer accept Claude's category (overwriting the human decision) or reject
it (keeping the human decision) one sentence at a time.
"""

from pathlib import Path
import json
import time

from flask import Blueprint, jsonify, render_template, request

import analyse
from annotate import load_document_sentences, load_annotations, save_annotations

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

REVIEWS_DIR = BASE_DIR / "data" / "reviews"
REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

ASPECTS = ("diagnosis", "proposal", "implementation", "testing")
RULES_ONLY_VARIANT = "candidate_rules_only"

review_bp = Blueprint("review", __name__)


def load_claude_decisions(policy_name: str) -> dict:
    """Returns {code: category}, category one of ASPECTS or 'none'.

    Reads results/<policy>/<aspect>/lines.txt, the same artefacts the viewer
    reads (see CLAUDE.md) — never an in-memory LLM response.
    """
    results_dir = PROJECT_ROOT / "results" / policy_name
    decisions = {}
    for aspect in ASPECTS:
        path = results_dir / aspect / "lines.txt"
        if not path.exists():
            continue
        for code in path.read_text(encoding="utf-8").splitlines():
            code = code.strip()
            if code:
                decisions[code] = aspect
    return decisions


def load_candidate_decisions(policy_name: str, variant: str) -> dict:
    """Returns {code: category} for a candidate prompt variant, category one
    of ASPECTS or 'none'.

    Reads annotations/<policy>/<variant>_<aspect>_lines.txt, the ad-hoc
    comparison artefacts written by run_candidate_extract.py. Missing files
    (variant never run for this policy) yield an empty dict, so the caller
    just sees no decisions rather than an error.
    """
    annotations_dir = PROJECT_ROOT / "annotations" / policy_name
    decisions = {}
    for aspect in ASPECTS:
        path = annotations_dir / f"{variant}_{aspect}_lines.txt"
        if not path.exists():
            continue
        for code in path.read_text(encoding="utf-8").splitlines():
            code = code.strip()
            if code:
                decisions[code] = aspect
    return decisions


def reviews_file(policy_name: str) -> Path:
    policy_dir = REVIEWS_DIR / policy_name
    policy_dir.mkdir(parents=True, exist_ok=True)
    return policy_dir / "reviews.json"


def load_reviews(policy_name: str) -> dict:
    path = reviews_file(policy_name)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_reviews(policy_name: str, reviews: dict) -> None:
    reviews_file(policy_name).write_text(json.dumps(reviews, indent=2, ensure_ascii=False), encoding="utf-8")


@review_bp.route("/review")
def review_page():
    policy_name = analyse.POLICY_NAME
    sentences = load_document_sentences(policy_name)
    human_annotations = load_annotations(policy_name)
    claude_decisions = load_claude_decisions(policy_name)
    rules_only_decisions = load_candidate_decisions(policy_name, RULES_ONLY_VARIANT)
    reviews = load_reviews(policy_name)

    diffs = []
    for sentence in sentences:
        code = sentence["code"]
        human_record = human_annotations.get(code)
        if human_record is None:
            continue  # not yet annotated by the human, nothing to diff against
        human_decision = human_record["decision"]
        claude_decision = claude_decisions.get(code, "none")
        if human_decision == claude_decision:
            continue
        rules_only_decision = rules_only_decisions.get(code, "none") if rules_only_decisions else None
        diffs.append({
            "code": code,
            "text": sentence["text"],
            "human_decision": human_decision,
            "claude_decision": claude_decision,
            "rules_only_decision": rules_only_decision,
            "resolved": code in reviews,
        })

    n_resolved = sum(1 for d in diffs if d["resolved"])

    return render_template(
        "review.jinja",
        document_title=policy_name.replace("_", " ").title(),
        policy_name=policy_name,
        diffs=diffs,
        n_resolved=n_resolved,
        decisions=ASPECTS + ("none",),
    )


@review_bp.route("/review/resolve", methods=["POST"])
def review_resolve():
    data = request.get_json(silent=True) or {}
    code = str(data.get("code", "")).strip()
    action = str(data.get("action", "")).strip().lower()
    claude_decision = str(data.get("claude_decision", "")).strip().lower()

    if not code:
        return jsonify({"status": "error", "message": "code is required"}), 400
    if action not in ("accept", "reject"):
        return jsonify({"status": "error", "message": "action must be 'accept' or 'reject'"}), 400

    policy_name = analyse.POLICY_NAME
    annotations = load_annotations(policy_name)
    record = annotations.get(code)
    if record is None:
        return jsonify({"status": "error", "message": f"no human annotation for {code}"}), 404

    # original_decision is set once, from whatever the human had recorded
    # before the first review action on this sentence, and never touched
    # again — so it survives even if the sentence is later re-reviewed.
    if "original_decision" not in record:
        record["original_decision"] = record["decision"]
    human_decision_before = record["original_decision"]

    if action == "accept":
        record["decision"] = claude_decision
    record["reviewed"] = True
    record["updated_at"] = time.time()
    annotations[code] = record
    save_annotations(policy_name, annotations)

    reviews = load_reviews(policy_name)
    reviews[code] = {
        "code": code,
        "action": action,
        "human_decision_before": human_decision_before,
        "claude_decision": claude_decision,
        "resolved_at": time.time(),
    }
    save_reviews(policy_name, reviews)

    return jsonify({"status": "ok", "review": reviews[code], "annotation": annotations.get(code)})
