"""Sentence-by-sentence annotation workflow.

Presents each coded sentence of the current document in its surrounding
context, lets the annotator classify it into one of the policy-cycle
aspects (or "none"), and persists the decision keyed by sentence code.
"""

from pathlib import Path
import json
import re
import time

from flask import Blueprint, jsonify, render_template, request

import analyse

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

DECISIONS = ("diagnosis", "proposal", "implementation", "testing", "none")

annotate_bp = Blueprint("annotate", __name__)


def load_document_sentences(policy_name: str) -> list[dict]:
    """Loads every coded sentence (e.g. 'S001 ...') from the document's input.txt."""
    doc_path = PROJECT_ROOT / "documents" / policy_name / "input.txt"
    if not doc_path.exists():
        raise FileNotFoundError(f"Source sentences file not found: {doc_path}")

    sentences = []
    with open(doc_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            match = re.match(r'^(S\d{3,})\s*(.*)$', line)
            if not match:
                continue
            code, text = match.group(1), match.group(2).strip()
            sentences.append({"code": code, "text": text})
    return sentences


def annotations_file(policy_name: str) -> Path:
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '', policy_name) or "default"
    policy_dir = ANNOTATIONS_DIR / safe_name
    policy_dir.mkdir(parents=True, exist_ok=True)
    return policy_dir / "annotations.json"


def load_annotations(policy_name: str) -> dict:
    """Returns {code: {text, decision, confidence, updated_at, ...}}."""
    path = annotations_file(policy_name)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_annotations(policy_name: str, annotations: dict) -> None:
    path = annotations_file(policy_name)
    path.write_text(json.dumps(annotations, indent=2, ensure_ascii=False), encoding="utf-8")


@annotate_bp.route("/annotate")
def annotate_page():
    policy_name = analyse.POLICY_NAME
    sentences = load_document_sentences(policy_name)
    annotations = load_annotations(policy_name)

    return render_template(
        "annotate.jinja",
        document_title=policy_name.replace("_", " ").title(),
        policy_name=policy_name,
        sentences=sentences,
        annotations=annotations,
        decisions=DECISIONS,
    )


@annotate_bp.route("/annotate/save", methods=["POST"])
def annotate_save():
    data = request.get_json(silent=True) or {}
    code = str(data.get("code", "")).strip()
    text = data.get("text", "")
    decision = str(data.get("decision", "")).strip().lower()
    confidence = data.get("confidence")  # optional, left flexible for future scoring

    if not code:
        return jsonify({"status": "error", "message": "code is required"}), 400
    if decision not in DECISIONS:
        return jsonify({"status": "error", "message": f"decision must be one of {DECISIONS}"}), 400

    policy_name = analyse.POLICY_NAME
    annotations = load_annotations(policy_name)

    record = annotations.get(code, {})
    record.update({
        "code": code,
        "text": text,
        "decision": decision,
        "confidence": confidence,
        "updated_at": time.time(),
    })
    annotations[code] = record
    save_annotations(policy_name, annotations)

    return jsonify({"status": "ok", "annotation": record})


@annotate_bp.route("/annotate/all")
def annotate_all():
    """Returns the full persisted annotation set for the current document."""
    policy_name = analyse.POLICY_NAME
    return jsonify(load_annotations(policy_name))
