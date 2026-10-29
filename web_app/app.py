from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from pathlib import Path
import os
import re
import json
import uuid
import logging
from sentence_ids import is_sentence_line, sentence_id
from upload import upload_bp
import analyse
from analyse import analyse_bp
from annotate import annotate_bp
from review import review_bp
from generate_report import build_report

# Set up logging for validation and LLM operations
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s in %(name)s: %(message)s")
logger = logging.getLogger("dbi_app")

# =============================================================================
# SECTION 1: CONFIGURATION & PATH RESOLUTION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

POLICY_NAME = "visitor_levy"

# Base document files
DOCUMENTS_DIR = PROJECT_ROOT / "documents" / POLICY_NAME
RESULTS_DIR = PROJECT_ROOT / "results" / POLICY_NAME
SENTENCES_FILE = DOCUMENTS_DIR / "input.txt"
REFERENCES_FILE = DOCUMENTS_DIR / "references.json"
METADATA_FILE = DOCUMENTS_DIR / "metadata.json"

# Results are the durable interface between the generation pipeline and the UI.
# A generator may create these files, but routes must always read the persisted
# artefacts rather than use an in-memory LLM response.
ASPECTS = ("diagnosis", "proposal", "implementation", "testing")
SCORED_ASPECTS = ("diagnosis", "proposal", "implementation", "testing")
ASPECT_LABELS = {
    "diagnosis": "Diagnosis",
    "proposal": "Proposal",
    "implementation": "Implementation",
    "testing": "Testing + Evaluation",
}


# Short UI labels for the levels of the Evidence transparency framework (Sense
# about Science / Institute for Government / Alliance for Useful Evidence,
# revised 2017).  These abbreviate the level wording the model scores against,
# which lives in web_app/prompts/score_<aspect>.txt -- edit those first and keep
# these labels in step with them.
SCORE_RUBRIC = {
    0: "No evidence provided, or no explanation of how it has been used",
    1: "Evidence mentioned, with some explanation of how it has been used",
    2: "Evidence properly cited and findable; chain of reasoning is clear",
    3: "Evidence base assessed; uncertainties and contradictions acknowledged"
}

# Testing + evaluation has its own levels in the framework: it scores plans for
# testing, evaluation and publication of results rather than citation of evidence.
TESTING_SCORE_RUBRIC = {
    0: "No indication of success measures, testing or evaluation",
    1: "Some indication of success measures, but no plans for testing",
    2: "Comprehensive success measures, with details of testing and evaluation",
    3: "Reasons for testing explained, and what happens to the results"
}
SCORE_RUBRICS_BY_ASPECT = {"testing": TESTING_SCORE_RUBRIC}

# Per-session caching directory: web_app/data/sessions/<session_id>/
SESSIONS_CACHE_DIR = BASE_DIR / "data" / "sessions"
SESSIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Strict validation toggle (raises error if True, logs and displays warning if False)
STRICT_VALIDATION = os.environ.get("STRICT_VALIDATION", "false").lower() in ("true", "1")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dbi-dev-secret-key-change-in-production")


app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

app.register_blueprint(upload_bp)
app.register_blueprint(analyse_bp)
app.register_blueprint(annotate_bp)
app.register_blueprint(review_bp)

def resolve_existing_path(primary: Path, fallback: Path = None) -> Path:
    """Helper to locate files across project root or web_app directories."""
    if primary.exists():
        return primary
    if fallback and fallback.exists():
        return fallback
    local_path = BASE_DIR / primary.name
    if local_path.exists():
        return local_path
    return primary


# Directory containing modular LLM prompt files
PROMPTS_DIR = BASE_DIR / "prompts"


def load_prompt_file(filename: str, fallback_text: str = "") -> str:
    """Loads prompt text from the web_app/prompts directory."""
    path = PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return fallback_text


def load_document_title() -> str:
    """Load the selected document's display title from metadata.json."""
    metadata_file = PROJECT_ROOT / "documents" / analyse.POLICY_NAME / "metadata.json"
    if metadata_file.exists():
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            title = metadata.get("title", "") if isinstance(metadata, dict) else ""
            if isinstance(title, str) and title.strip():
                return title.strip()
        except json.JSONDecodeError:
            logger.warning("Invalid document metadata file: %s", metadata_file)
    return analyse.POLICY_NAME.replace("_", " ").title()


def available_result_documents() -> list[dict]:
    """List documents with a results/ folder, labelled by their metadata.json title."""
    results_root = PROJECT_ROOT / "results"
    if not results_root.exists():
        return []
    documents = []
    for path in sorted(results_root.iterdir()):
        if not path.is_dir():
            continue
        alias = path.name
        metadata_file = PROJECT_ROOT / "documents" / alias / "metadata.json"
        title = alias.replace("_", " ").title()
        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                meta_title = metadata.get("title", "") if isinstance(metadata, dict) else ""
                if isinstance(meta_title, str) and meta_title.strip():
                    title = meta_title.strip()
            except json.JSONDecodeError:
                logger.warning("Invalid document metadata file: %s", metadata_file)
        documents.append({"alias": alias, "title": title})
    return documents


def result_file(aspect: str, filename: str) -> Path:
    """Return a file in the standard results/<policy>/<aspect>/ layout."""
    if aspect not in ASPECTS:
        raise ValueError(f"Unknown policy aspect: {aspect}")
    return PROJECT_ROOT / "results" / analyse.POLICY_NAME / aspect / filename


def load_result_lines(aspect: str) -> list[str]:
    """Load the reviewed line selection for an aspect from its results folder."""
    # lines_2.txt is the post-processed selection used to produce text.txt.
    # Keep lines.txt as a compatibility fallback for newly generated results.
    path = result_file(aspect, "lines_3.txt")
    if not path.exists():
        path = result_file(aspect, "lines.txt")
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_result_categories(aspect: str) -> dict:
    """Load classifications from the persisted JSON (or legacy text) artefact."""
    json_path = result_file(aspect, "classification.json")
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        themes = data.get("themes", []) if isinstance(data, dict) else []
        return {
            index: {
                "name": theme.get("name", "Unnamed Category"),
                "lines": theme.get("ids", []),
            }
            for index, theme in enumerate(themes)
            if isinstance(theme, dict)
        }

    text_path = result_file(aspect, "classification.txt")
    if text_path.exists():
        return parse_categories_text(text_path.read_text(encoding="utf-8"))
    return {}


def load_result_score(aspect: str) -> dict:
    """Load an aspect's persisted evidence-transparency assessment."""
    path = result_file(aspect, "scoring.json")
    if not path.exists():
        return {"score": 0, "reasoning": "No assessment report available."}
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# SECTION 2: MANAGED ERROR DEFINITIONS & VALIDATION
# =============================================================================

class AppValidationError(Exception):
    """Base exception for application validation errors."""
    pass


class LineNotFoundError(AppValidationError):
    """Raised when a classified line code is not found in the source document."""
    def __init__(self, line_id: str, category_name: str = "", context: str = ""):
        self.line_id = line_id
        self.category_name = category_name
        self.context = context
        message = f"Classified sentence code '{line_id}' was not found in the document"
        if category_name:
            message += f" (under category '{category_name}')"
        if context:
            message += f" - {context}"
        super().__init__(message)


class InvalidLLMOutputError(AppValidationError):
    """Raised when LLM output format is invalid or cannot be parsed."""
    pass


@app.errorhandler(AppValidationError)
def handle_validation_error(error):
    """Managed error handler that provides friendly JSON or HTML responses."""
    logger.error(f"Managed ValidationError encountered: {error}")
    if request.is_json or request.path.startswith("/get_references"):
        return jsonify({
            "status": "error",
            "error_type": error.__class__.__name__,
            "message": str(error)
        }), 400
    return render_template(
        "error.jinja",
        error_title="Validation Error",
        error_message=str(error),
        details=getattr(error, "context", "")
    ), 400


def validate_classified_lines(
    categories: dict,
    valid_sentence_ids: set[str],
    extracted_diagnosis_lines: set[str] = None,
    strict: bool = False
) -> tuple[dict, list[str]]:
    """
    Validates categories and their associated sentence codes:
    1. Ensures each classified line ID (e.g. S011) exists in the source document (valid_sentence_ids).
       - If missing and strict=True: raises LineNotFoundError (managed error).
       - If missing and strict=False: logs warning, collects managed warning message, and omits the invalid line.
    2. Logs informational warnings if a classified line was not in the extracted diagnosis set.

    Returns:
        tuple: (cleaned_categories, list_of_warning_messages)
    """
    warnings = []
    cleaned_categories = {}

    for cat_id, cat_data in categories.items():
        cat_name = cat_data.get("name", "Unnamed Category")
        lines = cat_data.get("lines", [])
        valid_cat_lines = []

        for line_id in lines:
            line_id = line_id.strip()
            if not line_id:
                continue

            # Check 1: Does this sentence code exist in the document?
            if line_id not in valid_sentence_ids:
                msg = f"Classified line '{line_id}' in theme '{cat_name}' was not found in the source document."
                if strict:
                    raise LineNotFoundError(
                        line_id=line_id,
                        category_name=cat_name,
                        context="Sentence code missing from pension_sentences.txt"
                    )
                else:
                    logger.warning(f"[Validation Warning] {msg}")
                    warnings.append(msg)
                    # Exclude the missing line so the rest of the application renders safely
                    continue

            # Check 2: Was this line in the extracted diagnosis lines?
            if extracted_diagnosis_lines and line_id not in extracted_diagnosis_lines:
                logger.info(f"[Validation Notice] Line '{line_id}' under '{cat_name}' was not in the extracted diagnosis set.")

            valid_cat_lines.append(line_id)

        cleaned_categories[cat_id] = {
            "name": cat_name,
            "lines": valid_cat_lines
        }

    return cleaned_categories, warnings


def validate_assessment_report(report_data: dict, aspect: str = "diagnosis") -> dict:
    """
    Validates that the final LLM assessment report has an integer score (0-3) and reasoning.
    Enriches the dictionary with rubric_label for display.
    """
    rubric = SCORE_RUBRICS_BY_ASPECT.get(aspect, SCORE_RUBRIC)

    if not isinstance(report_data, dict):
        logger.warning(f"[Validation Warning] Assessment data is not a dict: {type(report_data)}")
        report_data = {"score": 0, "reasoning": "Invalid assessment report format."}

    raw_score = report_data.get("score", 0)
    try:
        score = int(raw_score)
    except (ValueError, TypeError):
        logger.warning(f"[Validation Warning] Non-integer assessment score: {raw_score}")
        score = 0

    # Ensure score is within valid range
    if score not in rubric:
        logger.warning(f"[Validation Warning] Assessment score {score} out of standard range (0-3)")

    reasoning = report_data.get("reasoning", "")
    if not isinstance(reasoning, str) or not reasoning.strip():
        reasoning = "No assessment reasoning provided."

    rubric_label = rubric.get(score, "Unspecified evaluation score")

    # Per-theme levels are optional: results scored before theme_scores was
    # added to the Score schema simply have none.
    theme_scores = []
    for entry in report_data.get("theme_scores") or []:
        if not isinstance(entry, dict):
            continue
        try:
            theme_score = int(entry.get("score"))
        except (ValueError, TypeError):
            logger.warning("[Validation Warning] Non-integer theme score: %s", entry.get("score"))
            continue
        theme_scores.append({
            "theme": str(entry.get("theme", "")).strip() or "Untitled theme",
            "score": theme_score,
            "justification": str(entry.get("justification", "")).strip(),
        })

    return {
        "score": score,
        "reasoning": reasoning.strip(),
        "rubric_label": rubric_label,
        "theme_scores": theme_scores,
    }


# =============================================================================
# SECTION 3: LLM CALL PLACEHOLDERS
# =============================================================================

def call_llm_extract_diagnosis_lines(document_text: str) -> list[str]:
    """
    LLM Call 1: Extract sentences relevant to policy diagnosis.

    Prompt file: web_app/prompts/extract_diagnosis.txt
    Model: claude-sonnet-5-5  |  max_tokens: 20000
    Schema: Sentence(id: str)  →  structured JSON list of sentence codes

    Returns:
        A list of sentence codes: e.g. ['S002', 'S003', 'S010', ...]
    """
    # ---------------------------------------------------------------------------
    # LLM BOOSTER — uncomment when ready to use live Anthropic API
    # ---------------------------------------------------------------------------
    # import anthropic
    # from pydantic import BaseModel
    #
    # class Sentence(BaseModel):
    #     id: str
    #
    # task_prompt = load_prompt_file("extract_diagnosis.txt",
    #     fallback_text=(
    #         "Your task is to identify all sentences that contain claims or evidence "
    #         "for the diagnosis of the problem this policy aims to address. "
    #         "These should be factual not value statements. Do not consider claims "
    #         "regarding the proposed intervention, implementation or testing and "
    #         "evaluation of the intervention. Provide the outcome of your search as "
    #         "a list of sentence codes encoded in a structured json output"
    #     )
    # )
    # system_prompt = load_prompt_file("system.txt",
    #     fallback_text="You are an employee at a policy institute who is concerned "
    #     "that the public can see the evidence behind policy decisions and that a "
    #     "motivated citizen can understand why a policy decision has been made."
    # )
    # schema = Sentence.model_json_schema()
    # schema["additionalProperties"] = False
    #
    # client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    # response = client.messages.create(
    #     model="claude-sonnet-5-5",
    #     max_tokens=20000,
    #     system=system_prompt,
    #     messages=[{
    #         "role": "user",
    #         "content": f"<document>{document_text}</document><task>{task_prompt}</task>"
    #     }],
    #     output_config={
    #         "format": {
    #             "type": "json_schema",
    #             "schema": schema,
    #         }
    #     },
    # )
    # result = json.loads(response.content[0].text)
    # # result is a list of {"id": "S001"} objects; extract just the codes
    # return [item["id"] for item in result if isinstance(item, dict) and "id" in item]
    # ---------------------------------------------------------------------------
    logger.info("[LLM Placeholder] Calling call_llm_extract_diagnosis_lines...")

    # The route consumes the persisted artefact, never a response object.
    return load_result_lines("diagnosis")



# =============================================================================
# SECTION 4: SESSION CACHING & DATA MANAGEMENT
# =============================================================================

def get_or_create_session_id() -> str:
    """Retrieves session ID from query param, JSON body, or Flask session; generates one if absent."""
    # 1. Query parameter (?session_id=...)
    if request.args.get("session_id"):
        sid = request.args.get("session_id").strip()
        session["session_id"] = sid
        return sid

    # 2. JSON request body if present
    if request.is_json:
        data = request.get_json(silent=True) or {}
        if data.get("session_id"):
            sid = str(data.get("session_id")).strip()
            session["session_id"] = sid
            return sid

    # 3. Flask session cookie
    if "session_id" not in session:
        session["session_id"] = uuid.uuid4().hex[:12]

    return session["session_id"]


def get_session_dir(session_id: str) -> Path:
    """Returns and ensures directory path for the given session cache."""
    safe_sid = re.sub(r'[^a-zA-Z0-9_\-]', '', session_id)
    if not safe_sid:
        safe_sid = "default_session"
    session_dir = SESSIONS_CACHE_DIR / safe_sid
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def parse_categories_text(raw_text: str) -> dict:
    """
    Parses classification text into a structured dictionary:
    {
        0: {"name": "Theme Name", "lines": ["S011", "S032"]},
        ...
    }
    """
    categories = {}
    count = 0
    for line in raw_text.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue

        if is_sentence_line(line_clean):
            if count == 0:
                categories[0] = {"name": "General Diagnosis", "lines": []}
                count = 1
            categories[count - 1]["lines"].append(line_clean)
        else:
            categories[count] = {"name": line_clean, "lines": []}
            count += 1

    return categories


def get_or_generate_session_data(
    session_id: str,
    valid_sentence_ids: set[str],
    document_text: str,
    strict: bool = False
) -> tuple[list[str], dict, list[str]]:
    """Load persisted diagnosis results for the viewer without invoking an LLM."""
    diagnosis_lines = load_result_lines("diagnosis")
    raw_categories = load_result_categories("diagnosis")

    # Validate persisted classifications against the persisted source document.
    categories, warnings = validate_classified_lines(
        raw_categories,
        valid_sentence_ids=valid_sentence_ids,
        extracted_diagnosis_lines=set(diagnosis_lines),
        strict=strict
    )

    assessment = validate_assessment_report(load_result_score("diagnosis"), aspect="diagnosis")

    return diagnosis_lines, categories, warnings, assessment


def load_source_sentences():
    """Loads source sentences from pension_sentences.txt."""
    doc_path = PROJECT_ROOT / "documents" / analyse.POLICY_NAME / "input.txt"
    if not doc_path.exists():
        raise FileNotFoundError(f"Source sentences file not found: {doc_path}")

    with open(doc_path, "r", encoding="utf-8") as f:
        raw_lines = [line.rstrip("\n") for line in f]
# TODO: adjust for not labelled sections of markdown
    valid_sentence_ids = {
        sentence_id(line) for line in raw_lines if is_sentence_line(line)
    }
    full_document_text = "\n".join(raw_lines)
    return raw_lines, valid_sentence_ids, full_document_text


def highlight_sentence(
    code: str,
    diagnosis_lines: set,
    proposal_lines: set,
    implementation_lines: set,
    testing_lines: set,
) -> str | None:
    """Determines which aspect (if any) a sentence code should be highlighted as."""
    if code in diagnosis_lines:
        return 'diagnosis'
    elif code in proposal_lines:
        return 'proposal'
    elif code in implementation_lines:
        return 'implementation'
    elif code in testing_lines:
        return 'testing'
    return None


# =============================================================================
# SECTION 5: FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    try:
        session_id = get_or_create_session_id()
        raw_lines, valid_sentence_ids, document_text = load_source_sentences()

        # Check for strict validation toggle via query string or environment
        strict = request.args.get("strict", "").lower() in ("true", "1") or STRICT_VALIDATION

        # Retrieve cached session data or trigger LLM generation
        diagnosis_lines, categories, validation_warnings, _diagnosis_assessment = get_or_generate_session_data(
            session_id=session_id,
            valid_sentence_ids=valid_sentence_ids,
            document_text=document_text,
            strict=strict
        )

        # All highlights come from the same persisted results tree as diagnosis.
        proposal_lines = set(load_result_lines("proposal"))
        implementation_lines = set(load_result_lines("implementation"))
        testing_lines = set(load_result_lines("testing"))
        assessments = {
            "diagnosis": {"label": ASPECT_LABELS["diagnosis"], **_diagnosis_assessment},
            **{
                aspect: {
                    "label": ASPECT_LABELS[aspect],
                    **validate_assessment_report(load_result_score(aspect), aspect=aspect),
                }
                for aspect in SCORED_ASPECTS
                if aspect != "diagnosis"
            },
        }

        diagnosis_set = set(diagnosis_lines)

        lines = []
        for line in raw_lines:
            code = sentence_id(line) or ""
            category = highlight_sentence(
                code, diagnosis_set, proposal_lines, implementation_lines, testing_lines
            )
            lines.append({
                "text": line,
                "category": category,
            })

        return render_template(
            'index.jinja',
            lines=lines,
            document_title=load_document_title(),
            session_id=session_id,
            warnings=validation_warnings,
            assessments=assessments,
            aspect_labels=ASPECT_LABELS,
            available_documents=available_result_documents(),
            current_alias=analyse.POLICY_NAME
        )
    except AppValidationError:
        raise
    except Exception as e:
        logger.exception("Error processing index route")
        return render_template("error.jinja", error_title="Application Error", error_message=str(e)), 500


@app.route("/switch-document/<alias>")
def switch_document(alias):
    try:
        analyse.set_policy_name(alias)
    except ValueError as e:
        return render_template("error.jinja", error_title="Unknown document", error_message=str(e)), 404
    return redirect(url_for("index"))


@app.route("/report")
def report():
    alias = analyse.POLICY_NAME
    summary_only = request.args.get("summary", "").lower() in ("1", "true")
    try:
        html_content = build_report(alias, summary_only=summary_only)
    except FileNotFoundError as e:
        return render_template("error.jinja", error_title="Report unavailable", error_message=str(e)), 404
    return html_content


@app.route("/get_references", methods=["POST"])
def get_references():
    try:
        data = request.get_json(silent=True) or {}
        aspect = data.get("aspect")
        if aspect not in ASPECTS:
            return jsonify({"error": f"aspect must be one of {ASPECTS}"}), 400

        # References must correspond to the persisted line selection shown by the
        # page, rather than an old session cache or an LLM response.
        aspect_lines = set(load_result_lines(aspect))
        if not aspect_lines:
            return jsonify({"error": f"No {aspect} lines found"}), 404

        # Load reference definitions. references.json is a list of
        # {"number": ..., "text": ...} entries; index it by number for lookup.
        ref_file = PROJECT_ROOT / "documents" / analyse.POLICY_NAME / "references.json"
        reference_dict = {}
        if ref_file.exists():
            with open(ref_file, "r", encoding="utf-8") as f:
                raw_references = json.load(f)
            if isinstance(raw_references, list):
                reference_dict = {
                    str(entry.get("number")): entry
                    for entry in raw_references
                    if isinstance(entry, dict) and entry.get("number") is not None
                }
            elif isinstance(raw_references, dict):
                reference_dict = raw_references

        # Search document lines for reference citations within this aspect, in
        # document order, with no theme/claim grouping.
        ref_nums = []
        doc_file = PROJECT_ROOT / "documents" / analyse.POLICY_NAME / "input.txt"
        if doc_file.exists():
            with open(doc_file, "r", encoding="utf-8") as f:
                for line in f:
                    line_code = sentence_id(line) or ""
                    if line_code in aspect_lines and "**" in line:
                        match = re.search(r'Reference:\s*(\d+(?:\s*,\s*\d+)*)', line)
                        if match:
                            ref_text = match.group(1)
                            for num in ref_text.split(','):
                                num_clean = num.strip()
                                if num_clean and num_clean not in ref_nums:
                                    ref_nums.append(num_clean)

        references = []
        for num in ref_nums:
            ref_info = reference_dict.get(num, {"text": f"Reference #{num} citation not found."})
            references.append({num: ref_info})

        return jsonify(references)
    except AppValidationError:
        raise
    except Exception as e:
        logger.exception("Error processing get_references")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
