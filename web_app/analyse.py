"""Analysis workflow and status routes.

This module owns LLM-facing work.  The viewer in app.py only reads the
persisted document and result artefacts produced here.
"""

from pathlib import Path
import json
import math
import re
from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
import os
from dotenv import load_dotenv
from preprocess import construct_xml, extract_lines
from string import Template
from xml.sax.saxutils import unescape
from llm_monitor import track_llm_call, write_run_summary

load_dotenv()

from flask import Blueprint, Response, current_app, render_template, request, stream_with_context

# List of Pydantic schema

class Category(str, Enum):
        model_config = ConfigDict(extra="forbid")
        A = "A"
        B = "B"
        C = "C"
        D = "D"
        NONE = "None"

class SentenceClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentence_id: str
    category: Category = Field(description="Best-fitting category for this sentence")

class ClassificationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy_intervention: str
    policy_problem: str = Field(description="The problem the intervention is meant to solve, in 2-5 words")
    classifications: list[SentenceClassification]


class ContinuationClassificationOutput(BaseModel):
    """Output schema for later chunks: the intervention/problem anchors are
    already fixed from the first chunk, so this omits them."""
    model_config = ConfigDict(extra="forbid")
    classifications: list[SentenceClassification]


class Theme(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    ids: list[str]

class Themes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    themes: list[Theme]

class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    number: str

class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    data: list[str]
    reasoning: str
    references: list[Reference]

class SummaryTheme(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    claims: list[Claim]

class SummaryThemes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    themes: list[Theme]

class ThemeWeight(str, Enum):
    core = "core"
    not_core = "not core"

class WeightedTheme(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme: str = Field(description="Name of the theme, exactly as given in the document extract")
    # The reason comes before the weight so the label follows from it.
    reason: str = Field(description="What part of the argument the theme carries, and which problem or should-explain point it bears on")
    weight: ThemeWeight

class ThemeWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")
    themes: list[WeightedTheme]

class CoreLine(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="A short name for this line of argument")
    # The reason comes before the ids so the selection follows from it.
    reason: str = Field(description="Why the case for this aspect rests on this line of argument")
    ids: list[str] = Field(description="Sentence codes that make up this line of argument, copied exactly from the input")

class CoreLines(BaseModel):
    model_config = ConfigDict(extra="forbid")
    core_lines: list[CoreLine]

class CoreTheme(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="A short name, of the kind that could head a section")
    claim: str = Field(description="One sentence stating the claim the document makes in this theme")

class AspectCore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    core_themes: list[CoreTheme]
    note: str = Field(description="Anything unusual about this aspect, or an empty string")

class DocumentCore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    diagnosis: AspectCore
    proposal: AspectCore
    implementation: AspectCore

class ThemeScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme: str = Field(description="Name of the theme, as given in the document extract")
    primary_evidence: bool = Field(description="Whether the theme relies on the department's own primary evidence, such as its consultation responses, data or analysis")
    score: int = Field(description="Level from 0 to 3 this theme reaches on its own")
    justification: str = Field(description="One sentence naming the lines or reference numbers behind this theme's level")

class Score(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Per-theme levels come first so the overall reasoning and score are decided
    # after the model has worked through each theme, making the aggregation
    # visible rather than hidden.
    theme_scores: list[ThemeScore]
    reasoning: str
    # Anthropic's structured-output schema rejects integer minimum/maximum
    # constraints, so the 0-3 range is stated in the description instead of
    # enforced via Field(ge=..., le=...).
    score: int = Field(description="Final score from 0 to 3, decided after the reasoning above")

class BlockScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning: str
    score: int = Field(description="Level from 0 to 3, decided after the reasoning above")

analyse_bp = Blueprint("analyse", __name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICY_NAME = "visitor_levy"
DOCUMENTS_DIR = PROJECT_ROOT / "documents" / POLICY_NAME
RESULTS_DIR = PROJECT_ROOT / "results" / POLICY_NAME
ASPECTS = ("diagnosis", "proposal", "implementation", "testing")
# How the weighting prompt names each aspect; worded as in the opening line of
# the matching score_<aspect>.txt.
ASPECT_DESCRIPTIONS = {
    "diagnosis": "the DIAGNOSIS: why something is proposed, that is, what the issue is that will be addressed",
    "proposal": "the PROPOSAL: the government's chosen intervention",
    "implementation": "IMPLEMENTATION: how the chosen intervention will be rolled out",
    "testing": "TESTING AND EVALUATION: how we will know if the policy has worked",
}
WEB_APP_ROOT = Path(__file__).resolve().parent

# Documents at or below this many sentences are extracted in a single call.
# Larger documents are split into chunks of CHUNK_SIZE sentences so each
# extraction call stays within a manageable context/output size.
CHUNK_SENTENCE_THRESHOLD = 600
CHUNK_SIZE = 500

_client = None
_system_prompt = ""

def available_policies() -> list[str]:
    """Return valid document aliases available for analysis."""
    documents_root = PROJECT_ROOT / "documents"
    if not documents_root.exists():
        return []
    return sorted(path.name for path in documents_root.iterdir() if path.is_dir())


def most_recently_uploaded_policy(policies: list[str]) -> str:
    """Return the alias with the newest metadata.json 'uploaded_at', if any."""
    documents_root = PROJECT_ROOT / "documents"
    latest_alias = ""
    latest_uploaded_at = ""
    for alias in policies:
        metadata_path = documents_root / alias / "metadata.json"
        try:
            uploaded_at = json.loads(metadata_path.read_text(encoding="utf-8")).get("uploaded_at", "")
        except (OSError, ValueError):
            continue
        if uploaded_at > latest_uploaded_at:
            latest_uploaded_at = uploaded_at
            latest_alias = alias
    return latest_alias


def set_policy_name(policy_name: str) -> None:
    """Point this analysis run at a selected documents/<alias> directory."""
    if policy_name not in available_policies():
        raise ValueError("Select a valid document directory")

    global POLICY_NAME, DOCUMENTS_DIR, RESULTS_DIR
    POLICY_NAME = policy_name
    DOCUMENTS_DIR = PROJECT_ROOT / "documents" / POLICY_NAME
    RESULTS_DIR = PROJECT_ROOT / "results" / POLICY_NAME

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
        _client = Anthropic(api_key=api_key)
    return _client

def _load_source_text() -> str:
    path = DOCUMENTS_DIR / "input.txt"
    return path.read_text(encoding="utf-8") if path.exists() else ""

def result_file(aspect: str, filename: str) -> Path:
    return RESULTS_DIR / aspect / filename

def _get_prompt(task):
    with open(WEB_APP_ROOT / f"prompts/{task}.txt", "r") as f:
        prompt = f.read()
    return prompt

def _set_up_dirs():
    for aspect in ASPECTS:
        (RESULTS_DIR / aspect).mkdir(parents=True, exist_ok=True)
    return None

def _set_system_prompt() -> list[dict]:
    global _system_prompt
    with open(WEB_APP_ROOT / "prompts/system.txt", "r") as f:
        text = f.read()
    _system_prompt = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
    return _system_prompt


# LLM call boundary ---------------------------------------------------------
# Replace these implementations with the live Anthropic calls when enabled.
# Their return values are always persisted by run_analysis before the viewer
# reads them, so no API response is ever rendered directly.
def _chunk_document_lines(document_text: str) -> list[str]:
    """Split document_text into as many equal-sized chunks as it takes to
    keep each one at or under CHUNK_SIZE sentence lines.

    Returns a single-element list (the whole document) when the document is
    at or below CHUNK_SENTENCE_THRESHOLD. Otherwise splits into
    ceil(total / CHUNK_SIZE) chunks of as-equal-as-possible size, rather than
    fixed CHUNK_SIZE chunks plus a small remainder chunk.
    """
    lines = document_text.splitlines()
    total = len(lines)
    if total <= CHUNK_SENTENCE_THRESHOLD:
        return [document_text]

    num_chunks = math.ceil(total / CHUNK_SIZE)
    base_size, remainder = divmod(total, num_chunks)

    chunks = []
    start = 0
    for i in range(num_chunks):
        size = base_size + (1 if i < remainder else 0)
        chunks.append("\n".join(lines[start:start + size]))
        start += size
    return chunks


def _raise_if_refused(response, step: str) -> None:
    """Claude Haiku 5.5 (and Sonnet 5.5) run safety classifiers that can decline
    a request: HTTP 200, stop_reason "refusal", no usable content.  Fail with a
    clear message instead of a confusing parse error."""
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        raise RuntimeError(f"{step}: request declined by the model (stop_reason=refusal, category={category})")


def _run_extract_call(prompt: str, document_text: str, schema: dict, step: str):
    """Run one extraction call (first chunk or a continuation chunk) and
    return the parsed JSON output alongside its usage record."""
    with track_llm_call(step, POLICY_NAME, "anthropic", "claude-sonnet-5-5") as call:
        with _client.messages.stream(
        model="claude-sonnet-5-5",
        max_tokens=75000,
        system=_system_prompt,
        messages=[{
            "role": "user",
            "content": [
                {
                    # Fixed across every call regardless of which document is
                    # being analysed, so this breakpoint hits on every run.
                    "type": "text",
                    "text": "<task>" + prompt + "</task>",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    # Varies per document, so this breakpoint only hits when
                    # the same document is analysed again (e.g. reproducibility runs).
                    "type": "text",
                    "text": "<document>" + document_text + "</document>",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        }],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        },
        ) as stream:
            message = stream.get_final_message()
        call.set_usage_from_response(message)
    if message.stop_reason == "max_tokens":
        raise RuntimeError(
            f"{step}: response truncated (stop_reason=max_tokens, "
            f"output_tokens={call.record.output_tokens}); raise max_tokens or split the chunk further"
        )
    text_block = next(block for block in message.content if block.type == "text")
    data = json.loads(text_block.text)
    return data, call.record


def call_llm_extract_lines():
    schema = ClassificationOutput.model_json_schema()
    continuation_schema = ContinuationClassificationOutput.model_json_schema()
    document_text = _load_source_text()
    chunks = _chunk_document_lines(document_text)
    chunked = len(chunks) > 1

    records = []
    all_classifications = []
    intervention = ""
    problem = ""

    for index, chunk_text in enumerate(chunks):
        if index == 0:
            prompt = _get_prompt("extract")
            step = "extract_lines:chunk_1" if chunked else "extract_lines"
            data, record = _run_extract_call(prompt, chunk_text, schema, step)
            intervention = data["policy_intervention"]
            problem = data["policy_problem"]
        else:
            prompt = _get_prompt("extract_continuation").format(
                intervention_description=intervention,
                problem_description=problem,
            )
            step = f"extract_lines:chunk_{index + 1}"
            data, record = _run_extract_call(prompt, chunk_text, continuation_schema, step)
        records.append(record)
        all_classifications.extend(data["classifications"])

    names = {
        "A" : "diagnosis",
        "B" : "proposal",
        "C" : "implementation",
        "D" : "testing"
    }
    # intervention.txt is substituted into the scoring prompts as $intervention,
    # so it stays a bare one-liner; the problem anchor gets its own file.
    with open(RESULTS_DIR / "intervention.txt", "w") as f:
        f.write(intervention)
    with open(RESULTS_DIR / "problem.txt", "w") as f:
        f.write(problem)

    for aspect_key, aspect_name in names.items():
        with open(result_file(aspect_name, "lines.txt"), "w") as f:
            for classification in all_classifications:
                if classification["category"] == aspect_key:
                    f.write(classification["sentence_id"])
                    f.write("\n")
        extract_lines(DOCUMENTS_DIR, RESULTS_DIR, aspect_name)
    return records


def call_llm_classify_themes(aspect_name: str, model: str = "claude-haiku-5-5"):
    schema = Themes.model_json_schema()
    prompt = _get_prompt("classify")
    with open(result_file(aspect_name, "text.txt"), "r") as f:
        aspect_text = f.read()
    # Streamed so the larger max_tokens (room for Sonnet's thinking on long
    # aspects) does not hit the SDK's non-streaming timeout.
    with track_llm_call(f"classify_themes:{aspect_name}", POLICY_NAME, "anthropic", model) as call:
        with _client.messages.stream(
            max_tokens=64000,
            model=model,
            system=_system_prompt,
            messages=[{"role": "user", "content": "<document>"+ aspect_text +"</document>" + "<task>" + prompt + "</task>"}],
            output_config={
                    "effort": "low",  # mechanical grouping; Haiku 5.5 thinks by default
                    "format": {
                        "type": "json_schema",
                        "schema": schema,
                    }
                },
            ) as stream:
            response = stream.get_final_message()
        call.set_usage_from_response(response)
    _raise_if_refused(response, f"classify_themes:{aspect_name}")
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"classify_themes:{aspect_name}: response truncated (stop_reason=max_tokens, "
            f"output_tokens={call.record.output_tokens}); raise max_tokens or split the aspect further"
        )
    with open(result_file(aspect_name, "classification.json"), "w") as f:
        f.write(next(b.text for b in reversed(response.content) if b.type == "text"))
    return call.record



# def call_llm_summarise_claims(classified_themes_text: str, document_text: str, references_data: dict | None = None) -> dict:
#     schema = SummaryThemes.model_json_schema()
#     prompt = _get_prompt("summarise")
#     response = _client.messages.create(
#     max_tokens=5000,
#     model="claude-haiku-5-5",
#     system="You are an employee at a policy institute who is concerned that the public can see the evidence behind policy decisions and that a motivated citizen can understand why a policy decision has been made.",
#     messages=[{"role": "user", "content": "<document>"+ total_text + "</document>" + "<prompt>" + prompt + "</prompt>"}],
#     output_config={
#                 "format": {
#                     "type": "json_schema",
#                     "schema": schema,
#                 }
#             },
#     )
#     data = json.loads(response.content[0].text)
#     return {}


def _anchor_texts() -> tuple[str, str]:
    """The intervention and problem anchors recorded at extraction.  Results
    extracted before problem.txt existed (e.g. emissions_trading) get a
    placeholder rather than failing the substitution."""
    intervention_text = (RESULTS_DIR / "intervention.txt").read_text()
    problem_path = RESULTS_DIR / "problem.txt"
    problem_text = problem_path.read_text().strip() if problem_path.exists() else ""
    return intervention_text, problem_text or "not recorded for this document"


def _read_summary_xml(aspect_name: str) -> str:
    with open(result_file(aspect_name, "summary.xml"), "r") as f:
        # summary.xml is escaped for well-formedness on disk; unescape it here
        # so the model reads the document's original text (e.g. "RD&I", not
        # "RD&amp;I") rather than XML-entity-encoded text.
        xml = unescape(f.read(), {"&quot;": '"', "&apos;": "'"})
    # Sentence ids are only used for traceability in the stored file; the
    # prompts never refer to them, so drop them to save tokens.
    return re.sub(r'<line id="[^"]*">', "<line>", xml)


def _weighting_input(xml: str) -> str:
    """Reduce summary.xml to theme names and sentence text, so the weighting
    call cannot see which themes are cited or how well.  The <ungrouped>
    bucket is not a theme and gets no weight, so it is dropped too, as is any
    weight attribute left over from an earlier run."""
    xml = re.sub(r"<references>.*?</references>\n?", "", xml, flags=re.S)
    xml = re.sub(r"<links>.*?</links>\n?", "", xml, flags=re.S)
    xml = re.sub(r"<ungrouped[^>]*>.*?</ungrouped>\n?", "", xml, flags=re.S)
    xml = re.sub(r"\s*\*\*\s*\((?:Reference|Link):[^)]*\)", "", xml)
    return re.sub(r' weight="[^"]*"', "", xml)


def call_llm_weight_themes(aspect_name: str):
    """Mark each theme core or not core before scoring, in a call that sees
    the argument but not the evidence, so a theme's weight cannot follow from
    how well it is evidenced.  Writes weights.json, which construct_xml then
    folds into summary.xml for the scorer; it can be corrected by hand before
    scoring, and must be regenerated whenever classification.json is."""
    schema = ThemeWeights.model_json_schema()
    xml = _weighting_input(_read_summary_xml(aspect_name))
    intervention_text, problem_text = _anchor_texts()
    prompt = Template(_get_prompt("weight_themes")).substitute(
        intervention=intervention_text,
        problem=problem_text,
        aspect=ASPECT_DESCRIPTIONS[aspect_name],
        should_explain=_get_prompt(f"explain_{aspect_name}").rstrip("\n"),
    )
    with track_llm_call(f"weight_themes:{aspect_name}", POLICY_NAME, "anthropic", "claude-sonnet-5-5") as call:
        response = _client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=16000,
            thinking={"type": "adaptive", "display": "summarized"},
            system=_system_prompt,
            messages=[{"role": "user", "content": "<document>" + xml + "</document>" + "<task>" + prompt + "</task>"}],
            output_config={
                "format": {
                "type": "json_schema",
                "schema": schema,
            }}
            )
        call.set_usage_from_response(response)
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"weight_themes:{aspect_name}: response truncated (stop_reason=max_tokens, "
            f"output_tokens={call.record.output_tokens}); raise max_tokens"
        )
    data = json.loads(response.content[-1].text)

    # construct_xml matches weights to themes by name, so a renamed, dropped
    # or invented theme would silently lose its weight: fail loudly instead.
    # The yardstick is the themes the call was shown: construct_xml leaves out
    # a theme with no sentences from this aspect, so classification.json can
    # hold names that summary.xml does not.
    expected = re.findall(r"<theme[^>]*>\s*<name>(.*?)</name>", xml, flags=re.S)
    returned = [entry["theme"] for entry in data["themes"]]
    if sorted(returned) != sorted(expected):
        raise RuntimeError(
            f"weight_themes:{aspect_name}: themes returned {returned} do not match "
            f"the themes in classification.json {expected}"
        )
    with open(result_file(aspect_name, "weights.json"), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return call.record


def _strip_citations(text: str) -> str:
    """Remove in-text citation markers, as _weighting_input does for themes."""
    return re.sub(r"\s*\*\*\s*\((?:Reference|Link):[^)]*\)", "", text)


# The aspects find_core names core themes for.  Testing is scored as one
# block: its rubric is about success measures and plans, not an argument.
CORE_ASPECTS = ("diagnosis", "proposal", "implementation")


def _labelled_document() -> str:
    """The whole document, one sentence per line in order, each tagged with
    the aspect extraction put it in ([None] if none), citations stripped.
    The labels come from each aspect's text.txt, which already merges the
    chunks of a long document."""
    labels = {}
    for aspect in ASPECTS:
        path = result_file(aspect, "text.txt")
        if path.exists():
            for sid in re.findall(r"^(S\d{4,})", path.read_text(), flags=re.M):
                labels.setdefault(sid, aspect.capitalize())
    lines = []
    for line in (DOCUMENTS_DIR / "input.txt").read_text().splitlines():
        match = re.match(r"(S\d{4,})\s?(.*)", line)
        if not match:
            continue
        sid, body = match.groups()
        lines.append(f"{sid} [{labels.get(sid, 'None')}] {_strip_citations(body)}")
    return "\n".join(lines)


def call_llm_find_core():
    """Name each aspect's core themes from the whole document, before any
    grouping and without sight of the evidence.  Writes core_themes.json."""
    schema = DocumentCore.model_json_schema()
    intervention_text, problem_text = _anchor_texts()
    aspects = "\n\n".join(
        f"{ASPECT_DESCRIPTIONS[a]}. The document should explain:\n"
        + _get_prompt(f"explain_{a}").rstrip("\n")
        for a in CORE_ASPECTS
    )
    prompt = Template(_get_prompt("find_core")).substitute(
        intervention=intervention_text, problem=problem_text, aspects=aspects,
    )
    with track_llm_call("find_core", POLICY_NAME, "anthropic", "claude-sonnet-5-5") as call:
        with _client.messages.stream(
            model="claude-sonnet-5-5",
            max_tokens=32000,
            thinking={"type": "adaptive", "display": "summarized"},
            system=_system_prompt,
            messages=[{"role": "user", "content": "<document>" + _labelled_document() + "</document>" + "<task>" + prompt + "</task>"}],
            output_config={
                "format": {
                "type": "json_schema",
                "schema": schema,
            }}
            ) as stream:
            response = stream.get_final_message()
        call.set_usage_from_response(response)
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"find_core: response truncated (stop_reason=max_tokens, "
            f"output_tokens={call.record.output_tokens}); raise max_tokens"
        )
    data = json.loads(next(b.text for b in reversed(response.content) if b.type == "text"))
    with open(RESULTS_DIR / "core_themes.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return call.record


def call_llm_classify_with_core(aspect_name: str, core_themes: list[dict], model: str = "claude-haiku-5-5"):
    """Group an aspect's sentences into themes, filling the given core themes
    (from find_core) first.  Writes classification.json and, since core is
    already decided, weights.json directly: the named core themes are core,
    every other theme is not.  Returns the call record and a list of problems
    (core themes missing from the output or left with no sentences)."""
    schema = Themes.model_json_schema()
    with open(result_file(aspect_name, "text.txt"), "r") as f:
        aspect_text = f.read()
    aspect_ids = set(re.findall(r"^(S\d{4,})", aspect_text, flags=re.M))
    intervention_text, _ = _anchor_texts()
    prompt = Template(_get_prompt("classify_with_core")).substitute(
        intervention=intervention_text,
        aspect=ASPECT_DESCRIPTIONS[aspect_name],
        core_themes="\n".join(f"- {t['name']}: {t['claim']}" for t in core_themes),
    )
    with track_llm_call(f"classify_with_core:{aspect_name}", POLICY_NAME, "anthropic", model) as call:
        with _client.messages.stream(
            max_tokens=64000,
            model=model,
            system=_system_prompt,
            messages=[{"role": "user", "content": "<document>" + aspect_text + "</document>" + "<task>" + prompt + "</task>"}],
            output_config={
                "effort": "low",  # mechanical grouping; Haiku 5.5 thinks by default
                "format": {
                "type": "json_schema",
                "schema": schema,
            }}
            ) as stream:
            response = stream.get_final_message()
        call.set_usage_from_response(response)
    _raise_if_refused(response, f"classify_with_core:{aspect_name}")
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"classify_with_core:{aspect_name}: response truncated (stop_reason=max_tokens, "
            f"output_tokens={call.record.output_tokens}); raise max_tokens"
        )
    data = json.loads(next(b.text for b in reversed(response.content) if b.type == "text"))
    with open(result_file(aspect_name, "classification.json"), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    core_names = [t["name"] for t in core_themes]
    returned = {t["name"]: [sid for sid in t["ids"] if sid in aspect_ids] for t in data["themes"]}
    problems = [f"core theme missing: {name}" for name in core_names if name not in returned]
    problems += [f"core theme empty: {name}" for name in core_names if name in returned and not returned[name]]
    weights = {"themes": [
        {"theme": t["name"],
         "reason": "Core theme named by find_core." if t["name"] in core_names else "Not a core theme.",
         "weight": "core" if t["name"] in core_names else "not core"}
        for t in data["themes"]
    ]}
    with open(result_file(aspect_name, "weights.json"), "w") as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)
    return call.record, problems


def call_llm_core_lines(aspect_name: str):
    """Experimental alternative to grouping then weighting: name the one to
    three lines of argument the aspect rests on, straight from its sentences,
    with citations removed so the choice cannot follow the evidence.  Writes
    core.json; ids outside the aspect are dropped and counted."""
    schema = CoreLines.model_json_schema()
    with open(result_file(aspect_name, "text.txt"), "r") as f:
        aspect_text = _strip_citations(f.read())
    aspect_ids = set(re.findall(r"^(S\d{4,})", aspect_text, flags=re.M))
    intervention_text, problem_text = _anchor_texts()
    prompt = Template(_get_prompt("weight_core")).substitute(
        intervention=intervention_text,
        problem=problem_text,
        aspect=ASPECT_DESCRIPTIONS[aspect_name],
        should_explain=_get_prompt(f"explain_{aspect_name}").rstrip("\n"),
    )
    with track_llm_call(f"core_lines:{aspect_name}", POLICY_NAME, "anthropic", "claude-sonnet-5-5") as call:
        with _client.messages.stream(
            model="claude-sonnet-5-5",
            max_tokens=32000,
            thinking={"type": "adaptive", "display": "summarized"},
            system=_system_prompt,
            messages=[{"role": "user", "content": "<document>" + aspect_text + "</document>" + "<task>" + prompt + "</task>"}],
            output_config={
                "format": {
                "type": "json_schema",
                "schema": schema,
            }}
            ) as stream:
            response = stream.get_final_message()
        call.set_usage_from_response(response)
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"core_lines:{aspect_name}: response truncated (stop_reason=max_tokens, "
            f"output_tokens={call.record.output_tokens}); raise max_tokens"
        )
    data = json.loads(next(b.text for b in reversed(response.content) if b.type == "text"))
    outside = 0
    for line in data["core_lines"]:
        kept = [sid for sid in line["ids"] if sid in aspect_ids]
        outside += len(line["ids"]) - len(kept)
        line["ids"] = kept
    data["ids_outside_aspect"] = outside
    with open(result_file(aspect_name, "core.json"), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return call.record


def _load_weights(aspect_name: str) -> dict[str, str]:
    path = result_file(aspect_name, "weights.json")
    if not path.exists():
        return {}
    with open(path) as f:
        return {t["theme"]: t["weight"] for t in json.load(f)["themes"]}


def section_score(theme_scores: list[dict], weights: dict[str, str]) -> dict:
    """The section score: the mean of the core themes' levels, rounded to the
    nearest level with an exact half rounding down (1.5 -> 1).

    Core/not core comes from weights (weights.json), never from the scorer.
    With no core theme the mean is over all themes.  A core theme the scorer
    gave no level is reported in missing_core and left out of the mean."""
    scores = {t["theme"]: int(t["score"]) for t in theme_scores}
    core = [name for name, weight in weights.items() if weight == ThemeWeight.core.value]
    missing_core = [name for name in core if name not in scores]
    used = [name for name in core if name in scores] or list(scores)
    if not used:
        return {"core_themes": [], "core_mean": None, "score": None, "missing_core": missing_core}
    mean = round(sum(scores[name] for name in used) / len(used), 6)
    return {"core_themes": used, "core_mean": mean,
            "score": math.ceil(mean - 0.5), "missing_core": missing_core}


# Aspects scored as a single block of sentences, with no themes or weights.
# Testing's rubric is about success measures and plans, not an argument with
# lines of support, and there is usually little of it.
BLOCK_ASPECTS = ("testing",)

NO_BLOCK_REASONING = ("No sentences in the document were classified as {label}, so there is no "
                      "indication of success measures, testing or evaluation.")


def _write_single_theme(aspect_name: str) -> None:
    """classification.json with every sentence of the aspect in one theme, so
    the viewer and report show the block as a single theme."""
    name = ASPECT_DESCRIPTIONS[aspect_name].split(":")[0].capitalize()
    with open(result_file(aspect_name, "classification.json"), "w") as f:
        json.dump({"themes": [{"name": name, "ids": _aspect_sentence_ids(aspect_name)}]}, f, indent=2)


def _block_input(aspect_name: str) -> str | None:
    """The aspect's sentences with their citations, as one <text>/<references>/
    <links> block with no theme structure.  Built beside summary.xml, which is
    left alone.  None when the aspect holds no sentences."""
    coverage = construct_xml(DOCUMENTS_DIR, RESULTS_DIR, aspect_name,
                             classification={"themes": []}, out_name="block.xml")
    if not coverage["total"]:
        return None
    with open(result_file(aspect_name, "block.xml"), "r") as f:
        xml = unescape(f.read(), {"&quot;": '"', "&apos;": "'"})
    inner = re.search(r"<ungrouped[^>]*>\n(.*)</ungrouped>", xml, flags=re.S).group(1)
    return re.sub(r'<line id="[^"]*">', "<line>", inner)


def _aspect_sentence_ids(aspect_name: str) -> list[str]:
    path = result_file(aspect_name, "text.txt")
    if not path.exists():
        return []
    return re.findall(r"^(S\d{4,})", path.read_text(), flags=re.M)


def _write_empty_score(aspect_name: str) -> None:
    """An aspect with no sentences scores 0 without a call."""
    label = ASPECT_DESCRIPTIONS[aspect_name].split(":")[0].lower().removeprefix("the ")
    with open(result_file(aspect_name, "scoring.json"), "w") as f:
        json.dump({"score": 0, "reasoning": NO_BLOCK_REASONING.format(label=label), "block": True}, f, indent=2)


def call_llm_score_block(aspect_name: str):
    """Score an aspect as one block.  An aspect with no sentences scores 0
    without a call."""
    block = _block_input(aspect_name)
    if block is None:
        _write_empty_score(aspect_name)
        return None
    schema = BlockScore.model_json_schema()
    template = Template(_get_prompt(f"score_{aspect_name}") + "\n" + _get_prompt("score_block"))
    intervention_text, problem_text = _anchor_texts()
    prompt = template.substitute(
        intervention=intervention_text,
        problem=problem_text,
        should_explain=_get_prompt(f"explain_{aspect_name}").rstrip("\n"),
    )
    with track_llm_call(f"score:{aspect_name}", POLICY_NAME, "anthropic", "claude-sonnet-5-5") as call:
        response = _client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=20000,
            thinking={"type": "adaptive", "display": "summarized"},
            system=_system_prompt,
            messages=[{"role": "user", "content": "<document>" + block + "</document>" + "<task>" + prompt + "</task>"}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        call.set_usage_from_response(response)
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"score:{aspect_name}: response truncated (stop_reason=max_tokens, "
            f"output_tokens={call.record.output_tokens}); raise max_tokens"
        )
    result = json.loads(next(b.text for b in reversed(response.content) if b.type == "text"))
    result["block"] = True
    with open(result_file(aspect_name, "scoring.json"), "w") as f:
        json.dump(result, f, indent=2)
    return call.record


def call_llm_score(aspect_name: str):
    if aspect_name in BLOCK_ASPECTS:
        return call_llm_score_block(aspect_name)
    schema = Score.model_json_schema()
    xml = _read_summary_xml(aspect_name)
    weights = _load_weights(aspect_name)
    # The aspect prompt supplies the role, load-bearing claims and rubric; the
    # shared block supplies the input format, aggregation rule and output spec
    # that are identical across aspects.
    template = Template(_get_prompt(f"score_{aspect_name}") + "\n" + _get_prompt("score_common"))
    intervention_text, problem_text = _anchor_texts()
    prompt = template.substitute(
        intervention=intervention_text,
        problem=problem_text,
        should_explain=_get_prompt(f"explain_{aspect_name}").rstrip("\n"),
    )
    with track_llm_call(f"score:{aspect_name}", POLICY_NAME, "anthropic", "claude-sonnet-5-5") as call:
        response = _client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=20000,
            thinking={"type": "adaptive", "display": "summarized"},
            system=_system_prompt,
            messages=[{"role": "user", "content": "<document>" + xml + "</document>" + "<task>" + prompt + "</task>"}],
            output_config={
                "format": {
                "type": "json_schema",
                "schema": schema,
            }}
            )
        call.set_usage_from_response(response)
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"score:{aspect_name}: response truncated (stop_reason=max_tokens, "
            f"output_tokens={call.record.output_tokens}); raise max_tokens"
        )
    result = json.loads(next(b.text for b in reversed(response.content) if b.type == "text"))
    # The scorer applies the rule itself so its reasoning can state the score;
    # the code recomputes it from the theme levels and has the final say.  A
    # disagreement is an arithmetic slip, recorded rather than raised since the
    # call is already paid for.
    check = section_score(result["theme_scores"], weights)
    result["model_score"] = result["score"]
    result["core_mean"] = check["core_mean"]
    if check["score"] is not None:
        result["score"] = check["score"]
    result["score_consistent"] = result["model_score"] == result["score"]
    result["missing_core"] = check["missing_core"]
    if not result["score_consistent"] or check["missing_core"]:
        print(f"WARNING score:{POLICY_NAME}/{aspect_name}: scorer said {result['model_score']}, "
              f"core mean {check['core_mean']} gives {result['score']}; "
              f"core themes with no level: {check['missing_core']}", flush=True)
    with open(result_file(aspect_name, "scoring.json"), "w") as f:
        json.dump(result, f, indent=2)
    return call.record



# def _load_lines(aspect: str) -> list[str]:
#     for filename in ("lines_2.txt", "lines.txt"):
#         path = result_file(aspect, filename)
#         if path.exists():
#             return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
#     return []



def _step_event(label: str, record) -> dict:
    event = {"label": label, "status": "Success"}
    if record is not None:
        event["usage"] = record.model_dump(mode="json")
    return event


def run_analysis():
    """Generate missing result artefacts, yielding each workflow step as it completes."""
    records = []
    _ = get_client()
    _ = _set_system_prompt()
    _set_up_dirs()

    extract_records = call_llm_extract_lines()
    records.extend(extract_records)
    total_chunks = len(extract_records)
    for index, record in enumerate(extract_records, start=1):
        label = "Sentence extraction" if total_chunks == 1 else f"Sentence extraction (chunk {index}/{total_chunks})"
        yield _step_event(label, record)

    # Core themes come first, from the whole document and without sight of the
    # evidence; the grouping then fills them, and the scorer judges each core
    # claim wherever its evidence sits.  Testing is scored as one block.
    record = call_llm_find_core()
    records.append(record)
    yield _step_event("Finding core themes", record)
    core = json.loads((RESULTS_DIR / "core_themes.json").read_text())

    for aspect in CORE_ASPECTS:
        if not _aspect_sentence_ids(aspect):
            construct_xml(DOCUMENTS_DIR, RESULTS_DIR, aspect, classification={"themes": []})
            _write_empty_score(aspect)
            yield _step_event(f"Scoring for {aspect} (no sentences, scored 0)", None)
            continue

        themes = core[aspect]["core_themes"]
        label = f"Grouping for {aspect}"
        if themes:
            record, problems = call_llm_classify_with_core(aspect, themes)
            records.append(record)
            if problems:
                label += f" (PROBLEMS: {'; '.join(problems)})"
        else:
            # The aspect holds sentences but no argument to name core themes
            # for: score it as one block, as Testing is.
            _write_single_theme(aspect)
            result_file(aspect, "weights.json").unlink(missing_ok=True)
            record = None
        # construct_xml reports how many of the aspect's sentences the grouping
        # call left out of every theme; they are still scored (via <ungrouped>),
        # but the number is worth seeing.
        coverage = construct_xml(DOCUMENTS_DIR, RESULTS_DIR, aspect)
        if coverage["ungrouped"]:
            label += f" ({coverage['ungrouped']} of {coverage['total']} sentences ungrouped)"
        yield _step_event(label, record)

        record = call_llm_score_block(aspect) if not themes else call_llm_score(aspect)
        if record is not None:
            records.append(record)
        yield _step_event(f"Scoring for {aspect}", record)

    for aspect in BLOCK_ASPECTS:
        if _aspect_sentence_ids(aspect):
            _write_single_theme(aspect)
            construct_xml(DOCUMENTS_DIR, RESULTS_DIR, aspect)
        else:
            construct_xml(DOCUMENTS_DIR, RESULTS_DIR, aspect, classification={"themes": []})
        record = call_llm_score_block(aspect)
        if record is not None:
            records.append(record)
        yield _step_event(f"Scoring for {aspect}" + ("" if record else " (no sentences, scored 0)"), record)

    write_run_summary(POLICY_NAME, records)


@analyse_bp.route("/analyse")
def analyse_page():
    policies = available_policies()
    selected_policy = most_recently_uploaded_policy(policies) or (POLICY_NAME if POLICY_NAME in policies else (policies[0] if policies else ""))
    if selected_policy:
        set_policy_name(selected_policy)
    return render_template("analyse.jinja", policies=policies, selected_policy=selected_policy)


@analyse_bp.route("/analyse/run", methods=["POST"])
def analyse_run():
    data = request.get_json(silent=True) or {}
    try:
        set_policy_name(str(data.get("policy_name", POLICY_NAME)))
    except ValueError as error:
        return Response(json.dumps({"status": "error", "error": str(error)}) + "\n", status=400, mimetype="application/x-ndjson")

    def generate():
        try:
            for step in run_analysis():
                yield json.dumps(step) + "\n"
            yield json.dumps({"status": "complete"}) + "\n"
        except Exception as error:
            current_app.logger.exception("Analysis failed")
            yield json.dumps({"status": "error", "error": str(error)}) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")
