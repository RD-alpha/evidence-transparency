"""Generate a print-ready HTML evidence report from analysis results.

Reads the persisted artefacts produced by analyse.py (summary.xml and
scoring.json for each policy aspect) and renders a single self-contained
HTML file laid out for reading on screen or printing/exporting to PDF.

Usage:
    python generate_report.py <policy_alias> [-o OUTPUT.html] [--summary]
    python generate_report.py --all [--summary]

--summary produces a shorter report with just theme headings, scores and
reasoning (no extracted document text, references or links).

Does not call any LLM or web service; it only reads results/<policy>/...
"""

from __future__ import annotations

import argparse
import html
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
DOCUMENTS_ROOT = PROJECT_ROOT / "documents"

ASPECTS = ("diagnosis", "proposal", "implementation", "testing")
ASPECT_LABELS = {
    "diagnosis": "Diagnosis",
    "proposal": "Proposal",
    "implementation": "Implementation",
    "testing": "Testing + Evaluation",
}
SCORE_RUBRIC = {
    0: "No evidence provided",
    1: "Relevant evidence is mentioned",
    2: "Evidence properly cited & chain of reasoning is clear",
    3: "Strengths and weaknesses of evidence assessed",
}
TESTING_SCORE_RUBRIC = {
    0: "No discussion of testing or evaluation",
    1: "Some indication of evaluation",
    2: "Details provided for testing and evaluation",
    3: "Detailed plans for testing, evaluation and publication of results",
}
SCORE_RUBRICS_BY_ASPECT = {"testing": TESTING_SCORE_RUBRIC}


def get_score_rubric(aspect: str) -> dict:
    return SCORE_RUBRICS_BY_ASPECT.get(aspect, SCORE_RUBRIC)

REF_MARKER_RE = re.compile(r"\*\*\(Reference:\s*([0-9a-zA-Z_\-,\s]+?)\)")
LINK_MARKER_RE = re.compile(r"\*\*\(Link:\s*([0-9a-zA-Z_\-,\s]+?)\)")


# ---------------------------------------------------------------------------
# Parsing (summary.xml is well-formed XML: a single <root> of <theme>
# elements, each with <text><line id="..."> children, <reference number="">
# and <link number="" url=""> children).
# ---------------------------------------------------------------------------

def parse_summary_xml(path: Path) -> list[dict]:
    if not path.exists():
        return []
    root = ET.parse(path).getroot()
    themes = []
    # <ungrouped> holds the aspect's sentences that theme grouping missed; it is
    # rendered last, like a theme but without a name of its own.
    for theme_el in list(root.findall("theme")) + list(root.findall("ungrouped")):
        if theme_el.tag == "ungrouped":
            name = "Ungrouped sentences"
        else:
            name = (theme_el.findtext("name") or "").strip() or "Untitled theme"

        text_el = theme_el.find("text")
        paragraphs = [(line.text or "").strip() for line in (text_el.findall("line") if text_el is not None else [])]
        paragraphs = [p for p in paragraphs if p]

        references = {}
        refs_el = theme_el.find("references")
        if refs_el is not None:
            for ref_el in refs_el.findall("reference"):
                number = ref_el.get("number", "")
                references[number] = (ref_el.text or "").strip()

        links = {}
        links_el = theme_el.find("links")
        if links_el is not None:
            for link_el in links_el.findall("link"):
                number = link_el.get("number", "")
                links[number] = {"label": (link_el.text or "").strip(), "url": link_el.get("url", "")}

        themes.append({"name": name, "paragraphs": paragraphs, "references": references, "links": links})
    return themes


def load_scoring(path: Path) -> dict:
    if not path.exists():
        return {"score": None, "reasoning": "No assessment report available for this section."}
    data = json.loads(path.read_text(encoding="utf-8"))
    score = data.get("score")
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = None
    # theme_scores was added after some results were generated, so older
    # scoring.json files simply have none.
    theme_scores = []
    for entry in data.get("theme_scores") or []:
        try:
            theme_score = int(entry.get("score"))
        except (TypeError, ValueError, AttributeError):
            continue
        theme_scores.append({
            "theme": str(entry.get("theme", "")).strip(),
            "score": theme_score,
            "justification": str(entry.get("justification", "")).strip(),
        })
    return {
        "score": score,
        "reasoning": data.get("reasoning", "").strip(),
        "theme_scores": theme_scores,
    }


def load_weights(path: Path) -> dict[str, list[str]]:
    """Theme weights (core / not core) by theme name, from weights.json.
    Results scored before themes were weighted have no file, and their
    Weight cells are left blank.  A list per name keeps duplicate theme
    names apart, in order."""
    if not path.exists():
        return {}
    weights: dict[str, list[str]] = {}
    for entry in json.loads(path.read_text(encoding="utf-8")).get("themes") or []:
        weights.setdefault(str(entry.get("theme", "")).strip(), []).append(str(entry.get("weight", "")))
    return weights


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def linkify_paragraph(paragraph: str, aspect: str, theme_idx: int) -> str:
    escaped = html.escape(paragraph)

    def ref_repl(match: re.Match) -> str:
        numbers = [n.strip() for n in match.group(1).split(",") if n.strip()]
        anchors = ", ".join(
            f'<a href="#ref-{aspect}-{theme_idx}-{n}" class="citation">{html.escape(n)}</a>' for n in numbers
        )
        return f'<sup class="cite">[{anchors}]</sup>'

    def link_repl(match: re.Match) -> str:
        numbers = [n.strip() for n in match.group(1).split(",") if n.strip()]
        anchors = ", ".join(
            f'<a href="#link-{aspect}-{theme_idx}-{n}" class="citation link">{html.escape(n)}</a>' for n in numbers
        )
        return f'<sup class="cite">[{anchors}]</sup>'

    escaped = REF_MARKER_RE.sub(ref_repl, escaped)
    escaped = LINK_MARKER_RE.sub(link_repl, escaped)
    return escaped


def render_theme(theme: dict, aspect: str, theme_idx: int) -> str:
    paragraphs_html = "\n".join(
        f"<p>{linkify_paragraph(p, aspect, theme_idx)}</p>" for p in theme["paragraphs"]
    )

    refs_html = ""
    if theme["references"]:
        items = "\n".join(
            f'<li id="ref-{aspect}-{theme_idx}-{html.escape(num)}">'
            f'<span class="ref-number">{html.escape(num)}.</span> {html.escape(text)}</li>'
            for num, text in theme["references"].items()
        )
        refs_html = f'<div class="refs"><h5>References</h5><ol class="ref-list">{items}</ol></div>'

    links_html = ""
    if theme["links"]:
        items = []
        for num, entry in theme["links"].items():
            label = html.escape(entry["label"])
            url = html.escape(entry["url"])
            target = f'<a href="{url}">{url}</a>' if url else "(no URL)"
            items.append(
                f'<li id="link-{aspect}-{theme_idx}-{html.escape(num)}">'
                f'<span class="ref-number">{html.escape(num)}.</span> {label} &mdash; {target}</li>'
            )
        links_html = f'<div class="links"><h5>Links</h5><ol class="ref-list">{"".join(items)}</ol></div>'

    return f"""
    <div class="theme">
      <h4>{html.escape(theme["name"])}</h4>
      {paragraphs_html}
      {refs_html}
      {links_html}
    </div>
    """


def render_theme_heading(theme: dict) -> str:
    return f"<li>{html.escape(theme['name'])}</li>"


def render_aspect(policy_dir: Path, aspect: str, summary_only: bool = False) -> tuple[str, dict]:
    scoring = load_scoring(policy_dir / aspect / "scoring.json")
    themes = parse_summary_xml(policy_dir / aspect / "summary.xml")

    score = scoring["score"]
    score_class = f"score-{score}" if score is not None else "score-none"
    score_label = f"{score} / 3" if score is not None else "N/A"
    rubric_label = get_score_rubric(aspect).get(score, "Not yet assessed")

    reasoning_html = html.escape(scoring["reasoning"]).replace("\n", "<br>") or "No reasoning provided."

    theme_scores_html = ""
    if scoring["theme_scores"]:
        weights = load_weights(policy_dir / aspect / "weights.json")

        def weight_of(theme: str) -> str:
            matches = weights.get(theme)
            return matches.pop(0) if matches else ""

        rows = "\n".join(
            f'<tr><td>{html.escape(entry["theme"])}</td>'
            f'<td>{html.escape(weight_of(entry["theme"]))}</td>'
            f'<td><span class="score-badge {"score-%d" % entry["score"]}">{entry["score"]} / 3</span></td>'
            f'<td>{html.escape(entry["justification"])}</td></tr>'
            for entry in scoring["theme_scores"]
        )
        theme_scores_html = (
            '<div class="theme-scores"><h3>Score by theme</h3>'
            '<table class="theme-score-table">'
            "<thead><tr><th>Theme</th><th>Weight</th><th>Level</th><th>Basis</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    if summary_only:
        if themes:
            themes_html = (
                '<ul class="theme-list">'
                + "\n".join(render_theme_heading(theme) for theme in themes)
                + "</ul>"
            )
        else:
            themes_html = '<p class="empty-note">No themes were extracted for this section.</p>'
        themes_heading = "Themes covered"
    else:
        if themes:
            themes_html = "\n".join(render_theme(theme, aspect, idx) for idx, theme in enumerate(themes))
        else:
            themes_html = '<p class="empty-note">No supporting evidence was extracted for this section.</p>'
        themes_heading = "Themes and supporting evidence"

    section = f"""
    <section class="aspect" id="aspect-{aspect}">
      <h2>{html.escape(ASPECT_LABELS[aspect])}</h2>
      <div class="score-block">
        <span class="score-badge {score_class}">{score_label}</span>
        <span class="rubric-label">{html.escape(rubric_label)}</span>
      </div>
      <div class="reasoning">
        <h3>Assessment reasoning</h3>
        <p>{reasoning_html}</p>
      </div>
      {"" if summary_only else theme_scores_html}
      <h3>{themes_heading}</h3>
      {themes_html}
    </section>
    """
    summary_row = {
        "aspect": aspect,
        "label": ASPECT_LABELS[aspect],
        "score_label": score_label,
        "score_class": score_class,
        "rubric_label": rubric_label,
    }
    return section, summary_row


def load_document_title(alias: str) -> tuple[str, str]:
    metadata_path = DOCUMENTS_ROOT / alias / "metadata.json"
    title = alias.replace("_", " ").title()
    source_url = ""
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            title = metadata.get("title", title) or title
            source_url = metadata.get("source_url", "") or ""
        except json.JSONDecodeError:
            pass
    return title, source_url


CSS = """
:root {
  --ink: #1a1a1a;
  --muted: #5a5a5a;
  --border: #d8d8d8;
  --bg-page: #ffffff;
  --score-0: #b3261e;
  --score-1: #b06000;
  --score-2: #1a5fb4;
  --score-3: #1c7a3c;
  --score-none: #8a8a8a;
}
* { box-sizing: border-box; }
body {
  font-family: Georgia, 'Times New Roman', serif;
  color: var(--ink);
  background: var(--bg-page);
  max-width: 860px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 4rem;
  line-height: 1.55;
  font-size: 15px;
}
h1, h2, h3, h4, h5 {
  font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif;
  color: #111;
  line-height: 1.25;
}
header.cover {
  border-bottom: 3px solid var(--ink);
  padding-bottom: 1.5rem;
  margin-bottom: 2rem;
}
header.cover h1 { font-size: 1.9rem; margin: 0 0 0.25rem; }
header.cover .subtitle { font-size: 1.05rem; color: var(--muted); margin: 0 0 1rem; }
header.cover .intervention { font-size: 1rem; margin: 0.5rem 0; }
header.cover .meta { font-size: 0.85rem; color: var(--muted); }
header.cover .meta a { color: var(--muted); }

.summary-table { margin-bottom: 2.5rem; }
.summary-table table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.summary-table th, .summary-table td {
  text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border);
}
.summary-table th { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.03em; color: var(--muted); }
.summary-table a { color: inherit; text-decoration: none; }
.summary-table tr td:first-child a { font-weight: 600; }

section.aspect {
  margin-top: 3rem;
  padding-top: 0.5rem;
  border-top: 1px solid var(--border);
}
section.aspect h2 { font-size: 1.5rem; margin-bottom: 0.75rem; }

.score-block { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.25rem; }
.score-badge {
  display: inline-block; font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif;
  font-weight: 700; font-size: 0.95rem; color: #fff; padding: 0.25rem 0.7rem; border-radius: 4px;
}
.score-0 { background: var(--score-0); }
.score-1 { background: var(--score-1); }
.score-2 { background: var(--score-2); }
.score-3 { background: var(--score-3); }
.score-none { background: var(--score-none); }
.rubric-label { font-size: 0.9rem; color: var(--muted); font-style: italic; }

.theme-scores { margin-bottom: 1.5rem; }
.theme-scores h3 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.03em; color: var(--muted); }
.theme-score-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.theme-score-table th, .theme-score-table td {
  text-align: left; padding: 0.45rem 0.7rem; border-bottom: 1px solid var(--border); vertical-align: top;
}
.theme-score-table th { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.03em; color: var(--muted); }
.theme-score-table td:nth-child(2) { white-space: nowrap; }

.reasoning { background: #f7f6f3; border-left: 3px solid var(--border); padding: 0.9rem 1.1rem; margin-bottom: 1.5rem; }
.reasoning h3 { margin-top: 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.03em; color: var(--muted); }
.reasoning p { margin: 0; font-size: 0.95rem; }

.theme {
  margin: 1.5rem 0;
  padding-bottom: 1rem;
  border-bottom: 1px dashed var(--border);
  break-inside: avoid;
  page-break-inside: avoid;
}
.theme h4 { font-size: 1.05rem; margin-bottom: 0.5rem; }
.theme p { margin: 0.5rem 0; }

sup.cite a.citation {
  color: var(--score-2);
  text-decoration: none;
  font-weight: 600;
}
sup.cite a.citation.link { color: var(--score-3); }

.refs, .links { margin-top: 0.75rem; font-size: 0.82rem; color: var(--muted); }
.refs h5, .links h5 { margin: 0 0 0.25rem; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.03em; }
ol.ref-list { list-style: none; margin: 0; padding: 0; }
ol.ref-list li { margin-bottom: 0.3rem; }
ol.ref-list li:target { background: #fff3c4; }
.ref-number { font-weight: 700; color: var(--ink); }
.links a { color: var(--muted); word-break: break-all; }

.empty-note { color: var(--muted); font-style: italic; }

ul.theme-list { margin: 0.5rem 0 0; padding-left: 1.3rem; }
ul.theme-list li { margin: 0.4rem 0; }

footer.report-footer {
  margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
  font-size: 0.78rem; color: var(--muted); text-align: center;
}

@media print {
  body { max-width: none; padding: 0; font-size: 12.5px; }
  header.cover { break-after: page; page-break-after: always; }
  section.aspect { break-before: page; page-break-before: always; border-top: none; padding-top: 0; margin-top: 0; }
  .theme { break-inside: avoid; page-break-inside: avoid; }
  a { color: inherit !important; text-decoration: none; }
  @page { size: A4; margin: 2cm; }
}
"""


def build_report(alias: str, summary_only: bool = False) -> str:
    policy_dir = RESULTS_ROOT / alias
    if not policy_dir.exists():
        raise FileNotFoundError(f"No results found for policy '{alias}' at {policy_dir}")

    title, source_url = load_document_title(alias)
    intervention_path = policy_dir / "intervention.txt"
    intervention = intervention_path.read_text(encoding="utf-8").strip() if intervention_path.exists() else ""

    sections = []
    summary_rows = []
    for aspect in ASPECTS:
        section_html, row = render_aspect(policy_dir, aspect, summary_only=summary_only)
        sections.append(section_html)
        summary_rows.append(row)

    summary_table_rows = "\n".join(
        f'<tr><td><a href="#aspect-{row["aspect"]}">{html.escape(row["label"])}</a></td>'
        f'<td><span class="score-badge {row["score_class"]}">{row["score_label"]}</span></td>'
        f'<td>{html.escape(row["rubric_label"])}</td></tr>'
        for row in summary_rows
    )

    source_html = f'<a href="{html.escape(source_url)}">{html.escape(source_url)}</a>' if source_url else "N/A"
    generated_at = datetime.now().strftime("%d %B %Y, %H:%M")
    subtitle = "Evidence Transparency Report — Summary" if summary_only else "Evidence Transparency Report"
    title_suffix = " — Summary" if summary_only else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html.escape(title)} — Evidence Transparency Report{html.escape(title_suffix)}</title>
<style>{CSS}</style>
</head>
<body>
  <header class="cover">
    <h1>{html.escape(title)}</h1>
    <p class="subtitle">{html.escape(subtitle)}</p>
    {f'<p class="intervention"><strong>Policy intervention:</strong> {html.escape(intervention)}</p>' if intervention else ''}
    <p class="meta">Source document: {source_html}<br>Report generated {generated_at}</p>
  </header>

  <section class="summary-table">
    <h2>Summary of Findings</h2>
    <table>
      <thead><tr><th>Section</th><th>Score</th><th>Rubric</th></tr></thead>
      <tbody>{summary_table_rows}</tbody>
    </table>
  </section>

  {"".join(sections)}

  <footer class="report-footer">
    Generated from analysis results in results/{html.escape(alias)}/
  </footer>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", nargs="?", help="Policy alias under results/ (e.g. steel_strategy)")
    parser.add_argument("--all", action="store_true", help="Generate a report for every policy under results/")
    parser.add_argument("--summary", action="store_true", help="Omit document text/references/links; headings, scores and reasoning only")
    parser.add_argument("-o", "--output", type=Path, help="Output HTML path (single-policy mode only)")
    args = parser.parse_args()

    if args.all:
        aliases = sorted(p.name for p in RESULTS_ROOT.iterdir() if p.is_dir())
    elif args.policy:
        aliases = [args.policy]
    else:
        parser.error("Provide a policy alias or pass --all")
        return

    default_filename = "report_summary.html" if args.summary else "report.html"
    for alias in aliases:
        try:
            html_content = build_report(alias, summary_only=args.summary)
        except FileNotFoundError as error:
            print(f"Skipping {alias}: {error}")
            continue
        output_path = args.output if (args.output and len(aliases) == 1) else RESULTS_ROOT / alias / default_filename
        output_path.write_text(html_content, encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
