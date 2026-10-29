"""Combine several already-ingested documents into one merged document.

A policy is often argued across more than one publication -- a consultation
alongside its impact assessment, say -- and the evidence for a claim made in one
may only be cited in the other.  Combining them lets the pipeline treat the pair
as a single body of argument: one stream of ``S###`` sentences and one shared
reference list.

Merging happens over the *persisted artefacts* of the sources (processed.md,
references.json, links.json), not their originals, so nothing is re-extracted
and no LLM or OCR call is made.  Two things have to be renumbered on the way in,
because every source numbers them from 1:

* sentences -- ``S###`` ids continue across the whole combined document;
* references and links -- renumbered into one list, with every in-text
  ``(Reference: N)`` / ``(Link: N)`` marker rewritten to match.

A citation whose definition is missing from its source's references.json still
gets an entry in the combined list (flagged ``missing``), so the marker keeps
pointing at its own source rather than silently landing on another document's
reference.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_ids import format_sentence_id, format_sentence_line, split_sentence_line

# In-text markers, as written by preprocess.py and web_parser.py:
#   "** (Reference: 3, 4)" at the end of a sentence, and
#   "**(Link: 7)" inline, right after the linked phrase.
_REFERENCE_MARKER_RE = re.compile(r'(Reference:\s*)([0-9a-zA-Z_\-]+(?:\s*,\s*[0-9a-zA-Z_\-]+)*)')
_LINK_MARKER_RE = re.compile(r'(Link:\s*)([0-9a-zA-Z_\-]+(?:\s*,\s*[0-9a-zA-Z_\-]+)*)')


class CombineError(ValueError):
    """A source document can't be combined (missing or unreadable artefacts)."""


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    """Read a references.json/links.json-style list, tolerating an absent file."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CombineError(f"{path.parent.name}/{path.name} is not valid JSON: {exc}") from exc

    if isinstance(data, dict):  # legacy {number: entry} shape
        data = [
            {**entry, "number": str(number)} if isinstance(entry, dict) else {"number": str(number), "text": str(entry)}
            for number, entry in data.items()
        ]
    if not isinstance(data, list):
        raise CombineError(f"{path.parent.name}/{path.name} should contain a list of entries")
    return [entry for entry in data if isinstance(entry, dict) and entry.get("number") is not None]


def source_title(document_dir: Path, alias: str) -> str:
    """The source's display title from metadata.json, falling back to its alias."""
    metadata_file = document_dir / "metadata.json"
    if metadata_file.exists():
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = None
        if isinstance(metadata, dict) and str(metadata.get("title", "")).strip():
            return str(metadata["title"]).strip()
    return alias.replace("_", " ").title()


def _renumber_entries(
    entries: List[Dict[str, Any]],
    offset: int,
    source: Dict[str, str],
) -> tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Give each entry a new number continuing from `offset`.

    Returns the old-number -> new-number map and the renumbered entries, each
    keeping a note of where it came from.
    """
    number_map: Dict[str, str] = {}
    renumbered: List[Dict[str, Any]] = []
    for entry in entries:
        old_number = str(entry["number"])
        if old_number in number_map:  # duplicate definition; first one wins
            continue
        new_number = str(offset + len(renumbered) + 1)
        number_map[old_number] = new_number
        renumbered.append({
            **entry,
            "number": new_number,
            "source_alias": source["alias"],
            "source_title": source["title"],
            "source_number": old_number,
        })
    return number_map, renumbered


def _remap_marker_numbers(
    body: str,
    pattern: re.Pattern,
    number_map: Dict[str, str],
    on_unmapped,
) -> str:
    """Rewrite the numbers inside one kind of in-text marker."""
    def replace(match: re.Match) -> str:
        numbers = [number.strip() for number in match.group(2).split(",")]
        mapped = [number_map.get(number) or on_unmapped(number) for number in numbers]
        return match.group(1) + ", ".join(mapped)

    return pattern.sub(replace, body)


def combine_documents(sources: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    """Merge `sources` into one document written to `out_dir`.

    Each source is a dict with 'alias', 'title', and 'dir' (its documents/
    folder).  Sources are merged in the order given.  Writes processed.md,
    references.json, links.json, and input.txt, and returns a stats dict
    describing the result (including a per-source breakdown of which sentence
    and reference ranges came from where).
    """
    if len(sources) < 2:
        raise CombineError("Combining needs at least two documents")

    processed_lines: List[str] = []
    references: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []
    source_stats: List[Dict[str, Any]] = []
    sentence_index = 0

    for source in sources:
        source_dir = Path(source["dir"])
        processed_file = source_dir / "processed.md"
        if not processed_file.exists():
            raise CombineError(f"'{source['alias']}' has no processed.md to combine")

        reference_map, source_references = _renumber_entries(
            _read_json_list(source_dir / "references.json"), len(references), source
        )
        link_map, source_links = _renumber_entries(
            _read_json_list(source_dir / "links.json"), len(links), source
        )
        references.extend(source_references)
        links.extend(source_links)

        # A citation with no definition in its source gets a placeholder entry
        # rather than keeping its original number, which would now point at
        # some other source's reference.
        def placeholder(old_number: str, source=source, references=references, reference_map=reference_map) -> str:
            new_number = str(len(references) + 1)
            reference_map[old_number] = new_number
            references.append({
                "number": new_number,
                "text": (
                    f"Reference #{old_number} in '{source['title']}' "
                    "was cited in the text but has no definition in that document."
                ),
                "source_alias": source["alias"],
                "source_title": source["title"],
                "source_number": old_number,
                "missing": True,
            })
            return new_number

        first_sentence_index = sentence_index + 1
        first_reference_number = len(references) - len(source_references) + 1

        # A heading names each source in the combined text, so a reader (and the
        # classifier) can see where one document ends and the next begins.
        processed_lines.append(f"# {source['title']}")

        with open(processed_file, encoding="utf-8") as f:
            for line in f:
                sid, body = split_sentence_line(line)
                if sid is None:
                    processed_lines.append(body)
                    continue
                body = _remap_marker_numbers(body, _REFERENCE_MARKER_RE, reference_map, placeholder)
                body = _remap_marker_numbers(
                    body, _LINK_MARKER_RE, link_map, lambda number: link_map.get(number, number)
                )
                sentence_index += 1
                processed_lines.append(format_sentence_line(sentence_index, body))

        sentence_count = sentence_index - first_sentence_index + 1
        reference_count = len(references) - first_reference_number + 1
        source_stats.append({
            "alias": source["alias"],
            "title": source["title"],
            "sentence_count": sentence_count,
            "first_sentence": format_sentence_id(first_sentence_index) if sentence_count else None,
            "last_sentence": format_sentence_id(sentence_index) if sentence_count else None,
            "reference_count": reference_count,
            "first_reference": str(first_reference_number) if reference_count else None,
            "last_reference": str(len(references)) if reference_count else None,
            "link_count": len(source_links),
        })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "processed.md").write_text("\n".join(processed_lines) + "\n", encoding="utf-8")
    (out_dir / "references.json").write_text(
        json.dumps(references, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "links.json").write_text(
        json.dumps(links, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    input_lines = [line for line in processed_lines if split_sentence_line(line)[0] is not None]
    (out_dir / "input.txt").write_text("\n".join(input_lines) + "\n", encoding="utf-8")

    return {
        "sentence_count": sentence_index,
        "reference_count": len(references),
        "link_count": len(links),
        "missing_reference_count": sum(1 for entry in references if entry.get("missing")),
        "sources": source_stats,
    }


def collect_sources(documents_dir: Path, aliases: List[str]) -> List[Dict[str, Any]]:
    """Resolve source aliases to {'alias', 'title', 'dir'} dicts, in order."""
    sources = []
    for alias in aliases:
        source_dir = Path(documents_dir) / alias
        if not source_dir.is_dir():
            raise CombineError(f"No document found with alias '{alias}'")
        sources.append({"alias": alias, "title": source_title(source_dir, alias), "dir": source_dir})
    return sources


def combinable_documents(documents_dir: Path) -> List[Dict[str, Optional[str]]]:
    """List the documents that have the artefacts needed to be combined."""
    documents_dir = Path(documents_dir)
    if not documents_dir.is_dir():
        return []

    documents = []
    for source_dir in sorted(documents_dir.iterdir()):
        if not source_dir.is_dir() or not (source_dir / "processed.md").exists():
            continue
        input_file = source_dir / "input.txt"
        sentence_count = 0
        if input_file.exists():
            with open(input_file, encoding="utf-8") as f:
                sentence_count = sum(1 for line in f if line.strip())
        documents.append({
            "alias": source_dir.name,
            "title": source_title(source_dir, source_dir.name),
            "sentence_count": sentence_count,
            "reference_count": len(_read_json_list(source_dir / "references.json")),
        })
    return documents
