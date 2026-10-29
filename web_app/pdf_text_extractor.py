"""Text-layer-first PDF extraction.

Born-digital PDFs already contain a full, accurate text layer. Sending them to an
OCR model is both unnecessary and risky: investigating a real document (UK Space
Strategy) showed Mistral OCR silently dropped 16 of 37 endnote definitions on a
per-page, all-or-nothing basis, and none of that text appeared anywhere in its
output — a pure OCR block-detection miss, not something fixable downstream.

This module extracts text directly from the PDF's embedded text layer using
PyMuPDF, classifying content by font size:

- Reference/footnote entries sit at a font size clearly *smaller* than the page's
  dominant body text (validated: body ~12pt, references ~10pt with a 6.5pt marker,
  vs. a body-adjacent running header at 11pt that must NOT be caught by this rule).
- Headings sit at a font size clearly *larger* than body text.
- Running headers/page numbers are filtered separately (by repetition across pages
  and digit-only content), since they're too close in size to body text to be
  separated by the size rule alone.

Only pages with negligible extractable text *and* embedded raster images are
flagged as needing OCR (a scanned page) -- a low-text page with no images is
treated as genuinely blank, not sent to OCR.
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import pymupdf

# A block's dominant font size must differ from the page's body size by at least
# this many points to be classified as a heading or a reference entry, rather than
# ordinary body text. Calibrated against a real document where body=12pt,
# references=10pt (2pt gap, correctly caught) and the running header=11pt (1pt
# gap, correctly NOT caught by this rule -- headers are filtered separately).
SIZE_GAP_THRESHOLD = 1.5

# How far below body size a block's dominant font must sit before a superscript-
# style marker inside it is trusted to mean "references section" rather than an
# inline citation marker inside an ordinary body paragraph (whose dominant size
# is body_size itself, unaffected by the one or two marker characters it contains).
MARKER_SIZE_MARGIN = 0.5

# A page is only flagged for OCR fallback when its extractable text is at or below
# this length AND it contains embedded images -- distinguishes a scanned page from
# a page that is simply blank.
MIN_TEXT_CHARS = 20

# A block is a table-of-contents entry, not real body content, if it contains a
# dot-leader run this long -- ordinary prose (including an ellipsis, 3 dots)
# never does, so this has no observed false positives.
DOT_LEADER_PATTERN = re.compile(r"\.{4,}")

# Heading text (after whitespace normalization) that marks a table-of-contents
# page whose own heading should be dropped along with its dot-leader entries.
TOC_HEADING_TEXTS = {"table of contents", "contents"}

# A body-classified block is treated as a repeated running header/footer (and
# dropped) if it's short and appears verbatim on at least this fraction of pages.
HEADER_REPETITION_THRESHOLD = 0.3
HEADER_MAX_CHARS = 80

# Two blocks-mode blocks that both start with a superscript-style reference
# marker, and sit at most this far apart vertically, are treated as one split
# footer/endnote run (see _merge_adjacent_marker_blocks): PyMuPDF's own block
# reconstruction sometimes fails to glue a marker to its own continuation text
# (e.g. the marker rendered slightly offset onto its own line), which would
# otherwise leave it as an isolated single-marker block indistinguishable from
# a lone inline citation inside an ordinary body paragraph.
MARKER_BLOCK_MERGE_GAP = 20.0

# A PyMuPDF-detected table is only trusted (and excluded from the ordinary
# block loop) if at least this fraction of its cells are non-empty. A genuine
# multi-column table has nearly all cells filled; a false-positive detection of
# a run of footnote lines has only its first column or two populated.
MIN_TABLE_FILL_RATIO = 0.7


def analyze_pdf(filepath: str) -> Dict[str, Any]:
    """Extract structured content from a PDF's text layer, page by page.

    Returns:
        {
            "page_count": int,
            "pages": [
                {
                    "blocks": [{"type": "heading"|"paragraph"|"table", "text": str}, ...],
                    "references": [{"number": str, "text": str}, ...],
                    "needs_ocr": bool,
                    "ocr_reason": Optional[str],
                },
                ...
            ],
            "reference_gaps": [int, ...],  # missing numbers in the doc-wide 1..max run
            "reference_duplicates": [int, ...],  # numbers defined more than once
            "links": [{"number": str, "url": str, "text": str}, ...],  # body hyperlinks,
                # in document order, annotated inline in paragraph blocks as
                # "**(Link: N)" -- see _text_with_superscripts.
        }
    """
    doc = pymupdf.open(filepath)
    try:
        page_dicts = [page.get_text("dict") for page in doc]
        body_size = _document_body_size(page_dicts)
        repeated_lines = _find_repeated_header_lines(page_dicts)

        pages = []
        links: List[Dict[str, str]] = []
        for page, page_dict in zip(doc, page_dicts):
            pages.append(_analyze_page(page, page_dict, body_size, repeated_lines, links))

        harvard_references = promote_harvard_references(pages)

        all_numbers = [ref["number"] for page in pages for ref in page["references"]]
        gaps = find_reference_number_gaps(all_numbers)

        return {
            "page_count": len(doc),
            "pages": pages,
            "reference_gaps": gaps["missing"],
            "reference_duplicates": gaps["duplicated"],
            "links": links,
            "harvard_references": harvard_references,
        }
    finally:
        doc.close()


def find_reference_number_gaps(numbers: List[str]) -> Dict[str, List[int]]:
    """Check whether reference numbers, as extracted doc-wide, form a contiguous
    1..max run with no repeats.

    We only caught the water-impact-assessment document's missing 21-24 (and
    later 55-56) by manually diffing `sorted(int(n) for n in references)`
    against `range(1, max+1)` -- this makes that diff a standing, automatic
    check instead of a one-off manual one.

    Only digit-only numbers are checked; a document may legitimately use
    non-numeric reference labels (e.g. author-year style), which carry no
    contiguity expectation. Returns {"missing": [...], "duplicated": [...]},
    both sorted ascending and empty when nothing is wrong.

    Deliberately not fatal: a gap can be legitimate (e.g. a citation
    deliberately renumbered/removed by the source author). Callers should
    log/warn on a non-empty result, not reject the document.
    """
    digit_numbers = [int(n) for n in numbers if n.isdigit()]
    if not digit_numbers:
        return {"missing": [], "duplicated": []}

    counts = Counter(digit_numbers)
    max_number = max(digit_numbers)
    missing = sorted(n for n in range(1, max_number + 1) if counts[n] == 0)
    duplicated = sorted(n for n, c in counts.items() if c > 1)
    return {"missing": missing, "duplicated": duplicated}


def promote_harvard_references(pages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Peel Harvard/author-date reference entries out of the ordinary paragraph
    stream, after a References/Bibliography heading.

    Some documents (e.g. DWP's "Improving Lives: Helping Workless Families")
    list references as plain 'Author(s) (Year) Title...' paragraphs, sorted
    alphabetically, with no numbering or superscript marker at all -- so
    `_classify_size`'s font-size/marker heuristics (built for numbered
    footnote systems) never fire and these paragraphs would otherwise be
    lost to the body text. Mutates `pages` in place, removing any paragraph
    block that matches once inside such a section, and returns the parsed
    reference entries (empty if the document has no such section).
    """
    try:
        # Matches how the app itself imports sibling modules (run with web_app/ on sys.path).
        from preprocess import RE_HARVARD_ENTRY_START, RE_REFERENCES_HEADING_PLAIN, parse_harvard_reference_entries
    except ImportError:
        # Matches how tests import this package from the repo root (web_app.<module>).
        from web_app.preprocess import (
            RE_HARVARD_ENTRY_START,
            RE_REFERENCES_HEADING_PLAIN,
            parse_harvard_reference_entries,
        )

    in_references = False
    matched_texts: List[str] = []
    for page in pages:
        kept_blocks = []
        for block in page["blocks"]:
            text = block["text"].strip()
            if block["type"] == "heading" and RE_REFERENCES_HEADING_PLAIN.match(text):
                in_references = True
                kept_blocks.append(block)
                continue
            if in_references and block["type"] == "paragraph" and RE_HARVARD_ENTRY_START.match(text):
                matched_texts.append(text)
                continue
            kept_blocks.append(block)
        page["blocks"] = kept_blocks

    return parse_harvard_reference_entries(matched_texts)


def merge_ocr_page(ocr_body_markdown: str, ocr_footer_text: str = "") -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Convert a single flagged page's recovered OCR content into the shared shape.

    Mistral's OCR block content is itself markdown text (confirmed: raw.md
    contains real '#'/'|' syntax), so a flagged page's recovered body text is
    parsed the same way `preprocess.py` parses a full markdown document, keeping
    OCR-recovered pages and text-layer pages in one uniform block stream. The
    page's footer/reference text (already numbered-line text, the same shape
    `references_raw.md` has always used) is parsed with the same helper
    `preprocess.py` uses for that file.

    Returns (blocks, references).
    """
    try:
        # Matches how the app itself imports sibling modules (run with web_app/ on sys.path).
        from preprocess import MarkdownDocumentProcessor, _parse_md, parse_numbered_reference_lines
    except ImportError:
        # Matches how tests import this package from the repo root (web_app.<module>).
        from web_app.preprocess import MarkdownDocumentProcessor, _parse_md, parse_numbered_reference_lines

    tokens, lines = _parse_md(ocr_body_markdown)
    blocks = MarkdownDocumentProcessor._blocks_from_markdown_tokens(tokens, lines, endnote_start=None)
    references = parse_numbered_reference_lines(ocr_footer_text.splitlines()) if ocr_footer_text else []
    return blocks, references


# ==============================================================================
# Page-level classification
# ==============================================================================

def _analyze_page(
    page: "pymupdf.Page",
    page_dict: Dict[str, Any],
    body_size: float,
    repeated_lines: Set[str],
    links: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Build this page's blocks/references from PyMuPDF's *reflowed* text blocks.

    This document's PDF (like many InDesign-style exports) draws each visual line
    -- sometimes each styled run within a line -- as its own low-level content-stream
    block, so `page_dict["blocks"]` is fragmented far below paragraph granularity
    (e.g. "to world " and "leading companies..." as two separate blocks for one
    line). `page.get_text("blocks")` uses PyMuPDF's own layout-reconstruction pass
    to re-glue those fragments into real paragraph/heading/reference-entry
    groupings, which `page_dict`'s raw block IDs cannot do. Font size for
    classification, and the original span-level marker parsing for multi-line
    reference entries, still come from `page_dict` by spatially matching its
    spans/lines back into each reflowed block's bounding box.
    """
    blocks: List[Dict[str, str]] = []
    references: List[Dict[str, str]] = []

    span_index = _build_span_index(page_dict)
    char_records = _char_records(page.get_text("rawdict"))
    page_words = page.get_text("words", sort=True)
    page_links = _get_page_uri_links(page)
    tables = _extract_tables(page)
    table_rects = [bbox for _text, bbox in tables]

    candidates = []
    for x0, y0, x1, y1, raw_text, _block_no, block_type in page.get_text("blocks", sort=True):
        if block_type != 0:  # not a text block (e.g. an image)
            continue

        rect = (x0, y0, x1, y1)
        text = _normalize_block_text(raw_text)
        if not text or _is_toc_block(text) or _rect_in_tables(rect, table_rects):
            continue

        candidates.append({"rect": rect, "text": text, "lines": _lines_in_rect(page_dict, rect)})

    for candidate in _merge_adjacent_marker_blocks(candidates):
        rect, text, lines = candidate["rect"], candidate["text"], candidate["lines"]
        dominant_size = _dominant_size_in_rect(span_index, rect)
        kind = _classify_size(dominant_size, body_size)
        if kind == "paragraph":
            marker_count = _count_reference_markers(lines)
            if marker_count >= 2 and _first_line_is_marker(lines, rect[0]):
                # Two or more superscript-style markers, in a block that
                # itself *opens* on a marker, only happens when several
                # consecutive reference/footnote entries got glued into the
                # same block -- trusted regardless of the block's aggregate
                # font size, since some documents (e.g. one real footer block
                # that mixed a 9.96pt entry with three 11pt, body-sized ones)
                # don't consistently shrink every entry's text, only the
                # marker digit itself, which would otherwise average the
                # block's dominant size back up to body_size and defeat the
                # size-based check below. Requiring the block to *open* on a
                # marker (not just contain one deeper in) is what excludes an
                # ordinary paragraph whose several sentences each end in an
                # inline citation -- that block opens on plain prose, not a
                # marker, no matter how many citations it goes on to contain.
                kind = "reference"
            elif marker_count == 1 and dominant_size < body_size - MARKER_SIZE_MARGIN:
                # A single marker is a much more specific signal than
                # SIZE_GAP_THRESHOLD, and catches documents where body and
                # reference text sit only ~1pt apart -- too close for the
                # size-gap check alone. But with only one marker, still gated
                # on the block's dominant size sitting (even slightly) below
                # body size: an inline citation marker inside an ordinary body
                # paragraph has the same small-digit-then-larger-span shape,
                # but that paragraph's dominant size is body_size itself, not
                # something below it -- so this guard is what tells a lone
                # references-section entry from a body sentence with a citation.
                kind = "reference"

        if kind == "reference":
            references.extend(_parse_reference_entries(lines))
        elif kind == "heading":
            blocks.append({"type": "heading", "text": text})
        else:
            if not _is_header_or_page_number(text, repeated_lines):
                text = _text_with_superscripts(page_words, char_records, rect, text, page_links, links)
                blocks.append({"type": "paragraph", "text": text})

    for table_text, _bbox in tables:
        blocks.append({"type": "table", "text": table_text})

    char_count = len(page.get_text().strip())
    has_images = len(page.get_images()) > 0
    needs_ocr = char_count <= MIN_TEXT_CHARS and has_images
    ocr_reason = "little/no extractable text, contains embedded images" if needs_ocr else None

    return {
        "blocks": blocks,
        "references": references,
        "needs_ocr": needs_ocr,
        "ocr_reason": ocr_reason,
    }


def _classify_size(dominant_size: float, body_size: float) -> str:
    """Classify a font size as belonging to a "reference", "heading", or "paragraph"."""
    if dominant_size <= body_size - SIZE_GAP_THRESHOLD:
        return "reference"
    if dominant_size >= body_size + SIZE_GAP_THRESHOLD:
        return "heading"
    return "paragraph"


def _normalize_block_text(raw_text: str) -> str:
    """Collapse a reflowed block's internal line breaks into one flat string."""
    return " ".join(raw_text.split())


def _char_records(page_dict_raw: Dict[str, Any]) -> List[Tuple[float, float, str, int]]:
    """(center_x, center_y, char, span_flags) for every character on the page,
    from `page.get_text("rawdict")`.

    Character, not span or word, granularity: a citation marker is frequently
    glued directly onto the digits of an ordinary number with no separating
    space (e.g. "2009/10.10", where only the trailing "10" is the citation --
    confirmed on a real document). A word- or span-level signal can't tell
    which digits within such a run are the superscript-flagged citation and
    which are the ordinary number it's glued to; per-character flags can.
    """
    records = []
    for block in page_dict_raw["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                flags = span.get("flags", 0)
                for ch in span.get("chars", []):
                    cx0, cy0, cx1, cy1 = ch["bbox"]
                    records.append(((cx0 + cx1) / 2, (cy0 + cy1) / 2, ch["c"], flags))
    return records


def _get_page_uri_links(page: "pymupdf.Page") -> List[Dict[str, Any]]:
    """External (URI) hyperlinks on the page, as {"rect": (x0,y0,x1,y1), "uri": str}.

    Only `LINK_URI` links are kept -- internal navigation links (e.g.
    `LINK_GOTO`, a table-of-contents entry jumping to a heading) don't point at
    an external evidence source and are left untouched, mirroring how
    web_parser.py's `_extract_body_links` leaves in-page anchors (href="#...")
    untouched.
    """
    links = []
    for link in page.get_links():
        uri = link.get("uri")
        if link.get("kind") == pymupdf.LINK_URI and uri:
            links.append({"rect": tuple(link["from"]), "uri": uri})
    return links


def _link_index_for_word(
    word_rect: Tuple[float, float, float, float],
    page_links: List[Dict[str, Any]],
    pad: float = 1.0,
) -> Optional[int]:
    """Index of the page hyperlink (if any) whose rect a word's center falls inside."""
    wx0, wy0, wx1, wy1 = word_rect
    wcx, wcy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
    for i, link in enumerate(page_links):
        lx0, ly0, lx1, ly1 = link["rect"]
        if lx0 - pad <= wcx <= lx1 + pad and ly0 - pad <= wcy <= ly1 + pad:
            return i
    return None


def _text_with_superscripts(
    page_words: List[Tuple[float, float, float, float, str, int, int, int]],
    char_records: List[Tuple[float, float, str, int]],
    rect: Tuple[float, float, float, float],
    fallback_text: str,
    page_links: Optional[List[Dict[str, Any]]] = None,
    links_out: Optional[List[Dict[str, str]]] = None,
    pad: float = 1.0,
) -> str:
    """Rebuild a paragraph block's flat text from words, promoting superscripts
    and annotating hyperlinked phrases with a trailing "**(Link: N)" marker.

    `page.get_text("blocks")` (used for `fallback_text`) returns plain text with
    no font-size or superscript-flag info, so an inline citation glued directly
    onto a word with no separating punctuation (e.g. "EA17", "overflows18") is
    indistinguishable from an ordinary trailing number by the time it reaches
    `preprocess.py`. Translating any character PyMuPDF marks superscript
    (`flags & 1`) whose text is a digit into its Unicode superscript equivalent
    lets `UnicodeSuperscriptCitationExtractor` pick these up downstream with no
    change to the citation parser itself.

    Rebuilt word-by-word from `page.get_text("words", sort=True)` rather than
    from `page_dict`'s per-line spans: a superscript marker's bbox sits at a
    slightly different y than the baseline row it's glued to (observed: a
    citation ".1" at y0=503.9 immediately before a line at y0=504.4), which is
    enough to sort it into the wrong line -- and therefore the wrong word
    position -- when lines are independently re-sorted by (y, x) as
    `_lines_in_rect` does. `get_text("words")`'s own reading-order reflow (the
    same one `page.get_text("blocks")` uses for `fallback_text`) doesn't have
    that failure mode.

    Within each word, the replacement is rebuilt strictly from `char_records`
    (falling back to the word's own text unchanged if the characters found
    there don't reconstruct it exactly) rather than by regex over the word's
    text, precisely to avoid over-translating a glued-together case like
    "2009/10.10".

    `page_links` (this page's genuine external hyperlinks, from
    `_get_page_uri_links`) are matched against each word's center, the same way
    `web_parser.py`'s `_extract_body_links` walks HTML `<a href>` elements.
    Consecutive words covered by the same link are one phrase, so the marker is
    emitted once, right after the phrase, appended (in document order) to
    `links_out` -- analogous to `GovUKParser.self.links`.

    Only called for blocks already classified as "paragraph" -- reference
    entries (whose own marker digits must NOT be superscripted here, since they
    are parsed separately by `_parse_reference_entries`) never reach this path.
    """
    if not char_records:
        return fallback_text

    try:
        # Matches how the app itself imports sibling modules (run with web_app/ on sys.path).
        from preprocess import DIGIT_TO_SUPERSCRIPT
    except ImportError:
        # Matches how tests import this package from the repo root (web_app.<module>).
        from web_app.preprocess import DIGIT_TO_SUPERSCRIPT

    rx0, ry0, rx1, ry1 = rect
    words = []
    for wx0, wy0, wx1, wy1, wtext, *_rest in page_words:
        wcx, wcy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
        if rx0 - pad <= wcx <= rx1 + pad and ry0 - pad <= wcy <= ry1 + pad:
            words.append((wx0, wy0, wx1, wy1, wtext))

    if not words:
        return fallback_text

    char_pad = 0.5
    parts = []
    current_link_idx: Optional[int] = None
    current_link_words: List[str] = []

    def flush_link():
        nonlocal current_link_idx, current_link_words
        if current_link_idx is not None and links_out is not None:
            number = str(len(links_out) + 1)
            links_out.append({
                "number": number,
                "url": page_links[current_link_idx]["uri"],
                "text": " ".join(current_link_words),
            })
            parts.append(f"**(Link: {number})")
        current_link_idx = None
        current_link_words = []

    for wx0, wy0, wx1, wy1, wtext in words:
        if not any(ch.isdigit() for ch in wtext):
            word_out = wtext
        else:
            matched = sorted(
                (cx, ch, flags)
                for cx, cy, ch, flags in char_records
                if wx0 - char_pad <= cx <= wx1 + char_pad and wy0 - char_pad <= cy <= wy1 + char_pad
            )
            rebuilt_plain = "".join(ch for _cx, ch, _flags in matched)
            if rebuilt_plain != wtext:
                # Character reconstruction didn't exactly reproduce the word (e.g. a
                # ligature or a word-tokenizer/rawdict mismatch) -- leave it untouched
                # rather than risk a wrong translation.
                word_out = wtext
            else:
                word_out = "".join(
                    ch.translate(DIGIT_TO_SUPERSCRIPT) if (ch.isdigit() and flags & 1) else ch
                    for _cx, ch, flags in matched
                )

        link_idx = _link_index_for_word((wx0, wy0, wx1, wy1), page_links) if page_links else None
        if link_idx != current_link_idx:
            flush_link()
            current_link_idx = link_idx
        if link_idx is not None:
            current_link_words.append(word_out)
        parts.append(word_out)

    flush_link()
    return " ".join(parts)


def _is_toc_block(text: str) -> bool:
    """A table-of-contents heading or dot-leader entry -- navigational, not content."""
    if DOT_LEADER_PATTERN.search(text):
        return True
    return text.strip().lower() in TOC_HEADING_TEXTS


def _build_span_index(page_dict: Dict[str, Any]) -> List[Tuple[float, float, float, int]]:
    """Flatten every span in a page to (center_x, center_y, size, char_count).

    Used to look up which font size dominates an arbitrary rectangle -- namely a
    `page.get_text("blocks")` block's bounding box, which does not itself carry
    font information.
    """
    spans: List[Tuple[float, float, float, int]] = []
    for block in page_dict["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if not span["text"]:
                    continue
                sx0, sy0, sx1, sy1 = span["bbox"]
                spans.append(((sx0 + sx1) / 2, (sy0 + sy1) / 2, round(span["size"], 1), len(span["text"])))
    return spans


def _dominant_size_in_rect(
    spans: List[Tuple[float, float, float, int]],
    rect: Tuple[float, float, float, float],
    pad: float = 1.0,
) -> float:
    """The font size covering the most characters whose span center falls in `rect`."""
    x0, y0, x1, y1 = rect
    counts: Counter = Counter()
    for cx, cy, size, length in spans:
        if x0 - pad <= cx <= x1 + pad and y0 - pad <= cy <= y1 + pad:
            counts[size] += length
    return counts.most_common(1)[0][0] if counts else 0.0


def _lines_in_rect(
    page_dict: Dict[str, Any],
    rect: Tuple[float, float, float, float],
    pad: float = 1.0,
) -> List[Dict[str, Any]]:
    """Non-blank dict lines whose center falls in `rect`, in reading order.

    Reassembles a reflowed reference-entry block back into the original
    per-line span structure that `_parse_reference_entries` needs to tell a
    superscript-style marker line from a wrapped continuation line.
    """
    x0, y0, x1, y1 = rect
    lines = []
    for block in page_dict["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            if not _line_text(line).strip():
                continue
            lx0, ly0, lx1, ly1 = line["bbox"]
            cx, cy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
            if x0 - pad <= cx <= x1 + pad and y0 - pad <= cy <= y1 + pad:
                lines.append(line)
    lines.sort(key=lambda l: (round(l["bbox"][1], 1), l["bbox"][0]))
    return lines


def _is_marker_line(spans: List[Dict[str, Any]]) -> bool:
    """A short pure-digit first span rendered smaller than the rest of the line.

    This is the superscript-style number marker that opens a reference entry.
    """
    return bool(spans) and (
        spans[0]["text"].strip().isdigit()
        and len(spans) > 1
        and spans[0]["size"] < spans[1]["size"]
    )


def _count_reference_markers(lines: List[Dict[str, Any]]) -> int:
    count = 0
    for line in lines:
        spans = [s for s in line["spans"] if s["text"]]
        if _is_marker_line(spans):
            count += 1
    return count


def _first_line_is_marker(lines: List[Dict[str, Any]], rect_x0: float, tolerance: float = 5.0) -> bool:
    """Whether the block's leftmost-aligned line opens on a reference marker.

    A raised inline-citation superscript (e.g. "...Innovation (UKRI)9 and...")
    sits mid-line, so PyMuPDF's line grouping can still give it a smaller y0
    than the paragraph's own opening line (its baseline is vertically offset),
    which would otherwise make it sort first and be mistaken for the marker
    that opens a genuine reference entry. A real entry's marker is always
    flush with the block's left margin; a mid-sentence superscript is not, so
    restricting to left-aligned lines is what tells them apart.
    """
    for line in lines:
        if line["bbox"][0] - rect_x0 > tolerance:
            continue
        spans = [s for s in line["spans"] if s["text"]]
        if not spans:
            continue
        return _is_marker_line(spans)
    return False


def _merge_adjacent_marker_blocks(
    candidates: List[Dict[str, Any]],
    max_gap: float = MARKER_BLOCK_MERGE_GAP,
) -> List[Dict[str, Any]]:
    """Glue together consecutive reflowed blocks that each open with a marker.

    PyMuPDF's block reconstruction can leave a marker isolated from its own
    continuation text (observed: a marker rendered slightly offset onto its
    own line), and/or leave two consecutive entries as separate single-marker
    blocks. Neither, alone, can be told apart from a lone inline citation
    inside a body paragraph by font size alone (see the classification logic
    below) -- but chaining consecutive marker-led blocks together recovers the
    multi-marker signal that a real footer/endnote run always has, without
    ever touching a block that isn't itself already marker-led: a body
    paragraph's own block opens on plain prose, not a marker, so it is never
    an eligible chain member here regardless of how many inline citations it
    goes on to contain.
    """
    merged: List[Dict[str, Any]] = []
    i = 0
    while i < len(candidates):
        current = candidates[i]
        if _first_line_is_marker(current["lines"], current["rect"][0]):
            j = i + 1
            while j < len(candidates) and _first_line_is_marker(candidates[j]["lines"], candidates[j]["rect"][0]):
                gap = candidates[j]["rect"][1] - current["rect"][3]
                if gap > max_gap:
                    break
                nxt = candidates[j]
                x0 = min(current["rect"][0], nxt["rect"][0])
                y0 = min(current["rect"][1], nxt["rect"][1])
                x1 = max(current["rect"][2], nxt["rect"][2])
                y1 = max(current["rect"][3], nxt["rect"][3])
                current = {
                    "rect": (x0, y0, x1, y1),
                    "text": current["text"] + " " + nxt["text"],
                    "lines": current["lines"] + nxt["lines"],
                }
                j += 1
            merged.append(current)
            i = j
        else:
            merged.append(current)
            i += 1
    return merged


def _parse_reference_entries(lines: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Parse a reference block's lines into number/text entries.

    A new entry starts on a line whose first span is a short pure-digit string
    rendered smaller than the rest of that line (the superscript-style number
    marker); any following line without such a marker is a wrapped continuation
    of the previous entry's text.
    """
    entries: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None

    for line in lines:
        spans = [s for s in line["spans"] if s["text"]]
        if not spans:
            continue

        first_text = spans[0]["text"].strip()
        is_marker = _is_marker_line(spans)

        if is_marker:
            if current is not None:
                entries.append(current)
            rest_text = "".join(s["text"] for s in spans[1:]).strip()
            current = {"number": first_text, "text": rest_text}
        else:
            line_text = _line_text(line).strip()
            if not line_text:
                continue
            if current is not None:
                current["text"] = (current["text"] + " " + line_text).strip()

    if current is not None:
        entries.append(current)

    return [e for e in entries if e["text"]]


def _extract_tables(page: "pymupdf.Page") -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """Best-effort table reconstruction as simple pipe-delimited text, with each
    table's bounding box so its cells can be excluded from the ordinary
    heading/paragraph/reference block loop (otherwise a numbered ruled table --
    e.g. a numbered list of company names -- would also get parsed as reference
    entries there, since a small numeral column followed by larger cell text has
    the same shape as a superscript-style reference marker).

    PyMuPDF's table detector only finds ruled/bordered tables -- borderless
    layout tables (e.g. a two-column table of contents) are not detected and
    fall through to ordinary paragraph classification. That's an accepted
    cosmetic limitation: correctness of reference/citation extraction does not
    depend on table fidelity.
    """
    tables = []
    try:
        finder = page.find_tables()
    except Exception:
        return tables

    for table in finder.tables:
        rows = table.extract()
        cells = [cell for row in rows for cell in row]
        fill_ratio = sum(1 for c in cells if c and str(c).strip()) / len(cells) if cells else 0.0
        if fill_ratio < MIN_TABLE_FILL_RATIO:
            # PyMuPDF's ruled-table detector can misfire on a run of closely-
            # spaced same-margin lines -- observed on a page footnote block,
            # detected as a "table" whose cells were ~35% filled (one real
            # column of text, the rest None). A real table's cells are almost
            # all populated; skip anything this sparse and let it fall through
            # to ordinary heading/paragraph/reference classification instead.
            continue
        lines = [" | ".join(str(cell or "").strip() for cell in row) for row in rows]
        text = "\n".join(line for line in lines if line.strip())
        if text:
            tables.append((text, tuple(table.bbox)))
    return tables


def _rect_in_tables(
    rect: Tuple[float, float, float, float],
    table_rects: List[Tuple[float, float, float, float]],
) -> bool:
    """Whether `rect`'s center falls inside any detected table's bounding box."""
    cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    for tx0, ty0, tx1, ty1 in table_rects:
        if tx0 <= cx <= tx1 and ty0 <= cy <= ty1:
            return True
    return False


# ==============================================================================
# Font-size helpers
# ==============================================================================

def _line_text(line: Dict[str, Any]) -> str:
    return "".join(span["text"] for span in line["spans"])


def _document_body_size(page_dicts: List[Dict[str, Any]]) -> float:
    """The font size covering the most characters across the whole document.

    Computed document-wide rather than per-page: a page with unusually little
    body text relative to its footnotes could otherwise skew a per-page estimate.
    """
    counts: Counter = Counter()
    for page_dict in page_dicts:
        for block in page_dict["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    counts[round(span["size"], 1)] += len(span["text"])
    return counts.most_common(1)[0][0] if counts else 12.0


# ==============================================================================
# Running header / page-number filtering
# ==============================================================================

def _find_repeated_header_lines(page_dicts: List[Dict[str, Any]]) -> Set[str]:
    """Short body-sized lines that repeat verbatim across many pages (running headers)."""
    counts: Counter = Counter()
    total_pages = len(page_dicts)

    for page_dict in page_dicts:
        seen_this_page: Set[str] = set()
        for block in page_dict["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                text = _line_text(line).strip()
                if text and len(text) <= HEADER_MAX_CHARS:
                    seen_this_page.add(text)
        for text in seen_this_page:
            counts[text] += 1

    if total_pages == 0:
        return set()

    # Require the line to repeat on at least 2 pages, not just clear the ratio
    # threshold -- on a 1-2 page document, a ratio alone would trivially treat
    # every short line as a "repeated" header.
    return {
        text for text, count in counts.items()
        if count >= 2 and count / total_pages >= HEADER_REPETITION_THRESHOLD
    }


def _is_header_or_page_number(text: str, repeated_lines: Set[str]) -> bool:
    if text.strip().isdigit():
        return True
    return text in repeated_lines
