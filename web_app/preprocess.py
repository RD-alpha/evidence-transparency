"""Document preprocessing and reference extraction pipeline.

This module splits documents into sentences, extracts in-text citations,
parses reference definitions from footers/endnotes, and generates structured
artefacts (processed.md, references.json, input.txt).
"""

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.sax.saxutils import escape as xml_escape, quoteattr as xml_quoteattr

from markdown_it import MarkdownIt
import spacy

from sentence_ids import format_sentence_line, is_sentence_line, sentence_id, split_sentence_line

md = MarkdownIt("gfm-like")
nlp = spacy.load("en_core_web_sm")

SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
SUPERSCRIPT_TO_DIGIT = str.maketrans(SUPERSCRIPT_DIGITS, "0123456789")
DIGIT_TO_SUPERSCRIPT = str.maketrans("0123456789", SUPERSCRIPT_DIGITS)

# Common regex patterns
RE_SUPERSCRIPT_CHARS = re.compile(r'[⁰¹²³⁴⁵⁶⁷⁸⁹]+')
RE_LATEX_SUPERSCRIPT = re.compile(r'\$\^\{([^}]+)\}\$')
RE_MARKDOWN_FOOTNOTE = re.compile(r'\[\^([^\]]+)\]')
RE_HTML_SUPERSCRIPT = re.compile(r'<sup>([^<]+)</sup>', re.IGNORECASE)
RE_ENDNOTE_HEADING = re.compile(
    r'^\s*#{1,6}\s*(?:\d+[\.\s]+)?(?:endnotes?|footnotes?|references?|bibliography)\b.*$',
    re.IGNORECASE
)
RE_FOOTNOTE_DEF = re.compile(r'^\s*\[\^([^\]]+)\]:\s*(.+)$')

# Author-date ("Harvard") citation support -- see HarvardCitationExtractor
# and harvard_reference_alias_map.
RE_HARVARD_CITATION_GROUP = re.compile(r'\(([A-Z][^()]{1,200})\)')
RE_HARVARD_PIECE = re.compile(r'^(.+?),?\s+(\d{4}[a-z]?)\s*$')
RE_HARVARD_ENTRY_START = re.compile(r'^([A-Z].{1,200}?)\s*\((\d{4}[a-z]?)\)')
RE_REFERENCES_HEADING_PLAIN = re.compile(r'^(?:\d+[\.\s]+)?(?:references?|bibliography)\s*$', re.IGNORECASE)
RE_AUTHOR_SURNAME = re.compile(r"([A-Z][\w'’\-]+),?\s+[A-Z]\.")
HARVARD_ACRONYM_STOPWORDS = {"for", "of", "the", "and", "in", "on"}


ORDINAL_SUFFIXES = {"st", "nd", "rd", "th"}


def expand_numeric_range_or_list(val_str: str) -> List[str]:
    """Expand strings like '1, 2', '1-3', '1–3', '1/2' into individual numbers."""
    if not val_str:
        return []
    # Translate any Unicode superscripts and superscript minus first
    val_str = val_str.translate(SUPERSCRIPT_TO_DIGIT).replace('\u207b', '-')
    parts = re.split(r'[,;/]\s*', val_str.strip())
    results: List[str] = []
    for part in parts:
        part = part.strip()
        if not part or part.lower() in ORDINAL_SUFFIXES:
            continue
        # Support ranges like 1-3, 1–3, 1—3, 1⁻3
        range_match = re.match(r'^(\d+)\s*[\-\u2010-\u2015\u207b]\s*(\d+)$', part)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if 0 < end - start <= 50:
                results.extend(str(x) for x in range(start, end + 1))
            else:
                results.append(part)
        else:
            results.append(part)
    return results


def parse_numbered_reference_lines(lines: List[str]) -> List[Dict[str, str]]:
    """Parse plain-text numbered reference lines (`1 Text`, `[1] Text`, `¹ Text`, ...).

    Consecutive non-matching lines are treated as a continuation of the
    previous entry's text (e.g. a reference that wraps onto a second line).
    Shared by `RawFooterFileParser` (reading references_raw.md) and any other
    source of already-numbered reference text (e.g. OCR-recovered footer text).
    """
    refs: List[Dict[str, str]] = []
    current_num: Optional[str] = None
    current_text: List[str] = []

    def save():
        nonlocal current_num, current_text
        if current_num is not None:
            refs.append({"number": current_num, "text": " ".join(current_text).strip()})
            current_num = None
            current_text = []

    for line in lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        m = re.match(r'^\s*(?:\[\^?(\d+)\]|([⁰¹²³⁴⁵⁶⁷⁸⁹]+)|(\d+))\s*[:.)]?\s+(.+)$', text)
        if m:
            save()
            raw_num = m.group(1) or m.group(2) or m.group(3)
            current_num = raw_num.translate(SUPERSCRIPT_TO_DIGIT)
            current_text = [m.group(4)]
        elif current_num is not None:
            current_text.append(text)
    save()
    return refs


# ==============================================================================
# In-Text Citation Parsing (References in the main body text)
# ==============================================================================

class CitationExtractor(ABC):
    """Abstract base class for extracting citation references from in-text prose."""

    @abstractmethod
    def extract(self, text: str, known_numbers: Optional[Set[str]] = None) -> Tuple[str, List[str]]:
        """Extract citations from text and return cleaned text and extracted reference keys.

        Args:
            text: Sentence or paragraph segment.
            known_numbers: Optional set of reference numbers known to exist in the document.

        Returns:
            Tuple of (cleaned_text, list_of_extracted_reference_keys)
        """
        pass


class LatexSuperscriptCitationExtractor(CitationExtractor):
    """Extracts LaTeX-style math superscripts like $^{1}$ or $^{1,2,5-7}$."""

    def extract(self, text: str, known_numbers: Optional[Set[str]] = None) -> Tuple[str, List[str]]:
        refs: List[str] = []

        def _repl(match: re.Match) -> str:
            content = match.group(1).strip()
            if content.lower() in ORDINAL_SUFFIXES:
                return content
            expanded = expand_numeric_range_or_list(content)
            valid = [
                r for r in expanded
                if r.lower() not in ORDINAL_SUFFIXES
                and (r.isdigit() or (known_numbers and r in known_numbers))
            ]
            refs.extend(valid)
            return ""

        cleaned = RE_LATEX_SUPERSCRIPT.sub(_repl, text)
        return cleaned, refs


class HtmlSuperscriptCitationExtractor(CitationExtractor):
    """Extracts HTML superscripts like <sup>1</sup> or <sup>1,2</sup>."""

    def extract(self, text: str, known_numbers: Optional[Set[str]] = None) -> Tuple[str, List[str]]:
        refs: List[str] = []

        def _repl(match: re.Match) -> str:
            content = match.group(1).strip()
            if content.lower() in ORDINAL_SUFFIXES:
                return content
            expanded = expand_numeric_range_or_list(content)
            valid = [
                r for r in expanded
                if r.lower() not in ORDINAL_SUFFIXES
                and (r.isdigit() or (known_numbers and r in known_numbers))
            ]
            refs.extend(valid)
            return ""

        cleaned = RE_HTML_SUPERSCRIPT.sub(_repl, text)
        return cleaned, refs


class MarkdownFootnoteCitationExtractor(CitationExtractor):
    """Extracts Markdown footnote callouts like [^1] or [^ref]."""

    def extract(self, text: str, known_numbers: Optional[Set[str]] = None) -> Tuple[str, List[str]]:
        refs: List[str] = []

        def _repl(match: re.Match) -> str:
            ref_key = match.group(1).strip()
            if ref_key.lower() not in ORDINAL_SUFFIXES:
                refs.append(ref_key)
            return ""

        cleaned = RE_MARKDOWN_FOOTNOTE.sub(_repl, text)
        return cleaned, refs


class UnicodeSuperscriptCitationExtractor(CitationExtractor):
    """Extracts Unicode superscripts like ³, ¹², ¹,², or ¹⁻³."""

    # Matches sequences of superscripts and possible separators (commas, slashes, hyphens)
    SUPERSCRIPT_GROUP_PATTERN = re.compile(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+(?:[\s,/'\-\u2010-\u2015]+[⁰¹²³⁴⁵⁶⁷⁸⁹]+)*"
    )

    def extract(self, text: str, known_numbers: Optional[Set[str]] = None) -> Tuple[str, List[str]]:
        refs: List[str] = []

        def _repl(match: re.Match) -> str:
            matched_str = match.group(0)
            expanded = expand_numeric_range_or_list(matched_str)
            valid = [
                r for r in expanded
                if r.lower() not in ORDINAL_SUFFIXES
                and (r.isdigit() or (known_numbers and r in known_numbers))
            ]
            refs.extend(valid)
            return ""

        cleaned = self.SUPERSCRIPT_GROUP_PATTERN.sub(_repl, text)
        return cleaned, refs


class PunctuationAttachedCitationExtractor(CitationExtractor):
    """Extracts unformatted numbers attached to punctuation (e.g. 'Banking.24' or 'Vision,26').

    Only matches numbers that exist in `known_numbers` to avoid stripping dates or decimal figures.
    """

    def extract(self, text: str, known_numbers: Optional[Set[str]] = None) -> Tuple[str, List[str]]:
        if not known_numbers:
            return text, []

        refs: List[str] = []
        cleaned = text
        for number in known_numbers:
            pattern1 = re.compile(rf'(?<=[.,;:]){re.escape(number)}(?=\s|$)')
            pattern2 = re.compile(rf'(?<=\b\d{{4}}/\d{{2}}){re.escape(number)}(?=\s|$)')

            found = False

            def _repl(match: re.Match) -> str:
                nonlocal found
                # Distinguish a decimal fraction ("7.2 billion") from a citation
                # marker glued to punctuation ("2016.9 High capital..."): a
                # decimal is preceded by another digit and continues in
                # lowercase (its unit); a citation marker starts a new,
                # capitalised clause.
                if self._looks_like_decimal(cleaned, match):
                    return match.group(0)
                # Likewise a numbered paragraph's own label ("1.8 Industry
                # responses...") is a section number, not a citation: the
                # number is the tail of a dotted label opening the line.
                if self._looks_like_paragraph_number(cleaned, match):
                    return match.group(0)
                found = True
                return ''

            new_cleaned = pattern1.sub(_repl, cleaned)
            if found:
                refs.append(number)
                cleaned = new_cleaned
            elif pattern2.search(cleaned):
                refs.append(number)
                cleaned = pattern2.sub('', cleaned)

        return cleaned, refs

    @staticmethod
    def _looks_like_decimal(text: str, match: re.Match) -> bool:
        before = text[:match.start()]
        after = text[match.end():].lstrip()
        preceded_by_digit = len(before) >= 2 and before[-1] == '.' and before[-2].isdigit()
        followed_by_lowercase = bool(after) and after[0].islower()
        return preceded_by_digit and followed_by_lowercase

    @staticmethod
    def _looks_like_paragraph_number(text: str, match: re.Match) -> bool:
        before = text[:match.start()]
        return re.search(r'(?:^|\n)\s*\d+(?:\.\d+)*\.$', before) is not None


class BracketedCitationExtractor(CitationExtractor):
    """Extracts bracketed citations like [1] or [1, 2] when matching known numbers."""

    def extract(self, text: str, known_numbers: Optional[Set[str]] = None) -> Tuple[str, List[str]]:
        if not known_numbers:
            return text, []

        refs: List[str] = []

        def _repl(match: re.Match) -> str:
            candidate = match.group(1)
            parts = [p.strip() for p in re.split(r'[,;]\s*', candidate)]
            if all(p in known_numbers for p in parts if p):
                refs.extend(parts)
                return ""
            return match.group(0)

        cleaned = re.sub(r'\[(\d+(?:\s*[,;]\s*\d+)*)\]', _repl, text)
        return cleaned, refs


def _harvard_slug(text: str, year: str) -> str:
    """A stable, readable reference number for a Harvard-style entry, e.g. 'dwp-2017'."""
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    return f"{slug}-{year}"


def _harvard_acronym(author_part: str) -> Optional[str]:
    """'Department for Work and Pensions' -> 'DWP' (initials of capitalised, non-stopword words)."""
    words = re.findall(r"[A-Za-z][\w'’\-]*", author_part)
    initials = "".join(
        w[0].lower() for w in words
        if w[0].isupper() and w.lower() not in HARVARD_ACRONYM_STOPWORDS
    )
    return initials if len(initials) >= 2 else None


def harvard_intext_alias_keys(author_part: str) -> List[str]:
    """Candidate lookup keys for an in-text citation's author text (no year).

    In-text citations never carry initials ('DWP, 2017', 'Asmussen & Weizel,
    2010'), so authors are told apart from each other only by 'and'/'&'
    conjunctions and an optional trailing 'et al'.
    """
    stripped = author_part.strip()
    et_al = bool(re.search(r'\bet\s*al\.?\s*$', stripped, re.IGNORECASE))
    core = re.sub(r'\s*et\s*al\.?\s*$', '', stripped, flags=re.IGNORECASE).strip()

    keys = [re.sub(r'\s+', ' ', stripped.lower())]
    if core:
        keys.append(re.sub(r'\s+', ' ', core.lower()))

    parts = [p.strip() for p in re.split(r'\s*(?:,|&|\band\b)\s*', core, flags=re.IGNORECASE) if p.strip()]
    if parts:
        keys.append(parts[0].lower())
        if et_al:
            keys.append(f"{parts[0].lower()} et al")
        if len(parts) >= 2:
            keys.append(f"{parts[0].lower()} and {parts[1].lower()}")
            keys.append(f"{parts[0].lower()} et al")

    # Deduplicate while preserving order.
    return list(dict.fromkeys(keys))


def harvard_reference_alias_keys(author_part: str) -> List[str]:
    """Candidate lookup keys for a references-list entry's leading author text.

    Reference-list entries spell authors out in full with initials
    ('Asmussen, K. and Weizel, K.'), so surnames are recovered via
    `RE_AUTHOR_SURNAME` rather than by splitting on 'and' (which would
    misfire on an organisation name that itself contains 'and', e.g.
    'Department for Communities and Local Government').
    """
    stripped = author_part.strip()
    keys = [re.sub(r'\s+', ' ', stripped.lower())]

    surnames = [m.group(1) for m in RE_AUTHOR_SURNAME.finditer(stripped)]
    if surnames:
        keys.append(surnames[0].lower())
        keys.append(f"{surnames[0].lower()} et al")
        if len(surnames) >= 2:
            keys.append(f"{surnames[0].lower()} and {surnames[1].lower()}")
    else:
        acronym = _harvard_acronym(stripped)
        if acronym:
            keys.append(acronym)

    return list(dict.fromkeys(keys))


def harvard_reference_alias_map(references: List[Dict[str, str]]) -> Dict[str, str]:
    """Build the '<alias key>|<year>' -> reference number lookup used by
    `HarvardCitationExtractor`, from Harvard-style reference entries whose
    text starts 'Author(s) (Year) ...' (see `parse_harvard_reference_entries`,
    which sets `number` to the same slug this recomputes).
    """
    alias_map: Dict[str, str] = {}
    for ref in references:
        match = RE_HARVARD_ENTRY_START.match(ref["text"].strip())
        if not match:
            continue
        author_part, year = match.group(1).strip(), match.group(2)
        for key in harvard_reference_alias_keys(author_part):
            alias_map[f"{key}|{year}"] = ref["number"]
    return alias_map


def parse_harvard_reference_entries(paragraphs: List[str]) -> List[Dict[str, str]]:
    """Parse a run of references-section paragraph blocks (one per entry, no
    numbering) into {"number", "text"} entries, e.g. 'Asmussen, K. and
    Weizel, K. (2010) Evaluating the Evidence...' -> number
    'asmussen-k-and-weizel-k-2010' (see `_harvard_slug`; the in-text alias
    lookup, not this number, is what actually resolves citations).

    Only paragraphs that actually open on 'Author(s) (Year)' are kept --
    trailing boilerplate after the references section (contact details,
    licensing) doesn't match and is silently dropped, same as any other
    non-reference text found where a references parser expects entries.
    """
    entries: List[Dict[str, str]] = []
    seen_numbers: Set[str] = set()
    for text in paragraphs:
        stripped = text.strip()
        match = RE_HARVARD_ENTRY_START.match(stripped)
        if not match:
            continue
        author_part, year = match.group(1).strip(), match.group(2)
        number = _harvard_slug(author_part, year)
        if number in seen_numbers:
            continue
        seen_numbers.add(number)
        entries.append({"number": number, "text": stripped})
    return entries


class HarvardCitationExtractor(CitationExtractor):
    """Extracts Harvard/author-date citations like '(DWP, 2017)' or
    '(Whisman and Uebelacker, 2003; Harold et al, 2016)'.

    Unlike the other extractors, which only need to know which numbers are
    valid, resolving an author-date citation to the right reference entry
    requires comparing against the reference list's own author text (e.g.
    in-text 'DWP' must resolve to a references-list entry headed 'Department
    for Work and Pensions') -- so this extractor is constructed with an
    `alias_map` (built by `harvard_reference_alias_map`) rather than relying
    on the generic `known_numbers` set passed to `extract`.
    """

    def __init__(self, alias_map: Optional[Dict[str, str]] = None):
        self.alias_map = alias_map or {}

    def extract(self, text: str, known_numbers: Optional[Set[str]] = None) -> Tuple[str, List[str]]:
        if not self.alias_map:
            return text, []

        refs: List[str] = []

        def _repl(match: re.Match) -> str:
            group_content = match.group(1)
            pieces = [p.strip() for p in group_content.split(";") if p.strip()]
            resolved: List[str] = []
            for piece in pieces:
                piece_match = RE_HARVARD_PIECE.match(piece)
                if not piece_match:
                    return match.group(0)
                author_part, year = piece_match.group(1), piece_match.group(2)
                number = self._resolve(author_part, year)
                if number is None:
                    return match.group(0)
                resolved.append(number)
            refs.extend(resolved)
            return ""

        cleaned = RE_HARVARD_CITATION_GROUP.sub(_repl, text)
        return cleaned, refs

    def _resolve(self, author_part: str, year: str) -> Optional[str]:
        for key in harvard_intext_alias_keys(author_part):
            number = self.alias_map.get(f"{key}|{year}")
            if number is not None:
                return number

        # Fall back to a prefix match against this year's organisation-name
        # aliases, for a shortened in-text org name whose reference-list
        # entry continues past it (e.g. in-text 'Department of Health, 2007'
        # against a references entry headed 'Department of Health (England)
        # and the devolved administrations (2007)').
        normalized = re.sub(r'\s+', ' ', author_part.strip().lower())
        suffix = f"|{year}"
        for alias_key, number in self.alias_map.items():
            if not alias_key.endswith(suffix):
                continue
            key = alias_key[: -len(suffix)]
            if len(key) >= 6 and (key.startswith(normalized) or normalized.startswith(key)):
                return number
        return None


class InTextCitationParser:
    """Coordinates citation extractors to identify references and clean sentence text."""

    def __init__(self, extractors: Optional[List[CitationExtractor]] = None):
        if extractors is None:
            self.extractors: List[CitationExtractor] = [
                LatexSuperscriptCitationExtractor(),
                HtmlSuperscriptCitationExtractor(),
                MarkdownFootnoteCitationExtractor(),
                UnicodeSuperscriptCitationExtractor(),
                PunctuationAttachedCitationExtractor(),
            ]
        else:
            self.extractors = extractors

    def add_extractor(self, extractor: CitationExtractor) -> None:
        """Register an additional citation extractor."""
        self.extractors.append(extractor)

    def extract_citations(
        self,
        sentence: str,
        known_numbers: Optional[Set[str]] = None
    ) -> Tuple[str, List[str]]:
        """Parse all in-text citations from the sentence and clean the sentence text."""
        all_refs: List[str] = []
        clean_text = sentence

        for extractor in self.extractors:
            clean_text, refs = extractor.extract(clean_text, known_numbers=known_numbers)
            all_refs.extend(refs)

        # Deduplicate while maintaining numerical order
        unique_refs = list(dict.fromkeys(all_refs))
        unique_refs.sort(key=lambda k: (0, int(k)) if k.isdigit() else (1, k))

        # Tidy spacing and punctuation artifacts
        clean_text = self._clean_whitespace_and_punctuation(clean_text)
        return clean_text, unique_refs

    @staticmethod
    def _clean_whitespace_and_punctuation(text: str) -> str:
        # Normalize whitespace before punctuation (e.g. 'word .' -> 'word.')
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        # Collapse multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()


# ==============================================================================
# Reference Definition Parsing (Footers, Endnotes, Raw Files)
# ==============================================================================

class ReferenceDefinitionParser(ABC):
    """Abstract base class for extracting reference definitions (number -> text)."""

    @abstractmethod
    def parse(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Parse references from given context (lines, path, raw_text, etc.).

        Returns:
            List of dicts: [{"number": "1", "text": "..."}]
        """
        pass


class EndnoteSectionParser(ReferenceDefinitionParser):
    """Parses reference entries from a dedicated Endnotes/Footnotes/References section."""

    @staticmethod
    def find_endnote_start(lines: List[str]) -> Optional[int]:
        """Return the index of the line starting an Endnotes/Footnotes/References section."""
        for index, line in enumerate(lines):
            if RE_ENDNOTE_HEADING.match(line):
                return index
        return None

    def parse(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        lines = context.get("lines") or []
        start_line = context.get("endnote_start")
        if start_line is None:
            start_line = self.find_endnote_start(lines)

        if start_line is None:
            return []

        references: List[Dict[str, str]] = []
        current_number: Optional[str] = None
        current_text: List[str] = []

        def save_current():
            nonlocal current_number, current_text
            if current_number is not None:
                references.append({
                    "number": current_number,
                    "text": " ".join(current_text).strip()
                })
                current_number = None
                current_text = []

        source_lines = lines[start_line + 1:]
        for line in source_lines:
            text = line.strip()
            if not text:
                continue

            # Check if another major section heading starts
            if re.match(r'^\s*#{1,2}\s+\S', text) and not RE_ENDNOTE_HEADING.match(text):
                break

            # Look for numbered entry: 1. Text, 1 Text, [1] Text, [^1]: Text, or ¹ Text
            match = re.match(r'^(?:\[\^?(\d+)\]|([⁰¹²³⁴⁵⁶⁷⁸⁹]+)|(\d+))\s*[:.)]?\s+(.+)$', text)
            if match:
                save_current()
                raw_num = match.group(1) or match.group(2) or match.group(3)
                current_number = raw_num.translate(SUPERSCRIPT_TO_DIGIT)
                current_text = [match.group(4)]
            elif current_number is not None:
                current_text.append(text)

        save_current()
        return references


class RawFooterFileParser(ReferenceDefinitionParser):
    """Parses raw reference files (e.g. references_raw.md, ref_raw.txt, references.md)."""

    def parse(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        path = context.get("path")
        if not path:
            return []

        doc_dir = Path(path)
        candidates = [
            doc_dir / "references_raw.md",
            doc_dir / "ref_raw.txt",
            doc_dir / "references.md",
        ]

        references: List[Dict[str, str]] = []
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                file_refs = self._parse_file(candidate)
                if file_refs:
                    references.extend(file_refs)
                    break

        return references

    def _parse_file(self, file_path: Path) -> List[Dict[str, str]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()

        if not content:
            return []

        # Check if content is formatted as JSON list (e.g. ref_raw.txt)
        if content.startswith("[") and content.endswith("]"):
            try:
                items = json.loads(content)
                refs: List[Dict[str, str]] = []
                for item in items:
                    if isinstance(item, dict) and "number" in item and "text" in item:
                        refs.append({"number": str(item["number"]), "text": str(item["text"])})
                    elif isinstance(item, str):
                        m = re.match(r'^\s*(\d+)\s+(.+)$', item)
                        if m:
                            refs.append({"number": m.group(1), "text": m.group(2).strip()})
                return refs
            except Exception:
                pass

        # Text/Markdown line parsing
        return parse_numbered_reference_lines(content.splitlines())


class InlineFootnoteParser(ReferenceDefinitionParser):
    """Parses standard Markdown footnote definitions: [^1]: Reference text."""

    def parse(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        lines = context.get("lines") or []
        references = []
        for line in lines:
            match = RE_FOOTNOTE_DEF.match(line)
            if match:
                ref_num = match.group(1).translate(SUPERSCRIPT_TO_DIGIT)
                references.append({"number": ref_num, "text": match.group(2).strip()})
        return references


class SuperscriptDefinitionParser(ReferenceDefinitionParser):
    """Parses reference definitions starting with a superscript character (e.g. '¹ OECD (2024)...')."""

    def parse(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        lines = context.get("lines") or []
        references = []
        for line in lines:
            text = line.strip()
            match = re.match(r'^\s*([⁰¹²³⁴⁵⁶⁷⁸⁹]+)\s+(.+)$', text)
            if match:
                ref_num = match.group(1).translate(SUPERSCRIPT_TO_DIGIT)
                references.append({"number": ref_num, "text": match.group(2).strip()})
        return references


class PlainNumberedLineParser(ReferenceDefinitionParser):
    """Parses standalone numbered reference lines in documents without an Endnotes section."""

    def parse(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        lines = context.get("lines") or []
        # If an endnotes heading exists, the EndnoteSectionParser takes priority
        if EndnoteSectionParser.find_endnote_start(lines) is not None:
            return []

        references = []
        for line in lines:
            text = line.strip()
            if _is_plain_reference_line(text):
                match = re.match(r'^\s*(\d+)\s+(.+)$', text)
                if match:
                    references.append({"number": match.group(1), "text": match.group(2).strip()})
        return references


class ReferenceDefinitionExtractor:
    """Aggregates and coordinates reference definition parsers into a unified registry."""

    def __init__(self, parsers: Optional[List[ReferenceDefinitionParser]] = None):
        if parsers is None:
            self.parsers: List[ReferenceDefinitionParser] = [
                RawFooterFileParser(),
                EndnoteSectionParser(),
                InlineFootnoteParser(),
                SuperscriptDefinitionParser(),
                PlainNumberedLineParser(),
            ]
        else:
            self.parsers = parsers

    def add_parser(self, parser: ReferenceDefinitionParser) -> None:
        """Register an additional definition parser."""
        self.parsers.append(parser)

    def extract_references(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract and deduplicate reference definitions from all configured sources."""
        refs_by_number: Dict[str, str] = {}

        for parser in self.parsers:
            try:
                results = parser.parse(context)
                for item in results:
                    num = item["number"]
                    text = item["text"].strip()
                    if not text:
                        continue
                    # Prefer existing definition if current is significantly shorter
                    if num not in refs_by_number or len(text) > len(refs_by_number[num]):
                        refs_by_number[num] = text
            except Exception as e:
                # Keep other parsers running even if one encounters an error
                continue

        # Sort references numerically
        sorted_numbers = sorted(
            refs_by_number.keys(),
            key=lambda k: (0, int(k)) if k.isdigit() else (1, k)
        )
        return [{"number": num, "text": refs_by_number[num]} for num in sorted_numbers]


# ==============================================================================
# Document Processing Pipeline
# ==============================================================================

class MarkdownDocumentProcessor:
    """Processes markdown documents, coordinating citation extraction and output writing."""

    def __init__(
        self,
        citation_parser: Optional[InTextCitationParser] = None,
        definition_extractor: Optional[ReferenceDefinitionExtractor] = None,
    ):
        self.citation_parser = citation_parser or InTextCitationParser()
        self.definition_extractor = definition_extractor or ReferenceDefinitionExtractor()

    def process(
        self,
        path: Path,
        tokens: Optional[List[Any]] = None,
        lines: Optional[List[str]] = None,
        out_dir: Optional[Path] = None,
        parser_type: str = "auto",
    ) -> List[Dict[str, str]]:
        """Process document at path or from pre-parsed tokens/lines."""
        out_dir = out_dir or path

        if tokens is None or lines is None:
            raw_text = _get_text(path)
            tokens, lines = _parse_md(raw_text)

        endnote_start = EndnoteSectionParser.find_endnote_start(lines)
        has_plain_refs = any(_is_plain_reference_line(line) for line in lines)

        if parser_type == "type_2" and endnote_start is None and not has_plain_refs:
            raise ValueError("Type 2 requires an Endnotes, Footnotes, References, or Bibliography heading")

        context = {
            "path": path,
            "lines": lines,
            "tokens": tokens,
            "endnote_start": endnote_start,
        }

        references = self.definition_extractor.extract_references(context)
        blocks = self._blocks_from_markdown_tokens(tokens, lines, endnote_start)
        return self.process_structured(blocks, references, out_dir)

    @staticmethod
    def _blocks_from_markdown_tokens(
        tokens: List[Any],
        lines: List[str],
        endnote_start: Optional[int],
    ) -> List[Dict[str, str]]:
        """Convert parsed markdown tokens into type/text blocks (heading, table, paragraph).

        Mirrors the token-walking behaviour previously inlined in `process()`, so
        markdown-origin documents and structurally-extracted documents (e.g. from a
        PDF's text layer) can share the same downstream processing in `process_structured`.
        """
        blocks: List[Dict[str, str]] = []
        in_table = False

        for i, token in enumerate(tokens):
            # Skip endnotes / bibliography tokens
            if endnote_start is not None and token.map and token.map[0] >= endnote_start:
                continue

            if in_table:
                if token.type == "table_close":
                    in_table = False
                continue

            if token.type == "heading_open":
                blocks.append({"type": "heading", "text": tokens[i + 1].content})
            elif token.type == "table_open":
                in_table = True
                start, end = token.map
                original_markdown = "".join(lines[start:end])
                blocks.append({"type": "table", "text": original_markdown})
            elif token.type == "paragraph_open":
                blocks.append({"type": "paragraph", "text": tokens[i + 1].content})

        return blocks

    def process_structured(
        self,
        blocks: List[Dict[str, str]],
        references: List[Dict[str, str]],
        out_dir: Path,
    ) -> List[Dict[str, str]]:
        """Write processed.md/references.json from pre-built type/text blocks.

        This is the shared core used by both `process()` (markdown-origin documents)
        and structurally-extracted documents (e.g. a PDF's text layer), so both paths
        get identical citation extraction and reference handling.
        """
        known_numbers = {r["number"] for r in references}

        with open(out_dir / "processed.md", "w", encoding="utf-8") as f:
            idx = 1
            for block in blocks:
                block_type = block.get("type")
                if block_type in ("heading", "table"):
                    f.write(block["text"] + "\n")
                elif block_type == "paragraph":
                    idx = self._write_paragraph_sentences(
                        block["text"], f, idx, known_numbers, references
                    )

        # Sort references finally and write references.json
        references = sorted(
            list({entry["number"]: entry for entry in references}.values()),
            key=lambda k: (0, int(k["number"])) if k["number"].isdigit() else (1, k["number"])
        )
        with open(out_dir / "references.json", "w", encoding="utf-8") as f:
            json.dump(references, f, indent=2)

        return references

    def _write_paragraph_sentences(
        self,
        text: str,
        f,
        idx: int,
        known_numbers: Set[str],
        references: List[Dict[str, str]],
    ) -> int:
        """Split a paragraph block's text into sentences and write S### lines.

        `known_numbers`/`references` are shared, mutable state across the whole
        document: an inline superscript definition found here (legacy type 1
        format) becomes a known reference for every paragraph processed after it.
        """
        for line_text in text.split("\n"):
            if "![img" in line_text:
                continue
            if _is_superscript_reference_line(line_text):
                continue
            if _is_plain_reference_line(line_text):
                # A leading digit alone isn't enough -- an ordinary numbered
                # paragraph (e.g. Whitehall-style "1 This government response
                # follows..." body text) matches the same shape as a plain
                # numbered bibliography entry ("24 FCA ..."), and the two
                # numbering sequences can collide on the same digit (a
                # document's paragraph 1 and its footnote 1 are unrelated).
                # So a shared leading number isn't enough either: only drop
                # the line when it's actually (near-)identical to a reference
                # definition we already extracted from a references section
                # -- i.e. this line is a stray duplicate of real reference
                # text, not merely a paragraph that happens to start with a
                # matching digit.
                ref_num = _plain_reference_line_number(line_text)
                if ref_num is not None and ref_num in known_numbers:
                    line_body = line_text.strip()[len(ref_num):].strip()
                    ref_text = next(
                        (r["text"] for r in references if r["number"] == ref_num), None
                    )
                    if ref_text and _normalize_whitespace(line_body) == _normalize_whitespace(ref_text):
                        continue

            read = nlp(line_text)
            for sentence in read.sents:
                sentence_str = str(sentence).strip()
                if not sentence_str:
                    continue

                # Check if sentence is an inline superscript definition (legacy type 1 format)
                if sentence_str[0] in SUPERSCRIPT_DIGITS:
                    match = re.match(r'^([⁰¹²³⁴⁵⁶⁷⁸⁹]+)\s*(.+)$', sentence_str)
                    if match:
                        ref_num = match.group(1).translate(SUPERSCRIPT_TO_DIGIT)
                        ref_text = match.group(2).strip()
                        if ref_num not in known_numbers:
                            references.append({"number": ref_num, "text": ref_text})
                            known_numbers.add(ref_num)
                        continue

                clean_sentence, refs = self.citation_parser.extract_citations(
                    sentence_str,
                    known_numbers=known_numbers
                )

                if not clean_sentence:
                    continue

                if refs:
                    f.write(format_sentence_line(idx, f"{clean_sentence} ** (Reference: {','.join(refs)})") + "\n")
                else:
                    f.write(format_sentence_line(idx, clean_sentence) + "\n")
                idx += 1

        return idx


# ==============================================================================
# Helper & Compatibility Functions
# ==============================================================================

def _get_text(path: Path) -> str:
    """Read raw markdown text from path/raw.md or path/markdown_raw.md."""
    file_path = path / "raw.md"
    if not file_path.exists():
        file_path = path / "markdown_raw.md"

    full_text = ""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == "":
                continue
            full_text += line
    return full_text.replace("**", "")


def _parse_md(text: str) -> Tuple[List[Any], List[str]]:
    """Parse markdown string into markdown_it tokens and source lines."""
    tokens = md.parse(text)
    lines = text.splitlines(keepends=True)
    return tokens, lines


def _write_input(path: Path) -> None:
    """Filter processed.md sentences matching S\\d{3} to input.txt."""
    lines = []
    with open(path / "processed.md", "r", encoding="utf-8") as f:
        for line in f:
            if is_sentence_line(line):
                lines.append(line)
    with open(path / "input.txt", "w", encoding="utf-8") as f:
        f.writelines(lines)


def _numbered_endnote_start(lines: List[str]) -> Optional[int]:
    """Return the line index where an Endnotes/Footnotes section starts."""
    return EndnoteSectionParser.find_endnote_start(lines)


def _is_plain_reference_line(text: str) -> bool:
    """Recognise an inline numbered bibliography entry (e.g. `24 FCA ...`)."""
    return re.match(r'^\s*\d+\s+\S[^\n]*$', text.strip()) is not None


def _plain_reference_line_number(text: str) -> Optional[str]:
    """The leading number of a plain numbered line (e.g. `24 FCA ...` -> `24`)."""
    match = re.match(r'^\s*(\d+)\s+\S', text.strip())
    return match.group(1) if match else None


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _is_superscript_reference_line(text: str) -> bool:
    """Recognise a line beginning with a superscript numeral definition."""
    return re.match(r'^\s*[⁰¹²³⁴⁵⁶⁷⁸⁹]+\s+\S', text.strip()) is not None


def _extract_numbered_endnotes(lines: List[str], start_line: Optional[int]) -> List[Dict[str, str]]:
    """Legacy helper for backward compatibility."""
    parser = EndnoteSectionParser()
    return parser.parse({"lines": lines, "endnote_start": start_line})


def parse_raw_footers(path: Path) -> List[Dict[str, str]]:
    """Parse raw footer files (references_raw.md, ref_raw.txt, etc.) in directory."""
    parser = RawFooterFileParser()
    return parser.parse({"path": path})


def process_type_1(tokens: List[Any], lines: List[str], out_dir: Path) -> None:
    """Legacy Type 1 processing function for superscript-defined references."""
    processor = MarkdownDocumentProcessor()
    processor.process(out_dir, tokens=tokens, lines=lines, out_dir=out_dir, parser_type="type_1")


def process_type_2(tokens: List[Any], lines: List[str], out_dir: Path) -> None:
    """Legacy Type 2 processing function for documents with an Endnotes section."""
    processor = MarkdownDocumentProcessor()
    processor.process(out_dir, tokens=tokens, lines=lines, out_dir=out_dir, parser_type="type_2")


def preprocess(path: Path, parser_type: str = "auto") -> None:
    """Primary preprocessing entry point for documents."""
    processor = MarkdownDocumentProcessor()
    processor.process(path, parser_type=parser_type)
    _write_input(path)


def extract_lines(document_path: Path, result_path: Path, aspect: str) -> None:
    """Extract classified sentence lines into text.txt."""
    with open(document_path / "input.txt", "r", encoding="utf-8") as f1:
        lines_by_id = {sentence_id(line): line for line in f1 if is_sentence_line(line)}

    rows = []
    with open(result_path / f"{aspect}/lines.txt", "r", encoding="utf-8") as f2:
        for line in f2:
            sid = line.strip()
            if sid:
                rows.append(lines_by_id[sid])

    with open(result_path / f"{aspect}/text.txt", "w", encoding="utf-8") as f:
        f.writelines(rows)


def _collect_markers(line: str, ref_acc: List[str], link_acc: List[str]) -> None:
    """Pull the inline "**(Reference: N)" / "**(Link: N)" markers out of a line."""
    if "**" not in line:
        return
    match = re.search(r'Reference:\s*([0-9a-zA-Z_\-]+(?:\s*,\s*[0-9a-zA-Z_\-]+)*)', line)
    if match is not None:
        ref_text = match.group(1)
        if "," in ref_text:
            ref_acc.extend(num.strip() for num in ref_text.split(","))
        else:
            ref_acc.append(ref_text.strip())

    # A line can carry multiple inline "**(Link: N)" markers
    # (unlike references, links aren't merged into one marker).
    for link_match in re.finditer(r'Link:\s*([0-9a-zA-Z_\-]+(?:\s*,\s*[0-9a-zA-Z_\-]+)*)', line):
        link_text = link_match.group(1)
        if "," in link_text:
            link_acc.extend(num.strip() for num in link_text.split(","))
        else:
            link_acc.append(link_text.strip())


def _write_citation_blocks(f, ref_acc: List[str], link_acc: List[str], references: list, links: list) -> None:
    f.write("<references>\n")
    for i in ref_acc:
        try:
            entry = next(x for x in references if x["number"] == i)
            f.write(f'<reference number={xml_quoteattr(i)}>{xml_escape(entry["text"])}</reference>\n')
        except StopIteration:
            print(f"Reference '{i}' not found in references.json")
    f.write("</references>\n")

    f.write("<links>\n")
    for i in link_acc:
        try:
            entry = next(x for x in links if x["number"] == i)
            f.write(
                f'<link number={xml_quoteattr(i)} url={xml_quoteattr(entry["url"])}>'
                f'{xml_escape(entry["text"])}</link>\n'
            )
        except StopIteration:
            print(f"Link '{i}' not found in links.json")
    f.write("</links>\n")


def construct_xml(document_path: Path, result_path: Path, aspect: str,
                  classification: Optional[dict] = None, out_name: str = "summary.xml") -> dict:
    """Build summary.xml associating classified themes with their referenced citations and links.

    Produces well-formed XML (single root, all text content escaped) so it can
    be read back with a standard XML parser rather than ad hoc regexes.

    The theme-grouping LLM call routinely omits some of the aspect's sentences
    from every theme (observed loss ranges from 0% to over 60% of an aspect),
    which silently hid those sentences from scoring.  Any sentence in the
    aspect's text.txt that no theme claimed is therefore emitted in a trailing
    <ungrouped> block, so the scorer always sees the whole aspect.  Returns a
    coverage summary for the caller to log.

    Each <theme> carries a sentences attribute (its line count), and, once
    weights.json exists for the aspect, a weight attribute (core / not core)
    matched to the theme by name.

    Pass classification={"themes": []} and an out_name to write the whole
    aspect as a single <ungrouped> block without touching summary.xml (used
    to score Testing as one block).
    """
    with open(document_path / "references.json", "r", encoding="utf-8") as f:
        references = json.load(f)

    links_path = document_path / "links.json"
    if links_path.exists():
        with open(links_path, "r", encoding="utf-8") as f:
            links = json.load(f)
    else:
        links = []

    if classification is not None:
        data = classification
    else:
        with open(result_path / f"{aspect}/classification.json", "r", encoding="utf-8") as f:
            data = json.load(f)

    with open(document_path / "input.txt", "r", encoding="utf-8") as f:
        input_lines = f.readlines()

    # The aspect's own sentence set is the yardstick for coverage: every line
    # in text.txt belongs to this aspect and should reach the scorer.
    with open(result_path / f"{aspect}/text.txt", "r", encoding="utf-8") as f:
        aspect_ids = [sentence_id(line) for line in f if is_sentence_line(line)]

    themed_ids = {sid for theme in data["themes"] for sid in theme["ids"]}
    ungrouped_ids = [sid for sid in aspect_ids if sid not in themed_ids]
    # The grouping call sometimes lists ids from outside the aspect (once, 431
    # of them padding "Other").  input.txt holds the whole document, so those
    # would pull other aspects' sentences into this one: keep only the
    # aspect's own lines, and drop a theme left with none.
    aspect_id_set = set(aspect_ids)
    outside_ids = themed_ids - aspect_id_set

    # Weights are matched by name; a list per name keeps duplicate theme
    # names apart, in the order they appear.
    weights_by_name: Dict[str, List[str]] = {}
    weights_path = result_path / f"{aspect}/weights.json"
    if weights_path.exists():
        with open(weights_path, "r", encoding="utf-8") as f:
            for entry in json.load(f)["themes"]:
                weights_by_name.setdefault(entry["theme"], []).append(entry["weight"])

    # Core themes named by find_core carry the claim they make, so the scorer
    # knows what it is looking for evidence of.
    claims: Dict[str, str] = {}
    core_path = result_path / "core_themes.json"
    if core_path.exists():
        with open(core_path, "r", encoding="utf-8") as f:
            core = json.load(f)
        for theme in core.get(aspect, {}).get("core_themes", []):
            claims[theme["name"]] = theme["claim"]

    with open(result_path / f"{aspect}/{out_name}", "w", encoding="utf-8") as f:
        f.write("<root>\n")
        for theme in data["themes"]:
            ref_theme: List[str] = []
            link_theme: List[str] = []
            theme_lines = [line for line in input_lines
                           if split_sentence_line(line)[0] in theme["ids"]
                           and split_sentence_line(line)[0] in aspect_id_set]
            if not theme_lines:
                print(f"Theme '{theme['name']}' in '{aspect}' has no sentences from this aspect; left out.")
                continue
            attrs = f' sentences="{len(theme_lines)}"'
            if weights_path.exists():
                weights = weights_by_name.get(theme["name"])
                if weights:
                    attrs += f" weight={xml_quoteattr(weights.pop(0))}"
                else:
                    print(f"No weight in weights.json for theme '{theme['name']}' in '{aspect}'")
            f.write(f"<theme{attrs}>\n")
            f.write(f"<name>{xml_escape(theme['name'])}</name>\n")
            if theme["name"] in claims:
                f.write(f"<claim>{xml_escape(claims[theme['name']])}</claim>\n")
            f.write("<text>\n")
            for line in theme_lines:
                line_id, body = split_sentence_line(line)
                f.write(f'<line id={xml_quoteattr(line_id)}>{xml_escape(body)}</line>\n')
                _collect_markers(line, ref_theme, link_theme)
            f.write("</text>\n")
            _write_citation_blocks(f, ref_theme, link_theme, references, links)
            f.write("</theme>\n")

        if ungrouped_ids:
            ref_rest: List[str] = []
            link_rest: List[str] = []
            ungrouped_lines = [line for line in input_lines if split_sentence_line(line)[0] in ungrouped_ids]
            f.write(f'<ungrouped sentences="{len(ungrouped_lines)}">\n')
            f.write("<text>\n")
            for line in ungrouped_lines:
                line_id, body = split_sentence_line(line)
                f.write(f'<line id={xml_quoteattr(line_id)}>{xml_escape(body)}</line>\n')
                _collect_markers(line, ref_rest, link_rest)
            f.write("</text>\n")
            _write_citation_blocks(f, ref_rest, link_rest, references, links)
            f.write("</ungrouped>\n")
        f.write("</root>\n")

    coverage = {
        "aspect": aspect,
        "total": len(aspect_ids),
        "themed": len(aspect_ids) - len(ungrouped_ids),
        "ungrouped": len(ungrouped_ids),
        "outside": len(outside_ids),
    }
    if outside_ids:
        print(f"Theme grouping for '{aspect}' listed {len(outside_ids)} sentence ids from outside "
              f"the aspect; ignored.")
    if ungrouped_ids and classification is None:
        share = 100 * len(ungrouped_ids) / len(aspect_ids) if aspect_ids else 0
        print(
            f"Theme grouping for '{aspect}' left {len(ungrouped_ids)} of {len(aspect_ids)} "
            f"sentences ({share:.0f}%) in no theme; emitted in <ungrouped>."
        )
    return coverage
