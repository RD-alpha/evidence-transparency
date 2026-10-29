"""Shared handling of the ``S####`` sentence identifiers used in processed.md/input.txt.

Sentence ids are written zero-padded to four digits (``S0001``).  Documents
regularly run past 999 sentences -- combined documents, which concatenate
several sources, especially -- and three-digit padding made those documents'
ids change width partway through, which fixed-width readers silently truncated.
Four digits covers every document in the corpus; anything past 9999 widens
again, so readers go through the helpers here rather than slicing.
"""

import re

# Accepts three-digit ids as well, so an artefact written before the move to
# four digits still reads back correctly.
SENTENCE_LINE_RE = re.compile(r'^(S\d{3,})[ \t]?(.*)$', re.DOTALL)

SENTENCE_ID_WIDTH = 4


def split_sentence_line(line: str) -> tuple[str | None, str]:
    """Split a processed.md/input.txt line into its ``S####`` id and its body.

    Returns ``(None, line)`` (with the trailing newline stripped) for lines that
    carry no sentence id, e.g. the heading and table lines in processed.md.
    """
    match = SENTENCE_LINE_RE.match(line.rstrip("\n"))
    if match is None:
        return None, line.rstrip("\n")
    return match.group(1), match.group(2)


def sentence_id(line: str) -> str | None:
    """Return just the ``S####`` id of a line, or None if it has none."""
    return split_sentence_line(line)[0]


def is_sentence_line(line: str) -> bool:
    return SENTENCE_LINE_RE.match(line.rstrip("\n")) is not None


def format_sentence_id(index: int) -> str:
    """Format a 1-based sentence index as an id (``S0001``, widening past 9999)."""
    return f"S{index:0{SENTENCE_ID_WIDTH}d}"


def format_sentence_line(index: int, body: str) -> str:
    """Format one ``S#### <body>`` line (no trailing newline)."""
    return f"{format_sentence_id(index)} {body}"
