#!/usr/bin/env python3
"""
GOV.UK HTML Parser

Extracts main content from div.govspeak (excluding footnote references)
and extracts footnote references from div.footnotes or similar containers.

Usage:
    python govuk_parser.py <url>
    python govuk_parser.py <file.html>
"""

import sys
import re
import json
from pathlib import Path
from urllib.parse import urlparse

from sentence_ids import format_sentence_line

try:
    import requests
    from bs4 import BeautifulSoup, NavigableString, Comment
    USE_REQUESTS = True
except ImportError:
    USE_REQUESTS = False

def install_dependencies():
    """Install required packages if missing."""
    print("Installing required packages: requests, beautifulsoup4...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "requests", "beautifulsoup4"], check=True)
    print("Dependencies installed. Please re-run the script.")
    sys.exit(0)

class GovUKParser:
    """Parser for GOV.UK HTML documents."""

    # Patterns for footnote references in text
    FOOTNOTE_PATTERNS = [
        r'\[footnote\s+\d+\]',  # [footnote 1]
        r'\[fn:\d+\]',          # [fn:1]
        r'\[\d+\]',             # [1]
    ]

    # Internal placeholder used to mark a footnote reference's position in the
    # text before sentence-level merging collapses multiple references into a
    # single "** (Reference: X, Y)" marker. Uses NUL bytes so it can't collide
    # with real document text.
    _REF_PLACEHOLDER_RE = re.compile(r'\x00REF:([^\x00]+)\x00')

    # Matches a sentence boundary (end punctuation + whitespace), tolerating
    # reference placeholders sitting between the punctuation and the
    # whitespace (e.g. "...text.\x00REF:1\x00 Next sentence").
    _SENTENCE_BOUNDARY_RE = re.compile(
        r'([.!?])((?:\s*\x00REF:[^\x00]+\x00)*)\s+(?=[A-Z0-9"\'‘“(])'
    )

    # Classes used by kramdown/govspeak for the footnote "back to content"
    # link, which should not be treated as a citation source URL.
    _BACKREF_CLASS_RE = re.compile(r'revers[e]?footnote|footnote-?backref|footnote-?return', re.IGNORECASE)

    # Matches a run of two or more reference placeholders that sit back-to-back
    # with nothing but whitespace/comma/semicolon separating them (e.g. two
    # adjacent <sup> footnotes, or "<a>14</a>, <a>15</a>, <a>16</a>"). Used to
    # merge them into one reference as soon as they're seen, before any
    # sentence-level text is resolved, and to swallow the connecting
    # punctuation so it doesn't leak into the output as stray commas.
    _REF_CLUSTER_RE = re.compile(
        r'\x00REF:[^\x00]+\x00'
        r'(?:[ \t]*[,;][ \t]*\x00REF:[^\x00]+\x00|[ \t]+\x00REF:[^\x00]+\x00)+'
    )

    def __init__(self, html_content=None, url=None):
        """
        Initialize parser.

        Args:
            html_content: Raw HTML string
            url: URL to fetch HTML from
        """
        self.html_content = html_content
        self.url = url
        self.soup = None
        # Body hyperlinks found by the last extract_govspeak_text() call,
        # e.g. [{'number': '1', 'url': '...', 'text': '...'}], analogous to
        # the entries written to references.json for footnotes.
        self.links = []
        self._parse()

    def _parse(self):
        """Parse HTML content."""
        if not self.html_content:
            if self.url:
                self.html_content = self._fetch_url(self.url)
            else:
                raise ValueError("No HTML content or URL provided")

        self.soup = BeautifulSoup(self.html_content, 'html.parser')

    def _fetch_url(self, url):
        """Fetch HTML from URL."""
        if not USE_REQUESTS:
            install_dependencies()

        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; GovUKParser/1.0; +https://github.com)'
        }

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch URL {url}: {e}")

    def _clean_text(self, text):
        """Clean and normalize text."""
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    @staticmethod
    def _ref_placeholder(fn_id):
        """Build an internal placeholder marking a footnote reference's position."""
        return f"\x00REF:{fn_id}\x00"

    def _extract_body_links(self, element):
        """
        Replace hyperlinks in the main body with "{link text}**(Link: N)"
        markers, recording each one (in document order) on self.links.

        Unlike footnote references, link markers are left inline right after
        their own link text rather than merged to the end of the sentence,
        since each one annotates a specific phrase rather than the sentence
        as a whole. In-page anchors (href="#...") are left untouched.
        """
        for a in element.find_all('a', href=True):
            href = a['href'].strip()
            if not href or href.startswith('#'):
                continue

            link_text = a.get_text()
            index = str(len(self.links) + 1)
            self.links.append({
                'number': index,
                'url': href,
                'text': self._clean_text(link_text),
            })
            a.replace_with(f"{link_text}**(Link: {index})")

    def _replace_footnote_references(self, element):
        """
        Replace footnote reference elements in the content with (Reference: X).

        Multiple references within the same sentence are merged into a single
        "** (Reference: X, Y, Z)" marker at the end of that sentence.

        Args:
            element: BeautifulSoup element

        Returns:
            Text with footnote references replaced by (Reference: X)
        """
        if not element:
            return ""

        self.links = []

        # Create a copy to avoid modifying original
        import copy
        element_copy = copy.deepcopy(element)

        # Process superscript elements first (they may contain footnote links)
        for sup in element_copy.find_all('sup'):
            # Check if this sup contains a footnote link
            fn_link = sup.find('a', href=re.compile(r'#fn:\d+', re.IGNORECASE))
            if fn_link:
                # Extract the number from the link
                fn_id = re.search(r'fn:(\d+)', fn_link.get('href', ''), re.IGNORECASE)
                if fn_id:
                    sup.replace_with(self._ref_placeholder(fn_id.group(1)))
                else:
                    sup.decompose()
            else:
                # Check if sup contains a footnote marker in its text
                fn_id = re.search(r'\[(\d+)\]', sup.get_text(), re.IGNORECASE)
                if fn_id:
                    sup.replace_with(self._ref_placeholder(fn_id.group(1)))
                else:
                    sup.decompose()

        # Replace footnote link elements
        for fn_link in element_copy.find_all('a', href=re.compile(r'#fn:\d+', re.IGNORECASE)):
            fn_id = re.search(r'fn:(\d+)', fn_link.get('href', ''), re.IGNORECASE)
            if fn_id:
                fn_link.replace_with(self._ref_placeholder(fn_id.group(1)))

        for fn_link in element_copy.find_all('a', id=re.compile(r'fnref:\d+', re.IGNORECASE)):
            fn_id = re.search(r'fnref:(\d+)', fn_link.get('id', ''), re.IGNORECASE)
            if fn_id:
                fn_link.replace_with(self._ref_placeholder(fn_id.group(1)))

        for fn_link in element_copy.find_all('a', class_=re.compile(r'footnote', re.IGNORECASE)):
            link_text = fn_link.get_text()
            # Try to extract number from various formats: [footnote 1], [fn:1], [1], fn:1, etc.
            fn_id = re.search(r'(?:footnote\s+|fn:|fn\.|fn-)?(\d+)', link_text, re.IGNORECASE)
            if fn_id:
                fn_link.replace_with(self._ref_placeholder(fn_id.group(1)))
            else:
                fn_link.decompose()

        # Any <a href> left at this point is a genuine body link (footnote
        # links were already consumed above) — annotate it in place with a
        # "**(Link: N)" marker and record it on self.links, analogous to how
        # footnotes are recorded for references.json.
        self._extract_body_links(element_copy)

        # Get text
        text = element_copy.get_text(separator='\n')

        # Replace remaining footnote patterns from text with a reference placeholder
        for pattern in self.FOOTNOTE_PATTERNS:
            def replace_footnote(match):
                # Extract the number from the pattern
                match_str = match.group(0)
                num_match = re.search(r'\d+', match_str)
                if num_match:
                    return self._ref_placeholder(num_match.group())
                return ""

            text = re.sub(pattern, replace_footnote, text, flags=re.IGNORECASE)

        # Clean up whitespace, then merge back-to-back reference placeholders
        # (e.g. two adjacent footnotes) as soon as they're found, and finally
        # merge any remaining references scattered across the same sentence
        # into a single combined "** (Reference: X, Y)" marker.
        text = self._clean_text(text)
        text = self._consolidate_adjacent_refs(text)
        text = self._merge_sentence_references(text)

        return text

    def _consolidate_adjacent_refs(self, text):
        """
        Merge reference placeholders that sit back-to-back (separated only by
        whitespace/comma/semicolon) into a single placeholder, consuming the
        connecting punctuation so it doesn't leak into the final text.
        """
        if '\x00' not in text:
            return text

        def _repl(match):
            ref_ids = self._REF_PLACEHOLDER_RE.findall(match.group(0))
            unique_ids = list(dict.fromkeys(ref_ids))
            return self._ref_placeholder(', '.join(unique_ids))

        return self._REF_CLUSTER_RE.sub(_repl, text)

    def _merge_sentence_references(self, text):
        """
        Collapse per-occurrence reference placeholders into one combined
        "** (Reference: X, Y, Z)" marker at the end of each sentence.
        """
        if '\x00' not in text:
            return text

        # Mark sentence boundaries with a unique delimiter, keeping any
        # reference placeholders attached to the sentence they belong to.
        marked = self._SENTENCE_BOUNDARY_RE.sub(
            lambda m: f"{m.group(1)}{m.group(2)}\x01", text
        )

        merged_sentences = []
        for sentence in marked.split('\x01'):
            ref_ids = self._REF_PLACEHOLDER_RE.findall(sentence)
            cleaned = self._REF_PLACEHOLDER_RE.sub('', sentence)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()

            if ref_ids:
                unique_ids = list(dict.fromkeys(ref_ids))
                marker = f"** (Reference: {', '.join(unique_ids)})"
                cleaned = f"{cleaned} {marker}" if cleaned else marker

            if cleaned:
                merged_sentences.append(cleaned)

        return ' '.join(merged_sentences)

    def extract_govspeak_text(self, exclude_footnotes=True):
        """
        Extract text from div.govspeak, excluding the references section.

        Args:
            exclude_footnotes: If True, replace footnote references with (Reference: X)

        Returns:
            Tuple of (text, cleaned_text)
            - text: Raw text with footnote markers
            - cleaned_text: Text with footnote references replaced
        """
        govspeak_div = self.soup.find('div', class_='govspeak')

        if not govspeak_div:
            # Try alternative class names
            for class_name in ['govspeak', 'document', 'content', 'main-content']:
                govspeak_div = self.soup.find('div', class_=class_name)
                if govspeak_div:
                    break
            else:
                # Try main content area
                govspeak_div = self.soup.find('main') or self.soup.find('article')

        if not govspeak_div:
            return "", ""

        # Create a copy to avoid modifying the original soup
        import copy
        govspeak_copy = copy.deepcopy(govspeak_div)

        # Find and remove references/footnotes section from the copy
        # This ensures references text doesn't appear in the main content
        footnotes_in_govspeak = govspeak_copy.find('div', class_=re.compile(r'footnotes|references', re.IGNORECASE))
        if footnotes_in_govspeak:
            footnotes_in_govspeak.decompose()

        # Also check for h2 with "References" or "Footnotes"
        for heading in govspeak_copy.find_all(['h2', 'h3', 'h4']):
            if re.search(r'(references|footnotes)', heading.get_text(), re.IGNORECASE):
                # Remove this heading and all following siblings
                for sibling in heading.find_next_siblings():
                    sibling.decompose()
                heading.decompose()
                break

        # Get raw text from original (before modifications)
        raw_text = govspeak_div.get_text(separator='\n')
        raw_text = self._clean_text(raw_text)

        if not exclude_footnotes:
            return raw_text, raw_text

        # Replace footnote references with (Reference: X) on the copy
        cleaned_text = self._replace_footnote_references(govspeak_copy)

        return raw_text, cleaned_text

    def _extract_urls(self, element):
        """
        Extract external link URLs from a footnote element, excluding
        in-page anchors and footnote "back to content" links.

        Args:
            element: BeautifulSoup element containing a footnote's content

        Returns:
            List of URLs found in the element, in document order, deduped.
        """
        urls = []
        for a in element.find_all('a', href=True):
            href = a['href'].strip()
            if not href or href.startswith('#'):
                continue

            classes = a.get('class') or []
            if isinstance(classes, str):
                classes = [classes]
            if any(self._BACKREF_CLASS_RE.search(c) for c in classes):
                continue

            if href not in urls:
                urls.append(href)

        return urls

    @staticmethod
    def _normalize_fn_id(fn_id):
        """Strip a leading 'fn:' prefix so footnote keys are just '1', '2', etc."""
        return re.sub(r'^fn:', '', fn_id, flags=re.IGNORECASE)

    def extract_footnotes(self):
        """
        Extract footnote references from the document.

        Returns:
            Dictionary of {footnote_id: {'text': footnote_text, 'urls': [...]}}
        """
        footnotes = {}

        # Try to find footnotes section
        footnotes_section = self.soup.find('div', class_='footnotes')

        if not footnotes_section:
            # Try alternative selectors
            selectors = [
                'div.footnotes',
                'section.footnotes',
                'div#footnotes',
                'aside.footnotes',
                'div.fn',
                'div.footnote',
            ]
            for selector in selectors:
                footnotes_section = self.soup.select_one(selector)
                if footnotes_section:
                    break

        if footnotes_section:
            # Extract individual footnotes
            # Pattern 1: <div id="fn:1"> or <div class="footnote" id="fn:1">
            fn_divs = footnotes_section.find_all('div', id=re.compile(r'fn:\d+', re.IGNORECASE))

            for fn_div in fn_divs:
                fn_id = self._normalize_fn_id(fn_div.get('id', ''))
                fn_text = fn_div.get_text(separator='\n')
                fn_text = self._clean_text(fn_text)
                if fn_id and fn_text:
                    footnotes[fn_id] = {'text': fn_text, 'urls': self._extract_urls(fn_div)}

            # Pattern 2: <li id="fn:1">
            fn_lis = footnotes_section.find_all('li', id=re.compile(r'fn:\d+', re.IGNORECASE))
            for fn_li in fn_lis:
                fn_id = self._normalize_fn_id(fn_li.get('id', ''))
                fn_text = fn_li.get_text(separator='\n')
                fn_text = self._clean_text(fn_text)
                if fn_id and fn_text:
                    footnotes[fn_id] = {'text': fn_text, 'urls': self._extract_urls(fn_li)}

            # Pattern 3: <p class="footnote">
            fn_paras = footnotes_section.find_all('p', class_=re.compile(r'footnote', re.IGNORECASE))
            for i, fn_para in enumerate(fn_paras):
                fn_text = fn_para.get_text(separator='\n')
                fn_text = self._clean_text(fn_text)
                if fn_text:
                    footnotes[str(i + 1)] = {'text': fn_text, 'urls': self._extract_urls(fn_para)}
        else:
            # Try to find footnotes by ID pattern anywhere in document
            all_fn_divs = self.soup.find_all('div', id=re.compile(r'fn:\d+', re.IGNORECASE))
            for fn_div in all_fn_divs:
                fn_id = self._normalize_fn_id(fn_div.get('id', ''))
                fn_text = fn_div.get_text(separator='\n')
                fn_text = self._clean_text(fn_text)
                if fn_id and fn_text:
                    footnotes[fn_id] = {'text': fn_text, 'urls': self._extract_urls(fn_div)}

            # Also try list items
            all_fn_lis = self.soup.find_all('li', id=re.compile(r'fn:\d+', re.IGNORECASE))
            for fn_li in all_fn_lis:
                fn_id = self._normalize_fn_id(fn_li.get('id', ''))
                fn_text = fn_li.get_text(separator='\n')
                fn_text = self._clean_text(fn_text)
                if fn_id and fn_text:
                    footnotes[fn_id] = {'text': fn_text, 'urls': self._extract_urls(fn_li)}

        return footnotes

    def extract_footnote_references_from_text(self, text):
        """
        Extract footnote reference markers from text.

        Args:
            text: Text to search for footnote references

        Returns:
            List of footnote reference strings found
        """
        references = []

        # Find all footnote patterns
        for pattern in self.FOOTNOTE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend(matches)

        return references

    def parse(self):
        """
        Parse the document and return structured data.

        Returns:
            Dictionary with:
            - govspeak_text: Raw text from govspeak div
            - govspeak_text_clean: Text with footnote references replaced by (Reference: X)
              and body hyperlinks annotated in place with (Link: X)
            - footnotes: Dictionary of footnote IDs to text
            - links: List of body hyperlinks found, e.g. {'number': '1', 'url': ..., 'text': ...}
            - footnote_references: List of footnote reference markers found (kept for debugging)
        """
        raw_text, clean_text = self.extract_govspeak_text(exclude_footnotes=True)
        footnotes = self.extract_footnotes()
        # references = self.extract_footnote_references_from_text(raw_text)  # Commented out per request

        return {
            'govspeak_text': raw_text,
            'govspeak_text_clean': clean_text,
            'footnotes': footnotes,
            'links': self.links,
            # 'footnote_references': references,  # Commented out per request
        }

def parse_file(file_path):
    """Parse a local HTML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    parser = GovUKParser(html_content=html_content)
    return parser.parse()

def parse_url(url):
    """Parse a URL."""
    parser = GovUKParser(url=url)
    return parser.parse()


# ==============================================================================
# Document Artifact Generation
#
# Converts a GovUKParser.parse() result into the same processed.md /
# references.json / input.txt artifacts that preprocess.py produces for
# uploaded PDFs, so downstream analysis works the same regardless of whether
# a document came from a file upload or a GOV.UK URL.
# ==============================================================================

# A sentence boundary in the already-resolved govspeak_text_clean text: end
# punctuation (including the closing ")" of a "**(Reference: X)" marker)
# followed by whitespace and the start of the next sentence.
_PLAIN_SENTENCE_BOUNDARY_RE = re.compile(
    r'(?<=[.!?)])\s+(?=[A-Z0-9"\'‘“(])'
)


def split_into_sentences(text):
    """Split cleaned govspeak text (with references already resolved) into sentences."""
    if not text:
        return []
    return [s.strip() for s in _PLAIN_SENTENCE_BOUNDARY_RE.split(text) if s.strip()]


def build_processed_markdown(sentences):
    """Format sentences as S-numbered lines, matching preprocess.py's processed.md format."""
    return "".join(
        format_sentence_line(idx, sentence) + "\n" for idx, sentence in enumerate(sentences, start=1)
    )


def build_references_list(footnotes):
    """Convert a {id: {'text': ..., 'urls': [...]}} footnotes dict into a references.json-style list."""
    def sort_key(fn_id):
        return (0, int(fn_id)) if fn_id.isdigit() else (1, fn_id)

    return [
        {
            "number": fn_id,
            "text": footnotes[fn_id].get("text", ""),
            "urls": footnotes[fn_id].get("urls", []),
        }
        for fn_id in sorted(footnotes.keys(), key=sort_key)
    ]


def write_document_artifacts(results, document_dir):
    """
    Write processed.md, references.json, links.json, and input.txt for a page
    parsed by GovUKParser into document_dir.

    Args:
        results: The dict returned by GovUKParser.parse()
        document_dir: Directory to write the artifacts into (created if needed)

    Returns:
        Dict with 'sentence_count', 'reference_count', and 'link_count'.
    """
    document_dir = Path(document_dir)
    document_dir.mkdir(parents=True, exist_ok=True)

    sentences = split_into_sentences(results.get('govspeak_text_clean', ''))
    references = build_references_list(results.get('footnotes', {}))
    links = results.get('links', [])
    processed_md = build_processed_markdown(sentences)

    (document_dir / "processed.md").write_text(processed_md, encoding="utf-8")
    (document_dir / "references.json").write_text(
        json.dumps(references, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (document_dir / "links.json").write_text(
        json.dumps(links, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Every line in processed_md is already an "S###" sentence line (unlike
    # preprocess.py's version, there are no heading/table lines to filter out).
    (document_dir / "input.txt").write_text(processed_md, encoding="utf-8")

    return {
        "sentence_count": len(sentences),
        "reference_count": len(references),
        "link_count": len(links),
    }


def print_results(results, verbose=True):
    """Print results in a readable format."""
    print("=" * 80)
    print("GOV.UK Document Parser Results")
    print("=" * 80)

    if verbose:
        print("\n--- Main Content (with references replaced) ---")
        print(results['govspeak_text_clean'][:2000] + "..." if len(results['govspeak_text_clean']) > 2000 else results['govspeak_text_clean'])

    # print("\n--- Footnote References Found ---")
    # refs = results.get('footnote_references', [])
    # if refs:
    #     for ref in refs:
    #         print(f"  - {ref}")
    # else:
    #     print("  None found")

    print("\n--- Footnotes Extracted ---")
    footnotes = results['footnotes']
    if footnotes:
        for fn_id, fn_data in footnotes.items():
            print(f"\n[{fn_id}]")
            print(f"  {fn_data['text']}")
            if fn_data.get('urls'):
                print(f"  URLs: {', '.join(fn_data['urls'])}")
    else:
        print("  None found")

    print("\n--- Links Extracted ---")
    links = results.get('links', [])
    if links:
        for link in links:
            print(f"  [{link['number']}] {link['text']} -> {link['url']}")
    else:
        print("  None found")

    print("\n" + "=" * 80)
    print(f"Statistics:")
    print(f"  Main content length: {len(results['govspeak_text_clean'])} characters")
    # print(f"  Footnote references found: {len(refs)}")
    print(f"  Footnotes extracted: {len(footnotes)}")
    print(f"  Links extracted: {len(links)}")
    print("=" * 80)

def save_results(results, output_file):
    """Save results to a JSON file."""
    # Prepare for JSON serialization
    output = {
        'govspeak_text_clean': results['govspeak_text_clean'],
        # 'footnote_references': results.get('footnote_references', []),
        'footnotes': results['footnotes'],
        'links': results.get('links', []),
        'stats': {
            'content_length': len(results['govspeak_text_clean']),
            # 'reference_count': len(results.get('footnote_references', [])),
            'footnote_count': len(results['footnotes']),
            'link_count': len(results.get('links', [])),
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_file}")

def main():
    """Main entry point."""
    if not USE_REQUESTS:
        install_dependencies()

    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  {sys.argv[0]} <url>")
        print(f"  {sys.argv[0]} <file.html>")
        print(f"  {sys.argv[0]} <url> --output <file.json>")
        print(f"  {sys.argv[0]} <url> --quiet")
        sys.exit(1)

    target = sys.argv[1]
    output_file = None
    quiet = False

    # Parse arguments
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == '--output' and i + 1 < len(args):
            output_file = args[i + 1]
            i += 2
        elif args[i] in ('--quiet', '-q'):
            quiet = True
            i += 1
        else:
            i += 1

    # Determine if it's a URL or file
    if target.startswith(('http://', 'https://')):
        results = parse_url(target)
    else:
        file_path = Path(target)
        if file_path.exists():
            results = parse_file(target)
        else:
            print(f"Error: File not found: {target}")
            sys.exit(1)

    if output_file:
        save_results(results, output_file)

    if not quiet:
        print_results(results, verbose=not output_file)

    return results

if __name__ == '__main__':
    main()