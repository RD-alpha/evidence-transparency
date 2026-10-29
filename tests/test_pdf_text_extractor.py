import unittest
import tempfile
from pathlib import Path

import pymupdf

from web_app.pdf_text_extractor import analyze_pdf, find_reference_number_gaps, merge_ocr_page

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"

BODY_PARAGRAPH = (
    "This is a long paragraph of ordinary body text meant to dominate the character "
    "count of the page so the body font size is correctly identified as the most common "
    "size across the whole document, just as it would be in a real multi-page report."
)


def _build_pdf(path: Path, page_count: int = 2, blank_page: bool = False, scanned_page: bool = False):
    doc = pymupdf.open()
    for i in range(page_count):
        page = doc.new_page()
        page.insert_text((72, 40), "Sample Report", fontsize=11)
        page.insert_text((72, 100), f"Section {i + 1}", fontsize=22)
        page.insert_text((72, 140), BODY_PARAGRAPH, fontsize=12)
        page.insert_text((72, 200), BODY_PARAGRAPH, fontsize=12)
        # Two-tier marker+text, matching the real document's rendering style
        # (a small superscript-style number followed by smaller reference text).
        page.insert_text((72, 700), f"{i + 1}", fontsize=6.5)
        page.insert_text((82, 700), f" A reference entry number {i + 1} at small size.", fontsize=10)

    if blank_page:
        doc.new_page()

    if scanned_page:
        page = doc.new_page()
        pix = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 10, 10), False)
        pix.set_rect(pix.irect, (200, 0, 0))
        page.insert_image(pymupdf.Rect(72, 72, 300, 300), pixmap=pix)

    doc.save(str(path))
    doc.close()


class TestAnalyzePdf(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        for f in self.tmp_dir.glob("*"):
            f.unlink()
        self.tmp_dir.rmdir()

    def test_classifies_headings_paragraphs_and_references_by_font_size(self):
        pdf_path = self.tmp_dir / "doc.pdf"
        _build_pdf(pdf_path, page_count=1)

        result = analyze_pdf(str(pdf_path))
        self.assertEqual(result["page_count"], 1)

        page = result["pages"][0]
        types = [b["type"] for b in page["blocks"]]
        self.assertIn("heading", types)
        self.assertIn("paragraph", types)
        self.assertEqual(page["references"], [
            {"number": "1", "text": "A reference entry number 1 at small size."}
        ])

    def test_strips_repeated_running_header(self):
        pdf_path = self.tmp_dir / "doc.pdf"
        _build_pdf(pdf_path, page_count=2)

        result = analyze_pdf(str(pdf_path))
        for page in result["pages"]:
            texts = [b["text"] for b in page["blocks"]]
            self.assertNotIn("Sample Report", texts)

    def test_blank_page_is_not_flagged_for_ocr(self):
        pdf_path = self.tmp_dir / "doc.pdf"
        _build_pdf(pdf_path, page_count=1, blank_page=True)

        result = analyze_pdf(str(pdf_path))
        blank_page = result["pages"][-1]
        self.assertFalse(blank_page["needs_ocr"])
        self.assertEqual(blank_page["blocks"], [])
        self.assertEqual(blank_page["references"], [])

    def test_page_with_image_and_no_text_is_flagged_for_ocr(self):
        pdf_path = self.tmp_dir / "doc.pdf"
        _build_pdf(pdf_path, page_count=1, scanned_page=True)

        result = analyze_pdf(str(pdf_path))
        scanned_page = result["pages"][-1]
        self.assertTrue(scanned_page["needs_ocr"])
        self.assertIsNotNone(scanned_page["ocr_reason"])

    def test_body_hyperlink_is_annotated_inline_and_recorded(self):
        pdf_path = self.tmp_dir / "doc.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 140), "See the full report for more detail.", fontsize=12)
        rect = page.search_for("full report")[0]
        page.insert_link({"kind": pymupdf.LINK_URI, "from": rect, "uri": "https://example.com/report"})
        doc.save(str(pdf_path))
        doc.close()

        result = analyze_pdf(str(pdf_path))
        paragraph_texts = [b["text"] for b in result["pages"][0]["blocks"] if b["type"] == "paragraph"]
        self.assertTrue(any("full report **(Link: 1)" in text for text in paragraph_texts))
        self.assertEqual(result["links"], [
            {"number": "1", "url": "https://example.com/report", "text": "full report"}
        ])

    def test_internal_navigation_link_is_not_annotated(self):
        pdf_path = self.tmp_dir / "doc.pdf"
        doc = pymupdf.open()
        doc.new_page()
        doc.new_page()
        page = doc[0]
        page.insert_text((72, 140), "See the section below for more detail.", fontsize=12)
        rect = page.search_for("section below")[0]
        page.insert_link({"kind": pymupdf.LINK_GOTO, "from": rect, "page": 1})
        doc.save(str(pdf_path))
        doc.close()

        result = analyze_pdf(str(pdf_path))
        paragraph_texts = [b["text"] for b in result["pages"][0]["blocks"] if b["type"] == "paragraph"]
        self.assertTrue(any(text == "See the section below for more detail." for text in paragraph_texts))
        self.assertEqual(result["links"], [])


class TestFindReferenceNumberGaps(unittest.TestCase):
    def test_contiguous_run_has_no_gaps_or_duplicates(self):
        result = find_reference_number_gaps(["1", "2", "3"])
        self.assertEqual(result, {"missing": [], "duplicated": []})

    def test_reports_missing_numbers(self):
        result = find_reference_number_gaps(["1", "2", "4"])
        self.assertEqual(result, {"missing": [3], "duplicated": []})

    def test_reports_duplicated_numbers(self):
        result = find_reference_number_gaps(["1", "2", "2", "3"])
        self.assertEqual(result, {"missing": [], "duplicated": [2]})

    def test_ignores_non_digit_numbers(self):
        result = find_reference_number_gaps(["1", "2", "Smith2021"])
        self.assertEqual(result, {"missing": [], "duplicated": []})

    def test_empty_input_has_no_gaps(self):
        self.assertEqual(find_reference_number_gaps([]), {"missing": [], "duplicated": []})


@unittest.skipUnless(DOCUMENTS_DIR.is_dir(), "documents/ directory not present")
class TestNoReferenceGapsAcrossDocuments(unittest.TestCase):
    """Regression check for the water-impact-assessment doc, which silently
    dropped references 21-24 (and later 55-56) -- caught only by manually
    diffing reference numbers against range(1, max+1). Runs against every
    checked-in document so a repeat shows up here instead of by hand.
    """

    def test_no_reference_gaps_or_duplicates(self):
        pdf_paths = sorted(DOCUMENTS_DIR.glob("*/doc.pdf"))
        self.assertTrue(pdf_paths, "expected at least one documents/*/doc.pdf to check")

        failures = []
        for pdf_path in pdf_paths:
            result = analyze_pdf(str(pdf_path))
            if result["reference_gaps"] or result["reference_duplicates"]:
                failures.append(
                    f"{pdf_path.parent.name}: missing={result['reference_gaps']} "
                    f"duplicated={result['reference_duplicates']}"
                )

        self.assertEqual(failures, [], "\n" + "\n".join(failures))


class TestMergeOcrPage(unittest.TestCase):
    def test_converts_markdown_body_and_numbered_footer(self):
        body_markdown = "# A Heading\n\nSome recovered paragraph text.\n"
        footer_text = "1 A recovered reference entry.\n2 Another recovered entry."

        blocks, references = merge_ocr_page(body_markdown, footer_text)

        self.assertEqual(
            [b["type"] for b in blocks],
            ["heading", "paragraph"],
        )
        self.assertEqual(references, [
            {"number": "1", "text": "A recovered reference entry."},
            {"number": "2", "text": "Another recovered entry."},
        ])

    def test_empty_footer_yields_no_references(self):
        blocks, references = merge_ocr_page("Some text.\n", "")
        self.assertEqual(references, [])


if __name__ == "__main__":
    unittest.main()
