import unittest
import json
from pathlib import Path
import tempfile
import shutil

from web_app.preprocess import (
    InTextCitationParser,
    UnicodeSuperscriptCitationExtractor,
    LatexSuperscriptCitationExtractor,
    MarkdownFootnoteCitationExtractor,
    HtmlSuperscriptCitationExtractor,
    PunctuationAttachedCitationExtractor,
    expand_numeric_range_or_list,
    ReferenceDefinitionExtractor,
    EndnoteSectionParser,
    RawFooterFileParser,
    InlineFootnoteParser,
    SuperscriptDefinitionParser,
    PlainNumberedLineParser,
    MarkdownDocumentProcessor,
    preprocess,
    parse_raw_footers,
    construct_xml,
)
import xml.etree.ElementTree as ET


class TestInTextCitationParsing(unittest.TestCase):
    def setUp(self):
        self.parser = InTextCitationParser()

    def test_expand_numeric_range_or_list(self):
        self.assertEqual(expand_numeric_range_or_list("1"), ["1"])
        self.assertEqual(expand_numeric_range_or_list("1, 2, 3"), ["1", "2", "3"])
        self.assertEqual(expand_numeric_range_or_list("1-4"), ["1", "2", "3", "4"])
        self.assertEqual(expand_numeric_range_or_list("1–3"), ["1", "2", "3"])
        self.assertEqual(expand_numeric_range_or_list("¹⁻³"), ["1", "2", "3"])
        self.assertEqual(expand_numeric_range_or_list("¹,²"), ["1", "2"])

    def test_unicode_superscript_single(self):
        text = "This is a statement with evidence.¹"
        clean, refs = self.parser.extract_citations(text)
        self.assertEqual(refs, ["1"])
        self.assertEqual(clean, "This is a statement with evidence.")

    def test_unicode_superscript_multiple_digits(self):
        text = "Another statement with double digit reference.¹²"
        clean, refs = self.parser.extract_citations(text)
        self.assertEqual(refs, ["12"])
        self.assertEqual(clean, "Another statement with double digit reference.")

    def test_unicode_superscript_separated(self):
        text = "A statement with multiple references.¹,²"
        clean, refs = self.parser.extract_citations(text)
        self.assertEqual(refs, ["1", "2"])
        self.assertEqual(clean, "A statement with multiple references.")

    def test_latex_superscript(self):
        text = "Life sciences employment was based at companies registered in London.$^{94}$"
        clean, refs = self.parser.extract_citations(text)
        self.assertEqual(refs, ["94"])
        self.assertEqual(clean, "Life sciences employment was based at companies registered in London.")

    def test_latex_superscript_comma_and_range(self):
        text = "Investment in diagnostics was proven.$^{1,2,5-7}$"
        clean, refs = self.parser.extract_citations(text)
        self.assertEqual(refs, ["1", "2", "5", "6", "7"])
        self.assertEqual(clean, "Investment in diagnostics was proven.")

    def test_markdown_footnote_citation(self):
        text = "According to recent figures[^1], the deficit has narrowed."
        clean, refs = self.parser.extract_citations(text)
        self.assertEqual(refs, ["1"])
        self.assertEqual(clean, "According to recent figures, the deficit has narrowed.")

    def test_html_superscript(self):
        text = "Outcomes improved significantly<sup>3,4</sup> across all regions."
        clean, refs = self.parser.extract_citations(text)
        self.assertEqual(refs, ["3", "4"])
        self.assertEqual(clean, "Outcomes improved significantly across all regions.")

    def test_punctuation_attached_citation(self):
        known = {"24", "26"}
        text = "This matches Open Banking.24 Furthermore, in the Vision,26 progress was noted."
        clean, refs = self.parser.extract_citations(text, known_numbers=known)
        self.assertEqual(refs, ["24", "26"])
        self.assertEqual(clean, "This matches Open Banking. Furthermore, in the Vision, progress was noted.")

    def test_punctuation_attached_ignores_unknown_numbers_and_decimals(self):
        known = {"24"}
        text = "The rate was 3.14% in 2024. Next sentence."
        clean, refs = self.parser.extract_citations(text, known_numbers=known)
        self.assertEqual(refs, [])
        self.assertEqual(clean, "The rate was 3.14% in 2024. Next sentence.")

    def test_punctuation_attached_ignores_paragraph_numbers(self):
        known = {"1", "8"}
        text = "1.8 Industry responses argued that time was short."
        clean, refs = self.parser.extract_citations(text, known_numbers=known)
        self.assertEqual(refs, [])
        self.assertEqual(clean, text)

        text = "3.2.1 The following chapters summarise the responses."
        clean, refs = self.parser.extract_citations(text, known_numbers=known)
        self.assertEqual(refs, [])
        self.assertEqual(clean, text)

    def test_punctuation_attached_still_matches_after_paragraph_number(self):
        known = {"8"}
        text = "1.4 Responses were mixed.8 Some argued otherwise."
        clean, refs = self.parser.extract_citations(text, known_numbers=known)
        self.assertEqual(refs, ["8"])
        self.assertEqual(clean, "1.4 Responses were mixed. Some argued otherwise.")

    def test_clean_punctuation_and_whitespace(self):
        text = "Here is some text   ¹   with bad spaces."
        clean, refs = self.parser.extract_citations(text)
        self.assertEqual(refs, ["1"])
        self.assertEqual(clean, "Here is some text with bad spaces.")


class TestReferenceDefinitionParsing(unittest.TestCase):
    def test_endnote_section_parser(self):
        lines = [
            "# Main Title\n",
            "Some content here.\n",
            "## Endnotes\n",
            "1 Department for Transport (2025). National Travel Survey.\n",
            "   Continued on next line.\n",
            "2 Office for National Statistics (2024).\n",
        ]
        parser = EndnoteSectionParser()
        refs = parser.parse({"lines": lines})
        self.assertEqual(len(refs), 2)
        self.assertEqual(refs[0]["number"], "1")
        self.assertIn("Continued on next line.", refs[0]["text"])
        self.assertEqual(refs[1]["number"], "2")
        self.assertEqual(refs[1]["text"], "Office for National Statistics (2024).")

    def test_superscript_definition_parser(self):
        lines = [
            "¹ Calculated using 2024 UK GDP data (GDP = £2,848 billion).\n",
            "² DBT (March 2026): Potential Economic Impact.\n",
        ]
        parser = SuperscriptDefinitionParser()
        refs = parser.parse({"lines": lines})
        self.assertEqual(len(refs), 2)
        self.assertEqual(refs[0]["number"], "1")
        self.assertIn("Calculated using 2024 UK GDP", refs[0]["text"])
        self.assertEqual(refs[1]["number"], "2")

    def test_raw_footer_file_parser_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with open(tmp_path / "references_raw.md", "w") as f:
                f.write("1 First reference text\n2 Second reference text\n")
            parser = RawFooterFileParser()
            refs = parser.parse({"path": tmp_path})
            self.assertEqual(len(refs), 2)
            self.assertEqual(refs[0]["number"], "1")
            self.assertEqual(refs[0]["text"], "First reference text")
            self.assertEqual(refs[1]["number"], "2")
            self.assertEqual(refs[1]["text"], "Second reference text")

    def test_raw_footer_file_parser_json_strings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with open(tmp_path / "ref_raw.txt", "w") as f:
                f.write('["1 First ref", "2 Second ref"]')
            parser = RawFooterFileParser()
            refs = parser.parse({"path": tmp_path})
            self.assertEqual(len(refs), 2)
            self.assertEqual(refs[0]["number"], "1")
            self.assertEqual(refs[0]["text"], "First ref")

    def test_reference_definition_extractor_deduplication_and_sorting(self):
        extractor = ReferenceDefinitionExtractor()
        context = {
            "lines": [
                "¹ Short def\n",
                "## References\n",
                "1 A much more detailed description of reference 1.\n",
                "10 Tenth reference.\n",
                "2 Second reference.\n",
            ]
        }
        refs = extractor.extract_references(context)
        numbers = [r["number"] for r in refs]
        self.assertEqual(numbers, ["1", "2", "10"])
        # Should prefer detailed description
        ref1 = next(r for r in refs if r["number"] == "1")
        self.assertIn("much more detailed", ref1["text"])


class TestEndToEndPreprocessing(unittest.TestCase):
    def test_preprocess_endnotes_document(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_dir = Path(tmpdir)
            raw_content = (
                "# Document Title\n\n"
                "This is the first sentence with a reference.¹\n\n"
                "This is the second sentence without any reference.\n\n"
                "## Endnotes\n\n"
                "1 Source: Government Report 2026.\n"
            )
            with open(doc_dir / "raw.md", "w") as f:
                f.write(raw_content)

            preprocess(doc_dir)

            self.assertTrue((doc_dir / "processed.md").exists())
            self.assertTrue((doc_dir / "references.json").exists())
            self.assertTrue((doc_dir / "input.txt").exists())

            processed_text = (doc_dir / "processed.md").read_text()
            self.assertIn("S0001 This is the first sentence with a reference. ** (Reference: 1)", processed_text)
            self.assertIn("S0002 This is the second sentence without any reference.", processed_text)
            # Endnote heading and definition should not become sentences
            self.assertNotIn("S0003 Source:", processed_text)

            input_text = (doc_dir / "input.txt").read_text()
            self.assertIn("S0001", input_text)
            self.assertIn("S0002", input_text)

    def test_preprocess_superscript_definitions_document(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_dir = Path(tmpdir)
            raw_content = (
                "# Economic Report\n\n"
                "Growth was observed across regions.¹ Increased investment contributed to GDP.²\n\n"
                "¹ OECD (2024), Economic Survey.\n\n"
                "² Treasury (2025), Autumn Budget.\n"
            )
            with open(doc_dir / "raw.md", "w") as f:
                f.write(raw_content)

            preprocess(doc_dir)

            processed_text = (doc_dir / "processed.md").read_text()
            self.assertIn("Reference: 1,2", processed_text)
            # The definition lines should not become S00x sentences
            self.assertNotIn("OECD (2024)", processed_text.split("**")[0])


class TestProcessStructured(unittest.TestCase):
    """Exercises MarkdownDocumentProcessor.process_structured() directly with
    hand-built blocks, matching how a non-markdown source (e.g. a PDF's text
    layer) feeds the processing pipeline without going through markdown_it."""

    def test_writes_headings_paragraphs_and_references_from_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            blocks = [
                {"type": "heading", "text": "Section One"},
                {"type": "paragraph", "text": "This is a statement with evidence.1 A second sentence follows."},
                {"type": "table", "text": "col a | col b"},
            ]
            references = [{"number": "1", "text": "Source: Government Report 2026."}]

            processor = MarkdownDocumentProcessor()
            result_refs = processor.process_structured(blocks, references, out_dir)

            self.assertEqual(result_refs, references)
            processed_text = (out_dir / "processed.md").read_text()
            self.assertIn("Section One", processed_text)
            self.assertIn(
                "S0001 This is a statement with evidence. A second sentence follows. ** (Reference: 1)",
                processed_text,
            )
            self.assertIn("col a | col b", processed_text)

            references_json = json.loads((out_dir / "references.json").read_text())
            self.assertEqual(references_json, references)

    def test_process_delegates_to_process_structured_for_markdown(self):
        """process() must still produce byte-identical output after being
        refactored into a thin adapter over process_structured()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_dir = Path(tmpdir)
            raw_content = (
                "# Document Title\n\n"
                "This is the first sentence with a reference.¹\n\n"
                "## Endnotes\n\n"
                "1 Source: Government Report 2026.\n"
            )
            with open(doc_dir / "raw.md", "w") as f:
                f.write(raw_content)

            preprocess(doc_dir)
            processed_text = (doc_dir / "processed.md").read_text()

            self.assertIn("Document Title", processed_text)
            self.assertIn("S0001 This is the first sentence with a reference. ** (Reference: 1)", processed_text)


class TestConstructXml(unittest.TestCase):
    """summary.xml carries each theme's sentence count and, once weights.json
    exists, its weight, which the scorer aggregates by."""

    def _build(self, tmpdir: str, weights: dict | None = None, themes: list | None = None,
               core: dict | None = None) -> ET.Element:
        doc_dir = Path(tmpdir) / "doc"
        result_dir = Path(tmpdir) / "results"
        (result_dir / "diagnosis").mkdir(parents=True)
        doc_dir.mkdir()
        (doc_dir / "input.txt").write_text(
            "S0001 Rents are high. ** (Reference: 1)\n"
            "S0002 Fees are opaque.\n"
            "S0003 Research has found tenants pay more.\n"
            "S0004 The consultation closed in June.\n"
            "S0005 A sentence from another aspect.\n"
        )
        (doc_dir / "references.json").write_text(json.dumps([{"number": "1", "text": "Housing Survey"}]))
        (result_dir / "diagnosis" / "text.txt").write_text(
            "S0001 Rents are high. ** (Reference: 1)\n"
            "S0002 Fees are opaque.\n"
            "S0003 Research has found tenants pay more.\n"
            "S0004 The consultation closed in June.\n"
        )
        (result_dir / "diagnosis" / "classification.json").write_text(json.dumps({"themes": themes or [
            {"name": "Costs to tenants", "ids": ["S0001", "S0003"]},
            {"name": "Background", "ids": ["S0004"]},
        ]}))
        if weights is not None:
            (result_dir / "diagnosis" / "weights.json").write_text(json.dumps(weights))
        if core is not None:
            (result_dir / "core_themes.json").write_text(json.dumps(core))
        construct_xml(doc_dir, result_dir, "diagnosis")
        return ET.parse(result_dir / "diagnosis" / "summary.xml").getroot()

    def test_sentence_count_matches_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._build(tmpdir)
            blocks = root.findall("theme") + root.findall("ungrouped")
            self.assertEqual(len(blocks), 3)
            for block in blocks:
                self.assertEqual(int(block.get("sentences")), len(block.findall("text/line")))
            self.assertEqual(root.find("ungrouped").get("sentences"), "1")

    def test_ignores_ids_outside_the_aspect(self):
        # S0005 is in input.txt (the whole document) but not in this aspect.
        themes = [
            {"name": "Costs to tenants", "ids": ["S0001", "S0002", "S0003", "S0005"]},
            {"name": "Other", "ids": ["S0004", "S0005"]},
            {"name": "Elsewhere", "ids": ["S0005"]},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._build(tmpdir, themes=themes)
            names = [theme.findtext("name") for theme in root.findall("theme")]
            self.assertEqual(names, ["Costs to tenants", "Other"])
            self.assertNotIn("S0005", [line.get("id") for line in root.iter("line")])

    def test_core_theme_carries_its_claim(self):
        core = {"diagnosis": {"core_themes": [
            {"name": "Costs to tenants", "claim": "Tenants pay high, opaque fees."},
        ], "note": ""}}
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._build(tmpdir, core=core)
            claims = {theme.findtext("name"): theme.findtext("claim") for theme in root.findall("theme")}
            self.assertEqual(claims, {"Costs to tenants": "Tenants pay high, opaque fees.", "Background": None})

    def test_no_weight_without_weights_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._build(tmpdir)
            self.assertTrue(all(theme.get("weight") is None for theme in root.findall("theme")))

    def test_weight_matched_by_theme_name(self):
        weights = {"themes": [
            {"theme": "Background", "reason": "Procedure.", "weight": "not core"},
            {"theme": "Costs to tenants", "reason": "The problem itself.", "weight": "core"},
        ]}
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._build(tmpdir, weights)
            by_name = {theme.findtext("name"): theme.get("weight") for theme in root.findall("theme")}
            self.assertEqual(by_name, {"Costs to tenants": "core", "Background": "not core"})
            self.assertIsNone(root.find("ungrouped").get("weight"))


if __name__ == "__main__":
    unittest.main()
