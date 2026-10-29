import json
import shutil
import tempfile
import unittest
from pathlib import Path

from web_app.combine import (
    CombineError,
    collect_sources,
    combinable_documents,
    combine_documents,
    source_title,
)


def write_document(root: Path, alias: str, title, processed: str, references, links=None) -> Path:
    """Create a minimal ingested document (the artefacts combining reads)."""
    document_dir = root / alias
    document_dir.mkdir(parents=True)
    (document_dir / "processed.md").write_text(processed, encoding="utf-8")
    (document_dir / "references.json").write_text(json.dumps(references), encoding="utf-8")
    if links is not None:
        (document_dir / "links.json").write_text(json.dumps(links), encoding="utf-8")
    (document_dir / "input.txt").write_text(
        "".join(line + "\n" for line in processed.splitlines() if line.startswith("S")),
        encoding="utf-8",
    )
    if title is not None:
        (document_dir / "metadata.json").write_text(
            json.dumps({"title": title, "alias": alias}), encoding="utf-8"
        )
    return document_dir


class CombineTestCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.documents = self.root / "documents"
        self.documents.mkdir()
        self.out_dir = self.root / "out"

    def consultation(self, **overrides):
        return write_document(
            self.documents,
            "consultation",
            "Visitor Levy Consultation",
            "# Introduction\n"
            "S0001 The levy raises revenue. ** (Reference: 1)\n"
            "S0002 Evidence supports it. ** (Reference: 1,2)\n",
            [{"number": "1", "text": "OECD 2013"}, {"number": "2", "text": "ONS 2024"}],
            **overrides,
        )

    def impact_assessment(self, **overrides):
        return write_document(
            self.documents,
            "impact_assessment",
            "Visitor Levy Impact Assessment",
            "S0001 Costs fall on businesses. ** (Reference: 1)\n"
            "S0002 Take-up is uncertain. ** (Reference: 2)\n",
            [{"number": "1", "text": "HMRC costing"}, {"number": "2", "text": "DCMS survey"}],
            **overrides,
        )

    def combine(self, aliases=("consultation", "impact_assessment")):
        sources = collect_sources(self.documents, list(aliases))
        return combine_documents(sources, self.out_dir)


class TestSentenceMerging(CombineTestCase):
    def test_sentences_are_renumbered_continuously_across_sources(self):
        self.consultation()
        self.impact_assessment()
        stats = self.combine()

        input_lines = (self.out_dir / "input.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            [line.split(" ", 1)[0] for line in input_lines], ["S0001", "S0002", "S0003", "S0004"]
        )
        self.assertIn("Costs fall on businesses.", input_lines[2])
        self.assertEqual(stats["sentence_count"], 4)

    def test_sentence_ids_stay_four_digits_past_999(self):
        write_document(
            self.documents, "long_a", "Long A",
            "".join(f"S{i:04d} Sentence {i}.\n" for i in range(1, 999)), [],
        )
        write_document(
            self.documents, "long_b", "Long B",
            "".join(f"S{i:04d} Sentence {i}.\n" for i in range(1, 6)), [],
        )
        self.combine(["long_a", "long_b"])

        input_lines = (self.out_dir / "input.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(input_lines), 1003)
        self.assertTrue(input_lines[998].startswith("S0999 "))
        self.assertTrue(input_lines[999].startswith("S1000 "))

    def test_sentence_ids_widen_past_9999(self):
        write_document(
            self.documents, "huge", "Huge Doc",
            "".join(f"S{i:04d} Sentence {i}.\n" for i in range(1, 9999)), [],
        )
        write_document(
            self.documents, "small", "Small Doc",
            "".join(f"S{i:04d} Sentence {i}.\n" for i in range(1, 4)), [],
        )
        self.combine(["huge", "small"])

        input_lines = (self.out_dir / "input.txt").read_text(encoding="utf-8").splitlines()
        self.assertTrue(input_lines[9998].startswith("S9999 "))
        self.assertTrue(input_lines[9999].startswith("S10000 "))

    def test_source_written_before_the_four_digit_move_is_renumbered(self):
        write_document(
            self.documents, "legacy", "Legacy Doc",
            "S001 A legacy claim. ** (Reference: 1)\n"
            "S002 Another legacy claim.\n",
            [{"number": "1", "text": "Legacy source"}],
        )
        self.impact_assessment()
        stats = self.combine(["legacy", "impact_assessment"])

        input_lines = (self.out_dir / "input.txt").read_text(encoding="utf-8").splitlines()
        self.assertTrue(input_lines[0].startswith("S0001 A legacy claim."))
        self.assertTrue(input_lines[1].startswith("S0002 Another legacy claim."))
        self.assertEqual(stats["sentence_count"], 4)

    def test_each_source_is_introduced_by_a_heading(self):
        self.consultation()
        self.impact_assessment()
        self.combine()

        processed = (self.out_dir / "processed.md").read_text(encoding="utf-8")
        self.assertIn("# Visitor Levy Consultation", processed)
        self.assertIn("# Visitor Levy Impact Assessment", processed)
        # Non-sentence lines from the source are carried through.
        self.assertIn("# Introduction", processed)

    def test_input_txt_holds_only_sentence_lines(self):
        self.consultation()
        self.impact_assessment()
        self.combine()

        input_lines = (self.out_dir / "input.txt").read_text(encoding="utf-8").splitlines()
        self.assertTrue(all(line.startswith("S") for line in input_lines))


class TestReferenceMerging(CombineTestCase):
    def test_references_are_renumbered_into_one_list(self):
        self.consultation()
        self.impact_assessment()
        stats = self.combine()

        references = json.loads((self.out_dir / "references.json").read_text(encoding="utf-8"))
        self.assertEqual([entry["number"] for entry in references], ["1", "2", "3", "4"])
        self.assertEqual(
            [entry["text"] for entry in references],
            ["OECD 2013", "ONS 2024", "HMRC costing", "DCMS survey"],
        )
        self.assertEqual(stats["reference_count"], 4)

    def test_citation_markers_follow_the_renumbering(self):
        self.consultation()
        self.impact_assessment()
        self.combine()

        input_lines = (self.out_dir / "input.txt").read_text(encoding="utf-8").splitlines()
        self.assertTrue(input_lines[1].endswith("** (Reference: 1, 2)"))
        self.assertTrue(input_lines[2].endswith("** (Reference: 3)"))
        self.assertTrue(input_lines[3].endswith("** (Reference: 4)"))

    def test_each_reference_records_where_it_came_from(self):
        self.consultation()
        self.impact_assessment()
        self.combine()

        references = json.loads((self.out_dir / "references.json").read_text(encoding="utf-8"))
        self.assertEqual(references[2]["source_alias"], "impact_assessment")
        self.assertEqual(references[2]["source_number"], "1")

    def test_reference_fields_such_as_urls_are_preserved(self):
        write_document(
            self.documents, "web_doc", "Web Doc",
            "S0001 A claim. ** (Reference: 1)\n",
            [{"number": "1", "text": "ONS", "urls": ["https://ons.gov.uk"]}],
        )
        self.impact_assessment()
        self.combine(["web_doc", "impact_assessment"])

        references = json.loads((self.out_dir / "references.json").read_text(encoding="utf-8"))
        self.assertEqual(references[0]["urls"], ["https://ons.gov.uk"])

    def test_citation_without_a_definition_gets_its_own_placeholder(self):
        write_document(
            self.documents, "gappy", "Gappy Doc",
            "S0001 An uncited claim. ** (Reference: 7)\n", [],
        )
        self.impact_assessment()
        stats = self.combine(["gappy", "impact_assessment"])

        references = json.loads((self.out_dir / "references.json").read_text(encoding="utf-8"))
        input_lines = (self.out_dir / "input.txt").read_text(encoding="utf-8").splitlines()

        # The marker must not be left pointing at another source's reference #7.
        self.assertTrue(input_lines[0].endswith("** (Reference: 1)"))
        self.assertTrue(references[0]["missing"])
        self.assertEqual(references[0]["source_alias"], "gappy")
        self.assertEqual(stats["missing_reference_count"], 1)
        self.assertEqual([entry["text"] for entry in references[1:]], ["HMRC costing", "DCMS survey"])


class TestLinkMerging(CombineTestCase):
    def test_links_are_renumbered_and_markers_rewritten(self):
        write_document(
            self.documents, "linked_a", "Linked A",
            "S0001 See the guidance**(Link: 1) .\n", [],
            links=[{"number": "1", "url": "https://a.gov.uk", "text": "guidance"}],
        )
        write_document(
            self.documents, "linked_b", "Linked B",
            "S0001 See the review**(Link: 1) and the report**(Link: 2) .\n", [],
            links=[
                {"number": "1", "url": "https://b.gov.uk/review", "text": "review"},
                {"number": "2", "url": "https://b.gov.uk/report", "text": "report"},
            ],
        )
        stats = self.combine(["linked_a", "linked_b"])

        input_lines = (self.out_dir / "input.txt").read_text(encoding="utf-8").splitlines()
        self.assertIn("**(Link: 1)", input_lines[0])
        self.assertIn("**(Link: 2)", input_lines[1])
        self.assertIn("**(Link: 3)", input_lines[1])
        self.assertEqual(stats["link_count"], 3)

    def test_missing_links_json_is_tolerated(self):
        self.consultation()
        self.impact_assessment()
        stats = self.combine()
        self.assertEqual(stats["link_count"], 0)
        self.assertEqual(json.loads((self.out_dir / "links.json").read_text(encoding="utf-8")), [])


class TestSourceStats(CombineTestCase):
    def test_per_source_ranges_are_reported(self):
        self.consultation()
        self.impact_assessment()
        stats = self.combine()

        consultation, impact = stats["sources"]
        self.assertEqual(
            (consultation["first_sentence"], consultation["last_sentence"]), ("S0001", "S0002")
        )
        self.assertEqual((impact["first_sentence"], impact["last_sentence"]), ("S0003", "S0004"))
        self.assertEqual((impact["first_reference"], impact["last_reference"]), ("3", "4"))

    def test_source_without_references_reports_no_range(self):
        write_document(self.documents, "bare", "Bare Doc", "S0001 A claim.\n", [])
        self.impact_assessment()
        stats = self.combine(["bare", "impact_assessment"])

        self.assertEqual(stats["sources"][0]["reference_count"], 0)
        self.assertIsNone(stats["sources"][0]["first_reference"])


class TestSourceResolution(CombineTestCase):
    def test_unknown_alias_is_rejected(self):
        self.consultation()
        with self.assertRaises(CombineError):
            collect_sources(self.documents, ["consultation", "nope"])

    def test_a_single_source_is_rejected(self):
        self.consultation()
        with self.assertRaises(CombineError):
            self.combine(["consultation"])

    def test_source_without_processed_md_is_rejected(self):
        self.consultation()
        (self.documents / "empty").mkdir()
        with self.assertRaises(CombineError):
            self.combine(["consultation", "empty"])

    def test_title_falls_back_to_the_alias(self):
        write_document(self.documents, "no_title_doc", None, "S0001 A claim.\n", [])
        self.assertEqual(source_title(self.documents / "no_title_doc", "no_title_doc"), "No Title Doc")

    def test_combinable_documents_lists_ingested_documents_only(self):
        self.consultation()
        self.impact_assessment()
        (self.documents / "not_ingested").mkdir()

        listed = combinable_documents(self.documents)
        self.assertEqual([doc["alias"] for doc in listed], ["consultation", "impact_assessment"])
        self.assertEqual(listed[0]["sentence_count"], 2)
        self.assertEqual(listed[0]["reference_count"], 2)


if __name__ == "__main__":
    unittest.main()
