import re
import unittest
from string import Template

from web_app.analyse import ASPECTS, _get_prompt, _weighting_input, section_score


SUMMARY = """<root>
<theme sentences="2" weight="core">
<name>Costs to tenants</name>
<text>
<line>Rents are high. ** (Reference: 1)</line>
<line>Fees rose.**(Link: 4) **(Link: 5)</line>
</text>
<references>
<reference number="1">Housing Survey</reference>
</references>
<links>
<link number="4" url="https://example.org">Example</link>
</links>
</theme>
<ungrouped sentences="1">
<text>
<line>The consultation closed in June.</line>
</text>
<references>
</references>
<links>
</links>
</ungrouped>
</root>
"""


class TestWeightingInput(unittest.TestCase):
    """The weighting call must see the argument but none of the evidence."""

    def setUp(self):
        self.reduced = _weighting_input(SUMMARY)

    def test_keeps_theme_names_sentences_and_counts(self):
        self.assertIn("<name>Costs to tenants</name>", self.reduced)
        self.assertIn("<line>Rents are high.</line>", self.reduced)
        self.assertIn("<line>Fees rose.</line>", self.reduced)
        self.assertIn('<theme sentences="2">', self.reduced)

    def test_drops_citations_ungrouped_and_old_weights(self):
        for absent in ("Reference", "Link", "<references>", "<links>", "Housing Survey",
                       "example.org", "<ungrouped", "consultation", "weight="):
            self.assertNotIn(absent, self.reduced)


class TestPromptPlaceholders(unittest.TestCase):
    """Template.substitute raises on a missing or stray placeholder, so every
    prompt is filled here exactly as analyse.py fills it."""

    def test_scoring_prompts_substitute(self):
        for aspect in ASPECTS:
            template = Template(_get_prompt(f"score_{aspect}") + "\n" + _get_prompt("score_common"))
            prompt = template.substitute(intervention="X", problem="Y",
                                         should_explain=_get_prompt(f"explain_{aspect}"))
            self.assertNotIn("$", prompt)

    def test_weighting_prompt_substitutes(self):
        for aspect in ASPECTS:
            prompt = Template(_get_prompt("weight_themes")).substitute(
                intervention="X", problem="Y", aspect="Z",
                should_explain=_get_prompt(f"explain_{aspect}"))
            self.assertNotIn("$", prompt)



class TestSectionScore(unittest.TestCase):
    def score(self, core, other=()):
        weights = {f"c{i}": "core" for i in range(len(core))}
        weights.update({f"n{i}": "not core" for i in range(len(other))})
        scores = [{"theme": f"c{i}", "score": s} for i, s in enumerate(core)]
        scores += [{"theme": f"n{i}", "score": s} for i, s in enumerate(other)]
        return section_score(scores, weights)

    def test_not_core_themes_do_not_enter_the_mean(self):
        result = self.score([3, 0, 1, 0], other=[3, 3])
        self.assertEqual((result["core_mean"], result["score"]), (1.0, 1))

    def test_rounds_to_nearest_with_half_down(self):
        self.assertEqual(self.score([1, 2])["score"], 1)        # 1.5
        self.assertEqual(self.score([2, 3])["score"], 2)        # 2.5
        self.assertEqual(self.score([1, 2, 2])["score"], 2)     # 1.67
        self.assertEqual(self.score([1, 1, 2])["score"], 1)     # 1.33
        self.assertEqual(self.score([0, 1, 1, 1])["score"], 1)  # 0.75
        self.assertEqual(self.score([0, 0, 0, 1])["score"], 0)  # 0.25

    def test_no_core_theme_uses_all_themes(self):
        result = self.score([], other=[1, 2, 2, 2])
        self.assertEqual((result["core_mean"], result["score"]), (1.75, 2))

    def test_reports_core_theme_without_a_level(self):
        result = section_score([{"theme": "A", "score": 2}], {"A": "core", "B": "core"})
        self.assertEqual(result["missing_core"], ["B"])
        self.assertEqual(result["score"], 2)


if __name__ == "__main__":
    unittest.main()


class TestBlockScoring(unittest.TestCase):
    """Testing is scored as one block: no themes, weights or claims in its
    prompt or input, and an aspect with no sentences scores 0 without a call."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        from web_app import analyse
        self.analyse = analyse
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.docs, self.results = root / "documents", root / "results"
        (self.docs).mkdir()
        (self.results / "testing").mkdir(parents=True)
        (self.docs / "references.json").write_text('[{"number": "1", "text": "Survey"}]')
        (self.docs / "input.txt").write_text(
            "S0001 We will review in 2027. **(Reference: 1)\nS0002 Other text.\n")
        self.saved = (analyse.DOCUMENTS_DIR, analyse.RESULTS_DIR)
        analyse.DOCUMENTS_DIR, analyse.RESULTS_DIR = self.docs, self.results

    def tearDown(self):
        self.analyse.DOCUMENTS_DIR, self.analyse.RESULTS_DIR = self.saved
        self.tmp.cleanup()

    def test_prompt_substitutes_and_mentions_no_themes(self):
        prompt = Template(_get_prompt("score_testing") + "\n" + _get_prompt("score_block")).substitute(
            intervention="X", problem="Y", should_explain=_get_prompt("explain_testing"))
        words = set(re.findall(r"[a-z]+", prompt.lower()))
        for absent in ("theme", "themes", "core", "weight", "weights", "ungrouped", "claim"):
            self.assertNotIn(absent, words)

    def test_block_input_has_sentences_and_citations_but_no_theme_structure(self):
        (self.results / "testing" / "text.txt").write_text("S0001 We will review in 2027. **(Reference: 1)\n")
        block = self.analyse._block_input("testing")
        self.assertIn("<line>We will review in 2027.", block)
        self.assertIn("Survey", block)
        for absent in ("<theme", "ungrouped", "<root>", "id="):
            self.assertNotIn(absent, block)
        self.assertFalse((self.results / "testing" / "summary.xml").exists())

    def test_empty_aspect_scores_zero_without_a_call(self):
        import json
        (self.results / "testing" / "text.txt").write_text("")
        self.assertIsNone(self.analyse.call_llm_score_block("testing"))
        result = json.loads((self.results / "testing" / "scoring.json").read_text())
        self.assertEqual(result["score"], 0)
        self.assertIn("testing and evaluation", result["reasoning"])
