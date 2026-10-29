import unittest

from web_app.sentence_ids import (
    format_sentence_id,
    format_sentence_line,
    is_sentence_line,
    sentence_id,
    split_sentence_line,
)


class TestFormatting(unittest.TestCase):
    def test_ids_are_padded_to_four_digits(self):
        self.assertEqual(format_sentence_id(1), "S0001")
        self.assertEqual(format_sentence_id(999), "S0999")
        self.assertEqual(format_sentence_id(1000), "S1000")

    def test_ids_widen_past_9999(self):
        self.assertEqual(format_sentence_id(10000), "S10000")

    def test_line_is_id_then_body(self):
        self.assertEqual(format_sentence_line(42, "A claim."), "S0042 A claim.")


class TestParsing(unittest.TestCase):
    def test_splits_id_from_body(self):
        self.assertEqual(split_sentence_line("S0042 A claim.\n"), ("S0042", "A claim."))

    def test_reads_ids_written_before_the_four_digit_move(self):
        self.assertEqual(split_sentence_line("S042 A claim."), ("S042", "A claim."))

    def test_reads_ids_wider_than_four_digits(self):
        self.assertEqual(sentence_id("S12345 A claim."), "S12345")

    def test_non_sentence_lines_have_no_id(self):
        self.assertFalse(is_sentence_line("# A heading"))
        self.assertFalse(is_sentence_line("| a | table |"))
        self.assertIsNone(sentence_id("Some prose about S0042."))
        self.assertEqual(split_sentence_line("# A heading"), (None, "# A heading"))

    def test_too_few_digits_is_not_an_id(self):
        self.assertFalse(is_sentence_line("S42 not an id"))


if __name__ == "__main__":
    unittest.main()
