import json
import tempfile
import unittest
from src.data.human_review import save_review, validate_review


class HumanReviewTests(unittest.TestCase):
    def test_records_explicit_zero_spans_and_keeps_revisions(self):
        with tempfile.TemporaryDirectory() as directory:
            a = save_review(directory, "1", "something", [], "PRODUCT_SEARCH", "Reviewer", "ambiguous")
            b = save_review(directory, "1", "something", [], "PRODUCT_SEARCH", "Reviewer", "confirmed")
            self.assertNotEqual(a, b)
            self.assertEqual(json.loads(a.read_text())["chunks"], [])

    def test_overlap_and_fractional_offsets_rejected(self):
        for chunks in ([{"start_char": 0.5, "end_char": 3, "chunk_label": "PRODUCT"}],
                       [{"start_char": 0, "end_char": 3, "chunk_label": "PRODUCT"}] * 2):
            with self.assertRaises(ValueError):
                validate_review("bag", chunks, "PRODUCT_SEARCH", "Reviewer")
