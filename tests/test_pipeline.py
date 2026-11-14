import unittest
from src.pipeline import analyze


class PipelineTests(unittest.TestCase):
    def test_rule_offsets_reproduce_query(self):
        query = "airpods pro case without keychain"
        result = analyze(query)
        self.assertEqual([r["label"] for r in result["chunks"]], ["PRODUCT", "CONSTRAINT"])
        for row in result["chunks"]:
            self.assertEqual(query[row["start_char"]:row["end_char"]], row["text"])

    def test_empty_and_oversized_queries_rejected(self):
        for query in ("  ", "x" * 2001):
            with self.assertRaises(ValueError):
                analyze(query)
