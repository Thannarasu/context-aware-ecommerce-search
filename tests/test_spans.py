import unittest

from src.evaluation.spans import assign_predicted_spans, calculate_span_metrics


class SpanTests(unittest.TestCase):
    def test_repeated_text_uses_distinct_occurrences(self):
        rows = assign_predicted_spans("bag bag", [{"text": "bag", "label": "PRODUCT"}] * 3)
        self.assertEqual([r["start_char"] for r in rows], [0, 4, None])

    def test_invalid_explicit_offsets_are_not_repaired(self):
        row = assign_predicted_spans("bag", [{"text": "bag", "label": "PRODUCT",
                                             "start_char": 1, "end_char": 3}])[0]
        self.assertFalse(row["span_found"])

    def test_failed_query_retains_false_negatives_and_invalid_counts(self):
        gold = [{"query_id": "1", "chunk_label": "PRODUCT", "start_char": 0, "end_char": 3},
                {"query_id": "2", "chunk_label": "PRODUCT", "start_char": 0, "end_char": 3}]
        pred = [gold[0], dict(gold[0], start_char=None)]
        result = calculate_span_metrics(gold, pred)
        self.assertEqual((result["true_positive"], result["false_positive"],
                          result["false_negative"]), (1, 1, 1))
        self.assertEqual(result["f1"], 0.5)

    def test_fractional_gold_offsets_rejected(self):
        with self.assertRaises(ValueError):
            calculate_span_metrics([{"query_id": "1", "chunk_label": "PRODUCT",
                                     "start_char": 0.5, "end_char": 3}], [])


if __name__ == "__main__":
    unittest.main()
