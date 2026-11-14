import unittest
from src.block1_chunking.bio import encode_labels, decode_labels
from src.block1_chunking.train_transformer import make_splits, validate_splits


class BioTests(unittest.TestCase):
    def test_roundtrip_subwords_and_special_tokens(self):
        offsets = [(0, 0), (0, 3), (3, 7), (8, 11), (0, 0)]
        spans = [{"chunk_label": "PRODUCT", "start_char": 0, "end_char": 7}]
        decoded = decode_labels("handbag red", offsets, encode_labels(offsets, spans))
        self.assertEqual(decoded[0]["chunk_text"], "handbag")
        self.assertEqual(len(decoded), 1)

    def test_unrepresentable_boundary_rejected(self):
        with self.assertRaises(ValueError):
            encode_labels([(0, 7)], [{"chunk_label": "PRODUCT", "start_char": 0, "end_char": 3}])

    def test_duplicate_query_text_stays_in_one_split(self):
        queries = {str(i): f"query {i}" for i in range(20)}
        queries["20"] = " QUERY  1 "
        spans = {q: [] for q in queries}
        splits = make_splits(queries, spans, 42)
        validate_splits(splits, queries, spans)
        self.assertTrue(any("1" in ids and "20" in ids for ids in splits.values()))
        self.assertEqual(splits, make_splits(queries, spans, 42))
