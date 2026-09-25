from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.pipeline import candidates_for_source, clean, f05, number_jaccard, prepare, without_suffixes


def test_small_logic():
    assert clean("PAYNE-Énterprises, LLC") == "payne énterprises llc"
    assert clean("राम मार्केटिंग प्राइवेट लिमिटेड") == "राम मार्केटिंग प्राइवेट लिमिटेड"
    assert clean("ஆதித்யா கடை") == "ஆதித்யா கடை"
    assert without_suffixes("payne enterprises llc") == "payne enterprises"
    assert without_suffixes("the company store") == "the company store"
    assert number_jaccard("3315 Fremont St", "3315 Fremont Street") == 1.0
    assert f05(set(), set()) == 1.0
    assert f05({"a", "b"}, {"a"}) > f05({"a", "b"}, {"a", "x"})


def test_exact_block_reaches_candidate_outside_vector_top_k():
    left = prepare(pd.DataFrame([{
        "entity_id": "S1-1", "business_name": "Acme Ltd", "business_address": "1 Main St", "country": "US",
    }]))
    right = prepare(pd.DataFrame([
        {"entity_id": "S2-1", "business_name": "Other", "business_address": "2 Main St", "country": "US"},
        {"entity_id": "S2-2", "business_name": "Acme Inc", "business_address": "9 Side St", "country": "US"},
    ]))

    class TopOne:
        def search(self, vectors, top_k):
            return np.array([[0.9]], dtype="float32"), np.array([[0]], dtype="int64")

    with patch("src.pipeline.encode", return_value=np.zeros((1, 1), dtype="float32")):
        pairs = list(candidates_for_source(left, right, {"US": TopOne()}, None, None, 1, 1))
    assert {candidate for _, candidate, _ in pairs} == {"S2-1", "S2-2"}

    crowded = prepare(pd.DataFrame([{
        "entity_id": f"S2-{i + 2}", "business_name": "Acme Inc",
        "business_address": "9 Side St", "country": "US",
    } for i in range(101)]))
    with patch("src.pipeline.encode", return_value=np.zeros((1, 1), dtype="float32")):
        pairs = list(candidates_for_source(left, crowded, {"US": TopOne()}, None, None, 1, 1))
    assert len(pairs) == 1


def test_hybrid_unions_gpu_and_cpu_candidates():
    left = prepare(pd.DataFrame([{
        "entity_id": "S1-1", "business_name": "Left", "business_address": "1 A", "country": "US",
    }]))
    right = prepare(pd.DataFrame([
        {"entity_id": "S2-1", "business_name": "First", "business_address": "2 B", "country": "US"},
        {"entity_id": "S2-2", "business_name": "Second", "business_address": "3 C", "country": "US"},
    ]))

    class Search:
        def __init__(self, position):
            self.position = position

        def search(self, vectors, top_k):
            return np.array([[0.9]], dtype="float32"), np.array([[self.position]], dtype="int64")

    def fake_gpu(frame):
        return np.array([
            [0, 1] if entity_id in {"S1-1", "S2-2"} else [1, 0]
            for entity_id in frame["entity_id"]
        ], dtype="float32")

    output = StringIO()
    with patch("src.pipeline.encode", return_value=np.zeros((1, 1), dtype="float32")), redirect_stdout(output):
        pairs = list(candidates_for_source(
            left, right, {"US": Search(0)}, None, None, 1, 1,
            fake_gpu, 2, "train_source2",
        ))
    assert {candidate for _, candidate, _ in pairs} == {"S2-1", "S2-2"}
    assert all(len(features) == 17 for _, _, features in pairs)
    assert "E5 country=US indexed 2/2 (100%)" in output.getvalue()
    assert "country=US scored 1/1 S1 rows (100%)" in output.getvalue()
