#!/usr/bin/env python3
"""Bounded CPU validation with an S1-group train/calibration/test split."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer

from pipeline import NUMBER_RE, clean, f05, number_jaccard, without_suffixes

ROOT = Path(__file__).resolve().parents[3]
S1_ROWS, TARGET_ROWS, TOP_K, DIMENSIONS = 20_000, 100_000, 20, 512


def prepare(path: Path, rows: int) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=rows)
    frame["name"] = frame.business_name.map(clean)
    frame["name_core"] = frame.name.map(without_suffixes)
    frame["address"] = frame.business_address.map(clean)
    frame["numbers"] = frame.address.map(NUMBER_RE.findall)
    frame["house"] = frame.numbers.map(lambda x: x[0] if x else "")
    frame["postal"] = frame.numbers.map(lambda x: next((n for n in reversed(x) if len(n) in (5, 6)), ""))
    frame["text"] = frame.name + " " + frame.name + " " + frame.address
    return frame


def split(entity_id: str) -> int:
    return hashlib.blake2b(entity_id.encode(), digest_size=1).digest()[0] % 10


def metrics(entity_ids, truth, predictions):
    tp = predicted = actual = 0
    entity_scores = []
    for entity_id in entity_ids:
        expected, found = truth.get(entity_id, set()), predictions.get(entity_id, set())
        tp += len(expected & found)
        predicted += len(found)
        actual += len(expected)
        entity_scores.append(f05(expected, found))
    precision = tp / predicted if predicted else 1.0
    recall = tp / actual if actual else 1.0
    return precision, recall, float(np.mean(entity_scores))


def candidate_rows(source1, target, truth, vectorizer):
    query_vectors = vectorizer.transform(source1.text).toarray().astype("float32", copy=False)
    rows = []
    retrieved = defaultdict(set)
    for country in source1.country.unique():
        left_ids = np.flatnonzero(source1.country.to_numpy() == country)
        right_ids = np.flatnonzero(target.country.to_numpy() == country)
        right = target.iloc[right_ids]
        right_vectors = vectorizer.transform(right.text).toarray().astype("float32", copy=False)
        index = faiss.IndexFlatIP(DIMENSIONS)
        index.add(right_vectors)
        scores, neighbors = index.search(query_vectors[left_ids], TOP_K)

        exact_name, exact_core, exact_address = defaultdict(list), defaultdict(list), defaultdict(list)
        token_counts = Counter(token for value in right.name for token in set(value.split()) if len(token) >= 4)
        rare_tokens = defaultdict(list)
        for local, row in enumerate(right.itertuples(index=False)):
            if row.name: exact_name[row.name].append(local)
            if row.name_core: exact_core[row.name_core].append(local)
            if row.address: exact_address[row.address].append(local)
            for token in set(row.name.split()):
                if len(token) >= 4 and token_counts[token] <= 25:
                    rare_tokens[token].append(local)

        for q, left_id in enumerate(left_ids):
            left = source1.iloc[left_id]
            candidates = {}
            for rank, (neighbor, score) in enumerate(zip(neighbors[q], scores[q])):
                candidates[int(neighbor)] = [float(score), rank, 1, 0, 0, 0, 0]
            blocks = (
                (exact_name.get(left["name"], ()), 3),
                (exact_core.get(left.name_core, ()), 4),
                (exact_address.get(left.address, ()), 5),
            )
            for positions, flag in blocks:
                for local in positions:
                    candidates.setdefault(local, [0.0, TOP_K, 0, 0, 0, 0, 0])[flag] = 1
            tokens = sorted(
                {t for t in left["name"].split() if len(t) >= 4 and t in rare_tokens},
                key=lambda t: token_counts[t],
            )[:3]
            for token in tokens:
                for local in rare_tokens[token]:
                    candidates.setdefault(local, [0.0, TOP_K, 0, 0, 0, 0, 0])[6] = 1

            for local, (score, rank, char_block, name_block, core_block, address_block, rare_block) in candidates.items():
                match = right.iloc[local]
                retrieved[left.entity_id].add(match.entity_id)
                la, ra = left.address, match.address
                rows.append((
                    left.entity_id, match.entity_id,
                    int(match.entity_id in truth.get(left.entity_id, set())),
                    score, 1 / (rank + 1),
                    ratio(left["name"], match["name"]) / 100,
                    token_set_ratio(left["name"], match["name"]) / 100,
                    ratio(left.name_core, match.name_core) / 100,
                    ratio(la, ra) / 100 if la and ra else 0,
                    token_set_ratio(la, ra) / 100 if la and ra else 0,
                    number_jaccard(la, ra),
                    int(bool(left.house) and left.house == match.house),
                    int(bool(left.postal) and left.postal == match.postal),
                    int(bool(left["name"]) and left["name"] == match["name"]),
                    int(bool(left.name_core) and left.name_core == match.name_core),
                    int(bool(la) and la == ra),
                    int(not ra), char_block, name_block, core_block, address_block, rare_block,
                ))
    return rows, retrieved


def main() -> None:
    train_dir = ROOT / "dataset" / "train"
    source1 = prepare(train_dir / "train_source1.tsv", S1_ROWS)
    targets = [prepare(train_dir / f"train_source{i}.tsv", TARGET_ROWS) for i in (2, 3)]
    allowed = set(source1.entity_id)
    truth = {}
    for chunk in pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", dtype=str,
                             keep_default_na=False, chunksize=100_000):
        for row in chunk[chunk.source1_entity_id.isin(allowed)].itertuples(index=False):
            truth[row.source1_entity_id] = set(filter(None, row.matched_entity_ids.split(",")))

    vectorizer = HashingVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), n_features=DIMENSIONS,
        alternate_sign=False, norm="l2", dtype=np.float32,
    )
    all_rows, retrieved = [], defaultdict(set)
    for target in targets:
        rows, found = candidate_rows(source1, target, truth, vectorizer)
        all_rows.extend(rows)
        for entity_id, values in found.items(): retrieved[entity_id].update(values)
    columns = [
        "s1", "candidate", "label", "retrieval", "rank", "name", "name_tokens",
        "name_core", "address", "address_tokens", "numbers", "house", "postal",
        "exact_name", "exact_core", "exact_address", "missing_address",
        "char_block", "name_block", "core_block", "address_block", "rare_block",
    ]
    pairs = pd.DataFrame(all_rows, columns=columns)
    features = columns[3:]
    groups = pairs.s1.map(split)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=150, max_leaf_nodes=15,
        min_samples_leaf=30, l2_regularization=2, random_state=42,
    ).fit(
        pairs.loc[groups < 7, features], pairs.loc[groups < 7, "label"],
        sample_weight=np.where(pairs.loc[groups < 7, "label"].to_numpy() == 1, 5.0, 1.0),
    )
    pairs["probability"] = model.predict_proba(pairs[features])[:, 1]

    gates = [
        pd.Series(True, index=pairs.index),
        (
            (pairs.exact_name.eq(1) & ((pairs.address >= .35) | pairs.missing_address.eq(1) | pairs.house.eq(1)))
            | (pairs.exact_core.eq(1) & (pairs.address >= .45))
            | (pairs.exact_address.eq(1) & (pairs.name >= .45))
            | ((pairs.name >= .92) & (pairs.address >= .60))
            | ((pairs.name >= .82) & (pairs.address >= .78))
            | ((pairs.name >= .70) & (pairs.address >= .90))
            | (pairs.house.eq(1) & (pairs.name >= .78) & (pairs.address >= .55))
            | (pairs.postal.eq(1) & (pairs.name >= .72))
        ),
        (
            (pairs.exact_name.eq(1) & ((pairs.address >= .50) | pairs.missing_address.eq(1) | pairs.house.eq(1)))
            | (pairs.exact_core.eq(1) & (pairs.address >= .60))
            | (pairs.exact_address.eq(1) & (pairs.name >= .60))
            | ((pairs.name >= .95) & (pairs.address >= .70))
            | ((pairs.name >= .88) & (pairs.address >= .82))
            | ((pairs.name >= .78) & (pairs.address >= .92))
            | (pairs.house.eq(1) & (pairs.name >= .85) & (pairs.address >= .65))
            | (pairs.postal.eq(1) & (pairs.name >= .80))
        ),
    ]

    def predictions(ids, threshold, gate):
        selected = pairs[pairs.s1.isin(ids) & gate & (pairs.probability >= threshold)]
        return selected.groupby("s1").candidate.agg(set).to_dict()

    calibration_ids = source1.loc[source1.entity_id.map(split) == 7, "entity_id"].tolist()
    _, threshold, gate_id = max(
        (metrics(calibration_ids, truth, predictions(calibration_ids, t, gate))[2], t, gate_id)
        for gate_id, gate in enumerate(gates)
        for t in np.linspace(0.50, 0.999, 101)
    )
    test_ids = source1.loc[source1.entity_id.map(split) >= 8, "entity_id"].tolist()
    precision, recall, score = metrics(test_ids, truth, predictions(test_ids, threshold, gates[gate_id]))
    available = set().union(*(set(frame.entity_id) for frame in targets))
    all_positive = sum(len(truth.get(x, set())) for x in test_ids)
    available_positive = sum(len(truth.get(x, set()) & available) for x in test_ids)
    retrieved_positive = sum(len(truth.get(x, set()) & retrieved.get(x, set())) for x in test_ids)
    print(f"pairs={len(pairs):,} positives={int(pairs.label.sum()):,}")
    print(f"test_entities={len(test_ids):,} threshold={threshold:.3f} evidence_gate={gate_id}")
    print(f"precision={precision:.4f} recall={recall:.4f} macro_f0.5={score:.4f}")
    print(f"retrieved_truth={retrieved_positive:,}/{all_positive:,} full_recall_ceiling={retrieved_positive/all_positive:.4f}")
    print(f"sampled_target_truth={available_positive:,}/{all_positive:,} sampled_target_coverage={available_positive/all_positive:.4f}")


if __name__ == "__main__":
    main()
