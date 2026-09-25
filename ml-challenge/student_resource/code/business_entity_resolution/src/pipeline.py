#!/usr/bin/env python3
"""Memory-conscious entity-resolution baseline for the Amazon ML Challenge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import sqlite3
import tempfile
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize


LEGAL_SUFFIXES = {
    "co", "company", "corp", "corporation", "inc", "incorporated", "llc",
    "llp", "ltd", "limited", "pvt", "private", "sa", "sas", "sarl",
}
NUMBER_RE = re.compile(r"\d+")
SPACE_RE = re.compile(r"\s+")


def clean(value: object) -> str:
    value = unicodedata.normalize("NFKC", "" if pd.isna(value) else str(value)).casefold()
    value = value.replace("&", " and ")
    return SPACE_RE.sub(" ", "".join(
        c if c.isalnum() or unicodedata.category(c).startswith("M") else " " for c in value
    )).strip()


def without_suffixes(value: str) -> str:
    tokens = value.split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna("")
    df["name_norm"] = df["business_name"].map(clean)
    df["address_norm"] = df["business_address"].map(clean)
    df["name_core"] = df["name_norm"].map(without_suffixes)
    # Repeating the name stops long addresses from overwhelming retrieval.
    df["search_text"] = df["name_norm"] + " " + df["name_norm"] + " " + df["address_norm"]
    return df[["entity_id", "country", "name_norm", "address_norm", "name_core", "search_text"]]


def read_tsv(path: Path, limit: int | None = None) -> pd.DataFrame:
    return prepare(pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, nrows=limit,
        usecols=["entity_id", "business_name", "business_address", "country"],
    ))


def make_vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), n_features=2**17,
        alternate_sign=False, norm="l2", dtype=np.float32,
    )


def fit_encoder(paths: list[Path], rows_per_file: int, dimensions: int) -> tuple[HashingVectorizer, TruncatedSVD]:
    vectorizer = make_vectorizer()
    texts: list[str] = []
    for path in paths:
        texts.extend(read_tsv(path, rows_per_file)["search_text"].tolist())
    matrix = vectorizer.transform(texts)
    svd = TruncatedSVD(n_components=dimensions, n_iter=3, random_state=42)
    svd.fit(matrix)
    return vectorizer, svd


def encode(texts: list[str] | pd.Series, vectorizer: HashingVectorizer, svd: TruncatedSVD) -> np.ndarray:
    vectors = svd.transform(vectorizer.transform(texts)).astype("float32", copy=False)
    return l2_normalize(vectors, copy=False)


def encode_e5(frame: pd.DataFrame, tokenizer, model, batch_size: int, max_length: int) -> np.ndarray:
    import torch

    vectors = []
    report_every = max(batch_size, math.ceil(len(frame) / 10))
    next_report = report_every
    for start in range(0, len(frame), batch_size):
        part = frame.iloc[start:start + batch_size]
        texts = [
            f"query: business name: {name}; address: {address}"
            for name, address in zip(part["name_norm"], part["address_norm"])
        ]
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=1)
        vectors.append(pooled.cpu().numpy())
        done = start + len(part)
        if len(frame) >= 20_000 and (done >= next_report or done == len(frame)):
            print(f"E5 training-sample encoding {done:,}/{len(frame):,} ({done / len(frame):.0%})", flush=True)
            next_report = done + report_every
    return np.ascontiguousarray(np.concatenate(vectors), dtype=np.float32)


def load_e5(model_name: str, batch_size: int, max_length: int):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Hybrid retrieval requires a CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda").eval()
    return lambda frame: encode_e5(frame, tokenizer, model, batch_size, max_length), model.config.hidden_size


def build_country_index(target: pd.DataFrame, positions: np.ndarray, dimensions: int, encode_rows, batch_size: int, label: str = "") -> faiss.Index:
    started = time.monotonic()
    nlist = min(2048, max(32, int(math.sqrt(len(positions)))))
    if len(positions) < 20_000:
        index: faiss.Index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimensions))
    else:
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimensions), dimensions, nlist, faiss.METRIC_INNER_PRODUCT)
        sample = positions[np.linspace(0, len(positions) - 1, min(len(positions), nlist * 40), dtype=int)]
        if label:
            print(f"{label} training FAISS on {len(sample):,} rows", flush=True)
        index.train(encode_rows(target.iloc[sample]))
        index.nprobe = min(32, nlist)
    report_every = max(batch_size, math.ceil(len(positions) / 10))
    next_report = report_every
    for start in range(0, len(positions), batch_size):
        ids = positions[start:start + batch_size]
        index.add_with_ids(encode_rows(target.iloc[ids]), ids)
        done = start + len(ids)
        if label and (done >= next_report or done == len(positions)):
            print(f"{label} indexed {done:,}/{len(positions):,} ({done / len(positions):.0%}) elapsed={time.monotonic() - started:.0f}s", flush=True)
            next_report = done + report_every
    return index


def build_indexes(
    target: pd.DataFrame,
    vectorizer: HashingVectorizer,
    svd: TruncatedSVD,
    batch_size: int,
    label: str = "",
) -> dict[str, faiss.Index]:
    indexes: dict[str, faiss.Index] = {}
    dimensions = svd.n_components
    for country, positions in target.groupby("country", sort=False).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        indexes[str(country)] = build_country_index(
            target, positions, dimensions,
            lambda rows: encode(rows["search_text"], vectorizer, svd), batch_size,
            f"{label} CPU country={country}" if label else "",
        )
    return indexes


def number_jaccard(left: str, right: str) -> float:
    a, b = set(NUMBER_RE.findall(left)), set(NUMBER_RE.findall(right))
    return len(a & b) / len(a | b) if a or b else 0.0


def address_numbers(value: str) -> tuple[str, str]:
    numbers = NUMBER_RE.findall(value)
    house = numbers[0] if numbers else ""
    postal = next((number for number in reversed(numbers) if len(number) in (5, 6)), "")
    return house, postal


def pair_features(left: pd.Series, right: pd.Series, retrieval_score: float, rank: int) -> list[float]:
    ln, rn = left["name_norm"], right["name_norm"]
    la, ra = left["address_norm"], right["address_norm"]
    lc, rc = left["name_core"], right["name_core"]
    left_house, left_postal = address_numbers(la)
    right_house, right_postal = address_numbers(ra)
    return [
        retrieval_score,
        1.0 / (rank + 1),
        ratio(ln, rn) / 100,
        token_set_ratio(ln, rn) / 100,
        ratio(lc, rc) / 100,
        token_set_ratio(la, ra) / 100 if la and ra else 0.0,
        ratio(la, ra) / 100 if la and ra else 0.0,
        number_jaccard(la, ra),
        float(bool(left_house) and left_house == right_house),
        float(bool(left_postal) and left_postal == right_postal),
        float(bool(ln) and ln == rn),
        float(bool(lc) and lc == rc),
        float(bool(la) and la == ra),
        float(not la),
        float(not ra),
    ]


def candidates_for_source(
    source1: pd.DataFrame,
    target: pd.DataFrame,
    indexes: dict[str, faiss.Index],
    vectorizer: HashingVectorizer,
    svd: TruncatedSVD,
    top_k: int,
    batch_size: int,
    gpu_encoder=None,
    gpu_dimensions: int = 0,
    progress_label: str = "",
):
    block_columns = ("name_norm", "name_core", "address_norm")
    # ponytail: skip keys shared by >100 targets; add address ranking if these blocks limit recall.
    exact_blocks = []
    if progress_label:
        print(f"{progress_label} building exact blocks", flush=True)
    for column in block_columns:
        wanted = set(zip(source1["country"], source1[column]))
        matches = defaultdict(list)
        for right_id, (country, value) in enumerate(zip(target["country"], target[column])):
            key = (country, value)
            if value and key in wanted and len(matches[key]) <= 100:
                matches[key].append(right_id)
        exact_blocks.append(matches)
    if progress_label:
        print(f"{progress_label} exact blocks ready", flush=True)
    target_countries = target.groupby("country", sort=False).indices if gpu_encoder is not None else {}
    for country, positions in source1.groupby("country", sort=False).indices.items():
        index = indexes.get(str(country))
        if index is None:
            continue
        gpu_index = None
        if gpu_encoder is not None:
            right_positions = target_countries.get(country)
            if right_positions is not None:
                gpu_index = build_country_index(
                    target, np.asarray(right_positions, dtype=np.int64), gpu_dimensions,
                    gpu_encoder, batch_size,
                    f"{progress_label} E5 country={country}" if progress_label else "",
                )
        positions = np.asarray(positions, dtype=np.int64)
        report_every = max(1, math.ceil(len(positions) / 10))
        next_report = report_every
        for start in range(0, len(positions), batch_size):
            left_ids = positions[start:start + batch_size]
            scores, right_ids = index.search(
                encode(source1.iloc[left_ids]["search_text"], vectorizer, svd), top_k
            )
            if gpu_index is not None:
                gpu_scores, gpu_right_ids = gpu_index.search(gpu_encoder(source1.iloc[left_ids]), top_k)
            for row_number, left_id in enumerate(left_ids):
                left = source1.iloc[left_id]
                candidates = {
                    int(right_id): (float(score), rank, 0.0, top_k)
                    for rank, (right_id, score) in enumerate(zip(right_ids[row_number], scores[row_number]))
                    if right_id >= 0
                }
                if gpu_index is not None:
                    for rank, (right_id, score) in enumerate(zip(gpu_right_ids[row_number], gpu_scores[row_number])):
                        if right_id >= 0:
                            cpu_score, cpu_rank, _, _ = candidates.get(int(right_id), (0.0, top_k, 0.0, top_k))
                            candidates[int(right_id)] = (cpu_score, cpu_rank, float(score), rank)
                for column, matches in zip(block_columns, exact_blocks):
                    block = matches.get((country, left[column]), ())
                    for right_id in block if len(block) <= 100 else ():
                        candidates.setdefault(right_id, (0.0, top_k, 0.0, top_k))
                for right_id, (score, rank, gpu_score, gpu_rank) in candidates.items():
                    right = target.iloc[right_id]
                    features = pair_features(left, right, score, rank)
                    if gpu_encoder is not None:
                        features.extend((gpu_score, 1.0 / (gpu_rank + 1) if gpu_rank < top_k else 0.0))
                    yield int(left_id), str(right["entity_id"]), features
                done = start + row_number + 1
                if progress_label and (done >= next_report or done == len(positions)):
                    print(f"{progress_label} country={country} scored {done:,}/{len(positions):,} S1 rows ({done / len(positions):.0%})", flush=True)
                    next_report = done + report_every
        del gpu_index


def load_truth(path: Path, allowed: set[str]) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["source1_entity_id"] in allowed:
                truth[row["source1_entity_id"]] = set(filter(None, row["matched_entity_ids"].split(",")))
    return truth


def split_group(entity_id: str) -> int:
    return hashlib.blake2b(entity_id.encode(), digest_size=1).digest()[0] % 10


def f05(truth: set[str], predicted: set[str]) -> float:
    if not truth:
        return float(not predicted)
    if not predicted:
        return 0.0
    correct = len(truth & predicted)
    precision, recall = correct / len(predicted), correct / len(truth)
    return 1.25 * precision * recall / (0.25 * precision + recall) if correct else 0.0


def train(args: argparse.Namespace) -> None:
    train_dir = args.data_dir / "train"
    source_paths = [train_dir / f"train_source{i}.tsv" for i in (1, 2, 3)]
    # The unlabeled test sample exposes unseen countries to the retrieval encoder.
    print("stage=fit_cpu_encoder started", flush=True)
    vectorizer, svd = fit_encoder(
        source_paths + [args.data_dir / "test" / "test_source1.tsv"],
        args.encoder_rows, args.dimensions,
    )
    print("stage=fit_cpu_encoder done", flush=True)
    print("stage=read_source1 started", flush=True)
    source1 = read_tsv(source_paths[0], args.max_s1)
    truth = load_truth(train_dir / "train_ground_truth.tsv", set(source1["entity_id"]))
    print(f"stage=read_source1 done rows={len(source1):,} true_links={sum(map(len, truth.values())):,}", flush=True)
    if args.retriever == "hybrid":
        print(f"stage=load_e5 model={args.e5_model}", flush=True)
    gpu_encoder, gpu_dimensions = load_e5(args.e5_model, args.gpu_batch_size, args.max_length) if args.retriever == "hybrid" else (None, 0)
    if gpu_encoder is not None:
        print("stage=load_e5 done", flush=True)

    train_x: list[list[float]] = []
    train_y: list[int] = []
    held_out: dict[str, list[tuple[str, list[float]]]] = {}
    retrieved: dict[str, set[str]] = {entity_id: set() for entity_id in source1["entity_id"]}

    for path in source_paths[1:]:
        print(f"source={path.stem} stage=read_target started", flush=True)
        target = read_tsv(path, args.max_target)
        print(f"source={path.stem} stage=read_target done rows={len(target):,}", flush=True)
        indexes = build_indexes(target, vectorizer, svd, args.batch_size, path.stem)
        for left_pos, candidate_id, features in candidates_for_source(
            source1, target, indexes, vectorizer, svd, args.top_k, args.batch_size,
            gpu_encoder, gpu_dimensions, path.stem,
        ):
            source1_id = str(source1.iloc[left_pos]["entity_id"])
            retrieved[source1_id].add(candidate_id)
            label = int(candidate_id in truth.get(source1_id, set()))
            if split_group(source1_id) < 7:
                train_x.append(features)
                train_y.append(label)
            else:
                held_out.setdefault(source1_id, []).append((candidate_id, features))
        print(f"source={path.stem} stage=candidates done", flush=True)
        del indexes, target

    if not train_y or len(set(train_y)) < 2:
        raise RuntimeError("Training candidates contain only one class; raise --top-k or data limits")
    print(f"stage=fit_classifier started training_pairs={len(train_y):,} positives={sum(train_y):,}", flush=True)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=150, max_leaf_nodes=15,
        min_samples_leaf=30, l2_regularization=2.0, early_stopping=False, random_state=42,
    ).fit(
        np.asarray(train_x, dtype=np.float32), np.asarray(train_y, dtype=np.uint8),
        sample_weight=np.where(np.asarray(train_y) == 1, 5.0, 1.0),
    )
    print("stage=fit_classifier done; calibrating threshold", flush=True)

    scored: dict[str, list[tuple[str, float]]] = {}
    for source1_id, rows in held_out.items():
        probabilities = model.predict_proba(np.asarray([x[1] for x in rows], dtype=np.float32))[:, 1]
        scored[source1_id] = [(row[0], float(score)) for row, score in zip(rows, probabilities)]
    thresholds = np.linspace(0.05, 0.999, 96)
    results = []
    calibration_ids = [entity_id for entity_id in source1["entity_id"] if split_group(entity_id) == 7]
    evaluation_ids = [entity_id for entity_id in source1["entity_id"] if split_group(entity_id) >= 8]
    for threshold in thresholds:
        results.append(float(np.mean([
            f05(truth.get(entity_id, set()), {
                candidate for candidate, score in scored.get(entity_id, ()) if score >= threshold
            }) for entity_id in calibration_ids
        ])))
    best = int(np.argmax(results))
    threshold = float(thresholds[best])
    evaluation_score = float(np.mean([
        f05(truth.get(entity_id, set()), {
            candidate for candidate, score in scored.get(entity_id, ()) if score >= threshold
        }) for entity_id in evaluation_ids
    ]))
    positives = sum(len(v) for v in truth.values())
    hits = sum(len(truth.get(k, set()) & v) for k, v in retrieved.items())

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "vectorizer": vectorizer, "svd": svd, "model": model,
        "threshold": threshold, "top_k": args.top_k,
        "retriever": args.retriever, "e5_model": args.e5_model,
        "max_length": args.max_length,
    }, args.model)
    print(f"trained_pairs={len(train_y):,} positive_pairs={sum(train_y):,}")
    print(f"candidate_recall={hits / positives:.4f} ({hits:,}/{positives:,})")
    print(f"calibration_macro_f0.5={results[best]:.4f} threshold={threshold:.2f}")
    print(f"held_out_macro_f0.5={evaluation_score:.4f} entities={len(evaluation_ids):,}")
    print(f"saved={args.model}")


def write_source_predictions(
    path: Path,
    source1: pd.DataFrame,
    target_path: Path,
    artifact: dict,
    args: argparse.Namespace,
    gpu_encoder=None,
    gpu_dimensions: int = 0,
) -> None:
    target = read_tsv(target_path, args.max_target)
    indexes = build_indexes(target, artifact["vectorizer"], artifact["svd"], args.batch_size)
    database = sqlite3.connect(path.with_suffix(".db"))
    database.execute("PRAGMA journal_mode=OFF")
    database.execute("CREATE TABLE predictions (pos INTEGER PRIMARY KEY, candidates TEXT, matches TEXT)")
    pending: list[tuple[int, str, list[float]]] = []

    def score_pending() -> None:
        if not pending:
            return
        probabilities = artifact["model"].predict_proba(
            np.asarray([item[2] for item in pending], dtype=np.float32)
        )[:, 1]
        grouped = defaultdict(lambda: [[], []])
        for (left_pos, candidate_id, _), probability in zip(pending, probabilities):
            grouped[left_pos][0].append(candidate_id)
            if probability >= artifact["threshold"]:
                grouped[left_pos][1].append(candidate_id)
        database.executemany(
            "INSERT INTO predictions VALUES (?, ?, ?)",
            ((pos, ",".join(values[0]), ",".join(values[1])) for pos, values in grouped.items()),
        )
        database.commit()
        pending.clear()

    last_pos = None
    for left_pos, candidate_id, features in candidates_for_source(
        source1, target, indexes, artifact["vectorizer"], artifact["svd"],
        artifact["top_k"], args.batch_size, gpu_encoder, gpu_dimensions,
    ):
        if pending and left_pos != last_pos and len(pending) >= args.batch_size:
            score_pending()
        pending.append((left_pos, candidate_id, features))
        last_pos = left_pos
    score_pending()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        stored = iter(database.execute("SELECT pos, candidates, matches FROM predictions ORDER BY pos"))
        row = next(stored, None)
        for pos in range(len(source1)):
            writer.writerow(row[1:] if row and row[0] == pos else ("", ""))
            if row and row[0] == pos:
                row = next(stored, None)
    database.close()


def predict(args: argparse.Namespace) -> None:
    artifact = joblib.load(args.model)
    gpu_encoder, gpu_dimensions = load_e5(
        artifact["e5_model"], args.gpu_batch_size, artifact.get("max_length", args.max_length)
    ) if artifact.get("retriever") == "hybrid" else (None, 0)
    test_dir = args.data_dir / "test"
    source1 = read_tsv(test_dir / "test_source1.tsv", args.max_s1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="er-predictions-") as tmp:
        partials = []
        for source in (2, 3):
            partial = Path(tmp) / f"source{source}.tsv"
            write_source_predictions(
                partial, source1, test_dir / f"test_source{source}.tsv", artifact, args,
                gpu_encoder, gpu_dimensions,
            )
            partials.append(partial)
        with (
            partials[0].open(encoding="utf-8", newline="") as left,
            partials[1].open(encoding="utf-8", newline="") as right,
            (args.output_dir / "candidate_pairs.tsv").open("w", encoding="utf-8", newline="") as candidate_file,
            (args.output_dir / "matching_results.tsv").open("w", encoding="utf-8", newline="") as match_file,
        ):
            readers = [csv.reader(left, delimiter="\t"), csv.reader(right, delimiter="\t")]
            candidate_writer, match_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n"), csv.writer(match_file, delimiter="\t", lineterminator="\n")
            candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
            match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
            for entity_id, a, b in zip(source1["entity_id"], *readers, strict=True):
                candidate_writer.writerow([entity_id, ",".join(filter(None, (a[0], b[0])))])
                match_writer.writerow([entity_id, ",".join(filter(None, (a[1], b[1])))])
    print(f"wrote={args.output_dir / 'candidate_pairs.tsv'}")
    print(f"wrote={args.output_dir / 'matching_results.tsv'}")


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=root / "dataset")
    common.add_argument("--model", type=Path, default=root / "artifacts" / "baseline.joblib")
    common.add_argument("--max-s1", type=int)
    common.add_argument("--max-target", type=int)
    common.add_argument("--batch-size", type=int, default=4096)
    common.add_argument("--gpu-batch-size", type=int, default=256)
    common.add_argument("--max-length", type=int, default=128)
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train", parents=[common])
    training.add_argument("--encoder-rows", type=int, default=50_000)
    training.add_argument("--dimensions", type=int, default=64)
    training.add_argument("--top-k", type=int, default=10)
    training.add_argument("--retriever", choices=("cpu", "hybrid"), default="cpu")
    training.add_argument("--e5-model", default="intfloat/multilingual-e5-small")
    training.set_defaults(run=train)
    inference = sub.add_parser("predict", parents=[common])
    inference.add_argument("--output-dir", type=Path, default=root / "output")
    inference.set_defaults(run=predict)
    return command


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.run(arguments)
