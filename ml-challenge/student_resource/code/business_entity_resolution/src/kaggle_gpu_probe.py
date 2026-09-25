#!/usr/bin/env python3
"""Measure multilingual embedding candidate recall on a Kaggle GPU."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from pipeline import load_truth, read_tsv


def encode(frame, tokenizer, model, batch_size: int, max_length: int) -> np.ndarray:
    vectors = []
    for start in range(0, len(frame), batch_size):
        part = frame.iloc[start:start + batch_size]
        texts = [
            f"query: business name: {name}; address: {address}"
            for name, address in zip(part["name_norm"], part["address_norm"])
        ]
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=1)
        vectors.append(pooled.cpu().numpy())
    return np.ascontiguousarray(np.concatenate(vectors), dtype=np.float32)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "dataset")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--max-s1", type=int, default=2000)
    parser.add_argument("--max-target", type=int)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(2))
        index.add_with_ids(
            np.array([[1, 0], [0, 1]], dtype=np.float32),
            np.array([4, 9], dtype=np.int64),
        )
        assert index.search(np.array([[0, 1]], dtype=np.float32), 1)[1].tolist() == [[9]]
        print("PASS")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this probe")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to("cuda").eval()
    dimension = model.config.hidden_size

    train_dir = args.data_dir / "train"
    source1 = read_tsv(train_dir / "train_source1.tsv", args.max_s1)
    truth = load_truth(train_dir / "train_ground_truth.tsv", set(source1["entity_id"]))
    positive_ids = {entity_id for ids in truth.values() for entity_id in ids}
    total = sum(map(len, truth.values()))
    if not total:
        raise RuntimeError("No true links in the selected Source 1 rows")
    available: set[str] = set()
    retrieved = {entity_id: set() for entity_id in source1["entity_id"]}

    for source in (2, 3):
        target = read_tsv(train_dir / f"train_source{source}.tsv", args.max_target)[
            ["entity_id", "country", "name_norm", "address_norm"]
        ]
        available.update(entity_id for entity_id in target["entity_id"] if entity_id in positive_ids)
        for country, positions in target.groupby("country", sort=False).indices.items():
            left_positions = source1.groupby("country", sort=False).indices.get(country)
            if left_positions is None:
                continue
            positions = np.asarray(positions, dtype=np.int64)
            nlist = min(2048, max(32, int(math.sqrt(len(positions)))))
            if len(positions) < 20_000:
                index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
            else:
                index = faiss.IndexIVFFlat(
                    faiss.IndexFlatIP(dimension), dimension, nlist, faiss.METRIC_INNER_PRODUCT
                )
                sample = positions[np.linspace(
                    0, len(positions) - 1, min(len(positions), nlist * 40), dtype=int
                )]
                index.train(encode(target.iloc[sample], tokenizer, model, args.batch_size, args.max_length))
                index.nprobe = min(args.nprobe, nlist)
            for start in range(0, len(positions), args.batch_size):
                ids = positions[start:start + args.batch_size]
                index.add_with_ids(
                    encode(target.iloc[ids], tokenizer, model, args.batch_size, args.max_length), ids
                )
            left_positions = np.asarray(left_positions, dtype=np.int64)
            for start in range(0, len(left_positions), args.batch_size):
                ids = left_positions[start:start + args.batch_size]
                queries = encode(source1.iloc[ids], tokenizer, model, args.batch_size, args.max_length)
                _, neighbours = index.search(queries, args.top_k)
                for left_pos, right_positions in zip(ids, neighbours):
                    retrieved[source1.iloc[left_pos]["entity_id"]].update(
                        target.iloc[right_pos]["entity_id"] for right_pos in right_positions if right_pos >= 0
                    )
            print(f"source={source} country={country} indexed={len(positions):,}", flush=True)
            del index
        del target

    hits = sum(len(truth.get(entity_id, set()) & ids) for entity_id, ids in retrieved.items())
    print(f"target_coverage={len(available):,}/{total:,} ({len(available) / total:.4f})")
    print(f"candidate_recall_full={hits:,}/{total:,} ({hits / total:.4f})")
    if available:
        print(f"candidate_recall_available={hits:,}/{len(available):,} ({hits / len(available):.4f})")


if __name__ == "__main__":
    main()
