# Business Entity Resolution baseline

CPU-first retrieve-and-classify baseline designed for 16 GB RAM. It uses Unicode-aware normalization (including Indic combining marks), character n-gram retrieval, exact name/core/address blocks, country-partitioned FAISS indexes, fuzzy name/address features, hard negatives, and a boosted binary classifier. The retrieval encoder sees a sample of unlabeled test Source 1 text so it can represent France; no test labels are used. Prediction batches are assembled on disk to avoid keeping all scored pairs in memory.

## Environment

From the repository root:

```bash
uv sync --frozen
```

## Smoke run

```bash
uv run python code/business_entity_resolution/src/pipeline.py train \
  --max-s1 1000 --max-target 5000 --encoder-rows 2000 \
  --model artifacts/smoke.joblib

uv run python code/business_entity_resolution/src/pipeline.py predict \
  --max-s1 1000 --max-target 5000 \
  --model artifacts/smoke.joblib --output-dir output-smoke
```

Limits deliberately produce incomplete, non-submittable output. They only verify the pipeline.

## Initial full run

Train the first model on a bounded S1 sample while indexing all S2/S3 rows:

```bash
uv run python code/business_entity_resolution/src/pipeline.py train \
  --max-s1 100000 --model artifacts/baseline.joblib
```

Generate predictions for every test S1 row:

```bash
uv run python code/business_entity_resolution/src/pipeline.py predict \
  --model artifacts/baseline.joblib --output-dir output

python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

The model is CPU-only. More training rows, larger `--top-k`, multilingual embeddings, and GPU inference should be added only after measuring candidate recall and validation F0.5.

## CPU validation

```bash
uv run python code/business_entity_resolution/src/validate_cpu.py
```

This older bounded experiment uses a different retrieval and feature pipeline. Its score
is not a production estimate. To check production candidate recall and held-out F0.5
against all training targets while keeping Source 1 small:

```bash
uv run python code/business_entity_resolution/src/pipeline.py train \
  --max-s1 500 --encoder-rows 10000 --top-k 20 \
  --model artifacts/full-target-check.joblib
```

The training command reports candidate recall, calibration F0.5, and a separate
held-out F0.5. Run it without `--max-s1` for the eventual full model.

## Kaggle GPU candidate probe

The optional `src/kaggle_gpu_probe.py` measures retrieval recall from the
[MIT-licensed multilingual-e5-small model](https://huggingface.co/intfloat/multilingual-e5-small).
It does not change the production classifier or create a submission. On a Kaggle
GPU notebook with this repository and challenge data attached:

```bash
pip install -r code/business_entity_resolution/requirements.txt 'transformers==4.57.6'
python code/business_entity_resolution/src/kaggle_gpu_probe.py \
  --data-dir /kaggle/input/YOUR-DATASET/dataset --max-s1 2000 --top-k 50
```

Kaggle's GPU environment provides PyTorch. The model can be downloaded when
Internet is enabled, or attached to the notebook and passed via `--model PATH`.
For a short smoke run, add `--max-target 100000`; judge that run by
`candidate_recall_available` and `target_coverage`, not full recall. The full
run omits `--max-target` and processes both complete training target files.
Run `python code/business_entity_resolution/src/kaggle_gpu_probe.py --self-test`
to check its local FAISS ID mapping without a GPU.
