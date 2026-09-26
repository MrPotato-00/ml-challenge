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

## Hybrid CPU + GPU validation

The full-target GPU probe retrieved 5,758/6,986 true links (82.42%) on the
first 2,000 training S1 rows, versus 4,161/6,986 (59.56%) for the CPU
pipeline. This is candidate recall, **not** final matching F0.5. The next
experiment unions CPU, E5, and exact-block candidates and retrains the pair
classifier on that union:

```bash
pip install -r code/business_entity_resolution/requirements.txt 'transformers==4.57.6'
python code/business_entity_resolution/src/pipeline.py train \
  --data-dir /kaggle/input/YOUR-DATASET/dataset \
  --max-s1 2000 --top-k 50 --retriever hybrid \
  --model artifacts/hybrid.joblib
```

Do not set `--max-target` for this comparison. Keep a Kaggle GPU enabled and
Internet on for the E5 weights, or pass `--e5-model PATH` for an attached copy.
The command reports union candidate recall, mean/p95/max candidates per S1,
and independent held-out macro F0.5; compare recall and F0.5 with 59.56% and
0.7141 from the CPU run. This is a sampled
validation run, not a submission. The hybrid model records its retriever and
model path in the saved artifact so prediction uses the same candidate method.
Training prints flushed stage messages plus per-country indexing and S1 scoring
percentages. These messages require starting a new run with the updated code;
they cannot appear in a training process that was already running.
Prediction prints the same index/scoring progress for each test target source,
then output-writing progress. An already-running prediction cannot gain these logs.
