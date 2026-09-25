# Amazon ML Challenge 2026: Entity-Resolution Pipeline Status

## Objective

For every Source 1 business, predict all matching Source 2 and Source 3 records. The final decision is binary at the candidate-pair level:

```text
(S1 record, S2/S3 record) -> match or non-match
```

Accepted pairs are grouped into the required `matching_results.tsv`. Entity IDs are opaque record keys; among 345,997 sampled positive pairs, zero shared the same numeric suffix.

## Pipeline

```text
Raw TSV records
    -> Unicode/text normalization
    -> candidate generation
    -> pairwise feature calculation
    -> gradient-boosted match classifier
    -> F0.5-tuned threshold
    -> candidate_pairs.tsv and matching_results.tsv
```

## Text preprocessing

Original TSV files remain unchanged. The production pipeline keeps only IDs, country,
and normalized fields in memory to fit larger target partitions on a CPU machine.

### Business names

- Unicode NFKC normalization
- Unicode-aware `casefold()`
- Punctuation and repeated-whitespace normalization while preserving Unicode combining
  marks (required for Hindi and Tamil vowel signs)
- `&` normalized to `and`
- Auxiliary core name with trailing legal suffixes removed
- Unicode preserved for Hindi, Tamil, French accents, and mixed scripts

Legal suffixes currently include `inc`, `corporation`, `llc`, `llp`, `ltd`, `limited`, `pvt`, `private`, `sa`, `sas`, and `sarl`.

Example:

```text
PAYNE-Énterprises, LLC
name_norm -> payne énterprises llc
name_core -> payne énterprises
```

### Addresses

- Unicode and punctuation normalization
- Numeric-component extraction
- First number used as a potential house number
- Last five- or six-digit component used as a potential postal code
- Missing-address indicator

These rules are generic and do not depend on external geocoding or a fixed set of countries.

### Retrieval text

```text
normalized_name + normalized_name + normalized_address
```

The repeated name prevents a long address from overwhelming the name signal.

## Candidate generation

The system does not perform an all-pairs comparison.

### Production path

- Character n-grams of lengths 3-5
- Fixed-memory `HashingVectorizer`
- Compact SVD vectors
- SVD fitted on training text plus unlabeled test Source 1 text so unseen
  France is represented in the retrieval space
- Cosine similarity through normalized dot products
- Country-partitioned FAISS indexes
- Top-k retrieval from Source 2 and Source 3 independently
- IVF indexes for large target partitions
- Union with exact normalized name, trailing-suffix-stripped name, and address blocks
- Exact block keys matching more than 100 targets are skipped to bound candidate volume

Character retrieval handles corruption such as `Enterprises`/`Enterpires`, `Williams`/`Wilblims`, and `Wayne`/`Wanye`.

### Additional blocking evaluated

Rare name-token blocking recovered only two extra positive pairs in the earlier bounded experiment. It remains experimental because its full-scale in-memory index would be expensive for little measured gain.

## Pairwise features

### Retrieval

- Character-vector similarity
- Reciprocal retrieval rank
- Blocking-channel indicators in the validation experiment

### Name

- Edit/character similarity
- Token-set similarity
- Core-name similarity
- Exact normalized-name agreement
- Exact core-name agreement

### Address

- Edit/character similarity
- Token-set similarity
- Number-set Jaccard similarity
- House-number agreement
- Postal-code agreement
- Exact normalized-address agreement
- Missing-address indicators

## Training

Ground truth supplies positive labels:

```text
candidate listed for S1  -> 1
retrieved but unlisted   -> 0
```

Negatives are retrieved hard negatives rather than unrelated random businesses.

The current classifier is `HistGradientBoostingClassifier` with:

```text
max_iter             = 150
max_leaf_nodes       = 15
min_samples_leaf     = 30
l2_regularization    = 2
positive sample weight = 5
```

Automatic balanced weighting produced too many false positives. A moderate 5x positive weight increased held-out precision from 46.99% to 82.80%.

## Validation methodology

Splits are made by `source1_entity_id`, never by individual pairs:

```text
70% S1 entities -> model training
10% S1 entities -> threshold calibration
20% S1 entities -> untouched evaluation
```

The production pipeline now uses the same 70/10/20 entity split. The threshold is
selected on calibration entities, and macro F0.5 is reported separately on the
untouched evaluation entities. Automatic pair-level early stopping is disabled.
The challenge metric is:

```text
F0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)
```

F0.5 is computed independently for every S1 entity and macro-averaged. True singletons receive 1.0 for a correct empty prediction and 0.0 for any false match.

## Current CPU result

Bounded experiment:

```text
S1 records:  20,000
S2 records: 100,000
S3 records: 100,000
Candidate pairs: 963,977
Positive candidate pairs: 1,233
Untouched test S1 entities: 3,899
Selected threshold: 0.615
```

Held-out metrics:

```text
Precision:  0.8280
Recall:     0.0171
Macro F0.5: 0.0968
```

### Interpretation

Only 2.08% of the full true links were present in the truncated S2/S3 corpus. Candidate recall against complete ground truth was therefore capped at 1.91%, explaining the low recall and macro F0.5.

The precision improvement is encouraging but remains provisional because truncating the targets also removes some hard negatives. This result came from a separate experimental script with different retrieval and features; it must not be treated as the production pipeline's expected leaderboard score. A full-target validation is required.

## Production full-target CPU checks

Both checks used all 5,034,616 Source 2 and 5,285,603 Source 3 training records,
the first 500 Source 1 records, 64 SVD dimensions, and top 20 per source. The
evaluation split contained 86 Source 1 entities.

| Candidate strategy | Candidate recall | Held-out macro F0.5 |
| --- | ---: | ---: |
| Character retrieval alone | 556/1,778 = 0.3127 | 0.5382 |
| Character retrieval + exact blocks | 998/1,778 = 0.5613 | 0.7306 |

These results are directionally useful but the evaluation set is small. A larger
Source 1 sample and wider top-k are the next CPU checks before considering a
multilingual embedding model.

On 2,000 Source 1 records with all targets and top 50 per source, the same
classical pipeline reached 4,169/6,986 = 0.5968 candidate recall and 0.7239
held-out macro F0.5 across 399 evaluation entities with the original train-only
SVD encoder. Repeating the same check with the current encoder, which also fits
on unlabeled test Source 1 text, gave 4,161/6,986 = 0.5956 candidate recall and
0.7141 held-out macro F0.5. The test-text encoder is retained for unseen France.

An encoder-only diagnostic on 10,000 test Source 1 records outside the encoder's
fit sample found that including unlabeled test text raised the median SVD
projection norm for France from 0.267 to 0.529; US and India changed little.
This is not a France matching score.

## Output generation

The production pipeline writes:

```text
output/
|-- candidate_pairs.tsv
`-- matching_results.tsv
```

Inference probabilities are calculated in batches. A temporary SQLite table keeps
output assembly bounded in memory. A bounded smoke run produced one output row
for each of 1,000 sampled S1 records, plus headers, and every final match appeared
in its candidate list.

## Environment and files

Package management uses `uv`:

```text
pyproject.toml
uv.lock
```

Important files:

```text
entity_resolution_experiment.ipynb
code/business_entity_resolution/src/validate_cpu.py
code/business_entity_resolution/src/pipeline.py
code/business_entity_resolution/README.md
```

## Next step

The Kaggle GPU probe with multilingual-e5-small, 2,000 S1 rows, top 50, and
both complete S2/S3 training files reached 5,758/6,986 = 0.8242 candidate
recall at 100% target coverage. The comparable CPU pipeline achieved
4,161/6,986 = 0.5956. This is retrieval-only evidence, not a held-out final
matching score.

`pipeline.py train --retriever hybrid` now unions CPU, E5, and exact-block
candidates, adds E5 similarity/rank features, and retrains the existing
classifier. The immediate Kaggle experiment is the same 2,000-S1/full-target
split at top 50; measure union recall and independent held-out macro F0.5
before predicting test rows. This code has CPU-side checks but has not yet
completed a Kaggle GPU hybrid run. See the package README for the command.

Organizer update: `candidate_pairs.tsv` is part of the final submission, and
smaller candidate sets count toward final ranking beyond the matching-score
leaderboards. The current hybrid union may be large: at top 50 it can contribute
up to 50 CPU and 50 E5 results per target source, plus exact-block candidates.
Before fixing a candidate budget, measure average and tail candidates per S1,
true-link recall at smaller top-k values, and held-out F0.5. Candidate pruning
must happen before the matching model so the TSV remains the exact scored set.
