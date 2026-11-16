# Candidate Protocol

The main claim is a controlled candidate-ranking decision study unless a
full-catalog evaluation is explicitly marked as completed.

## Required audit

Run `main_audit_candidate_protocol.py` for every domain used in a main table.
The output is:

```text
outputs/summary/candidate_protocol_audit.csv
```

Required fields include:

- `domain`
- `split`
- `num_users`
- `num_events`
- `candidate_set_size_mean`
- `candidate_set_size_min`
- `candidate_set_size_max`
- `positives_per_event`
- `negative_sampling_strategy`
- `hard_negative_ratio`
- `popularity_bin_distribution`
- `duplicate_title_rate`
- `title_overlap_or_duplicate_rate`
- `user_overlap_valid_test`
- `item_overlap_train_test`
- `one_positive_setting`
- `hr_recall_equivalent_flag`
- `full_catalog_eval_available_flag`
- `status_label`

## HR@K and Recall@K

HR@K and Recall@K are numerically equivalent only in one-positive candidate
events. The audit sets `hr_recall_equivalent_flag=true` only when every audited
event has exactly one positive item.

## Ranking-metric input contract

`src.eval.ranking_metrics.compute_ranking_metrics` fails fast before computing
HR, NDCG, or MRR when the input frame is malformed. The frame must contain the
selected user and rank columns plus a binary `label` column. Labels must be
finite `0`/`1` values, ranks must be finite positive integers, user identifiers
must be present, and `k` must be a positive integer. A different rank-column
name is supported through the `rank_col` argument. Tied ranks remain valid for
diagnostic outputs, while the candidate protocol audit is responsible for
checking the exact candidate-key and ordering contract used by a main result.

This validation prevents coercion of fractional labels, missing ranks, or
non-numeric values from silently changing a reported metric.

## Candidate-set claim boundary

If `full_catalog_eval_available_flag=false`, the paper must not claim
full-catalog recommender SOTA. It may claim controlled candidate-ranking or
reranking behavior under the audited sampling protocol.

## Example

```bash
python main_audit_candidate_protocol.py \
  --domain beauty \
  --data_dir data/processed/amazon_beauty \
  --negative_sampling_strategy sampled_candidate_one_positive \
  --output_path outputs/summary/candidate_protocol_audit.csv
```
