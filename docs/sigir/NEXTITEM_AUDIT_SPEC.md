# Full-scale next-item audit: analysis specification (binding for `src/confrec/nextitem_audit.py`)

Registered in `idea-stage/PREREG_AMENDMENT_2.md` section N (scoring) and the judge's action 7
(`idea-stage/deliberation_2026-10-02/judge.md`). Scores: zero-shot Qwen3-8B fp16, prompt FROZEN at V0, questions `next`
and `like`, hist_len 5, on the four Lumen same-candidate panels (sports, toys, home, tools; 10,000 test events each,
1 positive + 100 popularity-sampled negatives) plus the first 2,000 VALID events per domain (temperature fits only).
Sports TEST events 1-1000 are reported **separately** from 1001-10000 everywhere (quarantine).

Scope of every claim: same-candidate evaluation with popularity-sampled negatives. No full-catalog claim, no SOTA claim,
no "uncertainty improves ranking" claim. The 8 official baselines and C-CRP v3 are context and a testbed.

## Inputs (per domain d)
- `S_test` = `<audit>/<d>_test/scores.csv.gz` and `S_valid` = `<audit>/<d>_valid2k/scores.csv.gz` (pyes_scorer contract:
  `source_event_id,user_id,item_id,cand_idx,label,question,lp_yes,lp_no,logit,yes_no_mass,censored`, questions `next`,
  `like`; a row with censored 2/3 or a non-finite logit is an unscored candidate).
- `P_test` = `ranking_test.jsonl`, `P_valid` = `ranking_valid.jsonl` (rows with `source_event_id, user_id, history_item_ids,
  candidate_item_ids, candidate_popularity_groups` (head/mid/tail), `positive_item_index`).
- `docs/sigir/ref_ranks/<d>/<method>.csv.gz` (positive ranks of ccrp_v3 + 8 baselines; schema in `export_ref_ranks.py`) and
  `docs/sigir/ref_exposure/<d>/<method>.csv.gz` (written by `scripts/sigir/export_ref_exposure.py`, see below).

## Conventions
- Event = one test event = one user (user-level independence): all CIs are percentile bootstraps over events
  (2,000 resamples, seed 0 unless stated); paired comparisons resample the same events.
- Ranks are tie-aware: expected 1-based rank under uniformly random tie-breaking
  (`stats.tie_aware_rank` semantics); an unscored positive ranks last, unscored negatives rank below the positive.
  NDCG@10 / HR@10 / MRR are reported at the expected rank and, as `*_tie_exact`, as the exact expectation over the
  positive's tie group (as `pilot_mirror._event_metrics`). Vectorise over the (events x 101) array.
- Popularity group of a candidate = `candidate_popularity_groups` (head/mid/tail, the existing Lumen split).
- Everything is computed per question q in {next, like}; `next` is primary, `like` secondary.
- All JSON strict (`stats.strict_json`); every quantity carries n and CI where it is an estimate.

## A. Ranking quality (context)
Per event NDCG@10, HR@10, MRR for the LLM (q) ; mean with CI. Paired dNDCG@10 / dHR@10 of the LLM minus each of the 9
reference methods on the joined events (positive_rank from ref_ranks). Report n_joined and censoring counts.

## B. Static exposure (S3) - the registered exposure endpoints
Top-10 of an event = the 10 highest-scored candidates (deterministic ties: lower `cand_idx` first; report the number of events whose
cutoff is tied). For the LLM (each q) and for each reference method (from `ref_exposure`: top-10 item ids and groups):
1. `head_share_top10` = mean over events of (#head in top-10)/10; `mid`, `tail` (= APLT) likewise.
2. `pool_head_share` = mean over events of (#head candidates)/101 (the expected head share of a random top-10);
   **`delta_head = head_share_top10 - pool_head_share`** with event-bootstrap CI (the S3 endpoint), and the same for tail.
3. `target_head_share` = P(positive is head) over the same events; `delta_head_vs_target = head_share_top10 -
   target_head_share` with CI.
4. `coverage_top10` = #distinct items in all top-10 lists / #distinct candidate items of the pool of that domain;
   `gini_exposure` = Gini coefficient of the exposure counts (number of events whose top-10 contains the item) over the pool
   items (items that are a candidate in >= 1 event of the domain's test panel; zero counts included).
5. Paired difference of `head_share_top10` between the LLM and each reference method (same events).
**Admission rule (registered):** an S3 effect is claimed for a method only if `delta_head` has a CI excluding 0 with the
same sign in at least 3 of the 4 domains; sports events 1-1000 and 1001-10000 are reported separately.

## C. List-normalised calibration and error anatomy (S1, S2, S4)
List-normalisation: `p_c = softmax(L_c / T)` over the scored candidates of an event. T is fitted ONCE per (domain, q) on
VALID2k by minimising the mean NLL of the positive's probability (convex in 1/T: use a bounded 1-D search on beta=1/T in
[1e-3, 10]); T is applied unchanged to TEST. Report T, NLL(T=1), NLL(T fitted), NLL of the uniform distribution (ln 101).
On TEST:
1. top-1 confidence `p_max`; correctness `1[top-1 is the positive]`; `ECE` (15 equal-width bins) and adaptive ECE,
   reliability bins (n, mean confidence, accuracy), multiclass Brier (mean over events of sum_c (p_c - y_c)^2), NLL.
2. Discrimination of confidence for correctness: AUROC of each signal for top-1 correctness and for `HR@10` hit:
   signals = `p_max`, `margin` (= p_1 - p_2), `neg_entropy`, `max_logit` (uncalibrated), and `random` (a fixed-seed uniform
   draw, the null). Paired AUROC differences vs `p_max` with CI.
3. Error anatomy (answers S1 "correct but sure?" and S2 "wrong because unsure?"): partition events into confidence tertiles by
   `p_max` (tertile edges computed on TEST): for each tertile the top-1 accuracy, HR@10, and the two registered rates
   `confident_error_rate` = P(top-1 wrong | top tertile) and `unsure_correct_rate` = P(top-1 correct | bottom tertile),
   plus the share of all top-1 errors that fall in the top tertile.
4. Pointwise S4: pooled AUROC of the logit for the label over all 1,010,000 candidate rows and the mean per-event AUC
   (UAUC; equal to 1 - (rank - 1)/100 on average), per q.

## D. Selective serving (cross-user decisions)
Serve the events with the highest confidence first. For each signal in {p_max, margin, neg_entropy, max_logit, random}:
risk-coverage curve over coverage in {0.1, 0.2, ..., 1.0} with utility = NDCG@10 of the event (and HR@1 as a second utility);
`AURC` (tie-invariant, `metrics.risk_coverage`); the paired difference AURC(signal) - AURC(random expectation = 1 - mean utility)
with CI; utility at 50% coverage minus the full-coverage utility (the registered gain of serving the confident half).
**Niche-user coverage at 50% coverage:** user popularity profile = mean over the user's history items of the group score
(head 2, mid 1, tail 0), using an item->group map built from all candidate lists of the domain's panel (history items missing
from the map are skipped; users with no mapped history item are excluded and counted); quintiles of the profile by
`stats.rank_bins(profile, 5)` (bin 0 = niche, bin 4 = mainstream). For every signal report, per quintile, the share of the
quintile's events that are served at 50% coverage (a signal that silences niche users shows a low share in bin 0), and the
utility among served events; plus the niche-minus-mainstream difference in served share with CI.

## E. Popularity-graded confidence and calibration (S5)
On the TOP-10 candidates of each event (the served set) with the list-normalised probability p and label y:
1. mean `p` by popularity group of the candidate (head/mid/tail) and mean event-centred logit by group;
2. calibration residual mean(y - p) by group with event-cluster CIs;
3. `metrics.bias_index_ci(conf=p, correct=y, group=popularity group, clusters=event id, adjust=True)` (500 resamples) :
   the confidence-matched head-vs-tail calibration gap;
4. head-minus-tail difference of mean p with CI. (Within-event: also report the fraction of events whose top-1 is head.)

## F. Cross-domain summary
`summary.json`: for each of S1..S5 the registered endpoint values per domain (and per sports split), Holm-adjusted
(m = 4 domains per endpoint) bootstrap-based sign statements, and the admission flags. Never write prose; numbers only.

## G. Endpoint registry, segments and freeze (added 2026-10-04, before any LLM audit score was analysed)
- **Registry.** The registered endpoints of the cross-domain summary (S1, S2, S4, S5, and A as context) are the entries of the
  dictionary `Q_ENDPOINTS` in `src/confrec/nextitem_audit.py` at the sha1 recorded in PILOT_LOG. Where sections C-E above name an
  endpoint (`confident_error_rate`, `unsure_correct_rate`, the AUROC of `p_max`, `bias_index`, ...) the dictionary uses it; the
  dictionary completes the sections where they were silent (the signed contrasts are those with a Holm sign statement).
- **Segments.** Sports yields two segments in panel file order: `events_1_1000` (quarantine) and `events_1001_10000` (main).
  The summary uses `main` as the sports entry in every Holm family (m = 4) and in the 3-of-4 rule; the quarantine segment is
  reported under its own unit and is never counted.
- **S3 admission** requires all four domains present, and a CI excluding 0 with the same sign in at least 3 of them; it is
  evaluated for the LLM (`next`, `like`) and for every reference method.
- **Unscored candidates** (censored 2/3 or non-finite logit) rank below all scored ones, an unscored positive ranks last, they
  get probability 0 in the list softmax; an event with no scored candidate stays in A and UAUC and is excluded from B-E (counted).
- **Addendum 2026-10-04 (niche bins).** The first real run (sports) returned the registered niche-minus-mainstream difference as
  undefined, because about half of the sports users have every mapped history item in the head group, so the popularity profile
  is heavily tied and `rank_bins` leaves the top quintile empty. Niche is now the lowest and mainstream the highest NON-EMPTY
  quintile (reported as `niche_bin`, `mainstream_bin`). The change was made after that field came back empty and before any niche
  value (served share, utility among served) had been looked at; nothing else changed.
- **Freeze.** Before the first run of `nextitem_audit run` on real LLM scores, PILOT_LOG records the sha1 of `nextitem_audit.py`,
  of this file, of `scripts/sigir/export_ref_exposure.py` and the manifest hash of `docs/sigir/ref_exposure` (the sha1 of the lines
  `<relative path> <sha1>` over its files in sorted order). Before this record the only audit output read was the scorer's own
  progress lines (chunk counters, prompts per second).

## CLI
```
python -m src.confrec.nextitem_audit run --domain sports --audit_dir <audit root> --panel_test P --panel_valid P \
    --ref_ranks docs/sigir/ref_ranks/sports --ref_exposure docs/sigir/ref_exposure/sports --out <dir>/sports.json \
    [--n_boot 2000] [--seed 0] [--questions next,like]
python -m src.confrec.nextitem_audit summarize --inputs <dir>/sports.json,<dir>/toys.json,... --out <dir>/summary.json
```

## `scripts/sigir/export_ref_exposure.py` (local, reads the ranking_eval_records.csv of C-CRP and the 8 baselines)
Writes `docs/sigir/ref_exposure/<d>/<method>.csv.gz` with columns `source_event_id,user_id,positive_rank,positive_group,
top10_item_ids,top10_groups,n_head_pool,n_mid_pool,n_tail_pool` (id / group lists space-joined, from the first 10 entries of
`pred_ranked_item_ids` and the matching groups looked up from `candidate_item_ids`/`candidate_popularity_groups`) and
`docs/sigir/ref_exposure/<d>/_pool.json` with `n_pool_items` (distinct candidate items over all events). Asserts that every
event of docs/sigir/panel_refs/<d>_test.json is present exactly once and the candidate lists are identical across methods.
