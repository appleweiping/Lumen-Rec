# Pre-registration amendment 1 (2026-10-01): pilots §3.1–§3.3

**Status: binding.** Written before any pilot data existed. The only GPU run so far is a 20-user ML-1M smoke
test (778 prompts) used to check the infrastructure; no pilot endpoint was computed from it.

**Trigger.** An adversarial pre-GPU code review (5 lenses × 3 skeptics, about 45 confirmed findings, 3 critical)
found estimands that were degenerate or ill-defined as implemented. The rule for this amendment:
- Change only operationalizations (what is computed, and on which unit).
- Change a design element only where the registered version cannot produce a non-degenerate answer.
- Keep every decision threshold of `triage_verdict.md` §3.

Anything not listed below is unchanged.

## C. Common to all pilots

**C0 Scoring numerics.**
- Qwen3-8B runs in **fp16, not bf16**.
  - bf16 rounds the output logits, so 57% of log P(Yes) − log P(No) values fall on a 0.25-logit grid.
  - fp16 gives a 1/32 grid.
  - On the 389-prompt probe, bf16 and fp16 logits correlate at ρ = 0.999, with no non-finite values in fp16.
- Yes/No are read by **token id**. The id sets are resolved from the tokenizer and logged.
- 50 top logprobs are requested. If the Yes or No side is missing from the top 50:
  - that side is imputed with the smallest returned logprob (an upper bound), and the row is flagged `censored=1`;
  - if both sides are missing, the logit is set to NaN, the row is flagged `censored=2`, and it is excluded and counted.
- Prompts are rendered by the chat template and tokenized once, with no extra special tokens, then passed to vLLM
  as token ids. The LoRA trainer uses the identical helper, which removes the double-BOS mismatch for Llama-3.1.
- Every history title and candidate title is capped at 200 characters in both paths. Prompts longer than the
  model context are skipped and counted, never silently truncated.

**C1 Inference units.**
- Every CI resamples the independent unit:
  - users, for pair-level statistics;
  - items, for item-level correlations.
- Spearman correlations average ties.
- Ranks on next-item panels are tie-aware: the expected rank under random tie-breaking.

**C2 Popularity (unchanged).**
- Popularity stays the all-time category event count, as in Lumen's own convention. It never reads ratings, and
  every use is rank or quantile based, so the candidate's own +1 is a uniform shift.
- The review's objection (data_integrity#10) was refuted 3/3.
- The count of events strictly before the candidate's timestamp is added as a robustness column.

**C3 Rated panels.**
- Keep the first event per (user, item). This removes a confirmed leak where a re-reviewed item appears in the
  history with its stars and again as a labelled candidate.
- Same-second ties keep the deterministic (ts, item) order. That objection (data_integrity#11) was refuted 2/3:
  the order is arbitrary but leaks nothing.

**C4 Panel integrity.**
- The pilots read the Lumen sports panel only after `panel_reference.py verify` passes.
- Status at amendment time:
  - sports, toys and home reproduce 10000/10000 events.
  - tools initially failed (0/10000) because the rebuild used the other domains' builder arguments. The original
    tools panel was built with `seed=42, max_history_len=10`, recovered from the baseline run summaries. Rebuilt
    with those, it reproduces 10000/10000.
  - All four panels are therefore exact.

## P1. Pilot 1 (§3.1 MIRROR)

**P1.1 SD(a) and corr(a, log-pop).**
- `SD(a)` is the pooled within-user SD of the user-centred acquiescence a(u, i).
  - This is the quantity that decides whether MIRROR re-ranks candidates within a user (rank relevance), which is
    the gate's stated purpose.
  - On Amazon panels almost every item occurs for one user (n_i ≈ 1), so an item-level SD is not identifiable there.
  - The between-item SD is still reported, but does not gate. It is computed by method of moments on items with
    n_i ≥ 2: Var(ā) − mean(s²_i / n_i).
- `corr(a, log-pop)` is the Spearman correlation over pairs of the centred a(u, i) with log1p(all-time popularity,
  C2), with a user-cluster bootstrap CI. Prior popularity is a robustness check.
- The item-mean version is a secondary measure.
- The same definitions apply to v and π.

**P1.2 Calibration.**
- Raw-scale ECE and Brier use MIRROR on its half-difference scale, (l_like − l_dislike)/2. Ranking metrics are
  unaffected.
- For the cross-arm calibration comparison, every arm is also Platt-mapped on a 50% user split and evaluated on
  the other half; this is the primary comparison.
- Evidence and PMI arms (residuals, not probabilities) get only the Platt version.

**P1.3 Confident-error detection.**
- The target is the same for every detector: the errors of the raw *like* decision (σ(l_like) ≥ 0.5 vs label).
- Detectors:
  - |l_like|;
  - |MIRROR|;
  - |evidence|;
  - MIRROR-vs-raw sign disagreement.
- Report AUROC and tie-invariant AURC for each, with paired user-bootstrap Δ vs |l_like|.

**P1.4 Next-item endpoints (sports, first 1,000 verified events).**
- Per-user NDCG@10 from tie-aware ranks.
- Paired user-bootstrap ΔNDCG@10 for:
  - MIRROR − raw(next);
  - MIRROR − raw(like);
  - MIRROR − placebo;
  - each arm − C-CRP v3 on the same events. C-CRP ranks: `docs/sigir/ref_ranks/sports/ccrp_v3.csv.gz`.
- The registered rule "no NDCG@10 loss > 0.005" must hold on the point estimate against **both** raw(next) and
  raw(like). CIs are reported.

**P1.5 Sanity gate.**
- The registered constants are kept: ML-1M UAUC ≥ 0.60 and sports NDCG@10 ≥ 0.8 × 0.2329.
- C-CRP on the same 1,000 events (0.2310, giving a threshold of 0.1848) is printed alongside.
- The gate script exits non-zero on FAIL and writes a `GATE_PASS` / `GATE_FAIL` marker.

**P1.5b Orchestrator sign-off (2026-10-02, before any pilot data).** The fix review left four
operationalizations open. They are fixed as follows; each rejected alternative is still computed as a sensitivity
in `decision.json`.
- `SD(a)` uses the textbook pooled within-user SD, with divisor N − n_users (`SD_pair_df`). Divisor N is biased low
  by √(1 − 1/n_u).
- Popularity is all-time (C2).
- NDCG@10 in the no-loss rule is the exact expectation over the positive's tie group. The plug-in expected rank is
  biased because NDCG is nonlinear.
- When NEGATIVE and NULL both hold, **NULL** wins. NULL's registered meaning ("acquiescence is large and
  popularity-linked, but there is no UAUC gain") describes exactly that case.

**P1.6 Swap prior.**
- Donors exclude every user who holds item i in their history or as a candidate.
- Exclusion counts are logged.

## P2. Pilot 2 (§3.2 KuaiRec): the registered MNAR view was degenerate

**Why the registered view fails.** KuaiRec removes every small-matrix interaction from the big matrix by design.
"Drop shared pairs" together with "unobserved-in-big = negative" therefore makes every MNAR label 0:
- AUC_MNAR = NaN;
- ECE_MNAR = mean(p);
- the POSITIVE rule is met mechanically.

**Replacement.** A logged view is built on the **same** fully observed pairs, so the labelling protocol is the
only difference between the two views:

- **Candidates (unchanged).**
  - 300 users × 200 small-matrix videos.
  - MAR label 1[watch_ratio ≥ 2.0].
  - Threshold sensitivity {1.0, 3.0} is recomputed on CPU from the stored watch_ratio.
- **Exposure.** O(u, v) ~ Bernoulli(e(u, v)), where:
  - e(u, v) = min(1, c · n_u · n_v^α);
  - n_v = the number of big-matrix users (all 7,176) exposed to v;
  - n_u = u's big-matrix exposure count;
  - α ∈ {0.5, **1 (primary)**, 2};
  - c is set so that mean e(u, v) equals the big-matrix density over small-matrix videos.
- **Logged label.** O · MAR label, with unobserved treated as negative; this is the convention under test.
  - 20 exposure draws, seeds 0–19.
  - Endpoints are averaged over draws. CIs come from user bootstraps pooled across draws.
- **Endpoint 1.**
  - The share of top-decile-confidence logged false positives that are MAR positives.
  - Also reported as a lift over the MAR-positive rate among all logged negatives, with CI.
  - The threshold is unchanged (≥ 20%).
- **Endpoint 2.** Exposure gap = ECE_logged − ECE_MAR, on the same pairs and scores, with a user-bootstrap CI.
- **Endpoint 3.**
  - Bias Index, head vs tail by n_v, under each view, with CI.
  - Bias Index is now residual-adjusted: per confidence bin it compares (acc − conf) between groups, so
    within-bin confidence differences cannot create a sign.
- **Endpoint 2 is close to mechanical (addendum 2026-10-02, still before data).**
  - With logged = O · MAR, |ECE_logged − ECE_MAR| ≤ MAR rate − logged rate, so a CI excluding 0 is close to
    guaranteed for any yes-leaning model.
  - POSITIVE therefore additionally requires the endpoint-1 lift CI to exclude 1 (lower bound > 1). This is
    conservative: it can only turn POSITIVE into AMBIGUOUS.
  - `gap_normalized` (gap / ceiling) is reported. Endpoint 3, the popularity-differential mislabelling, is the
    informative quantity for the observation study.
- **NULL gate.**
  - The gate is on **UAUC_MAR** (per-user AUC); pooled AUC is reported as well.
  - The LoRA retry is moved to M3 and is not run in the pilot. NULL therefore reads: "zero-shot UAUC_MAR < 0.58".
- **History (unchanged).** u's last 10 big-matrix positives over the whole log, excluding candidate videos.
  This is the standard KuaiRec protocol: the big matrix is the logged context and the small matrix the MAR test
  set. A time-bound objection (data_integrity#9) was refuted 3/3.
- **Captions.**
  - Parsing is NaN-safe, and `UNKNOWN` counts as missing.
  - Videos with no caption and no cover text are excluded.
- **Popularity.** Computed from the unfiltered big matrix.

## P3. Pilot 3 (§3.3 pseudonym knockout)

**Domains.** Toys and **Video_Games**. All_Beauty yields only 37 eligible users, against a minimum of 300.

**Substitution.**
- One single-pass, whole-word, case-insensitive regex.
- A stop-list excludes generic or placeholder stores and strings shorter than 4 characters or all-digit.
- Pseudonyms are injective within a panel and never equal a real store string.

**Placebo.**
- Maps **every** brand the pseudonym arm maps, history-only brands included.
- Brands are matched by tercile of category-wide brand review counts.
- Both arms must have identical substituted key sets (asserted), and per-row substitution counts are logged.

**Treated set and scope.**
- Treated candidates are those whose store string occurs as a whole word in their own title.
- "Brand tokens in titles" beyond the `store` field are not implemented. Scope note: the analysis population is
  store-in-title items, and their share is reported.

**Statistics.**
- Every Δ CI is a user-cluster bootstrap.
- The decile slope is the OLS of Δ on log-pop, with a user-cluster CI. Deciles are cut on average ranks, so no bin
  is empty.
- UAUC is computed over **all** candidates (primary) and over treated candidates.
- ΔUAUC is computed with paired user bootstraps for real − pseudo and real − placebo, overall and in the tail.
- The decision thresholds (0.20 / ±0.05 / 0.01 / 0.02) apply to the point estimates, with CIs reported.

**Cross-check (§3.3).**
- The real and pseudonym arms are scored with *like*, *dislike*, and an 8-donor swap prior.
- corr(a, log-pop) and corr(π, log-pop) are reported per arm, together with real − pseudo and its user-cluster CI.

## D. Downstream scope (refine-logs/EXPERIMENT_PLAN.md)

**Rated datasets.** ML-1M, Toys, Video_Games and **Sports**. Sports replaces the too-sparse All_Beauty and links
the rated study to the sports next-item panel.

**LoRA training panels.**
- Exclude every user in the evaluated domain's Lumen valid and test `selected_users`.
- The rated UAUC experiments use a global temporal split as the primary split:
  - training candidates are earlier than T and test candidates are at T or later;
  - T is the 80th percentile of candidate timestamps.
- A user-disjoint split serves as the robustness check.
