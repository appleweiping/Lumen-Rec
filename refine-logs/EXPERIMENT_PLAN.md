# Experiment Plan — SIGIR 2027 (ARIS Workflow 1.5 input)

> Status: **CONDITIONAL on Pilot 1** (`idea-stage/triage_verdict.md` §3.1). Branch A = MIRROR passes; Branch B =
> fallback (B9/A9 evidence decomposition). Verdicts are same-family/provisional. Old pre-SIGIR plans:
> `refine-logs/archive_pre_sigir_2026-05/`.

**Problem.** A yes/no LLM recommender's confidence P(Yes) mixes *what it knows about the item* (familiarity,
popularity, acquiescence) with *what it knows about the user*. Only the second is a recommendation signal; the
first produces confident errors, popularity-skewed confidence and head over-exposure — and, being item-level, it
is NOT removed by within-user monotone calibration (Lumen's negative result N1).

**Method thesis (Branch A).** Ask the mirrored question: MIRROR(u,i) = logit P(Yes|like?) − logit P(Yes|dislike?)
cancels the item-level acquiescence a(i) = ½(logit_like + logit_dislike), which changes within-user rankings,
improves personal discrimination (UAUC) and tail calibration; *mirror tuning* trains both questions with
consistent labels so the cancellation survives fine-tuning.
**Branch B thesis.** Evidence = logit_like(u,i) − π(i), π = user-marginalised item prior (user-swap), as the
confident-error detector and cross-user decision signal.

## Claim Map
| Claim | Why it matters | Minimum convincing evidence | Blocks |
|---|---|---|---|
| C1 (method) | The first confidence correction for LLM recs that legitimately changes within-user ranking, answering S1/S2/S5 with an actionable fix | UAUC gain of MIRROR over (i) raw, (ii) equal-compute paraphrase ensemble, (iii) PMI/no-history & user-swap prior subtraction, CI excluding 0, on ≥3/4 rated datasets × 2 backbones, zero-shot AND fine-tuned (3 seeds); no NDCG@10 loss on next-item panels | B1, B2 |
| C2 (observation) | Answers S1–S5 with correct labels: where confident errors come from, popularity-graded acquiescence, exposure skew | within-user SD of a(u,i) ≥ 0.2 logit & corr(a, log-pop) ≥ 0.2 (pair level, user-cluster CI; amendment P1.1); confident-error share carried by item term; residual-adjusted confidence-matched Bias Index with CI; head share of top-10 raw vs MIRROR; pseudonym knockout with full-coverage placebo (causal) | B3, B4 |

## Experiment Blocks
### B1 — Main result (MUST-RUN)
- **Data.** Rated panels (label = rating ≥4 vs ≤2; 3 dropped; first event per (user, item)): ML-1M, Amazon-2023
  Toys, Video_Games, Sports (≤5k users × ≤20 rated candidates each).
  - Sports replaces All_Beauty, which yields only 37 eligible users (`idea-stage/PREREG_AMENDMENT_1.md` §D).
  - Next-item panels: Lumen frozen sports/toys/home/tools (10k × 101). All four reproduce 10000/10000 events
    (tools with its original `seed=42, max_history_len=10`).
  - Per-event ranks of C-CRP v3 and the 8 official baselines: `docs/sigir/ref_ranks/<domain>/<method>.csv.gz`.
- **Scoring numerics.** fp16 (bf16 put 57% of logits on a 0.25 grid); Yes/No read by token id from the top-50
  logprobs, with censoring flags; prompts passed as token ids (the scorer and the LoRA trainer share one helper).
- **Leakage rules for fine-tuning.**
  - LoRA training panels exclude every user in the evaluated domain's Lumen valid/test `selected_users`.
  - Rated UAUC experiments use a global temporal split at the 80th percentile of candidate timestamps as the
    primary split, with a user-disjoint split as robustness.
- **Backbones.** Qwen3-8B, Llama-3.1-8B-Instruct (thinking off).
- **Regimes.** Zero-shot; fine-tuned LoRA (TALLRec-style yes/no SFT, r=16, α=32, 1 epoch, 3 seeds).
- **Systems (≤3 baseline families).** (F1) single-prompt confidence: token P(Yes) (TALLRec/CoLLM scoring rule),
  verbalized (C-CRP), self-consistency (K=5, T=0.7); (F2) prior-subtraction calibrators: PMI/no-history,
  contextual calibration (content-free input), batch calibration, user-swap prior; (F3) trained calibration:
  temperature scaling (val), ConfTuner-style Brier SFT. Ours: MIRROR (zero-shot), mirror tuning (fine-tuned).
- **Metrics.** Primary UAUC; secondary global AUC, ECE/Brier (val-temperature-scaled for fairness), tail-item
  UAUC, NDCG@10/HR@10 on next-item panels; paired user-bootstrap 95% CI; fine-tuned: mean ± sd over 3 seeds and
  gain > 2σ_seed.
- **Success.** C1 evidence above. **Failure interpretation.** If MIRROR ≈ PMI baseline → the item term is a plain
  prior (report as such; Branch B).

### B2 — Ablations (MUST-RUN)
like-only / dislike-only / mirrored; paraphrase sets (2 vs 4 prompts, equal compute); history length {0,5,10,20};
mirror tuning vs standard SFT vs SFT+dislike-data-only; MIRROR on verbalized confidence (does the effect need
token probabilities?).

### B3 — Observation study (MUST-RUN; answers S1, S2, S4, S5)
Reliability diagrams of token vs verbalized confidence vs rated labels; error anatomy (correct-but-unsure,
confidently-wrong, by item/personal term); confidence level AND confidence-matched gap by popularity decile
(ProCal Bias Index); a(i)/v(i)/π(i) vs log-pop; pseudonym knockout with real-brand placebo (Pilot 3 scaled up).

### B4 — Validity + exposure (MUST-RUN, small)
KuaiRec MAR vs logged calibration (Pilot 2 scaled to all 1,411 users); static exposure: head share of top-10 and
tail coverage under raw vs MIRROR on the 4 next-item panels (answers S3 statically).

### B5 — Appendix (NICE-TO-HAVE)
S6: detection of natural rating–review contradictions by {raw entropy, MIRROR |d|, evidence} + ≥5-seed
matched-random pruning; thinking-mode calibration (A11/B11); 2-round feedback loop on KuaiRec (only if Pilot 2
POSITIVE and §3.4 conditions hold); per-user offset lemma (UAUC invariance).

## Run Order
| Milestone | Goal | Runs | Decision gate | Cost (GPU-h) |
|---|---|---|---|---|
| M0 Sanity | server, panels verified, token channel | rebuild_panels + Pilot-1 gate | ML-1M UAUC ≥0.60; sports NDCG ≥0.186 | 3 |
| M1 Pilots | decide branch | Pilots 1–3 | pre-registered rules (triage §3) | 5 |
| M2 Zero-shot grid | B1/B3 zero-shot | 4 rated × 2 backbones × arms; 4 next-item × 2 | C1 zero-shot criterion | 15 |
| M3 Fine-tuned grid | B1 fine-tuned | 4 × 2 × {SFT, mirror tuning, Brier SFT} × 3 seeds | C1 fine-tuned criterion | 40 |
| M4 Ablations + B4 | B2, B4 | as listed | each component matters | 12 |
| M5 Appendix | B5 | as listed | — | 8 |

## Compute Budget
~83 GPU-h total on one A100/A800-80G-class GPU (≈2× on a 24 GB card with smaller batches). Bottleneck:
next-item panels (8M prompts per backbone; prefix caching essential) and the 72-run LoRA grid.

## Risks
- Item term is global, not item-specific (Braun EMNLP-F'25, "Yes is harder than No" CIKM'25) → Pilot 1 gate on
  SD(a); Branch B fallback.
- Negation handled poorly by 8B models → dislike-question validity check (AUC of −logit_dislike on labels).
- Amazon rating skew (≈80% ≥4★) → users filtered to ≥3 likes & ≥3 dislikes; report like-rate per dataset.
- Seed variance in LoRA (σ≈1.5pt in sibling project) → 3 seeds, gain > 2σ rule, paired tests.
- Same-family review bias → final human/cross-family check flagged in every review artifact.
