# IDEA_REPORT — Lumen-Rec → SIGIR 2027 (ARIS Workflow 1: idea-discovery)

> Executor: Claude Opus 5.5. Reviewer: fresh Claude Opus 5.5 sub-agents (one per verdict, read-only, absolute
> paths only). **review_independence = same-family; acceptance_status = provisional** for every gate below
> (ARIS mainline requires a cross-family reviewer; the user mandated Opus-only, so ARIS's documented
> same-family fallback is used and every verdict is labelled provisional). AUTO_PROCEED = true.

## Journey
| Step | Artifact | Result |
|---|---|---|
| research-lit (Hooi group) | `docs/sigir/LIT_HOOI_GROUP.md` | Hooi group has **no** LLM4Rec uncertainty paper; templates: confidence elicitation (ICLR'24), ProCal (NeurIPS'23), ConfTuner (NeurIPS'25), UNIT, MKJ |
| research-lit (recent) | `docs/sigir/LIT_RECENT_UNC_LLM4REC.md` | ~70 papers; closest: UQRec, UGR, EviRank, Ravikumar, KnowSA, GORACS/DEALRec; 4 open gaps |
| idea-creator Phase 2 | `idea-stage/brainstorm_{A,B}.md` | 22 ideas from 2 independent Opus generators (B with causal/decision-theory lens + 5 inverted assumptions) |
| Phase 3 consolidation | `idea-stage/triage_bundle.md` | 7 clusters + 4 singletons; none budget-infeasible |
| Phase 4 triage (jury) | `idea-stage/triage_verdict.md` | Top-1 **A1 MIRROR** (conditional on pilot); executor's CPU "complementarity" pilot **refuted** (see below) |
| Phase 4 novelty-check | `idea-stage/NOVELTY_DOSSIER.md` | A1 5/10 PROCEED WITH CAUTION; B1/A7 6/10 PROCEED (validity section); A2 6/10 PROCEED |
| Phase 5 pilots | `scripts/sigir/run_pilot1_mirror.sh` (+ P2, P3 pending) | **BLOCKED: GPU server SSH key not yet installed** |

## Refuted along the way (kept as negative evidence)
- **LLM↔CF deferral headroom** (`docs/sigir/PILOT_LOG.md`): the +0.05–0.08 NDCG@10 "oracle complementarity"
  is ~half a max-of-noisy-rankers artifact (vs uniform-random ranker +0.029–0.037, executor-verified);
  realistic tuned fusion +0.002–0.012 (reviewer-reported). Cluster C4 (A4/A5/A9/B3) deprioritised.

## Selected direction (Gate 1, AUTO_PROCEED; provisional)
**Spine:** *"Yes because it knows the item, or because it knows the user?"* — the token-level yes/no
confidence of an LLM recommender decomposes into an **item term** (popularity/familiarity-linked) and a
**personal-evidence term**; only the latter is the recommendation signal.
- **Method candidate (A1 MIRROR):** score = logit P(Yes | "like?") − logit P(Yes | "dislike?"), cancelling the
  item-level acquiescence a(i) and thereby changing within-user rankings (escaping the rank-preservation
  negative N1). Allowed novelty claim (dossier): the artifact has an **item-specific, popularity-graded**
  component in **personalised** recommendation; its removal changes rankings and improves UAUC/tail calibration
  **beyond an equal-compute paraphrase ensemble and beyond PMI-style item-prior subtraction**. NOT claimable:
  the two-prompt estimator itself (CCS; arXiv 2607.05552), negation-consistent training (REPAIR).
- **Fallback flagship (if pilot 1 NEGATIVE/NULL):** B9/A9 user-swap evidence decomposition
  (logit_like − π(i)), used for confident-error detection + cross-user decisions.
- **Diagnostics feeding the spine:** A2 pseudonym knockout (causal test of S5; interprets a(i)); B1/A7 MAR vs
  logged calibration on KuaiRec/Coat (validity of every "confidently wrong" claim, S2/S4); A11/B11 thinking
  mode; B6 niche users; A10/B7 per-user offset (UAUC-invariance lemma).

## Where the user's six questions land
| Seed question | Answered by |
|---|---|
| S1 correct but unsure? | §RQ1 error anatomy on rated panels: share of correct-but-unsure / confidently-wrong, by term |
| S2 wrong because low confidence? | §RQ1 confident-error AUROC of raw vs MIRROR vs evidence; MAR re-labelling (pilot 2) |
| S3 echo chamber? | §RQ4 static exposure (head share of top-10, raw vs MIRROR); dynamic 2-round loop in appendix if pilot 2 passes |
| S4 yes/no confidence vs ground truth | §RQ1 calibration of token P(Yes) vs **rated** labels (fixes CALIBRA's 1/101 label flaw); verbalized vs token |
| S5 popular high / niche low? | §RQ2 confidence level vs confidence-matched gap by popularity; a(i)/v(i)/π(i) vs popularity; pseudonym knockout |
| S6 prune training by uncertainty? | Appendix: natural rating–review contradiction detection by 3 uncertainty signals + ≥5-seed matched-random pruning |

## Pilot plan (pre-registered in `triage_verdict.md` §3; decision rules there are binding)
1. **Pilot 1 MIRROR** (~1.7 GPU-h): ML-1M + Amazon Toys rated panels (1.5k users × ≤20 rated cands), Lumen sports
   next-item (1k × 101). Arms: like / dislike / like_para (placebo) / user-swap prior (K=8) / no-history PMI prior.
   Gate first: ML-1M UAUC(raw) ≥ 0.60 and sports NDCG@10(raw next) ≥ 0.186. Code: `src/confrec/{pyes_scorer,
   build_rated_panels,pilot_mirror}.py`, tests green (8/8).
2. **Pilot 2 KuaiRec MAR vs logged** (≤2 GPU-h): gates B1/A7/B2/A6/B10.
3. **Pilot 3 pseudonym knockout** (~1.2 GPU-h): Beauty + Toys, with real-brand-swap placebo.

## Status
`idea-discovery`: **done** (executor) — **accepted: provisional (same-family)**. Next: Phase 5 pilots →
research-refine + experiment-plan → experiment-bridge. Blocked only on server access.
