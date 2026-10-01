# Research Brief — Lumen-Rec → SIGIR 2027 (uncertainty-aware LLM4Rec)

> ARIS input for `/research-pipeline` (Workflow 1 → 1.5 → 2 → 3). Executor **and** reviewer = Claude Opus 5.5
> (reviewer = fresh, context-free Opus sub-agent per round; no Codex/GPT). Long material lives in
> `docs/sigir/` (literature: `LIT_HOOI_GROUP.md`, `LIT_RECENT_UNC_LLM4REC.md`) and in the prior-evidence docs
> cited below. Created 2026-10-01.

## Problem Statement
LLM recommenders (pointwise yes/no scorers à la TALLRec, generative title generators à la BIGRec/D3) are
deployed as if their outputs were equally trustworthy. But an LLM can output the right item **without being
sure**, or be **confidently wrong**. The lab seed idea (#4) asks: (Q1) when the LLM recommends correctly, is it
confident? are wrong answers wrong *because* of low confidence, or confidently wrong? (Q2) how does the
**yes/no token confidence** correlate with ground truth? (Q3) are **popular items systematically high-confidence and
niche items low-confidence**, and does acting on high confidence create an **echo chamber**? (Q4) can we
**prune / reweight noisy training data by uncertainty**?

No published LLM4Rec work answers these jointly with a *method that changes outcomes* (accuracy, calibration,
tail exposure) rather than only measuring uncertainty. UQRec (WWW'25) measures/decomposes LLM-rec uncertainty and
uses it for prompting; RosePO / C-APO inject *external* reliability signals into preference optimization;
DEALRec prunes data for efficiency, not noise. The gap: a principled account of **what LLM-rec confidence actually
encodes** (relevance vs. item familiarity/popularity vs. label noise) and a method exploiting that account.

## Background
- **Field / sub-area**: IR / recommender systems; LLM-based recommendation; uncertainty & calibration.
- **Inspiration**: Bryan Hooi group (NUS) — confidence elicitation (ICLR'24), ConfTuner (NeurIPS'25),
  proximity-informed calibration ProCal (NeurIPS'23), NeighborAgg (TMLR) — see `docs/sigir/LIT_HOOI_GROUP.md`.
- **Assets already built (this repo, D:\Research\Lumen)**: C-CRP v3 zero-shot pointwise relevance score with
  Qwen3-8B/vLLM on 8 Amazon-2023 domains (10k users, 1+100 popularity-sampled negatives, rating≥4, 3-core).
  NOTE: p_i is a **verbalized** probability parsed from generated JSON (`"relevance_probability"`, T=0.1,
  ~100 output tokens, ~32 prompts/s on a 4090) — **not** a yes/no token probability. Token-level P(Yes) has
  never been measured in this project. First in 6/8 domains vs 8 official
  baselines (ELMRec, IRLLRec, LLM2Rec, LLMEmb, LLM-ESR, ProEx, ProMax, RLMRec); CALIBRA audit code `src/calibra/`.
- **Sibling evidence (D:\llm4rec\uncertainty-llm4rec, "UCalRec")**: generative Qwen3-8B+LoRA, SFT/S-DPO.

## What already did NOT work (must be respected — do not re-propose)
1. Uncertainty as a **ranking multiplier** `r=p(1-U)^η`: η=0 test-best in all domains (inert). Root cause: monotone
   calibration is **rank-preserving within a user's candidate list**.
2. **Popularity-aware post-hoc recalibration** (PopProCal, 2D conf×pop bin shift): no gain over popularity-blind
   recalibration; residual pop-dependence of the conf→acc map is small and **sign-flips** across domains.
3. **Uncertainty-based training-pair pruning/reweighting** in generative DPO (entropy, aleatoric decomposition,
   calibration loss, stable-conflict, cross-model typicality): never beat **random 25% prune**; training-seed
   σ≈1.5pt swamps typical gains → any claim needs ≥3 seeds and gain > 2σ.
4. Cheap LLM-uncertainty **gate for LLM↔CF routing**: oracle router +21% NDCG over LLM, but gate corr 0.07.

## What DID hold (reusable positives)
- AUROC(confidence→correctness) 0.60–0.76 on 8 domains; strong over-confidence; **selective serving** beats
  matched random coverage (+0.036–0.083 NDCG@10 @50%, p<0.001, 5 domains).
- Over-confidence graded head>mid>tail in all domains (mostly base-rate); confidence-ranked serving over-exposes
  head items by +1–5pp vs demand (static exposure bias).
- UCalRec: confident errors = **base-LM genericness/popularity prior** inherited and *amplified* by fine-tuning
  (δ_base −5…−7.5 nats; head-ECE 0.075→0.143 after training); item-vs-item preference is history-content-invariant.
- No-history ablation collapses P(yes) ranking to random (gain is user-conditional, not memorized item prior).

## Known methodological flaw to fix
CALIBRA scored P(yes|"would user like i") against "i is the single held-out next item" (base rate 1/101) → ECE is
not meaningful. A sound calibration study needs labels that match the question (explicit ratings like/dislike,
TALLRec-style), or a properly normalized list-level probability.

## Constraints
- **Compute**: one rented GPU server (SeetaCloud/AutoDL, `ssh lumen-gpu`; GPU type TBD on first login). Old
  server `pony-rec-gpu` is gone → regenerate all data/scores. No local GPU runs.
- **Backbones**: open 7–8B LLMs (Qwen3-8B primary; Llama-3.1-8B / a second family for generality), LoRA.
- **Timeline**: autonomous; target = SIGIR 2027 full paper (ACM `sigconf`, anonymous).
- **Integrity**: no fabricated numbers; ≥3 seeds for trained methods; test split touched once; three-way sync
  (local ↔ GitHub `appleweiping/Lumen-Rec` ↔ server).

## What I'm Looking For
- [x] Improvement on existing project: turn Lumen-Rec's uncertainty line into a SIGIR-grade paper with an
      **original, theoretically-motivated method** (not a kitchen sink), plus a rigorous observation study answering Q1–Q4.

## Domain Knowledge / Hypotheses
- H1: LLM-rec confidence conflates **user-conditional relevance** with **item familiarity/popularity priors**
  from pretraining; confident errors concentrate where the prior is strong.
- H2: Uncertainty only matters where decisions are *not* within-list rank-preserving: training (changes the
  model), cross-user/cross-item decisions (serve/abstain/route, exposure allocation, loop dynamics), fusion with
  other scorers, or explicit-feedback CTR-style prediction (global AUC/LogLoss).
- H3: Naive uncertainty pruning removes hard/tail samples (difficulty ≠ noise); a noise signal must separate
  "confidently contradicted" (noise) from "unsure" (hard) — e.g. via decomposition or stability over training.

## Non-Goals
- Not a pure benchmark/reranking leaderboard paper; not conformal coverage guarantees (sibling TRUCE-Rec owns
  that); not temporal modeling (sibling TGL-Rec).
