# SIGIR 2027 line — pilot log (append-only, newest first)

## 2026-10-01 · M0 infrastructure (server `lumen-gpu`, RTX 4090 D 24 GB, vLLM 0.30.0, torch 2.13, pandas 3.0.6)

**Panels.** Rebuilt from the Amazon-2023 HF mirror (slimmed, Range-resumable) and verified against
`docs/sigir/panel_refs`. Each domain was checked by `source_event_id`, positive item, and the sha1 of the ordered
101 candidates:

| domain | interactions (processed) | eligible users | verify |
|---|---|---|---|
| sports | 5,542,756 | — | 10000/10000 |
| toys | 5,502,054 | 992,296 | 10000/10000 |
| home | 31,541,397 | 5,056,360 | 10000/10000 |
| tools | 9,962,279 | 1,714,369 | 10000/10000 (after the arg fix below) |

- Tools first failed (0/10000). It had been built with `seed=42, max_history_len=10` (from the baseline run
  summaries), not with the `20260506 / 50` used by the other three domains.
- The pandas-3 timestamp-unit worry (review pipeline_ops#0, data_integrity#5) does not apply here: pandas 3.0.6
  reproduces the frozen ms keys exactly.

**Reference ranks.** `docs/sigir/ref_ranks/<d>/<method>.csv.gz` holds C-CRP v3 plus the 8 official baselines.
All 36 files cover the frozen event sets exactly.

**Token-channel smoke test** (20 ML-1M users, 389 candidates × {like, dislike}):
- 778 prompts in 3.8 s, about 200 prompts/s with prefix caching, so the 60 prompts/s pre-registered assumption is
  met.
- Yes+No mass ≥ 0.99998 on every prompt.
- Mean logit: like +3.66 (SD 5.66), dislike −5.39 (SD 3.74).
- This is an infrastructure check only; no endpoint was computed.

**dtype probe** (same 389 like-prompts):
- bf16 puts 57% of logit(Yes) − logit(No) values on a 0.25 grid; fp16 gives a 1/32 grid.
- bf16 vs fp16: Pearson 0.9991, Spearman 0.9989, mean |Δ| 0.17, no non-finite values.
- Decision: fp16 (amendment C0).

**Rated-panel eligibility.** All_Beauty has only 37 eligible users (≥3 likes, ≥3 dislikes, ≥3 history), so it is
replaced by Video_Games in Pilot 3 and by Sports in the full grid (amendment P3/D).

**Pre-GPU code review** (ARIS-style adversarial workflow, 5 lenses × 3 skeptics + critic):
- About 45 confirmed findings, including 3 critical:
  - KuaiRec MNAR labels are identically 0 by construction;
  - the pseudonymizer chains swaps and matches substrings, and its placebo skips history-only brands;
  - rated panels leak re-reviewed items' stars.
- 8 findings were rejected.
- Design changes: `idea-stage/PREREG_AMENDMENT_1.md`, dated before any pilot data.

## 2026-10-01 · CORRECTION to P0-CPU (raised by the Phase-4 triage reviewer, re-verified by executor)
The "+23–40% oracle complementarity" below is **mostly a max-of-two-noisy-rankers artifact, not exploitable
complementarity**. Executor re-check (20 random seeds): per-user max of the LLM with a *uniform-random* ranker
already gains **+0.029 / +0.029 / +0.037 / +0.035** NDCG@10 (sports/toys/home/tools, sd ≈ 0.001) — about half
of the LLMEmb "oracle gain". Reviewer-reported (to be re-verified before any paper use): LLM–LLMEmb per-user
NDCG correlation 0.34–0.56 (co-hit 1.7–2.0× chance); tuned rank interpolation recovers only +0.002..+0.012.
The "popularity-structured" pattern is defined on the HELD-OUT target's popularity (not observable at serving
time). ⇒ The LLM↔CF deferral/fusion cluster (A4, A5, A9, B3) is deprioritised; the winner's-curse lesson is
already published (RouteRec 2607.09908) and is at most a paragraph.

## 2026-10-01 · P0-CPU · LLM↔baseline per-user complementarity (no GPU)
- Script: `src/confrec/analysis_complementarity.py` → `outputs/confrec_pilot/complementarity.json`.
- Data: frozen Lumen 1+100 popularity-sampled panels, 10k users each (sports/toys/home/tools); LLM = C-CRP v3
  (verbalized, Qwen3-8B); 8 official baselines' full rankings (`ranking_eval_records.csv`, local).
- Event keys identical across all 9 methods (asserted).

| domain | LLM NDCG@10 | best complement | its NDCG@10 | per-user oracle | gain | base>LLM users |
|---|---|---|---|---|---|---|
| sports | 0.2329 | LLMEmb | 0.1795 | 0.3095 | +0.0766 (+33%) | 19.1% |
| toys | 0.2708 | LLMEmb | 0.2049 | 0.3353 | +0.0645 (+24%) | 15.5% |
| home | 0.1324 | LLMEmb | 0.0939 | 0.1850 | +0.0526 (+40%) | 12.3% |
| tools | 0.1661 | LLMEmb | 0.1159 | 0.2283 | +0.0622 (+37%) | 14.4% |

**Popularity structure of complementarity:** graph/CF-style baselines (IRLLRec, RLMRec, LLM-ESR) beat the
LLM almost exclusively when the target is a HEAD item (12–20% of head users vs <2% of mid/tail users);
LLMEmb's wins are spread over all groups (head 13–21%, tail 9–14%). LLM NDCG@10 is ~flat across
head/mid/tail. ⇒ headroom for a per-user LLM→baseline deferral is large and consistent (+0.05–0.08 NDCG@10,
4/4 domains), and it is popularity-structured. Open question for the GPU pilot: can LLM confidence (token
P(Yes), margin, entropy, consistency) identify the users to defer (prior cheap gate on verbalized p: corr 0.07)?
