# SIGIR 2027 line — pilot log (append-only, newest first)

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
