# SIGIR 2027 line — pilot log (append-only, newest first)

## 2026-10-04 · Infrastructure smoke test of the Gate-FT path (Amendment 2 carve-out O)

Synthetic panel only (`scripts/sigir/synth_smoke_panel.py`, invented titles and random labels); no real user, item or rating
data; nothing below enters any gate, table or claim.

- **First attempt failed after 20 s.** `TrainingArguments(warmup_ratio=...)` does not exist in transformers 5.18, the
  server's version. Fixed in `train_lora_yesno.py` (`warmup_steps=0.03`, where a float below 1 is a ratio of the total steps;
  `dtype=` instead of `torch_dtype=`). Without this test Gate-FT would have failed at its scheduled start.
- **Second attempt passed.** Qwen3-8B LoRA (r 16, alpha 32, bf16, gradient checkpointing, micro-batch 8 × accumulation 4,
  last-token loss), 256 synthetic examples (longest prompt 553 tokens): `train_runtime` 66.7 s = **3.84 examples/s**, 86 s wall
  including the model load, **peak GPU memory 19,958 MiB**. Planning figure for Gate-FT: about 24k examples at 3.84/s,
  i.e. ≈ 1.7 GPU-h per seed instead of the 3 assumed (the real prompts decide; BENCH=1 on the real panel replaces this).
- **vLLM 0.30.0 loads and applies the adapter** (`pyes_scorer --lora`): mean |Δlogit| = 3.75 against the base model on
  the same 768 prompts; every row finite, censored = 2 share 0, mean Yes+No mass 0.99998.
- **Trainer loss scale.** `lora_trainer.last_token_trainer` follows the stock `sum / num_items_in_batch` rule under gradient
  accumulation. A plain micro-batch mean (the first version) made the accumulated gradient `grad_accum` times too large:
  in the control run the weights moved 0.17 further. Against the stock Trainer on a tiny LoRA model (SGD, CPU) the
  last-token trainer now agrees to 2e-7 (accumulation 1) and 7e-8 (accumulation 2): `tests/test_confrec_lora_trainer.py`.

## 2026-10-04 · CPU forensics A2–A7 on the Pilot-1 score files (EXPLORATORY: burned users, hypothesis-generating only)

File: `docs/sigir/forensics/pilot1_forensics.json` (sha1 73507b31…, produced by `src.confrec.forensics`, server CPU,
panels ml1m_rated, toys_rated, sports_next_1k; A8 needs the local C-CRP ranking file and is not in this run). Nothing here
is a gate, a selection input or a citable result; each quantity is re-measured as a pre-declared endpoint on fresh users
(Amendment 3).

| quantity (UAUC unless stated) | ML-1M | Toys |
|---|---|---|
| zero-shot like-logit (Pilot 1) | 0.587 | 0.540 |
| item prior π alone (8-donor swap) | 0.603 [0.594, 0.611] | 0.512 [0.501, 0.522] |
| no-history logit alone | 0.546 | 0.488 |
| evidence = like − π | **0.507** [0.497, 0.518] | 0.525 [0.515, 0.536] |
| π split-half r, Spearman–Brown 8-donor | 0.677, 0.807 | 0.673, 0.804 |
| prior-only item mean (ratings before the candidate time) | 0.753 [0.744, 0.761] | 0.693 [0.682, 0.703] |
| temporal biased MF, post-T pairs | 0.804 [0.784, 0.823] | 0.554 [0.513, 0.593] (38% cold items) |
| MF personal residual (MF − b_u − b_i) | 0.677 [0.654, 0.697] | 0.497 [0.458, 0.536] |
| Platt slope of the zero-shot logit; ECE raw → Platt | 0.062; 0.305 → 0.016 | 0.022; 0.364 → 0.024 |
| partial Spearman(π, log popularity \| item mean), item level | 0.361 [0.328, 0.393] | 0.248 |
| MIRROR − correlation-matched ensemble null | −0.0048 [−0.0088, −0.0009] | (CI covers 0) |

Wording flags of Amendment 2 §A (wording only): π reliability ≥ 0.7 on both panels, so the evidence collapse is
interpretable; MF personal-residual UAUC ≤ 0.55 on Toys, so "no personal signal" is stated as a property of the Amazon
(consumed-item, median 3 history events) panels, not on ML-1M (0.68); the ensemble-null rule attributes MIRROR's ML-1M edge
to two-view diversity.

## 2026-10-02 · Pilot 2 (KuaiRec MAR vs logged) result: NULL

**Setup.**
- Qwen3-8B fp16, zero-shot, 300 users × 200 small-matrix videos (60k pairs).
- Target density D = 0.349 (distinct pairs).
- Local copy: `outputs/confrec_pilot/pilot2_kuairec.json`.

**Gate.**
- UAUC_MAR at thr = 2.0 is **0.469 [0.451, 0.486]** (279 users with both classes), below the 0.58 gate.
- Pooled AUC is 0.498.

**Registered consequence (§3.2 NULL; amendment P2).** The LLM cannot read KuaiRec zero-shot:
- B1/A7, B2, A6 and B10 are deprioritized.
- S3 (echo chamber) is answered only through static exposure on the next-item panels.

**Note.** UAUC is significantly *below* 0.5. A plausible mechanism: watch_ratio ≥ 2 favours short, replayed videos, which captions do not reveal. This is unverified and is not used for any decision.

## 2026-10-02 · Gate-failure diagnosis: the task is not hard; the zero-shot yes/no channel is weak

**Method.** `src/confrec/diag_rated_baselines.py` runs on the same Pilot-1 panel pairs. Item signals use only
*other* users' ratings (leave-user-out), so no label leaks.

| UAUC | ML-1M | Toys |
|---|---|---|
| item mean rating (LOO, non-personalized) | **0.760** | **0.712** |
| −item rating variance | 0.662 | 0.687 |
| item popularity | 0.637 | 0.535 |
| LLM like-logit | 0.587 | 0.540 |
| LLM MIRROR | 0.616 | 0.537 |

**Reading.**
- Spearman correlation of the LLM like-logit with item mean rating: 0.15 on ML-1M, 0.07 on Toys.
- The same correlation with popularity is 0.20 on ML-1M and 0.07 on Toys.
- The channel is intact but weak: it misses the item-quality signal.

**Correction (ARIS judge, 2026-10-02).**
- "The task is not hard" overstates the evidence.
- `item_mean_loo` is transductive: it uses other users' *later* ratings, on labels with 3 stars dropped. 0.760/0.712 is therefore an optimistic collaborative-filtering ceiling, not proof that the task is easy.
- The right statement: zero-shot content-only yes/no is weak on a task that collaborative filtering does well.
- Literature anchor (CoLLM, TKDE'25, Table 2, ML-1M): zero-shot ICL UAUC 0.527, MF 0.636, fine-tuned TALLRec 0.682.
- Prior-only (strictly-before) leave-user-out CF references are pending (Amendment 2 artifact checks).
- These are diagnostics. Any claim must be re-validated on fresh users and domains.

## 2026-10-02 · Pilot 1 result: token-channel GATE FAIL; decision = GATE_FAIL_UNINTERPRETABLE

**Setup.**
- Qwen3-8B, fp16, thinking off.
- No censoring on any of 1.13M scored rows; Yes+No mass about 1.
- Throughput about 191 prompts/s.
- Local copies: `outputs/confrec_pilot/pilot1/*.json`.

**Gate.**
- ML-1M raw(like) UAUC = **0.587** against the registered ≥ 0.60, so the gate **FAILS**.
- Sports raw(next) NDCG@10 = 0.209 against ≥ 0.186, which passes.
- Same-event C-CRP reference: 0.231.
- Per triage §3 nothing below the gate is interpretable. The registered remedy is "fix the prompt first".

**Diagnostics** (recorded for transparency; *not* decision inputs; any use must be re-validated on fresh users):

| panel | raw/like | MIRROR | placebo (2-prompt) | evidence (like − π) | PMI no-hist |
|---|---|---|---|---|---|
| ML-1M UAUC (1500 u) | 0.587 | 0.616 | 0.598 | 0.507 | 0.547 |
| Toys UAUC (1500 u) | 0.540 | 0.537 | 0.544 | 0.525 | 0.546 |
| Sports NDCG@10 (1000 ev) | 0.209 (next) / 0.212 (like) | 0.179 | 0.219 | – | – |

- **ΔUAUC(MIRROR − placebo)** by panel:
  - ML-1M: +0.019 [0.013, 0.024].
  - Toys: −0.007 [−0.014, −0.000].
  - Sports: −0.018.
- **Sports next-item:** MIRROR loses about 0.03 NDCG@10 against raw.
- **Overconfidence (ML-1M):**
  - Raw ECE is 0.30.
  - The Platt slope is 0.062, i.e. the logits are about 16× too extreme.
  - |logit| detects errors only weakly: AUROC 0.58.
- **Popularity link (ML-1M):**
  - corr(v, log-pop) = 0.30 at the pair level and 0.42 at the item level.
  - corr(π, log-pop) = 0.43.
  - corr(a, log-pop) is about 0 at the pair level and 0.12 at the item level.
- **Popularity link (Toys):** corr(π, log-pop) = 0.28.
- Within-user acquiescence spread is large (SD_pair_df 2.3 on ML-1M, 3.5 on Toys) but is not popularity-linked at the pair level.

## 2026-10-02 · M0 closed: panels byte-identical, code fixed, Pilot 1 launched

**Panel reproduction.**
- `panel_reference.py verify` now also checks the SHA-256 of the original files recorded in the old C-CRP
  provenance.
- All four rebuilt test panels are byte-identical to the originals for both `ranking_test.jsonl` and
  `candidate_items.csv`. The domains are sports, toys, home and tools, and each covers 10000/10000 events with no
  extra, duplicate or out-of-order rows.
- So the next-item comparison against C-CRP v3 and the 8 official baselines runs on exactly the frozen data.

**Pre-GPU fixes.**
- A fix workflow split the work into 5 disjoint groups. Each group was implemented, reviewed by 2 adversarial
  reviewers, then repaired, followed by an integration step.
- That integration step included a CPU end-to-end dry run of all three pilots and of the real shell scripts in a
  sandbox.
- 148 confrec tests pass.
- Additional real defects found and fixed during review:
  - common-word store names (Kids, Toys & Games) were rewritten in every prompt;
  - pseudonym generation looped forever above 22.7k stores;
  - next-item arms were ranked on different candidate sets when a value was censored;
  - an absent primary endpoint produced a definite NULL;
  - pilot 3 was not gated on pilot 1's token channel.

**Orchestrator sign-off before data** (amendment P1.5b and the P2 addendum):
- `SD_pair_df` gates;
- all-time popularity;
- exact tie-group NDCG;
- NULL wins the NEGATIVE/NULL overlap;
- Pilot-2 POSITIVE additionally needs the lift CI above 1.

**Pilot 1 launched** (`scripts/sigir/run_pilot1_mirror.sh`, Qwen3-8B fp16). ML-1M rated panel: 3,183 eligible users,
1,500 used, 29,365 candidates, like-rate 0.660, 0 duplicates removed.

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
