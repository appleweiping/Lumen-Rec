# Pre-registration amendment 2 (2026-10-02): registered remedy after the Pilot-1 gate failure

**Status: binding.**
- Written after Pilot 1 (GATE_FAIL) and Pilot 2 (NULL), before any further GPU job.
- Source: the ARIS research-review deliberation in `idea-stage/deliberation_2026-10-02/`.
  - Four independent lenses: gate-fix methodologist, paper strategist, method inventor, adversarial skeptic.
  - A judge synthesis, `judge.md`.
- review_independence: same-family (Claude), provisional.

**Freeze rule.** No GPU job of any kind, diagnostics included, starts until the following are written into
`docs/sigir/PILOT_LOG.md` (the only exceptions are the carve-outs N and O below):
- the sha1 of this file;
- the sha1 of the rendered prompt bank;
- the sha1 of the DEV and CONFIRM user-id lists.

## N. Carve-out to the freeze rule (added 2026-10-03, before any GPU job)

**Why.** The freeze exists to prevent forking paths in the *prompt selection*. The full-scale next-item audit cannot
fork that selection, and the one-month schedule cannot leave the GPU idle for the code review cycle. The carve-out is
narrow and adds no degrees of freedom.

**What may run before FREEZE_ACK.**
- Zero-shot token P(Yes) on the Lumen next-item panels with the prompt **frozen at V0** (G0), and nothing else:
  - questions `next` and `like` only (no dislike, paraphrase, swap or no-history arm);
  - TEST: all 10,000 events of each of sports, toys, home, tools;
  - VALID: the first 2,000 events of each domain, used only to fit list-normalized temperatures;
  - Qwen3-8B, fp16, `hist_len` 5, top-50 logprobs, `max_model_len` 3072, 100-user chunks.
- Code: the frozen checkout at commit `189c164` (the code that produced Pilot 1). Later code changes cannot affect
  these scores.
- Outputs: `/root/autodl-tmp/lumen-audit-out/<domain>_{valid2k,test}`.

**Why this cannot fork the prompt selection.**
- No gate-fix decision (G3–G7) reads these outputs.
- No variant, threshold or arm is chosen from them.
- Their analyses (head share, list-normalized calibration, selective serving) are pre-declared in the judge's action 7.

**Quarantine unchanged.** Sports TEST events 1–1000 are reported separately from 1001–10000.

**Preemption.** The scorer is chunked and resumable. These jobs are killed and resumed whenever a gate-fix or diagnosis
GPU stage becomes ready after FREEZE_ACK. Gate-fix and diagnosis stages have priority.

## O. Carve-out for infrastructure smoke tests (added 2026-10-04, before any gate-fix GPU job)

**Why.** Gate-FT (G9) is the most expensive registered stage (about 10 GPU-h) and its training and adapter-scoring path has
only been exercised on CPU with stand-ins. A failure discovered at its scheduled start (out of memory, an adapter that vLLM
does not load, a throughput far from the planning figure) would cost days. The carve-out buys that knowledge early without
touching a single real user, item or rating.

**What may run before FREEZE_ACK.** GPU jobs whose inputs contain **no real user, item or rating data**:
- a synthetic rated panel written by `scripts/sigir/synth_smoke_panel.py` (invented titles, random labels);
- LoRA training of Qwen3-8B for at most 32 optimizer steps on it (`train_lora_yesno`, the G9 recipe);
- `pyes_scorer` on at most 100 synthetic users, with and without the adapter;
- the checks that the adapter is applied (scores differ from the base model), that the readout is finite with Yes+No mass
  ≥ 0.95, and the peak GPU memory and examples/s of training.

**Why this cannot fork anything.** The inputs are synthetic, so no output can inform a variant, a threshold, an arm, a user
set or a claim. Nothing the job writes is read by any gate, table or analysis; its directory is deleted after the check; the
only trace is an infrastructure note in `docs/sigir/PILOT_LOG.md` (peak memory, throughput, return codes).

**Preemption.** As N: killed whenever a gate-fix or diagnosis stage becomes ready after FREEZE_ACK.

## 0. What Pilot 1 established (binding interpretation)

**The registered result stands.**
- GATE_FAIL for the registered prompt V0: ML-1M raw-like UAUC 0.5874 < 0.60.
- The sports next-item component passed (0.2093 ≥ 0.18632).
- Every Pilot-1 number below the gate is **exploratory**. It may generate hypotheses. It may never be cited as evidence
  or used to choose a prompt, an arm or a method.

**Diagnosis: the channel is not broken.**
- No censoring; Yes+No mass ≥ 0.99999.
- The dislike logit is consistently anti-informative, so the model reads question polarity.
- Next-item UAUC is 0.719.
- What is weak is zero-shot efficacy on rated within-user discrimination.

**The 0.60 bar.**
- It was an efficacy bar with no reference point. The only comparable published zero-shot yes/no UAUC on ML-1M is
  0.527 (CoLLM ICL, TKDE'25, Table 2).
- It is **not** amended: the statistic has been seen.

**The registered decision table had a hole, and Pilot 1 landed in it** (positive = negative = null = false). It is
closed conservatively:
- Any configuration that matches no branch is **INDETERMINATE**.
- INDETERMINATE means not POSITIVE and no automatic B9/A9 handoff.
- `pilot1_gate.py` must enumerate every condition combination in a unit test and map each to exactly one outcome.

## G. Registered remedy: one prompt-fix round

**G0 Invariants.**
- Qwen3-8B, fp16, chat template, thinking off, prefill-only.
- Yes/No read by token id from the top-50 logprobs.
- `max_model_len` 4096 for every variant.
- Never in any variant:
  - collaborative-filtering or popularity statistics;
  - the user's mean rating;
  - few-shot examples, thinking, verbalized or digit readout.
- The rated gate constant is unchanged: ML-1M UAUC ≥ 0.60 on the point estimate, computed as `pilot1_gate.py` does.
- The next-item component already passed under V0. Its prompt stays frozen at V0 and is not re-tested.

**G1 Candidate bank** (rated panels; exact strings in `src/confrec/prompting.py` VARIANTS, from
`idea-stage/deliberation_2026-10-02/gatefix_prompt_variants_v1.json`):

| id | change vs V0 | grounding |
|---|---|---|
| V0 | none (registered control; must reproduce pilot prompts_sha1 `12e83c4fcca40db398ac58acbba2a7eb9c770524`) | – |
| V1 | label-aligned question "Will this user rate the candidate item 4 stars or higher (on a 1-5 scale)?" (threshold family) | registered label; Kang et al. 2023 |
| V2 | TALLRec layout: the same 10 events split into liked (rating > 3) / disliked (≤ 3) lists | TALLRec, RecSys'23 |
| V3 | last 20 history events | ReLLa, WWW'24 (zero-shot AUC peaks near K = 15) |
| V4 | history lines annotated with genres (ML-1M) or categories[:80] (Amazon) | Kang et al. 2023 |
| V5 | system message with role and 1–5 scale; user template without the persona line | Kang et al. 2023 |
| V7 | V1 + V3 + V4 + V5 | combined |

- V6 (ReLLa semantic retrieval) is dropped: it needs new embedding code and defeats prefix caching.
- Each variant's dislike and paraphrase strings (the threshold family for V1/V7) are fixed now. They are never scored
  before GATE_PASS.

**G2 Users.**
- **DEV** is the burned Pilot-1 panels: ML-1M (1,500 users) and Toys (1,500 users).
- **CONFIRM** is every fresh eligible ML-1M user:
  - rebuild with seed 0 and n_users ≥ all eligible;
  - assert that the sha1 of the first 1,500 lines (at the pilot `hist_len` = 10 rendering) equals
    `985494c7b44d010bec4b11ec62f35ad274dfe91f`;
  - positions 1,500 onward are the expected 1,683 fresh users, asserted disjoint by user_id from the pilot panel;
  - if the sha1 differs, select by user_id exclusion and log the mismatch.
- Video_Games rated and Sports rated stay untouched until replication.
- Toys fresh users are used only if Toys has ≥ 2,000 eligible users.

**G3 Stage 1: dev scoring, blind to MIRROR.**
- Score only the `like` question of V0–V5 and V7 on both dev panels.
- No dislike, paraphrase, swap or no-history arm of any new variant is scored on dev.

**G4 Eligibility.**
- E1, per panel: censored = 2 rows ≤ 0.5%; overlength = 0; mean Yes+No mass ≥ 0.95.
- E2: Toys dev UAUC(V) ≥ Toys dev UAUC(V0) − 0.010.

**G5 Selection.**
- V\* is the eligible variant with the highest ML-1M dev UAUC(like).
- Variants within 0.005 of the maximum count as tied. Break ties by higher Toys dev UAUC, then by the simplicity order
  V0 < V1 < V5 < V3 < V4 < V2 < V7.
- A **FIX is FOUND** only if V\* ≠ V0 and the one-sided paired user-bootstrap lower bound of UAUC(V\*) − UAUC(V0) on
  ML-1M dev is > 0. The bootstrap uses 2,000 resamples, seed 0, at level 1 − 0.05/6.
- All 7 × 2 dev UAUCs are published, in an appendix table on prompt sensitivity.

**G6 Stage 2: one confirmatory gate** (only if a fix is found).
- Score V\* like on the CONFIRM ML-1M users. V0 like is also scored there, as reported context only.
- **GATE_PASS** if and only if UAUC(V\*) ≥ 0.60 on the point estimate and E1 holds. Report the 95% user-bootstrap CI.

**G7 Stage 3, only on GATE_PASS.**
- Under V\*, score like, dislike, like_para, the swap prior (K = 8) and the no-history prior on:
  - CONFIRM ML-1M;
  - the second rated domain: fresh Toys if ≥ 500 fresh users are eligible, otherwise Video_Games.
- MIRROR's next-item no-loss check runs under V0 on the first 1,000 events of the **sports VALID** panel. Sports TEST
  events 1–1000 are quarantined.
- Apply the §3.1 table with P1 and the closed hole.
- Any MIRROR or two-prompt claim must beat both the registered paraphrase placebo and a **correlation-matched ensemble
  null**: the within-user z-sum of like and −dislike, compared with like + like_para.

**G8 Contingency, written now.**
- F0 (no fix found on dev): record GATE_FAIL_AFTER_REMEDY(dev) and skip Stage 2.
- F1 (CONFIRM UAUC < 0.60, or E1 fails): record GATE_FAIL_AFTER_REMEDY(confirm).
- On F0 or F1:
  - no second prompt round, no threshold change, no backbone rescue (Llama-3.1-8B is replication only);
  - zero-shot rated results become the descriptive Finding 1, labelled "below the registered bar", re-measured on
    CONFIRM users with pre-declared descriptive endpoints;
  - Pilot 3 (pseudonym knockout) moves to the fine-tuned regime;
  - next-item zero-shot endpoints remain interpretable.

**G9 Gate-FT** (registered now; runs in every branch).
- TALLRec-style LoRA yes/no SFT (`src/confrec/train_lora_yesno.py`):
  - r = 16, alpha = 32, lr 1e-4, 1 epoch, 3 seeds;
  - prompt V\*, or V0 under F0.
- Training uses only events of ML-1M users outside the CONFIRM set, before the global p80 timestamp (amendment 1 §D).
- Evaluation is on the CONFIRM users' test-period candidates. Zero-shot is re-scored on the same subset for a paired
  regime contrast.
- **PASS** if and only if the mean UAUC over seeds is ≥ 0.65. This is anchored between published MF (0.636) and
  fine-tuned TALLRec (0.682), both on harder labels.
- Report every seed, with the CPU references (prior-only item mean, MF) as context.
- Gate-FT FAIL closes the rated-panel method line.

**G9 operationalization (added 2026-10-04, before FREEZE; code `src/confrec/gateft_data.py`, `gateft_eval.py`,
`scripts/sigir/run_gateft.sh`).**
- T is the 0.8-quantile (numpy linear) of all candidate timestamps of the ML-1M rated panel of the DEV and the CONFIRM
  users together, computed once and recorded in `gateft_split.json`.
- TRAIN = the DEV users' candidates with ts < T (histories untouched). No CONFIRM user and no event at or after T enters
  training (asserted).
- The adapters score the whole CONFIRM panel; the primary endpoint restricts the rows to ts ≥ T by the panel's
  `candidate_timestamps`. The all-CONFIRM UAUC and the paired zero-shot contrast on identical rows are context.
- Seeds are exactly 0, 1, 2. PASS iff the mean over the 3 seeds of UAUC(post-T) ≥ 0.65 on the point estimate AND every run
  passes the integrity checks (censored = 2 share ≤ 0.5%, no overlength prompt, mean Yes+No mass ≥ 0.95). A different
  number of seeds or a failed integrity check is INCOMPLETE (neither PASS nor FAIL): rerun the broken run; no seed is ever
  replaced.
- Training settings: LoRA r = 16, alpha = 32, dropout 0.05 on q/k/v/o, lr 1e-4, cosine schedule with 3% warmup, 1 epoch,
  bf16, per-device batch 8 × gradient accumulation 4, max prompt length 1,024 tokens (longer examples are skipped and
  counted, never truncated). The loss is the cross-entropy of the answer token, computed from the last two positions'
  logits (`logits_to_keep=2`), which equals the shifted full-sequence loss with the prompt masked (unit-tested).
- Power is recorded (CONFIRM users with both classes among their post-T candidates); fewer than 150 only prints a
  warning. The quantile and the 0.65 constant are not changed.

**G7 stage-3 gate source (clarification).** In stage 3 the token-channel gate is the recorded one (`pilot1_gate.py
--stage3_gate/--pilot1_decision`): the G6 `gate.json` must be GATE_PASS and the Pilot-1 decision must record the sports
next-item component as passed (G0: not re-tested); re-measured CONFIRM UAUC and sports VALID NDCG@10 are context.

**Section N analysis plan.** `docs/sigir/NEXTITEM_AUDIT_SPEC.md` (sections A–F) is the registered analysis plan for the
next-item audit: its endpoints, the S3 admission rule and the sports 1–1000 / 1001–10000 split are binding.

## D. Diagnosis battery (interpretive only; cannot change G0–G9)

All on burned users, Qwen3-8B fp16:

| test | what it does | how it reads |
|---|---|---|
| T0 | No-user item-quality probe ("Is this movie widely considered good?" P(Yes)) on the 2,909 ML-1M items; Spearman with the prior-only item mean | ≥ 0.5 together with a like-logit ρ of about 0.15: the knowledge exists but is not used. < 0.3: knowledge-limited |
| T1 | Positive control: append "(This user later rated this item r/5.)" | UAUC < 0.90: readout broken; stop and debug |
| T2 | Append "Average rating by other users before this date: m/5 (n ratings)" (prior-only leave-user-out mean); also on 500 burned Toys users. Transmission = (UAUC_T2 − 0.5)/(UAUC_itemmean_prior − 0.5) | ≥ 0.8: knowledge-limited. < 0.5: readout-limited |
| T3 | Digit readout, E[r] over the tokens 1–5 | ≥ yes/no + 0.03: format bottleneck, reported as a secondary channel, never a gate rescue |
| Star permutation | Permute the "(rated r/5)" suffixes among the same history items (K = 2); τ_P = L − mean L_perm | \|ΔUAUC\| < 0.005 and within-user SD(τ_P) < 0.10: the model ignores the user's ratings. A hypothesis to re-test on fresh users |
| T4 | Llama-3.1-8B-Instruct, V0 like | ≥ 0.62 is exploratory only |

## A. CPU artifact checks (exploratory; recorded before any paper claim)

1. Prefix/fresh-user check:
   - the sha1 above;
   - eligible counts for ML-1M, Toys, Video_Games and Sports rated;
   - Toys prefix sha1 `69252b4806bb4ffe05101967ea2a1d94ada57dee`.
2. Prior quantities:
   - UAUC of π(i) alone and of L_nohist alone;
   - split-half Spearman reliability of π (donors 1–4 vs 5–8);
   - 4-donor evidence UAUC with a Spearman–Brown projection.
3. Non-transductive CF references on the same pairs:
   - prior-only leave-user-out item mean (ratings strictly before each candidate's timestamp);
   - temporal biased MF;
   - MF personal residual (MF − b_u − b_i);
   - global AUC of the history-star mean.
4. MIRROR against the correlation-matched ensemble null (per user).
5. Calibration decomposition:
   - ECE after intercept-only and after temperature-only recalibration;
   - user-centred ECE.
6. Error targets redefined: errors after Platt scaling, and per-user top-k errors (k = number of likes).
7. Quality-partialled popularity: partial Spearman of π and v against log-pop, controlling for the prior-only item
   mean. ML-1M also controls for release year; Toys also for description length and has_store.
8. Sports next-item: expected top-10 head share under random ranking per event, and C-CRP's top-10 head share.
9. Throughput reconciliation, measured per panel: rated ML-1M 87.1, Toys 35.0, sports 174.8 prompts/s.

Decision use of these checks (wording only, never gates):
- π split-half reliability < 0.7: the evidence collapse is labelled uninterpretable.
- MF personal-residual UAUC ≤ 0.55: "no personal signal" is stated as a property of consumed-item panels.
- Ensemble-null residual CI covers 0 on ML-1M: MIRROR's ML-1M edge is attributed to two-view diversity.

## F. Forking-path rules (binding)

- **Immutable constants:**
  - zero-shot rated UAUC ≥ 0.60;
  - next-item NDCG@10 ≥ 0.18632;
  - Gate-FT ≥ 0.65.
  - No post-hoc re-anchoring, no CI-based reinterpretation, no "near miss" rule.
- **One round only:** 7 variants, one selection, one confirmatory gate. A CONFIRM failure is final for the zero-shot
  gate.
- **Unit roles are fixed:**
  - DEV = burned Pilot-1 users; CONFIRM = fresh ML-1M users, disjoint by user_id;
  - Video_Games and Sports rated are untouched until replication;
  - sports next-item TEST events 1–1000 are quarantined and always reported separately from 1001–10000;
  - next-item prompts stay frozen at V0.
- **Selection is MIRROR-blind, and no arm is promoted.** like_para or MIRROR may not become the base arm because they
  cleared 0.60 on burned users.
- **Every rated-panel table carries non-LLM references** (prior-only item mean, popularity, MF) as context. They are
  never gates.
- **Claim admission.** A finding enters the abstract only if:
  - it is confirmed on fresh users;
  - it replicates in sign on at least one fresh Amazon domain;
  - it replicates on the second backbone.
  - Until then, wording stays direction-free.
- **Fine-tuned seeds:**
  - 3 for the main grid, 5 for pruning; every seed reported;
  - a gain must exceed 2 σ_seed with all seeds agreeing in sign;
  - hyperparameters and kill dates are fixed before training.

## P. Paper framing (strategy decision recorded with this amendment)

**MIRROR is retired as the flagship but not killed.** It stays a registered confidence channel and is evaluated only
after GATE_PASS.

**The paper becomes a pre-registered audit comparing the zero-shot and fine-tuned regimes.** Working title: *What does
a yes/no LLM recommender's confidence know? Item priors vs personal evidence, and when uncertainty is actionable.*

Contributions:
- **C1 Protocol.**
  - Rated panels (ML-1M, Toys, Video_Games, Sports; all fresh eligible users).
  - Full-scale next-item panels: 4 domains × 10k users × 101 candidates.
  - Two backbones; zero-shot and LoRA × 3 seeds.
  - Non-LLM references; user-cluster CIs; Holm correction within each RQ; confirmation only on untouched users.
- **C2 Decomposition.**
  - logit = user offset + item prior π + personal evidence.
  - Disattenuated personal share.
  - Information gain over the item-mean prior.
  - Star-permutation contrast.
  - Two-axis popularity analysis, partialled on quality.
  - Pseudonym knockout, in the fine-tuned regime.
- **C3 Actionability.**
  - A lemma: per-user strictly increasing maps cannot change UAUC or within-user NDCG (non-strict maps can, through ties).
  - Each route that can change outcomes is tested against its proper control:
    - item-dependent corrections vs the placebo and the ensemble null;
    - cross-user selective serving and exposure at 10k scale vs random and vs the baselines' own confidence;
    - S6 pruning vs matched-random pruning over 5 seeds.
- **C4 A one-table guide answering S1–S6.**
  - S3 is answered statically on the next-item panels.
  - The dynamic loop is off because Pilot 2 was NULL.

**One nested method slot** (default: prior-offset LoRA, logit = f_θ(u,i) + b·q̂(i) with q̂ under stop-gradient):
- Hard kill date: **2026-11-30**.
- It is killed if its gain over post-hoc stacking is < +0.01, or its CI covers 0, on ≥ 2 of 4 datasets.

**Context, not headline.** C-CRP and the 8 baselines are context and testbed:
- C-CRP is cited in the third person.
- Every claim is scoped to same-candidate evaluation.

**Banned claims:**
- the two-prompt estimator as new;
- "uncertainty improves ranking";
- "6/8 SOTA";
- any number not confirmed on fresh users.
