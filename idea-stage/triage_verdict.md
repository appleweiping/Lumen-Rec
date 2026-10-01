# ARIS Phase-4 Triage Verdict: Lumen-Rec → SIGIR 2027

Reviewer: fresh Claude Opus 5.5, acting as an adversarial senior SIGIR/RecSys reviewer. Date: 2026-10-01.
review_independence = same-family: the generators (opus-A, opus-B) and this reviewer are the same model family,
so any blind spot we share goes uncorrected. acceptance_status = provisional.

Inputs read in full:
- `triage_bundle.md`, `brainstorm_A.md`, `brainstorm_B.md`, `brainstorm_bundle.md`, `RESEARCH_BRIEF.md`
- `docs/sigir/LIT_HOOI_GROUP.md`, `docs/sigir/LIT_RECENT_UNC_LLM4REC.md`
- `docs/sigir/PILOT_LOG.md`, plus `src/confrec/analysis_complementarity.py` and its output
  `outputs/confrec_pilot/complementarity.json`

Status of numbers:
- Every number tagged **[measured]** below I recomputed on CPU in this session, from the local
  `ranking_eval_records.csv` files. No file other than this verdict was written.
- Every other number is either quoted from the inputs or is a pre-registered threshold.

---

## 0. Evidence that changes the triage (read this first)

### 0.1 The "+23–40% oracle complementarity" is mostly an artifact of single-target evaluation

The executor's `so_what` annotation treats the per-user LLM↔LLMEmb oracle (+0.053..+0.077 NDCG@10) as
exploitable headroom. On the project's own data it is not.

Every user has exactly one held-out target, so the per-user oracle is the maximum of two noisy single-item
outcomes. Four checks [measured]:

| domain | oracle gain vs a **uniform-random** ranker (analytic) | ELMRec (NDCG ≈ 0.046–0.049, i.e. ≈ random 0.045) oracle gain | LLMEmb actual oracle gain | LLMEmb oracle under **independence** (ranks shuffled across users) | corr(per-user NDCG, LLM vs LLMEmb) | P(both hit@10) ÷ P(LLM hit)·P(LLMEmb hit) | **cross-fitted** LLM+LLMEmb rank interpolation (weight tuned on the other half of users) |
|---|---|---|---|---|---|---|---|
| sports | +0.0303 | +0.0321 | +0.0766 | +0.1247 | 0.42 | 1.66× | 0.2329 → 0.2452 (**+0.0123**) |
| toys | +0.0292 | +0.0308 | +0.0645 | +0.1387 | 0.56 | 1.80× | 0.2708 → 0.2772 (**+0.0064**) |
| home | +0.0364 | +0.0370 | +0.0526 | +0.0771 | 0.35 | 2.00× | 0.1324 → 0.1360 (**+0.0036**) |
| tools | +0.0346 | +0.0349 | +0.0622 | +0.0905 | 0.34 | 1.79× | 0.1661 → 0.1680 (**+0.0019**) |

Reciprocal-rank fusion (k = 60) gave 0.2415 / 0.2708 / 0.1325 / 0.1607, i.e. −0.005 to +0.009.

What the table shows:
1. **Roughly half of the headline oracle comes free.** A ranker with zero information already yields
   +0.03. ELMRec is at random level and reproduces that +0.03 almost exactly.
2. **LLM and LLMEmb are redundant, not complementary.**
   - Their hits co-occur 1.7–2.0× more often than independence predicts.
   - Their observed oracle gain is *below* what two independent rankers with the same marginals would give.
   - This is inconsistent with conditional independence given relevance, which is the premise of A4's
     product-of-experts fusion.
3. **Achievable item-level fusion recovers 3–16% of the oracle**, i.e. +0.002..+0.012 NDCG@10.
4. **The "popularity structure" sits on the wrong side.** It is defined on the *target's* popularity
   (`positive_popularity_group`), which a serving-time gate cannot observe. Also, "best complement" was picked
   post hoc among 8 baselines on test data.

Consequences:
- B3's winner's-curse diagnosis is essentially **confirmed at zero GPU cost**. RouteRec (2607.09908) already
  published "oracle headroom is not enough" under strict out-of-fold selection.
- The LLM↔CF serving cluster (C4: A4, A5, A9-as-routing, B3) loses its motivating number and is demoted.
- The corrected numbers are themselves a finding for S1/S2: "when the LLM is wrong, is anyone else right?
  Rarely, and unpredictably."

### 0.2 Novelty checks run in this session (details in §2; sources at the end)

- **A3's unresolved prior-work paper is PerRecBench** ("Can LLMs Understand Preferences in Personalized
  Recommendation?", 2501.13391). It uses infini-gram item-name counts and correlates them with *accuracy*.
  LRWorld (2512.17389) does **not** do this; I checked the full HTML. This partly pre-empts A3.
- **Incumbent Advantage (2606.17443)** is zero-shot, non-personalized, skincare-only, with 3 commercial LLMs
  and 1 real brand vs 9 fictional ones. A2 remains differentiable: a consistent pseudonym map, personalized
  histories, confidence/calibration, and a training regularizer.
- **Yes–no bias (2607.05552)** uses crossed symmetrization, which is effectively MIRROR's two-prompt design,
  and finds the bias is mostly global (answer order plus a lexical pull toward "no"). A global bias is
  rank-inert. MIRROR's scoring rule ½[p(like) + 1 − p(dislike)] is also the contrast-consistent inference rule
  of CCS (Burns et al., ICLR'23).
  - **What A1 may still claim as new:** item-conditional, popularity-linked acquiescence in personalized
    recommendation.
  - **What A1 may not claim:** the two-prompt estimator itself.
- **User-marginalized prior (A9, B9)** is per-item Batch Calibration (Zhou et al., ICLR'24) / PMI-DC, and the
  LLM analogue of MACR-style counterfactual popularity debiasing. Only its *use* (cross-user decisions) is new.
- **The B2 theory restates two known results:**
  - the naive-loss minimizer exposure × preference is standard MNAR/PU (Saito et al. WSDM'20; Schnabel et
    al. ICML'16);
  - the "self-confirming fixed point" is performative stability (Perdomo et al. ICML'20).

  The LLM-specific "prior erosion" prediction is the only new part.
- **Routing:** besides RouteRec, there is a serving-time generative/collaborative profile gate (2609.39043)
  and ReliGRec's user-risk prompt routing (2609.16560). The space is crowded.
- **Thinking-mode calibration:** QA results exist (Yoon et al. 2505.14489; "Don't Think Twice" 2508.15050;
  CDUR 2606.11211). I found nothing for recommendation.

### 0.3 A hidden shared cost

The frozen Lumen panels are next-item 1+100 with a 1/101 base rate. They support NDCG, not per-candidate ECE.

Ideas that need explicit like/dislike labels (A1, A2, A10, A11, B7, B11, and the calibration half of A3) all
require **new rated-candidate panels**. Amazon-2023 ratings are heavily skewed toward 5 stars, so few users
have ≥3 ratings of ≤2 stars.
- ML-1M (and Steam's recommend flag) are the safe domains for this protocol.
- Amazon panels need pooling or a minimum-dislike filter.

The old server is gone, so panel construction is on the critical path for any rated-label spine.

---

## 1. Ranked table (all 22 candidates; cluster tag in brackets)

Ranking criterion: expected information × upside within a ≤2 GPU-h pilot, i.e. which results matter whichever
way they come out. Risk is my own rating and may differ from the generator's.

| Rank | ID [cluster] | One-line verdict | Risk | Expected info value |
|---|---|---|---|---|
| 1 | **A1** MIRROR [singleton; C5-adjacent] | Best flagship. It is the only one-mechanism method that legitimately changes within-user order, and its pilot is cheap, placebo-controlled and decisive. It has two named identification risks (valence; non-commitment) and a mis-specified kill gate (fixed in §3). | MED-HIGH | HIGH |
| 2a | **B1** confidently wrong vs confidently unlabelled [C2] | The sharpest reframe in the pool and the validity check every S2/S4 claim needs. One pilot gates 5 KuaiRec-dependent ideas. Too few datasets to host a method. | MED | HIGH |
| 2b | **A7** MNAR calibration illusion [C2] | Same pilot as B1; adds the "MNAR recalibration can't fix item-dependent shift" prediction. Its fix (DR with 2% MAR data) is classical (Kweon AAAI'22). | MED | HIGH (jointly with B1) |
| 3 | **A2** pseudonym knockout [C5] | The only causal test of S5 in the pool, and the experiment that interprets A1's a(i). A strong diagnostic but a weak method (name-dropout will likely cost head accuracy). | MED-HIGH | HIGH |
| 4 | **B9** counterfactual personalisation confidence [C4/C5 bridge] | The cleanest definition of "how much of this yes comes from *this* user"; the fallback flagship if A1's gate fails. The estimator is batch calibration / PMI. | MED | MED-HIGH |
| 5a | **A11** thinking and confident errors [C1] | Free knob, informative either way, nothing published for recommendation. One RQ, not a paper. | LOW-MED | MED |
| 5b | **B11** deliberation manufactures confidence [C1] | Same as A11, with a sharper tail-item prediction. Join the two into one RQ. | LOW-MED | MED |
| 6 | **A9** evidence-gated serving [C4] | The same latent decomposition as B9, with a cheaper user-independent prior. Selective serving is textbook; only the gated quantity is new. | MED | MED-HIGH (measured inside pilot 1) |
| 7 | **B2** self-confirming calibration [C3] | Highest ceiling in the pool, but the theory restates MNAR/PU and performative stability, the full study is 25–70 GPU-h, and it is gated on KuaiRec readability. Named high-upside slot, conditional on pilot 2. | HIGH | MED now; HIGH if pilot 2 is positive |
| 8a | **A10** BUCE [C7] | The lemma "within-user rules are invariant to per-user monotone maps; cross-user regret ≤ BUCE + ECE" is the paper's formal bridge from N1. It is nearly trivial, and PerK/Kweon own user-wise calibration. | LOW-MED | MED |
| 8b | **B7** user leniency [C7] | A one-paragraph corollary of A10 (fix is a per-user logit offset); section-sized, as the generator itself concedes. | LOW | LOW-MED |
| 9 | **B3** ask the fallback / winner's curse [C4] | Its diagnosis is now answered (§0.1) and RouteRec already said it. The remaining pre-hoc deferral method has ≤ ~+0.01 NDCG headroom. Keep it as a paragraph. | MED | LOW-MED (already answered) |
| 10 | **A3** exposure counts (OLMo + infini-gram) [C5] | Partly owned (PerRecBench; Mozafari SIGIR'26 short). Amazon keyword-string titles make counts noisy. The "sign flip" it explains was measured on mismatched labels. | MED-HIGH | MED |
| 11 | **B6** selective serving silences niche users | CPU-only and likely true, but expected from Jones et al. ICLR'21 and mainstream-bias work. Good S5 (user-side) appendix. | LOW-MED | MED |
| 12 | **A5** confidence = redundancy [C4] | My CPU check already supports the outcome-level redundancy. The confidence-decile figure can be run now on verbalized p, no pilot slot needed. | LOW | MED (cheap) |
| 13 | **A6** ground-truth echo chamber on KuaiRec [C3] | Fixes the simulator critique, but a 1,411 × 3,327 matrix with static preferences can only show model-side homogenization. Gated on pilot 2. | MED-HIGH | MED |
| 14 | **A8** natural-noise oracle for pruning [C6] | The cheap half (detection AUROC against rating–text contradictions) is the right S6 appendix. The full 90-run study belongs to a companion paper. | MED | MED (cheap half) |
| 15 | **B4** prune the familiar [C6] | Bold and principled, but every targeted pruning signal so far has tied random (N3, GORACS, MiniRec), and 3 seeds × 2 arms cannot resolve effects under 2σ. The pilot is over budget. | HIGH | LOW-MED |
| 16 | **B5** mediation by concept erasure [C5] | Popularity steering is owned (SPREE, PopSteer). The title-likelihood familiarity probe is confounded with length and genericness. A2 does the same causal job at the input more cleanly. | MED-HIGH | MED-LOW |
| 17 | **A4** calibrated log-odds pooling [C4] | Its conditional-independence premise is contradicted by the project's own data (per-user NDCG corr 0.34–0.56). Tuned interpolation already captures the little headroom there is. | MED-HIGH | LOW-MED |
| 18 | **B8** decision-weighted proper scoring | Principled, but gains are likely inside σ_seed. Setting the band at the validation threshold is circular, and it answers only S4. | MED | LOW |
| 19 | **B10** abstention feedback loop [C3] | A novel inversion, but it depends on B2's loop infrastructure, and KuaiRec's niche/mainstream split is weak. | HIGH | LOW |

(20 table rows for 22 IDs: A11/B11, B1/A7 and A10/B7 are near-duplicate pairs judged jointly and kept visible
as a/b rows.)

**Is the top set all low-risk?** No.
- Ranks 1–3 are MED-HIGH, MED and MED-HIGH.
- The high-upside idea that most deserves a *conditional* slot beyond them is **B2**; §3.4 gives the result
  that would convince me.

---

## 2. Per-candidate both-ways analysis (concise)

### A1: MIRROR (rank 1)

**FOR.**
- One mechanism, restatable in a breath: "LLM recommenders say yes to items they recognize; ask both ways and
  take the difference."
- An item-dependent correction legitimately changes within-user order, which escapes N1 without a gate or
  extra module.
- It comes with an identifiability statement (under the additive model, the difference is unbiased for s) and
  a proper-scoring training variant (mirror tuning with BCE).
- Zero-shot scoring works on any dataset and backbone, so a "consistent gains on 4 datasets × 2 backbones"
  table is feasible.
- It can be dropped onto Lumen's 4 next-item panels against the 8 official baselines, reusing the strongest
  existing asset.
- The paraphrase-ensemble placebo makes the pilot decisive.
- A citable named phenomenon (the acquiescence gap) if it exists.

**STRONGEST OBJECTION (identification).**
1. *Valence survives the difference.* Familiarity plausibly acts through valence ("well-known brand ⇒ the
   user will like it"): like = a + v(i) + s and dislike = a − v(i) − s, so v(i) passes straight through.
   MIRROR removes yes-saying, not the familiarity prior.
2. *"Dislike" is not "not like".* For obscure items an 8B model may answer No to both questions
   (non-commitment). That produces a popularity-correlated a(i) with no yes-saying at all, so the "it knows
   the item" reading is confounded.
3. *Negation blindness.* "Language models are not naysayers" applies at 8B: dislike logits may track like
   logits, leaving the difference noisy.
4. *Prior art.* The estimator is crossed symmetrization (2607.05552, which finds the bias mostly global) and
   the CCS inference rule.

**MOST LIKELY FAILURE.**
- The item-dependent spread of a(i) is small. MIRROR then reduces to a global recalibration, which is
  rank-inert (N1), and UAUC versus the paraphrase placebo falls inside the CI.
- **The generator's own kill gate is mis-specified.** "Mean |a| < 0.1 logit" is passed by a *global* yes-bias
  that cannot change ranking. Pre-register instead on the across-item SD of the within-user-centred a(i) and
  on corr(a, log-popularity); see §3.1.

**PRIOR WORK.** A real but differentiable problem. The novelty is the item-conditional, personalized finding
and its ranking consequence, not the two-prompt estimator. Cite CCS, crossed symmetrization, contextual and
batch calibration, and RankSteer (2602.03422, pointwise LLM-ranker calibration in IR).

**Info:** HIGH, because each outcome redirects the paper: a ≈ 0 means familiarity enters through valence or
evidence, and the flagship passes to B9/A9.

### A2: pseudonym knockout (rank 3)

**FOR.**
- The only intervention in the pool that is causal on the *input*.
- A consistent pseudonym map keeps in-prompt collaborative structure (brand loyalty, series) and removes only
  pretraining recognition.
- It explains N2's null ("conditioned on the wrong variable").
- It interprets A1: if a(i)'s popularity slope vanishes under pseudonyms, a(i) is familiarity; if not, it is
  in-prompt evidence or non-commitment.
- A reviewer remembers this experiment.

**OBJECTION.**
- The brand is evidence (a quality signal), so removing it removes information. "Of course confidence
  dropped."
- Head items may simply have more recognizable brands, a composition effect.
- Amazon titles mix brand, model number and keywords.

**FAILURE MODE.**
- A uniform Δlogit across popularity deciles: a perturbation effect, not familiarity.
- This needs a **real-brand-swap placebo** (map to another real brand of matched popularity); the shared
  substrate's "placebo for every intervention" rule already requires it.
- Name-dropout LoRA likely costs head accuracy (KnowSA: knowledge helps).

**PRIOR WORK.** Incumbent Advantage (2606.17443) shows real > fictional brands, but zero-shot, non-personalized,
commercial LLMs, choice only. Differentiable.

**Info:** HIGH as a diagnostic, LOW as a method. Use it as a diagnostic RQ, not the method.

### A3: exposure counts (rank 10)

**FOR.** Pretraining exposure is the principled latent variable (KnowSA, Mozafari). A PIECE-style EICE metric
has the "theorem shows ECE hides it" shape that Hooi-group papers use.

**OBJECTION.**
- PerRecBench already correlates infini-gram item-name counts with LLM-rec accuracy; I verified this.
- Amazon titles are long keyword strings, so n-gram counts mostly measure genericness.
- OLMo-2-7B is a weak recommender, so findings may not transfer to Qwen or Llama.
- The CALIBRA "sign flip" it sets out to explain was measured on 1/101 next-item labels, so it may be
  explaining an artifact.

**FAILURE MODE.** Partial correlations ≈ 0 after accounting for count noise.

**Info:** MED. At most one RQ paragraph inside S5.

### A4: calibrated log-odds pooling (rank 17)

**FOR.** A clean theory ("calibration is the exchange rate between recommenders") that is not rank-inert and
reuses the 8-baseline rankings.

**OBJECTION (now measured).**
- The conditional-independence premise is contradicted on the project's data (§0.1).
- Cross-fitted rank interpolation already captures +0.002..+0.012, so tuned interpolation, the baseline A4
  itself fears, absorbs the headroom.
- Official baselines expose rankings; calibrated per-candidate scores may not be available for all 8.

**FAILURE MODE.** PoE within ±0.005 of tuned interpolation.

**Info:** LOW-MED. Keep only as a theory remark.

### A5: confidence measures redundancy (rank 12)

**FOR.** Cheap, and both outcomes are informative. Outcome-level redundancy is already confirmed (§0.1).

**OBJECTION.** RouteRec has already made the policy point. It is a figure, not a paper.

**Action.** Run the LLM-confidence-decile × {LLM hit, LLMEmb hit} table on verbalized p on CPU now. No GPU slot.

**Info:** MED.

### A6: ground-truth echo chamber (rank 13)

**FOR.** Real recorded responses rather than a simulated user, which fixes the main critique of EchoTrace and
RecLoop. It is S3's primary candidate.

**OBJECTION.**
- Static preferences on a tiny dense matrix can only show *model-side* homogenization.
- Watch-ratio labels come from forced exposure.
- Short Chinese captions may be unreadable zero-shot.
- 30–45 LoRA runs.

**FAILURE MODE.** No drift within 4–5 rounds: an ambiguous null.

**Info:** MED, gated on pilot 2. The zero-shot 2-round in-context protocol is the right S3 appendix if KuaiRec
is readable.

### A7 and B1: exposure-confounded calibration (rank 2)

**FOR.**
- INV-1 is the most important validity question for the whole observation study. Some confidently-wrong
  predictions may be confidently right about pairs that were never labelled.
- KuaiRec gives the fully observed answer.
- "Fine-tuning re-learns the logging policy" is an LLM-specific, falsifiable prediction.
- One pilot decides the fate of 5 ideas (B1, A7, B2, A6, B10).

**OBJECTION.**
- Zero-shot P(Yes) on short Chinese captions may be near chance.
- KuaiRec's forced exposure changes behaviour.
- Coat is tiny and text-poor.
- MNAR-calibration ideas are classical (Kweon AAAI'22; doubly-calibrated MNAR 2403.00817).
- At most 2–3 datasets, so it cannot carry the method.

**FAILURE MODE.**
- AUC ≈ 0.5, so the result is uninformative and a LoRA arm is required.
- Big/small matrix pair overlap must be verified.

**Info:** HIGH.

### A8: natural-noise oracle (rank 14)

**FOR.** It fixes the evaluation problem behind S6: real noise ground truth instead of injected noise. Its
detection-AUROC half is cheap and directly tests H3 ("difficulty ≠ noise").

**OBJECTION.**
- Rating–text contradiction includes legitimate cases (sarcasm, seller vs product ratings).
- LLM noise judges are crowded (LLM4DSR, LLMHD, LLMHNI).
- The full grid is 90 LoRA runs.

**FAILURE MODE.** Judge precision below 70%.

**Info:** MED for the cheap half, which is the S6 appendix.

### A9 and B9: prior vs personal-evidence decomposition (ranks 4 and 6)

**FOR.**
- The most general decomposition of a yes into item prior (acquiescence, valence and popularity together)
  plus this user's evidence.
- It answers S2 directly ("are confident errors prior-dominated?").
- Used for *cross-user* decisions it is outside N1.
- A9's prior is user-independent, so it can be cached and costs one extra scoring pass.
- B9's prediction that "LoRA collapses CPC" (UCalRec) would be striking either way.

**OBJECTION.**
- Prior art for the estimator: per-item batch calibration, PMI-DC, MACR.
- Selective serving is textbook; only the gated quantity is new.
- Subtracting the prior removes *legitimate* item appeal, so as a ranking score it can hurt next-item
  prediction. The sibling project found exactly this domain-conditionality.
- K = 4–16 swapped users is a noisy estimate for tail items.

**FAILURE MODE.** AURC(evidence) ≈ AURC(confidence), because both mostly rank users by how easy their candidate
list is.

**Info:** MED-HIGH, measured for free inside pilot 1 as A1's competitor estimator. A1's own composition already
names it as such.

### A10 and B7: cross-user calibration (rank 8)

**FOR.**
- The lemma formalizes N1 and its exact complement, which is the paper's needed bridge from "uncertainty is
  rank-inert" to "where it matters".
- Exact UAUC invariance under the per-user offset is a clean demonstration.
- The causal history-length test is cheap.

**OBJECTION.**
- The lemma is near-trivial, and user-wise calibration is owned (PerK, Kweon).
- Global AUC is not a metric SIGIR top-K reviewers value.
- Lumen has no budgeted cross-user task to show utility.

**Info:** MED. A half-page lemma plus one table, not a spine.

### A11 and B11: thinking-mode calibration (rank 5)

**FOR.** A zero-cost switch in Qwen3. The QA literature predicts both directions. Nothing exists for
recommendation. It gives S1/S2 a modern hook.

**OBJECTION.**
- After a reasoning trace, P(Yes) at the answer token saturates to 0/1, so ECE is dominated by saturation.
  Verbalized confidence is needed alongside it.
- The result is derivative of QA findings (2505.14489, 2508.15050, CDUR 2606.11211).
- About 400 extra tokens per prompt.

**Info:** MED. One RQ table in the observation study or the appendix.

### B2: self-confirming calibration (rank 7; named high-upside)

**FOR.**
- Reverses S3's causal arrow ("the loop produces the confidence").
- Makes a sharp on-path versus off-path prediction: logged calibration improves while MAR tail calibration
  degrades.
- Its LLM-specific "prior erosion" prediction would be a genuinely new and citable result.
- It comes with a fix (an IPS-weighted proper score).

**OBJECTION.**
- The proposition is the textbook MNAR/PU minimizer, and the fixed point is performative stability, so
  reviewers will call the theory a restatement.
- IPS requires a stochastic serving policy, which changes the served utility.
- The full study needs 25–70 GPU-h, and KuaiRec readability is unproven.
- The MF-only CPU check would merely confirm the textbook result.

**FAILURE MODE.** One LLM round shows Δ inside noise, which settles nothing.

**Info:** MED now; HIGH if pilot 2 passes.

### B3: ask the fallback (rank 9)

**FOR.**
- Its central claim, that the oracle is inflated, is **correct**: §0.1 confirms it.
- The pre-hoc constraint (decide before paying for the LLM) is operationally right.

**OBJECTION.**
- The diagnosis is now answered for free, and RouteRec already published its policy lesson.
- The remaining L2D method chases ≤ ~+0.01 NDCG.
- The leave-two-out split needs a re-split of the frozen panels.
- The routing space is crowded (S-LLMR, 2609.39043, ReliGRec).

**Info:** LOW-MED. Report the corrected-oracle table as a paragraph.

### B4: prune the familiar (rank 15)

**FOR.** A bold, principled inversion (the IPS view of high-propensity positives; analogous to word2vec
frequent-word subsampling). Either sign is publishable, and it answers S6 directly.

**OBJECTION.**
- Every targeted pruning signal so far has tied random (N3, GORACS, MiniRec).
- σ_seed ≈ 1.5 pt.
- Familiar positives *do* carry information (KnowSA).
- UCalRec's head label-smoothing was null.
- The 3 GPU-h pilot cannot resolve sub-2σ effects.

**Info:** LOW-MED within budget.

### B5: concept erasure (rank 16)

**FOR.** Separating the platform-popularity mediator from the pretraining-familiarity mediator is
conceptually sharp, and item-specific erasure escapes N1.

**OBJECTION.**
- Popularity steering already exists (SPREE, PopSteer, activation steering).
- LEACE gives only a linear lower bound.
- A familiarity probe built from title log-likelihood is confounded by length and genericness.
- Caching hidden states at 101-candidate scale is slow.
- A2 answers the same causal question at the input more cleanly.

**Info:** MED-LOW.

### B6: selective serving silences niche users (rank 11)

**FOR.** CPU-only and likely true. The LLM-specific twist (abstaining to CF may hurt niche users twice) is
supported by §0.1's pattern: CF baselines win almost only on head targets. It answers S5 on the user side.

**OBJECTION.** The result is expected from Jones et al. ICLR'21, and the Mondrian CRC fix is standard.

**Info:** MED. Appendix.

### B8: decision-weighted proper scoring (rank 18)

**FOR.** Principled (Schervish and Buja–Stuetzle–Shen threshold mixtures) and proper.

**OBJECTION.** Gains are likely inside σ_seed, the band set at the validation threshold is circular, and it
covers only S4.

**Info:** LOW.

### B10: abstention feedback loop (rank 19)

**FOR.** A novel inversion of S3: acting on *low* confidence builds the echo chamber.

**OBJECTION.** It needs B2's infrastructure, inherits the simulator critique, and KuaiRec's niche/mainstream
split is weak.

**Info:** LOW.

---

## 3. Top-3 pilot picks (pre-registered; each ≤2 GPU-h)

Common gate for pilots 1 and 3 (token channel sanity):
- On ML-1M rated pairs, UAUC(raw token P(Yes)) must be ≥ 0.60.
- On the Lumen sports next-item panel, NDCG@10 of raw P(Yes) must be ≥ 0.8 × C-CRP verbalized (0.2329).
- If either fails, the token channel or prompt is broken and nothing below is interpretable. Fix the prompt
  first (no idea is killed on a broken channel).

Compute assumption: ≥ 60 prefill prompts/s on the new server with vLLM prefix caching. If throughput is
lower, scale users down proportionally and keep the arms.

Pilots are deliberately small; the paper itself must be full-scale (10k users, 101 candidates; CLAUDE.md
rule 3).

### 3.1 Pilot 1: A1 MIRROR, with the A9/B9 prior measured head-to-head (≈1.7 GPU-h)

**Data.**
- Rated panels:
  - ML-1M: 1.5k users × 20 rated candidates (rating ≥4 = Yes, ≤2 = No, 3 dropped);
  - Amazon-2023 Toys: users with ≥3 likes and ≥3 dislikes, up to 1.5k.
- Next-item panel: Lumen sports, 1k users × 101 candidates.

**Arms (Qwen3-8B, thinking off, prefill-only P(Yes)):**
- *like*;
- *dislike*;
- *paraphrased-like* (the placebo; equal compute, so MIRROR is compared against a two-prompt ensemble);
- the user-marginalized prior: *like* under K = 8 random other users' histories, once per unique item.

**Estimands (pre-registered).**
- **Acquiescence a(i)** = (logit_like + logit_dislike)/2.
  - Centre it within each user to remove the global yes-bias, which is rank-inert.
  - Average over users per item.
  - Report SD across items and Spearman corr(a, log-popularity).
- **Valence v(i)** = the user-average of (logit_like − logit_dislike)/2. Report corr(v, log-popularity).
  This tests the valence objection directly.
- **Prior π(i)** from user swaps; **evidence** = logit_like − π(i).
- **Primary endpoint:** ΔUAUC(MIRROR − paraphrase placebo), paired bootstrap over users, per rated domain.
- **Secondary endpoints:**
  - ECE and Brier on rated labels;
  - AUROC of {raw, MIRROR, evidence} for confident-error detection;
  - ΔNDCG@10 on the next-item panel vs raw P(Yes) and vs C-CRP;
  - head share of top-10.

**Decision rule.**
- **POSITIVE.** All three hold:
  - SD(a) ≥ 0.20 logit, and corr(a, log-pop) ≥ 0.20 in ≥1 domain;
  - the ΔUAUC CI excludes 0 in ≥1 rated domain;
  - no NDCG@10 loss > 0.005 on the next-item panel.

  Then MIRROR is the flagship: proceed to mirror tuning (3 seeds) and the full 4 × 2 grid.
  *Teaches:* item-conditional yes-saying is real, popularity-linked and rank-relevant, and it is H1's
  mechanism.
- **NEGATIVE.** SD(a) < 0.10, or the ΔUAUC CI ≤ 0 in both rated domains.
  - MIRROR dies as a method; the acquiescence measurement stays as one observation paragraph.
  - If |corr(v or π, log-pop)| ≥ 0.20, the flagship passes to **B9/A9**: the evidence decomposition, used
    for confident-error detection and cross-user decisions.

  *Teaches:* LLM-rec confidence is negation-coherent, and familiarity enters through valence or prior, not
  yes-saying.
- **NULL / AMBIGUOUS.** a(i) is large and popularity-linked, but there is no UAUC gain beyond the placebo.
  - Acquiescence exists but is rank-irrelevant. Report it as a calibration finding.
  - The flagship moves to B9/A9 unless pilot 3 shows a(i) is familiarity-driven.

  *Teaches:* acquiescence is a calibration artifact, not a ranking one.

### 3.2 Pilot 2: B1 ≈ A7, KuaiRec MNAR vs MAR calibration (≤2 GPU-h; gates 5 ideas)

**Data.**
- KuaiRec small matrix: 300 users × 200 items (60k prompts, about 20–30 min).
- Prompt inputs: Chinese captions plus categories.
- Label: watch_ratio ≥ 2.0; sensitivity at 1.0 and 3.0 on CPU.
- First verify on CPU how big/small-matrix pairs overlap, and drop shared pairs.

**Arms.**
- Qwen3-8B zero-shot.
- If zero-shot AUC under MAR labels is < 0.58, spend the remaining ~1.5 GPU-h on **one** short LoRA on the
  big matrix and re-score. Pre-registered: no other changes.

**Endpoints.**
1. Share of top-decile-confidence "false positives" under the MNAR view (unobserved-in-big-matrix treated as
   negative) that are positives under MAR labels.
2. Exposure gap = logged ECE − MAR ECE, with bootstrap CI.
3. Sign of the confidence-matched head-vs-tail Bias Index under each view.
4. If the LoRA arm runs: MAR ECE before vs after LoRA (B1's "fine-tuning re-learns the logging policy").

**Decision rule.**
- **POSITIVE.** ≥ 20% of confident false positives are MAR positives, and the exposure-gap CI excludes 0.
  - S2's "confidently wrong" must be reported against MAR labels.
  - The observation study gains an exposure-validity RQ.
  - **B2 earns its conditional slot (§3.4).**
- **NEGATIVE.** < 10%, with exposure gap ≈ 0. Logged rated labels are an adequate proxy, so drop B1, A7 and
  B2; the main study runs on rated ML and Amazon panels with a one-paragraph validity check.
- **NULL.** AUC < 0.58 even after LoRA. The LLM cannot read KuaiRec, so B1, A7, B2, A6 and B10 are all
  deprioritized and S3 is answered only through static exposure. *Teaches:* where *not* to build the loop
  infrastructure, which saves weeks.

### 3.3 Pilot 3: A2 pseudonym knockout, diagnostic only (≈1.2 GPU-h; reuses pilot 1's *like* arm on Toys)

**Data.**
- Amazon-2023 Beauty and Toys, both brand-dominated; ≤1.5k users × 20 rated candidates.
- Pseudonym map built from the `store`/brand field plus brand tokens in titles, hash-seeded and consistent
  within each example.

**Arms.**
- Real names (reused from pilot 1 on Toys).
- Consistent pseudonyms.
- **Placebo:** a real-brand swap to another real brand of matched popularity tercile. This is required:
  without it, a uniform shift cannot be told apart from a perturbation effect.

**Endpoints.**
- Δ mean logit (real − pseudonym) by popularity decile: slope, and head minus tail.
- Confidence-matched Bias Index before and after.
- ΔUAUC overall and in the tail.
- **Cross-check with pilot 1:** corr(a, log-pop) and corr(π, log-pop) under pseudonyms vs real names.

**Decision rule.**
- **POSITIVE.** Head-minus-tail Δlogit ≥ 0.20 with CI, the placebo Δ is within ±0.05, and tail UAUC changes
  by ≤ 0.01.
  - Familiarity causally inflates head confidence; S5 is answered causally and N2's null is explained.
  - Whichever of a(i) or π(i) loses its popularity slope under pseudonyms is the familiarity estimator, which
    tells the flagship which decomposition is principled.
- **NEGATIVE.** Flat Δ across deciles, with the placebo ≈ the pseudonym arm. The popularity skew of confidence
  is not name recognition but base rate or in-prompt evidence; H1's "familiarity" wording must be dropped.
- **NULL.** UAUC falls everywhere by more than 0.02. Pseudonymization destroys evidence in that domain:
  restrict to brand tokens on Beauty, or report the design limitation.

### 3.4 High-upside conditional slot: B2

What would convince me to fund B2's one-round LLM loop (about 2 GPU-h) after pilot 2 is POSITIVE:
- zero-shot or LoRA AUC ≥ 0.60 on KuaiRec;
- a naive retraining round raises MAR tail ECE ≥ 1.5× (CI) while logged ECE stays within ±0.01;
- the IPS arm shows no such divergence.

That pattern is the "logged calibration is uninformative about counterfactual calibration" result, which no
amount of textbook theory pre-empts. Without it, B2 is a restatement of performative prediction.

### 3.5 Which ONE carries a full SIGIR paper

**A1 MIRROR,** conditional on pilot 1.
- **Observation:** a rated-label calibration study of token P(Yes) on 4 datasets (ML-1M, Amazon Toys,
  Amazon Beauty, Steam) × 2 backbones (Qwen3-8B, Llama-3.1-8B).
- **Method:** MIRROR scoring (zero-shot) plus mirror tuning (LoRA, 3 seeds).
- **Gains reported:**
  - UAUC and ECE on the rated panels;
  - NDCG on Lumen's 4 next-item panels against the 8 official baselines;
  - head share in top-10;
  - every gain against the paraphrase placebo.

**Fallback if pilot 1 fails:** B9/A9 (evidence decomposition), using the same observation study and panels; the
method becomes evidence-based confident-error detection and serving.
**Not a fallback:** A's suggested A4+A5. §0.1 removes its motivation.

---

## 4. Recommended paper spine, and where S1–S6 land

**Spine.** "Yes because it knows the item, or because it knows the user? Separating item familiarity from
personal evidence in LLM-recommender confidence."

There is one decomposition, logit P(Yes) = item term + personal term, and one method, MIRROR (or, on the
fallback, evidence scoring). Every seed question is answered *in terms of that decomposition*, which is what
keeps the paper from becoming a kitchen sink.

**Main body.**

| Section | Content | Seeds answered |
|---|---|---|
| §3 RQ1 | Reliability of token P(Yes) against rated labels (fixes CALIBRA's 1/101 flaw): ECE, Brier, AUROC of confidence for correctness, reliability diagrams; validity under MAR labels from pilot 2 ("confidently wrong vs confidently unlabelled") | **S4** |
| §3 RQ2 | Correct-but-unsure mass and confidently-wrong mass; whether errors are low-confidence or confident; what share of confident errors the item term carries; paragraph with the corrected-oracle numbers from §0.1 ("when the LLM is wrong, is CF right? rarely, and unpredictably") | **S1, S2** |
| §3 RQ3 | Confidence *level* vs confidence-matched *calibration gap* by popularity (ProCal Bias Index); a(i)/v(i)/π(i) vs popularity; the causal pseudonym knockout (pilot 3) | **S5** |
| §4 | MIRROR scoring and mirror tuning; identifiability statement; the A10 lemma (N1 and its complement) as the formal reason only an item-dependent correction can change ranking | — |
| §5 | 4 datasets × 2 backbones: UAUC, NDCG, ECE/Brier, PIECE-style popularity-informed error; Lumen 4-domain next-item panels against 8 baselines; paraphrase placebo; 3 seeds for the tuned variant | — |
| §5 / §6 | Does confidence-ranked serving concentrate exposure on familiar items (head share, Gini against demand), and does MIRROR reduce that at equal NDCG? This is the static echo-chamber answer | **S3** (static) |

**Appendix.**

| Topic | Content | Seeds answered |
|---|---|---|
| Dynamic echo chamber | A6's zero-shot 2-round in-context loop on KuaiRec if pilot 2 is POSITIVE; otherwise a replay loop on Amazon rated items, stated honestly as model-side only | **S3** (dynamic) |
| Prune by uncertainty | Detection AUROC of three signals against natural rating–text contradiction noise (A8's cheap half): raw uncertainty (low \|logit\|), MIRROR contradiction with the label, prior-congruence (B4's signal). Then one ≥5-seed matched-random 25% prune on one domain, using the mirror-tuning pipeline. Expected and acceptable answer: raw uncertainty ≈ random (consistent with N3/GORACS) because it flags difficulty, not noise | **S6** |
| Further RQs | A11/B11 thinking-mode table; B6 user-side niche analysis; A10/B7 per-user offset (exact UAUC invariance) | — |

**Kept out of the spine:**
- C4 serving and routing: one paragraph only;
- the B2 and B10 loops: future work unless §3.4 fires;
- A3 OLMo counts: one sentence citing PerRecBench and Mozafari;
- B5, B8.

**Named spine risks.**
1. Pilot 1 may hand the flagship to B9/A9, whose estimator is prior art. The paper would then be
   "observation + an established estimator used in a new place", a weaker method claim.
2. Rated-panel construction on Amazon (5-star skew) is the critical path.
3. Same-family review: every ranking here should be re-challenged by a different-family reviewer before the
   full grid is funded.

---

## Sources (novelty checks this session)

- [PerRecBench: Can LLMs Understand Preferences in Personalized Recommendation? (2501.13391)](https://arxiv.org/html/2501.13391v1): uses infini-gram item-name counts vs accuracy.
- [LRWorld: The Mental World of LLMs in Recommendation (2512.17389)](https://arxiv.org/html/2512.17389): no infini-gram or confidence analysis.
- [Incumbent Advantage: Brand Bias in LLM Recommendation (2606.17443)](https://arxiv.org/abs/2606.17443)
- [The yes–no bias of LLMs reflects answer order and wording (2607.05552)](https://arxiv.org/abs/2607.05552)
- [Batch Calibration (Zhou et al., ICLR'24; 2309.17249)](https://arxiv.org/abs/2309.17249)
- [RouteRec: Strict Evaluation of Recommender-Agent Selection (2607.09908)](https://arxiv.org/html/2607.09908)
- [Routing Between Generative and Collaborative User Profiles (2609.39043)](https://arxiv.org/abs/2609.39043)
- [ReliGRec: User-Risk-Aware Prompt Routing (2609.16560)](https://arxiv.org/abs/2609.16560)
- [Pretraining Exposure Explains Popularity Judgments in LLMs (2605.12382)](https://arxiv.org/html/2605.12382)
- [Calibration Drift Under Reasoning (2606.11211)](https://arxiv.org/abs/2606.11211)
- [Reasoning Models Better Express Their Confidence (2505.14489)](https://arxiv.org/html/2505.14489)
- [Don't Think Twice! Over-Reasoning Impairs Confidence Calibration (2508.15050)](https://pith.science/paper/2508.15050)
- [Performative Prediction (Perdomo et al., 2002.06673)](https://arxiv.org/pdf/2002.06673)
- [Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias (MACR, 2010.15363)](https://arxiv.org/pdf/2010.15363)
- [SPREE: Aligning Recommendations with User Popularity Preferences (2604.01036)](https://arxiv.org/pdf/2604.01036)
- [RankSteer: Can Pointwise LLM Rankers Be Calibrated at the Representation Level? (2602.03422)](https://arxiv.org/html/2602.03422)
- [Hard vs. Noise: LLMHNI (2511.07295)](https://arxiv.org/html/2511.07295)
- [Buja, Stuetzle, Shen 2005: Loss Functions for Binary Class Probability Estimation](http://stat.wharton.upenn.edu/~buja/PAPERS/paper-proper-scoring.pdf)
- [KuaiRec (2202.10842)](https://arxiv.org/html/2202.10842v3)

Triage-Verdict: A1 | review_independence: same-family
