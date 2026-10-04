# METHOD INVENTOR

**Headline.** In Pilot 1, Qwen3-8B's zero-shot yes/no logit tracked how familiar an item is, not how good it is or what this user thinks of it. That is the NLP "demonstration shortcut" showing up in recommendation. So the methods worth piloting change one thing at a time: scramble only the user's item-to-rating mapping (VCC), validate on the user's own past ratings (SPJ), or compare each candidate against an equally familiar foil (FMD). None of these is expected to beat leave-user-out item-mean CF (UAUC 0.76/0.71) on rated panels, so every rated result must be reported next to it.

## Analysis
STATUS OF EVIDENCE. Every number below comes from Pilot 1 or from the gate-failure diagnosis appended to docs/sigir/PILOT_LOG.md on 2026-10-02. All of it is diagnostic: it motivates the designs but feeds no decision rule. Every rule below is evaluated only on users and domains that Pilot 1 never touched, and is pre-registered before any scoring.

1. WHAT THE FACTS SAY ABOUT THE CHANNEL

(a) On rated panels the yes logit is an item-familiarity score.
- Like-logit UAUC is 0.587 on ML-1M and 0.540 on Toys.
- The leave-user-out item mean rating, a non-personalised signal, reaches 0.760 and 0.712 on the same pairs (outputs/confrec_pilot/pilot1/*.diag_baselines.json).
- Spearman correlation of the like-logit with item quality (item mean rating) is only 0.15 (ML-1M) and 0.07 (Toys).
- The user-swap prior pi(i) correlates with log-popularity at 0.43 and 0.28.
- Subtracting pi (the "evidence" arm) collapses UAUC to 0.507 and 0.525.
- Reading: the within-user ranking signal sits almost entirely in an item term, and that term behaves like recognition or familiarity, not quality. The user-specific residual carries close to nothing, and the no-history PMI arm keeps only 0.547.

(b) The two-prompt corrections tried so far failed for identifiable reasons.
- MIRROR negates the question. An 8B model handles negation inconsistently across domains: the dislike logit has UAUC 0.408 on ML-1M (so its negation gives 0.592) but 0.480 on Toys. MIRROR also costs about 0.03 NDCG@10 on next-item.
- Swap and no-history priors delete the user's entire context, including topical match. Topical match is exactly what makes the LLM competitive on next-item: raw 0.209–0.219 against C-CRP 0.231, with all 8 baselines below that.

(c) The logit's overall scale is wrong, which matters as soon as it is combined with anything.
- The Platt slope is 0.062, i.e. the logits are about 16 times too extreme. That is rank-inert inside one user.
- It becomes decisive when the logit is fused with any other signal, since a naive sum is swamped by the LLM term, and for any cross-user threshold.

(d) Confidence magnitude does not know when it is wrong. |logit| detects errors with AUROC 0.58, consistent with magnitude measuring familiarity rather than evidence.

(e) Pilot 2 (KuaiRec) was NULL: UAUC_MAR 0.469. The dynamic echo-chamber loop is off the table, and S3 is answered statically through head share.

UNIFYING HYPOTHESIS. This is the NLP "demonstration shortcut":
- Jang et al., NAACL'24: in-context learners lean on the semantic prior of the demonstrations rather than on the input–label mapping.
- Min et al., EMNLP'22, and Wei et al. 2023: small models barely react when demonstration labels are corrupted.
- The recommendation analogue: a user's rated history is a demonstration set, and the item→rating mapping is the user's taste. The 8B model appears to use the items (topic, familiarity) and ignore the mapping.
- This can be tested directly, and it tells a method what to hold fixed and what to vary.

2. DESIGN RULES DERIVED FROM THE FACTS

- R1. A counterfactual must vary only the factor being measured. It should keep the user's items (topic and familiarity) when measuring taste, and keep the item's familiarity when measuring preference. Question negation, no-history and user-swap all changed too much.
- R2. Every correction must be item-dependent, or depend on user × channel, so that it is not rank-inert (negative N1).
- R3. Every multi-prompt method gets an equal-compute, information-preserving placebo:
  - reorder the history lines, keeping each item's rating attached;
  - or use a paraphrase ensemble.
- R4. Rated-panel claims are made as a gain over the LLM base and the placebo. They are always reported next to item-mean CF and an LLM+CF fusion. Next-item claims are made against C-CRP and the 8 official baselines on the frozen panels.
- R5. Tuning knobs (lambda, kappa, stacking weights) are chosen on a dev split. Test users are touched once.

3. CANDIDATES

A summary score of gain × novelty × feasibility (each 1–5) ranks them:

| Rank | Candidate | Score |
|---|---|---|
| 1 | VCC | 3 × 3 × 5 = 45 |
| 2 | SPJ | 2.5 × 2.5 × 5 = 31 |
| 3 | FMD | 3.5 × 2.5 × 3.5 = 31 |
| 4 | TC-LoRA | 3.5 × 2.5 × 2.5 = 22 |
| 5 | UGF | 2.5 × 1.5 × 5 = 19 |

UGF is nonetheless mandatory reporting, because it is the honesty anchor against item-mean CF.

C1. VALENCE-COUNTERFACTUAL CONTRAST (VCC)

Notation.
- h_u = [(t_j, r_j)] is the rendered history: titles with "(rated r/5)".
- Counterfactuals:
  - c_P(h) permutes the ratings among the same items. It keeps the item set, the line order and the rating multiset, i.e. the user's leniency.
  - c_F(h) maps each rating r_j to 6 − r_j.
  - c_O(h) reorders the lines with ratings attached. This is the placebo, since it preserves all information.
- For next-item histories, which carry no ratings: L+ is the history annotated "(liked)" and L− the history annotated "(disliked)".

Estimators.
- tau_X(u,i) = L(h_u, i) − L(c_X(h_u), i), with tau_P averaged over K = 2 permutations.
- tau_next = L+ − L−.
- Score: s_lambda = L + lambda · tau, which is (1 + lambda) · L − lambda · L_cf.

Identification.
- Assume an additive model: L = a_u + f(items(h), i) + g(mapping(h), i) + pi(i) + e.
- Then tau_P = g(V, i) − g(V_perm, i).
- Familiarity pi(i), topical match f, and per-user leniency a_u all cancel exactly. Only the taste mapping survives.
- tau_F additionally carries a leniency × item interaction; tau_P is the clean check.

Why it is not a dead idea.
- MIRROR negates the question; VCC negates the evidence and keeps the question fixed.
- Item-prior subtraction (swap prior, no-history PMI, CFT, NPRec) deletes the user's items. VCC keeps them.
- lambda = 0 recovers the base score, so dev selection guards against loss.
- tau varies across items within a user, so VCC is not rank-inert.

Novelty check.
- The score form is prior art: contrastive decoding (Li et al., ACL'23), CAD (Shi et al., NAACL'24), and above all ICCD (Peng et al., ACL'25), which contrasts correct against corrupted input–label demonstrations.
- What is new (to my search):
  - using a user's rated history as the demonstration set;
  - an item-preserving rating permutation as the counterfactual, so familiarity and topic are held fixed;
  - reading |tau| as "taste-attributable confidence" for S1, S2 and S5 decisions;
  - showing that the choice of counterfactual matters, against swap-CAD and no-history-CAD (CFT-style) controls.
- Realistic novelty: 5/10, the same footing as MIRROR. The estimator form may not be claimed.

C2. PROMPT-LEVEL JACKKNIFE SELF-PROBES (SPJ)

Idea. An in-context recommender can be validated on each user for free. Ask the same prompt about the user's own earlier rated items, whose labels are known, without any retraining.

Estimators.
- Per-user probe AUC and log-loss, computed both for the LLM channel and for leave-user-out item-mean CF on the same probes.
- Shrinkage: r~_u = (n+ · n− · AUC_u + kappa · mean AUC) / (n+ · n− + kappa).

Uses.
- (A) Cross-user selective serving: send the LLM list to users with high r~, otherwise fall back to CF.
- (B) Per-user stacking: s = w_u · z_u(L) + (1 − w_u) · z_u(q), with w_u = sigmoid(b0 + b1 · (r~_LLM − r~_CF)). This changes within-user order whenever the LLM and CF orders differ.
- (C) Per-user Platt fitted on the probes. This is rank-inert within a user but changes global AUC, ECE and cross-user thresholds (S4).

Novelty.
- Per-user algorithm selection and stacking are old: Ekstrand & Riedl RecSys'12; FWLS, Sill et al. 2009.
- In-Context Calibration (Jang et al., NAACL'24, the POSTECH group, i.e. likely reviewers) already runs leave-one-out over demonstrations, but unsupervised, to estimate a global label prior.
- SPJ's delta: supervised per-user reliability from the user's own labelled history inside a personalised LLM recommender, used for cross-user decisions. Realistic novelty: about 4.5/10.

Power arithmetic.
- With 20 candidates and 10–20 probes per user, the measurement reliabilities are roughly 0.3 for test UAUC and 0.2 for probe AUC.
- So an observed correlation of 0.10–0.25 is achievable only if the true between-user SD of AUC is at least about 0.06.
- Hence the pre-registered threshold of 0.10. The earlier verbalized-confidence gate reached 0.07.

C3. FAMILIARITY-MATCHED FOIL DUELS (FMD)

Idea. Elicit preference by comparing each candidate against M = 2 foils of equal LLM familiarity (nearest no-history logit pi_0), with both orders. The item's own familiarity then cancels by design, while the user context stays intact.

Estimator.
- d(u,i,f) = ½ · [(l_A − l_B | i first) − (l_A − l_B | i second)].
- Score: s = mean over foils f of d.
- Order-inconsistency rate is the "is it sure" readout.

Support and prior art.
- Support: PerRecBench (Tan et al. 2025) reports that pairwise and listwise beat pointwise when personalisation is isolated; PRP (Qin et al., NAACL-F'24) shows the same in IR.
- Prior art also includes anchor-based pointwise ranking (GCCP, Long et al., SIGIR'25).
- So the novelty rests on familiarity matching. Realistic novelty: about 4.5/10.
- It is costlier than VCC: 4 prompts per (candidate, foil) pair, and longer prompts.

C4. TASTE-CONTRASTIVE LoRA (TC-LoRA)

Conditional on VCC showing a non-degenerate tau.

Objective: BCE(sigmoid(L_theta(h,i)), y) + mu · BCE(sigmoid(L_theta(h,i) − L_theta(c_P(h), i) + b), y). The auxiliary term trains the model to use the rating mapping, which is the anti-shortcut target.

Prior art and novelty.
- Closest prior: CFT (Zhang et al., arXiv 2410.22809). It fits labels on L(h) − L(no history).
- The delta is an item-preserving counterfactual. Realistic novelty: about 4.5/10.

Why it matters. This is where SIGIR-grade rated results would have to live: the zero-shot 0.59 UAUC is far below item-mean CF.

C5. UNCERTAINTY-GATED LLM×CF FUSION (UGF)

Model: logit P = b0 + b1 · logit q(i) + (b2 + b3 · phi(i) + b4 · |tau| + b5 · r~_u) · z_u(L).
- q is the shrunk leave-user-out item like-rate.
- phi(i) = pi_0(i) is the LLM's familiarity of the item.

The LLM's weight depends on the item, so UGF changes within-user rankings. Novelty is low (FWLS-style meta-features). Its value is the mandatory quantity ΔUAUC(fusion − item-mean): does zero-shot LLM confidence add anything beyond item quality?

4. MAPPING TO THE SIX SEED QUESTIONS

S1, correct but is it sure?
- VCC: the share of correct predictions with |tau| near 0 is "right for the wrong reason (familiarity)".
- FMD: order consistency.
- SPJ: per-user probe reliability.

S2, wrong because of low confidence?
- Error-detection AUROC of −|tau|, order inconsistency, and r~_u, each against |L| (0.58).
- Tested prediction: errors concentrate where taste support is absent, not where |L| is small.

S3, does high confidence cause echo chambers?
- Static only, since Pilot 2 was NULL.
- Measure top-10 head share and tail coverage on next-item under L vs s_lambda, vs FMD, and for SPJ's served lists.
- Prediction: confidence driven by familiarity over-exposes head items; taste-attributable scoring reduces head share at equal NDCG.

S4, yes/no confidence vs ground truth:
- Reliability and ECE of each channel on rated labels after dev-Platt.
- SPJ per-user Platt validated on labelled probes; its gain shows up in global AUC and ECE.

S5, popular items high confidence, niche low?
- Decompose L into tau and L − tau. The prediction is that the popularity link (0.3–0.4 in Pilot 1, to be re-measured) sits in the taste-free part, not in tau.
- FMD holds familiarity fixed by design.

S6, prune training data by uncertainty because of noise?
- Detection only, inside the budget.
- "Taste-contradicted" labels (large |tau_P| with sign opposite to the label) vs "prior-contradicted" labels (pi opposite to the label) as noise vs hardness (H3). Evaluate against natural rating–review contradictions on Amazon.
- SPJ probe self-inconsistency flags noisy users.
- Any pruning claim still needs at least 5 seeds against matched-random pruning, which is beyond pilot scope.

5. SHARED PILOT LOGISTICS (applies to all five)

Step 0, the registered remedy "fix the prompt first" (about 0.2 GPU-h):
- Use 400 dev ML-1M users drawn from the 1,683 eligible users Pilot 1 did not use (eligible 3,183, of which 1,500 were used).
- Four pre-declared prompt variants:
  - P0, the current prompt;
  - P1, star ratings verbalised as liked/disliked;
  - P2, a 1–5 rating question read off the digit tokens 1–5, scored by the probability-weighted mean (G-Eval style), with Var[r] as a side readout;
  - P3, a user-relative question ("rate it above this user's usual?").
- Pick the variant with the best dev UAUC. The registered gate constant is unchanged: base UAUC ≥ 0.60 on ML-1M test users, and next-item NDCG@10 ≥ 0.8 × C-CRP on the same events.
- If no variant passes, the zero-shot rated endpoints become exploratory only. In that case C4 becomes the route for rated-panel claims, and next-item endpoints stay interpretable if they pass their own gate.

Data partitions.

| Panel | Dev | Test |
|---|---|---|
| ML-1M rated (fresh users) | 400 | 1,283 |
| Video_Games rated (new domain; fall back to Sports rated if fewer than 800 eligible) | 400 | the rest |
| Next-item | sports events 1,001–1,500 | toys events 1–1,000 |

- Video_Games eligibility must be checked on CPU first.
- Toys next-item test events exclude any user in Pilot 1's Toys rated panel. Disjointness is asserted.
- C-CRP and the 8 baseline ranks exist for the toys next-item events.

Pre-registration.
- Write idea-stage/PREREG_AMENDMENT_2.md before any scoring. It holds all five rules, the lambda/kappa grids, and a Holm correction across the zero-shot primaries (VCC, SPJ-P2, FMD).
- VCC, SPJ and UGF share base arms and can run as one job of about 1.9 GPU-h. FMD and TC-LoRA follow only if their pre-conditions hold.
- Total if everything runs: about 4.9 GPU-h.

Optional robustness: replicate the winning candidate on Llama-3.1-8B, about 0.5 GPU-h, using a separate output root.

6. HONESTY NOTES

- No zero-shot LLM-only method here is expected to reach item-mean CF on rated UAUC (0.76/0.71). Claims are framed as a gain over the LLM base and placebo, plus the fusion quantity.
- If VCC's mechanism check fails, the paper gains a clean negative observation: "the zero-shot yes is about the item, not about you". That is consistent with the ICL label-insensitivity literature, and it moves the method story to C4.
- Two-sided outcomes are pre-specified for every candidate. Pilot-1 diagnostics are not reused as evidence, and the correlations are re-measured on fresh users.

7. VERIFICATION LIMITS

Checked in this session:
- ICCD (arXiv 2502.13738; ACL 2025 per arXiv; formula z + alpha·(z − z−); negatives keep labels and swap inputs).
- In-Context Calibration (NAACL'24 long).
- CFT's counterfactual (history set to None).
- NPRec (CIKM'26).
- PerRecBench (arXiv, venue not verified).
- PRP (NAACL-F'24).
- GCCP (SIGIR'25).
- CAD (arXiv 2305.14739).
- FWLS (arXiv 0911.0460).
- Ekstrand & Riedl (RecSys'12).
- LiDu (arXiv 2507.23208).
- CoPe (EMNLP'25).

Cited from memory and not re-verified this session:
- Min et al. 2022 (2202.12837)
- Wei et al. 2023 (2303.03846)
- Li et al. Contrastive Decoding (2210.15097)
- G-Eval (2303.16634)

Venues not verified:
- PerRecBench
- CFT
- EviRank

The UQRec, UGR, EviRank, KnowSA, TALLRec, CoLLM, ConfTuner, ProCal, CCS, Calibrate-Before-Use, Batch Calibration, PMI-DC and 2607.05552 entries rely on the project's dossier verification.

"Not found" statements mean that 10 web searches surfaced no such paper. That is absence of evidence, not an exhaustive survey.

No file was written in the repository or in the deliberation folder.

## Recommendations
### #1 VCC: Valence-Counterfactual Contrast, i.e. amplify the part of P(Yes) that depends on the user's item-to-rating mapping  (≈1.5 GPU-h)
- Rationale: The Pilot-1 diagnostics say the yes logit carries item familiarity, not taste:
- the swap-evidence arm collapses UAUC to about 0.51;
- correlation with item quality is 0.15 / 0.07;
- pi correlates with log-popularity at 0.43.

The counterfactuals used so far each changed too much. Question negation is unreliable at 8B, and removing the history also removes the topical match that wins next-item.

VCC permutes or flips only the ratings attached to the user's own history items. Under the additive model, familiarity, topic and leniency then cancel exactly, and only taste survives.

The score s = L + lambda·tau is item-dependent, so it is not rank-inert. Setting lambda = 0 recovers the base, which protects against loss.

The score form is ICCD/CAD prior art. The claim must rest on three things:
- the recommendation counterfactual that preserves the items;
- the taste-attribution readout;
- the proof against swap and no-history controls that the choice of counterfactual matters.
- Protocol: 0) Write the PREREG_AMENDMENT_2 rules before any scoring. Run the shared Step 0 prompt fix: 4 variants on 400 dev ML-1M users; the gate constants stay as registered.

1) Data:
- ML-1M rated: the 1,683 users Pilot 1 did not use. Dev 400 / test 1,283. Assert disjointness from Pilot 1.
- Video_Games rated: a new domain, dev 400 / test the rest. If fewer than 800 users are eligible, use Sports rated.
- Next-item: sports events 1,001–1,500 as dev; toys events 1–1,000 as test, excluding Pilot-1 Toys-rated users.

2) Arms, rated (hist_len 10):
- base L(h);
- flip L(c_F h), with r -> 6 - r;
- perm, K = 2 rating permutations among the same items (seeded by hash(u, k); resampled if identical);
- reorder placebo L(c_O h), lines shuffled with ratings attached;
- paraphrase L_para;
- controls: swap prior pi (K = 8 donors, exclusions as in P1.6) and no-history L_0.

3) Arms, next-item:
- base;
- L+ with every history line suffixed "(liked)";
- L- suffixed "(disliked)";
- reorder placebo.

4) Estimators:
- tau_F = L - L_F; tau_P = L - mean_k L_Pk; tau_O = L - L_O; tau_next = L+ - L-.
- s_lambda = L + lambda·tau.
- Controls: swapCAD = L + lambda(L - pi) and nohistCAD = L + lambda(L - L_0).
- lambda is chosen per arm from {0, 0.25, 0.5, 1, 2, 4} on dev users: mean UAUC for rated, tie-exact NDCG@10 for next-item.

5) Mechanism M: pooled within-user SD of tau_P (divisor N - U) >= 0.20 logit, AND UAUC(tau_P alone) CI > 0.5, in at least 1 rated domain.

6) Primary P, on test users with a paired user bootstrap (2,000 resamples): dUAUC = s_lambda*(tau_F) - s_lambda*(tau_O).
- POSITIVE requires all of:
  - CI lower bound > 0 in at least 1 rated domain, and point estimate >= -0.002 in the other;
  - point dUAUC versus both swapCAD_lambda* and nohistCAD_lambda* >= 0 in the positive domain;
  - toys next-item dNDCG@10(s_lambda* - base) >= -0.005.
- STRONG: additionally, next-item dNDCG@10 versus the reorder placebo has CI > 0, or top-10 head share falls by at least 2 pp with no NDCG loss.
- NEGATIVE: SD(tau_P) < 0.10 and UAUC(tau_P) CI covers 0.5 in both rated domains. Report "the zero-shot yes ignores the user's ratings" (S1/S2 finding) and stop VCC.
- NULL: M holds and P fails. Keep tau only as an uncertainty readout. Claim error detection only if AUROC(-|tau_P|) - AUROC(-|L|) >= 0.03 with CI > 0.

7) Report alongside, as non-decision inputs: item-mean CF (leave-user-out), C-CRP and the 8 baselines on the toys events, head share, and the re-measured corr(tau, log-pop) vs corr(L - tau, log-pop).

Prompt budget: about 165k ML-1M + about 100k Video_Games + about 450k next-item + about 140k swap/no-history ≈ 0.86M prompts at about 190/s.
- Risk: - Most likely failure: tau is near 0. 8B in-context learners are known to be insensitive to label corruption (Min et al. 2022; Wei et al. 2023), so the model may ignore star ratings entirely. The method dies, but the observation survives.
- The flip variant confounds a leniency × item interaction. The perm variant is the identification check; if flip works and perm does not, report it as leniency rather than taste.
- Amazon histories skew to 5 stars, so perm is often degenerate. Report the share of degenerate users; flip is used there.
- Selecting lambda on 400 dev users is noisy.
- Reviewers will cite ICCD (ACL'25) and CAD as the estimator. Novelty is about 5/10 and hinges on the item-preserving counterfactual plus the readout.
- On next-item, the liked/disliked annotation changes the prompt format relative to the base.
- Paper value: If POSITIVE, VCC is the flagship method behind the decomposition "the yes = a taste-attributable part + a taste-free part", which answers all six seeds with one estimator:
- S1: the correct-but-not-personal share;
- S2: errors concentrate where |tau| is small;
- S3: head share of top-10 under L vs s_lambda;
- S4: calibration of the tau-based score;
- S5: the popularity link lives in the taste-free part;
- S6: taste-contradicted vs prior-contradicted labels.

It can be dropped onto the 4 frozen next-item panels against the 8 baselines.

If NEGATIVE, it still gives a strong, citable observation ("demonstration shortcut in LLM recommenders: the yes is about the item, not you"), and it motivates #4.

### #2 SPJ: Prompt-level jackknife self-probes, i.e. per-user reliability from the user's own labelled history, used for cross-user serving and per-user LLM/CF stacking  (≈0.2 GPU-h)
- Rationale: Confidence magnitude cannot tell when the LLM is wrong: |logit| detects errors with AUROC 0.58, and the earlier verbalized gate correlated only 0.07 with per-user success.

An in-context recommender admits exact per-user validation at almost no cost: ask the same prompt about the user's own earlier rated items, whose labels are known, with no retraining. That gives a direct per-user reliability estimate in place of a confidence proxy.

This is where uncertainty is not rank-inert (H2):
- cross-user decisions (serve vs fall back to CF);
- per-user channel weights, which change within-user order whenever the LLM and CF orders differ.

Prior art: per-user algorithm selection and stacking (Ekstrand & Riedl RecSys'12; FWLS), and unsupervised leave-one-out over demonstrations (In-Context Calibration, NAACL'24, by the likely reviewer group). The delta is supervised per-user reliability inside a personalised LLM recommender, applied to serving.
- Protocol: 1) Users and prompt: the same fresh ML-1M and Video_Games dev/test users and the same Step-0 prompt as #1. The base arm is shared.

2) Probes per user:
- up to 20 events with rating != 3 immediately preceding the history window, so every probe is strictly earlier than every candidate;
- each probe is scored with the identical history block h_u, as a candidate prompt;
- label = 1[r >= 4];
- if fewer than 6 probes exist, fall back to leave-one-out within h_u (context h_u minus j).

3) Per-user statistics:
- probe AUC and log-loss for the LLM logit;
- the same for CF q(i), the leave-user-out shrunk item like-rate (m = 5 pseudo-ratings), on the same probe items;
- shrinkage r~ = (n+ n- AUC_u + kappa·mean AUC) / (n+ n- + kappa), with kappa from {5, 10, 20, 40} chosen on dev.

4) Uses:
- (A) Selective serving: serve the LLM ranking to users with r~_LLM above a coverage threshold, otherwise the CF ranking. Trace the risk–coverage curve of served UAUC against selection by user-mean |L| and against random selection.
- (B) Per-user stacking: s = w_u·z_u(L) + (1 - w_u)·z_u(q), with w_u = sigmoid(b0 + b1(r~_LLM - r~_CF)). (b0, b1) is grid-fit on dev. Compare with global stacking (constant w).
- (C) Per-user Platt fitted on the probes. Report global AUC and ECE (S4), recognising this is rank-inert within a user.

5) Decision:
- M: Spearman(r~_LLM, test UAUC_u) >= 0.10 with user-bootstrap CI excluding 0, in at least 1 rated domain.
- P1: at 50% coverage, mean served UAUC (r~-selected) minus mean served UAUC (|L|-selected) >= +0.01 with CI > 0.
- P2: dUAUC(per-user stacking - global stacking) CI > 0 in at least 1 domain.
- NEGATIVE: the M CI covers 0 in both domains. Per-user reliability is then not estimable from 20 probes; report as such.

6) Also report:
- head share of the served lists (S3);
- probe self-inconsistency (users whose probe log-loss is in the top decile) as a noisy-user flag, detection only (S6).

About 40k extra prompts.
- Risk: - Measurement noise. With 10–20 probes the reliability is about 0.2 and test UAUC about 0.3, so the observed correlation clears 0.10 only if the true between-user SD of AUC is at least about 0.06.
- Temporal shift between earlier probes and later candidates.
- CF dominates on rated panels (0.76 vs 0.59), so stacking gains may be small (expected +0.003 to +0.01).
- Moderate novelty (about 4.5/10): FWLS, Ekstrand & Riedl, and In-Context Calibration by the POSTECH group (likely reviewers).
- Probe availability is thin on Amazon users with short histories.
- Paper value: Gives a direct answer to S1/S2 at the user level ("for whom is the LLM's yes trustworthy?") with a validated estimator instead of a confidence proxy.

It yields a legitimate, non-rank-inert use of uncertainty: cross-user serving and fallback plus per-user channel weights. It is cheap enough to run on every full-grid dataset and both backbones.

It also supplies a calibration audit on labelled per-user probes (S4) and a noisy-user detector (S6 appendix). It combines naturally with #1, with |tau| as an item-level readout and r~ as a user-level one.

### #3 FMD: Familiarity-matched foil duels, i.e. elicit preference against an equally familiar item so the candidate's familiarity cancels by design  (≈1.2 GPU-h)
- Rationale: Pointwise yes is dominated by item familiarity: pi correlates with log-popularity at 0.43, and the like-logit correlates with quality at only 0.15.

A forced comparison between the candidate and a foil with the same LLM familiarity (matched on the no-history logit pi_0) cancels the candidate's familiarity by construction. Unlike swap or no-history subtraction, it keeps the user's full context, topic included.

The pairwise format is independently known to extract more personalisation than pointwise (PerRecBench; PRP). Order-swap inconsistency gives a natural "is it sure" readout.

Prior art: anchor-comparison pointwise ranking (GCCP, SIGIR'25) and AlpacaEval-style reference comparison. The new part is per-candidate familiarity matching in personalised recommendation.
- Protocol: 1) Compute pi_0(i), the no-history like-logit (one prompt per item):
- for every test candidate;
- for a foil pool: all domain items appearing in any built panel, excluding the user's history and candidates.

2) For each (u, i), pick M = 2 foils with nearest pi_0. For Amazon, restrict to the same top-level category. Fix foil RNG seeds.

3) Duel prompt: same history block, then "Item A: <title/desc>  Item B: <title/desc>  Which item would this user like more? Answer with only A or B." Read the A/B token log-probs by token id, as in the Yes/No reader. Score both orders.

4) Estimators:
- d = 0.5·[(lA - lB | i first) - (lA - lB | i second)];
- s_FMD = mean over foils of d;
- order-inconsistency c = share of foils whose two orders disagree in sign.

5) Controls:
- random unmatched foils (M = 2, both orders);
- an equal-compute pointwise ensemble (base + 3 paraphrases averaged).

6) Data: test users only, since there are no tuning knobs:
- ML-1M test, 1,283 users;
- Video_Games test;
- toys next-item, 500 test events, matched foils only.

7) Decision:
- M: mean within-user Spearman(s_FMD, pi_0) <= 0.5 × Spearman(L, pi_0).
- POSITIVE: dUAUC(FMD - pointwise ensemble) CI > 0 in at least 1 rated domain, AND point dUAUC(matched - random foils) >= 0, AND next-item dNDCG@10(FMD - base) >= -0.005.
- Readout: AUROC of c for base-decision errors vs |L|.
- Report head share and corr with log-pop.

About 0.6M prompts, which are longer than pointwise ones.
- Risk: - Novelty about 4.5/10: pairwise prompting and anchor comparison are known (PRP, PerRecBench, GCCP). Reviewers may call it PRP with a twist.
- Foil-induced variance with M = 2.
- Position bias at 8B can be large; the two-order average handles only the linear part.
- Matching on pi_0 may also match on quality or topic and remove real signal.
- Next-item cost is 4 × 101 prompts per event.
- A/B token reading needs the same id-set logging as Yes/No.
- Paper value: A familiarity-controlled elicitation that directly tests S5 by design: does confidence still track popularity once familiarity is held fixed?

It also gives an order-consistency "is it sure" signal (S1/S2) and a static echo-chamber readout (S3 head share).

If POSITIVE it is a second, complementary method to VCC. VCC controls the evidence side and FMD the item side, which together make a clean two-counterfactual story for one paper spine.

### #4 TC-LoRA: Taste-contrastive tuning, i.e. train the yes/no recommender so its logit must move when the history's rating mapping is scrambled (conditional on #1's mechanism check)  (≈1.8 GPU-h)
- Rationale: Zero-shot rated UAUC (0.59/0.54) is far below item-mean CF (0.76/0.71). SIGIR-grade rated results will therefore have to come from the fine-tuned regime, where plain BCE may inherit the same demonstration shortcut.

An auxiliary BCE on L_theta(h,i) - L_theta(perm(h), i) directly rewards using the item-to-rating mapping, with the item set and leniency held fixed.

Closest prior: CFT (arXiv 2410.22809), which fits labels on L(h) - L(no history). Its counterfactual deletes the items, so it rewards the topic as well as taste. The delta is the item-preserving counterfactual.
- Protocol: Pre-condition: #1's mechanism check M holds, i.e. tau is non-degenerate at zero-shot. Otherwise skip.

1) Training data:
- 2,000 (h, i, y) examples built with build_rated_panels from users disjoint from every pilot user;
- ML-1M users outside the eligible pool, plus Video_Games users outside dev/test;
- assert disjointness.

2) Setup: Qwen3-8B, 4-bit QLoRA, r = 16, alpha = 32, lr 2e-4, 1 epoch, max length 3072, shared prompting helper.

3) Arms:
- A: BCE on 4,000 examples, which matches the forward passes of B;
- B: BCE(sigmoid(L_theta(h,i)), y) + mu·BCE(sigmoid(L_theta(h,i) - L_theta(c_P(h), i) + b), y), with mu = 1 fixed a priori, b a learnable scalar, and a fresh rating permutation each step.

4) Seeds {0, 1} per arm, 4 runs, about 15 min each.

5) Evaluation on ML-1M test users (fresh): UAUC of L_theta, within-user SD of tau_theta, and ECE after dev-Platt.

6) Go/no-go only, no claims:
- go if the mean dUAUC(B - A) >= +0.02;
- AND both seeds are positive;
- AND the gap exceeds |seed0 - seed1| within arm A;
- AND SD_within(tau_theta) in B >= 1.5 × A.
- Otherwise drop.

A later full study needs at least 3 seeds and a gain > 2 sigma_seed (sigma ≈ 1.5 pt in the sibling project).
- Risk: - Two seeds cannot establish an effect. The pilot is go/no-go only.
- The model may learn to copy the user's rating level (leniency) rather than taste; permutations hold leniency fixed, so check with the perm-only tau.
- CFT is a close prior, so novelty is about 4.5/10.
- The budget is tight on a 24 GB card: two forward passes per example with gradient checkpointing.
- Training panels must exclude the Lumen valid/test selected_users (amendment D).
- Paper value: The training-time counterpart of #1, giving one mechanism across both the zero-shot and fine-tuned regimes. That is the minimum a SIGIR reviewer will ask for, given the zero-shot rated UAUC.

It answers whether fine-tuning re-learns the familiarity shortcut (S5) or can be pushed toward taste.

It creates a principled hook for S6: taste-contradicted training labels can be down-weighted, but not pruned, in a later at-least-5-seed study.

### #5 UGF: Uncertainty-gated LLM×CF fusion, i.e. let the LLM's weight depend on item familiarity, taste support and per-user reliability (honesty anchor; CPU-only)  (≈0 GPU-h)
- Rationale: Item-mean CF beats the zero-shot LLM by about 0.17 UAUC on rated panels. A rated-panel paper therefore has to report whether the LLM adds anything beyond item quality.

The overconfidence (Platt slope 0.062) means a naive fusion is dominated by the LLM, so it must be standardised.

Making the LLM's weight a function of familiarity pi_0, of |tau| from #1 and of r~_u from #2 is item- and user-dependent, so it changes within-user rankings. It also directly answers "where should yes/no confidence be trusted?"

Novelty is low (FWLS-style meta-features), but this is the mandatory anchor for any honest rated-panel claim.
- Protocol: 1) Inputs from #1 and #2 on the same dev/test users:
- z_u(L), the within-user standardised logit;
- q(i), the leave-user-out shrunk like-rate (m = 5) excluding u;
- phi(i) = pi_0(i), centred within user;
- |tau_P|;
- r~_u.

2) Fit an L2-logistic regression on dev users: logit P = b0 + b1·logit q + (b2 + b3·phi + b4·|tau| + b5·r~_u)·z_u(L), with the L2 strength chosen by 5-fold CV over dev users.

3) Baselines:
- item-mean only;
- global fusion (b3 = b4 = b5 = 0);
- LLM only.

4) Test metrics: UAUC, global AUC, ECE, and head share among top-ranked items.

5) Decision:
- Always report dUAUC(global fusion - item-mean) with CI.
- POSITIVE if dUAUC(UGF - global fusion) CI > 0 in at least 1 rated domain.
- If the CI of dUAUC(global fusion - item-mean) covers 0, record that zero-shot LLM confidence adds nothing beyond item quality on rated panels, and move every rated-panel method claim to the fine-tuned regime (#4).
- Risk: - Low novelty: stacking with meta-features (FWLS 2009) and CoLLM-style CF integration.
- The expected increment over global fusion is at most about 0.01.
- Fitting on 400 dev users risks overfitting with 6 coefficients; mitigated by L2 and CV.
- Its outcome can deflate the zero-shot rated story. That is a feature for honesty, not a reason to skip it.
- Paper value: Gives the reviewer-proof anchor table (LLM vs CF vs fusion) that any rated-panel claim needs.

Answers S4/S5 operationally: the LLM's yes is worth trusting on items with taste support and low familiarity, not on familiar items with no taste support.

Costs no GPU time and reuses #1/#2 outputs. It determines whether the paper's rated-panel contribution stays zero-shot or moves to fine-tuning.

## Citations
- Peng, Ding, Ouyang, Fang, Yuan, Tao. Enhancing Input-Label Mapping in In-Context Learning with Contrastive Decoding (ICCD). ACL 2025 (per arXiv comment; ACL Anthology 2025.acl-short.77). arXiv 2502.13738 [verified this session]
- Jang, Jang, Kweon, Jeon, Yu. Rectifying Demonstration Shortcut in In-Context Learning (In-Context Calibration). NAACL 2024 (long). arXiv 2403.09488 [verified this session]
- Shi, Han, Lewis, Tsvetkov, Zettlemoyer, Yih. Trusting Your Evidence: Hallucinate Less with Context-aware Decoding. NAACL 2024 (short). arXiv 2305.14739 [verified this session]
- Li et al. Contrastive Decoding: Open-ended Text Generation as Optimization. ACL 2023. arXiv 2210.15097 [from memory, not re-verified this session]
- Min et al. Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? EMNLP 2022. arXiv 2202.12837 [from memory, not re-verified this session]
- Wei et al. Larger Language Models Do In-Context Learning Differently. arXiv 2303.03846, 2023 [from memory, not re-verified this session]
- Zhang, You, Bai, et al. Causality-Enhanced Behavior Sequence Modeling in LLMs for Personalized Recommendation (CFT; counterfactual = history set to None). arXiv 2410.22809 [content verified; venue unverified]
- Li, Yang, Liu, Wu, Xia, Dai. Neutralizing Popularity Bias in LLM-based Recommendation via Counterfactual Reasoning Guidelines (NPRec). CIKM 2026. arXiv 2503.08051 [verified this session]
- Tan, Zeng, Zeng, Wu, Liu, Mo, Jiang. Can LLMs Understand Preferences in Personalized Recommendation? (PerRecBench). arXiv 2501.13391 [content verified; venue unverified]
- Qin et al. Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting (PRP). Findings of NAACL 2024. arXiv 2306.17563 [verified this session]
- Long, Li, Xu, Tang, Wang. Precise Zero-Shot Pointwise Ranking with LLMs through Post-Aggregated Global Context Information (GCCP). SIGIR 2025. arXiv 2506.10859 [verified this session]
- Sill, Takacs, Mackey, Lin. Feature-Weighted Linear Stacking. arXiv 0911.0460, 2009 [verified this session]
- Ekstrand, Riedl. When Recommenders Fail: Predicting Recommender Failure for Algorithm Selection and Combination. RecSys 2012. doi:10.1145/2365952.2366002 [verified this session]
- Li, Ye, Jian, Guo, Ma, Ai, Zhang. Are Recommenders Self-Aware? Label-Free Recommendation Performance Estimation via Model Uncertainty (LiDu). arXiv 2507.23208 [verified this session]
- Bu, Jung, Kang, Kim. Personalized LLM Decoding via Contrasting Personal Preference (CoPe). EMNLP 2025. arXiv 2506.12109 [verified this session]
- Liu et al. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment (probability-weighted score readout). EMNLP 2023. arXiv 2303.16634 [from memory, not re-verified this session]
- Kang et al. Do LLMs Understand User Preferences? Evaluating LLMs on User Rating Prediction. arXiv 2305.06474 [verified in project docs]
- Hou et al. Large Language Models are Zero-Shot Rankers for Recommender Systems. ECIR 2024. arXiv 2305.08845 [verified in project dossier]
- Burns et al. Discovering Latent Knowledge in Language Models Without Supervision (CCS). ICLR 2023. arXiv 2212.03827 [verified in project dossier]
- Zhao et al. Calibrate Before Use. ICML 2021. arXiv 2102.09690; Zhou et al. Batch Calibration. ICLR 2024. arXiv 2309.17249; Holtzman et al. Surface Form Competition / PMI-DC. EMNLP 2021. arXiv 2104.08315; Fei et al. Domain-context calibration. ACL 2023. arXiv 2305.19148 [verified in project dossier]
- Yes-no bias of LLMs reflects answer order and wording (crossed symmetrization). arXiv 2607.05552 [verified in project dossier]
- Li, Xiong, Wu, Hooi. ConfTuner. NeurIPS 2025. arXiv 2508.18847; Xiong et al. Proximity-Informed Calibration (ProCal). NeurIPS 2023. arXiv 2306.04590; Xiong et al. Can LLMs Express Their Uncertainty? ICLR 2024. arXiv 2306.13063 [verified in project docs]
- Kweon, Jang, Kang, Yu. UQRec: Uncertainty Quantification and Decomposition for LLM-based Recommendation. WWW 2025. arXiv 2501.17630; Fan et al. UGR. KDD 2026. arXiv 2602.11719; Yan et al. EviRank. arXiv 2606.04727 (venue unverified); Lee et al. KnowSA_CKP. SIGIR 2026. arXiv 2604.07825 [verified in project docs]
- Bao et al. TALLRec. RecSys 2023. arXiv 2305.00447; Zhang et al. CoLLM. TKDE 2025. arXiv 2310.19488 [verified in project docs]
- RankSteer: Can Pointwise LLM Rankers Be Calibrated at the Representation Level? arXiv 2602.03422; Ravikumar. Do LLM Recommenders Know When They're Hallucinating? CIKM 2026 short. arXiv 2608.10008 [verified in project docs]
- Project evidence: D:/Research/Lumen/docs/sigir/PILOT_LOG.md (2026-10-02 entries: Pilot 1 gate fail, gate-failure diagnosis, Pilot 2 NULL); D:/Research/Lumen/outputs/confrec_pilot/pilot1/{ml1m_rated,toys_rated,sports_next_1k}.pilot_mirror.json, decision.json, ml1m_rated.diag_baselines.json, toys_rated.diag_baselines.json; D:/Research/Lumen/src/confrec/{prompting,build_rated_panels,pilot_mirror,diag_rated_baselines}.py