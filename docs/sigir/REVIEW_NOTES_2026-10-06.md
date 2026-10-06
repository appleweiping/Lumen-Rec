# Independent same-family (Opus) reviews of the filled draft, 2026-10-06 (provisional: same model family as the executor)

Three read-only reviewers were launched on `Paper/sigir2027/filled/main.pdf` (12 pages with references; rated panels, Qwen3-8B/Llama results, FT-C/FT-Q and the next-item
audits of Sports and Toys filled; home/tools audits, Z2, S6, the method slot, stage-3, FT-Q on Video_Games/Sports and the serving controls pending). Their reports are model output,
not user authority; every item below was checked by me against the result files where it concerns a number. This file keeps the findings and the decisions so that they survive
context compaction. (R-C, the methodology reviewer, is recorded below when it arrives.)

## R-A. Senior PC-member review (lens: SIGIR full-paper committee)

Recommendation as the draft stood: **weak reject**; with the pending items landing reasonably: **borderline**, **weak accept** with changes 1-4 and 6. Scores (1-5): originality 3, significance 2, soundness 3,
clarity 2, reproducibility 4. About 40 table values were recomputed from the JSON files and matched.

Weaknesses (W) and what I verified:
- W1 no positive control outside ML-1M: the MF personal-residual UAUC is 0.50-0.52 on the Amazon panels (G_CF 0.0015 / -0.0020 / 0.0024), so the LLM null there cannot be told from a panel without learnable personal signal; ML-1M TEST has 366 users. VERIFIED (Table 3B).
- W2 the comparator m(i) is not information-matched (full rating log: 1.0M / 16.3M / 4.6M / 19.6M events); against m_T Video_Games is a tie (+0.0003), Llama ML-1M equals m (+0.001). VERIFIED.
- W3 scope: the decomposition exists only for rated like/dislike panels; next-item (where yes/no LLMs are deployed) has no swap-prior decomposition and no adapter.
- W4 weak models: zero-shot Amazon UAUC 0.493-0.527; one epoch, r = 16; the under-training objection (FT-L, addendum 9) is registered but unanswered.
- W5 estimator dependence and selective reporting: Sports zero-shot G is confirmed NEGATIVE by E-D (-0.0095) but +0.0016 by E-W; Llama ML-1M fine-tuned G is 'not replicated' by E-D (0.0071, p = 0.064) but 0.0098 [0.0031, 0.0173] by E-W (the size of Qwen's); P1 is computed on every panel
  (Sports +0.0118 [0.0003, 0.0230], p = 0.041, P1_HOLDS; Toys -0.003; Video_Games -0.004; Llama ML-1M +0.004, p = .38) but Table 3 prints '--' outside ML-1M. VERIFIED from grid/*.json (P1 blocks) and Table 3.
- W6 absence claims without equivalence tests: upper 95% bounds reach 0.011 (Sports fine-tuned G), 0.012 (Llama Toys zero-shot G), 0.017 (Sports G_LLM|CF).
- W7 actionability rests on a random-signal control only (addenda 11-12 pending).
- W8 registration: 12 addenda over five days, several after outcomes; deviation rows 21-22 missing from the paper's table (added to DEVIATIONS.md 2026-10-06); same-family reviews; the machinery costs about 1.5 pages.
- W9 presentation: internal codes on every page, no figures, tables shrunk to unreadability, text spills onto p. 10; the '+- half-width' format hides skewed intervals (FT-Q Toys R 1.335 +- 1.186 vs the file's [0.674, 3.046]: VERIFIED); Table 7's Brier 0.972 is the 101-way list Brier (not the Brier of Section 5); the rated panels use V1 ('4 stars or higher') but Section 3 defines q as 'Would this user like...'.
- W10 ELMRec's NDCG@10 (0.040-0.049) is random level; 'less head exposure than eight published recommenders' needs their accuracy beside it.

Claims the tables do not support (abstract/intro/conclusion): below the item mean 'on all four' (against m_T Video_Games ties, Llama ML-1M equals m, ML-1M gap exploratory); 'not replicated with Llama' (estimator-dependent); '85-92% of the gain' (ML-1M only; Toys R_Q 1.34 [0.67, 3.05]);
'zero-shot Sports -0.009 confirmed negative' (+0.0016 under E-W); P1 omits Sports/Toys/Games; 'knockout INDETERMINATE, so not familiarity-linked' (an INDETERMINATE label cannot support a negative; Toys zero-shot head-tail drop 0.40 [0.27, 0.52] vs placebo 0.14);
'registered before any confirmatory result was analysed' (the admission and wording rules and the dagger endpoints came after the ML-1M report); 'what the confidence knows is mostly the item' (consumed-item panels only) and 'each route was tested against its own control' (not yet true).

Top changes (R-A ranking): (1) positive control and power (content-similarity personal reference G_content, planted-signal power curve, equivalence at delta = 0.01); (2) re-anchor to m_T and a TRAIN-only item mean, state the data asymmetry; (3) swap-prior decomposition of next-item logits
(about 5 GPU-h per domain, 1,000 events x 101 candidates x 8 donors); (4) finish routing and cheap-signal controls (4 domains + Z2 Llama, oracle router, risk-coverage figure); (5) training strength (FT-L, 5-10x TRAIN users on ML-1M and Toys); (6) rewrite (plain endpoint names, 2-3 figures, [lo, hi] for ratios, fix Brier/V1, deviation rows 21-22);
(7) print baseline accuracy beside exposure; (8) demote MIRROR, the knockout, S6 and the method slot unless one is admitted. Cuts suggested for 9 pages: move the registration machinery (Table 1, most of Table 9, hashes) to the artefact; drop the knockout and MIRROR rows; Table 10 to one sentence; Proposition 3.2 to a remark;
Table 2 to a list; keep only Delta_head in Table 5; move Tables 11-14 to the artefact (add a Llama column to Table 3's headline rows); drop Table 3C oracle rows and Table 4B.

## R-B. Positioning and novelty review (lens: uncertainty-aware LLM4Rec literature)

Estimated acceptance probability about 20% (15-30%); about 30-35% if a controlled, replicated positive use (D1/D2 below) succeeds; 10-15% if every route is null. The most novel claim: the item-only TRAINING controls (FT-C R = 0.92, FT-Q R = 0.85) with G_LLM|CF <= 0.005: a causal statement about what TALLRec-style SFT learns.
The claim most likely to be dismissed as known: 'LLM recommenders follow item quality/popularity' (PerRecBench, rating-prediction studies, Zero-Shot Rankers, Flower/SPRec). Sharpened contribution statement (adopted as the basis of the abstract and introduction): 'With training controls that remove all user-item evidence, we show that LoRA-tuned Yes/No LLM recommenders
learn item-level label rates rather than personal preference ... Since no per-user recalibration can reorder a user's candidates, we test the decisions where such confidence can still act (cross-user serving, exposure, item-dependent corrections, data selection) each against a matched control, and find [X].'

Missing or uncited works (all verified to exist by the reviewer; to be verified again before they enter references.bib): GUIDER (AAAI 2026); Disentangling User Interest and Conformity for Recommendation with Causal Embedding (WWW 2021; the decomposition is the biased-MF baseline predictor applied to log-odds: say so); 'Large Language Models Must Be Taught to Know What They Don't Know' (NeurIPS 2024) and ConfTuner (NeurIPS 2025) (both in the bib, uncited);
Clark et al., 'Don't Take the Easy Way Out' (EMNLP 2019: the prior-offset LoRA is a product of experts with a bias-only expert); Dataset Cartography (EMNLP 2020), 'Beyond neural scaling laws' (NeurIPS 2022), SelectIT (NeurIPS 2024) for S6; 'Item-side Fairness of LLM-based Recommendation System' (WWW 2024) for S3; 'Uncertainty-Calibrated Recommendations for Low-Active Users' (KDD 2026); kweon2022calibrated (in the bib, uncited).
Misdescriptions to fix: MIRROR/CCS (CCS is an unsupervised probe on hidden states; say 'the log-odds analogue of CCS's symmetrised inference rule'); UNIT (it deletes claims inside responses for truthfulness; P2 deletes whole examples: drop 'in the spirit of UNIT'); Lost in Sequence (order sensitivity, not 'under-use of the history'); GORACS (state its finding that loss/gradient scores fall below random); 'None asks how much of the confidence is a user-independent item prior' holds at the logit level only.
Constructive directions (D1 prior-aware group-conditional serving; D2 prior-cancelling LoRA with within-item pairs and a within-item-permutation falsification arm; D3 prior-corrected exposure; D4 decomposition-informed pruning) cost 130-210 of the 600 GPU-hours. Gaps against strong papers of this kind: one uncertainty estimator (a seed-ensemble disagreement estimate is free from the three adapters), two 8B backbones, one recipe, power (366 / 943 / 820 / 997 TEST users),
no pruning baselines (DEALRec, GORACS, GraNd/EL2N), no figures, unreadable shrunk tables, text before the references over the limit.

## Decisions (executor, 2026-10-06)

| item | decision | action |
|---|---|---|
| claims not supported (R-A) | accept all eight | rewrite abstract, introduction, conclusion, Findings text: state registered and information-matched comparisons, estimator dependence, P1 on all panels, intervals for ratios, 'INDETERMINATE' without a negative, registration wording |
| positive control / power (R-A 1) | accept | exploratory CPU module (addendum 14, to be registered before running): planted-signal power curve of G, upper bounds and equivalence at 0.01, content-based personal reference, TRAIN-only item mean; FT-B/FT-S/FT-N (addendum 13, in implementation) is the GPU probe of objective versus model |
| re-anchor (R-A 2) | accept | m_T and the TRAIN-only mean become primary figure/table rows beside m and MF |
| next-item decomposition (R-A 3) | deferred | decide after FT-B and the GPU queue (about 5 GPU-h per domain) |
| routing and cheap signals (R-A 4, R-B D1) | in progress | addenda 11-12, code by an implementer, run on 4 domains and Z2 |
| training strength (R-A 5) | partly | FT-L (queued); more TRAIN users is possible on Amazon only (ML-1M has about 3,200 eligible users) and Amazon has no personal signal: not run unless GPU is idle |
| rewrite and figures (R-A 6, R-B) | accept | figure generator in progress; editor pass 3 restructures to about 9 pages |
| baseline accuracy beside exposure (R-A 7) | accept | add the eight NDCG@10 values and the positives' head share |
| demote MIRROR / knockout / S6 / slot (R-A 8) | accept | one compact paragraph or appendix row each; S6 and the slot stay only as small tables if complete |
| missing citations and misdescriptions (R-B) | accept | citation agent with lookups; fix the five misdescriptions |
| CF-augmented Yes/No positive control, pruning baselines (DEALRec, GORACS) | decline | stated as limitations |
