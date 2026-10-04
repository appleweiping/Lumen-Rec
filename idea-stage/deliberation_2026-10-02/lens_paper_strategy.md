# SIGIR area chair / paper strategist: which honest SIGIR 2027 full paper can be reached by 2027-01-21 on one 4090, given that Pilot 1 failed its token-channel gate

**Headline.** Retire MIRROR as the flagship. The strongest honest paper is a pre-registered audit of yes/no LLM-recommender confidence that compares the zero-shot and fine-tuned regimes. Its organizing lens is to split confidence into an item prior and personal evidence. A short invariance lemma, tested at full scale, says where uncertainty can change outcomes and where it cannot. This framing holds whatever the gate does, because the zero-shot gate outcome becomes Finding 1 rather than a blocker, and the main object is the TALLRec-style LoRA regime. Expected acceptance is about 22-30%. MIRROR-as-method is about 3%. A zero-shot-only audit is 6-10%.

## Analysis
0. BOTTOM LINE

Pilot 1 did not just "fail a gate". It reproduced a known property of zero-shot yes/no LLM recommenders: on explicit feedback they are close to chance.
- TALLRec (RecSys'23) reports un-tuned in-context LLMs at about AUC 0.5.
- Kang et al. (arXiv 2305.06474) report that "zero-shot LLMs lag behind traditional recommender models", while fine-tuned LLMs catch up.

The diagnostics are not decision inputs and must be re-validated. Read as an AC, they sketch one coherent picture:
- the zero-shot channel is weak on preference-among-consumed-items;
- the signal it does carry is mostly a user-independent, popularity-flavoured item prior;
- its overconfidence is a single global temperature;
- the mirrored correction is not a robust lever.

A method paper built on that channel is a poor bet. A measurement paper that treats it as Finding 1 and moves the main analysis to the fine-tuned regime is gate-robust and is the best expected-value use of the 15.5 weeks left. The deadline is 2027-01-21, with the abstract due 01-14; SIGIR 2027 CFP details are still placeholders as of today.

1. WHAT THE PILOT ACTUALLY TELLS AN AC
All numbers in this section are exploratory and from burned users.

(a) The task is not hard; the channel is weak.
- Leave-user-out item mean rating gets UAUC 0.760 on ML-1M and 0.712 on Toys.
- Item popularity gets 0.637 and 0.535.
- The LLM like-logit gets 0.587 and 0.540, so on ML-1M it is *below popularity*.
- Its Spearman correlation with item mean rating is only 0.15 / 0.07.
- Any paper that reports zero-shot P(Yes) on rated panels without these non-personalized references will be rejected by a reviewer who computes them, as in the Dacrema et al. RecSys'19 genre of critique. They must appear in every table.

(b) The weakness is task-specific.
- On the Sports next-item panel (1+100, popularity-sampled negatives), token P(Yes) "next" has UAUC 0.719 and NDCG@10 0.209 on 1k events.
- C-CRP verbalized gets 0.231 on the same events.
- The strongest official baseline on the full 10k panel is LLMEmb at 0.1795. This must be recomputed on the full 10k before any statement.
- So the LLM is a strong candidate-relevance matcher but a poor judge of liked vs disliked among consumed items. That contrast is itself reportable.

(c) MIRROR is dominated.
- ΔUAUC against the placebo is +0.019 on ML-1M, −0.007 on Toys and −0.018 on Sports.
- On Sports, NDCG@10 against raw is −0.031 [−0.043, −0.020], about 6x the registered 0.005 tolerance.
- The one novelty premise the dossier allowed is a *popularity-linked* item-conditional acquiescence. It is absent at the pair level in both domains: corr(a, log-pop) is 0.007 and 0.074.
- The within-user spread of a(u,i) is large (SD 2.3-4.4 logits) but behaves like noise rather than a familiarity term.
- The ML-1M gain is fully explained by "the dislike prompt is a second, partly independent noisy reading": UAUC of −dislike is 0.592, about the same as like. On Toys, −dislike is only 0.52, and the gain vanishes.
- For MIRROR to become a flagship, three independent things would have to flip on fresh users after a prompt change selected on burned users. I put that at ≤10%.

(d) Evidence collapses: like − π gives UAUC 0.507 / 0.525.
- The discriminative part of zero-shot P(Yes) is almost entirely item prior π(i), which tracks popularity (0.43 on ML-1M, 0.28 on Toys).
- Caveat that must be handled before this becomes a claim: π uses K=8 donors, so subtracting a noisy estimate attenuates. A split-half donor reliability (4+4) and a disattenuated "personal share" are required.

(e) Overconfidence is a global temperature.
- ECE is 0.30 and the Platt slope is 0.062.
- After Platt scaling, ECE is 0.016.
- This is rank-inert and fixable with one scalar. It is not a contribution on its own (Tian et al. EMNLP'23; Xiong et al. ICLR'24). It does support a useful sentence: "calibration is not the bottleneck; discrimination and what the confidence encodes are."

(f) Error detection by |logit| is weak (AUROC 0.58).

2. CANDIDATE FRAMINGS
Scores use a generic 1-5 scale: 1 = strong reject, 3 = borderline, 5 = strong accept. A SIGIR accept usually needs a mean of about 3.5 plus a champion. The base rate is about 20%; SIGIR'25 accepted 1,134 of 5,797 submissions across tracks.

A. MIRROR-as-method, conditional on the fixed prompt rescuing it
- Contribution: "LLM recommenders say yes to items they recognize; ask both ways."
- Novelty dossier score was 5/10. The estimator is CCS / crossed symmetrization (arXiv 2607.05552), and that work found per-item residuals idiosyncratic.
- Probability of being viable: prompt fix passes the gate on fresh users (about 0.55-0.65) × MIRROR beats the placebo with CIs on ≥3/4 datasets × 2 backbones, with no NDCG loss on 10k×101 (about 0.10-0.15). That gives about 7-9%.
- If viable, scores look like {2,3,3,4}, with acceptance about 25-30% (expect "this is CCS" and "the gain is a 2-prompt ensemble").
- Unconditional acceptance: about 2-4%.
- Minimal experiment set:
  - prompt fix plus gate;
  - 4 rated datasets × 2 backbones × {like, dislike, paraphrase, PMI, swap-prior, batch-calibration};
  - 4 next-item panels × 10k × 3 questions × 2 backbones (about 24M prompts, about 36 GPU-h);
  - mirror tuning: 4 × 2 × 3 seeds;
  - pseudonym identification of a(i);
  - about 130 GPU-h in total.
- Gate robustness: none. The paper does not exist if the gate or the method fails, and 6-8 weeks are lost.

B. Zero-shot-only measurement/audit answering the six questions
- Gate-robust in a weak sense: the failure becomes the finding.
- Reviewer reaction: "an audit of an untuned 8B whose confidence is beaten by item mean rating; overconfidence is known; what do I do with this?"
- Closest precedent: Ravikumar, an audit of zero-shot verbalized confidence stratified by popularity, which landed as a CIKM'26 *short* paper.
- Scores about {2,2,3,3}; acceptance 6-10%.
- About 30 GPU-h.

C. RECOMMENDED: regime-comparative, pre-registered audit with a decomposition lens and an actionability lemma
- Working title: "What Does a Yes/No LLM Recommender's Confidence Know? Item Priors, Personal Evidence, and When Uncertainty Is Actionable."
- Object: pointwise yes/no LLM recommenders, the family with probability semantics (TALLRec, CoLLM, BinLLM). Calibration of this family against preference labels has never been reported (LIT doc §B, verified PDF text search).
- Factors (the structure mirrors Xiong et al. ICLR'24's prompt × sampling × aggregation framework):
  - regime: zero-shot vs LoRA-tuned;
  - channel: token P(Yes), mirrored, paraphrase, rating-token expectation, verbalized;
  - label validity: rated, MAR on KuaiRec, list-normalized next-item;
  - use: calibration, failure prediction, cross-user decisions, exposure, pruning.
- Scores:
  - typical {3,3,3,4}, mean about 3.2;
  - downside {2,3,3,3}, from a "no new method" reviewer;
  - upside {3,4,4,4}, if the pseudonym knockout and KuaiRec both produce clean positives.
- Acceptance about 22-30% (point estimate 25%), rising to about 35% if the nested method slot (below) survives.
- Gate robustness: full.
  - If the fixed prompt passes, the zero-shot regime is a valid object, and prompt fragility (original vs fixed) is a reported sensitivity, a known LLM-judge phenomenon (Thomas et al. SIGIR'24).
  - If it fails, Finding 1 is "zero-shot yes/no confidence fails a minimal adequacy bar on explicit feedback and is dominated by the item prior". The fine-tuned regime, which does not depend on the zero-shot prompt, carries the decomposition and actionability results.
- About 220-280 GPU-h. This fits: one 4090 over ~10 experiment weeks gives roughly 1,000+ GPU-h at realistic utilization, so the binding constraints are calendar time and writing, not compute.

D. Different-mechanism method: confidence for cross-user decisions
- Examples: prior-residualized selective serving, per-user offset recalibration for thresholding, exposure-aware gating.
- It escapes N1 legitimately and reuses the next-item channel, which passed its half of the gate.
- Novelty is thin:
  - UGR (KDD'26) owns user rejection and truncation;
  - PerK (WWW'24) and Kweon (AAAI'22) own calibrated probabilities for personalized K;
  - Zou et al. (KDD'26 ADS) own uncertainty-differentiated cross-user policies;
  - Jones et al. (ICLR'21) predict the niche-user disparity.
- Selective serving already beat random in Lumen (+0.036-0.083 NDCG@10 at 50% coverage), so the gain is near-certain but expected.
- Scores {2,3,3,3}; acceptance 12-18%.
- About 40-60 GPU-h.
- Best used as the "so-what" section of C, not as a standalone paper.

E. Fine-tuned-regime method paper
- Candidates: prior-offset (residual) LoRA, mirror tuning, name-dropout, IPS-weighted Brier on KuaiRec.
- Must beat TALLRec-style SFT, post-hoc stacking, ConfTuner-style Brier SFT, and SPRec/Flower-style debiasing, by more than 2σ_seed on ≥3/4 datasets × 2 backbones.
- Lumen's record is N2/N3 nulls with σ_seed ≈ 1.5 pt, so P(success) is about 20%.
- If it succeeds, scores look like {3,3,4,4} and acceptance is about 40%. Standalone unconditional acceptance is about 10%.
- About 150-200 GPU-h.
- Gate-robust, because it never touches the zero-shot prompt.
- Best nested inside C as one pre-registered slot with a hard kill date. That adds about +3-5 points of expected acceptance at about 30 GPU-h.

F. Hedge outside the full-paper track
- A negative-results/reproducibility write-up: "uncertainty does not improve LLM reranking", built from N1-N3, MIRROR and pruning.
- Only if SIGIR 2027 announces such a track; that is not verified, since the CFP is a placeholder. A CIKM/RecSys fallback is the other route.

3. THE RECOMMENDED PAPER (C) IN DETAIL

Contribution statement. This is pre-results wording: direction-free phrasing is mandatory until confirmatory data exist.

(C1) Protocol.
- The first pre-registered confidence audit of pointwise yes/no LLM recommenders against labels that match the question:
  - explicit like/dislike on ML-1M, Amazon Toys, Video_Games and Sports, using all eligible *fresh* users (ML-1M has about 1,683 fresh eligible users left after the 1,500 burned in the pilot; state this as a dataset limit, not a toy choice);
  - fully observed MAR labels (KuaiRec small matrix);
  - list-normalized next-item targets on 4 frozen Amazon panels (10k users × 101 candidates).
- Qwen3-8B and Llama-3.1-8B, zero-shot and LoRA (3 seeds).
- Item-mean, popularity and MF/LightGCN references in every table.
- User-cluster CIs, Holm correction within each RQ family, and confirmation on users never touched during design.

(C2) Decomposition.
- logit P(Yes) = user offset + item prior π(i) (history-swap estimator, split-half disattenuated) + personal evidence.
- Report the personal share and a "personal information gain" (ΔUAUC and Δlog-loss when LLM evidence is added to the item-mean prior in a 3-parameter stacker fit on dev users), by regime.
- A history-consistent pseudonym knockout with a real-brand placebo tests whether the prior is name familiarity.
- A ProCal-style two-axis popularity test, separating confidence *level* from the confidence-matched *gap*, with a popularity-informed ECE.

(C3) Actionability.
- A half-page invariance lemma, not claimed as deep. Per-user monotone maps leave UAUC and within-user NDCG unchanged. Only item-dependent corrections, cross-user decisions or training can change outcomes.
- Each route is tested against its proper control:
  - item-dependent corrections (mirrored, PMI, batch calibration) vs an equal-compute paraphrase placebo;
  - cross-user selective serving and thresholding at 10k scale vs random and vs the 8 baselines' own confidence, with niche-user coverage;
  - uncertainty pruning vs matched random over 5 seeds.

(C4) A one-table practitioner guide answering the six seed questions:
- S1 correct-but-unsure mass;
- S2 confident-error share and failure-prediction AUROC/AURC;
- S3 static exposure vs demand at full scale, plus a dynamic loop only if KuaiRec is readable, stated as model-side only;
- S4 reliability vs rated, MAR and list-normalized targets;
- S5 two-axis popularity plus the causal knockout;
- S6 pruning ≈ random is the expected and acceptable answer.

Banned claims:
- the two-prompt estimator as new;
- "first to find yes/no bias";
- "uncertainty improves ranking";
- any 6/8-domain SOTA claim (that belongs to the C-CRP reranker work, cited in the third person);
- anything not confirmed on fresh users.

What the 8-baseline next-item comparison contributes. It is context and testbed, not headline.
- (i) Anti-strawman anchor: the audited channel is a competitive same-candidate recommender, so the audit is about a model people would deploy.
- (ii) Population for a full-scale S3 answer. The local baseline `ranking_eval_records.csv` files contain full ranked lists (`pred_ranked_item_ids`, `topk_popularity_groups`; columns verified this session). This allows top-10 head share, Gini and tail coverage of 10 scorers against target demand on identical candidates. It directly tests whether LLM ranking is more or less head-concentrated than CF and LLM-embedding baselines; Lichtenberg et al. Gen-IR'24 found zero-shot LLMs *less* popularity-biased. Pilot-scale LLM top-10 head share is 0.745.
- (iii) Cross-user selective serving: LLM list-normalized confidence vs each baseline's own confidence/margin (where the `confidence` column or scores exist) vs random, with niche-user disparity.
- (iv) S2 "when the LLM is wrong, is anyone right?": the corrected oracle, which shows little exploitable headroom (+0.002-0.012 after cross-fitting).
- (v) The correct fix for the old 1/101 calibration flaw: softmax-normalize token logits over the 101 candidates, fit T on dev events, and report top-label ECE against the actual target.
- Limits:
  - Sampled-candidate metrics (Krichene & Rendle KDD'20) restrict every claim to the same-candidate scope.
  - The local ref_ranks files hold only positive ranks, so exposure must come from ranking_eval_records.
  - Old per-candidate C-CRP verbalized probabilities may have been lost with the old server. Verbalized-vs-token calibration on next-item would need a rerun at about 32 prompts/s, roughly 35 GPU-h per domain. Restrict that comparison to rated panels, or one domain.

Page plan (9 pages; plan for references counting until the 2027 CFP says otherwise):
- §1 Intro, with 4 findings and a Fig. 1 showing the decomposition: 1.0
- §2 Setup and protocol: 1.25
- §3 RQ1 reliability by regime/label: 1.0
- §4 RQ2 decomposition, causal knockout and popularity: 1.5
- §5 RQ3 actionability (lemma, item corrections, cross-user decisions, exposure vs 8 baselines): 1.75
- §6 RQ4 pruning: 0.5
- Related work: 0.5
- Guide/conclusion: 0.25
- References: about 1.25
Proofs, prompts and extended tables go in an anonymized supplement, if allowed.

4. HONESTY AND FORKING-PATH PROTOCOL
- Freeze PREREG_AMENDMENT_2 before any confirmatory GPU run. It should list:
  - hypotheses with direction-free reporting;
  - primary endpoints per RQ;
  - the dev/confirm split (all Pilot-1 users are dev; Video_Games and Sports rated stay untouched);
  - seeds: 3 for the main LoRA runs, 5 for pruning;
  - the Holm families;
  - the nested-slot kill rule;
  - a rule that the prompt is selected on dev users only, with exactly one confirmatory gate.
- The registered gate keeps its constants: ML-1M UAUC ≥ 0.60, Sports NDCG ≥ 0.186.
- Under the pilot pre-registration, pilot diagnostics cannot justify a method. In the measurement paper they are re-measured as confirmatory endpoints, never quoted as evidence.
- Every finding requires fresh users AND a second backbone before it can enter the abstract.
- Same-family review is flagged; get one cross-family review before freezing (project rule).

5. TIMELINE (2026-10-05 to 2027-01-21)

| Weeks | Dates | Work |
|---|---|---|
| W1 | to Oct 11 | Prereg-2, prompt-fix dev plus confirmatory gate, fresh rated panels, Pilot 2 (KuaiRec readability), Pilot 3 zero-shot |
| W2-3 | | Zero-shot rated grid on both backbones; full-scale next-item scoring; MF/LightGCN references; LoRA temporal splits |
| W4-6 | | LoRA grid (Qwen 4×3 seeds, Llama 4×3 or 2×3), scoring, swap priors, knockout under LoRA |
| W7-8 | | KuaiRec scale-up or limitation; pruning 5 seeds; nested slot, killed or kept by **Nov 30** |
| W9-10 | | Robustness (user-disjoint split, prior-popularity column), figures, cross-family review of claims |
| W11-12 | Dec 21-Jan 3 | Full draft |
| W13-15 | | Revision; abstract Jan 14; submit Jan 21 |

6. LIKELY REVIEWERS AND PRE-EMPTIONS
The likely pool is the POSTECH group (UQRec, KnowSA, PerK) and the USTC/NUS He-Feng group (TALLRec, CoLLM, UGR, SPRec, Flower).

| Objection | Answer |
|---|---|
| "UQRec did LLM-rec uncertainty" | Theirs is listwise, correlating uncertainty with NDCG. Ours is pointwise yes/no against preference labels, with a regime contrast and decomposition. |
| "Ravikumar audited calibration by popularity" | Theirs is zero-shot verbalized confidence vs catalog membership. Ours is token P(Yes) vs preferences, includes fine-tuning, and has a causal knockout. |
| "KnowSA: popularity is a weak knowledge proxy" | Consistent with our results; the knockout tests familiarity directly. |
| "Overconfidence is known; temperature scaling fixes it" | Agreed and shown. That is why the paper is about what the confidence encodes. |
| "Your zero-shot LLM is beaten by item mean" | Reported up front. The fine-tuned regime is the main object, and the LLM vs CF comparison is reported honestly. |
| "Sampled 1+100 metrics" | All claims are scoped to same-candidate. |

7. KEY RISKS
- Fine-tuned P(Yes) may be well calibrated and personal. That is still a clean, publishable contrast with zero-shot ("supervision fixes calibration but changes what confidence encodes").
- KuaiRec may be unreadable: zero-shot UAUC_MAR ≥ 0.58 is about 40% likely, rising to about 65% with LoRA. If unreadable, it becomes a one-paragraph validity limitation.
- The pseudonym knockout may be flat: about 55-65% chance of no clean positive. If so, drop the word "familiarity" and report "popularity-linked prior, not name recognition".
- C-CRP overlap is a dual-submission/self-plagiarism risk if that paper is under review. Cite it; never re-present its table as a contribution.
- SIGIR 2027 page rules are unknown. Assume the stricter reading (references included).

8. UNVERIFIED OR APPROXIMATE ITEMS
- The acceptance-probability figures are judgment, not data.
- The fine-tuned UAUC expectation (about 0.70-0.78 on ML-1M) is an unverified recollection of CoLLM-type results; the CoLLM abstract does not report numbers.
- Venues not confirmed:
  - EviRank: MM'26 appears only in a template header;
  - UNIT: unknown;
  - SLLM4CTR: ACM journal (doi 10.1145/3763789), venue and year unverified;
  - a 2027 reproducibility track: unverified.
- Dacrema et al. and Jones et al. arXiv ids are from memory.
- LoRA throughput on the 4090 (about 2 GPU-h per 30k-example run) is an estimate; benchmark it in W1.

## Recommendations
### Adopt framing C, retire MIRROR as flagship, and freeze PREREG_AMENDMENT_2 before any confirmatory run  (≈0 GPU-h)
- Rationale: MIRROR's only allowed novelty claim was a popularity-linked, item-conditional acquiescence. It is absent at the pair level in both rated domains (corr(a, log-pop) 0.007 and 0.074). MIRROR also loses 0.031 NDCG@10 on Sports and loses to the placebo on 2 of 3 panels. A regime-comparative audit is the only framing that survives either gate outcome. Pre-registration turns the forking-paths problem into a selling point.
- Protocol: Write idea-stage/PREREG_AMENDMENT_2.md. It must contain:
(1) Hypotheses H1-H9, one per RQ endpoint, with direction-free reporting.
(2) Dev/confirm split. All Pilot-1 users are dev. Confirm on fresh ML-1M users (about 1,683 eligible remain), fresh Toys users, and untouched Video_Games and Sports rated users. Next-item uses the full 10k×101 frozen panels.
(3) Primary endpoints per RQ: UAUC, ECE/Brier, AUROC/AURC, personal share, ΔNDCG@10 at matched coverage, head share of top-10.
(4) Holm correction within each RQ family; user-cluster bootstrap with 2,000 resamples.
(5) Seeds: 3 for main LoRA runs, 5 for pruning; any gain must exceed 2σ_seed.
(6) Mandatory reference rows: item-mean LOO, popularity, MF, LightGCN.
(7) MIRROR is reclassified as one confidence channel.
(8) Kill rule and date for the nested method slot.
- Risk: Reviewers may read it as "no method". Mitigate with the actionability section and the nested method slot. Same-family pre-registration review: get one cross-family read.
- Paper value: Turns the failed gate into Finding 1 rather than a dead end. A pre-registered, fresh-user-confirmed audit is rare in RecSys and is a credible rigor differentiator against UQRec, Ravikumar and KnowSA.

### Execute the registered remedy: dev-only prompt fix, then one confirmatory gate, plus a fine-grained rating channel  (≈3 GPU-h)
- Rationale: The registered remedy is "fix the prompt first". The current prompt does not follow TALLRec's liked/disliked layout. Fine-grained label scoring is known to beat yes/no for zero-shot pointwise rankers (Zhuang et al., NAACL 2024). Selecting on burned users and confirming once on fresh users avoids forking paths.
- Protocol: On the 1,500 ML-1M and 1,500 Toys pilot (dev) users, score a fixed bank:
- P1: TALLRec layout, with the history split into a liked list (≥4 stars) and a disliked list (≤2 stars), then "Will the user enjoy the target item? Answer Yes or No."
- P2: P1 with 20 history items.
- P3: P1 plus metadata (ML-1M year and genres; Amazon category and a 200-character description).
- R: a rating channel, "Predict the user's star rating (1-5)", scored as E[r] = sum over r of r·softmax(logits of the tokens '1'..'5').
Pick exactly one yes/no prompt by dev UAUC. Run the registered gate once: fresh ML-1M UAUC ≥ 0.60, and Sports NDCG@10 ≥ 0.186 on events 1001-2000. Log both outcomes in PILOT_LOG and report the original prompt as a sensitivity row.
- Risk: The fixed prompt may still fail; the literature predicts zero-shot ≈ weak. That is acceptable in framing C. Do not iterate further on the fresh users.
- Paper value: Either a valid zero-shot regime or a clean, literature-consistent Finding 1. Adds a token yes/no vs fine-grained rating channel comparison for S4.

### Make the LoRA-tuned yes/no regime the main object of the paper  (≈75 GPU-h)
- Rationale: Deployed probability-semantic LLM recommenders (TALLRec, CoLLM, BinLLM) are fine-tuned with log loss, a proper scoring rule. ConfTuner's tokenized Brier with N=1 reduces to an ordinary Brier score. This yields a falsifiable prediction: in-distribution calibration should be good, so residual miscalibration must come from temporal shift, exposure, or popularity strata. This regime does not depend on the zero-shot prompt.
- Protocol: Use src/confrec/train_lora_yesno.py in standard mode. Settings: r=16, α=32, lr 1e-4, 1 epoch, bf16 with gradient checkpointing, ≤40k examples per dataset.
Split: global temporal split (T = 80th percentile of candidate timestamps). Training users exclude all confirm users and Lumen valid/test users. Robustness: a user-disjoint split.
Grid: 4 rated datasets × Qwen3-8B × 3 seeds, plus Llama-3.1-8B × 4 datasets × 3 seeds (drop to 2 datasets if behind schedule by W5). Add ConfTuner-style Brier SFT as an alternative objective on 2 datasets.
Score each adapter in vLLM on {like, dislike, paraphrase} plus an 8-donor swap prior.
Report UAUC, AUC, ECE (15-bin and adaptive), Brier, Platt slope, and failure-prediction AUROC/AURC (mean ± sd over seeds), against item-mean, popularity, MF and LightGCN trained on the same split.
- Risk: Fine-tuned P(Yes) may be well calibrated and mostly item-quality-driven. That is still informative. A 24 GB card may force 4-bit or shorter histories, so benchmark throughput in W1. LoRA seed variance (σ ≈ 1.5 pt).
- Paper value: Gives the paper real headline numbers and the zero-shot vs fine-tuned contrast that no LLM-rec uncertainty paper reports for the yes/no family. Removes the "you audited a broken model" objection.

### Decomposition and causal identification: item prior vs personal evidence, with a pseudonym knockout  (≈15 GPU-h)
- Rationale: Exploratory pilot data suggest zero-shot discrimination is almost entirely item prior: like − π gives UAUC 0.507/0.525, and π correlates with popularity at 0.43/0.28. The K=8 prior is noisy, so the collapse may be partly an estimator artifact. Only a reliability-corrected measure plus an input-level intervention makes this a claim. This is the paper's organizing lens and answers S2 and S5.
- Protocol: For each regime × backbone × dataset:
(a) Estimate π(i) from two independent donor halves (4+4) and compute the reliability ρ.
(b) Personal share = 1 − R² of the within-user-centred like-logit on π, disattenuated by ρ.
(c) Personal information gain = ΔUAUC and Δlog-loss of a logistic stacker [user-centred item-mean LOO logit + LLM evidence] over item-mean alone. Fit on dev users, evaluate on confirm users.
(d) Pseudonym knockout (src/confrec/pilot_pseudonym.py) on Toys and Video_Games, with arms real / pseudo / real-brand placebo, under both zero-shot and LoRA. Registered thresholds: head−tail Δ ≥ 0.20; placebo within ±0.05.
(e) ProCal-style two-axis test: confidence level by popularity decile, plus the confidence-matched Bias Index with a Wilcoxon test and a popularity-informed ECE.
- Risk: The knockout may be flat (about 55-65% chance); then drop the word "familiarity" and report a popularity-linked prior. The prior may also not be popularity-linked in the fine-tuned regime; report the change either way.
- Paper value: The citable finding, "what does the confidence know", stated with reliability-corrected numbers and a causal control. Neither UQRec, Ravikumar nor KnowSA has this.

### Full-scale next-item audit against C-CRP and the 8 official baselines (context, exposure, cross-user decisions, list-level calibration)  (≈25 GPU-h)
- Rationale: This is what the 8-baseline comparison contributes:
- it shows the audited channel is a competitive same-candidate recommender;
- it provides the 10k-user population for S3 exposure and for cross-user decisions, where the actionability lemma says confidence can matter;
- it fixes the old 1/101 calibration flaw through list normalization.
The local ranking_eval_records.csv files contain full ranked lists and popularity groups for all baselines.
- Protocol: Score the 4 frozen panels (10k × 101) with token P(Yes) for "next" and "like" (Qwen3-8B), and "next" only for Llama (≈12M prompts).
Compute:
(a) Full-panel HR/NDCG@5/10/20 and MRR next to C-CRP and the 8 baselines, as a cited context table, never as a SOTA claim.
(b) List-normalized calibration: q_i = softmax over the 101 candidates of l_i/T, with T fit on 2k dev events; report top-label ECE, Brier and a reliability diagram against the target.
(c) Exposure: top-10 head share, Gini and APLT per method vs target-demand head share, from pred_ranked_item_ids / topk_popularity_groups.
(d) Cross-user selective serving: NDCG@10 risk-coverage by max q, margin and entropy vs random and vs each baseline's own confidence (where the column is populated), plus niche-user coverage at 50% coverage.
(e) Corrected oracle.
Optional: next-item LoRA on Sports with 3 seeds, to test whether fine-tuning raises confidence-popularity coupling and head share.
- Risk: Sampled-candidate metrics critique (Krichene and Rendle, KDD 2020): scope every claim to same-candidate. Overlap with the C-CRP paper: cite it in the third person and do not restate the 6/8 claim. Some baselines may lack a usable confidence column; fall back to rank-based proxies.
- Paper value: A full-scale answer to S3 (static) and the "so-what" decision results. It also satisfies the no-toy rule and removes the strawman objection.

### MAR validity on KuaiRec: confidently wrong vs confidently unlabelled, and does fine-tuning re-learn the logging policy  (≈25 GPU-h)
- Rationale: Every "confidently wrong" claim (S2/S4) depends on label validity. The zero-shot vs fine-tuned contrast under MAR labels is the sharpest LLM-specific prediction in the dossier (6/10, PROCEED). No paper audits LLM-recommender confidence against fully observed labels.
- Protocol: Run the registered Pilot 2 first: 300 × 200 small matrix with the amended logged view, 20 exposure draws, and the lift-CI rule. NULL if zero-shot UAUC_MAR < 0.58.
If readable: score 600 users × 3,327 videos zero-shot. Train LoRA on big-matrix logged feedback (positives plus sampled negatives, 3 seeds) and re-score.
Endpoints:
- logged ECE − MAR ECE (normalized gap);
- top-decile false positives that are MAR positives (lift CI > 1);
- residual-adjusted head/tail Bias Index under each view;
- ΔECE_MAR vs ΔECE_logged from zero-shot to LoRA.
If still unreadable after LoRA (UAUC < 0.60), write one validity-limitation paragraph and stop.
- Risk: Chinese captions may be near chance, and forced exposure may alter behaviour. KuaiRec small-matrix size limits generality. P(readable) is about 40% zero-shot and about 65% with LoRA.
- Paper value: If positive, a distinctive validity RQ that upgrades the paper (probably +5 points of acceptance odds). If negative, an honest limitation that costs little.

### Answer S6 properly: uncertainty pruning vs matched random with 5 seeds and a natural-noise oracle  (≈45 GPU-h)
- Rationale: The user explicitly asked S6. Lumen's prior evidence (N3) and the literature (GORACS: GraNd and EL2N fall below random on TALLRec CTR; LLMHNI: hard-noise confusion) predict ≈ random. A properly seeded negative result with UNIT's exact quantile rule pre-empts the "strawman pruner" objection.
- Protocol: Setup: one rated dataset (ML-1M or Toys) on the temporal split, Qwen3-8B LoRA.
Arms (25% pruning each, except the full-data arm):
- full data;
- random 25%;
- UNIT-style cut of the top 25% by zero-shot uncertainty 1 − max(P(Yes), P(No));
- prior-congruent cut;
- high-loss cut (EL2N proxy).
5 seeds each, 25 runs in total.
Primary endpoints: confirm-user UAUC and Brier. Claim only gains > 2σ_seed with a paired Wilcoxon test across seeds.
Noise-detection side study: AUROC of each signal against rating-review text contradictions. Check first that review text survived the data slimming. Otherwise use an LLM judge with a 200-example human audit.
- Risk: The expected null may be read as low-value, so keep it to 0.5 page plus the appendix. Review text may be missing from the slimmed Amazon data.
- Paper value: A rigorous, literature-consistent answer to S6. It reinforces the actionability map: uncertainty flags difficulty, not noise.

### One nested method slot with a hard kill date (Nov 30): prior-offset LoRA  (≈30 GPU-h)
- Rationale: A small, mechanism-derived intervention raises acceptance odds if it survives and costs little if it does not. Prior-offset training follows directly from the decomposition. It pushes the adapter to model personal evidence on top of a CF quality prior, and it acts outside N1 because it changes the model.
- Protocol: Train yes/no LoRA with logit = f_θ(u,i) + b·q̂(i), where q̂ is the logit of the shrunk item-mean like rate from the training split, b is learned, and q̂ is held fixed with stop-gradient (a GLM offset). Inference uses the full sum.
Compare against standard SFT and against post-hoc stacking (SFT logit + q̂), on 4 datasets × Qwen3-8B × 3 seeds.
Kill rule (pre-registered): retire the slot to the appendix if ΔUAUC vs post-hoc stacking < +0.01, or if its CI includes 0, on ≥2 of 4 datasets.
A CPU-only alternative slot: per-user offset recalibration for cross-user thresholding on the full-scale panels.
- Risk: Likely null: hybrid CF+LLM is crowded (CoLLM, A-LLMRec), and post-hoc stacking may absorb the gain. Must not delay the core grid.
- Paper value: If it survives, it turns the audit into an "audit plus principled fix" in the style of ProCal's phenomenon → metric → plug-in structure. If not, it becomes one more honest entry in the actionability map.

### Integrity hedges: C-CRP overlap, venue fallback, and page rules  (≈0 GPU-h)
- Rationale: The 8-baseline table and C-CRP belong to a separate reranker paper whose submission status is unclear. Re-presenting them risks a dual-submission or self-plagiarism issue under anonymous review. The SIGIR 2027 CFP is still a placeholder, and the user's stated constraint (references inside 9 pages) is stricter than the 2026 rule.
- Protocol: (1) Confirm the status of the C-CRP paper. If it is under review or on arXiv, cite it in the third person and use its ranks only as a reference row.
(2) Draft the paper assuming 9 pages including references, with an anonymized supplement for proofs, prompts and extended tables.
(3) Keep a fallback version for a negative-results or reproducibility track, if SIGIR 2027 announces one, or for CIKM/RecSys. Build it from N1-N3, MIRROR and pruning.
(4) Schedule one cross-family review of the claims list in W10.
- Risk: Policy changes when the 2027 CFP appears. Reviewer identification through the self-citation pattern.
- Paper value: Removes a desk-reject-class risk and protects the January submission against page-limit surprises.

## Citations
- Xiong, Hu, Lu, Li, Fu, He, Hooi. Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. ICLR 2024. arXiv 2306.13063 (verified in repo LIT_HOOI_GROUP.md)
- Xiong et al. (Hooi group). Proximity-Informed Calibration for Deep Neural Networks (ProCal, PIECE). NeurIPS 2023. arXiv 2306.04590 (verified in repo)
- Li, Xiong, Wu, Hooi. ConfTuner: Training LLMs to Express Their Confidence Verbally. NeurIPS 2025. arXiv 2508.18847 (verified in repo)
- Wu, Ni, Hooi et al. UNIT: Uncertainty-Aware Instruction Fine-Tuning. arXiv 2502.11962 (venue unverified)
- Li, Wang, ..., Hooi, ... Fact or Guesswork? Evaluating LLMs' Medical Knowledge with Structured One-Hop Judgments (MKJ). arXiv 2502.14275
- Kweon, Jang, Kang, Yu. Uncertainty Quantification and Decomposition for LLM-based Recommendation (UQRec). WWW 2025. arXiv 2501.17630
- Fan, Gao, Gong, Liu, Feng, He. Uncertainty-aware Generative Recommendation (UGR). KDD 2026. arXiv 2602.11719
- Yan, Xu, Wang, Guan, Zhao. EviRank: Evidence-Based Confidence Estimation for LLM-Based Ranking. arXiv 2606.04727 (venue unverified; MM'26 template header only)
- Lee, Jang, Kang, Yu. Filling the Gaps: Selective Knowledge Augmentation for LLM Recommenders (KnowSA_CKP). SIGIR 2026. arXiv 2604.07825
- Ravikumar. Do LLM Recommenders Know When They're Hallucinating? Auditing Confidence Calibration in Catalog Faithfulness. CIKM 2026 (short). arXiv 2608.10008
- Bao et al. TALLRec: An Effective and Efficient Tuning Framework to Align LLM with Recommendation. RecSys 2023. arXiv 2305.00447
- Zhang et al. CoLLM: Integrating Collaborative Embeddings into LLMs for Recommendation. IEEE TKDE 2025. arXiv 2310.19488 (verified this session; abstract reports no ML-1M numbers)
- Kang et al. Do LLMs Understand User Preferences? Evaluating LLMs on User Rating Prediction. arXiv 2305.06474, 2023 (verified this session: zero-shot LLMs lag traditional recommenders; fine-tuned catch up)
- Zhuang, Qin, Hui, Wu, Yan, Wang, Bendersky. Beyond Yes and No: Improving Zero-Shot LLM Rankers via Scoring Fine-Grained Relevance Labels. NAACL 2024. arXiv 2310.14122 (verified this session)
- Thomas, Spielman, Craswell, Mitra. Large Language Models can Accurately Predict Searcher Preferences. SIGIR 2024. arXiv 2309.10621, doi 10.1145/3626772.3657707 (verified this session)
- Krichene, Rendle. On Sampled Metrics for Item Recommendation. KDD 2020. doi 10.1145/3394486.3403226 (verified this session)
- Ferrari Dacrema, Cremonesi, Jannach. Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches. RecSys 2019. arXiv 1907.06902 (from memory; not verified this session)
- Burns, Ye, Klein, Steinhardt. Discovering Latent Knowledge in Language Models Without Supervision (CCS). ICLR 2023. arXiv 2212.03827
- The yes-no bias of LLMs reflects answer order and wording (crossed symmetrization). arXiv 2607.05552, 2026
- Zhou et al. Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering. ICLR 2024. arXiv 2309.17249
- Kweon, Kang, Yu. Obtaining Calibrated Probabilities with Personalized Ranking Models. AAAI 2022. doi 10.1609/aaai.v36i4.20326
- Kweon, Kang, Jang, Yu. Top-Personalized-K Recommendation (PerK). WWW 2024. arXiv 2402.16304
- Gao et al. KuaiRec: A Fully-Observed Dataset and Insights for Evaluating Recommender Systems. CIKM 2022. arXiv 2202.10842
- LLM-as-a-Judge for Offline Evaluation in Top-K Recommendation. KDD 2026. arXiv 2606.22961
- Chu, Hou. Incumbent Advantage: Brand Bias in LLM Recommendation. arXiv 2606.17443, 2026
- Mozafari, Piryani, Jatowt. Pretraining Exposure Explains Popularity Judgments in LLMs. SIGIR 2026 (short). arXiv 2605.12382
- Ni, Bi, Guo, Cheng. Popular but Wrong: Understanding and Mitigating LLM Overconfidence through Knowledge Popularity. EMNLP 2026 (per arXiv comment). arXiv 2505.17537
- Abdallah, Ali, Piryani, Abdalla, Jatowt. Large Language Models Systematically Favor Popular Options: Evidence and Mitigation Across MCQs. EMNLP 2026 main (per arXiv comment). arXiv 2608.29257 (verified this session)
- Mei et al. GORACS: Group-level Optimal Transport-guided Coreset Selection for LLM-based Recommender Systems. KDD 2025. arXiv 2506.04015
- Lin et al. DEALRec: Data-efficient Fine-tuning for LLM-based Recommendation. SIGIR 2024. arXiv 2401.17197
- Gao et al. SPRec: Self-Play to Debias LLM-based Recommendation. WWW 2025. arXiv 2412.09243
- Gao et al. Flower: Process-Supervised LLM Recommenders via Flow-guided Tuning. SIGIR 2025. arXiv 2503.07377
- Kapoor et al. Large Language Models Must Be Taught to Know What They Don't Know. NeurIPS 2024. arXiv 2406.08391
- Tian et al. Just Ask for Calibration. EMNLP 2023
- Zou, Li, Sun, Guo, Wang. Uncertainty-Calibrated Recommendations for Low-Active Users. KDD 2026 ADS. arXiv 2605.17788
- Jones, Sagawa, Koh, Kumar, Liang. Selective Classification Can Magnify Disparities Across Groups. ICLR 2021 (arXiv id 2010.14134 from memory; unverified)
- Lichtenberg, Buchholz, Schwoebel. Large Language Models as Recommender Systems: A Study of Popularity Bias. Gen-IR@SIGIR 2024 workshop. arXiv 2406.01285
- RouteRec: Strict Evaluation of Recommender-Agent Selection. SIGIR 2026 AgentSearch workshop. arXiv 2607.09908
- Self-Monitoring Large Language Models for Click-Through Rate Prediction (SLLM4CTR). ACM journal, doi 10.1145/3763789 (venue/year unverified)
- SIGIR 2025 overall acceptance 1,134/5,797 (about 20%), ACM DL proceedings listing (verified this session via search snippet); SIGIR 2027 CFP pages are placeholders as of 2026-10-02 (verified this session: https://sigir2027.org/pages/submit-tracks.html)
- Repository evidence: D:/Research/Lumen/outputs/confrec_pilot/pilot1/{decision.json, ml1m_rated.pilot_mirror.json, toys_rated.pilot_mirror.json, sports_next_1k.pilot_mirror.json, ml1m_rated.diag_baselines.json, toys_rated.diag_baselines.json}; D:/Research/Lumen/docs/sigir/PILOT_LOG.md; D:/Research/Lumen/idea-stage/{triage_verdict.md, NOVELTY_DOSSIER.md, PREREG_AMENDMENT_1.md}; baseline full rankings at D:/Research/Lumen/outputs/baselines/official_adapters/*/tables/ranking_eval_records.csv