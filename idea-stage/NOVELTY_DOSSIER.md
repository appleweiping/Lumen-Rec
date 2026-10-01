# Novelty Dossier — Lumen-Rec idea stage (A1 MIRROR · B1/A7 MNAR calibration · A2 pseudonym knockout)

- Reviewer: fresh Claude Opus 5.5 sub-agent acting as the ARIS `novelty-check` reviewer. Date: 2026-10-01.
- **review_independence: same-family. acceptance_status: provisional.**
  - The ideas (opus-A, opus-B), the triage reviewer and this reviewer are all the same model family.
  - Phase C of the skill (a cross-model Codex/GPT call) was **not** run, on the caller's instruction: this agent is the reviewer.
  - Any blind spot the family shares goes uncorrected. A different-family check is still owed before the full grid is funded.
- Procedure: `D:\Research\Auto-claude-code-research-in-sleep\skills\novelty-check\SKILL.md`, followed in full except Phase C (see above).
- Inputs read in full:
  - `idea-stage/brainstorm_A.md` (ideas 1, 2, 7) and `idea-stage/brainstorm_B.md` (idea 1);
  - `idea-stage/triage_verdict.md`;
  - `RESEARCH_BRIEF.md`, `docs/sigir/LIT_RECENT_UNC_LLM4REC.md`, `docs/sigir/LIT_HOOI_GROUP.md`.
- No file other than this dossier was written.

---

## 0. NOVELTY VERDICT LIMITS (verbatim; every verdict below is judged under this block)

```
=== NOVELTY VERDICT LIMITS (these bound how you judge, never how widely you search) ===
Search exhaustively; judge calibrated. Two failures waste months equally:
passing an idea a published paper already contains, and killing a viable idea
because the territory has neighbors.
1. Proximity is information, not a verdict. Someone working nearby goes in the
   report; it is not by itself a reason to reject.
2. ABANDON has exactly one qualification: a specific published paper already
   contains this result — name that paper. No named paper, no ABANDON.
3. Crowded-but-deltaed is PROCEED: state the delta in one sentence a reviewer
   could verify. Thin or contested delta is PROCEED WITH CAUTION — say what
   would make it carry, not why it should die. CAUTION is not a safe middle:
   if you cannot name the specific thing that makes the delta thin, the
   verdict is PROCEED.
4. Concurrent or competing work is not a veto. That is a race — report it and
   let the user decide whether to run it.
5. A direct attack on a central problem is legitimate novelty when nobody has
   executed it well. "This area is hot" does not mean "this area is taken."
6. This check is an early gate, never the last one — more triage, pilots, or
   external review still stand between any idea and a paper, whatever order
   this run uses. A wrongly passed idea dies cheaply at one of them; a wrongly
   killed idea is never seen again. When torn between two verdicts, choose the
   more permissive one.
Say plainly when an idea clears the check. Do not manufacture overlap.
```

The skill's three reviewer questions are answered for each idea:
- Is this method novel?
- What is the closest prior work?
- What is the delta?

---

## 1. Verdicts at a glance

| Idea | Score | Verdict | One-line reason |
|---|---|---|---|
| **A1 MIRROR** (like/dislike mirrored prompts; item-level acquiescence a(i); difference score) | **5/10** | **PROCEED WITH CAUTION** | The two-prompt symmetrized estimator and its additive identifiability are prior art: the CCS inference rule, and crossed symmetrization with P = σ((θ ± m)/s) in 2607.05552. That paper also reports per-item artifact residuals that are "item-idiosyncratic and cancel". What remains new is the *popularity-linked, rank-relevant* item term in personalized recommendation, which is unmeasured and directly contested by that paper. |
| **B1/A7** exposure-confounded (MNAR) calibration of LLM-rec P(Yes) against fully observed KuaiRec / MAR Coat labels | **6/10** | **PROCEED** | The MNAR-vs-MAR calibration gap is textbook for classical rankers: Kweon et al. (AAAI'22) evaluate calibration on Yahoo!R3/Coat unbiased test sets and fix it with unbiased ERM. No paper audits *LLM-recommender* confidence this way, and none tests the LLM-specific "zero-shot prior is exposure-free, fine-tuning re-confounds it" prediction. The closest LLM work (2606.22961, KDD'26) uses KuaiRec/Coat only to validate an LLM *judge*. |
| **A2** pseudonym knockout (consistent pseudonym map; causal familiarity effect on confidence; name-dropout LoRA) | **6/10** | **PROCEED** | Real-vs-fictional brand substitution is published, but only for zero-shot, non-personalized *choice* by commercial LLMs (Incumbent Advantage, 2606.17443). Entity substitution as a causal probe is old in NLP (Longpre et al. 2021). Nobody applies a history-consistent pseudonym map to the *confidence-matched calibration* of personalized LLM recommenders, or uses name-dropout as a recommender fine-tuning regularizer. |

No idea is ABANDON: for none of the three does a published paper already contain the result.

---

## 2. Search protocol and coverage

### 2.1 Sources

| Source | How it was used | Status |
|---|---|---|
| arXiv API (`export.arxiv.org/api/query`) | About 35 fielded queries (`abs:` / `all:` boolean); ID look-ups for every cited arXiv paper | Worked |
| Semantic Scholar | Graph API `paper/search` | **HTTP 429 on every call in this session** (no API key, several back-off retries) |
| Semantic Scholar (fallback) | Domain-restricted web search (`semanticscholar.org`), one query per idea family | Worked |
| Google Scholar / general web | About 40 web-search queries | Worked |
| OpenReview, ACM DL, ACL Anthology | Domain-restricted web search | Worked |
| Full text read | 2607.05552 (HTML), CCS (ar5iv), NPRec (HTML), 2606.22961 (HTML), Kweon AAAI'22 (PDF via pdftotext) | Read |
| Abstract only | All other papers in the tables | Read |
| CrossRef / OpenAlex | DOI metadata | Worked |

The recent window was checked explicitly: arXiv 2026-04 to 2026-09 results appear in every family below.

### 2.2 Query log (≥3 formulations per claim, across sources)

- **A1 claim 1. Item-conditional yes-bias exists in LLM recommendation.**
  - arXiv: `all:"yes bias" AND all:recommendation`; `abs:acquiescence AND abs:"language model"`; `abs:"yes-no bias" AND abs:"language model"`.
  - Web: "acquiescence OR yes-bias … recommendation items familiar"; "pointwise LLM recommender assigns higher Yes probability to popular items …"; "LLM recommendation memorized familiar items … overconfidence".
  - S2 (domain): "yes no bias language model recommendation like dislike negation item popularity".
- **A1 claim 2. The term is popularity-linked, and confident errors concentrate there.**
  - arXiv: `abs:"popularity" AND abs:"confidence" AND abs:"LLM-based recommend"`; `abs:"item familiarity" AND abs:"language model"`.
  - Web: "item-level prior bias LLM recommendation P(Yes) calibration counterfactual prompt popularity".
  - ACM DL (domain): "LLM-based recommendation confidence calibration popularity bias yes/no".
  - OpenReview (domain): "LLM recommender calibration popularity yes probability negation acquiescence".
- **A1 claim 3. The mirrored difference score debiases and changes ranking.**
  - Web: "mitigate yes bias by asking negated question and combining answers log-odds"; "Contrast-Consistent Search recommendation"; "pointwise LLM relevance ranking ask relevant and irrelevant both prompts"; "LLM-as-a-judge agreeableness bias negated prompt"; "LLM CTR prediction will the user click / will the user not"; "LLM recommender logical consistency like vs dislike"; "Prompt-Reverse Inconsistency"; "LLM relevance judgments … is the document not relevant".
  - arXiv: `abs:"negated" AND abs:"prompt" AND abs:"recommend"`; `abs:"dislike" AND abs:"like" AND abs:LLM AND abs:recommendation AND abs:prompt`; `abs:negation AND abs:recommend AND abs:"large language model"`; `abs:"negation" AND abs:"consistency" AND abs:"yes/no"`.
- **A1 claim 4. Mirror tuning (paired phrasings, flipped BCE targets).**
  - OpenReview (domain): "negation-consistent fine-tuning yes no answers flipped targets paired prompts".
  - Web: "training language models for negation consistency paired affirmative negated questions flipped labels"; "Language models are not naysayers".
  - Abstract fetched: REPAIR (2410.02205).
- **A1 claim 5. Identifiability under the additive model / calibration family.**
  - arXiv: `abs:"contextual calibration" AND abs:recommendation`; `abs:"label bias" AND abs:"in-context" AND abs:calibration`; `abs:calibration AND abs:"pointwise" AND abs:"LLM" AND abs:ranker`.
  - Web: "surface form competition OR domain conditional PMI recommendation"; "Calibrate Before Use … LLM recommendation"; "Batch Calibration"; "domain-context calibration"; "contrastive decoding LLM recommendation popularity bias subtract history-free".
  - Full text read: 2607.05552 (HTML) and CCS (ar5iv).
- **B1 claim 1. MNAR-vs-MAR calibration gap of LLM-rec P(Yes).**
  - arXiv: `abs:KuaiRec AND abs:"large language model"`; `abs:KuaiRec AND abs:LLM`; `abs:KuaiRand AND abs:"language model"`; `abs:"fully-observed" AND abs:"language model" AND abs:recommend`; `abs:"missing not at random" AND abs:"large language model" AND abs:recommend`; `abs:"missing-at-random" AND abs:"LLM"`.
  - Web: "LLM-based recommender evaluation KuaiRec fully observed small matrix"; "KuaiRec large language model captions …"; "LLM zero-shot recommendation evaluated on randomized unbiased test set Coat Yahoo R3 KuaiRand".
  - S2 (domain): "LLM recommendation exposure bias fully observed KuaiRec calibration".
  - OpenReview + ACM DL (domain): "large language model recommender MNAR exposure bias calibration unbiased test".
- **B1 claim 2. Fine-tuning re-learns the logging policy.**
  - Web: "… MNAR … fine-tuning learns logging policy"; "positivity bias OR exposure LLM yes/no recommender fine-tuned TALLRec evaluated on randomly exposed items".
  - arXiv: `abs:"exposure bias" AND abs:"LLM-based recommend"`; `abs:"selection bias" AND abs:"large language model" AND abs:"recommend"`.
  - Abstracts fetched: ABPO, CALMRec.
- **B1 claim 3. Item-dependent shift; a small MAR slice plus doubly-robust (DR) calibration fixes it.**
  - arXiv: `abs:calibration AND abs:"selection bias" AND abs:recommend`; `abs:"exposure" AND abs:"calibration" AND abs:"recommend" AND abs:"counterfactual"`.
  - Web: "calibration of recommender predictions biased logged feedback vs randomized test set ECE Yahoo R3 Coat"; "LLM imputation model doubly robust …"; "large language model propensity estimation debiasing Yahoo R3 Coat MCAR".
  - Full text read: Kweon AAAI'22 (PDF).
- **B1 claim 4. "Confidently wrong" may be "confidently unlabelled".**
  - Web: "confidently wrong OR false negatives unobserved items LLM recommender evaluation …"; "LLM user simulator predict like dislike evaluated on KuaiRec".
  - Abstracts / HTML fetched: 2607.11354, 2606.22961.
- **A2 claim 1. A consistent pseudonym map as a knockout in personalized LLM recommendation.**
  - arXiv: `abs:"pseudonym" AND abs:"language model" AND abs:"recommend"`; `abs:anonymiz AND abs:"item title" AND abs:"LLM"`; `abs:"fake" AND abs:"titles" AND abs:"recommend" AND abs:"language model"`; `abs:"obfuscat" AND abs:"item" AND abs:"LLM" AND abs:"recommend"`.
  - Web: "LLM recommendation replace item titles with random IDs or fake names ablation"; "LLM sequential recommendation anonymized item names …"; "LLM recommender item title replaced random OR placeholder names causal effect …"; "in-context collaborative filtering LLM synthetic anonymized items".
- **A2 claim 2. Familiarity causally inflates confidence and the calibration gap.**
  - arXiv: `abs:"brand" AND abs:"bias" AND abs:"language model" AND abs:"recommend"`; `abs:"fictional" AND abs:"brand" AND abs:"LLM"`; `abs:"name" AND abs:"familiarity" AND abs:"LLM"`.
  - Web: "brand name bias LLM product recommendation fictitious brands identical specifications"; "fictional entities replaced names LLM confidence calibration familiarity overconfidence"; "Global is Good, Local is Bad"; "Do LLMs Memorize Recommendation Datasets"; "LLM recommender items released after knowledge cutoff …".
  - OpenReview + ACM DL + ACL Anthology (domain): "brand name familiarity LLM recommendation fictitious brand pseudonym causal confidence".
- **A2 claim 3. Name-dropout as a training regularizer.**
  - arXiv: `abs:"entity" AND abs:"dropout" AND abs:"fine-tuning" AND abs:"memorization"`; `abs:"entity substitution" AND abs:"parametric knowledge"`; `abs:"synthetic item" …`; `abs:"memoriz" AND abs:"recommend" AND abs:"LLM" AND abs:"title"`.
  - Web: "counterfactual data augmentation entity name swap fine-tuning reduce reliance on parametric knowledge overconfidence"; "DropoutNet".
  - Abstracts fetched: CogCalib, 2606.17276, CFT.

### 2.3 Anti-hallucination verification (skill rule; Policy D1)

**Helper run.** `tools/verify_papers.py` (3-layer: arXiv → CrossRef → Semantic Scholar) was run on all 56 cited entries:
- verdict **WARN**, hallucination_rate **0.0**;
- **55 verified**;
- **1 `verify_pending`**: DropoutNet. Its S2 layer was rate-limited and it has no arXiv ID or DOI. It was confirmed manually from the NeurIPS 2017 PDF on cs.toronto.edu and the ACM DL listing 10.5555/3295222.3295249, and is tagged below as **[verified manually; helper pending]**.

**Checked directly.** Eight further entries:
- arXiv IDs via the arXiv API: 2502.07717, 2604.16318, 2607.12520;
- DOIs via CrossRef: 10.1145/3437963.3441799, 10.1145/3511808.3557220, 10.1145/3240323.3240355.

**Never fabricated.** No arXiv ID, DOI or title in this dossier comes from memory alone. Every entry was resolved in this session.

### 2.4 Limitations
- The Semantic Scholar API was blocked. Web search restricted to S2 domains is a weaker substitute.
- DBLP was not used.
- Google Scholar was reached only through a general web engine.
- Most papers were read at abstract level only. Statements of the form "X does not do Y" mean absence of evidence in the abstract or HTML read. They are not exhaustive full-text proof.

---

## 3. A1 — MIRROR

### 3.1 Proposed method (as stated by opus-A, refined by triage)

- Score each (user, item) pair with two prompts: "Will the user *like* X?" and "Will the user *dislike* X?".
- Model the two logits additively:
  - `logit_like = a(i) + s(u,i)`;
  - `logit_dislike = a(i) − s(u,i)`.
- Rank and calibrate on `(logit_like − logit_dislike)/2`, which removes an item-level acquiescence/familiarity term a(i).
- Because a(i) depends on the item, removing it changes within-user order. That is the escape from the internal N1 negative.
- Optional "mirror tuning": train on both phrasings with flipped BCE targets.

### 3.2 Core claims, closest prior, and what stays unknown

| # | Claim | Closest prior | What stays unknown or different |
|---|---|---|---|
| 1 | LLM yes/no recommenders carry an **item-conditional** acquiescence term a(i), beyond a global yes/no bias. | 2607.05552 (crossed symmetrization): reports the artifact's per-item residuals are "real (up to ±0.5) but item-idiosyncratic and cancel" (moral dilemmas). Braun, EMNLP-F'25: LLMs lean "no" globally. "Yes is Harder than No", CIKM'25: framing asymmetry across downstream tasks. | Not measured for **personalized recommendation**, nor for items with a familiarity gradient. The nearest evidence (idiosyncratic per-item residuals) *contests* the premise. |
| 2 | a(i) rises with item popularity / pretraining exposure, and confident errors concentrate where a(i) is large. | MKJ (2502.14275): Acc_pos ≠ Acc_neg, with accuracy tied to term frequency (QA judgments). Popular-but-Wrong (2505.17537): popularity-tied overconfidence in QA. Hou et al. (2305.08845): LLM rankers favour popular items (listwise). | **No paper links a negation-symmetrized yes-bias term to item popularity**, in recommendation or elsewhere. This is the open empirical delta. |
| 3 | The mirrored difference changes within-user ranking and beats an equal-compute paraphrase ensemble on UAUC / NDCG / tail calibration. | CCS inference rule p̃ = ½(p(x⁺) + 1 − p(x⁻)) (Burns et al., ICLR'23). 2607.05552: stance from symmetric averaging over crossed frames. PRIN (COLM'25): "correct?" vs "incorrect?" disagreement. Agreeableness-bias mitigation for LLM judges (2510.11822). | The estimator is **not new**. Its *use as an item-dependent, rank-changing correction in recommendation*, and the test against a paraphrase placebo, are new. No recommendation or IR paper using a negated-question score was found: TALLRec, CoLLM and BinLLM each score with a single "Yes" prompt. |
| 4 | Mirror tuning (both phrasings, flipped targets, BCE) yields a negation-coherent recommender. | REPAIR, "Aligning with Logic" (ICML'25; 2410.02205): augmentation that enforces negation invariance in preference judgments. "Making LMs Robust Against Negation" (2502.07717). "LMs are not naysayers" (*SEM'23; 2306.08189). | Negation-consistency training is a known template. A recommender instance with a proper BCE score and a calibration readout is new, but *derivative*. |
| 5 | Identifiability: under the additive model the difference is unbiased for s whatever a(i) is. | 2607.05552's own summary model is P = σ((θ ± m)/s), i.e. exactly the additive stance ± artifact decomposition. Calibrate-Before-Use, domain-context calibration, Batch Calibration and PMI-DC remove a global or context-level answer prior. | Not new as a statement. Writing it in the recommender's (u, i) form is a presentation step, not a contribution. |

### 3.3 Closest prior work

| Paper | Year | Venue | Overlap | Key difference |
|---|---|---|---|---|
| [Burns et al., Discovering Latent Knowledge in LMs Without Supervision (CCS), 2212.03827](https://arxiv.org/abs/2212.03827) | 2023 | ICLR | Inference rule ½(p(x⁺) + 1 − p(x⁻)) over a statement and its negation; MIRROR's P_M is the log-odds analogue | Answer-flipped pairs, hidden-state probe, unsupervised truth discovery; no items, popularity or ranking |
| [The yes-no bias of LLMs reflects answer order and wording, 2607.05552](https://arxiv.org/abs/2607.05552) | 2026 | arXiv (Jul) | Crossed symmetrization (approve/oppose verb × answer order × label words); P = σ((θ ± m)/s); per-item artifact residuals up to ±0.5 | Moral dilemmas, frontier plus 2 open MoE models. The artifact is mostly order/label-driven and the per-item part is "idiosyncratic and cancels"; no popularity link and no ranking use. **This is the strongest prior work against A1.** |
| [Prompt-Reverse Inconsistency (PRIN), 2504.01282](https://arxiv.org/abs/2504.01282) | 2025 | COLM | "Which are correct?" vs "Which are incorrect?" yields conflicting answers | LLM-as-judge on math/logic; diagnostic only |
| [Braun, Acquiescence Bias in LLMs, 2509.08480](https://arxiv.org/abs/2509.08480) | 2025 | EMNLP Findings | Measures yes-saying in LLMs | LLMs show a global *no*-bias; survey-style statements; not item-conditional |
| [Zhang et al., Yes is Harder than No, doi:10.1145/3746252.3761350](https://doi.org/10.1145/3746252.3761350) | 2025 | CIKM | Positive vs negative framings of the same question across downstream tasks; asymmetry; debiasing instructions | Global asymmetry; recommendation is not listed in its abstract or concepts |
| [Calibrate Before Use, 2102.09690](https://arxiv.org/abs/2102.09690) · [Domain-context calibration, 2305.19148](https://arxiv.org/abs/2305.19148) · [Batch Calibration, 2309.17249](https://arxiv.org/abs/2309.17249) · [Surface-form competition / PMI-DC, 2104.08315](https://arxiv.org/abs/2104.08315) | 2021–24 | ICML / ACL / ICLR / EMNLP | Estimate and remove the answer prior | Single prompt wording; label-level or context-level prior, not an item-by-negation decomposition |
| [Aligning with Logic / REPAIR, 2410.02205](https://arxiv.org/abs/2410.02205) · [Making LMs Robust Against Negation, 2502.07717](https://arxiv.org/abs/2502.07717) · [LMs are not naysayers, 2306.08189](https://arxiv.org/abs/2306.08189) | 2023–25 | ICML / arXiv / *SEM | Negation-invariance training and negation-insensitivity diagnostics | Generic NLP; no recommendation, no item prior |
| [TALLRec, 2305.00447](https://arxiv.org/abs/2305.00447) · [CoLLM, 2310.19488](https://arxiv.org/abs/2310.19488) | 2023–25 | RecSys / TKDE | Yes/no LLM recommenders with P(Yes) under BCE | Single "like/Yes" prompt; AUC/UAUC only; no negated prompt, no calibration analysis (per LIT_RECENT) |
| [Hou et al., LLMs are Zero-Shot Rankers, 2305.08845](https://arxiv.org/abs/2305.08845) | 2024 | ECIR | LLM rankers are popularity- and position-biased | Listwise; fixed by bootstrapping/prompting; no acquiescence decomposition |
| [CFT, 2410.22809](https://arxiv.org/abs/2410.22809) · [NPRec, 2503.08051](https://arxiv.org/abs/2503.08051) (CIKM'26) · [MACR, 2010.15363](https://arxiv.org/abs/2010.15363) | 2021–26 | arXiv / CIKM / KDD | Remove the non-personal part of a prediction by counterfactual subtraction (history removed, popularity-only activation, item-only branch) | A different estimator of the same "item prior vs user evidence" split. These are **competitor baselines** for MIRROR's ranking claim |
| [MKJ, 2502.14275](https://arxiv.org/abs/2502.14275) | 2025 | arXiv | Acc_pos vs Acc_neg decomposition; frequency-linked errors | Medical QA judgments; no negation-paired prompts per item |
| [RankSteer, 2602.03422](https://arxiv.org/abs/2602.03422) | 2026 | arXiv | Pointwise LLM-ranker calibration gap in IR | Representation steering, not prompt symmetrization |
| [Rank-one popularity component, 2606.21275](https://arxiv.org/abs/2606.21275) | 2026 | arXiv | Additive item-marginal (log p(i)) term in recommender scores; separating it cuts popularity energy by 98.6% | Dot-product CF decoders, not LLM yes/no |

### 3.4 Is it novel? Closest work? Delta?
- **Is the method novel?** No. The scoring rule and its identifiability are the symmetrized estimator of CCS and the crossed-symmetrization designs.
- **Is the finding novel?** Yes, if it holds. Nobody has measured an item-conditional, popularity-linked yes-bias in personalized LLM recommendation, or shown that removing it changes rankings.
- **Closest work:** 2607.05552 (design and identifiability), then CCS (rule), then PRIN and "Yes is Harder than No" (framing asymmetry).
- **Delta:** the item axis and its popularity link inside a *personalized* recommender, plus the rank consequence measured against an equal-compute placebo.

### 3.5 Assessment
- **Score: 5/10.** Clear neighbours; the delta is defensible but is an empirical premise, not a method.
- **Recommendation: PROCEED WITH CAUTION.** Two specific things make the delta thin:
  1. The only new ingredient is item-conditionality. If a(i) has no item structure, MIRROR collapses to the published global symmetrized estimator, which is rank-inert (N1).
  2. The one published measurement of per-item artifact residuals (2607.05552) found them idiosyncratic and cancelling. Braun and "Yes is Harder than No" find the dominant LLM bias is a *global* lean toward "no".
- **What would make it carry.** The triage's pre-registered pilot 1 is the right gate. For novelty, all four of the following must hold:
  - (a) the SD of within-user-centred a(i) is ≥ 0.20 logit, **and** corr(a, log-pop) ≥ 0.20, *with the valence term v(i) reported separately* (the triage's valence objection);
  - (b) ΔUAUC(MIRROR − paraphrase placebo) has a CI excluding 0;
  - (c) MIRROR beats, or is complementary to, **an item-prior subtraction baseline**: the user-marginalized prior (A9/B9 ≈ per-item PMI / Batch Calibration) and a CFT/NPRec-style history-ablation subtraction. If PMI-style item-prior removal matches MIRROR, the negated prompt adds nothing new;
  - (d) mirror tuning is compared with REPAIR-style negation augmentation, or cited as its instance.
- **Key differentiator (if the pilot is positive):** "LLM recommenders say yes to items they recognise", stated as a measured, popularity-linked, rank-relevant item term in personalized recommendation.
- **Risk (what a reviewer will cite):**
  - CCS and 2607.05552 ("this is crossed symmetrization");
  - Calibrate-Before-Use / PMI-DC / Batch Calibration ("this is prior removal");
  - Braun and "Yes is Harder than No" ("the bias is global, and toward no");
  - "LMs are not naysayers" (dislike logits may track like logits).

### 3.6 Allowed differentiation sentence

> "Prior symmetrization work (CCS; crossed symmetrization, arXiv 2607.05552) removes a global framing artifact from yes/no judgments. We show that in personalized LLM recommendation the artifact has an item-specific component that grows with item popularity, and that removing it with the mirrored like/dislike difference changes within-user rankings and improves UAUC and tail calibration beyond an equal-compute paraphrase ensemble and beyond item-prior (PMI-style) subtraction."

The last two clauses may be stated only after the pilot confirms them.

**Not allowed:**
- claiming the two-prompt estimator or the additive identifiability as new;
- "first to find yes/no bias in LLMs";
- "negation-consistent training" as a new idea. Mirror tuning is an application of negation augmentation (REPAIR).

---

## 4. B1 / A7 — Exposure-confounded (MNAR) calibration of LLM-recommender P(Yes)

### 4.1 Proposed method / study
- Measure the calibration of yes/no LLM recommenders (prefill P(Yes)) under two kinds of labels:
  - (a) logged MNAR labels, with unobserved pairs treated as sampled negatives;
  - (b) fully observed (KuaiRec small matrix) or MAR (Coat) preference labels.
- Quantify the "exposure gap", defined as logged ECE − MAR ECE.
- Find the share of top-decile "false positives" that are true positives under full observation.
- Test the LLM-specific prediction that LoRA fine-tuning on MNAR logs improves logged ECE but worsens MAR ECE relative to zero-shot ("fine-tuning re-learns the logging policy").
- A7 adds:
  - the head-vs-tail sign under each view;
  - the claim that MNAR-fit monotone recalibration cannot fix an item-dependent shift, while a 1–5% MAR slice with DR calibration can.

### 4.2 Core claims, closest prior, and what stays unknown

| # | Claim | Closest prior | What stays unknown or different |
|---|---|---|---|
| 1 | Calibration of LLM-rec P(Yes) differs materially between logged MNAR labels and fully observed / MAR labels. | Kweon et al., AAAI'22: naive calibration on MNAR implicit data gives biased probabilities; evaluated on the **Yahoo!R3 and Coat unbiased test sets**; fix via unbiased ERM (read in the PDF). Schnabel ICML'16; Saito WSDM'20; Yang RecSys'18. KuaiRec (CIKM'22): partial vs full observation changes evaluation. | All classical (MF/NCF/LightGCN) ranking models. **No audit of LLM-recommender confidence against fully observed or MAR labels was found**: arXiv, web, S2-domain, ACM and OpenReview searches all came back empty. |
| 2 | Fine-tuning on MNAR logs trades MAR calibration for logged calibration relative to zero-shot (LLM-specific re-confounding). | ABPO (2605.18899): logged bandit feedback pulls continual LLM-rec updates off target; IPS correction. CALMRec (2607.23647): LLM profile memory conflates exposure with preference (simulated). CogCalib (ACL'25): fine-tuning on prior-aligned data → overconfidence. | No paper compares **zero-shot vs fine-tuned LLM-rec calibration under MAR labels**. This is the sharpest, LLM-specific, falsifiable delta. |
| 3 | The head/tail calibration gap shrinks or flips under MAR; MNAR-fit monotone recalibration cannot fix it; a small MAR slice with DR calibration does. | Wang et al., WSDM'21, "Combating Selection Biases with a Few Unbiased Ratings". Doubly Calibrated Estimator (2403.00817). Propensity calibration (2303.12973). | The fix is classical and **must not be claimed**. The sign-flip-under-MAR measurement for LLM confidence is new. |
| 4 | Many "confidently wrong" LLM predictions are confidently *unlabelled* (true positives under full observation). | LLM-as-a-Judge offline evaluation (KDD'26; 2606.22961): uses **KuaiRec and Coat unbiased tests** as ground truth and argues that MNAR histories distort evaluation, but the LLM is a *judge* of other recommenders. User Preference Induction (2607.11354): LLM judgments of unobserved pairs reduce popularity distortion. World knowledge for missing labels in XC (2408.09585). Generative pseudo-labeling of unexposed items (2602.20995). | The thesis that unobserved ≠ negative is established, and LLMs are already used to fill label holes. Auditing the **LLM recommender's own** confident errors against a fully observed matrix is not published. |

### 4.3 Closest prior work

| Paper | Year | Venue | Overlap | Key difference |
|---|---|---|---|---|
| [Kweon, Kang, Yu, Obtaining Calibrated Probabilities with Personalized Ranking Models, doi:10.1609/aaai.v36i4.20326](https://ojs.aaai.org/index.php/AAAI/article/view/20326) | 2022 | AAAI | MNAR-aware calibration; evaluated on Yahoo!R3/Coat unbiased test sets; IPS-style unbiased ERM | Classical rankers; no LLM; no zero-shot vs fine-tuned contrast |
| [Combating Selection Biases with a Few Unbiased Ratings, doi:10.1145/3437963.3441799](https://doi.org/10.1145/3437963.3441799) | 2021 | WSDM | Uses a small MAR slice to debias | Classical; A7's proposed fix is this |
| [Doubly Calibrated Estimator on MNAR data, 2403.00817](https://arxiv.org/abs/2403.00817) | 2024 | arXiv | Calibration of imputation/propensity models for MNAR debiasing | Calibrates the debiasing machinery, not an LLM's confidence |
| [KuaiRec, 2202.10842](https://arxiv.org/abs/2202.10842) · [doi:10.1145/3511808.3557220](https://doi.org/10.1145/3511808.3557220) | 2022 | CIKM | Fully observed matrix; partial vs full observation changes evaluation conclusions | Classical models; no confidence |
| [LLM-as-a-Judge for Offline Evaluation in Top-K Recommendation, 2606.22961](https://arxiv.org/abs/2606.22961) | 2026 | KDD | Uses KuaiRec (captions) and Coat unbiased tests as ground truth; discusses MNAR exposure | The LLM evaluates *other* recommenders' lists; no P(Yes) calibration audit. Useful side evidence: LLMs can take KuaiRec text as input |
| [User Preference Induction with LLMs for Offline Top-N Evaluation, 2607.11354](https://arxiv.org/abs/2607.11354) | 2026 | arXiv | LLM judgments on unobserved pairs mitigate popularity-sensitive distortion | Evaluation methodology; no fully observed validation (per abstract) |
| [ABPO, 2605.18899](https://arxiv.org/abs/2605.18899) | 2026 | arXiv | Exposure-biased logs in LLM-rec updates; IPS; self-certainty | Training method on Amazon/ML; no MAR calibration audit |
| [CALMRec, 2607.23647](https://arxiv.org/abs/2607.23647) | 2026 | arXiv | LLM profiles conflate exposure with preference; propensity-weighted memory | Simulated environments; no calibration |
| [Ravikumar, Do LLM Recommenders Know When They're Hallucinating?, 2608.10008](https://arxiv.org/abs/2608.10008) | 2026 | CIKM (short) | Popularity-stratified ECE/Brier of zero-shot LLM recommenders | Label is catalog membership, not preference; no MAR |
| [Schnabel et al., Recommendations as Treatments, 1602.05352](https://arxiv.org/abs/1602.05352) · [Saito et al., Unbiased Learning from MNAR Implicit Feedback, 1909.03601](https://arxiv.org/abs/1909.03601) · [Yang et al., Unbiased Offline Evaluation, doi:10.1145/3240323.3240355](https://doi.org/10.1145/3240323.3240355) | 2016–20 | ICML / WSDM / RecSys | Naive-loss minimizer = exposure × preference; IPS evaluation | Foundations. Any "theory" in B1/B2 restates them |
| [On the Necessity of World Knowledge for Missing Labels in XC, 2408.09585](https://arxiv.org/abs/2408.09585) · [Generative Pseudo-Labeling for Pre-Ranking with LLMs, 2602.20995](https://arxiv.org/abs/2602.20995) | 2024–26 | arXiv | LLM priors as exposure-independent labels for missing or unexposed pairs | Not recommender-confidence calibration |
| [Uncertainty Calibration for Counterfactual Propensity Estimation, 2303.12973](https://arxiv.org/abs/2303.12973) | 2023 | arXiv | ECE on Coat/Yahoo/KuaiRand for propensity models | Propensity models, not LLMs |

### 4.4 Is it novel? Closest work? Delta?
- **Concept:** not novel. MNAR calibration bias and unbiased calibration are Kweon AAAI'22 plus the MNAR canon.
- **Setting and LLM-specific predictions:** novel.
  - No published audit of LLM-recommender P(Yes) against fully observed or MAR preference labels was found.
  - No published test of zero-shot vs fine-tuned re-confounding was found.
- **Closest work:** Kweon AAAI'22 (concept), then 2606.22961 (LLM plus KuaiRec/Coat, but as a judge).
- **Delta:** LLM confidence; the exposure-free-prior vs fine-tuned-re-confounding contrast; confident-error relabelling on a fully observed matrix.

### 4.5 Assessment
- **Score: 6/10.**
- **Recommendation: PROCEED** (crowded-but-deltaed). The delta is one verifiable sentence, given below. No found paper contests it.
- **Scope caveats.** These concern the paper shape, not novelty:
  - only KuaiRec (small matrix) and Coat qualify (Yahoo!R3 has no text, per 2606.22961);
  - so this is a validity RQ or section, not a method paper;
  - the DR/MAR-slice fix and the MNAR theory must be credited to the classical literature.
- **Key differentiator:** the zero-shot-vs-LoRA contrast under MAR labels. If fine-tuning improves logged ECE while worsening MAR ECE, that result is LLM-specific and not predicted by the classical work.
- **Risk (what a reviewer will cite):**
  - Kweon AAAI'22 ("already known that MNAR calibration is biased");
  - WSDM'21 few-unbiased-ratings ("your fix exists");
  - 2606.22961 ("LLMs on KuaiRec/Coat unbiased tests already done"). The answer: that paper is a judge, not recommender confidence;
  - forced-exposure artifacts in KuaiRec;
  - Coat's tiny size.

### 4.6 Allowed differentiation sentence

> "Classical work calibrates ranking scores under MNAR feedback (Kweon et al., AAAI'22) and recent work uses fully observed KuaiRec/Coat labels to validate LLM *judges* (KDD'26); we are the first to audit the confidence of yes/no LLM *recommenders* against fully observed and MAR preference labels, separating exposure-induced from preference-induced miscalibration, and to test whether fine-tuning on logged data trades counterfactual (MAR) calibration for logged calibration relative to the zero-shot model."

**Not allowed:**
- MNAR-calibration as a new concept;
- the DR / few-unbiased-ratings fix as a contribution;
- "first LLM study on KuaiRec";
- "unobserved ≠ negative" as a new insight.

---

## 5. A2 — Pseudonym knockout (causal familiarity effect on LLM-rec confidence) + name-dropout LoRA

### 5.1 Proposed method / study
- **Map.** Replace every real item or brand name in history and candidate with a **consistent** invented pseudonym. The map is hash-seeded, keeps category and attributes, and is consistent within each prompt.
- **What it isolates.**
  - In-prompt collaborative structure (brand loyalty, series, repeat purchase) survives the map.
  - Pretraining recognition of the name does not.
  - The resulting confidence shift is read as the causal effect of familiarity.
- **Placebo.** A real-brand swap to another brand of matched popularity (triage).
- **Measurements.**
  - Δ logit by popularity decile;
  - confidence-matched Bias Index (ProCal);
  - tail UAUC.
- **Method.** Name-dropout LoRA: pseudonymize 30–50% of training examples, then evaluate on real names.

### 5.2 Core claims, closest prior, and what stays unknown

| # | Claim | Closest prior | What stays unknown or different |
|---|---|---|---|
| 1 | A history-consistent pseudonym map is a valid input-level knockout of name familiarity in *personalized* LLM recommendation. | Incumbent Advantage (2606.17443): 1 real + 9 validated fictional brands with identical specs; the real brand was chosen in all 670 valid trials. Longpre et al. (2109.05052): entity substitution separates parametric from contextual knowledge. Faithfulness-QA (2604.25313). | Prior substitutions are **non-personalized** (single query, no history) or QA. Keeping substitutions consistent *across a user history*, so that brand-loyalty evidence is preserved, is not published for recommendation. |
| 2 | Familiarity causally inflates head over-confidence and the confidently-wrong mass; measured on the **confidence-matched calibration gap**, not the confidence level. | Incumbent Advantage (causal on *choice*). Global-is-Good (EMNLP'24) and Bias Beware (EMNLP'25): brand bias in LLM product recommendation. Brand Retrieval & Ranking (2609.16304): visibility tracks search interest. Popular-but-Wrong (2505.17537) and CogCalib (2505.20903): prior knowledge or popularity → overconfidence (QA / general fine-tuning). Di Palma et al. (2505.10212): popular ML-1M items are more memorized. Mozafari et al. (2605.12382): pretraining exposure drives popularity judgments. | The *direction* (familiar brand ⇒ more "yes") is already known for choice. The **calibration** consequence, i.e. whether familiarity inflates confidence beyond accuracy at matched confidence, in personalized LLM recommenders with a placebo, is not published. |
| 3 | Name-dropout LoRA reduces familiarity reliance (lower head over-confidence) with acceptable accuracy cost. | CogCalib (ACL'25): knowledge-aware fine-tuning cuts ECE when data overlaps prior knowledge. DropoutNet (NeurIPS'17): input dropout to train for missing inputs. Counterfactual entity-swap augmentation for context faithfulness (Faithfulness-QA). CFT (2410.22809): counterfactual history-ablation fine-tuning for LLM recommenders. | No LLM-recommender fine-tuning regularizer based on name pseudonymization was found. The template (input dropout / entity-swap augmentation) is known, so the novelty is the instance and its calibration readout. |
| 4 | This explains why interaction-popularity recalibration (internal N2) failed: it conditioned on the wrong variable. | KnowSA (SIGIR'26; 2604.07825): popularity is an imperfect proxy for LLM item knowledge. Mozafari SIGIR'26 short. PerRecBench (2501.13391). | The narrative is partly anticipated (popularity ≠ knowledge). A *causal* demonstration is new. |

### 5.3 Closest prior work

| Paper | Year | Venue | Overlap | Key difference |
|---|---|---|---|---|
| [Chu & Hou, Incumbent Advantage: Brand Bias … in LLM Recommendation, 2606.17443](https://arxiv.org/abs/2606.17443) | 2026 | arXiv (Jun; rev. Aug) | Real vs validated fictional brands with identical specs; causal brand-recognition effect on LLM recommendation | Zero-shot, non-personalized, skincare, 3 commercial LLMs; outcome is **choice**, with no confidence/calibration and no training fix. **The strongest prior work against A2.** |
| [Kamruzzaman et al., "Global is Good, Local is Bad?" Brand Bias in LLMs, 2406.13997](https://arxiv.org/abs/2406.13997) | 2024 | EMNLP | Brand bias in LLM recommendations | Global vs local brands; no knockout, no calibration |
| [Bias Beware: Cognitive Biases in LLM-Driven Product Recommendations, 2502.01349](https://arxiv.org/abs/2502.01349) | 2025 | EMNLP | Manipulable biases in LLM product recommendations | No familiarity knockout or calibration |
| [Evaluating Brand Retrieval and Ranking in LLM Recommendations, 2609.16304](https://arxiv.org/abs/2609.16304) | 2026 | arXiv (Sep) | Brand visibility tracks search interest / online conversation (a familiarity proxy) | Real brands only; observational |
| [Towards Objective Fine-tuning (CogCalib), 2505.20903](https://arxiv.org/abs/2505.20903) | 2025 | ACL | Prior-knowledge-aligned fine-tuning data → overconfidence; knowledge-aware training reduces ECE | General tasks, not recommendation; no name pseudonymization |
| [Popular but Wrong, 2505.17537](https://arxiv.org/abs/2505.17537) | 2025/26 | EMNLP'26 (arXiv comment) | Popularity → overconfidence; popularity-aware calibration | QA |
| [Longpre et al., Entity-Based Knowledge Conflicts in QA, 2109.05052](https://arxiv.org/abs/2109.05052) · [Faithfulness-QA, 2604.25313](https://arxiv.org/abs/2604.25313) | 2021–26 | EMNLP / arXiv | Entity substitution as a causal probe and as training augmentation | QA / RAG |
| [DropoutNet](https://www.cs.toronto.edu/~mvolkovs/nips2017_deepcf.pdf) (doi:10.5555/3295222.3295249) | 2017 | NeurIPS | Input dropout during training conditions the model for missing inputs | CF cold-start; pre-LLM. **[verified manually; helper pending]** |
| [Di Palma et al., Do LLMs Memorize Recommendation Datasets?, 2505.10212](https://arxiv.org/abs/2505.10212) · [Benchmark Leakage Trap, 2602.13626](https://arxiv.org/abs/2602.13626) · [Memorization in Generative Rec, 2606.17276](https://arxiv.org/abs/2606.17276) | 2025–26 | arXiv | Memorization or leakage drives LLM-rec performance; popular items more memorized | No pseudonym knockout; no calibration |
| [KnowSA_CKP, 2604.07825](https://arxiv.org/abs/2604.07825) · [Mozafari et al., 2605.12382](https://arxiv.org/abs/2605.12382) · [PerRecBench, 2501.13391](https://arxiv.org/abs/2501.13391) | 2025–26 | SIGIR'26 / SIGIR'26 short / arXiv | Knowledge ≠ popularity; pretraining exposure explains popularity judgments; preference isolated from item quality | Observational; no input intervention |
| [CFT, 2410.22809](https://arxiv.org/abs/2410.22809) · [NPRec, 2503.08051](https://arxiv.org/abs/2503.08051) | 2024–26 | arXiv / CIKM'26 | Counterfactual emphasis of behaviour / removal of pretraining popularity priors in LLM rec | History-ablation or popularity-embedding subtraction, not name knockout. Competing regularizers |

### 5.4 Is it novel? Closest work? Delta?
- **Not novel:**
  - "LLMs prefer recognized brands" (Incumbent Advantage; Kamruzzaman);
  - substitution as a causal probe (Longpre);
  - "prior knowledge → overconfidence" (CogCalib; Popular-but-Wrong).
- **Novel:**
  - a *history-consistent* pseudonym map inside *personalized* recommendation prompts;
  - with a real-brand-swap placebo;
  - read out on the *confidence-matched calibration gap*;
  - plus name-dropout as an LLM-recommender fine-tuning regularizer.
- **Closest work:** Incumbent Advantage, then CogCalib.

### 5.5 Assessment
- **Score: 6/10.**
- **Recommendation: PROCEED** (crowded-but-deltaed). The delta spans four verifiable axes against the closest paper: personalized, calibration rather than choice, consistent mapping with a placebo, and a training regularizer.
- **Design condition for the claim to count.** This is not a novelty veto. A level drop in confidence under pseudonyms is *predicted* by Incumbent Advantage, so it is not a contribution. The claim must rest on the confidence-matched Bias Index (calibration gap) and on the placebo separating familiarity from perturbation.
- **Name-dropout as a method:** novel as an instance but derivative in template (DropoutNet, entity-swap augmentation, CogCalib). Expect "of course it costs head accuracy" (KnowSA). The idea is stronger as a diagnostic RQ, which matches the triage.
- **Risk (what a reviewer will cite):**
  - Incumbent Advantage ("already shown brands matter causally");
  - CogCalib ("prior knowledge → overconfidence is known");
  - "the brand is legitimate quality evidence" (the construct-validity objection).
- **Race note (verdict rule 4).** Incumbent Advantage was revised in August 2026. One co-author shares a name with the first author of "LLMs are Zero-Shot Rankers"; whether they are the same person was not verified. A personalized or calibration follow-up is plausible. Treat this as a race, not a veto.

### 5.6 Allowed differentiation sentence

> "Fictional-brand experiments establish that LLMs prefer recognized brands in zero-shot, non-personalized product choice (Incumbent Advantage, arXiv 2606.17443); we instead apply a history-consistent pseudonym map inside personalized recommendation prompts, with a real-brand-swap placebo, to estimate the causal effect of name familiarity on the confidence-matched calibration gap of LLM recommenders, and test name-dropout fine-tuning as a regularizer against it."

**Not allowed:**
- "first to show brand/name bias in LLM recommendation";
- "first to show prior knowledge causes overconfidence";
- entity substitution as a new probing method.

---

## 6. Cross-idea notes

1. **A1 and A2 are coupled.** A2 is the identification experiment for A1's a(i): if a(i)'s popularity slope vanishes under pseudonyms, a(i) is familiarity. A1's novelty claim becomes much stronger with A2's causal result attached. Without it, a reviewer can read a(i) as non-commitment or valence (triage objections 1–2).
2. **A1's ranking claim needs item-prior baselines** that the generator listed only partially:
   - user-marginalized prior / per-item PMI (≈ Batch Calibration);
   - CFT-style history-ablation subtraction;
   - NPRec / MACR-style counterfactual subtraction.

   If any of these matches MIRROR, the paper's method contribution moves to the decomposition finding.
3. **B1 is a validity section, not a spine.** Its novelty is clean but narrow (2 qualifying datasets). It pairs naturally with the observation study's S2/S4 RQs, as the triage recommends.
4. **Shared warning from the NLP literature.** Three independent recent papers say LLM binary-answer artifacts are mostly *global*, lean toward **"no"**, and are order- or label-driven:
   - 2607.05552;
   - Braun, EMNLP-F'25;
   - "Yes is Harder than No", CIKM'25.

   That is the prior that A1's pilot must overturn. Cite all three regardless of outcome. If a(i) ≈ 0, they explain the negative.
5. **Useful side evidence for pilot 2 (B1).** 2606.22961 (KDD'26) already feeds KuaiRec text to an LLM judge. This lowers, but does not remove, the "KuaiRec captions unreadable" risk the triage flagged. Their judge accuracy was not checked in this session.

---

## 7. Source list (all resolved in this session; verification per §2.3)

**A1:**
- [2212.03827 CCS](https://arxiv.org/abs/2212.03827) (ar5iv full text read)
- [2607.05552 yes-no bias / crossed symmetrization](https://arxiv.org/abs/2607.05552) (HTML read)
- [2504.01282 PRIN](https://arxiv.org/abs/2504.01282)
- [2509.08480 Acquiescence Bias in LLMs](https://arxiv.org/abs/2509.08480)
- [doi:10.1145/3746252.3761350 Yes is Harder than No](https://doi.org/10.1145/3746252.3761350) (OpenAlex metadata)
- [2102.09690](https://arxiv.org/abs/2102.09690)
- [2305.19148](https://arxiv.org/abs/2305.19148)
- [2309.17249](https://arxiv.org/abs/2309.17249)
- [2104.08315](https://arxiv.org/abs/2104.08315)
- [2410.02205](https://arxiv.org/abs/2410.02205)
- [2502.07717](https://arxiv.org/abs/2502.07717)
- [2306.08189](https://arxiv.org/abs/2306.08189)
- [2309.06991 CCR](https://arxiv.org/abs/2309.06991)
- [2510.11822](https://arxiv.org/abs/2510.11822)
- [2305.00447](https://arxiv.org/abs/2305.00447)
- [2310.19488](https://arxiv.org/abs/2310.19488)
- [2305.08845](https://arxiv.org/abs/2305.08845)
- [2410.22809](https://arxiv.org/abs/2410.22809)
- [2503.08051](https://arxiv.org/abs/2503.08051) (HTML read)
- [2010.15363](https://arxiv.org/abs/2010.15363)
- [2502.14275](https://arxiv.org/abs/2502.14275)
- [2602.03422](https://arxiv.org/abs/2602.03422)
- [2606.21275](https://arxiv.org/abs/2606.21275)
- [2601.09478](https://arxiv.org/abs/2601.09478)
- [2306.04590](https://arxiv.org/abs/2306.04590)

**B1/A7:**
- [doi:10.1609/aaai.v36i4.20326](https://ojs.aaai.org/index.php/AAAI/article/view/20326) (PDF read)
- [doi:10.1145/3437963.3441799](https://doi.org/10.1145/3437963.3441799)
- [2403.00817](https://arxiv.org/abs/2403.00817)
- [2202.10842](https://arxiv.org/abs/2202.10842)
- [doi:10.1145/3511808.3557220](https://doi.org/10.1145/3511808.3557220)
- [2606.22961](https://arxiv.org/abs/2606.22961) (HTML read)
- [2607.11354](https://arxiv.org/abs/2607.11354)
- [2605.18899](https://arxiv.org/abs/2605.18899)
- [2607.23647](https://arxiv.org/abs/2607.23647)
- [2608.10008](https://arxiv.org/abs/2608.10008)
- [1602.05352](https://arxiv.org/abs/1602.05352)
- [1909.03601](https://arxiv.org/abs/1909.03601)
- [doi:10.1145/3240323.3240355](https://doi.org/10.1145/3240323.3240355)
- [2408.09585](https://arxiv.org/abs/2408.09585)
- [2602.20995](https://arxiv.org/abs/2602.20995)
- [2303.12973](https://arxiv.org/abs/2303.12973)
- [2402.16325](https://arxiv.org/abs/2402.16325)

**A2:**
- [2606.17443](https://arxiv.org/abs/2606.17443)
- [2406.13997](https://arxiv.org/abs/2406.13997)
- [2502.01349](https://arxiv.org/abs/2502.01349)
- [2609.16304](https://arxiv.org/abs/2609.16304)
- [2505.20903](https://arxiv.org/abs/2505.20903)
- [2505.17537](https://arxiv.org/abs/2505.17537)
- [2109.05052](https://arxiv.org/abs/2109.05052)
- [2604.25313](https://arxiv.org/abs/2604.25313)
- [DropoutNet (NeurIPS'17)](https://www.cs.toronto.edu/~mvolkovs/nips2017_deepcf.pdf) [verified manually; helper pending]
- [2505.10212](https://arxiv.org/abs/2505.10212)
- [2602.13626](https://arxiv.org/abs/2602.13626)
- [2606.17276](https://arxiv.org/abs/2606.17276)
- [2604.07825](https://arxiv.org/abs/2604.07825)
- [2605.12382](https://arxiv.org/abs/2605.12382)
- [2501.13391](https://arxiv.org/abs/2501.13391)

**Context, checked but not decisive:**
- [2604.16318](https://arxiv.org/abs/2604.16318)
- [2607.12520](https://arxiv.org/abs/2607.12520)
- [2604.25456](https://arxiv.org/abs/2604.25456) (dialect bias; not a PMI calibration paper, contrary to a search snippet)
- [2601.15721](https://arxiv.org/abs/2601.15721)
- [2504.11889](https://arxiv.org/abs/2504.11889)
- [2505.20730](https://arxiv.org/abs/2505.20730)
- [doi:10.1145/3726302.3730181 Dual Debiasing in LLM-based Rec (SIGIR'25 short)](https://doi.org/10.1145/3726302.3730181)

Novelty-Check: A1 PROCEED-WITH-CAUTION (5/10) | B1/A7 PROCEED (6/10) | A2 PROCEED (6/10) | review_independence: same-family | acceptance_status: provisional
