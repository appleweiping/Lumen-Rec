# Bryan Hooi Group (NUS) — Uncertainty / Confidence / Calibration in LLMs and LLM4Rec

Literature survey to guide a SIGIR paper on **uncertainty-aware LLM-based recommendation (LLM4Rec)**.
Compiled 2026-10-01. Every factual claim is tagged **[verified: URL]** or **[unverified]**.
Statements of our own interpretation or design advice are tagged **[unverified: our analysis]**.

---

## 0. TL;DR

1. **Nobody in Hooi's group has published a paper on uncertainty, confidence or calibration in LLM-based recommendation** (checked through 2026-10-01). The group's recommender papers are GNN/multimodal (GRE-MC SIGIR'26, MIG-GT AAAI'25, MGDCF TKDE'24, next-POI) plus one LLM4Rec workshop paper, RecLAIF (KDD'25 OARS), which has no confidence component. Their calibration line (ICLR'24 → NeurIPS'23 → NeurIPS'25) has never been applied to recommendation. [verified: https://bhooi.github.io/ ; https://scholar.google.com/citations?hl=en&user=ErEL3bgAAAAJ ; arXiv author query, 210 entries: https://export.arxiv.org/api/query?search_query=au:%22Bryan%20Hooi%22]
2. These 5 papers map most directly onto our research questions:
   - **Can LLMs Express Their Uncertainty?** (ICLR'24). A black-box confidence-elicitation framework (prompt × sampling × aggregation), evaluated with ECE and AUROC.
   - **ProCal / proximity bias** (NeurIPS'23). Samples in sparse regions are *more overconfident*. It introduces PIECE, a debiased ECE. This is the template for popularity-aware calibration.
   - **ConfTuner** (NeurIPS'25). The tokenized Brier score is a proper scoring rule over confidence tokens.
   - **UNIT** (arXiv'25). It cuts training claims the model is unfamiliar with, using the 75th-quantile CCP uncertainty as the threshold. This is the template for pruning noisy fine-tuning data by uncertainty.
   - **Fact or Guesswork? (MKJ)** (arXiv'25). For yes/no judgments, LLMs are skeptical (Acc_neg > Acc_pos), overconfident, and worse on low-frequency entities. This is the template for yes/no-token LLM4Rec.
3. **ProCal predicts the opposite of the naive hypothesis.** The naive view is "popular = high confidence, niche = low confidence". ProCal found that low-proximity (sparse, long-tail) samples are *more overconfident* relative to their accuracy, even when their raw confidence is lower. We should therefore measure **confidence level** and **calibration gap** per popularity bucket separately. [verified: https://arxiv.org/abs/2306.04590]
4. **Direct competitors outside the group already exist:**
   - Kweon et al., WWW'25: uncertainty decomposition for LLM4Rec.
   - Fan et al., KDD'26: UGR, uncertainty-aware generative recommendation.
   - EviRank (2026): confidence for LLM ranking.
   - Ravikumar (2026): verbalized-confidence calibration audit of LLM recommenders, stratified by popularity.

   We must position against them (see §8). [verified: https://arxiv.org/abs/2501.17630 ; https://arxiv.org/abs/2602.11719 ; https://arxiv.org/abs/2606.04727 ; https://arxiv.org/abs/2608.10008]

---

## 1. Method and coverage (how "all papers" was enumerated)

| Source | What was done | Result |
|---|---|---|
| Homepage | Full publication list parsed (2015–2026) | ~200 entries, curated by Hooi [verified: https://bhooi.github.io/] |
| arXiv API | `au:"Bryan Hooi"`, sorted by date | 210 entries, 2015-10 to 2026-09-30, all abstracts keyword-scanned [verified: https://export.arxiv.org/api/query?search_query=au:%22Bryan%20Hooi%22&max_results=400] |
| Google Scholar | Profile ErEL3bgAAAAJ sorted by date, 200 most recent works | Includes non-arXiv items such as RecLAIF [verified: https://scholar.google.com/citations?hl=en&user=ErEL3bgAAAAJ] |
| Semantic Scholar | Author search returned **≥19 fragmented "Bryan Hooi" IDs**; all merged and deduplicated | 297 unique titles [verified: https://api.semanticscholar.org/graph/v1/author/search?query=Bryan%20Hooi] |
| DBLP | **Blocked by an anti-bot challenge (Anubis)**; not bypassed by policy | Not used [verified: attempted https://dblp.org/search/author/api?q=Bryan%20Hooi] |
| Student checks | Google Scholar for Miao Xiong (now at Amazon) and Jiaying Wu; arXiv and web search for Yibo Li, Xiaoxin He, Yuan Sui, Yiwei Wang, Junfeng Fang, Kun Wang | No student LLM4Rec+uncertainty paper found [verified: https://scholar.google.com/citations?user=yQ4U_5IAAAAJ ; https://scholar.google.com/citations?user=mrfO62wAAAAJ] |
| GitHub API | Stars and last push for every repo (snapshot 2026-10-01) | See tables below |

Keywords scanned across all 210 arXiv abstracts: recommend, calibrat, confiden, uncertain, popular, echo, filter bubble, conformal, hallucin, overconfiden, self-consisten, abstain, prun, data selection, mislabel, long-tail.

**Co-author note.** For "Zhiyuan Liu", no co-authored Hooi paper was found in the arXiv or Scholar lists; the Hooi-group "Zhiyuan **Hu**" is a frequent co-author. [unverified]

---

## 2. Master table of relevant papers (2022–2026)

Relevance key:
- ★★★ = core for our SIGIR paper
- ★★ = directly borrowable component
- ★ = context or background

Stars and push dates come from the GitHub API on 2026-10-01.

| # | Paper | Venue | arXiv | Code (★ / last push) | Rel. |
|---|---|---|---|---|---|
| A1 | Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs (Xiong, Hu, Lu, Li, Fu, He, **Hooi**) | ICLR 2024 | 2306.13063 | MiaoXiong2320/llm-uncertainty (149 / 2024-03-14) | ★★★ |
| A2 | ConfTuner: Training LLMs to Express Their Confidence Verbally (Yibo Li, Miao Xiong, Jiaying Wu, **Hooi**) | NeurIPS 2025 | 2508.18847 | liushiliushi/ConfTuner (38 / 2026-09-22) | ★★★ |
| A3 | Proximity-Informed Calibration for Deep Neural Networks, ProCal (Xiong, Deng, Koh, Wu, Li, Xu, **Hooi**) | NeurIPS 2023 (Spotlight) | 2306.04590 | MiaoXiong2320/ProximityBias-Calibration (19 / 2023-11-11) | ★★★ |
| A4 | Balancing Truthfulness and Informativeness with Uncertainty-Aware Instruction Fine-Tuning, UNIT (Tianyi Wu, Jingwei Ni, **Hooi**, …) | arXiv 2025 (venue unknown) | 2502.11962 | AndrewWTY/UNIT (33 / 2025-06-24) | ★★★ |
| A5 | Fact or Guesswork? Evaluating LLMs' Medical Knowledge with Structured One-Hop Judgments, MKJ (Jiaxi Li, Yiwei Wang, …, **Hooi**, …) | arXiv 2025 | 2502.14275 | not found | ★★★ |
| A6 | Test-Time Scaling in Reasoning Models Is Not Effective for Knowledge-Intensive Tasks Yet (James Xu Zhao, **Hooi**, See-Kiong Ng) | COLM 2026 (per repo) | 2509.06861 | XuZhao0/tts-knowledge (9 / 2026-08-04) | ★★ |
| A7 | RecLAIF: Reinforcement Learning from AI Feedback for Recommendation Systems (Xiaoxin He, …, **Hooi**, Bresson, Subbian) | KDD 2025 OARS Workshop | — | no official repo | ★★ |
| B1 | Birds of a Feather Trust Together: NeighborAgg (Xiong, Li, Feng, Deng, Zhang, **Hooi**) | TMLR 08/2022 | 2211.16466 | MiaoXiong2320/NeighborAgg (8 / 2022-08-16) | ★★ |
| B2 | Enhancing Multi-Agent Debate via Confidence Expression, ConfMAD (Zijie Lin, **Hooi**) | EMNLP 2025 Findings | 2509.14034 | Enqurance/ConfMAD (2 / 2025-11-13) | ★★ |
| B3 | Uncertainty of Thoughts, UoT (Zhiyuan Hu, …, Koh, **Hooi**) | NeurIPS 2024 | 2402.03271 | zhiyuanhubj/UoT (110 / 2024-08-05) | ★★ |
| B4 | Primacy Effect of ChatGPT (Yiwei Wang, …, **Hooi**) | EMNLP 2023 | 2310.13206 | wangywUST/PrimacyEffectGPT (4 / 2023-10-19) | ★★ |
| B5 | GraphCleaner: Detecting Mislabelled Samples in Popular Graph Learning Benchmarks (Yuwen Li, Miao Xiong, **Hooi**) | ICML 2023 | 2306.00015 | lywww/GraphCleaner (7) | ★★ |
| B6 | Great Models Think Alike: Inter-Model Latent Agreement (Ailin Deng, Miao Xiong, **Hooi**) | ICML 2023 | 2305.01481 | d-ailin/latent-agreement (1 / 2023-07-20) | ★ |
| B7 | Trust, but Verify: Self-Supervised Probing (Deng, Li, Xiong, Chen, **Hooi**) | ECCV 2022 | 2302.02628 | d-ailin/SSProbing (6 / 2022-12-20) | ★ |
| B8 | Towards a Unified View of Answer Calibration for Multi-Step Reasoning (Shumin Deng, Ningyu Zhang, Nay Oo, **Hooi**) | NLRSE Workshop 2024 (arXiv 2023) | 2311.09101 | 231sm/Eval_Multi-Step_Reasoning (4 / 2024-08-18) | ★ |
| B9 | Exploring Collaboration Mechanisms for LLM Agents: A Social Psychology View (Zhang, …, **Hooi**, Shumin Deng) | ACL 2024 | 2310.02124 | zjunlp/MachineSoM (121 / 2025-06-06) | ★ |
| B10 | Rewarding the Rare: Uniqueness-Aware RL (Zhiyuan Hu, …, **Hooi**) | ACL 2026 Findings | 2601.08763 | not found | ★ |
| B11 | Towards Natural Personalization: RealPref (Qianyun Guo, Yibo Li, Yue Liu, **Hooi**) | EMNLP 2026 Findings | 2603.04191 | GG14127/RealPref (9 / 2026-03-13) | ★ |
| B12 | FineVerify (Zhao, Chen, **Hooi**, Ng) | EMNLP 2026 | 2606.00660 | XuZhao0/fineverify (5 / 2026-06-02) | ★ |
| B13 | Enabling Self-Improving Agents to Learn at Test Time with HITL Guidance, ARIA (Yufei He, …, **Hooi**) | EMNLP 2025 Industry | 2507.17131 | — | ★ |
| B14 | JudgeLRM (Nuo Chen, …, **Hooi**, Bingsheng He) | arXiv 2025 | 2504.00050 | NuoJohnChen/JudgeLRM (42 / 2026-05-06) | ★ |
| B15 | Turning Bias into Bugs: Bandit-Guided Style Attacks on LLM Judges, BITE (Yang, **Hooi**, …) | ICML 2026 | 2605.26156 | xianglinyang/llm-as-a-judge-attack (3) | ★ |
| B16 | When Order Matters: First-Speaker Bias in Sequential MAD (Xu, **Hooi**, Qiao) | arXiv 2026-09-30 | 2609.38964 | — | ★ |
| B17 | Can KGs Make LLMs More Trustworthy? OKGQA (Yuan Sui, …, **Hooi**) | ACL 2025 | 2410.08085 | Y-Sui/OKGQA (15 / 2025-05-08) | ★ |
| C1 | Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion, GRE-MC (Yuan Li, Jun Hu, Jiaxin Jiang, **Hooi**, Bingsheng He) | SIGIR 2026 | 2605.00670 | — | ★ (SIGIR format reference) |
| C2 | Modality-Independent GNNs with Global Transformers for Multimodal Rec, MIG-GT (Jun Hu, **Hooi**, …) | AAAI 2025 | 2412.13994 | CrawlScript/MIG-GT (55 / 2025-05-13) | ★ |

Verification sources for the table:
- Titles, authors and venues: [verified: https://bhooi.github.io/] and [verified: https://scholar.google.com/citations?hl=en&user=ErEL3bgAAAAJ]
- arXiv IDs: [verified: https://export.arxiv.org/api/query?search_query=au:%22Bryan%20Hooi%22]
- Repos and stars: [verified: https://api.github.com/repos/<owner>/<repo>, snapshot 2026-10-01]
- UNIT venue: listed as arXiv preprint only on Google Scholar [unverified]
- TTS-knowledge venue: "COLM 2026" taken from the repo description [verified: https://github.com/XuZhao0/tts-knowledge]
- MKJ and GraphCleaner repos: not linked from the paper; GraphCleaner repo found by GitHub search, authorship of lywww/GraphCleaner [unverified]

---

## 3. Tier-A paper cards (full: framework, experiments, findings, borrow and gap)

### A1. Can LLMs Express Their Uncertainty? (ICLR 2024)

arXiv 2306.13063 · OpenReview gjeQKFxFpZ · code: github.com/MiaoXiong2320/llm-uncertainty (149★)
[verified: https://arxiv.org/abs/2306.13063 ; https://openreview.net/forum?id=gjeQKFxFpZ ; https://github.com/MiaoXiong2320/llm-uncertainty]

**Problem formulation.** "Confidence elicitation" means black-box estimation of P(answer correct) for closed APIs without token logits. It is evaluated on two orthogonal tasks: **calibration** and **failure prediction**. [verified: https://arxiv.org/html/2306.13063]

**Framework (3 components).** [verified: https://arxiv.org/html/2306.13063 §3]
1. **Prompting strategy**, for verbalized confidence:
   - Vanilla
   - CoT
   - Self-Probing: judge the answer in a separate session
   - Multi-Step: C = ∏ of the per-step confidences C_i
   - Top-K: K guesses, each with a probability
2. **Sampling strategy**, to obtain M responses:
   - Self-random: temperature
   - Prompting: paraphrase the question
   - Misleading: inject "I think the answer might be …" hints
3. **Aggregation strategy:**
   - Consistency: C = (1/M) Σ 1{Ŷ_i = Ỹ}
   - Avg-Conf: consistency weighted by verbalized confidence
   - Pair-Rank: MLE of a categorical distribution from pairwise rank events in Top-K lists, with P(S_u ≻ S_v) = P(S_u) / (P(S_u) + P(S_v)), plus a proposition with proof

**Experiment design.** [verified: https://arxiv.org/html/2306.13063 §4, App. E]
- 8 datasets across 5 reasoning types:
  - GSM8K and SVAMP (arithmetic)
  - StrategyQA and SportUND (commonsense)
  - DateUnd and ObjectCou (symbolic)
  - Prof. Law (MMLU)
  - Business Ethics (MMLU)
- 5 LLMs: Vicuna-13B, GPT-3 175B, GPT-3.5-turbo, GPT-4, LLaMA-2-70B
- Metrics: ECE, AUROC, AUPRC-Positive, AUPRC-Negative
- White-box baselines in the appendix: sequence probability, length-normalized probability, …

**Main findings.** [verified: https://arxiv.org/abs/2306.13063]
- LLMs are overconfident when verbalizing. Values cluster between 80 and 100% and in multiples of 5, imitating human speech.
- Calibration and failure prediction improve with scale.
- Human-inspired prompts reduce ECE but leave failure prediction weak; GPT-4 averages only about 62.7% AUROC.
- Sampling and aggregation help failure prediction.
- White-box beats black-box, but the gap is narrow: AUROC 0.522 vs 0.605.

**Paper architecture.** [verified: https://arxiv.org/html/2306.13063]
- §1 Introduction ends in 5 numbered findings, each tied to a subsection.
- §2 Related Work (short; long version in the appendix).
- §3 Framework (3.1 Motivation, 3.2 Prompting, 3.3 Sampling, 3.4 Aggregation).
- §4 Setup (one paragraph each for datasets, models, metrics).
- §5 Evaluation and Analysis: **every subsection title is a finding**, e.g. "5.1 LLMs tend to be overconfident…", "5.3 Variance among multiple responses improves failure prediction".
- §6 Discussion.
- Appendix: A proof; B detailed results (B.1–B.8 phrased as questions, e.g. "How much does the role-play prompt affect…?"); C related work; **D "Best practice and recommendations for practitioners"**; E setup; F prompts.
- Counts: 8 figures (main body Figs 1–3) and 18 tables (main body Tables 1–4).

**Borrow for LLM4Rec.** [unverified: our analysis]
- Re-instantiate the 3-component taxonomy for recommendation:
  - Prompt: P(Yes) token, verbalized 0–100, Top-K items with probabilities.
  - Sampling: candidate-order permutation (see B4, primacy effect), history paraphrase or truncation, misleading popularity hints such as "this item is a bestseller".
  - Aggregation: consistency of the top-1 item across permutations; Pair-Rank over sampled ranked lists maps naturally onto recommendation lists.
- Report both calibration (ECE, Brier) and failure prediction (AUROC, AUPRC-N), copying this split exactly.

**Gap.** Only QA and reasoning are covered. No ranking or list outputs, no user-personalized ground truth, no popularity analysis.

---

### A2. ConfTuner (NeurIPS 2025)

arXiv 2508.18847 · OpenReview VZQ04Ojhu5 · code: github.com/liushiliushi/ConfTuner (38★, actively maintained, last push 2026-09-22; checkpoints on Hugging Face per README)
[verified: https://arxiv.org/abs/2508.18847 ; https://openreview.net/pdf?id=VZQ04Ojhu5 ; https://github.com/liushiliushi/ConfTuner]

**Problem formulation.** Calibrate *verbalized* confidence without ground-truth confidence labels, proxy targets, or repeated sampling. The central question is whether an LLM can become naturally calibrated during training. [verified: https://arxiv.org/html/2508.18847 §1]

**Method.** [verified: https://arxiv.org/html/2508.18847 §3]
1. Take the logits of the confidence tokens 𝒯_N = {0..N} at the position after the fixed prefix "Confidence:", and softmax them to get q ∈ Δ^{N+1}. The paper uses N = 100 for LLaMA and N = 9 for Qwen and Ministral.
2. **Tokenized Brier score:** ℓ(q, y) = Σ_i q_i (y − i/N)², where y is answer correctness. For HotpotQA and TruthfulQA, correctness is judged by GPT-4o.
3. **Theorem 1:** the tokenized Brier score is a *proper scoring rule for verbalized confidence*. The risk is minimized by putting all mass on the token closest to η(x) = P(correct | x).
4. LoRA training (r = 8, α = 32). For LLaMA, an SFT-loss regularizer is added to prevent it from dropping the confidence output.

**Experiment design.** [verified: https://arxiv.org/html/2508.18847 §4, App. D]
- Training data: HotpotQA only.
- Evaluation, in-distribution: HotpotQA.
- Evaluation, out-of-distribution: GSM8K, TriviaQA, StrategyQA, TruthfulQA.
- Backbones: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Ministral-8B.
- Baselines (4 main): Base, Ensemble (from A1), SaySelf, LACIE. The appendix adds P(True), a classifier, and black-box methods.
- Metrics: ECE (10 bins) and AUROC.
- Downstream: (i) self-correction of low-confidence answers; (ii) confidence-based **model cascade** to GPT-4o under a fixed revision budget.
- Efficiency: about 2,000 training samples suffice; 4×A40 GPUs.

**Main findings.** [verified: https://arxiv.org/abs/2508.18847 ; https://arxiv.org/html/2508.18847]
- Up to 54.7% better ECE and 14.4% better AUROC than the best baseline.
- Generalizes to implicit (high/medium/low) confidence expressions and can score GPT-4o's answers.
- In the cascade, refined accuracy is up to +9.3% (HotpotQA) and +5.5% (TruthfulQA) at the same budget.

**Paper architecture.** [verified: https://arxiv.org/html/2508.18847]
- §1 Intro, with Fig. 1 showing a high-stakes medical example; the central RQ is stated in bold.
- §2 Background (proper scoring rules).
- §3 Method (3.1 / 3.2 / 3.3 Theory).
- §4 Experiments, organized as **question-titled subsections**: "4.2 Can ConfTuner learn effective verbalized confidence estimation?" and "4.3 Can ConfTuner help build more reliable and cost-effective LLM systems?", then 4.4 efficiency.
- §5 Related Work (placed late).
- §6 Conclusion.
- Appendix: proof, prompts, reproducibility, ablations (regularizer, data size, confidence form, distribution shift), extra results F.1–F.7.
- Counts: about 6 figures and 19 tables (main body Figs 1–5, Tables 1–6).

**Borrow for LLM4Rec.** [unverified: our analysis]
- For a yes/no LLM4Rec model (TALLRec style), take 𝒯 = {No, Yes}, i.e. N = 1. The tokenized Brier score then reduces to the standard Brier score on P(Yes). This gives a *principled* calibration loss to add to recommendation fine-tuning, with ConfTuner's proper-scoring theorem as the justification.
- For generative top-K recommendation: append "Confidence: x" after each recommended item and train with the tokenized Brier score against hit/miss.
- Copy the downstream framing for recommendation:
  - Cascade: LLM recommendation → fall back to a collaborative-filtering model when confidence is low.
  - Selective recommendation / abstention.
  - Confidence-gated exploration of niche items.

**Gap.** Correctness is binary per QA item. Nothing addresses **list-level** correctness (hit@K), user heterogeneity, or popularity-conditioned miscalibration.

---

### A3. Proximity-Informed Calibration (ProCal) (NeurIPS 2023 Spotlight)

arXiv 2306.04590 · code: github.com/MiaoXiong2320/ProximityBias-Calibration (19★)
[verified: https://arxiv.org/abs/2306.04590 ; https://bhooi.github.io/ ; https://github.com/MiaoXiong2320/ProximityBias-Calibration]

**Problem formulation.** **Proximity bias**: models are more overconfident on low-proximity samples, i.e. samples in sparse regions of the data distribution. [verified: https://arxiv.org/html/2306.04590 §3]
- Proximity: D(X) = exp(−(1/K) Σ_{X_i ∈ N_K(X)} dist(X, X_i)), computed with K = 10 on penultimate-layer features against the validation set.
- Definition 3.1: proximity bias is present if P(Ŷ = Y | P̂ = p, D = d1) ≠ P(Ŷ = Y | P̂ = p, D = d2).

**Empirical protocol.** [verified: https://arxiv.org/html/2306.04590 §3.1]
1. Split samples into 5 proximity groups.
2. Confidence-match the highest and lowest groups, drawing 10,000 samples in each direction.
3. Apply a **Wilcoxon rank-sum test**.
4. Compute **Bias Index = Acc(B_H) − Acc(B_L)** at matched confidence.

This was run on **504 timm ImageNet models**:
- More than 80% of models have p < 0.05.
- Transformers are more biased than CNNs.
- Temperature scaling does not remove the bias.
- Low-proximity samples overfit more: the train–validation gap is 31.67% vs 0.6%.

**Metric.** **PIECE** = E_{P̂,D}[ |P(Ŷ = Y | P̂, D) − P̂| ]. Theorem 4.2 shows PIECE ≥ ECE; ECE hides a "cancellation effect" between over- and under-confident subgroups. [verified: https://arxiv.org/html/2306.04590 §4]

**Method (plug-and-play).** [verified: https://arxiv.org/html/2306.04590 §5]
1. **Density-Ratio Calibration** for continuous scores. Estimate P(correct | P̂, D) by Bayes' rule, using 2-D KDE of (P̂, D) separately for correct and incorrect samples.
2. **Bin-Mean-Shift** for binned scores. Build a 2-D quantile binning over (D, P̂), then set P̂ ← P̂ + λ (Acc(B) − Conf(B)) with λ = 0.5.
3. **Theorem 5.1:** the Brier score after Bin-Mean-Shift is asymptotically ≤ the Brier score before it.

**Experiment design.** [verified: https://arxiv.org/html/2306.04590 §6, App. C]
- Balanced: ImageNet, Yahoo Answers Topics, MultiNLI-match.
- Long-tail: iNaturalist 2021, ImageNet-LT.
- Distribution shift: ImageNet-C, MultiNLI-mismatch, ImageNet-Sketch.
- 8 baselines: Conf, TS, ETS, PTS, PTSK, HB, IR, MIR; ProCal is applied *on top of* each.
- Metrics: ECE, ACE, MCE, PIECE; significance at p < 0.1.

**Paper architecture.** [verified: https://arxiv.org/html/2306.04590]
- §1 Intro, with a Fig. 1 teaser comparing high- and low-proximity reliability; contributions are stated as **Findings / Metrics / Method Effectiveness**.
- §2 Related.
- §3 "What is Proximity Bias?" (phenomenon plus 4 findings at scale).
- §4 Metric (PIECE plus theorem).
- §5 Method (two variants plus theorem).
- §6 Experiments, opening with a list of questions (performance across settings, efficiency, sensitivity, ablation), grouped by balanced / long-tail / shift.
- §7 Conclusion.
- Appendix: proofs, setup, extra findings, efficiency, ablations, pseudo-code.
- Counts: 14 figures and 7 tables (main body Figs 1–3, Tables 1–2).

**Borrow for LLM4Rec (the single most important template).** [unverified: our analysis]
1. Replace "proximity" with **item popularity** (log interaction count) or **user activity / embedding density**. This yields a *popularity bias in confidence*.
2. Hypothesis test: confidence-matched head vs tail items, Wilcoxon test, Bias Index.
3. Report a **popularity-informed ECE** (a PIECE analogue) to expose cancellation, where overconfidence on niche items offsets underconfidence on popular ones.
4. Use Bin-Mean-Shift over (popularity, confidence) as a cheap post-hoc fix and baseline.
5. Prediction from ProCal: niche items may get **lower raw confidence but still be overconfident** relative to their accuracy. Our "popular = confident, niche = uncertain" hypothesis must be tested on two axes: confidence level and calibration gap.

**Gap.** ProCal covers vision and text classifiers only. No LLMs, no recommendation, no feedback loop or echo chamber.

---

### A4. UNIT: Uncertainty-Aware Instruction Fine-Tuning (arXiv 2502.11962, 2025)

Earlier title: "Navigating the Helpfulness-Truthfulness Trade-Off with Uncertainty-Aware Instruction Fine-Tuning". Code: github.com/AndrewWTY/UNIT (33★).
[verified: https://arxiv.org/abs/2502.11962 ; https://scholar.google.com/citations?hl=en&user=ErEL3bgAAAAJ ; https://github.com/AndrewWTY/UNIT]

**Problem formulation.** The paper is framed as two RQs. [verified: https://arxiv.org/html/2502.11962]
- **RQ1:** does unfamiliar knowledge in human-written instruction fine-tuning data reduce truthfulness?
- **RQ2:** how can high-quality instruction data be used without hurting truthfulness?

**Method.** [verified: https://arxiv.org/html/2502.11962 §3.1, §4.1]
1. Extract atomic claims from each training response.
2. Score each claim with **CCP** (claim-conditioned probability) under the *target* model: CCP_claim = 1 − ∏_{x ∈ C} CCP(x).
3. Threshold at **τ = 75th quantile** of all claim scores. Claims above τ are "uncertain/unfamiliar".
4. **UNIT_cut:** drop the uncertain claims and have an LLM rewrite the response from the certain claims only.
5. **UNIT_ref:** keep the full response and append a "reflection" listing the uncertain claims.

**Experiment design.** [verified: https://arxiv.org/html/2502.11962 §3.2–3.3]
- Training data: LIMA and LFRQA, with LFRQA subsets varied from 10% to 100%, each also augmented with LIMA.
- Backbones: Llama-3.1-8B and Qwen2.5-14B, full fine-tuning for 3 epochs.
- Truthfulness: FactScore (500 biographies) and WildFactScore (500 entities).
- Informativeness: GPT-4o pairwise win rate.
- Also reports CCP balanced accuracy, honesty balanced accuracy, and Wilcoxon signed-rank tests.

**Findings.** UNIT_cut substantially increases truthfulness, at some informativeness cost. UNIT_ref keeps informativeness while flagging uncertain claims. [verified: https://arxiv.org/abs/2502.11962]

**Paper architecture.** Sections are titled by RQ: "§3 RQ1: Does unfamiliar knowledge… ?" and "§4 RQ2: …", each with its own Methodology / Data / Evaluation / Results subsections. Counts: 2 figures and 9 tables. [verified: https://arxiv.org/html/2502.11962]

**Borrow for LLM4Rec.** [unverified: our analysis]
- This is the closest Hooi-group template for "pruning noisy fine-tuning data by uncertainty".
- Recommendation analogue:
  1. Compute the base LLM's familiarity or confidence on each (history → target) training pair, e.g. P(Yes) for positives and P(No) for negatives, or the target-item likelihood.
  2. Mark the top-25% most uncertain pairs as noisy or unfamiliar, keeping the quantile threshold.
  3. **cut** them, or **ref**lect: train the model to output low confidence on them.
  4. Measure ranking accuracy, calibration, and tail exposure.
- Also adopt the "vary subset size 10%–100%" design and Wilcoxon tests.

**Gap.** UNIT targets long-form factual QA. It has no ranking metrics, and it does not separate *noise* (mislabeled interactions) from *rare-but-true* signal (genuine niche taste). A recommendation version must avoid pruning away the long tail.

---

### A5. Fact or Guesswork? MKJ (arXiv 2502.14275, 2025)

Authors: Jiaxi Li, Yiwei Wang, Kai Zhang, Yujun Cai, **Hooi**, Nanyun Peng, Kai-Wei Chang, Jin Lu. Code: not found.
[verified: https://arxiv.org/abs/2502.14275]

**Problem formulation.** Isolate parametric medical knowledge using **binary true/false one-hop judgments** built from UMLS triplets. Each item has 1 positive and k = 3 negative statements. Accuracy is decomposed as Acc = Acc_pos/(k+1) + k·Acc_neg/(k+1). [verified: https://arxiv.org/html/2502.14275 §3–4.2]

**Experiment design.** [verified: https://arxiv.org/html/2502.14275 §4.1]
- About 20 LLMs:
  - GPT-3.5, GPT-4o-mini, GPT-4o
  - Claude-3 Haiku and Sonnet
  - Llama-3.x (1B–8B)
  - Ministral-8B
  - Qwen2.5 (0.5B–3B)
  - Phi-3 / 3.5
  - Meditron-7B, MeLlama-13B
- Zero-shot, temperature 0.
- Calibration curves with M = 20 bins, ECE, and RAG (sparse and dense retrieval) as a mitigation.

**Findings.** [verified: https://arxiv.org/html/2502.14275 §4.3–4.5]
- Most LLMs have a **negative Acc_gap (Acc_pos − Acc_neg)**, i.e. a "skepticism" bias toward rejecting statements. Examples: Llama-3 series −0.38, Ministral −0.31.
- Calibration is poor, with both over- and underconfidence.
- Accuracy correlates with **term frequency**: low-frequency semantic types are worse ("long-tail knowledge").
- RAG raises accuracy, e.g. GPT-4o-mini Acc_pos +25 points.

**Paper architecture.** [verified: https://arxiv.org/html/2502.14275]
- Experiments are written as **RQ1–RQ4 subsection titles**:
  - RQ1: to what extent can LLMs judge accurately?
  - RQ2: how well are they calibrated?
  - RQ3: why do they fail?
  - RQ4: what strategies help?
- Counts: about 9 numbered figures (many sub-plots) and 5 tables.

**Borrow for LLM4Rec.** [unverified: our analysis] This is the closest analogue of yes/no LLM4Rec (TALLRec-style "Will the user like item X? Yes/No"). Copy:
1. The Acc_pos vs Acc_neg decomposition, to test whether LLM recommenders are biased toward "No".
2. Calibration curves of P(Yes).
3. Accuracy and calibration vs **item frequency** (popularity) as the RQ3 "why" analysis.
4. The RQ1–RQ4 ladder as an empirical-study skeleton.

**Gap.** No fine-tuning, no ranking, no user modeling.

---

### A6. Test-Time Scaling Is Not Effective for Knowledge-Intensive Tasks Yet (COLM 2026)

arXiv 2509.06861 · code: github.com/XuZhao0/tts-knowledge (9★)
[verified: https://arxiv.org/abs/2509.06861 ; https://github.com/XuZhao0/tts-knowledge]

**Problem formulation.** Does more test-time compute help closed-book factual recall? [verified: https://arxiv.org/html/2509.06861]

**Experiment design.** [verified: https://arxiv.org/html/2509.06861 §3.1]
- 14 reasoning models: GPT-5 family, o3-mini, o4-mini, gpt-oss-20b, Grok-3 mini, Gemini 2.5, Claude Sonnet 4, R1-Distill, Qwen3.
- 3 benchmarks: SimpleQA, FACTS Parametric, FRAMES.
- Scaling knobs: reasoning effort, thinking budget, budget forcing.
- Metrics: accuracy, hallucination ratio, F-score, with abstention allowed. The GPT-4o-mini grader disagreed with humans in only 2 of 300 cases.

**Findings.** [verified: https://arxiv.org/abs/2509.06861 ; https://arxiv.org/html/2509.06861 §3–5]
- More thinking rarely improves accuracy and often **increases hallucination**.
- Changes are driven by **willingness to answer**: 85–95% of new hallucinations come from previously abstained questions.
- Long reasoning shows **confirmation bias** and an inflation of verbalized confidence ("maybe 2005" → "I am fairly sure it's 2005").
- An information-theoretic theorem: compute-only scaling is post-processing of a fixed model and cannot add information about arbitrary (long-tail) facts.

**Paper architecture.** [verified: https://arxiv.org/html/2509.06861]
- §3 is phrased as a question: "How does TTS affect accuracy and hallucination?"
- §4 is phrased as a question: "Why…?"
- §5 is the theory section.
- Counts: 9 figures and 11 tables (mostly case-study traces).

**Borrow for LLM4Rec.** [unverified: our analysis]
- Reasoning-augmented LLM recommenders (CoT or R1-style) may produce **more confidently-wrong recommendations**, especially for niche items, where the "fact" is the user's idiosyncratic preference.
- Measure recommend-vs-abstain rates and confidence drift with reasoning length.
- The information-theoretic argument supports the claim that "more reasoning cannot recover preference signal absent from the history".

---

### A7. RecLAIF (KDD 2025 Workshop on Online and Adaptive Recommender Systems)

Authors: Xiaoxin He (NUS), Nurendra Choudhary, Jieyi Jiang, Edward W. Huang (Amazon), **Bryan Hooi**, Xavier Bresson (NUS), Karthik Subbian (Amazon). 10 pages. No official code; only a third-party re-implementation was found (rajeevrpandey/RecLAIF-implementation, 0★).
[verified: https://nurendra.com/papers/KDDW2025_1.pdf ; https://scholar.google.com/citations?hl=en&user=ErEL3bgAAAAJ]
The original workshop link oars-workshop.github.io/papers/He2025.pdf returned 404 on 2026-10-01. [verified]

**Framework.** [verified: https://nurendra.com/papers/KDDW2025_1.pdf §3–4]
- Recommender: Mistral-7B-Instruct, first SFT-distilled from Claude 3 Sonnet outputs. It outputs *key features (preferences) + top-k items with reasons*.
- Judge: Claude 3 Sonnet. It scores two sampled outputs (high temperature, nucleus sampling) on **relevance, diversity, explainability** (0–5 each, chain-of-thought style) and picks chosen vs rejected.
- Training: **iterative DPO**, Algorithm 1, for N iterations.

**Experiment design.** [verified: same PDF §5]
- ESCI: retrieval, 15 candidates → top-5.
- Amazon Beauty 2023: sequential, 253 users, 341 items, 10 candidates.
- LastFM: sequential, 1,220 users, 4,606 items, 20 candidates.
- Baselines: BM25, BLAIR, SASRec, Mistral-7B, Llama-3-8B, Qwen2.5-7B, Claude 3 Sonnet.
- Metrics:
  - Prec@5, NDCG@5 (ESCI)
  - Hit@1, NDCG@3 (Beauty, LastFM)
  - Diversity as mean pairwise Sentence-Transformer cosine distance
  - Explainability score (Claude 1–100)
  - Validity ratio
- Ablation: DPO iterations 1–4; gains saturate around iteration 3.

**Results.** [verified: same PDF, Tables 2–3]

| Dataset | Metric | RecLAIF | Claude 3 Sonnet |
|---|---|---|---|
| ESCI | NDCG@5 | 0.8279 | 0.8059 |
| Beauty | Hit@1 | 0.3874 | 0.2806 |
| LastFM | Hit@1 | 0.5126 | 0.4080 |

**Architecture.** §1 Intro (Fig. 1 example) → §2 Related → §3 Formalization → §4 Method (4.1 Recommender / 4.2 Judge / 4.3 DPO) → §5 Experiments (5.1 Retrieval / 5.2 Recommendation / 5.3 Ablation) → §6 Conclusion. 3 figures, 3 tables. [verified: same PDF]

**Borrow / gap.** [unverified: our analysis]
- This is the group's *only* LLM4Rec generation paper, and it has **no confidence, calibration or popularity analysis**.
- Its judge-scored *diversity* is the only bias-adjacent signal.
- Its datasets (Beauty, LastFM, ESCI with sampled candidate sets) and LLM baselines are a reasonable minimum scale, but too small for SIGIR. Beauty has only 253 users.
- Gap we can fill: uncertainty-aware preference construction. Use confidence to pick DPO pairs, or penalize *confidently wrong* chosen responses in the spirit of UGR (§8).

---

## 4. Tier-B cards (condensed)

**B1. NeighborAgg: Birds of a Feather Trust Together** (TMLR 08/2022; OpenReview p5V8P2J61u; code MiaoXiong2320/NeighborAgg)
[verified: https://arxiv.org/abs/2211.16466 ; https://openreview.net/forum?id=p5V8P2J61u ; https://arxiv.org/html/2211.16466]
- **Method:**
  - Trustworthiness score t = Agg(W_h h, W_p p). Here h is a class-wise kNN similarity vector and p is the classifier softmax; the score is trained with NLL.
  - Theorem: this is a generalized 1-hop GCN.
  - Extension NeighborAgg-CMD (**conformal mislabel detection**): reliability r = (2·1[ŷ = y] − 1)·t_y, with threshold τ_α = r_(B_α), B_α = ⌈(N+1)(1−α) + αNp⌉. This gives a noise-robust coverage guarantee: a correctly labeled sample scores above τ with probability ≥ 1 − α.
- **Experiments:**
  - Data: CIFAR10, FashionMNIST, MNIST; UCI LetterRecognition, Landsat, CardDefault.
  - Base classifiers: LR, RF, MLP, shallow CNN, ResNet18/50.
  - Baselines: MSP, temperature scaling, TCP, Trust Score, Top-label calibration, GCN-khop.
  - Metrics: AUC-ROC, APM, APC; 5 seeds.
  - The experiments section opens with 6 questions (mechanism, effectiveness, ablation, sensitivity, cost, case study).
- **Borrow** [unverified: our analysis]:
  - Neighborhood-aware trust maps well to collaborative filtering: similar users or items should agree. A "CF-neighborhood agreement" signal can complement LLM confidence.
  - The CMD conformal threshold gives a *guaranteed* rule for pruning noisy interactions, with false-removal rate ≤ α.

**B2. ConfMAD** (EMNLP 2025 Findings) [verified: https://arxiv.org/abs/2509.14034 ; https://bhooi.github.io/]
- **Method:** agents express confidence in two ways:
  - LN: key-token length-normalized sequence probability, seqprob^{1/n}.
  - SV: self-verbalized 0–100.
  - Confidences are calibrated with Platt scaling, histogram binning, or temperature scaling before the debate.
- **Benchmarks:** BIGGSM, BBH, MMLU, MATH.
- **Structure:** **RQ1–RQ4** — performance vs baselines; effect on individual LLMs; why it helps; ablation of the calibration scheme.
- **Counts:** 11 figures and 11 tables.
- **Key insight:** uncalibrated confidence causes stubbornness or premature consensus. Calibrated confidence improves the "win rate" (the correct agent having higher confidence). [verified: https://arxiv.org/html/2509.14034]
- **Borrow** [unverified: our analysis]: the "win rate" metric becomes, in recommendation, the probability that the correct item gets higher confidence than a wrong item, i.e. the confidence-AUROC we need for "does the LLM know when it is right".

**B3. Uncertainty of Thoughts (UoT)** (NeurIPS 2024) [verified: https://arxiv.org/abs/2402.03271 ; https://arxiv.org/html/2402.03271 §2–3]
- **Method:**
  - Simulate answer trees over a possibility set Ω.
  - Reward each question by expected information gain, IG = −p_A log p_A − p_N log p_N, sharpened by λ.
  - Propagate accumulated and expected rewards; pick the arg-max question.
- **Setup:** 5 recent LLMs (Llama-3-70B, Mistral-Large, Gemini-1.5-Pro, Claude-3-Opus, GPT-4) plus older ones; baselines DP, PP, CoT, CoT-SC, Reflexion, ToT; 20Q, medical diagnosis, troubleshooting.
- **Metrics:** success rate (SR), mean successful conversation length (MSC), mean conversation length (MCL). Average success-rate gain: +38.1%.
- **Borrow** [unverified: our analysis]: uncertainty-aware *preference elicitation* in conversational recommendation. Ask the question that halves the plausible item set, and stop when entropy is low.

**B4. Primacy Effect of ChatGPT** (EMNLP 2023) [verified: https://arxiv.org/abs/2310.13206]
- ChatGPT's choice is order-sensitive and favors earlier labels.
- **Borrow** [unverified: our analysis]: LLM rankers over candidate lists inherit position bias. Confidence measured on a single ordering is confounded, so permutation-consistency is a natural uncertainty signal and a required control.

**B5. GraphCleaner** (ICML 2023) [verified: https://arxiv.org/abs/2306.00015]
- **Method:** synthetic mislabel generation plus neighborhood-aware detection.
- **Results:** +0.14 F1 and +0.16 MCC over the closest baseline; at least 6.91% of PubMed is mislabeled; removing those labels raises accuracy from 86.71% to 89.11%.
- **Borrow** [unverified: our analysis]: inject *synthetic noisy interactions* into recommendation training data to get ground truth for evaluating an uncertainty-based pruner, then report clean-up gains.

**B6. Great Models Think Alike** (ICML 2023) [verified: https://arxiv.org/abs/2305.01481]
- Reliability is measured as neighborhood agreement between the model's latent space and a foundation model's latent space, then fused into confidence for failure detection.
- **Borrow** [unverified: our analysis]: agreement between the LLM recommender's top-K and a CF model's top-K as a black-box confidence signal.

**B7. Trust, but Verify** (ECCV 2022) [verified: https://arxiv.org/abs/2302.02628]
- Self-supervised probing to detect and mitigate overconfidence.
- Tasks: misclassification detection, calibration, OOD detection.

**B8. Towards a Unified View of Answer Calibration** (NLRSE 2024 workshop) [verified: https://arxiv.org/abs/2311.09101]
- A taxonomy of step-level vs path-level answer calibration (self-consistency etc.); combining both works best.

**B9. Exploring Collaboration Mechanisms for LLM Agents** (ACL 2024) [verified: https://arxiv.org/abs/2310.02124]
- Agents with an "overconfident" vs "easy-going" trait show conformity and consensus, mirroring social psychology.
- **Borrow** [unverified: our analysis]: conformity is a mechanism analogue for echo chambers.

**B10. Rewarding the Rare** (ACL 2026 Findings) [verified: https://arxiv.org/abs/2601.08763]
- RL post-training suffers exploration collapse. The fix reweights advantages inversely to the size of the LLM-judged strategy cluster; pass@k and AUC@K improve without hurting pass@1.
- **Borrow** [unverified: our analysis]: an anti-collapse rule for RL-tuned LLM recommenders. Reward correct-but-rare (niche) recommendations to counter confidence-driven popularity concentration.

**B11. RealPref** (EMNLP 2026 Findings) [verified: https://arxiv.org/abs/2603.04191 ; https://bhooi.github.io/]
- 100 synthetic users, 1,300 preferences, 4 expression types (explicit → implicit).
- Task types: multiple-choice, true/false, open-ended.
- Performance drops with context length and implicitness.
- **Borrow** [unverified: our analysis]: implicit-preference users are a natural "high-uncertainty" stratum.

**B12. FineVerify** (EMNLP 2026) [verified: https://arxiv.org/abs/2606.00660]
- Score-based best-of-N selection depends on calibration. Decomposing into checkable sub-questions turns selection into local judgments.
- GPT-5-mini gains +8.2 points with 4 samples.

**B13. ARIA** (EMNLP 2025 Industry) [verified: https://arxiv.org/abs/2507.17131]
- The agent assesses its own uncertainty by structured self-dialogue and asks human experts.
- Deployed in TikTok Pay (150M+ MAU).

**B14–B16. LLM-as-judge reliability:**
- JudgeLRM, RL-trained judges [verified: https://arxiv.org/abs/2504.00050]
- BITE: a LinUCB style attack inflating judge scores by 1–2 points on a 9-point scale, with over 65% attack success [verified: https://arxiv.org/abs/2605.26156]
- First-speaker bias in sequential debate [verified: https://arxiv.org/abs/2609.38964]
- **Relevance** [unverified: our analysis]: if we use an LLM judge for explanation or diversity (as RecLAIF does), judge bias is a threat to validity to discuss.

**B17. OKGQA** (ACL 2025) [verified: https://arxiv.org/abs/2410.08085]
- KG-augmented LLM hallucination benchmark, including a perturbed-KG variant (OKGQA-P).

---

## 5. Paper-architecture synthesis (what to imitate for SIGIR)

| Paper | Body skeleton | RQ style | #Fig / #Tab | Scale |
|---|---|---|---|---|
| A1 ICLR'24 | Intro (5 numbered findings) → Framework → Setup → **finding-titled** analysis subsections → Discussion; appendix holds a practitioner "best practice" section | Implicit; sections are findings | 8 / 18 | 5 LLMs × 8 datasets × 5 prompts × 3 samplers × 3 aggregators |
| A2 NeurIPS'25 | Intro (central RQ in bold) → Background → Method (with theorem) → **question-titled** experiment subsections → Related → Conclusion | 2 big questions | ~6 / 19 | 3 backbones × 5 datasets × 4 baselines; 2 downstream uses |
| A3 NeurIPS'23 | Intro (Findings / Metric / Method) → **phenomenon section** → **new metric** + theorem → **method** + theorem → Experiments (opens with a question list) | Question bullet list | 14 / 7 | **504 models**, 8 datasets, 8 baselines, 4 metrics, significance tests |
| A4 arXiv'25 | Intro (RQ1, RQ2) → **§3 = RQ1**, **§4 = RQ2**, each with method / data / eval / results | Section-level RQs | 2 / 9 | 2 LLMs × 2 training sets × 10 subset ratios |
| A5 arXiv'25 | Dataset construction → Experiments with **RQ1–RQ4 subsection titles** | Subsection RQs | ~9 / 5 | ~20 LLMs |
| A6 COLM'26 | "How does…?" → "Why…?" → information-theoretic limit | Question sections | 9 / 11 | 14 models × 3 benchmarks |
| B2 EMNLP-F'25 | Framework → Experiments RQ1–RQ4 | RQ list | 11 / 11 | 4 benchmarks, 3–4 LLMs |
| A7 KDD-W'25 | Formalization → Method (3 parts) → Exp (3 parts) | none | 3 / 3 | 3 datasets, 1 backbone |
| C1 SIGIR'26 | Intro → Prelim → Related → Method (4 parts plus complexity) → Exp (setup, main, **7 analyses**) | none | ~10 (32 panels) / 3 | Baby, Sports, Clothing; R@10/20, N@10/20 |

Sources: [verified: https://arxiv.org/html/2306.13063 ; https://arxiv.org/html/2508.18847 ; https://arxiv.org/html/2306.04590 ; https://arxiv.org/html/2502.11962 ; https://arxiv.org/html/2502.14275 ; https://arxiv.org/html/2509.06861 ; https://arxiv.org/html/2509.14034 ; https://nurendra.com/papers/KDDW2025_1.pdf ; https://arxiv.org/html/2605.00670]

**Recurring "Hooi-group recipe"** (synthesized from A1, A3, A2, A4) [unverified: our analysis]:
1. Discover a phenomenon at scale (hundreds of models, or many LLMs × datasets), backed by a hypothesis test.
2. Name it and give it a metric with a theorem showing the old metric hides it (PIECE ≥ ECE).
3. Propose a simple plug-and-play fix with a proper-scoring or Brier guarantee.
4. Run experiments as an explicit question list, including downstream utility (cascade, self-correction) and efficiency.
5. Put the bulk of the material in a heavy appendix with proofs and a practitioner guide.

**Proposed SIGIR skeleton (about 9 pages plus references)** [unverified: our analysis]:
- §1 Intro.
  - Fig. 1: a confidently-wrong popular recommendation vs a correctly-but-low-confidence niche one.
  - 3–4 numbered findings.
  - Contributions: Findings / Metric / Method.
- §2 Preliminaries: LLM4Rec settings (yes/no pointwise; candidate-list ranking; generative); confidence sources.
- §3 Empirical study, "Do LLM recommenders know when they are right?":
  - RQ1 calibration and failure prediction.
  - RQ2 confidently-wrong analysis.
  - RQ3 popularity-conditioned confidence and calibration, with Wilcoxon test and Bias Index.
  - RQ4 feedback-loop / echo-chamber simulation.
  - Scale target: at least 3 datasets (e.g. ML-1M, Amazon Beauty/Books, LastFM or Steam), at least 4–6 backbones (open-weight 1.5B–8B fine-tuned plus 2 API models), 3–5 elicitation methods.
- §4 Metric: popularity-informed calibration error, with an ECE ≤ PIECE-style theorem.
- §5 Method: (a) a calibration-aware fine-tuning loss (Brier on P(Yes), or tokenized Brier); (b) uncertainty-based data pruning (UNIT_cut-style quantile, or conformal threshold à la NeighborAgg-CMD).
- §6 Experiments:
  - RQ5 effectiveness: NDCG/HR + ECE/AUROC/Brier + tail coverage, Gini, APLT.
  - RQ6 downstream: cascade to CF, selective recommendation.
  - RQ7 ablations: threshold quantile, loss weight, pruning ratio.
  - RQ8 efficiency.
- Budget: 5–7 figures and 4–6 tables in the main body.

---

## 6. Mapping our research questions to Hooi-group tools

| Our question | Closest Hooi-group evidence | What to measure (borrowed) |
|---|---|---|
| Does the LLM know when its recommendation is right? | A1 (calibration vs failure prediction), A2, B2 win rate | ECE, Brier, **AUROC / AUPRC-N** of confidence vs hit; reliability diagrams (A1 Fig. 2 style) |
| Are wrong answers low-confidence or confidently wrong? | A1 (80–100% clustering), A6 (confirmation bias), A5 (overconfidence on errors) | Confidence histogram split by correct/incorrect; share of errors with confidence ≥ 0.8; drift with CoT length |
| Yes/no-token confidence vs ground truth | A2 (token-distribution confidence), A5 (Acc_pos vs Acc_neg skepticism) | P(Yes) calibration curve (M = 20 bins); Acc_pos − Acc_neg gap; Spearman correlation of P(Yes) with label |
| Popular high-confidence vs niche low-confidence | **A3 proximity bias** (sparse = *more overconfident*), A5 (low frequency = lower accuracy) | Confidence *and* calibration gap per popularity decile; confidence-matched Bias Index; Wilcoxon test; popularity-informed ECE |
| Does high confidence cause echo chambers or popularity bias? | Nothing in-group on recommendation; analogues B9 (conformity), B2 (premature consensus), B10 (exploration collapse) | Closed-loop simulation that retrains on confident recommendations; track Gini, coverage, tail share over rounds |
| Prune noisy fine-tuning data by uncertainty | **A4 UNIT_cut** (75th-quantile CCP), B1 NeighborAgg-CMD (conformal), B5 GraphCleaner (synthetic noise) | Prune the top-q% uncertain pairs; inject synthetic noise to score precision/recall of the pruner; guard that the tail is not pruned |

The cells above summarize the verified paper cards in §3–4; the measurement choices are [unverified: our analysis].

---

## 7. Gaps we can fill (design implications)

[unverified: our analysis, grounded in §3–4 and §8]

1. **Popularity-informed calibration for LLM4Rec (the main novelty hook).** ProCal's proximity bias has never been studied for LLM recommenders. Ravikumar (2026) stratifies verbalized confidence by popularity for *zero-shot* models and catalog hallucination only (§8). Our angle adds:
   - fine-tuned LLM4Rec;
   - token-level P(Yes) as well as verbalized confidence;
   - **confidence-matched** tests in the PIECE style;
   - a fix.
2. **Two-axis popularity hypothesis.** Test "popular → high confidence" (raw level) *separately* from "niche → overconfident relative to accuracy" (calibration gap). ProCal and MKJ predict the second effect may appear even when the first holds. Either outcome is a publishable finding.
3. **Principled calibration loss for recommendation fine-tuning.** Port ConfTuner's proper-scoring argument to recommendation: Brier on P(Yes) for pointwise models, tokenized Brier for generative models. No group paper does this for recommendation. UGR (KDD'26) does uncertainty-weighted rewards for generative recommendation but, per its abstract, not a proper-scoring guarantee. Check this against the UGR paper itself.
4. **Uncertainty-based pruning that protects the long tail.** UNIT_cut prunes "unfamiliar" knowledge, which in recommendation may be legitimate niche taste. A popularity-stratified or conformal (NeighborAgg-CMD) threshold that bounds false removal is an unaddressed design point.
5. **Confidence in a feedback loop (echo chambers).** The group only touches conformity in multi-agent LLM debate (B2, B9, B16) and exploration collapse (B10). A closed-loop simulation showing whether confidence-gated recommendation amplifies popularity, and whether calibration plus tail-aware rewards mitigate it, is open.
6. **Reasoning-augmented LLM4Rec.** A6's willingness-to-answer and confirmation-bias effects have not been examined for CoT/R1-style recommenders; this could be a short RQ.
7. **Experimental scale.** To match the group's standard, aim for:
   - at least 3 recommendation datasets;
   - at least 5 backbones (mixing sizes and open vs API);
   - at least 4 confidence-elicitation baselines: P(Yes) token, verbalized, self-consistency (A1), Top-K/Pair-Rank (A1);
   - plus LLM4Rec UQ baselines from Kweon WWW'25 and EviRank, and calibration baselines (temperature scaling, histogram binning, Bin-Mean-Shift);
   - significance tests (Wilcoxon as in A3 and A4).

---

## 8. Appendix: non-Hooi LLM4Rec-uncertainty papers found during search (competitors)

None of these papers have Hooi as an author.

| Paper | Venue | Key point | Source |
|---|---|---|---|
| Uncertainty Quantification and Decomposition for LLM-based Recommendation (Kweon, Jang, Kang, Yu) | WWW 2025 | Predictive uncertainty indicates recommendation reliability; decomposes it into recommendation vs prompt uncertainty; proposes uncertainty-aware prompting; code WonbinKweon/UNC_LLM_REC_WWW2025 (3★) | [verified: https://arxiv.org/abs/2501.17630 ; https://dl.acm.org/doi/10.1145/3696410.3714601] |
| Uncertainty-aware Generative Recommendation, UGR (Fan, Gao, Gong, Liu, Feng, Xiangnan He) | KDD 2026 | Combines an uncertainty-weighted reward (penalizes confident errors), difficulty-aware optimization, and explicit confidence alignment; code cxfann/UGR (7★) | [verified: https://arxiv.org/abs/2602.11719 ; https://github.com/cxfann/UGR] |
| EviRank: Evidence-Based Confidence Estimation for LLM-Based Ranking (Yan et al.) | arXiv 2026-06 | Position-level confidence for LLM ranking; three evidences aggregated, position-aware calibration, used to optimize ranking | [verified: https://arxiv.org/abs/2606.04727] |
| Do LLM Recommenders Know When They're Hallucinating? (Ravikumar) | arXiv 2026-08 | Audits OOD@10 and verbalized ECE/Brier for 4 zero-shot LLMs on MovieLens-25M, Amazon Toys and Yelp, **stratified by popularity**. Confidence is near-constant per model; a conformal abstention threshold barely helps. Closest to our RQs. | [verified: https://arxiv.org/abs/2608.10008] |
| SPECTRA: distributional LLM inference over preferences | arXiv 2025 | Softmax probing over category tokens recovers long-tail preferences | [verified: https://arxiv.org/abs/2509.24189] |
| Echoes in Filter Bubble: popularity bias in generative recommenders (Ghost) | arXiv 2026 | Popular items make up more than 97% of recommendation lists; asymmetric unlikelihood loss plus tokenization fix | [verified: https://arxiv.org/abs/2605.16825] |
| BiasRecBench: LLM-as-a-Recommender agents hacked by biases | arXiv 2026 | Injected contextual biases flip agent choices | [verified: https://arxiv.org/abs/2603.17417] |

**Mis-attributions caught during search.** Some search-engine summaries attributed these papers to Hooi; arXiv metadata shows he is not an author:
- DINCO, "Calibrating Verbalized Confidence with Self-Generated Distractors" (Wang & Stengel-Eskin) [verified: https://arxiv.org/abs/2509.25532]
- MetaFaith (Liu et al., EMNLP 2025) [verified: https://aclanthology.org/2025.emnlp-main.1505/]
- CritiCal (Zong et al., HKUST) [verified: https://arxiv.org/abs/2510.24505]

---

## 9. Limitations of this survey

- DBLP could not be queried (bot challenge), so venue cross-checks rely on the homepage and Google Scholar. [verified: attempted https://dblp.org]
- Semantic Scholar splits Hooi into at least 19 author IDs; the merged list may still miss items. [verified: https://api.semanticscholar.org/graph/v1/author/search?query=Bryan%20Hooi]
- Venues still unknown: UNIT and MKJ; "COLM 2026" for TTS-knowledge comes only from the repo description. [unverified]
- Figure and table counts were produced by parsing arXiv HTML captions. They count numbered figures and tables, including the appendix; sub-panels are counted separately only where stated. [verified: arXiv HTML pages listed in §5]
- Papers by students without Hooi as co-author were only spot-checked: the Scholar pages of Miao Xiong and Jiaying Wu, plus web search. [verified: https://scholar.google.com/citations?user=yQ4U_5IAAAAJ ; https://scholar.google.com/citations?user=mrfO62wAAAAJ]
