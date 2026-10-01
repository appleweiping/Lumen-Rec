# Recent literature: uncertainty, confidence, calibration and reliability in LLM-based recommendation (2023–2026)

Compiled 2026-10-01 to position a new SIGIR full paper. Sources: the arXiv API (abstracts and comments), arXiv PDFs (searched as text with pdftotext), ACM DL, ACL Anthology, OpenReview/ICLR virtual pages, GitHub (HTTP 200 checks), and the sigir2026.org / sigir2027.org sites.

**Verification legend**
- `[verified: URL]` means the claim was checked against that page or PDF in this session.
- `[verified: PDF text]` means the claim was read from the arXiv PDF of the paper being described (the arXiv link for that paper is given in its header line).
- `[unverified]` means the claim was not checked in this session, or comes only from a secondary source. Treat it as provisional.
- "Not found" statements mean that a keyword search of the PDF text found nothing. They are absence-of-evidence and should be read with that caveat.

Seeds of our idea (for the "gap vs. ours" lines):
- (S1) Is the LLM confident when it is right, and unconfident when it is wrong?
- (S2) How does yes/no P(yes) confidence correlate with ground truth?
- (S3) Do popular items get high confidence and niche items low confidence, and does confidence drive echo chambers?
- (S4) Prune noisy training data by uncertainty.

Prior internal negative results:
- (N1) In fixed-candidate pointwise reranking, monotone calibration preserves rank, so uncertainty has no effect on ranking.
- (N2) Popularity-aware recalibration gave no gain.
- (N3) Uncertainty-based pair pruning did not beat random pruning in generative DPO-style training (σ_seed ≈ 1.5 pt).

---

## 0. Executive summary

### Closest competitors (the "do-not-overlap" owners)

| # | Paper | Venue | What it owns, so we must not claim it as new |
|---|---|---|---|
| 1 | **UQRec**: Kweon, Jang, Kang, Yu, "Uncertainty Quantification and Decomposition for LLM-based Recommendation" | WWW 2025 | Predictive-uncertainty framework for **listwise** LLM recommendation, computed in a single inference through Plackett–Luce. Decomposes uncertainty into recommendation and prompt components. Shows that uncertainty predicts NDCG (Kendall τ@K, C-index). Proposes uncertainty-aware prompting (choosing the number of history items and candidates per user). Benchmarks label-probability, semantic-uncertainty and verbalized baselines. |
| 2 | **UGR**: Fan, Gao, Gong, Liu, Feng, He, "Uncertainty-aware Generative Recommendation" | KDD 2026 | Uses uncertainty **during RL training**: an uncertainty-weighted reward that penalizes *confident errors*, difficulty-aware optimization, and explicit `[C_HIGH]/[C_LOW]` confidence tokens. Uses confidence downstream for **user-level rejection**, **item-level truncation** and confidence reranking. States that "absolute calibration is less meaningful" in recommendation. |
| 3 | **EviRank**: Yan, Xu, Wang, Guan, Zhao, "Evidence-Based Confidence Estimation for LLM-Based Ranking" | arXiv 2606.04727 (the PDF header says MM'26; acceptance not confirmed) | **Position-level** confidence for LLM ranking lists, built by aggregating evidence. Position-aware calibration toward an NDCG-discount target. Confidence-guided reranking. Shows that label-probability confidence is flat and under-confident across positions. |
| 4 | **Ravikumar**, "Do LLM Recommenders Know When They're Hallucinating? Auditing Confidence Calibration in Catalog Faithfulness" | CIKM 2026 (short) | **Verbalized-confidence ECE and Brier** of *zero-shot* frontier LLM recommenders against catalog membership, **stratified by popularity**. Finds that the catalog sets whether models are over- or under-confident. A split-conformal abstention threshold changes hallucination by at most 1.65 pp. |
| 5 | **KnowSA_CKP**: Lee, Jang, Kang, Yu, "Filling the Gaps: Selective Knowledge Augmentation for LLM Recommenders" | SIGIR 2026 (full) | Per-item estimates of what the LLM knows, through comparative knowledge probing. Shows that **popularity, Min-K% and EigV-uncertainty proxies align poorly with Recall@1**. Augments only items with low estimated knowledge (selective augmentation). |
| 6 | **GORACS** (Mei et al., KDD 2025) and **DEALRec** (Lin et al., SIGIR 2024) | KDD'25 / SIGIR'24 | **Coreset and data pruning for LLM-recommender fine-tuning.** DEALRec uses influence regularized by effort. GORACS uses group-level optimal transport plus gradients, and reports that GraNd/EL2N (loss- and gradient-based) fall **below random** on TALLRec-style CTR tuning. |

Also near: **SPRec** (WWW'25) and **Flower** (SIGIR'25) own "SFT/DPO amplifies popularity bias in LLM recommenders" and its fixes. **PerK** (WWW'24) owns "calibrated probabilities → personalized list size". **Zou et al.** (KDD'26 ADS) own "model-uncertainty → deboost for low-activity users and UCB exploration for high-activity users" in production (not LLM-based).

### Open gaps (none of these was found as a published contribution in our search)

1. **Calibration of pointwise yes/no LLM recommenders against held-out labels** (TALLRec, CoLLM and BinLLM style). These papers use P(Yes) only as a score for AUC/UAUC. We found no ECE, Brier or reliability analysis in any of them, and no popularity-stratified correctness–confidence analysis for *fine-tuned* yes/no LLM recommenders. Ravikumar does this only for *zero-shot verbalized* confidence on *catalog membership*.
2. **Confidence × popularity × feedback loops.** EchoTrace (CIKM'26) and RecLoop simulate loops of LLM or generative recommenders with no confidence signal. UGR and UQRec model confidence with no loop or popularity analysis. Nobody tests whether high-confidence exposure on popular items drives homogenization, or whether confidence-gated exposure (abstain, fall back, explore) slows it.
3. **Decisions that are not within-list ranking**, as the escape from N1:
   - cross-user selective recommendation, measured with risk–coverage or AURC;
   - abstention with **fallback to a CF/sequential model**;
   - conformal list truncation for LLM recommenders.

   UGR shows user rejection and truncation only on one dataset (Office), with no fallback model, no risk–coverage comparison across UQ estimators, and no guarantee. RouteRec (SIGIR'26 workshop) reports that request-level LLM escalation did **not** help. That is a useful negative result to cite.
4. **A multi-seed study of uncertainty-based data pruning for LLM-recommender preference optimization**, with a hard-vs-noise analysis. The pieces exist separately:
   - GORACS and MiniRec report that importance or coverage selection can fall below random;
   - LLMHNI (AAAI'26) documents "hard–noisy confusion".

   No paper reports seed variance, or shows *when* uncertainty pruning beats random (for example under injected noise) for LLM-recommender DPO.

### SIGIR format facts (verified)

- **SIGIR 2026 full papers** (Melbourne):
  - at most **9 pages including appendices**, plus unlimited references;
  - ACM `sigconf`, using `\documentclass[sigconf,natbib=true,anonymous=true]{acmart}`;
  - double-anonymous review; arXiv preprints allowed;
  - **+1 page in the camera-ready version**;
  - submission through OpenReview;
  - one author per paper must serve on the PC;
  - abstract deadline Jan 15, 2026; full paper Jan 22, 2026 (AoE).
- **SIGIR 2027** (San Jose, July 18–24, 2027):
  - abstract **Jan 14, 2027**, full paper **Jan 21, 2027**, notification Apr 5, camera-ready Apr 30;
  - page limit and policies are still "TBA" or placeholders on the site.

  Details are in §F.

---

## A. Uncertainty quantification in LLM4Rec

### A1. UQRec: Uncertainty Quantification and Decomposition for LLM-based Recommendation
- **Authors / venue:** Wonbin Kweon, Sanghwan Jang, SeongKu Kang, Hwanjo Yu. WWW 2025. arXiv **2501.17630** [verified: https://arxiv.org/abs/2501.17630] [verified: https://dl.acm.org/doi/10.1145/3696410.3714601]
- **Repo:** https://github.com/WonbinKweon/UNC_LLM_REC_WWW2025 (HTTP 200) [verified]
- **Framework:**
  - Estimates the predictive distribution over rankings in a single inference by fitting a **Plackett–Luce** model.
  - Takes total predictive uncertainty as its entropy.
  - Introduces a latent "prompt" variable to split total uncertainty into **recommendation uncertainty** and **prompt uncertainty** (mutual information).
  - **Uncertainty-aware prompting** chooses, per user, the number of history items and candidates that minimizes uncertainty, which improves NDCG with negligible extra tokens. [verified: PDF text]
- **Experimental design:**
  - Datasets: MovieLens-1M, Amazon Grocery, Steam.
  - Backbones: Llama3-8B, Gemma-7B, Mistral-7B (fine-tuned on one A100-80G), plus GPT-3.5-Turbo zero-shot.
  - Task: listwise ranking over candidates retrieved by BPR-MF or at random.
  - Evaluation of the uncertainty measure: **Kendall τ@K and Concordance index C@K** between uncertainty and NDCG@K. These play the role that AUROC plays in classification.
  - UQ baselines: Label Prob., Semantic Uncertainty, Verbalized (1S top-1, and others).

  [verified: PDF text]
- **Gap vs. ours:**
  - Listwise and generative, not pointwise yes/no (S2).
  - Correlates uncertainty with *per-user NDCG*, not item-level correctness calibration (ECE/Brier).
  - No popularity, echo-chamber (S3) or training-data pruning (S4) component.
  - Its use of uncertainty is *prompt adaptation*. That is one per-user decision that escapes N1, and it is theirs.

### A2. UGR: Uncertainty-aware Generative Recommendation
- **Authors / venue:** Chenxiao Fan, Chongming Gao, Yaxin Gong, Haoyan Liu, Fuli Feng, Xiangnan He. **KDD 2026**. arXiv **2602.11719** [verified: https://arxiv.org/abs/2602.11719] [verified: https://dl.acm.org/doi/10.1145/3770855.3817975]
- **Repo:** https://github.com/cxfann/UGR (HTTP 200) [verified]
- **Framework:** Built on GRPO for SID-based generative recommendation, with three mechanisms:
  1. an **uncertainty-weighted reward** that penalizes confident errors, using logit intensity, more than tentative ones;
  2. **difficulty-aware optimization** that re-weights gradient budget toward hard samples to avoid premature convergence;
  3. **explicit confidence alignment**: new `[C_HIGH]/[C_LOW]` tokens trained with a hierarchical margin loss so that confidence ranks correct > partially correct > wrong.

  Downstream (RQ4, Office dataset only), a hybrid score `exp(gen) + 0.5·conf`:
  - **confidence reranking**: NDCG@1 rises from 0.0904 to 0.0933 (+0.003);
  - **user-level rejection**: NDCG@10 rises monotonically as the filter ratio goes from 0% to 50%;
  - **item-level truncation**: precision rises.

  The paper argues that "absolute calibration is less meaningful" in recommendation because of sparsity. [verified: PDF text]
- **Experimental design:**
  - Datasets: Amazon Office, Industrial & Scientific, Yelp, with an 8:1:1 chronological split.
  - Backbone: **Qwen3-8B** with 4-layer SIDs, trained on 8×A100.
  - Baselines: SASRec, BIGRec, SPRec, TIGER, MiniOneRec, ReaRec, R2ec.
  - Metrics: HR@1/5/10, NDCG@5/10, with beam size 10.
  - Seed counts and significance tests: none found in the text. [verified: PDF text]
- **Gap vs. ours:**
  - Owns "confident-error penalty", explicit confidence tokens, rejection and truncation for *generative SID* recommendation.
  - Has **no popularity or echo-chamber analysis** ("popular" appears zero times in the PDF text).
  - No ECE or reliability analysis.
  - No pointwise P(yes).
  - No data pruning: it *re-weights* by difficulty and does not drop samples.
  - No fallback to CF.

  It is the strongest "training-time uncertainty changes ranking" precedent against N1 and N3.

### A3. EviRank: Evidence-Based Confidence Estimation for LLM-Based Ranking
- **Authors / venue:** Meng Yan, Cai Xu, Xujing Wang, Ziyu Guan, Wei Zhao. arXiv **2606.04727** (June 2026) [verified: https://arxiv.org/abs/2606.04727]. Venue: the PDF header carries an "MM'26" ACM template with placeholder text such as "Trovato et al.", and the arXiv comment states no acceptance [unverified venue].
- **Repo:** anonymous link https://anonymous.4open.science/r/EviRank-CDE0 (taken from the PDF; not opened) [unverified].
- **Framework:**
  - Extracts three single-pass evidences (semantic, attention, output) and aggregates them with subjective-logic opinion fusion.
  - Applies **position-aware calibration**: a squared loss pushing position confidence toward an NDCG-discount target.
  - Reranks with a confidence-weighted score.
  - Motivation: label-probability confidence is "uniformly low" and flat across positions, that is, under-confident. [verified: PDF text]
- **Experimental design:**
  - Three datasets, including Amazon Grocery.
  - Candidate generator: GMF.
  - Backbones: Llama3, Mistral, Qwen2.5.
  - Baselines: GMF, SASRec, BERT4Rec, PepRec, RankGPT; UQ baselines are Label Prob., Semantic Unc., Verb. 1S top-1 (the UQRec set).

  [verified: PDF text]
- **Gap vs. ours:**
  - Listwise and position-level; does not cover pointwise P(yes), popularity, feedback loops or pruning.
  - It is a second precedent where a *calibration loss changes the ranking*, not a monotone post-hoc map.

### A4. Ravikumar: Do LLM Recommenders Know When They're Hallucinating?
- **Authors / venue:** Srijith Ravikumar. **CIKM 2026 short**. arXiv **2608.10008** (v3; the author says to cite v2 or later) [verified: https://arxiv.org/abs/2608.10008]
- **Repo:** https://github.com/rsrijith/cikm26-catalog-faithfulness (HTTP 200) [verified]
- **Framework:**
  - Joint audit of the out-of-catalog rate (OOD@10) and **verbalized-confidence calibration (ECE, Brier, reliability)**.
  - Four zero-shot LLMs: Mistral Large, Llama-3.3-70B, GPT-OSS-120B, Claude Sonnet 4.6.
  - Three catalogs: MovieLens-25M, Amazon-2023 Toys, Yelp.
  - **Stratified by popularity quartile** of the target item.
- **Key findings:**
  - Each model keeps a nearly constant confidence level, so the *catalog* decides the sign of miscalibration: all models are under-confident on MovieLens and all are over-confident on Amazon Toys.
  - A split-conformal abstention threshold over verbalized confidence changes hallucination by **≤ 1.65 pp**, because the confidence channel cannot separate correct items from hallucinations.
  - The author recommends catalog-anchored elicitation.

  [verified: PDF text and arXiv abstract]
- **Gap vs. ours:**
  - Zero-shot, verbalized confidence only, and the correctness target is *catalog membership*, not user preference.
  - No fine-tuning, no P(yes), no ranking, no loops.
  - Its negative conformal-abstention result is consistent with our N1 and N2 and is worth citing.

### A5. USD: Uncertainty-Aware Semantic Decoding for LLM-Based Sequential Recommendation
- **Authors / venue:** Chenke Yin, Li Fan, Jia Wang, Dongxiao Hu, Haichao Zhang, Chong Zhang, Yang Xiang. **APWeb 2025** (not a top venue). arXiv **2508.07210** [verified: https://arxiv.org/abs/2508.07210] [verified: https://link.springer.com/chapter/10.1007/978-981-95-5719-6_25 listed in search]
- **Framework:**
  - Clusters items with similar logit vectors into semantic-equivalence groups.
  - Redistributes probability mass within each cluster.
  - Uses cluster-level entropy to control item scoring and **sampling temperature** at inference.
  - Results: six Amazon domains, +18.5% HR@3; also H&M and Netflix. [verified: arXiv abstract]
- **Gap vs. ours:** Uncertainty-aware *decoding* for generative recommendation. Cite it as the "uncertainty-aware decoding" precedent. Low venue tier.

### A6. Ask to Be Sure: Informative Interactions for Confident Multi-Turn LLM Recommendation
- **Authors / venue:** Cedar Site Bai et al. **CIKM 2026**. arXiv **2608.15949** [verified: https://arxiv.org/abs/2608.15949]
- **Framework:** Uses *entropy reduction* over recommendations as a reward (no ground truth needed) to fine-tune a conversational recommender (SFT and DPO) to ask informative questions. Datasets: INSPIRED, ReDial. [verified: arXiv abstract]
- **Gap vs. ours:** Conversational setting. Uses uncertainty as a reward for *information gathering*, not for calibration of recommendations.

### A7. KnowSA_CKP: Filling the Gaps: Selective Knowledge Augmentation for LLM Recommenders
- **Authors / venue:** Jaehyun Lee, Sanghwan Jang, SeongKu Kang, Hwanjo Yu. **SIGIR 2026 full**. arXiv **2604.07825** [verified: https://arxiv.org/abs/2604.07825] [verified: SIGIR 2026 accepted list, https://sigir2026.org/en-AU/pages/program/accepted-papers]
- **Repo:** https://github.com/nowhyun/KnowSA_CKP (HTTP 200) [verified]
- **Framework:**
  - LLM knowledge is uneven because of pretraining exposure (the "knowledge gap").
  - **Comparative Knowledge Probing** scores each item by how well the LLM captures its collaborative relations.
  - Augmentation (adding item attributes to the prompt) is applied only to low-knowledge items.
  - Training-free.
- **Key analysis:** Proxies for "what the model knows" correlate poorly with Recall@1 (Spearman). The proxies tested were:
  - popularity;
  - Min-K% pretraining detection;
  - **uncertainty** (EigValLaplacian);
  - adaptive-RAG (SeaKR).

  The paper states that "popularity alone serves as an imperfect indicator". [verified: PDF text]
- **Experimental design:**
  - Datasets: Amazon-Beauty, Amazon-Gift Cards, ML-1M, Steam.
  - Backbones: Llama-8B, Mistral-7B, Qwen-7B, Qwen-32B.
  - Metric: Recall@K, with context efficiency also measured.
- **Gap vs. ours:**
  - Directly relevant to S3: it already shows that popularity is a weak proxy for LLM knowledge in recommendation, which is consistent with N2.
  - It is a **selective-intervention** (route by knowledge) precedent.
  - The same POSTECH group wrote UQRec. **Expect them as reviewers.**

### A8. Pre-LLM calibration foundations from the same group (still cited by reviewers)
- **Kweon, Kang, Yu, "Obtaining Calibrated Probabilities with Personalized Ranking Models," AAAI 2022.** Post-hoc maps from ranking scores to calibrated preference probabilities, using unbiased ERM under MNAR data. [verified: https://ojs.aaai.org/index.php/AAAI/article/view/20326]. Repo: https://github.com/WonbinKweon/CalibratedRankingModels_AAAI2022 (HTTP 200) [verified]
- **Kweon, Kang, Jang, Yu, "Top-Personalized-K Recommendation" (PerK), WWW 2024.** Picks a *per-user list size* by maximizing expected utility under calibrated probabilities. [verified: https://arxiv.org/abs/2402.16304]
- **Kweon PhD dissertation, "Confidence Calibration for Recommender Systems and Its Applications."** Covers calibration, using teacher confidence for distillation, and adapting the number of presented items. [verified: https://arxiv.org/abs/2402.16325]
- **Sato, "Calibrating the Predictions for Top-N Recommendations," RecSys 2024 (short).** Calibration measured on all items is miscalibrated on the top-N; proposes top-N-focused calibration. [verified: https://dl.acm.org/doi/10.1145/3640457.3688177]
- **Gap vs. ours:** These own "calibrated probabilities → list size and utility". Any list-truncation contribution of ours must be LLM-specific, for example confidence derived from P(yes), verbalized confidence, or multiple samples, and compared against PerK.

### A9. Industrial cross-user use of uncertainty (not LLM-based)
- **Zou, Li, Sun, Guo, Wang, "Uncertainty-Calibrated Recommendations for Low-Active Users," KDD 2026 ADS track.** arXiv **2605.17788** [verified: https://arxiv.org/abs/2605.17788]
- **Framework:** Calibrated model uncertainty drives *differentiated cross-user policies*:
  - a risk-averse **deboost** of uncertain items for low-activity users;
  - **UCB exploration** for high-activity users.

  Validated online on a livestream platform for retention and diversity. [verified: arXiv abstract]
- **Gap vs. ours:** This is the clearest published template for "uncertainty is useful for cross-user and exploration decisions, not for within-list ranking". It is not LLM-based, which leaves an opening for an LLM version.

### A10. Others (lower relevance or tier)
- **Sah et al., "Uncertainty and Fairness Awareness in LLM-Based Recommendation Systems," IASEAI'26.** Entropy-based uncertainty with Gemini 1.5 Flash; fairness across 8 demographic attributes. [verified: https://arxiv.org/abs/2602.02582]
- **Peng et al., "Uncertainty-Aware Explainable Recommendation with LLMs" (2024).** GPT-2 prompt learning for explanations; uncertainty is nominal. [verified: https://arxiv.org/abs/2402.03366]
- **Survey: "Trustworthy Recommendation in the Era of LLMs" (Wang et al., 2026).** Covers 200+ studies, 13 opportunities and 18 challenges. [verified: https://arxiv.org/abs/2606.00540]
- **Survey: "Uncertainty Quantification and Confidence Calibration in LLMs," KDD 2025.** [verified: https://dl.acm.org/doi/abs/10.1145/3711896.3736569]

---

## B. Pointwise yes/no LLM recommendation and probability scores

How P(Yes) is used: the yes/no family scores an item with the LLM's probability of answering "Yes" and evaluates **only discrimination** (AUC, or UAUC averaged per user). **None of TALLRec, CoLLM or BinLLM reports calibration (ECE, Brier, log-loss reliability)** [verified: PDF text search for calibrat/ECE/logloss; all hits were unrelated].

### B1. TALLRec
- **Authors / venue:** Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, Xiangnan He. **RecSys 2023** (DOI 10.1145/3604915.3608857 in the PDF). arXiv **2305.00447** [verified: https://arxiv.org/abs/2305.00447]
- **Repo:** https://github.com/SAI990323/TALLRec (HTTP 200) [verified]
- **Framework:** Alpaca-tuning followed by rec-tuning (LoRA, LLaMA-7B, one RTX 3090). The prompt asks for a binary answer: the instruction output is "Yes" or "No". Few-shot setting (16/64/256 samples). Movie and Book domains, with ratings binarized at a threshold. **AUC** as the metric. In-context ChatGPT/Davinci baselines score about AUC 0.5. [verified: PDF text]
- **P(yes):** The output is restricted to "Yes"/"No" and AUC is computed on the model's score. The exact formula (raw token probability or softmax over Yes/No) was not quoted in the text we extracted [unverified for the exact formula; the CoLLM text below states it explicitly].
- **Gap vs. ours:** No calibration and no popularity stratification. This is the canonical base model for S2.

### B2. CoLLM
- **Authors / venue:** Yang Zhang, Fuli Feng, Jizhi Zhang, Keqin Bao, Qifan Wang, Xiangnan He. **IEEE TKDE 37(5), 2025**. arXiv **2310.19488** [verified: https://dl.acm.org/doi/10.1109/TKDE.2025.3540912] [verified: https://arxiv.org/abs/2310.19488]
- **Repo:** https://github.com/zyang1580/CoLLM (HTTP 200) [verified]
- **P(yes):** "ŷ represents the prediction probability for the label being 1, i.e., the likelihood of answering 'Yes' for LLM". The model is trained with a BCE-style loss on ŷ. [verified: PDF text]
- **Design:**
  - Backbone: Vicuna-7B with LoRA.
  - A CIE module maps CF embeddings (MF, LightGCN, SASRec, DIN) into the token space.
  - Datasets: ML-1M and Amazon-Book.
  - Metrics: **AUC, UAUC** (and NDCG), with warm/cold splits. [verified: PDF text]
- **Gap vs. ours:** BCE training makes P(yes) a natural calibration target, but calibration is never evaluated. Warm/cold splits already exist, which suits S3 stratification.

### B3. BinLLM: Text-like Encoding of Collaborative Information in LLMs for Recommendation
- **Authors / venue:** Yang Zhang, Keqin Bao, Ming Yan, Wenjie Wang, Fuli Feng, Xiangnan He. **ACL 2024**. arXiv **2406.03210** [verified: https://aclanthology.org/2024.acl-long.497/]
- **Repo:** https://github.com/zyang1580/BinLLM (HTTP 200) [verified]
- **Design:**
  - Encodes CF embeddings as binary strings, with optional dot-decimal compression.
  - Prompt ends "Answer with 'Yes' or 'No'".
  - Loss: BCE on the predicted likelihood.
  - Hyperparameters follow CoLLM; Vicuna-7B.
  - Datasets: ML-1M and Amazon-Book. Metrics: AUC, UAUC. [verified: PDF text]
- **Gap vs. ours:** Same as CoLLM.

### B4. LLaRA: Large Language-Recommendation Assistant
- **Authors / venue:** Jiayi Liao et al. **SIGIR 2024**. arXiv 2312.02445 [verified: https://dl.acm.org/doi/10.1145/3626772.3657690]
- **Repo:** https://github.com/ljy0ustc/LLaRA (HTTP 200) [verified]
- **Design:**
  - Hybrid ID+text item tokens with curriculum learning.
  - The task is **choosing the next item from a candidate list** (not yes/no).
  - Datasets: MovieLens(100K), Steam, LastFM.
  - Metrics: **HitRatio@1 and ValidRatio**.
  - Reports "average outcomes of five runs using different random seeds" (stated in the baseline-training paragraph). [verified: PDF text]
- **Gap vs. ours:** Not P(yes). It is still a common baseline and protocol source (the 20-candidate setting).

### B5. A-LLMRec: Large Language Models meet Collaborative Filtering
- **Authors / venue:** Sein Kim, Hongseok Kang, Seungyoon Choi, Donghyun Kim, Minchul Yang, Chanyoung Park. **KDD 2024**. arXiv 2404.11343 [verified: https://dl.acm.org/doi/10.1145/3637528.3671931]
- **Repo:** https://github.com/ghdtjr/A-LLMRec (HTTP 200) [verified]
- **Design:** Freezes the LLM and the CF model and trains only an alignment network. Evaluated for cold and warm items, cold users, few-shot and cross-domain. Task: generative next-item selection from candidates. [verified: search summary and ACM DL]
- **Gap vs. ours:** No probability semantics and no uncertainty.

### B6. "Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction"
- **Authors / venue:** Wang-Cheng Kang, Jianmo Ni, Nikhil Mehta, Maheswaran Sathiamoorthy, Lichan Hong, Ed Chi, Derek Zhiyuan Cheng (Google). arXiv **2305.06474**, May 2023 [verified: https://arxiv.org/abs/2305.06474]. Venue: arXiv only; not confirmed at a top venue [unverified].
- **Design:**
  - Rating prediction on MovieLens-1M and Amazon-Books.
  - Zero-shot, few-shot and fine-tuned models: Flan-T5 from base to XXL, GPT-3 Curie/Davinci, Flan-U-PaLM 540B.
  - Metrics: **RMSE and AUC-ROC**.
  - Finding: fine-tuned LLMs are data-efficient. [verified: PDF text]
- **Gap vs. ours:** No calibration.

### B7. Data-pruning evidence on the yes/no task (cross-reference to D2)
GORACS (KDD'25) runs coreset selection on a **TALLRec CTR-prediction task** (Amazon Games, Food, Movies) and reports test AUC per selection method:
- **GraNd (0.46–0.47) and EL2N (0.45–0.47)** are far below **Random (0.59–0.66)**.

  [verified: PDF text, Table 2 of 2506.04015]

This is direct published support for our N3 pattern on yes/no tuning.

**B-summary gap:** For S2, there is no published reliability or ECE analysis of fine-tuned P(Yes) against held-out labels. There is also none of
- how confidence on correct predictions versus wrong ones differs;
- popularity-stratified calibration;
- AUC versus UAUC as a measure of *cross-user* comparability.

That last point is relevant to N1: a per-user monotone recalibration leaves UAUC unchanged but **can change global AUC and cross-user thresholds**. That is the one ranking-like metric where per-user calibration matters.

---

## C. Popularity bias, echo chambers and fairness in LLM recommenders

### C1. Lichtenberg, Buchholz, Schwöbel, "Large Language Models as Recommender Systems: A Study of Popularity Bias"
- **Venue:** **Gen-IR@SIGIR 2024 workshop**. arXiv **2406.01285** [verified: https://arxiv.org/abs/2406.01285]
- **Design:**
  - Proposes a principled popularity-bias metric.
  - Compares a simple prompted LLM recommender with traditional recommenders on **movie recommendation** (MovieLens; the PDF also lists GPT-4 and Claude model identifiers in its setup [verified: PDF text]).
  - Tests prompt-based mitigations: "match the user's popularity" and "recommend niche/indie".
- **Finding:** The LLM recommender shows **less** popularity bias without mitigation. [verified: arXiv abstract]
- **Gap vs. ours:** Zero-shot, no confidence. A useful counterpoint: LLMs are not always more popularity-biased. **Fine-tuning** is what amplifies popularity (C3, C5).

### C2. IFairLRS: Item-side Fairness of LLM-based Recommendation
- **Authors / venue:** Meng Jiang, Keqin Bao, Jizhi Zhang, Wenjie Wang, Zhengyi Yang, Fuli Feng, Xiangnan He. **WWW 2024**. arXiv 2402.15215 [verified: https://arxiv.org/abs/2402.15215]
- **Repo:** https://github.com/JiangM-C/IFairLRS (HTTP 200) [verified]
- **Design:** Fine-tunes LLaMA on MovieLens and Steam. Item-side fairness is affected both by interaction history and by the LLM's semantic biases. IFairLRS reweights and reranks to calibrate exposure. Introduced the **MGU/DGU** group-unfairness metrics later reused by SPRec and Flower. [verified: abstract; MGU attribution from the SPRec PDF text]

### C3. SPRec: Self-Play to Debias LLM-based Recommendation
- **Authors / venue:** Chongming Gao, Ruijun Chen, Shuai Yuan, Kexin Huang, Yuanqing Yu, Xiangnan He. **WWW 2025**. arXiv **2412.09243** [verified: https://arxiv.org/abs/2412.09243] [verified: https://dl.acm.org/doi/abs/10.1145/3696410.3714524]
- **Repo:** https://github.com/RegionCh/SPRec (HTTP 200) [verified]
- **Framework:**
  - Shows theoretically and empirically that **DPO amplifies popularity bias**: recommendations concentrate on the most popular group.
  - Each self-play round runs SFT and then DPO, with the *previous round's own outputs as negatives*. This re-weights the DPO loss by the model's logits and suppresses over-recommended items.
- **Design:**
  - Datasets: MovieLens, Steam, Goodreads, Amazon CDs & Vinyl.
  - Metrics: NDCG@5, HR@5, DivRatio, ORRatio (over-recommendation), MGU. [verified: PDF text]
- **Gap vs. ours:** Owns "DPO→popularity amplification" and the self-play fix. It does not frame its mechanism as *confidence*, although logit re-weighting is confidence-like. We should position S3 as **confidence diagnostics and decisions**, not as another debiasing loss.

### C4. D3: Decoding Matters
- **Full title:** "Decoding Matters: Addressing Amplification Bias and Homogeneity Issue for LLM-based Recommendation."
- **Authors / venue:** Keqin Bao, Jizhi Zhang, Yang Zhang, Xinyue Huo, Chong Chen, Fuli Feng. **EMNLP 2024 main**. arXiv 2406.14900 [verified: https://aclanthology.org/2024.emnlp-main.589/]
- **Repo:** https://github.com/SAI990323/DecodingMatters (HTTP 200) [verified]
- **Framework:**
  - Length normalization inflates scores of items containing **"ghost tokens"** whose probability is close to 1. This is an amplification bias.
  - D3 disables length normalization on ghost tokens and adds a text-free assistant model to counter homogeneity.
- **Design:** Six Amazon datasets (Instruments, CDs, Games, Toys, Sports, Books). Metrics: HR/NDCG@5/10. [verified: PDF text]
- **Gap vs. ours:** A key warning for any sequence-level confidence. **Token-probability confidence is distorted by ghost tokens and length normalization**, so the confidence definition must account for it.

### C5. Flower: Process-Supervised LLM Recommenders via Flow-guided Tuning
- **Authors / venue:** Chongming Gao, Mengyao Gao, Chenxiao Fan, Shuai Yuan, Wentao Shi, Xiangnan He. **SIGIR 2025**. arXiv 2503.07377 [verified: https://arxiv.org/abs/2503.07377] [verified: https://dl.acm.org/doi/10.1145/3726302.3729981]
- **Repo:** https://github.com/Mr-Peach0301/Flower (HTTP 200) [verified]
- **Framework:** **SFT amplifies popularity bias** through likelihood maximization. Flower replaces SFT with a GFlowNet that propagates token-level rewards, matching the empirical distribution and preserving diversity.
- **Design:**
  - Datasets: Amazon CDs, Video Games, and one more Amazon domain.
  - Backbone: **Qwen2.5-1.5B-Instruct** (3B in the scaling experiment).
  - Metrics: NDCG@5, HR@5, DGU@10, MGU@10, entropy, TTR. [verified: PDF text]
- **Gap vs. ours:** Owns "SFT likelihood → popularity bias, fixed by process supervision".

### C6. RosePO
- **Title:** "RosePO: Aligning LLM-based Recommenders with Human Values" (arXiv) / "Customized Preference Alignment in LLM-Based Recommendation" (journal version).
- **Authors / venue:** Jiayi Liao, Xiangnan He, Ruobing Xie, Jiancan Wu, Yancheng Yuan, Xingwu Sun, Zhanhui Kang, Xiang Wang. arXiv 2410.12519; **ACM TOIS 44(7)** (DOI 10.1145/3833420) [verified: https://arxiv.org/abs/2410.12519] [verified: https://doi.org/10.1145/3833420 via search]
- **Framework:**
  - Rejected-sample strategies: self-hard, semantic-similar, **popularity-aware** (sampling popular items as rejected).
  - A **personalized smoothing factor** predicted by a preference oracle, giving robustness to *uncertain labels* in auto-constructed preference pairs.
- **Design:**
  - Backbone: **Llama-3-8B-Instruct**.
  - Datasets: MovieLens, Steam, Goodreads.
  - Metrics: HR@1/5/10, NDCG@5/10.
  - **Mean of 5 runs with different random seeds.** [verified: PDF text]
- **Gap vs. ours:** Owns "label-uncertainty smoothing in recommender DPO" and "popular items as rejected samples". This is the soft alternative to our hard pruning (N3).

### C7. Other popularity work (2025–2026)
- **FairLRM, "Bridging Semantic Understanding and Popularity Bias with LLMs," WWW 2026.** Decomposes popularity bias into item-side and user-side components through structured prompts. [verified: https://arxiv.org/abs/2601.09478]. Repo: https://github.com/LuoRenqiang/FairLRM (HTTP 200) [verified]
- **NPRec, "Neutralizing Popularity Bias in LLM-based Recommendation via Counterfactual Reasoning Guidelines," 2025.** Inference-time counterfactual guidelines. arXiv only; venue unknown [verified: https://arxiv.org/abs/2503.08051]
- **"Pretraining Exposure Explains Popularity Judgments in LLMs," Mozafari, Piryani, Jatowt, SIGIR 2026 (short).** Uses OLMo and Dolma. LLM popularity priors track **pretraining exposure** more closely than Wikipedia pageviews, including in the long tail. [verified: https://arxiv.org/abs/2605.12382]
- **"Popular but Wrong: Understanding and Mitigating LLM Overconfidence through Knowledge Popularity," Ni, Bi, Guo, Cheng, EMNLP 2026 main** (QA, not recommendation).
  - Confidence is tied to the popularity of the generated answer; popular-but-wrong answers drive overconfidence.
  - Adding knowledge popularity reduces ECE from 0.356 to 0.050.
  - [verified: https://arxiv.org/abs/2505.17537 (comment "EMNLP2026 Main")]. The ECE numbers come from the search snippet [unverified against the PDF].
  - **Relevance:** This is the QA analogue of S3. It suggests that popularity-informed calibration can work when the popularity signal is the *answer's* exposure, not catalog interaction counts. Contrast it with our N2.

### C8. Feedback-loop and echo-chamber simulations with LLM or generative recommenders
- **EchoTrace (Park, Lee, Lee), "Diagnosing Recursive Risks in LLM-Powered Recommender Systems," CIKM 2026.**
  - A role-aware, phase-wise feedback-loop simulation in which the LLM acts as augmenter, profiler or recommender.
  - LLM components **amplify popularity bias**, inject hallucinated signals, and produce polarized, self-reinforcing exposure.
  - [verified: https://arxiv.org/abs/2602.07442]. Repo https://github.com/DongUk-Park/EchoTrace [unverified; not opened]
  - **No confidence signal** in the abstract.
- **RecLoop (Yang et al. 2026), "Do Generative Recommenders Deepen the Information Cocoon?"**
  - Closed loop with LLM user agents. Compares SID-based generative recommenders with SASRec-type models on two Amazon datasets.
  - Generative recommenders show *less* exposure-level cocoon, but concentration grows in SID code space. Effects depend on tokenization and scale.
  - [verified: https://arxiv.org/abs/2606.17707]. Repo https://github.com/Dregen-Yor/RecLoop (HTTP 200) [verified]. arXiv only.
- **ABPO (Kim, Yun, Choo, Park 2026), "Don't Let Bandit Feedback Pull Continual LLM-Recommender Updates Off Target."**
  - Continual GRPO updates from logged bandit feedback, with self-normalized IPS.
  - **Penalties from no-response are tempered by self-certainty**, that is, output-token confidence used as a reliability signal.
  - Five domains from Amazon and MovieLens.
  - [verified: https://arxiv.org/abs/2605.18899]. arXiv only.
- **Survey: Stoecker, Bayer, Weber, "Bias Mitigation for AI-Feedback Loops in Recommender Systems," FAccTRec'25 workshop.** Reviews 24 studies; notes there are few shared simulators. [verified: https://arxiv.org/abs/2509.00109]
- **Gap vs. ours (S3):** No paper links *model confidence* to *loop dynamics*. Open questions include:
  - whether high-confidence slots go disproportionately to popular items;
  - whether confidence-gated exposure (abstain, fall back to CF, or explore) reduces Gini or homogenization over rounds.

  Ravikumar stratifies calibration by popularity but has no loop.

---

## D. Data pruning, denoising and preference optimization for fine-tuning LLM recommenders

### D1. DEALRec: Data-efficient Fine-tuning for LLM-based Recommendation
- **Authors / venue:** Xinyu Lin, Wenjie Wang, Yongqi Li, Shuo Yang, Fuli Feng, Yinwei Wei, Tat-Seng Chua. **SIGIR 2024**. arXiv **2401.17197** [verified: https://arxiv.org/abs/2401.17197] [verified: https://doi.org/10.1145/3626772.3657807]
- **Repo:** https://github.com/Linxyhaha/DEALRec (HTTP 200; the repo title names the SIGIR'24 paper) [verified]
- **Framework:**
  - A surrogate model (SASRec) computes an **influence score** for each sample.
  - The influence score is regularized by an **effort score**, the LLM-side gradient norm.
  - Coverage-enhanced sampling stratifies by group.
- **Design:**
  - Datasets: Amazon Games, MicroLens-50K, Amazon Book.
  - Backends: **BIGRec (LLaMA-7B, LoRA)** and TIGER.
  - Pruning baselines: Random, GraNd, EL2N, CCS, TF-DCon, RecRanker.
  - Metrics: Recall/NDCG@10/20.
  - **3 random seeds; one-sample t-tests at p < 0.01.**
  - Headline: 2% of the data gives +7.1% Recall@10 on Games versus full-data training, with 95% less time. [verified: PDF text]
- **Gap vs. ours:** The SIGIR reference point for LLM-recommender pruning. It treats Random as a "strong baseline". Influence plus effort is the method to beat.

### D2. GORACS: Group-level Optimal Transport-guided Coreset Selection for LLM-based Recommender Systems
- **Authors / venue:** Tiehua Mei, Hengrui Chen, Peng Yu, Jiaqing Liang, Deqing Yang. **KDD 2025**. arXiv **2506.04015** [verified: https://arxiv.org/abs/2506.04015]
- **Repo:** https://github.com/Mithas-114/GORACS (HTTP 200) [verified]
- **Framework:** A proxy objective bounds the test loss using OT distance plus gradient information. A two-stage algorithm does greedy group-level initialization and then refinement.
- **Design:**
  - Datasets: Amazon Games, Food, Movies.
  - Tasks: **SeqRec on BIGRec**, and **CTR prediction on TALLRec**.
  - Baselines: Random, CCS, D2, GraNd, EL2N, MODERATE, FDMat, DEALRec.
  - Metrics: test loss, NDCG@5, HR@10, AUC.
- **Findings:**
  - Importance-based methods (GraNd, EL2N) are biased toward hard samples and generally lose to distribution-based methods.
  - On CTR, GraNd and EL2N fall **below random**. [verified: PDF text]
- **Gap vs. ours:** Published support for N3 on the yes/no task. GORACS does not study DPO pairs or seed variance.

### D3. MiniRec: Data-Efficient Reinforcement Learning for LLM-based Recommendation
- **Authors / venue:** Lin Wang, Yang Zhang, Jingfan Chen, Xiaoyan Zhao, Fengbin Zhu, Qing Li, Tat-Seng Chua. arXiv **2602.04278** (Feb 2026; venue not stated) [verified: https://arxiv.org/abs/2602.04278]
- **Code:** anonymous link in the PDF [unverified].
- **Framework:**
  - Selection for GRPO-style recommender RL using a **reward band**: drop samples that are too easy (high reward) or consistently zero-reward.
  - Gradient alignment with an approximate "ideal" trajectory.
  - Diversity control and an easy-to-hard curriculum.
- **Findings:**
  - Loss and gradients mis-measure learnability in GRPO; **27.2% of samples stay at near-zero reward** throughout training yet dominate the gradient norm.
  - **Random beats K-means coverage selection.**
  - Backbones: Gemma-2-2B-it, Qwen2.5-3B-Instruct. Datasets include Amazon CDs & Vinyl and Instruments. Baselines: Random, K-means, GraNd, EL2N, DEALRec. [verified: PDF text]
- **Gap vs. ours:** The closest analogue to N3 in a preference/RL setting. Its remedy is reward-band and trajectory alignment, not uncertainty.

### D4. LLM-based denoising of interactions
- **LLM4DSR (Wang et al., TOIS 2025).** Self-supervised fine-tuning to detect and replace noisy items in sequences, with an **uncertainty-estimation module so that only high-confidence corrections are applied**. Reports +12.9% NDCG@20 on average across 3 datasets and 3 backbones. [verified: https://arxiv.org/abs/2408.08208] [verified: https://dl.acm.org/doi/10.1145/3762182]. Repo: https://github.com/WANGBohaO-jpg/LLM4DSR (HTTP 200; README says TOIS2025) [verified]
- **LLaRD (Wang, Zheng, Sui, Xiong), WWW 2025.** LLM-generated preference and relation knowledge, chain-of-thought over the interaction graph, and an information-bottleneck alignment. Datasets: Amazon-Book, Yelp, Steam. [verified: https://arxiv.org/abs/2502.09058]
- **LLMHD (Song, Chao, Liu 2024).** An LLM scorer separates hard samples from noise, with variance-based pre-pruning. arXiv only [verified: https://arxiv.org/abs/2409.10343]
- **LLMHNI ("Hard vs. Noise"), AAAI 2026.** Documents **hard–noisy confusion**: noisy and hard samples share high-loss patterns, so loss-based dropping removes valuable hard samples. Uses LLM semantic and logical relevance to tell them apart. [verified: https://arxiv.org/abs/2511.07295]
- **DC4SR (2026).** Disagreement between an LLM semantic prior and a model-side learning-dynamics posterior estimates noise. arXiv only [verified: https://arxiv.org/abs/2604.24048]
- **IADSR, CIKM 2025.** LLM plus CF embeddings for sequence denoising; addresses over-denoising of cold items. [verified: https://arxiv.org/abs/2510.04239]
- **Gap vs. ours:** These clean *the interaction data feeding a recommender*, mostly conventional backbones. None prunes **DPO pairs for an LLM recommender by the LLM's own uncertainty**. The hard–noisy-confusion result is the best mechanistic explanation for N3.

### D5. Preference optimization for LLM recommenders (the training regime of N3)
- **S-DPO (Chen et al.), NeurIPS 2024.**
  - Softmax-DPO with multiple negatives; related to softmax loss and hard-negative mining.
  - Datasets: Goodreads, LastFM, MovieLens100K. Backbone: LLaMA2-7B. Metrics: HitRatio@1, ValidRatio.
  - [verified: https://arxiv.org/abs/2406.09215]. Repo: https://github.com/chenyuxin1999/S-DPO (HTTP 200) [verified]
- **DMPO (Bai, Wu, Cai, Zhu, Xiong), CIKM 2024.** "Aligning Large Language Model with Direct Multi-Preference Optimization for Recommendation": multiple negatives with a dynamic margin. [verified: https://dl.acm.org/doi/10.1145/3627673.3679611]. arXiv id and repo not found [unverified].
- **C-APO (Park, Yun, Kim, Hong, Hong, Myong, Oh, Cho, Park, Choi, Seok, Choo), ICLR 2026.** "More Than What Was Chosen: LLM-based Explainable Recommendation Beyond Noisy User Preferences."
  - Revealed preference is **noisy**. Adds "Coherent Preference" (logical consistency with history) and adaptively reconciles the two when they agree or conflict.
  - Data: Amazon Reviews. About 20 baselines. **1.65× CTR in deployment.** Backbone: gemma-3-4b-it (from the README).
  - [verified: https://iclr.cc/virtual/2026/poster/10009044]. Repo: https://github.com/cpark88/C-APO (HTTP 200) [verified]. arXiv id not found [unverified].
- **DynamicPO, DASFAA 2026 Best Paper.**
  - "Preference optimization collapse": more negatives can hurt performance even as the loss keeps falling, because easy negatives suppress gradients.
  - Fixes: boundary-negative selection and a per-sample β set by boundary ambiguity.
  - [verified: https://arxiv.org/abs/2605.00327]. Repo: https://github.com/xingyuHuxingyu/DynamicPO (HTTP 200) [verified]
- **NAPO (2025).** In-batch negative sharing and a **dynamic reward margin scaled by negative-sample confidence**; reduces popularity bias. arXiv only [verified: https://arxiv.org/abs/2508.09653]
- **Mult-DPO (Zhu, Steck, McInerney, …, Kallus, Li), 2026.** Multinomial set-wise DPO surrogate. arXiv only [verified: https://arxiv.org/abs/2606.10078]. Repo: https://github.com/yaochenzhu/Mult_DPO (HTTP 200) [verified]
- **BEAR (Yang et al.), SIGIR 2026.** Beam-search-aware regularization, which enforces top-B ranking of each positive token during SFT.
  - Datasets: Amazon Office, Book, Toy, Clothing. Base model: Llama-3.2-3B (also 1B and 8B).
  - [verified: https://arxiv.org/abs/2601.22925]. Repo: https://github.com/Tiny-Snow/BEAR-SIGIR-2026 (HTTP 200) [verified]
- **BLADE (Chen, Gao, Chen, Yang, He), SIGIR 2026.** Bayesian list-wise Best-of-N alignment covering NDCG, fairness and diversity.
  - Datasets: Amazon CDs, Steam, Goodreads. Backbone: Llama-3.2-1B-Instruct.
  - [verified: https://arxiv.org/abs/2605.04559]. Repo: https://github.com/RegionCh/BLADE (HTTP 200) [verified]
- **Gap vs. ours:** The trend is to use confidence or uncertainty as a **continuous weight, margin or β**, not as a hard filter: UGR, RosePO, NAPO, DynamicPO, ABPO. That is the published way around N3.

---

## E. Transferable calibration and UQ methods (core references)

| Method | Ref | Status |
|---|---|---|
| P(True) / P(IK) self-evaluation | Kadavath et al., "Language Models (Mostly) Know What They Know," 2022 (arXiv) | [verified: https://arxiv.org/abs/2207.05221] |
| Verbalized confidence beats token probabilities for RLHF LMs | Tian et al., "Just Ask for Calibration," EMNLP 2023 | [verified: https://aclanthology.org/2023.emnlp-main.330/] |
| Black-box elicitation (verbalized, sampling, consistency); LLMs are overconfident | Xiong et al., "Can LLMs Express Their Uncertainty?," ICLR 2024 | [verified: https://arxiv.org/abs/2306.13063]. Repo https://github.com/MiaoXiong2320/llm-uncertainty (HTTP 200) |
| Semantic entropy | Kuhn, Gal, Farquhar, ICLR 2023; Farquhar, Kossen, Kuhn, Gal, *Nature* 630:625–630 (2024) | [verified: https://arxiv.org/abs/2302.09664] [verified: https://pubmed.ncbi.nlm.nih.gov/38898292/] |
| Self-consistency | Wang et al., ICLR 2023 | [verified: https://arxiv.org/abs/2203.11171] |
| Calibration needs training (about 1k graded examples, LoRA) | Kapoor et al., "LLMs Must Be Taught to Know What They Don't Know," NeurIPS 2024 | [verified: https://arxiv.org/abs/2406.08391] |
| ConfTuner (tokenized Brier score, a proper scoring rule for verbalized confidence) | Li, Xiong, Wu, Hooi, NeurIPS 2025 | [verified: https://arxiv.org/abs/2508.18847]. Repo https://github.com/liushiliushi/ConfTuner (HTTP 200) |
| Temperature scaling / isotonic regression | Guo et al., ICML 2017 (classic) | [unverified in this session] |
| Conformal risk control | Angelopoulos, Bates, Fisch, Lei, Schuster, ICLR 2024 | [verified: https://arxiv.org/abs/2208.02814]. Repo https://github.com/aangelopoulos/conformal-risk (HTTP 200) |
| Recommendation sets with FDR control | Angelopoulos, Krauth, Bates, Wang, Jordan, COPA 2023 (PMLR v204; best paper) | [verified: https://proceedings.mlr.press/v204/angelopoulos23a.html] |
| Two-stage risk control for ranked retrieval | Xu, Ying, Guo, Wei 2024 (MSLR-Web, Yahoo LTRC) | [verified: https://arxiv.org/abs/2404.17769] |
| Conformal risk control for unwanted recommendations | De Toni et al., RecSys 2025 | [verified: https://arxiv.org/abs/2507.16829]. Repo https://github.com/geektoni/mitigating-harm-recsys (HTTP 200) |
| When abstention in ranked systems is monotone (structural vs contextual uncertainty) | Doku, "The Confidence Gate Theorem," 2026 (arXiv) | [verified: https://arxiv.org/abs/2603.09947] |

The Confidence Gate Theorem gives conditions ("rank-alignment, no inversion zones") under which confidence-based abstention monotonically improves decision quality. On a MovieLens *temporal* split, count-based confidence violates monotonicity as often as random abstention does. This gives a **theoretical lens for N1 and N2**: popularity or count-based confidence is "structural", and it fails under contextual drift.

---

## G. Published routes around our negative results

| Negative | Route found in the literature | Paper(s) | Strength |
|---|---|---|---|
| **N1**: monotone calibration preserves rank within a user | **Cross-user decisions**: reject or abstain on whole requests | UGR (KDD'26) user-level rejection; Zou et al. (KDD'26 ADS) deboost low-activity users; Confidence Gate Theorem | UGR on one dataset only; Zou is not LLM-based |
| N1 | **List truncation / personalized K** | PerK (WWW'24); UGR item truncation; Angelopoulos FDR sets (COPA'23); De Toni CRC (RecSys'25) | No LLM-specific conformal truncation found |
| N1 | **Abstain, then fall back or route** to a CF or sequential model | S-LLMR gate on history length, popularity and uncertainty, trained offline (arXiv 2512.21526) [verified]; serving-time gate between generative and collaborative profiles (arXiv 2609.39043, +6.5% Novelty@10 at a 5% NDCG budget) [verified]; LSC4Rec (KDD'25) device–cloud LLM-SRM with consistency-triggered requests [verified: https://arxiv.org/abs/2501.05647]; **negative**: RouteRec (SIGIR'26 AgentSearch workshop) found request-level selective LLM escalation did not help in a sparse fixed-candidate setting [verified: https://arxiv.org/abs/2607.09908] | Mixed. The RouteRec negative mirrors N1 and suggests item-level aggregation instead |
| N1 | **Exploration** | Zou et al. UCB for high-activity users; Google RecSys'24 LLM interest exploration (hybrid LLM plus classic) [verified: https://arxiv.org/abs/2405.16363]; ACL'25-industry user-feedback alignment for LLM exploration [verified: https://arxiv.org/abs/2504.05522]; **negative**: LLMP-UCB (RLC 2026), where embedding bandits match LLM-uncertainty bandits [verified: https://arxiv.org/abs/2604.05859] | LLM-uncertainty-driven exploration is still unproven |
| N1 | **Training-time calibration or uncertainty that changes the ranking** | UGR (confident-error penalty plus confidence tokens; reranking +0.003 NDCG@1); EviRank (position-aware calibration loss plus confidence-weighted rerank) | Gains are small. This direction is owned by UGR and EviRank |
| N1 | **Uncertainty-aware decoding or prompting** | UQRec uncertainty-aware prompting; USD (APWeb'25) entropy-controlled temperature; D3 ghost-token length-normalization fix | UQRec owns per-user prompt adaptation |
| **N2**: popularity-aware recalibration gave no gain | Evidence that popularity is the wrong conditioning variable | KnowSA_CKP: popularity is a poor proxy for LLM knowledge; Ravikumar: the catalog, not the item, sets the sign of miscalibration; Mozafari (SIGIR'26 short): exposure ≠ interaction popularity; Popular-but-Wrong (EMNLP'26): popularity *of the answer* reduces QA ECE | Supports reframing N2 as a finding ("interaction popularity is not the latent driver of LLM-recommender miscalibration; pretraining exposure is the better candidate") rather than hiding it |
| **N3**: uncertainty pruning ≈ random | Published evidence that loss or gradient importance underperforms random | GORACS (GraNd/EL2N < Random on TALLRec CTR); MiniRec (Random > K-means; low-reward samples dominate the gradient norm); LLMHNI (hard–noisy confusion) | Strong. N3 is consistent with the literature |
| N3 | **Soft weighting instead of hard pruning** | UGR difficulty-aware re-weighting; RosePO personalized smoothing; DynamicPO per-sample β; NAPO confidence margin; ABPO self-certainty tempering | Dominant published pattern |
| N3 | **Other selection signals that beat random** | DEALRec influence+effort; GORACS OT plus gradients; MiniRec reward band; LLM4DSR LLM judgment with an uncertainty gate | Our pruning study must compare against these, not only against Random |

---

## F. SIGIR format facts and typical SIGIR experimental design for LLM4Rec

### F1. SIGIR 2026 (Melbourne, July 20–24, 2026)
All of the following are [verified: https://sigir2026.org/en-AU/pages/submissions/full-papers-track and https://sigir2026.org/en-AU/pages/submissions/submission-policies-and-information, page text extracted].

- **Page limit:** "at most 9 pages (including figures, tables, proofs, appendixes, acknowledgments, and any content except references) … with unrestricted space for references."
- **Template:** ACM two-column `sigconf`, with `\documentclass[sigconf,natbib=true,anonymous=true]{acmart}`. ACM CCS concepts and keywords are required.
- **Camera-ready:** accepted papers get **one extra page** (10 pages plus references).
- **Anonymity:** double-anonymous review.
  - Cite your own work in the third person.
  - Share code through Anonymous GitHub (anonymous.4open.science).
  - "Authors may post drafts of their papers on the web, submit them to arXiv, or give talks about them." The 2026 CFP "relaxed the requirements regarding preprints and anonymity".
- **Other rules:**
  - Submission moved from EasyChair to **OpenReview**.
  - Each submission must nominate one author to serve on the PC.
  - Dual submissions are desk-rejected.
  - The ACM AI policy applies.
  - Reviewers may not use AI to write reviews.
- **2026 dates (AoE):** abstract Jan 15, 2026; full paper Jan 22, 2026; notification Apr 2, 2026; camera-ready Apr 29, 2026.
- **SIGIR 2025 for comparison:** same 9-page rule and the same `documentclass` line; abstract Jan 16, 2025; full paper Jan 23, 2025. [verified: https://sigir2025.dei.unipd.it/call-full-papers.html]

### F2. SIGIR 2027 (San Jose, CA, Signia by Hilton, July 18–24, 2027)
- Abstract **Jan 14, 2027**; full paper **Jan 21, 2027**; notification Apr 5, 2027; camera-ready and author registration Apr 30, 2027. [verified: https://sigir2027.org]
- The submission and policy pages are still **placeholders or TBA**: "Placeholder: add the 2027 pre-print, anonymity, and review policy". [verified: https://sigir2027.org/pages/submit-tracks.html, https://sigir2027.org/pages/policies.html]
- Assume the 2026 rules (9 pages plus references, `sigconf`, OpenReview, arXiv allowed) until the 2027 CFP is posted [unverified for 2027].

### F3. SIGIR 2026 accepted-paper scan (titles only, 468 full-paper entries)
- **No accepted full paper has uncertainty or calibration of LLM recommenders as its title topic.** [verified: title scan of https://sigir2026.org/en-AU/pages/program/accepted-papers]
- Related SIGIR'26 papers:
  - KnowSA_CKP, BEAR and BLADE (full papers);
  - "Pretraining Exposure Explains Popularity Judgments in LLMs" (short);
  - "CRED: Calibrated Relational Enhanced Distillation for LLM-Based Pointwise Reranking" (short; reranking for IR);
  - "CoCo: Conformal Confidence Suppression to Optimize Search Results" (short; search);
  - "Calibrating Uncertainty with Cross-Model Consistency for LLM Hallucination Mitigation" (short);
  - "Uncertainty Quantification for Retrieval-Augmented Reasoning" (full; IR, not recommendation);
  - "FairSpec: Expert Specialization for Fair LLM-based Recommendation" (full).

  [verified: same page]

### F4. Typical SIGIR / KDD / WWW LLM4Rec experimental design (from the papers above)

**Datasets**
- Amazon categories, most common: Beauty, Toys, Sports, Games, CDs & Vinyl, Books, Office, Industrial, Instruments, Grocery, Gift Cards.
- MovieLens-100K or 1M.
- Steam, Goodreads, LastFM, Yelp, MicroLens-50K.
- Typically **3 or 4 datasets** per paper. [verified across DEALRec, GORACS, SPRec, Flower, BEAR, BLADE, KnowSA, UGR, UQRec, RosePO, S-DPO, LLaRA PDFs]

**Backbones**
- Llama-3.2-1B/3B, Llama-3-8B, Qwen2.5-1.5B/3B/7B, Qwen3-8B, Gemma-2/3, Mistral-7B.
- Older papers: LLaMA-7B, LLaMA2-7B, Vicuna-7B.
- Often a size or architecture ablation (BEAR: 1B/3B/8B; KnowSA: 7B–32B; UQRec: three 7–8B models).

**Protocols and metrics**
- *Generative next-item:* HR@K and NDCG@K for K ∈ {1, 5, 10, 20}, beam size 10–20, leave-one-out or 8:1:1 chronological split.
- *Candidate selection:* HitRatio@1 and ValidRatio on about 20 candidates (LLaRA, S-DPO).
- *Pointwise yes/no:* AUC and UAUC (TALLRec, CoLLM, BinLLM).
- *Bias:* MGU/DGU@K, DivRatio, ORRatio, entropy, TTR.

**Baselines**
- Usually 6–10, rising to about 20 (C-APO).
- Typical mix: SASRec, GRU4Rec, Caser, BERT4Rec, LightGCN, TALLRec, BIGRec, LLaRA, A-LLMRec, TIGER, S-DPO, DMPO, SPRec, MiniOneRec, D3.
- For pruning: Random, GraNd, EL2N, CCS, D2, MODERATE, TF-DCon, DEALRec.

**Seeds and significance tests (sparse in this literature)**
- DEALRec: **3 seeds + one-sample t-tests, p < 0.01**.
- RosePO: **5 seeds**.
- LLaRA: **5 runs**.
- UGR, UQRec, SPRec, Flower, BEAR, BLADE, GORACS, MiniRec, KnowSA: no explicit seed count or significance test was found by text search [verified: PDF text grep; an absence-of-evidence caveat applies].

**Recommendation for us:** With σ_seed ≈ 1.5 pt, a claim of a 1–2 pt effect needs:
- ≥ 5 seeds (preferably about 10 for the pruning study);
- mean ± std;
- paired tests (paired t-test or Wilcoxon over seeds or users);
- bootstrap CIs for ECE and AURC;
- a pre-declared random-pruning control at matched budget.

This would itself distinguish the paper, since most competitors run a single seed.

---

## H. Do-not-overlap checklist and positioning suggestions

**Do not claim as novel:**
1. Plackett–Luce single-pass predictive uncertainty and recommendation/prompt decomposition; uncertainty-aware prompting (UQRec).
2. Confident-error penalties in RL, explicit confidence tokens, user rejection and item truncation for generative recommendation (UGR).
3. Position-level confidence with an NDCG-discount calibration target (EviRank).
4. Popularity-stratified ECE and Brier of verbalized confidence for zero-shot LLM recommenders, plus conformal abstention on hallucination (Ravikumar CIKM'26).
5. Popularity, Min-K or EigV as proxies for LLM item knowledge, and selective augmentation (KnowSA_CKP SIGIR'26).
6. "DPO/SFT amplifies popularity" (SPRec, Flower); ghost-token amplification bias (D3); popular items as rejected samples (RosePO).
7. Influence or OT coreset selection for LLM-recommender fine-tuning (DEALRec, GORACS); reward-band selection for recommender RL (MiniRec).
8. Calibrated probability → personalized K (PerK); cross-user deboost and UCB using calibrated uncertainty (Zou et al. KDD'26 ADS).

**Defensible novelty directions:**
- **(a) First calibration audit of fine-tuned pointwise yes/no LLM recommenders.** Measure P(Yes) and verbalized confidence against held-out labels (ECE, Brier, reliability, AUROC of correct versus wrong), across TALLRec, CoLLM and BinLLM-style models, with popularity-quartile and pretraining-exposure stratification. State N1 formally: a *global* monotone map changes neither AUC nor UAUC. A *user- or item-conditional* monotone map leaves UAUC (within-user order) unchanged but can change global AUC and any cross-user threshold. That is exactly where calibration can matter.
- **(b) Uncertainty for cross-user decisions.** Risk–coverage curves for selective LLM recommendation, with **fallback to SASRec or LightGCN** and conformal (CRC) guarantees on the abstention rate or risk. Compare UQ estimators: P(Yes), entropy, verbalized, semantic or self-consistency, a trained probe as in Kapoor et al., and EviRank-style evidence.
- **(c) Confidence × popularity × loop dynamics.** Use a RecLoop or EchoTrace-style simulator with confidence-gated exposure. Test whether gating slows homogenization (Gini, coverage, MGU) at a fixed accuracy budget.
- **(d) A rigorous negative and diagnostic result on uncertainty pruning for recommender DPO.** Use multiple seeds and the hard–noisy confusion lens, with controlled injected noise to locate *when* uncertainty pruning helps. Compare against soft re-weighting (RosePO, DynamicPO, UGR).

**Expected reviewer pool:** the POSTECH/UIUC group (Kweon, Kang, Jang, Yu: UQRec, KnowSA, PerK, SPRINT SIGIR'26) and the USTC/NUS He–Feng group (UGR, SPRec, Flower, BLADE, D3, TALLRec, CoLLM, BinLLM, RosePO, DEALRec).

---

## I. Unverified items and caveats
- EviRank's venue (MM'26) is taken from a PDF template header, and the code link was not opened. **[unverified]**
- The DMPO arXiv id and repo, and the C-APO arXiv id, were not found. Both venues are verified through ACM DL and ICLR. **[unverified ids]**
- The exact TALLRec scoring formula (raw P("Yes") versus softmax over Yes/No) was not quoted in the extracted text. CoLLM states "likelihood of answering 'Yes'". **[unverified for TALLRec]**
- The "Popular but Wrong" ECE numbers (0.356 → 0.050) come from a search snippet. **[unverified against the PDF]**
- The EchoTrace repo was not opened. **[unverified]**
- "Not found" statements about seeds or significance tests come from PDF keyword search and may miss tables or appendices.
- SIGIR 2027 page limits and policies are not yet published. The 2026 rules are assumed.
