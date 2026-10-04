# CITATION_MAP — SIGIR 2027 (uncertainty / confidence in LLM-based recommenders)

Companion to `references.bib` (same folder). Built 2026-10-01.

## How the entries were verified

- **No entry was written from memory.** Every entry was fetched from a source over HTTP and then normalized.
- **DBLP could not be used.** Both `dblp.org/search/publ/api` and the WebFetch tool hit an Anubis anti-bot challenge, and the challenge was not bypassed. OpenReview's API also returned `ChallengeRequiredError`.
- **Fallback order actually used:**
  1. **CrossRef** BibTeX (`api.crossref.org/works/<doi>/transform/application/x-bibtex`) for every paper with a DOI. This includes ACM, IEEE, ACL Anthology, AAAI, Springer and Nature papers, and NeurIPS 2023–2025 via Curran `10.52202` DOIs.
  2. **The venue's own BibTeX export** where no DOI exists:
     - PMLR (ICML), the `bibtex` block on the paper page;
     - NeurIPS 2017 proceedings `-Bibtex.bib`;
     - ICLR 2024/2026 proceedings `-Bibtex-Conference.bib`;
     - TMLR `jmlr.org/tmlr/papers/bib/<id>.bib`.
  3. **arXiv** BibTeX (`arxiv.org/bibtex/<id>`) for arXiv-only papers.
- **Accepted-but-unpublished venues.** For arXiv papers whose acceptance is stated only in the arXiv comments, a `note` field records the venue claim; the entry stays `@misc`.
- **ICLR 2022/2023 without DOI or proceedings BibTeX** (`hu2022lora`, `burns2023ccs`):
  - the authors and title come from the arXiv BibTeX;
  - `booktitle`, `year` and `url` come from the paper's own listing on iclr.cc (`/virtual/2022/poster/6319` and `/virtual/2023/poster/11480`), checked in this session.
- **Title check.** Each fetched title was compared with the title in the source lists (`docs/sigir/LIT_*.md`, `idea-stage/NOVELTY_DOSSIER.md`, `idea-stage/triage_verdict.md`). All 70 matched; differences are listed in §Discrepancies.
- **Only formatting was normalized:**
  - citation keys;
  - typographic quotes and dashes changed to LaTeX;
  - `&` escaped;
  - month macros;
  - DOI URLs changed to `https://doi.org/...`;
  - acronyms in titles brace-protected;
  - clutter fields dropped (abstract, pdf, editor, issn, isbn, collection, and `volume` where it equals the year).
- **Author lists truncated.** For the Llama 3 (561 authors) and Qwen3 (60 authors) reports, the first 5 authors are kept, followed by `and others`.
- **Compile check.** `references.bib` passes BibTeX with `ACM-Reference-Format.bst` (MiKTeX acmart): 70/70 entries emitted, 0 errors. The only warnings are the usual ACM "empty address/publisher" and "missing pages" ones.

**Counts: 70 verified entries, 0 `[VERIFY]` entries.**

## Key → what it supports → source URL used

### Closest competitors

| Key | Supports (one line) | Venue (year) | Source URL used |
|---|---|---|---|
| `kweon2025uqrec` | UQRec: Plackett-Luce predictive uncertainty for listwise LLM rec, decomposed into recommendation vs prompt uncertainty; uncertainty-aware prompting. | Proceedings of the ACM on Web Conference 2025 (2025) | https://api.crossref.org/works/10.1145/3696410.3714601/transform/application/x-bibtex |
| `fan2026ugr` | UGR: uncertainty-weighted reward penalizing confident errors, explicit confidence tokens, user rejection / item truncation in generative rec. | Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (2026) | https://api.crossref.org/works/10.1145/3770855.3817975/transform/application/x-bibtex |
| `yan2026evirank` | EviRank: position-level evidence-based confidence and position-aware calibration for LLM ranking. | arXiv preprint (2026) | https://arxiv.org/bibtex/2606.04727 |
| `ravikumar2026hallucinating` | Popularity-stratified ECE/Brier audit of verbalized confidence of zero-shot LLM recommenders vs catalog membership; conformal abstention barely helps. | arXiv preprint; Accepted at CIKM 2026 (short paper), per arXiv comments; cite v2 or later (2026) | https://arxiv.org/bibtex/2608.10008 |
| `lee2026knowsa` | KnowSA_CKP: popularity / Min-K% / EigV-uncertainty are poor proxies of LLM item knowledge; selective augmentation. | Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (2026) | https://api.crossref.org/works/10.1145/3805712.3809656/transform/application/x-bibtex |
| `zhou2026routerec` | RouteRec: negative result -- request-level selective LLM escalation did not help in a sparse fixed-candidate setting. | arXiv preprint; AgentSearch 2026 Workshop, co-located with SIGIR 2026, per arXiv comments (2026) | https://arxiv.org/bibtex/2607.09908 |

### Hooi group (calibration / confidence)

| Key | Supports (one line) | Venue (year) | Source URL used |
|---|---|---|---|
| `xiong2024llmuncertainty` | Black-box confidence elicitation (verbalized / sampling / aggregation); LLMs are overconfident; calibration vs failure-prediction split. | International Conference on Learning Representations (2024) | https://proceedings.iclr.cc/paper_files/paper/2024/file/6733cf15e10e2cd1d59af033c3bb8507-Bibtex-Conference.bib |
| `xiong2023procal` | ProCal: proximity bias (low-density samples more overconfident), PIECE metric, Bin-Mean-Shift; template for popularity-informed calibration. | Advances in Neural Information Processing Systems 36 (2023) | https://api.crossref.org/works/10.52202/075280-2996/transform/application/x-bibtex |
| `li2025conftuner` | ConfTuner: tokenized Brier score is a proper scoring rule for verbalized confidence (reduces to Brier on P(Yes) for binary). | Advances in Neural Information Processing Systems 38 (2025) | https://api.crossref.org/works/10.52202/085713-1783/transform/application/x-bibtex |
| `xiong2022neighboragg` | NeighborAgg: neighborhood-aware trustworthiness score and conformal mislabel detection. | Transactions on Machine Learning Research (2022) | https://jmlr.org/tmlr/papers/bib/p5V8P2J61u.bib |
| `wu2025unit` | UNIT: uncertainty-aware instruction fine-tuning; cut or reflect claims the model is unfamiliar with (quantile-threshold pruning). | arXiv preprint (2025) | https://arxiv.org/bibtex/2502.11962 |
| `li2025mkj` | MKJ: binary true/false judgments; Acc_pos vs Acc_neg skepticism gap, overconfidence, worse on low-frequency entities. | arXiv preprint (2025) | https://arxiv.org/bibtex/2502.14275 |

### LLM recommenders (yes/no + generative)

| Key | Supports (one line) | Venue (year) | Source URL used |
|---|---|---|---|
| `bao2023tallrec` | TALLRec: canonical yes/no LoRA-tuned LLM recommender evaluated with AUC. | Proceedings of the 17th ACM Conference on Recommender Systems (2023) | https://api.crossref.org/works/10.1145/3604915.3608857/transform/application/x-bibtex |
| `zhang2025collm` | CoLLM: yes/no LLM rec with CF embeddings; P(Yes) trained by BCE, evaluated by AUC/UAUC. | IEEE Transactions on Knowledge and Data Engineering (2025) | https://api.crossref.org/works/10.1109/TKDE.2025.3540912/transform/application/x-bibtex |
| `zhang2024binllm` | BinLLM: binary-string CF encoding for yes/no LLM rec (AUC/UAUC). | Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (2024) | https://api.crossref.org/works/10.18653/v1/2024.acl-long.497/transform/application/x-bibtex |
| `liao2024llara` | LLaRA: hybrid ID+text item tokens; candidate-list next-item selection (HitRatio@1). | Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (2024) | https://api.crossref.org/works/10.1145/3626772.3657690/transform/application/x-bibtex |
| `kim2024allmrec` | A-LLMRec: aligns a frozen CF model with a frozen LLM for cold/warm recommendation. | Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (2024) | https://api.crossref.org/works/10.1145/3637528.3671931/transform/application/x-bibtex |
| `lin2024rella` | ReLLa: retrieval-enhanced yes/no (CTR-style) LLM recommendation over lifelong behavior. | Proceedings of the ACM Web Conference 2024 (2024) | https://api.crossref.org/works/10.1145/3589334.3645467/transform/application/x-bibtex |
| `geng2022p5` | P5: recommendation cast as text-to-text generation. | Proceedings of the 16th ACM Conference on Recommender Systems (2022) | https://api.crossref.org/works/10.1145/3523227.3546767/transform/application/x-bibtex |
| `bao2025bigrec` | BIGRec: generative LLM rec with grounding of generated text to actual items. | ACM Transactions on Recommender Systems (2025) | https://api.crossref.org/works/10.1145/3716393/transform/application/x-bibtex |
| `bao2024d3` | D3: ghost-token / length-normalization amplification bias in LLM-rec decoding; sequence-level confidence is distorted. | Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (2024) | https://api.crossref.org/works/10.18653/v1/2024.emnlp-main.589/transform/application/x-bibtex |
| `chen2024sdpo` | S-DPO: softmax DPO with multiple negatives for LLM recommenders. | Advances in Neural Information Processing Systems 37 (2024) | https://api.crossref.org/works/10.52202/079017-0863/transform/application/x-bibtex |
| `gao2025sprec` | SPRec: DPO amplifies popularity bias in LLM rec; self-play fix. | Proceedings of the ACM on Web Conference 2025 (2025) | https://api.crossref.org/works/10.1145/3696410.3714524/transform/application/x-bibtex |
| `gao2025flower` | Flower: SFT amplifies popularity bias; GFlowNet process supervision. | Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (2025) | https://api.crossref.org/works/10.1145/3726302.3729981/transform/application/x-bibtex |
| `liao2026rosepo` | RosePO: popularity-aware rejected sampling and personalized smoothing for uncertain preference labels. | ACM Transactions on Information Systems (2026) | https://api.crossref.org/works/10.1145/3833420/transform/application/x-bibtex |
| `park2026capo` | C-APO: revealed preferences are noisy; coherent-preference alignment for LLM rec. | International Conference on Learning Representations (2026) | https://proceedings.iclr.cc/paper_files/paper/2026/file/2a581c3f0fcaf50b6f87eac048f18369-Bibtex-Conference.bib |

### Prompt symmetrization / yes-no bias

| Key | Supports (one line) | Venue (year) | Source URL used |
|---|---|---|---|
| `burns2023ccs` | CCS: statement/negation consistency; symmetrized inference rule 1/2 (p(x+) + 1 - p(x-)). | International Conference on Learning Representations (ICLR) (2023) | https://arxiv.org/bibtex/2212.03827 + venue: https://iclr.cc/virtual/2023/poster/11480 |
| `huang2026yesno` | Crossed symmetrization: LLM yes-no bias is order/wording-driven; per-item residuals idiosyncratic. | arXiv preprint (2026) | https://arxiv.org/bibtex/2607.05552 |
| `zhao2021calibrate` | Contextual calibration: estimate and remove the answer prior via content-free inputs. | Proceedings of the 38th International Conference on Machine Learning (2021) | https://proceedings.mlr.press/v139/zhao21c.html |
| `holtzman2021surface` | Surface form competition; domain-conditional PMI (PMI-DC) scoring. | Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (2021) | https://api.crossref.org/works/10.18653/v1/2021.emnlp-main.564/transform/application/x-bibtex |
| `zhou2024batch` | Batch Calibration: remove the contextual label prior estimated from the batch. | International Conference on Learning Representations (2024) | https://proceedings.iclr.cc/paper_files/paper/2024/file/003e438cf04e3caf0a5c908495e484fe-Bibtex-Conference.bib |
| `braun2025acquiescence` | LLMs show a global lean toward 'no' rather than human-like acquiescence. | Findings of the Association for Computational Linguistics: EMNLP 2025 (2025) | https://api.crossref.org/works/10.18653/v1/2025.findings-emnlp.607/transform/application/x-bibtex |
| `zhang2025yesharder` | Positive vs negative framing asymmetry of LLM answers across downstream tasks. | Proceedings of the 34th ACM International Conference on Information and Knowledge Management (2025) | https://api.crossref.org/works/10.1145/3746252.3761350/transform/application/x-bibtex |
| `liu2025repair` | REPAIR: negation/transitivity-consistent preference training (template for mirror tuning). | Proceedings of the 42nd International Conference on Machine Learning (2025) | https://proceedings.mlr.press/v267/liu25u.html |
| `ahn2025prin` | PRIN: 'correct?' vs 'incorrect?' prompts yield inconsistent LLM judgments. | arXiv preprint; Accepted at COLM 2025, per arXiv comments (2025) | https://arxiv.org/bibtex/2504.01282 |

### Calibration & UQ fundamentals

| Key | Supports (one line) | Venue (year) | Source URL used |
|---|---|---|---|
| `guo2017calibration` | Modern NNs are miscalibrated; temperature scaling; reliability diagrams / ECE. | Proceedings of the 34th International Conference on Machine Learning (2017) | https://proceedings.mlr.press/v70/guo17a.html |
| `naeini2015ece` | BBQ calibration; origin of the binned ECE / MCE metrics. | Proceedings of the AAAI Conference on Artificial Intelligence (2015) | https://api.crossref.org/works/10.1609/aaai.v29i1.9602/transform/application/x-bibtex |
| `kadavath2022language` | P(True) / P(IK) self-evaluation; LM token probabilities are fairly calibrated on multiple choice. | arXiv preprint (2022) | https://arxiv.org/bibtex/2207.05221 |
| `farquhar2024semantic` | Semantic entropy for LLM hallucination / uncertainty detection. | Nature (2024) | https://api.crossref.org/works/10.1038/s41586-024-07421-0/transform/application/x-bibtex |
| `geifman2017selective` | Selective prediction / risk-coverage with a reject option. | Advances in Neural Information Processing Systems (2017) | https://papers.nips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Bibtex.bib |

### RecSys calibration / MNAR / popularity

| Key | Supports (one line) | Venue (year) | Source URL used |
|---|---|---|---|
| `kweon2022calibrated` | Calibrating ranking-model scores under MNAR implicit feedback; evaluated on Yahoo!R3 / Coat unbiased test sets. | Proceedings of the AAAI Conference on Artificial Intelligence (2022) | https://api.crossref.org/works/10.1609/aaai.v36i4.20326/transform/application/x-bibtex |
| `steck2018calibrated` | Calibrated recommendations (genre-proportion calibration of lists); a different notion from probability calibration. | Proceedings of the 12th ACM Conference on Recommender Systems (2018) | https://api.crossref.org/works/10.1145/3240323.3240372/transform/application/x-bibtex |
| `abdollahpouri2017controlling` | Popularity bias in learning-to-rank recommendation and its control. | Proceedings of the Eleventh ACM Conference on Recommender Systems (2017) | https://api.crossref.org/works/10.1145/3109859.3109912/transform/application/x-bibtex |
| `wei2021macr` | MACR: counterfactual subtraction of the item-popularity effect. | Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining (2021) | https://api.crossref.org/works/10.1145/3447548.3467289/transform/application/x-bibtex |
| `zhang2021pda` | PDA: causal intervention to deconfound popularity. | Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (2021) | https://api.crossref.org/works/10.1145/3404835.3462875/transform/application/x-bibtex |
| `gao2022kuairec` | KuaiRec: fully observed user-item matrix for unbiased evaluation. | Proceedings of the 31st ACM International Conference on Information \& Knowledge Management (2022) | https://api.crossref.org/works/10.1145/3511808.3557220/transform/application/x-bibtex |
| `schnabel2016treatments` | MNAR debiasing via propensities; introduces the Coat MAR test set. | Proceedings of The 33rd International Conference on Machine Learning (2016) | https://proceedings.mlr.press/v48/schnabel16.html |
| `marlin2009collaborative` | Non-random missing data in CF; source of the Yahoo! R3 randomly exposed test set. | Proceedings of the third ACM conference on Recommender systems (2009) | https://api.crossref.org/works/10.1145/1639714.1639717/transform/application/x-bibtex |
| `chu2026incumbent` | Real vs fictional brands with identical specs: LLMs prefer recognized brands (causal brand effect on choice). | arXiv preprint (2026) | https://arxiv.org/bibtex/2606.17443 |
| `kamruzzaman2024brand` | Brand bias (global vs local) in LLM product recommendations. | Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (2024) | https://api.crossref.org/works/10.18653/v1/2024.emnlp-main.707/transform/application/x-bibtex |

### Data selection / denoising

| Key | Supports (one line) | Venue (year) | Source URL used |
|---|---|---|---|
| `lin2024dealrec` | DEALRec: influence + effort scores for data pruning in LLM-rec fine-tuning. | Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (2024) | https://api.crossref.org/works/10.1145/3626772.3657807/transform/application/x-bibtex |
| `mei2025goracs` | GORACS: OT-guided coreset selection; GraNd/EL2N fall below random on TALLRec CTR tuning. | Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (2025) | https://api.crossref.org/works/10.1145/3711896.3736985/transform/application/x-bibtex |
| `wang2021tce` | T-CE / R-CE: loss-based adaptive denoising of implicit feedback during training. | Proceedings of the 14th ACM International Conference on Web Search and Data Mining (2021) | https://api.crossref.org/works/10.1145/3437963.3441800/transform/application/x-bibtex |

### Feedback loops

| Key | Supports (one line) | Venue (year) | Source URL used |
|---|---|---|---|
| `chaney2018feedback` | Simulated feedback loops: algorithmic confounding increases homogeneity. | Proceedings of the 12th ACM Conference on Recommender Systems (2018) | https://api.crossref.org/works/10.1145/3240323.3240370/transform/application/x-bibtex |
| `mansoury2020feedback` | Feedback loops amplify popularity bias over iterations. | Proceedings of the 29th ACM International Conference on Information \& Knowledge Management (2020) | https://api.crossref.org/works/10.1145/3340531.3412152/transform/application/x-bibtex |
| `zhang2024agent4rec` | Agent4Rec: LLM-agent user simulator for recommendation, including filter-bubble simulation. | Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (2024) | https://api.crossref.org/works/10.1145/3626772.3657844/transform/application/x-bibtex |

### Baselines (Lumen main table)

| Key | Supports (one line) | Venue (year) | Source URL used |
|---|---|---|---|
| `wang2024elmrec` | ELMRec baseline. | Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (2024) | https://api.crossref.org/works/10.18653/v1/2024.emnlp-main.653/transform/application/x-bibtex |
| `wang2025irllrec` | IRLLRec baseline. | Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (2025) | https://api.crossref.org/works/10.1145/3726302.3730011/transform/application/x-bibtex |
| `he2025llm2rec` | LLM2Rec baseline. | Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (2025) | https://api.crossref.org/works/10.1145/3711896.3737029/transform/application/x-bibtex |
| `liu2025llmemb` | LLMEmb baseline. | Proceedings of the AAAI Conference on Artificial Intelligence (2025) | https://api.crossref.org/works/10.1609/aaai.v39i11.33327/transform/application/x-bibtex |
| `liu2024llmesr` | LLM-ESR baseline. | Advances in Neural Information Processing Systems 37 (2024) | https://api.crossref.org/works/10.52202/079017-0839/transform/application/x-bibtex |
| `zhang2026proex` | ProEx baseline. | Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (2026) | https://api.crossref.org/works/10.1145/3770854.3780284/transform/application/x-bibtex |
| `zhang2026promax` | ProMax baseline. | Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (2026) | https://api.crossref.org/works/10.1145/3805712.3809600/transform/application/x-bibtex |
| `ren2024rlmrec` | RLMRec baseline. | Proceedings of the ACM Web Conference 2024 (2024) | https://api.crossref.org/works/10.1145/3589334.3645458/transform/application/x-bibtex |
| `kang2018sasrec` | SASRec sequential baseline / CF fallback model. | 2018 IEEE International Conference on Data Mining (ICDM) (2018) | https://api.crossref.org/works/10.1109/ICDM.2018.00035/transform/application/x-bibtex |
| `hou2024llmrank` | LLM rankers are position- and popularity-biased (zero-shot listwise); extra context entry. | Advances in Information Retrieval (2024) | https://api.crossref.org/works/10.1007/978-3-031-56060-6_24/transform/application/x-bibtex |

### Backbones & infrastructure

| Key | Supports (one line) | Venue (year) | Source URL used |
|---|---|---|---|
| `yang2025qwen3` | Qwen3 backbone (Qwen3-8B). | arXiv preprint (2025) | https://arxiv.org/bibtex/2505.09388 |
| `grattafiori2024llama3` | Llama 3 backbone (Llama-3.1-8B). | arXiv preprint (2024) | https://arxiv.org/bibtex/2407.21783 |
| `kwon2023vllm` | vLLM (PagedAttention) inference engine used for prefill P(Yes) scoring. | Proceedings of the 29th Symposium on Operating Systems Principles (2023) | https://api.crossref.org/works/10.1145/3600006.3613165/transform/application/x-bibtex |
| `hu2022lora` | LoRA parameter-efficient fine-tuning. | International Conference on Learning Representations (ICLR) (2022) | https://arxiv.org/bibtex/2106.09685 + venue: https://iclr.cc/virtual/2022/poster/6319 |

## Discrepancies vs. the source lists (venue / year / metadata)

| Key | Source lists said | Fetched record says | Action |
|---|---|---|---|
| `yan2026evirank` | Authors include "Cai Xu". Venue "MM'26 per PDF header (unconfirmed)". | arXiv lists the author as **"Cai Xv"**. No venue in arXiv metadata, and no DOI in CrossRef. | Cited as an arXiv preprint, with the arXiv spelling. Do not claim MM'26. |
| `ravikumar2026hallucinating` | CIKM 2026 short. | Not yet in CrossRef (CIKM'26 proceedings not out). arXiv comment: "Accepted at CIKM 2026 (short paper)… cite v2 or later". | arXiv `@misc` plus a `note`. Swap in the ACM DOI once it is minted. |
| `zhou2026routerec` | "RouteRec (SIGIR'26 AgentSearch workshop)". | arXiv comment confirms the **AgentSearch 2026 workshop** co-located with SIGIR 2026. This is not a SIGIR main-track paper. | arXiv `@misc` plus a `note`. Cite as a workshop paper. |
| `bao2024d3` | Title "Decoding Matters: … for LLM-based Recommendation" (arXiv title). | The EMNLP 2024 published title is "…Homogeneity Issue **in Recommendations for Large Language Models**". | The published (CrossRef) title is used. |
| `liao2026rosepo` | TOIS 44(7). arXiv title "RosePO: Aligning LLM-based Recommenders with Human Values". Eight authors listed. | TOIS **44(7), Sept 2026**, pp. 1–25. Title "**RosePO: Customized Preference Alignment in LLM-Based Recommendation**". **10 authors** (adds Jingru Duan and Wenyu Zang). | The CrossRef record is used, with year 2026. |
| `bao2025bigrec` | BIGRec (venue not stated; usually cited as arXiv 2023). | **ACM TORS 3(4), 2025** (DOI 10.1145/3716393). | The TORS journal version is used. |
| `zhang2026promax` | CLAUDE.md lists the ProMax "DOI/Crossref visibility" as a submission blocker. The old `Paper/references.bib` had no pages. | CrossRef now resolves 10.1145/3805712.3809600: SIGIR'26, **pp. 2431–2441**, July 2026. | The CrossRef record is used. **The visibility blocker appears resolved.** |
| `zhang2026proex` | KDD 2026. | CrossRef: KDD'26 **V.1**, pp. 1940–1951, issued **Apr 2026** (cycle-1 proceedings). | As fetched. |
| `fan2026ugr` | KDD 2026. | CrossRef: KDD'26 **V.2**, pp. 1015–1026, Aug 2026. | As fetched. |
| `lee2026knowsa` | SIGIR 2026 full paper (arXiv 2604.07825). | CrossRef DOI 10.1145/3805712.3809656, pp. 891–901. | The venue version is used, not arXiv. |
| `li2025conftuner`, `xiong2023procal`, `chen2024sdpo`, `liu2024llmesr` | NeurIPS (no DOI expected). | NeurIPS 2023/2024/2025 papers now carry CrossRef DOIs (`10.52202/…`) with page ranges. | The CrossRef records are used. |
| `wu2025unit` (UNIT), `li2025mkj` (MKJ) | Venue unknown. | Still arXiv-only. No journal-ref or comment on arXiv. A web search found no ACL Anthology or OpenReview venue. | Cited as arXiv preprints. |
| `ahn2025prin` | COLM'25. | COLM has no DOI or BibTeX endpoint reachable here (OpenReview is challenge-blocked). The arXiv comment says "accepted in COLM2025". | arXiv `@misc` plus a `note`. |
| `abdollahpouri2017controlling` | "Abdollahpouri popularity bias" (paper not specified). | Chosen: RecSys'17 "Controlling Popularity Bias in Learning-to-Rank Recommendation". | Swap if a different Abdollahpouri paper is intended, for example the UMAP'20 or FLAIRS'19 work. |
| `marlin2009collaborative` / `schnabel2016treatments` | "Coat/Yahoo!R3 (Schnabel ICML'16 / Marlin)". | Yahoo! R3 is from Marlin & Zemel, **RecSys 2009**. Coat is from Schnabel et al., ICML 2016 (PMLR 48). | As fetched. |
| `hou2024llmrank` | ECIR 2024. | CrossRef types it `@inbook` (an LNCS chapter in "Advances in Information Retrieval"). | Emitted as `@inproceedings` with `series = Lecture Notes in Computer Science` (taken from the CrossRef container-title). |
| `huang2026yesno` | "crossed symmetrization arXiv 2607.05552". | Single author (Haonan Huang). The full title ends "…not shifts in moral judgment". | As fetched. |

No other venue or year disagreed with the source lists.

## [VERIFY] entries

None. Every requested paper resolved to a fetched CrossRef, venue or arXiv record. There are three caveats to re-check before camera-ready:
1. `ravikumar2026hallucinating`: replace with the CIKM'26 DOI once it is published.
2. `ahn2025prin`: the COLM'25 acceptance comes only from the arXiv comment.
3. `yan2026evirank`: no venue; keep it as a preprint.

## Added 2026-10-04 (text integration)

Verified on the web on 2026-10-04 by two verification passes (CrossRef JSON/BibTeX, publisher pages, JSTOR, PubMed, arXiv;
DBLP and OpenReview were not used: anti-bot challenge). Only formatting was normalised (same rules as above). The first block is
cited in the draft; the second block was verified for the MISSING_REF notes of the section writers but is **not cited** in the
current draft (no room within the 9-page budget) and is kept so that it can be cited later without re-verification.

**Counts after this addition: 92 entries in `references.bib` (70 + `kweon2024perk` + 21 added); 0 `[VERIFY]` entries.**

### Cited in the draft

| Key | Supports (one line) | Venue (year) | Source URL used |
|---|---|---|---|
| `kweon2024perk` | PerK (Top-Personalized-K): user-specific list size chosen by expected user utility from calibrated interaction probabilities (cited for list truncation by confidence). Key existed; row was missing. Pages 3388--3399 and DOI confirmed. | Proceedings of the ACM Web Conference 2024 (2024) | https://api.crossref.org/works/10.1145/3589334.3645417 |
| `krichene2020sampled` | Sampled metrics are inconsistent with exact metrics and need not preserve the order of two systems (scope of same-candidate evaluation). | Proceedings of the 26th ACM SIGKDD Conference (KDD '20) (2020) | https://api.crossref.org/works/10.1145/3394486.3403226/transform/application/x-bibtex |
| `holm1979simple` | Holm's sequentially rejective multiple test procedure (family-wise error control). No DOI resolves (10.2307/4615733 returns 404). | Scandinavian Journal of Statistics 6(2) (1979) | https://www.jstor.org/citation/ris/4615733 |
| `platt2000probabilities` | Platt scaling: a fitted sigmoid maps classifier scores to probabilities. Chapter title as registered by the publisher. | Advances in Large-Margin Classifiers, MIT Press (2000) | https://api.crossref.org/works/10.7551/mitpress/1113.003.0008 |
| `zadrozny2002transforming` | Calibration of ranking scores into probabilities, including isotonic regression (the isotonic attribution is from Niculescu-Mizil and Caruana, ICML 2005; full text not reached). | Proceedings of the 8th ACM SIGKDD Conference (KDD '02) (2002) | https://api.crossref.org/works/10.1145/775047.775151 |
| `harper2015movielens` | MovieLens datasets: history, use and limitations (ML-1M panel). | ACM Transactions on Interactive Intelligent Systems 5(4) (2015; issue dated Jan 2016) | https://api.crossref.org/works/10.1145/2827872 |
| `hou2024bridging` | Introduces the Amazon Reviews 2023 dataset (and BLaIR); arXiv v1. | arXiv preprint (2024) | https://arxiv.org/abs/2403.03952v1 |
| `kang2023llmrating` | Zero-shot and few-shot LLMs trail interaction-trained models on rating prediction; fine-tuned LLMs match them with a fraction of the data (basis of prompt variants V1, V4, V5). | arXiv preprint (2023) | https://arxiv.org/bibtex/2305.06474 |
| `koren2009matrix` | Biased matrix factorisation (user and item biases), the temporal MF reference row. | Computer 42(8) (2009) | https://api.crossref.org/works/10.1109/MC.2009.263 |
| `ferraridacrema2019progress` | Neural recommenders often fail to beat simple, well-tuned non-neural baselines (need for non-LLM references). | Proceedings of the 13th ACM Conference on Recommender Systems (RecSys '19) (2019) | https://api.crossref.org/works/10.1145/3298689.3347058 |
| `nogueira2020document` | Pointwise relevance ranking by the probability of the ``true'' token (IR counterpart of yes/no scoring). | Findings of EMNLP 2020 (2020) | https://aclanthology.org/2020.findings-emnlp.63/ |
| `jones2021selective` | Confidence-based abstention can raise average accuracy while widening accuracy gaps between groups (niche-user coverage). | International Conference on Learning Representations (2021) | https://iclr.cc/virtual/2021/poster/3060 + arXiv 2010.14134 |
| `spearman1910correlation` | Spearman--Brown prophecy formula (with `brown1910experimental`), used for the reliability $r_8$. | British Journal of Psychology 3(3) (1910) | https://api.crossref.org/works/10.1111/j.2044-8295.1910.tb00206.x |
| `brown1910experimental` | Spearman--Brown prophecy formula (independent derivation). | British Journal of Psychology 3(3) (1910) | https://api.crossref.org/works/10.1111/j.2044-8295.1910.tb00207.x |

### Verified, not cited in the current draft

| Key | Supports (one line) | Venue (year) | Source URL used |
|---|---|---|---|
| `tian2023justask` | For RLHF models, verbalised confidence is often better calibrated than conditional token probabilities. | Proceedings of EMNLP 2023 (2023) | https://aclanthology.org/2023.emnlp-main.330/ |
| `kapoor2024taught` | Fine-tuning on about 1,000 graded examples yields better calibrated uncertainty than prompting alone. | Advances in NeurIPS 37 (2024) | https://api.crossref.org/works/10.52202/079017-2729 |
| `zhuang2024beyond` | Pointwise yes/no LLM rankers improve with fine-grained relevance labels. | NAACL 2024, Volume 2 (Short Papers) (2024) | https://aclanthology.org/2024.naacl-short.31/ |
| `thomas2024searcher` | LLM relevance labels agree with searcher preferences about as well as human labellers. | Proceedings of SIGIR 2024 (2024) | https://api.crossref.org/works/10.1145/3626772.3657707 |
| `lichtenberg2024popularity` | A prompted LLM recommender showed less popularity bias than traditional recommenders, without mitigation. | arXiv preprint; Gen-IR@SIGIR24 workshop per arXiv comments (2024) | https://arxiv.org/bibtex/2406.01285 |
| `mozafari2026pretraining` | LLM popularity judgements align more with pretraining exposure than with page views. | Proceedings of SIGIR 2026 (2026) | https://api.crossref.org/works/10.1145/3805712.3809958 |
| `ni2026popular` | LLM confidence tracks the popularity of the generated answer, even when wrong (title of arXiv v2). | arXiv preprint; EMNLP 2026 Main per arXiv comments (2026) | https://arxiv.org/bibtex/2505.17537 |
| `hanley1982roc` | AUC as the probability that a positive is ranked above a negative, with its standard error (support for the binormal proposition, which was cut). | Radiology 143(1) (1982) | https://api.crossref.org/works/10.1148/radiology.143.1.7063747 |

### Discrepancies found in this pass

| Key | Notes said | Fetched record says | Action |
|---|---|---|---|
| `platt2000probabilities` | "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods", 1999 | MIT Press chapter "Probabilities for SV Machines", pp. 61--74, book year 2000; the familiar title is the 1999 preprint, which could not be fetched | Publisher record used |
| `hou2024bridging` | arXiv 2403.03952 | v1 (2024) introduces Amazon Reviews 2023; v2 (Apr 2026) adds a subtitle and an author and is published at ACL 2026 (DOI 10.18653/v1/2026.acl-long.147, pp. 3251--3265) | v1 cited (it introduces the dataset); switching to the ACL 2026 record is an open decision |
| `harper2015movielens` | TiiS 5(4), 2015 | online Dec 2015, issue Jan 2016; article 19 | key keeps 2015; pages written 19:1--19:19 as the GroupLens README asks |
| `holm1979simple` | -- | no working DOI | JSTOR stable URL |
| `kapoor2024taught` | "... Do Not Know" | "... Don't Know"; CrossRef lists "Andrew Wilson", the PDF byline "Andrew Gordon Wilson" | published title used |
| `mozafari2026pretraining` | "... in LLMs", SIGIR 2026 short | "... in Large Language Models"; short-paper status not confirmed (ACM page 403) | published title used |
| `ni2026popular` | arXiv 2505.17537 (2025) | v1 (2025) had a different title; v2 (2026) carries this title | arXiv year 2026 |
| `nogueira2020document` | "monoT5" | the paper says "T5 reranker" | the name monoT5 is not used in the draft |
| `jones2021selective` | -- | no DOI or pages; OpenReview not fetched | URL of the iclr.cc poster page |

### Check of the G9 anchor (no new entry)

`zhang2025collm` (CoLLM, TKDE 2025; author order Zhang, Feng, Zhang, Bao, Wang, He as in the existing entry), Table 2 read from
arXiv HTML v1--v3: ML-1M UAUC of MF 0.6361, TALLRec 0.6818 and ICL 0.5268 (AUC 0.6482, 0.7097, 0.5320). Labels: ratings above 3
are positive and all others, 3 stars included, negative; split by timestamp over the last 20 months (10 / 5 / 5 for
train / validation / test). This supports the protocol text "published ML-1M UAUCs of MF (0.636) and fine-tuned TALLRec (0.682)
under harder labels (3 stars as negatives)".

### Not cited (cannot be verified)

- The C-CRP paper and the benchmark protocol it defines (MISSING_REF of the related-work and results writers): no anonymous or public
  record can be cited without de-anonymising; the draft names C-CRP and "an existing same-candidate benchmark protocol" in the third
  person only.
