# CITATION_NOTES 2026-10-06 (SIGIR 2027 paper, D:\Research\Lumen\Paper\sigir2027)

Companion to `citations_added_2026-10-06.bib` (8 new entries). `references.bib`, the `sections\*.tex` files and everything else were
not edited. Searches used only generic technical queries (titles, author surnames, venues); nothing from the unpublished paper, the
repository or its authors was sent. Third-party papers are summarised in my own words; the only quotations are lines of the
unpublished paper and arXiv comment fields.

## 0. Result in brief

- All 12 works exist and match the requested title/venue/year; none is left unverified. Two need a caveat:
  - item 10: the requested title is only the arXiv metadata title; the KDD'26 proceedings title is different (section 1, item 10);
  - item 2: the publisher metadata and the printed PDF byline disagree on the order of the last two authors (item 2).
- 4 of 12 are already in `references.bib` (items 3, 4, 11, 12) with correct, complete fields; they are not repeated in the new .bib.
- 8 are new and are in the new .bib (keys below). The new file was test-compiled together with `references.bib` using
  `ACM-Reference-Format.bst` (MiKTeX bibtex): 0 errors, 0 repeated entries, 0 key collisions; only the usual "empty address" warnings.
- Part 2: of the five descriptions, (a) CCS is wrong as written, (b) UNIT and (c) Lost in Sequence are overstated or mis-scoped,
  (e) PerRecBench is overstated, (d) GORACS/DEALRec are described accurately but their real finding is not used. Corrected wordings
  are in section 2. Spot checks (section 3) found no further mischaracterisation among 6 sampled citations.

## 1. Part 1: the 12 works

| # | Work | Verified | Key | Already in references.bib? |
|---|------|----------|-----|----------------------------|
| 1 | GUIDER (AAAI 2026) | yes | `xu2026guider` | no, added |
| 2 | Disentangling User Interest and Conformity (WWW 2021) | yes, author-order caveat | `zheng2021dice` | no, added |
| 3 | LLMs Must Be Taught to Know What They Don't Know (NeurIPS 2024) | yes | `kapoor2024taught` | yes, correct |
| 4 | ConfTuner (NeurIPS 2025) | yes | `li2025conftuner` | yes, correct |
| 5 | Don't Take the Easy Way Out (EMNLP-IJCNLP 2019) | yes | `clark2019easyway` | no, added |
| 6 | Dataset Cartography (EMNLP 2020) | yes | `swayamdipta2020cartography` | no, added |
| 7 | Beyond neural scaling laws (NeurIPS 2022) | yes | `sorscher2022pruning` | no, added |
| 8 | SelectIT (NeurIPS 2024) | yes | `liu2024selectit` | no, added |
| 9 | Item-side Fairness of LLM-based RS (WWW 2024) | yes | `jiang2024itemfairness` | no, added |
| 10 | "Uncertainty-Calibrated Recommendations for Low-Active Users" (KDD 2026) | yes, TITLE DIFFERS | `zou2026uncertainty` | no, added |
| 11 | Obtaining Calibrated Probabilities with Personalized Ranking Models (AAAI 2022) | yes | `kweon2022calibrated` | yes, correct |
| 12 | Discovering Latent Knowledge in Language Models Without Supervision (ICLR 2023) | yes | `burns2023ccs` | yes, correct |

Source types: "Crossref" below means the publisher-deposited record at api.crossref.org/works/<doi>. The ACM Digital Library returned
HTTP 403 for every page I tried, DBLP is behind an anti-bot challenge (not bypassed), and OpenReview was not fetched, so for ACM
papers Crossref (deposited by ACM) is the publisher record, cross-checked with arXiv and, for item 2, Semantic Scholar's DBLP key.

### Item 1. GUIDER
- Verified: yes. Authors as published: Cai Xu, Xujing Wang, Ziyu Guan, Wei Zhao, Meng Yan. Proceedings of the AAAI Conference on
  Artificial Intelligence, vol. 40, no. 19, pp. 16049-16057, 14 March 2026, DOI 10.1609/aaai.v40i19.38639.
- Sources: https://ojs.aaai.org/index.php/AAAI/article/view/38639 ; https://api.crossref.org/works/10.1609/aaai.v40i19.38639/transform/application/x-bibtex ;
  https://doi.org/10.1609/aaai.v40i19.38639
- Key `xu2026guider`; not in references.bib. No arXiv version found (the arXiv hit 2603.11901 returned by a search is an unrelated
  paper). Family name is "Xu" (the same group's EviRank preprint, key `yan2026evirank`, uses the arXiv spelling "Xv").

### Item 2. Disentangling User Interest and Conformity (DICE)
- Verified: yes. Proceedings of the Web Conference 2021 (WWW '21), Ljubljana, pp. 2980-2991, April 2021, DOI 10.1145/3442381.3449788.
- Sources: https://api.crossref.org/works/10.1145/3442381.3449788 ; https://arxiv.org/abs/2006.11011 (v2, 19 Feb 2021, "Accepted by WWW'21") ;
  https://arxiv.org/pdf/2006.11011 (page 1 read) ; Semantic Scholar (DBLP key conf/www/ZhengGLHLJ21) ; OpenAlex ; ECNU Pure.
- Key `zheng2021dice`; not in references.bib.
- CAVEAT (author order): Crossref/ACM, DBLP (the key's initials Z-G-L-H-L-J encode Yong Li before Depeng Jin), Semantic Scholar,
  OpenAlex and ECNU give Yu Zheng, Chen Gao, Xiang Li, Xiangnan He, Yong Li, Depeng Jin (used in the .bib). The printed byline and the ACM Reference Format on page 1 of the camera-ready PDF (arXiv v2)
  give ... Xiangnan He, Depeng Jin, Yong Li (Yong Li is the corresponding author). Swap the last two names in the .bib if you prefer
  the PDF byline. I could not open the ACM page to see which order it shows beside the PDF.

### Item 3. Kapoor et al. (already in references.bib as `kapoor2024taught`)
- Verified: yes. Sanyam Kapoor, Nate Gruver, Manley Roberts, Katherine Collins, Arka Pal, Umang Bhatt, Adrian Weller, Samuel Dooley,
  Micah Goldblum, Andrew Gordon Wilson. NeurIPS 2024 (Advances in Neural Information Processing Systems 37), pp. 85932-85972,
  DOI 10.52202/079017-2729.
- Sources: https://proceedings.neurips.cc/paper_files/paper/2024/hash/9c20f16b05f5e5e70fa07e2a4364b80e-Abstract.html ;
  https://api.crossref.org/works/10.52202/079017-2729 (Crossref lists the last author as "Andrew Wilson"; the NeurIPS page has the full name) ;
  arXiv 2406.08391.
- Existing entry: title, 10 authors, volume 37, pages, DOI, URL all correct and complete. No change needed.

### Item 4. ConfTuner (already in references.bib as `li2025conftuner`)
- Verified: yes. Yibo Li, Miao Xiong, Jiaying Wu, Bryan Hooi. NeurIPS 2025 (Advances in Neural Information Processing Systems 38),
  pp. 59517-59546, DOI 10.52202/085713-1783.
- Sources: https://neurips.cc/virtual/2025/poster/117676 (title, authors; OpenReview id VZQ04Ojhu5) ;
  https://api.crossref.org/works/10.52202/085713-1783/transform/application/x-bibtex ; arXiv 2508.18847.
- Existing entry: correct and complete. Cosmetic option: add `volume = {38}` (the entry for item 3 has its volume; this one has it only
  inside `booktitle`). Not an error.

### Item 5. Don't Take the Easy Way Out
- Verified: yes. Christopher Clark, Mark Yatskar, Luke Zettlemoyer. EMNLP-IJCNLP 2019, Hong Kong, pp. 4069-4082, November 2019,
  DOI 10.18653/v1/D19-1418. Source: https://aclanthology.org/D19-1418/ (BibTeX endpoint D19-1418.bib). Key `clark2019easyway`; not in references.bib.

### Item 6. Dataset Cartography
- Verified: yes. Swabha Swayamdipta, Roy Schwartz, Nicholas Lourie, Yizhong Wang, Hannaneh Hajishirzi, Noah A. Smith, Yejin Choi.
  EMNLP 2020 (main), online, pp. 9275-9293, November 2020, DOI 10.18653/v1/2020.emnlp-main.746.
  Source: https://aclanthology.org/2020.emnlp-main.746/ (BibTeX endpoint). Key `swayamdipta2020cartography`; not in references.bib.

### Item 7. Beyond neural scaling laws
- Verified: yes. Title as printed on the paper and on the proceedings page (lower-case after the colon). Ben Sorscher, Robert Geirhos,
  Shashank Shekhar, Surya Ganguli, Ari S. Morcos. NeurIPS 2022 (Advances in Neural Information Processing Systems 35),
  pp. 19523-19536, DOI 10.52202/068431-1419.
- Sources: https://proceedings.neurips.cc/paper_files/paper/2022/hash/7b75da9b61eda40fa35453ee5d077df6-Abstract-Conference.html ;
  the proceedings PDF (page 1 read: byline "Ari S. Morcos"; the proceedings page and Crossref print "Ari Morcos") ; arXiv 2206.14486 ;
  Crossref (found by title query; page range from there, the NeurIPS page lists none).
- Key `sorscher2022pruning`; not in references.bib. Used the printed name "Morcos, Ari S.".

### Item 8. SelectIT
- Verified: yes. Liangxin Liu, Xuebo Liu, Derek F. Wong, Dongfang Li, Ziyi Wang, Baotian Hu, Min Zhang. NeurIPS 2024
  (Advances in Neural Information Processing Systems 37), pp. 97800-97825, DOI 10.52202/079017-3102.
- Sources: https://proceedings.neurips.cc/paper_files/paper/2024/hash/b130a5691815f550977e331f8bec08ae-Abstract.html ; the proceedings
  PDF (page 1 read: "Derek F. Wong"; Crossref has "Derek Wong") ; Crossref (page range; the NeurIPS page lists none) ; arXiv 2402.16705.
- Key `liu2024selectit`; not in references.bib.

### Item 9. Item-side Fairness of Large Language Model-based Recommendation System
- Verified: yes. Meng Jiang, Keqin Bao, Jizhi Zhang, Wenjie Wang, Zhengyi Yang, Fuli Feng, Xiangnan He. Proceedings of the ACM Web
  Conference 2024 (WWW '24), pp. 4717-4726, May 2024, DOI 10.1145/3589334.3648158.
- Sources: https://api.crossref.org/works/10.1145/3589334.3648158/transform/application/x-bibtex ; https://arxiv.org/abs/2402.15215
  (comment: accepted by the ACM Web Conference 2024; same 7 authors in the same order). ACM page not reachable (403).
- Key `jiang2024itemfairness`; not in references.bib.

### Item 10. "Uncertainty-Calibrated Recommendations for Low-Active Users" (KDD 2026)
- Verified: yes, it exists in the KDD 2026 proceedings, but under a different title. Published title: **Uncertainty-Aware Adaptive
  Recommendation across User Lifecycle**. Authors: Bob Junyi Zou, Sai Li, Tianyun Sun, Wentao Guo, Qinglei Wang. Proceedings of the
  32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '26), Jeju Island, pp. 8595-8603 (9 pages), August 2026,
  DOI 10.1145/3770855.3818501.
- Evidence that the two titles are one paper: arXiv 2605.17788 (v1 18 May 2026, v2 25 May 2026) still carries the title you gave in
  its metadata, states "Accepted to the Applied Data Science (ADS) track at KDD 2026", and lists the KDD'26 V.2 journal reference and
  the DOI above; but page 1 of the v2 PDF is titled "Uncertainty-Aware Adaptive Recommendation across User Lifecycle" with the ACM
  Reference Format and that DOI. Crossref (ACM deposit), OpenAlex and Semantic Scholar all show the proceedings title, and
  Semantic Scholar links that DOI to arXiv 2605.17788.
- Sources: https://arxiv.org/abs/2605.17788 ; https://arxiv.org/pdf/2605.17788v2 (page 1 read) ;
  https://api.crossref.org/works/10.1145/3770855.3818501 ; OpenAlex and Semantic Scholar records for the DOI.
- Not reached: the ACM DL page (403) and the official KDD 2026 program page (not found by search). So the program listing itself is
  unchecked; the Crossref proceedings record plus the PDF's own ACM Reference Format stand in for it.
- Key `zou2026uncertainty` (this is the key named in the header comment of sections/related_work.tex as "not cited"; it is not in
  references.bib). Cite the proceedings title; a reader who searches the arXiv title will still land on the same paper.

### Item 11. Kweon, Kang, Yu (already in references.bib as `kweon2022calibrated`)
- Verified: yes. Wonbin Kweon, SeongKu Kang, Hwanjo Yu. Proceedings of the AAAI Conference on Artificial Intelligence 36(4),
  pp. 4083-4091, 28 June 2022, DOI 10.1609/aaai.v36i4.20326.
- Sources: https://ojs.aaai.org/index.php/AAAI/article/view/20326 ; https://api.crossref.org/works/10.1609/aaai.v36i4.20326/transform/application/x-bibtex ; arXiv 2112.07428.
- Existing entry: correct and complete. (No section currently cites it.)

### Item 12. CCS (already in references.bib as `burns2023ccs`)
- Verified: yes. Collin Burns, Haotian Ye, Dan Klein, Jacob Steinhardt. ICLR 2023 (poster). No DOI and no pages exist for ICLR 2023.
- Sources: https://arxiv.org/abs/2212.03827 (comment "ICLR 2023"; v2 PDF read, sections 2.2 and 3.1) ;
  https://iclr.cc/virtual/2023/poster/11480 (lists the OpenReview forum https://openreview.net/forum?id=ETKGuby0hcs ; I did not open the OpenReview page).
- Existing entry: title, 4 authors, booktitle, year, eprint and url all correct. Optional: use the OpenReview forum link as `url`.

## 2. Part 2: how the paper describes five works

Line numbers refer to `sections\*.tex` (the copies under `filled\sections\` have the same lines and need the same edits).

### (a) CCS (burns2023ccs) and the 2026 yes/no-bias paper

Paper text: related_work.tex L32 "the two-view contrast \method{} belongs to contrast-consistent search"; introduction.tex L80
"a two-view contrast~\citep{burns2023ccs} that we evaluate, not propose".

What CCS is (read in the paper, sections 2.2 and 3.1): an unsupervised probing method on a frozen LM's hidden states (last token of the
last layer, mean-normalised separately over the set of x+ states and the set of x- states), not on output probabilities. For each yes/no question it builds
a contrast pair, the question answered "Yes" (x+) and answered "No" (x-), and trains a linear probe with a sigmoid so that the two truth
probabilities are consistent, p(x+) = 1 - p(x-), and confident (not both near 0.5); there is no label in training. At test time it
predicts with the symmetrised average (1/2)[p(x+) + 1 - p(x-)]; labels only fix which side of the probe counts as "Yes" (the paper
reports the better of the two orientations).

Accuracy: "belongs to CCS" is wrong as written. Mirror's like-minus-dislike log-odds is read from model outputs, with no probe, no hidden
states, no consistency or confidence training. What they share is the negation-paired, symmetrised inference. The reviewer's relation
is correct: since logit(1 - p) = -logit(p), CCS's average becomes (1/2)[logit p(x+) - logit p(x-)] in log-odds, and l_like - l_dislike
is twice that quantity when the dislike prompt is read as the negation of the like prompt, so any additive Yes-bias common to the two
prompts cancels. One difference worth a clause if space allows: CCS flips the answer appended to one question, Mirror flips the
polarity of the question and reads the Yes/No log-odds of preliminaries.tex Eq. (1) in both prompts ("a mirrored question" is the
second readout named at preliminaries.tex L77); for a consistent model these coincide.

Corrected wording (related_work.tex L32):
`the two-view contrast \method{} is the log-odds analogue of the symmetrised inference rule of contrast-consistent search~\citep{burns2023ccs}`
Corrected wording (introduction.tex L80):
`a two-view log-odds contrast, the analogue of the symmetrised inference rule of CCS~\citep{burns2023ccs}, that we evaluate, not propose`

The 2026 arXiv paper that crosses a statement with its negation: **Haonan Huang, "The yes-no bias of large language models reflects
answer order and wording, not shifts in moral judgment"**, arXiv:2607.05552, submitted 6 July 2026 (cs.CL; also cs.AI, cs.CY), single
author, Princeton University. It is already in references.bib as `huang2026yesno` and the entry is correct (the PDF prints an en dash
in "yes-no"). I read the abstract and the first three pages. Summary: it poses 20 moral dilemmas to seven frontier chat-model
configurations and two small open-weight models, flipping in balanced pairs and crossing the question's verb (approve versus oppose, i.e.
the statement versus its opposite), the printed order of the Yes/No options, and the answer labels; the symmetric part of a flip-pair
estimates the stance and the antisymmetric part the format artifact. It finds the apparent yes/no bias is an order bias toward the
last-printed option plus a lexical pull toward the word "no", substantial only for the Claude models it tests (about zero for GPT-5.5
and the Gemini models) and smaller under extended reasoning; per-item residuals up to about 0.5 exist but are item-idiosyncratic and
cancel across dilemmas.

Accuracy of related_work.tex L31 ("yes/no answers show wording-driven bias with idiosyncratic per-item residuals"): partly accurate.
It omits the answer-order component (half of the title) and the scope (moral-dilemma vignettes asked of chat models, not token
probabilities or recommendation). Corrected wording:
`yes/no answers carry order and wording biases that crossing a statement with its negation separates, leaving idiosyncratic per-item residuals~\citep{huang2026yesno}`

### (b) UNIT (wu2025unit)

Paper text: introduction.tex L30 "fine-tuning data can be pruned by uncertainty"; related_work.tex L45 "instruction tuning prunes at
an uncertainty quantile"; method.tex L83-84 "P2 the zero-shot model's most uncertain examples ... in the spirit of UNIT".

What UNIT does (arXiv 2502.11962 v3, 25 June 2025; read pages 1-5, Fig. 2 and section 3.1; still arXiv-only, no venue found; v1 carried an
earlier title that begins "Navigating the Helpfulness-Truthfulness Trade-Off"): it works inside the
responses of an instruction-tuning set. It splits each response into atomic claims, scores each claim's uncertainty with
claim-conditioned probability under the base model, and UNIT_cut deletes the claims above the 75th percentile of all claim scores in the
training set, then has an auxiliary LLM rewrite the response from the kept claims. All instruction-response pairs are kept (a response
whose claims are all removed becomes an apology; one with none removed is unchanged). UNIT_ref instead keeps the response and appends
a reflection listing the uncertain claims. The aim is truthfulness (less hallucination on FactScore and WildFactScore, Llama-3.1-8B and
Qwen2.5-14B tuned on LIMA and LFRQA), at some cost in informativeness; it is not example selection, data-efficiency pruning or ranking.

Accuracy: loosely true for "pruned at an uncertainty quantile" if read as claims, but "fine-tuning data can be pruned" invites the
reading "examples are removed", which UNIT never does. "In the spirit of UNIT" is defensible only as an analogy (the base model's own
uncertainty, thresholded at a data-wide quantile, decides what fine-tuning content to withhold); P2 drops whole examples to study
ranking quality, UNIT cuts claims to study truthfulness.

Corrected wordings:
- introduction.tex L30: `fine-tuning responses can be stripped of the claims the base model is unsure of~\citep{wu2025unit}`
- related_work.tex L45: `instruction tuning cuts, from responses, the claims above a quantile of the base model's uncertainty~\citep{wu2025unit}`
- method.tex L84: `... ; an example-level analogue of the claim-level cut of UNIT~\citep{wu2025unit})`

### (c) Lost in Sequence (kim2025lostinsequence)

Paper text: introduction.tex L37-38 "LLM recommenders under-use the user's history~\citep{kim2025lostinsequence,zhang2024cft}".

What it shows (arXiv 2502.13909 v3; read pages 1-3; KDD '25 record confirmed in Crossref, pp. 1160-1171): for four LLM recommenders
(TALLRec, LLaRA, CoLLM, A-LLMRec, run as next-item recommenders) against SASRec, training on histories whose order is shuffled gives
about the same accuracy as training on the true order, and shuffling the history at inference barely hurts them, whereas SASRec
degrades sharply (sections 2.3.1-2.3.2); user representations from shuffled and original histories are far more similar for the LLM
recommenders than for SASRec (section 2.3.3). The conclusion is that they do not capture the sequential (order) information of the
history. The same items stay in the prompt under shuffling, so this says nothing about whether the content of the history is used (the
authors stress that text and sequence information are both needed); the paper then proposes LLM-SRec, which distils a CF sequence
model's user representation into the LLM.

Accuracy: too broad for this source. It shows order-insensitivity, not under-use of the history. The second cite (zhang2024cft,
arXiv 2410.22809) does say that current LLM recommenders fail to fully leverage user behaviour sequences, so the sentence is
supportable only via that source or with the narrowed wording.

Corrected wording: `that LLM recommenders barely use the order of the user's history~\citep{kim2025lostinsequence} and do not fully exploit behaviour sequences~\citep{zhang2024cft}`

### (d) GORACS (mei2025goracs) and DEALRec (lin2024dealrec)

Paper text: introduction.tex L30-31 "fine-tuning data can be ... coreset scores"; related_work.tex L44-45 "selected by influence and
effort (DEALRec) or coresets (GORACS)". These descriptions are accurate (DEALRec combines an influence score with an effort score;
GORACS is optimal-transport-guided coreset selection). What the paper does not use is what both report about hardness scores versus
random subsets, which is the evidence behind RQ6's random control.

- GORACS (KDD '25 V.2; arXiv 2506.04015, read pages 1-3 and 6-8; Tables 1-2 and section 5.2). One sentence: the hardness-based
  importance scores GraNd (gradient norm) and EL2N (prediction error), which keep the hardest examples, pick subsets that are worse than
  uniform random in 29 of 30 SeqRec cells (Table 1, BIGRec on Amazon Games/Food/Movies; for example Games NDCG@10 0.180 and 0.145
  against 0.207 for Random; the one exception is GraNd's HR@10 on Food) and in all 12 CTR cells (Table 2, TALLRec; AUC 0.45-0.47
  against 0.59-0.66 for Random, i.e. below chance), which the authors blame on a biased subset heavy in difficult samples (section 5.2,
  observation 2); DEALRec is level with Random in Table 1 (Games NDCG@10 0.2046 versus 0.2074) and is not run on the CTR task.
- DEALRec (SIGIR '24; arXiv 2401.17197 v2, read pages 5-7; Table 2 and section 4.2). One sentence: at 1,024-shot fine-tuning (1-2% of
  the data; BIGRec and TIGER on Games, MicroLens-50K, Book) GraNd and EL2N, computed on a surrogate SASRec, are not reliably better than
  Random (below it in 24 of 24 TIGER cells; mixed on BIGRec, below it in 8 and 6 of 12 cells, far below on MicroLens), the authors note
  that random sampling is competitive with or better than several coreset methods because it keeps coverage, and only DEALRec's
  influence-plus-effort score with coverage-aware sampling beats Random significantly. (In the arXiv v2 table the EL2N MicroLens R@50
  entry, 0.0045, is below its R@20 entry, 0.0096; do not quote that cell.)
- Caveat for RQ6: neither paper tests an uncertainty score of the zero-shot model (entropy, margin) nor removing the most uncertain
  examples; GraNd and EL2N retain the hardest examples, the opposite orientation to P2, which removes the most uncertain ones. They
  support having a random control and doubting difficulty-based selection, not any claim about P2.
- Optional sentence if wanted: `plain difficulty scores did not beat random subsets in LLM-recommender fine-tuning~\citep{lin2024dealrec,mei2025goracs}`

### (e) PerRecBench (tan2025perrecbench)

Paper text: introduction.tex L35-37 "under pointwise rating prompts, LLM recommenders follow item quality and user rating bias more than
personal preference".

What it does (arXiv 2501.13391 v1, 23 Jan 2025, arXiv-only, no peer-reviewed version found; read pages 1-2 and 4-6): rating-error
metrics can be won without personalisation, because user-average and item-average rating baselines match existing LLM rating
predictors on MAE and RMSE (Fig. 1). PerRecBench therefore groups users who bought the same item in a short window (item quality
fixed), ranks them by relative rating, the rating minus the user's own mean (user bias removed), and scores LLMs by Kendall's tau under
pointwise (predict each user's rating), pairwise and listwise prompts. Across 19 LLMs, tau is low (per-model averages 0.02-0.18),
pairwise and listwise prompts beat pointwise ones (Table 1, section 4), and tau correlates weakly with MAE/RMSE (Fig. 3). (The text
quotes method averages of 0.19, 0.38 and 0.35 that I could not reproduce from Table 1; cite the ordering, not those numbers.)

Accuracy: overstated. The paper shows that rating-error metrics reward user and item averages, that LLMs which predict ratings well
rank users poorly once both factors are removed, and that pointwise is the weakest format. It does not measure that LLM recommenders
"follow item quality and user rating bias more than personal preference"; that is an inference.

Corrected wording: `that, under pointwise rating prompts, LLM recommenders recover personal preference poorly once user rating bias and item quality are controlled~\citep{tan2025perrecbench}`
(longer option: `that LLMs which predict ratings well rank users poorly by preference once user rating bias and item quality are controlled, with pointwise prompts trailing pairwise and listwise ones`)

## 3. Spot checks of other citations in the three files (6)

Method: for each, I read the cited work's arXiv abstract page and, where the sentence is specific, the relevant section of its arXiv
HTML text; results are my paraphrase.

| # | Sentence (file, line) | Cited work | How checked | Verdict |
|---|------------------------|------------|-------------|---------|
| S1 | UGR "uses confidence tokens for rejection and truncation" (related_work L33-34; introduction L29) | fan2026ugr, arXiv 2602.11719 | abstract + HTML: section 3.4 (verbalised confidence tokens), section 4.5.2 (user-level rejection, item-level truncation) | accurate |
| S2 | KnowSA "finds popularity and uncertainty poor proxies of item knowledge" (related_work L35) | lee2026knowsa, arXiv 2604.07825 | abstract + HTML section 3.2 and Table 2 (popularity, Min-K% and eigenvalue-uncertainty selective variants underperform the proposed method) | accurate |
| S3 | "selective escalation to an LLM did not help in a sparse study" (related_work L36-37) | zhou2026routerec, arXiv 2607.09908 | abstract: negative result for selective LLM escalation; request-level routing too coarse for sparse settings | accurate (a workshop paper) |
| S4 | ReLLa "scores a pair by the two-way Yes/No softmax ... reporting AUC, log loss and accuracy" (related_work L14-16) | lin2024rella, arXiv 2308.11131 | HTML: section 2.3 (softmax over the Yes and No answer-word scores), section 4.1.2 (AUC, Log Loss, ACC) | accurate |
| S5 | "Generative recommenders ... face distorted sequence-level confidence" (related_work L19-20) | bao2024d3, arXiv 2406.14900 | abstract: length normalisation inflates the score of items with near-deterministic ("ghost") tokens (amplification bias), plus a homogeneity issue | accurate in substance; "confidence" is this paper's word for the sequence score |
| S6 | "fairly calibrated in token probabilities on multiple-choice questions" (related_work L28-29) | kadavath2022language, arXiv 2207.05221 | abstract: larger models are well calibrated on multiple-choice and true/false questions in the right format | accurate (scope: larger models, suitable format) |

Incidental checks made while doing Part 2: `huang2026yesno` (partly accurate, see (a)); `zhang2024cft` (supports "fail to fully leverage
behaviour sequences", see (c)); `kim2025lostinsequence` Crossref entry matches references.bib.

Minor point found while reading: introduction.tex L32-33 lists "AUC, per-user AUC, UAUC" and log loss with a single cite to ReLLa;
ReLLa reports AUC, log loss and accuracy but not UAUC (CITATION_MAP.md records UAUC under CoLLM and BinLLM). Add
`zhang2025collm,zhang2024binllm` to that cite or drop UAUC from the ReLLa-attributed list.

Not re-checked in this pass (no concern found, not read): xiong2024llmuncertainty, nogueira2020document, zhuang2024beyond, qin2024prp,
hou2024llmrank, dipalma2025memorize, chu2026incumbent, ni2026popular, lichtenberg2024popularity, gao2025flower, gao2025sprec,
wang2021tce, wang2026llm4dsr, fayyazi2025facter, kweon2025uqrec, yan2026evirank, ravikumar2026hallucinating, xiong2023procal,
holtzman2021surface, geifman2017selective, jones2021selective, bellogin2011predicting, kang2023llmrating.

## 4. Practical notes

- `Paper\sigir2027\filled\` holds second copies of `sections\` and `references.bib` (same line numbers). Apply any text edit and the
  new entries there too if that copy is the build source.
- A third directory, `Paper\sigir2027\src\` (own `main.tex`, `references.bib`, `sections\design.tex`, `sections\setting.tex`), appeared
  while I was working (files dated 15:40-15:46 on 2026-10-06). I did not edit or study it; a read-only key check shows its
  `references.bib` also has 102 entries, contains the four already-present keys, and has no collision with the 8 new keys.
- To use the new entries, either append `citations_added_2026-10-06.bib` to `references.bib` or list both in `\bibliography{...}`; the
  keys do not collide with the existing 102 entries.
- The new keys are not cited by any section yet; a work cited only in the .bib does not print. Items 3, 4, 11 and 12 (already in
  references.bib) are cited only for item 12 (`burns2023ccs`).
