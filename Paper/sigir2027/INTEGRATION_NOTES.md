# Integration notes (text integrator, 2026-10-04)

Scope: alignment with the binding texts, page budget, consistency audit, citations, compile check. No result slot was filled and
no number was invented. All compiling was done in a private copy; only the nine section files, `references.bib` and
`CITATION_MAP.md` were written to this directory (main.tex untouched).

## 1. Page counts (pre-reference part, appendix included; SIGIR 2025 rule: at most 9 pages)

Measured on the compiled PDF of a measurement copy of `main.tex` with `\clearpage` before the bibliography (so every float is
placed before the references), as page index of the last content plus the used fraction of that page.

| state | skeleton (slots as red text) | typical filled cells |
|---|---|---|
| before integration | 11.61 | 11.86 |
| **after integration** | **8.86** | **8.91** |
| after integration, METHOD-SLOT blocks removed (information only) | -- | 8.71 |

"Typical filled": every table slot replaced by the results writer's typical cell (`\textbf{0.123 {\scriptsize$\pm$.012}}`, the
Gate-FT per-seed cell by three values), prose slots by neutral filler of the intended length (introduction findings 36 words,
remedy and Gate-FT branches 30, conclusion 55 / 40 / 22 words, abstract 17, protocol outcome slot 35, method-slot outcome 20; for
alternative-branch slots the longest alternative; the two "else delete" branches of the introduction removed). The filled draft
ends at page 9, right column about 93% full: the margin is about four lines. Re-run the measurement after the real fill
(the filler is an estimate; real reading-template fills with several numbers per slot can be longer).

## 2. Alignment with the binding texts (task 1)

a. **Non-prior share.** Wherever the statistic is 1 - rho^2/r_8 the text now says "non-prior share (an upper bound on the personal
   share)": abstract (slot description), introduction (C2 and findings slot), conclusion (slot description), preliminaries,
   tools, related work. Preliminaries rewritten to Amendment 3 section 3: personal share S defined as the target; non-prior share
   1 - rho^2/r_8 with rho the pooled within-unit correlation and r_8 = 2 r_c/(1 + r_c) (the r_8c of the amendment; the paper writes
   r_8 everywhere, as the tables do) "an upper bound on S (it also contains donor noise and user-by-prior interactions)";
   G = dUAUC(M2 - M1) of cross-fitted stackers, G_CF from M3 versus M0 (estimators in the tools section: cross-fitting over the two
   user halves of S_d's TEST rows). The old "stacker fitted on dev, evaluated on confirm users" and r_K are gone. Lemma 3.1 and
   Proposition 3.2 are unchanged. Proposition 3.3 (binormal, PROP3) was cut for space (text in section 9 below).
b. **Holm.** Introduction C1 now says: Holm within registered endpoint families, with the primary fine-tuning hypothesis and the
   pruning contrast as single tests. The protocol lists the families exactly as Amendment 3 section 11 (E-B over the four Qwen
   domains; G and the star-permutation contrast per domain and regime; the slot over its datasets) plus the next-item families of
   the audit spec (each endpoint over the four domains); "decomposition endpoints per domain and regime" was narrowed to "G and the
   star-permutation contrast".
c. **"Familiarity".** Related-work paragraph renamed "Yes-bias, answer priors and brand effects". The word now appears only in the
   wording rule (Table tab:rq, RQ5 rule; RQ5 endpoint sentence) and in the RQ5 branch slot. ("unfamiliar" in the UNIT sentence is a
   different word.)
d. **Registration timing.** Introduction and conclusion: "registered before any confirmatory result was analysed". Protocol
   ("Registration"): the next-item panels were scored under a registered carve-out with endpoints declared beforehand; the analysis
   specification was written while the scoring ran and frozen before any audit score was analysed; one later addendum, made before
   any niche value was seen, redefined the niche bins (PILOT_LOG and spec section G). Appendix amendments paragraph likewise.
e. **Baselines' own confidence.** The two "baselines' own margin" rows of tab:serving (10 slots) were deleted; the caption of the
   table that now holds selective serving (tab:anatomy, block C) and the RQ3 endpoint sentence say the comparison is not run (no
   score files), as Amendment 3 section 11 requires. Nothing promises it any more (the introduction's C3 compares exposure with the
   baselines, which is run).
f. **Second backbone (Z2).** Protocol: Llama replicates on next-item TEST events 1,001--3,000 (zero-shot, V0, next and like,
   temperature from 500 VALID events); tools: T fitted per domain, question and backbone (2,000 VALID events; 500 for the second
   backbone). Claim admission (protocol): Llama rated arms for rated findings, the Llama next-item sample for next-item findings,
   pruning and method slot single-backbone (Amendment 2 F, Amendment 3 sections 4 and 11). The `aud2q/aud2l` alias now says sections
   A--E; the guide comment now says that an S1/S2 next-item claim meets the second-backbone clause only through Z2's error-anatomy
   rows, and **two rows were added to tab:app-z2** (acc_high - acc_low; top-1 errors in the top tertile; 8 new slots
   `aud2q/aud2l:C.acc_top_minus_bottom_tertile`, `...C.share_errors_in_top_tertile`) so that such a claim can be shown.

Other inconsistencies found and fixed:
- Abstract: "per-user monotone calibration" -> "strictly increasing per-user calibration map" (non-strict maps can change AUC).
- Niche/mainstream users = lowest/highest **non-empty** popularity quintile (spec addendum) in the tools section.
- The knockout's registered seen/unseen-item split is reported as not run (Amendment 3 addendum 1, item 9).
- G9 anchor made precise and verified (section 5 below): "published ML-1M UAUCs of MF (0.636) and fine-tuned TALLRec (0.682) under
  harder labels (3 stars as negatives)".
- The protocol no longer says "otherwise the fine-tuned regime carries the decomposition and knockout analyses" (Amendment 3 runs
  the zero-shot decomposition in every branch; the knockout belongs to the fine-tuned programme).
- E1 is defined once (protocol); the full G0 variant exclusion list moved to the caption of tab:app-bank.
- S3 "Actionable via" slot description "exposure-aware serving" (not a registered route) -> "selective serving vs. random signal:
  niche served share".

## 3. Page-budget cuts and moves (task 2)

Removed:
- Table tab:rules (gate rules, table*) -> one paragraph "Registered gates and the one-round remedy" with every G0--G9 constant;
  eligibility/tie/simplicity-order details moved to the appendix text beside tab:app-sens.
- Proposition 3.3 (PROP3, binormal; text kept in section 9).
- Diagnosis battery: appendix section app:battery and tab:app-battery (6 slots `diag:*`) removed (results writer's cut order 1--2);
  one clause in the Findings and in the amendments paragraph says the battery is interpretive and its readings are released.
- "Baselines' own margin" rows of tab:serving (task 1e).
- Introduction RQ list folded into one paragraph; the "Findings in brief" scope sentence removed (scope stays in the abstract and
  the limitations); C3 no longer repeats the eight baseline citations (they are cited in the protocol).
- Endpoint paragraphs of the Findings reduced to pointers into tab:rq (wording rules kept); RQ1 and RQ2 share one subsection; the
  slot subsection of the Findings became a run-in paragraph (inside its METHOD-SLOT markers).
- Related work shortened (0.88 -> 0.6 page); "Exposure-biased evaluation" folded into "Popularity, exposure and feedback loops".
- Captions shortened throughout; `\arraystretch` 0.9 in the result tables; tab:rq at `\scriptsize`.
- Appendix: sections merged into one ("Prompts, Registration and Reproducibility") plus the per-panel tables (their heading and
  lead-in sentence removed); amendments paragraph shortened; reproducibility items that repeat the protocol removed (model names,
  main seeds, rerun rule).
- Tools: Bias Index equation inlined (eq:bi was unreferenced); MIRROR's acquiescence term dropped (not an endpoint); knockout
  decision rules (NEGATIVE/NULL/INCOMPLETE) moved to the appendix knockout paragraph.

Merged (no slot deleted, slots moved):
- tab:decomp merged into tab:reliability as block B (one table*, same columns); references now read "Table tab:reliability B".
- tab:anatomy transposed into a single-column table (rows = metrics, columns = the five panels) that also holds block A of
  tab:popularity (now block B) and the appendix table tab:serving (now block C, 20 slots moved from appendix.tex to
  experiments.tex). tab:popularity now holds only the knockout and sits inside one BRANCH-GATEFT pair (the pair formerly around its
  block B).
- tab:guide: the "Registered endpoint (table)" column folded into the question column (table numbers kept; endpoint names are in
  tab:rq; the full list is in the table's source comment).
- Appendix tables now precede the appendix prose (better float placement); single-column floats use `[!htbp]`.

Not cut (the cut orders allowed it, not needed): ECE/Brier of tab:anatomy, APLT block, per-domain rows of tab:app-seeds, share/ECE
rows of tab:app-llama.

## 4. Consistency audit (task 3)

- Every `\ref` resolves (0 undefined references or citations in the final compile). Unreferenced labels are section labels and
  `eq:logit`, `eq:decomp` (harmless). Removed labels: tab:rules, prop:contam, eq:bi, app:battery, tab:app-battery, tab:serving,
  tab:decomp, app:tables (none referenced any more).
- Anonymity: no "Lumen", author, institution, URL or "our previous work" in any section file; C-CRP and the benchmark protocol are
  third person. (references.bib keeps an old comment "Baselines (Lumen main table)" and the section comments contain repository
  paths: harmless in the PDF, but strip them before any source upload.)
- British spelling harmonised: verbalised, towards, programme, artefact, normalised, centred, behaviours (no -ize/-yze forms left).
- Notation: K = number of donors only (the star permutation is "two permutations"); the popularity link has no symbol any more
  (calligraphic L removed); the time split is T_d everywhere (T is the list temperature); the per-user top-k decision uses k_u
  (k is the NDCG/HR cut-off); "partial rho" row labels -> "partial Spearman" (rho is the decomposition correlation).
- Slots (`\DATANEEDED`, non-comment) before -> after: abstract 3 -> 3; introduction 8 -> 8; related 0 -> 0; preliminaries 0 -> 0;
  observation 2 -> 2; method 1 -> 1; experiments 544 -> 564 (+20 moved from tab:serving); conclusion 7 -> 7; appendix 220 -> 192
  (-6 battery, -10 baselines' own margin, +8 tab:app-z2, -20 moved). **Total 785 -> 777.** Slot descriptions edited (not filled):
  abstract slot 1, introduction findings slot 2, conclusion slot 1 (non-prior share wording); guide S3 "Actionable via" (see 2).
- Markers unchanged in number and position semantics: METHOD-SLOT pairs in abstract, introduction, related work, method,
  experiments (2), conclusion, appendix; BRANCH-GATEFT 4 (experiments) + 2 (appendix); BRANCH-GATEPASS 1 + 1. PROP3 markers removed
  with the cut.

## 5. Citations (task 4)

Verified on the web (CrossRef, publisher pages, JSTOR, PubMed, arXiv) and added to references.bib with CITATION_MAP rows:
- **cited (13):** krichene2020sampled (preliminaries, related, limitations), holm1979simple (protocol), platt2000probabilities
  (preliminaries, tools), zadrozny2002transforming (preliminaries), harper2015movielens and hou2024bridging (protocol),
  koren2009matrix (protocol), kang2023llmrating (appendix, variant bank V1/V4/V5), ferraridacrema2019progress (introduction),
  nogueira2020document and jones2021selective (related), spearman1910correlation and brown1910experimental (preliminaries).
- **verified, not cited (no room; kept for later):** tian2023justask, kapoor2024taught, zhuang2024beyond, thomas2024searcher,
  lichtenberg2024popularity, mozafari2026pretraining, ni2026popular, hanley1982roc.
- **kweon2024perk:** pages 3388--3399 and DOI confirmed; CITATION_MAP row added (it was missing).
- **CoLLM Table 2 (G9 anchor):** read in arXiv HTML v1--v3: ML-1M UAUC MF 0.6361, TALLRec 0.6818 (ICL 0.5268); labels rating > 3
  positive, 3 stars negative; temporal 10/5/5-month split. The protocol text is kept, made precise, and cited.
- **Not cited (cannot be verified without de-anonymising):** the C-CRP paper / benchmark-protocol paper; the text keeps them in the
  third person ("an existing benchmark protocol", "an existing verbalised reranker (C-CRP)").
- Differences from the writers' notes: Platt's record is the MIT Press chapter "Probabilities for SV Machines" (2000), not the 1999
  preprint title; Mozafari's title ends "in Large Language Models"; Kapoor's title has "Don't"; Ni et al.'s title is that of arXiv
  v2 (2026); Holm has no working DOI (JSTOR URL used); Hou et al. v2 is now ACL 2026 with a subtitle and an added author (v1 cited,
  see open question 1).

## 6. Compile (task 5)

Private copy, `main.tex` as in this directory, pdflatex + bibtex + pdflatex x2: 0 errors, 0 undefined references, 0 undefined
citations; BibTeX: only the usual ACM "empty address/publisher, pages missing" warnings. Overfull boxes: in skeleton form only
inside red slot text (slot names, acceptable); in the filled measurement one 3.9 pt box from a filler token
("GATE_FAIL_AFTER_REMEDY(confirm)"); no prose overfull box.

## 7. Open questions for the main session

1. Hou et al. (Amazon Reviews 2023): keep arXiv v1 (2024, the version that introduces the dataset) or switch to the peer-reviewed
   ACL 2026 version (different subtitle, added author)?
2. PILOT_LOG now records S_d = all TEST users with both classes in every domain. Rows "UAUC of l on S_d", the item-mean and popularity
   references and "Users in S_d" of block B duplicate block A; deleting them (20 slots) would save about 0.03 page. Not done (slots).
3. G5/G6 outcomes are now in PILOT_LOG (FIX_FOUND, V* = V1; GATE_PASS on the point estimate with an interval extending below 0.60).
   The branch slots and gate rows are untouched and must be filled from the committed files; when filling, report both the pass on
   the point estimate and the interval, as the log does.
4. The diagnosis battery is no longer reported in the paper (only "released with the artefact"). Acceptable, or restore a compact table
   if a page frees up?
5. Single-column floats now use `[!htbp]`; re-check after filling that no float drifts past the references (run the measurement
   with `\clearpage` before the bibliography).
6. The method slot costs about 0.2 page; with it the filled estimate is 8.91, without it 8.71.

## 8. Where a registered text and the paper may still differ

- Amendment 2 P still says "Holm correction within each RQ" and promises the serving comparison with the baselines' own
  confidence; the paper follows Amendment 3 section 11 (explicit families; comparison not run). Amendment 3 supersedes; no change
  needed, but the main session may want Amendment 2 P annotated.
- Star permutation: Amendment 3 addendum 1 (item 4) defines dUAUC as UAUC(L) minus the per-user AUC averaged over the two
  permutations; the tools text writes |UAUC(l) - UAUC(l_perm)| without spelling out the averaging (consistent, less explicit).
- Amendment 3 section 3 lists all-rows and seen/unseen UAUC as secondaries; the paper lists them in tab:rq (RQ4) and keeps per-seed
  all-rows UAUC in tab:app-seeds; seen/unseen values are not tabulated (released files).

## 9. Cut text of Proposition 3.3 (restore verbatim if a page frees up; rename its rho, e.g. to kappa, because rho is now the
## decomposition correlation)

```latex
% PROP3-BEGIN
\begin{proposition}[Item prior and discriminability]\label{prop:contam}
Fix a unit; let $A(\cdot)$ be the population AUC, $\Pr(\text{a positive outscores a negative})$, of a score on its
candidates (UAUC estimates its mean over units). Assume that, given $y$, $\pi(i)$ and $e(u,i)$ are independent Gaussians
with class means $\nu_y,\mu_y$ (unit constants are absorbed in $b_u$) and class-independent standard deviations
$\tau_\pi,\tau_e$. With $d_e=(\mu_1-\mu_0)/\tau_e$, $d_\pi=(\nu_1-\nu_0)/\tau_\pi$ and $\rho=\tau_\pi^2/(\tau_\pi^2+\tau_e^2)$,
\[
  A(\ell)=\Phi\Big(\tfrac{1}{\sqrt2}\big[\sqrt{1-\rho}\,d_e+\sqrt{\rho}\,d_\pi\big]\Big),\qquad A(e)=\Phi\big(d_e/\sqrt2\big).
\]
If the prior carries no label information ($d_\pi=0$), then $\Phi^{-1}(A(\ell))=\sqrt{1-\rho}\,\Phi^{-1}(A(e))$.
\end{proposition}
\begin{proof}
For a positive $i$ and a negative $j$, $\ell(u,i)-\ell(u,j)$ ($b_u$ cancels) is Gaussian with mean
$\tau_\pi d_\pi+\tau_e d_e$ and variance $2(\tau_\pi^2+\tau_e^2)$; divide, using
$(\tau_\pi,\tau_e)/\sqrt{\tau_\pi^2+\tau_e^2}=(\sqrt\rho,\sqrt{1-\rho})$. Dropping $\pi$ gives $A(e)$.
\end{proof}
So a label-free item prior shrinks the sensitivity index of $e$ by $\sqrt{1-\rho}$, where $\rho$ is the item-prior share of
within-class variance (the analogue of $1-\mathcal S$). Subtracting a noisy $\hat\pi=\pi+\eta$ ($\eta$ Gaussian, independent
of $y$) leaves $e-\eta$, the case $d_\pi=0$ with $\tau_\eta$ for $\tau_\pi$, so the reliability of $\hat\pi$ must accompany any
prior-subtraction result. The binormal form is a benchmark, not a claim about the data.
% PROP3-END
```
(`hanley1982roc` in references.bib is the verified reference for the binormal AUC if the proposition returns.)
