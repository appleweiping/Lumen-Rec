# Integration notes

## 0. Editor pass 1 (2026-10-05; brief `docs/sigir/PAPER_REVISION_BRIEF.md`)

Scope: the revision brief (story: what the confidence tracks; what fine-tuning teaches; which uses survive a matched control; the
registered protocol and its deviations), the page budget, the anonymity and wording rules, the merge of the verified citations, and
the specs and tests of `scripts/sigir/fill_paper.py`. No number was typed: every result is a slot; the new endpoints of addendum 6
are `ext:` slots that stay red until `ftgrid_extra.py`'s files exist (editor pass 2 specifies them).

**Pages before the references** (`scripts/sigir/page_budget.py` on the synthetic result set; SIGIR rule 9 pages, appendix included):
9.24 before the pass, 8.90 after it (8.91 on a "typical" variant of the synthetic set with the minimum-n flags cleared, because
the synthetic Toys world has 140 users and flags every Toys cell "(descriptive)", which widens the rated tables). The measuring tool
had two bugs, fixed in this pass and reported: it measured the end of the text only, so floats still deferred at that point (flushed
by its own `\clearpage`) were not counted (at one stage the text ended on page 9 while the guide and every appendix table sat on
page 10); and its count of undefined references matched all warnings as one. It now records where every float ends and measures to
the later of the text end and the last float end. Latexmk needs Perl: run the tool with Git's `usr\bin` on PATH.

**What moved or changed** (section by section):
- Abstract: claims kept; the pruning route named "uncertainty-selected pruning" (addendum 7).
- Introduction: claims of the 2026-10-04 rewrite kept; the output-level citations corrected after verification (PerRecBench under
  "pointwise rating prompts"; Kang et al. only for "zero-shot LLMs trail CF, fine-tuned ones approach it"); RQ6 worded as in
  addendum 7; the guide named outside the contributions; the findings-in-brief slots follow the three-part story; the gate sentence
  says the bar is a point estimate. PROSE_SPECS anchors updated to this text (two slots read `skeleton_changed` at the start of the
  pass, four tests failed).
- Related work: about 0.5 page; three paragraphs (pointwise scoring; confidence, calibration and answer priors; popularity,
  exposure, pruning and serving) with the verified works listed below.

**Citations merged from the verification pass (2026-10-05).** Into `references.bib`, unchanged: `tan2025perrecbench`,
`zhang2024cft`, `kim2025lostinsequence` (introduction), `dipalma2025memorize` (related work, limitations), `qin2024prp`,
`wang2026llm4dsr`, `fayyazi2025facter`, `bellogin2011predicting`, `park2026echotrace` (related work); `anon2026benchmark` (`@misc`,
author "Anonymous", title "Omitted for double-blind review") added by hand. Already present and now cited: `hou2024llmrank`,
`zhuang2024beyond`, `lichtenberg2024popularity` (as finding less popularity bias), `ni2026popular` (entity QA), `wang2021tce`. Not
merged or not cited: `zhang2025shapley` (not LLM-based), `yang2026recloop` (Semantic-ID recommenders), `zou2026uncertainty` (non-LLM,
dropped for space), `cronentownsend2002predicting` (optional), `mozafari2026pretraining` (kept uncited). No discrepancy was left: the
verifier's notes required no field correction; the wording follows its notes (ReLLa's score is the two-way Yes/No softmax; FACTER
thresholds a fairness score; LLM4DSR thresholds a generation probability to clean histories).
- Setting: lemma and proposition shrunk to one display each, proofs inlined; decomposition paragraph unchanged in substance.
- Protocol: the gate paragraph of the protocol and the gate subsection of the findings merged into one paragraph plus the small
  tab:gate-outcomes (moved here; "Rule" column folded into the row labels; the pilot rows dropped, their values stay in the text);
  repeated constants dropped (the bars stay); new: the families E-F, E-H, E-J, the robustness rule of E-W, the count rule and the
  replication rule (interval, not sign); tab:rq compacted, RQ4/RQ5 cells gain the new endpoints, RQ6 says the arms are matched per
  class only and points to addendum 7.
- Tools: paragraphs on the deployable top-k decision (E-C'), the item-prior share, the e-share and the item shares of MF and labels
  (E-G), the within-user estimator (E-W), M4 (E-F), the strata (E-H), the matched item mean (E-J) and the FT-C reading rule; the
  other paragraphs shortened; the method slot is a run-in paragraph.
- Findings: 6.1 "What the confidence tracks" (tab:tracks, the former tab:reliability: same-row references including m_T, the E-J
  contrasts, item-prior share, e-share, item shares, G and G_wu beside G_CF, P1 and P1_wu, ECE and the oracle and deployable error
  anatomy); 6.2 "What fine-tuning teaches" (new tab:teaches: E-F, E-H, the FT-C reading, the knockout rows of the former
  tab:popularity, shortened); 6.3 "Which uses survive a matched control" (S1/S2, S3 with tab:anatomy and the shrunk tab:exposure,
  corrections, S6 with addendum-7 wording, the method slot); the guide subsection became the caption of tab:guide. Prose lists that
  repeated table cells were dropped; the prose keeps the registered words and adds count slots. The full-width tables are defined
  ahead of the text so that none is deferred past the appendix.
- Conclusion: "two statements do not depend on the data" replaced (an accounting identity); e is the pair-specific residual;
  limitations paragraph as listed in the brief.
- Appendix: the variant-bank and sensitivity tables became two sentences and a pointer (tab:app-sens spec kept dormant);
  tab:deviations added (18 rows, condensed; row 7 carries three slots); tab:app-llama follows tab:tracks and carries the Llama Toys
  knockout; tab:app-z2 keeps four rows; amendments paragraph lists addenda 6 and 7.
- Anonymity: "C-CRP" became "a verbalised reranker" in the text, the tables and `REF_METHODS`; the benchmark protocol is
  `\citep{anon2026benchmark}`; repository and path names, the project name and a research-group name were removed from TeX and bib
  comments.

**Open for the main session.** (1) The sha1 of the edited `fill_paper.py` must be recorded in PILOT_LOG before any further result file
is pulled (A3-6 section 9.3): this pass added specs (tables tab:tracks, tab:teaches, tab:deviations, the exposure summary rows, the
item-prior share row) and three new decision functions (`c_eb_count`, `c_g_count` and `p_mir_decision`; the count rule of A3-6 item
9.2 and the verbatim stage-3 label); no existing decision function changed. (2) `pull_results.ps1` must pull the ext files to the
paths of the data map (alias `ext`). (3) `CITATION_MAP.md` (not part of the paper sources) still names the project and a research group
in two headers: strip them before any source upload. (4) The title is three lines long; "... Item Priors, Personal Evidence and
Actionable Uncertainty" would fit two (the authors' decision).

## 0b. Editor pass 2 (2026-10-05; the ext slots against the real schema, the FT-Q control, the page budget)

Scope: every `ext:` slot written against the real schema of `src/confrec/ftgrid_extra.py` (sha1 8d0e5d86 during this pass; the
summary's structure as `tests/test_confrec_ftgrid_extra.py` builds it) and the real ML-1M file `docs/sigir/results/extra/ml1m.json`
(exploratory, addendum 6); the FT-Q control of addendum 8; rows 19-20 of the deviations record; the page budget. No Toys, Video
Games or Sports result was read (none is pulled locally) and nothing was pulled.

**Pages before the references** (the same tool, sha1 51c3cfaf, and the same synthetic sets as pass 1): 8.90 at the end of pass 1
(8.91 typical); 9.17 once the pass-2 content was in (FT-Q rows and text, deviations rows 19-20, the limitation); 8.85 after the trims
below (8.86 typical; 8.81 with synthetic extra-analysis files, built by the test module's `ext_synthetic_files` and summarized by
`ftgrid_extra` itself, whose prose slots carry the handlers' real text instead of 30-word fillers). LaTeX errors 0, undefined
references 0.

**What changed.**
- tab:teaches block C: FT-C is the ML-1M LoRA cell only ("Toys is added to FT-C" is gone, addendum 8); FT-Q rows "$R_Q$ / reading"
  and "UAUC of $q_0$ / $q_1$ / $m$" (addendum 8's descriptive companion) in the LoRA cells of the four panels.
- 6.2 names the two item-only controls with a pointer to Section 5 and carries one slot for summarize's addendum-8 wording with the
  labels per control and dataset; the E-F, E-H, E-J and robust-reading counts of 6.1/6.2 are filled from `extra/summary.json`.
- Section 5: FT-C (ML-1M only, and why) and FT-Q (positive for the round(beta n) TRAIN examples with the largest m; no label of the
  example itself, no user information, nothing after T_d), and the retention R / R_Q with its reading.
- Section 4: tab:rq RQ4 cells (FT-C, FT-Q retention; "teaches the item" needs FT-C (ML-1M) and FT-Q (each dataset run)
  ITEM_DRIVEN); the registration paragraph's "two later addenda", wrong once addendum 8 exists, became "later addenda ... the
  item-only teacher".
- Introduction C1, findings slot (ii), abstract slot 2 and the conclusion slot name the item-only controls; limitations: the controls
  are partial (the permutation informs on ML-1M only, 97.7% of its TRAIN examples in repeated items, from addendum 8's table; the
  FT-Q teacher departs from the real labels, so a high R_Q shows what an item-level target can teach).
- Appendix: tab:deviations rows 19-20 (condensed from DEVIATIONS.md); the amendments paragraph names addendum 8; the reproducibility
  checklist gives the FT-Q seeds.

**Trims for the page budget** (the guide and the three findings tables keep every row): the FT-C/FT-Q definitions live in Section 5
only (6.2 points there; the tie rule of the teacher is left to addendum 8); limitations without the item-mean sentence (stated in
Section 4 and deviation row 16); shorter captions of tab:rq (no estimator pointer), tab:tracks (the popularity-link definition is in
Section 5), tab:teaches, tab:deviations, tab:corrections ("after GATE_PASS" dropped), tab:app-llama and tab:app-z2; shorter tab:rq
cells (RQ4; RQ5 "every Qwen model ... is POSITIVE and Llama on Toys agrees"; RQ6 "arms match classes, not items (diagnostics; after a
win, an item-stratified arm)"); the wording-flag sentence of 6.1 condensed; deviation rows 1, 6, 9, 10, 17, 19 and 20 condensed; the
amendments paragraph condensed (the content of A1 and A2 stays in Section 4); the knockout paragraph condensed (the logit-scale
clause is in the limitations); the reproducibility checklist without "every number ... committed result file" (stated at the head of
Section 6).

**fill_paper.py** (sha1 d4948d21be28deefa22402bb42ee583c0bc8582f). The ext section replaces the placeholder (`EXT_DETAIL` removed):
per-regime cells, the matched-mean span, item shares, G_CF,wu, P1_wu, the deployable rows, FT-C (ML-1M) and FT-Q cells, and seven
prose handlers (`c_ext_ej`, `c_ext_robust`, `p_ext_p1_reading`, `c_ext_ft_wording`, `c_ext_ef`, `c_ext_eh("ZS")`, `c_ext_eh("FT")`).
Italics come from the file's `status.items_2_to_7`; a block the file marks unavailable stays red with the file's reason; counts are
summarize's confirmations only, with the sign of each confirmed H-J member as found and recorded cuts (`not_run`) named; the wording
is never decided on an incomplete `ft_wording`. New consistency check: every summarize input present locally has the sha1 summarize
recorded (prose counts and table cells from one build). No existing decision function changed.

**Tests** (41, all pass, about 1.5 minutes): the ext slots stay red with the file they need while no extra file exists (157 slots);
the real ML-1M file fills the ML-1M columns (italics), the P1 reading, and keeps the FT-C cell red with the file's own reason;
synthetic files of the real schema (copies of the real file, summarized by `ftgrid_extra.summarize`) fill the Amazon, Llama, FT-C and
FT-Q cells and every count and wording branch (H-F 1 of 3; H-S per regime with the min-n note; H-J with both signs; robust and
estimator-dependent readings; mixed, item, not-supported, incomplete and not-requested wording; a recorded cut; an empty and a missing
regime; FAILED_INTEGRITY, incomplete_regime, R not defined; the sha1 check); the deviations table keeps the record's 20 rows; the
pull script pulls every extra file to the path the fill reads.

**Open for the main session.** (1) Record the sha1 of the edited `fill_paper.py` (above) in PILOT_LOG before any further result file
is pulled (A3-6 section 9.3). (2) The real `extra/ml1m.json` was built by `ftgrid_extra.py` 70684526; the module is now 8d0e5d86 and
`summarize` refuses a file built by another version, so ML-1M must be rebuilt by the final module, with `--pilot_log` once the A3-6
item 11 sha1s are recorded (its `FT_C_reading` reads "record missing" now, so the FT-C cell and the wording stay red). (3) The
agreement of the FT-Q teacher with the real labels (66-72% in this pass's brief) is in no registered document or result file, so the
paper does not state it; record it (e.g. from the teacher panel's manifest) if it should be printed. (4) A cut FT-Q dataset has no
file: its cells stay red (`result_file_missing`) until the main session decides how a cut FT-Q run is marked. (5) The TeX sources,
`fill_paper.py` and the test file are LF: a Python `write_text` on Windows writes CRLF and breaks the unfilled-report test
(multi-line slot texts); it happened once in this pass and was reverted. (6) The pass-1 paragraph of PAPER_DATA_MAP.md section 5
carried five control characters from a PowerShell backtick escape; repaired.

# Integration notes (text integrator, 2026-10-04)

Scope: alignment with the binding texts, page budget, consistency audit, citations, compile check. No result slot was filled and
no number was invented. All compiling was done in a private copy; only the nine section files, `references.bib` and
`CITATION_MAP.md` were written to this directory (main.tex untouched). Sections 1-9 below describe that pass; where they differ from
section 0, section 0 is current (for example the page counts of section 1 and the gate table of section 3).

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
