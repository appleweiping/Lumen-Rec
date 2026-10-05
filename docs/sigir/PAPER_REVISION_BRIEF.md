# Paper revision brief (editor pass 1, 2026-10-05)

Owner of this pass: one editor agent, who owns `Paper/sigir2027/sections/*.tex`, `Paper/sigir2027/main.tex`, `Paper/sigir2027/references.bib`,
the specs of `scripts/sigir/fill_paper.py` (TABLE_SPECS, PROSE_SPECS and their handlers), `tests/test_confrec_fillpaper.py`,
`docs/sigir/PAPER_DATA_MAP.md` and `Paper/sigir2027/INTEGRATION_NOTES.md`. Nobody else edits these files during the pass.

## 1. Goal
After the revision the pre-reference part of the paper, appendix included, fits **9.0 pages with every slot filled by a result of
typical size** (target 8.9; SIGIR full papers: 9 pages including appendices, references excluded), and the paper tells this story:
(i) what the token confidence of a yes/no recommender encodes (item prior versus a pair-specific residual; personal information
measured by prediction, beside an item mean and matrix factorisation on the same pairs); (ii) what fine-tuning teaches (within-item
label-permutation control, store-name knockout, sparse/unseen strata, complementarity to collaborative filtering); (iii) which uses of
the confidence survive a matched control (corrections, selective serving, static exposure, pruning); (iv) the registered protocol and its
deviations. Read first: `docs/sigir/REVIEW_NOTES_2026-10-05.md` (two same-family reviews, decisions), `idea-stage/PREREG_AMENDMENT_3_ADDENDUM_6.md`
(binding: wording, admission, new endpoints), `docs/sigir/DEVIATIONS.md` (source of the appendix table), `docs/sigir/PAPER_DATA_MAP.md`
(slot pipeline), `docs/sigir/FTEXTRA_IMPL_SPEC.md` (the result file the new endpoints will come from), `Paper/sigir2027/INTEGRATION_NOTES.md`.
The abstract, the introduction (body and contributions), the decomposition paragraph of `preliminaries.tex` and the gate sentences were
already rewritten on 2026-10-04: keep their claims (adjust only for length and consistency).

## 2. Hard rules
1. **No number is typed by hand.** Every result is a `\DATANEEDED{alias:field}` slot filled by `scripts/sigir/fill_paper.py` from the committed
   result files (`docs/sigir/results/<alias>/...`). New or changed slots need entries in TABLE_SPECS / PROSE_SPECS with handlers and tests;
   `python scripts/sigir/fill_paper.py` must report no `skeleton_changed` and `python -m pytest tests/test_confrec_fillpaper.py -q -p no:cacheprovider`
   must pass (31 tests at the start; the suite takes about 4 minutes; keep it green and extend it for what you add). Numbers that are part of the
   registered design (bars 0.60, 0.65; sample sizes; thresholds) and the three pilot outcomes already in the text (0.587, 0.2093, 0.1863,
   0.469, 0.58) may stay as text because they are constants of registered documents; add no other literal result.
2. **Wording rules are frozen** (addendum 6 section 9, `fill_paper.py` decision functions recorded in PILOT_LOG): direction words only from
   registered tests and Holm families; mixed outcomes are counts per regime; replication = same sign with its own interval excluding 0;
   "personal evidence" is reserved for what G measures; e is the "pair-specific residual"; the non-prior share is never an upper bound; "familiarity" only
   under the knockout wording rule (otherwise "popularity-linked"); the knockout is scoped to store-name strings. You may add specs for new result
   files; you may not change an existing decision function (if a rule seems wrong, report it).
3. **Anonymity** (double blind): no "Lumen", no repository, host or path names, no names of the authors' earlier systems or papers (the next-item
   panels come from "an earlier benchmark protocol" cited as `\citep{anon2026benchmark}`, an `@misc` entry with author "Anonymous" and title "Omitted for
   double-blind review"; the verbalised same-candidate reranker of that protocol is "a verbalised reranker", never "C-CRP": rename the row labels in
   the tables AND in `REF_METHODS` of `fill_paper.py` consistently); strip such strings from TeX comments too; no acknowledgements.
4. **No new claim without a registered test.** Branch slots (`branch:` and the METHOD-SLOT blocks) stay as they are unless a gate outcome that has
   occurred decides them (GATE_PASS and GATE_FT_PASS are already resolved). British spelling. No emoji, no exclamation marks.
5. **Citations**: every cited key must exist in `references.bib` with verified fields. The citation-verification agent's output
   (`new_refs.bib`, `new_refs_notes.md` in the session scratchpad: `D:\_Organized\Temp-Review\_RootDirs\temp\claude\D--\22c9b1b2-12a5-4956-a490-79bcb9beb0d8\scratchpad\`)
   may not exist yet when you start; the introduction already uses the keys `lin2024rella`, `tan2025perrecbench`, `kim2025lostinsequence`,
   `zhang2024cft` and `anon2026benchmark`. Add `anon2026benchmark` yourself; when `new_refs.bib` appears, merge the entries (rename keys to those above where they
   correspond, fix any that the notes mark as discrepant), cite each new work in the section the notes name, and report the final list. Never invent a field.
6. Do not run git in `D:\Research\Lumen`; do not touch `idea-stage/*` or any bound code file; do not start GPU work; the server is not needed. Keep the
   working directory `D:\Research\Lumen` (subshells for any cd).

## 3. Page budget and the measuring tool
`python scripts/sigir/page_budget.py --results <results dir> --out <dir>` fills the skeleton, gives every slot that is still red a typical filler
(table cell `0.123 +- .012`, prose slots 30 words, branch slots their longest alternative), compiles and prints the pages used before the references. A
complete synthetic result set with the real schemas is at
`D:\_Organized\Temp-Review\_RootDirs\temp\claude\D--\22c9b1b2-12a5-4956-a490-79bcb9beb0d8\scratchpad\synres` (the extra-analysis files of addendum 6 are not in it: the
tool fills unknown slots with the filler). **Measured now: 9.24 pages**, before any of the additions below, so the net cut is about 1.3 pages.

Allocation to aim at (pages): abstract and title block 0.35; introduction 1.0; related work 0.45; setting 0.55; protocol with tab:rq 0.9; tools 0.65;
findings 3.1; conclusion with limitations 0.4; appendix 1.5 (sum 8.9). Cut order proposed (change it if you can show better): (1) the variant bank table
and the prompt-sensitivity table of the appendix become two sentences and a pointer to the artefact; (2) the eight-baseline rows of the exposure table
shrink to the LLM, the pool, and the min / median / max of the eight published recommenders (new specs); (3) the gates become one paragraph plus a
small table (or only the paragraph); (4) the protocol's gate and backbone paragraphs lose their repeated constants (keep the bars); (5) the second-backbone
next-item appendix table keeps four rows; (6) Lemma and Proposition stay but shrink to one display each; (7) the nested method slot stays as one
compact table only if it survives, else one sentence (blocks `METHOD-SLOT-BEGIN/END`). Keep the one-table guide (tab:guide), it is the paper's deliverable
for the six practitioner questions, but it is not a contribution. Floats must keep the specifier `p` (see FILL RULE 6 in experiments.tex).

## 4. What to add (all registered in addendum 6; the result file is `outputs/confrec/ftgrid/extra/<d>.json` and `extra/summary.json`, pulled to
`docs/sigir/results/extra/`, alias `ext`; its schema is in `docs/sigir/FTEXTRA_IMPL_SPEC.md` and, once written, `src/confrec/ftgrid_extra.py`)
1. **Findings 6.x "What the confidence tracks"** (one table, Qwen3-8B, zero-shot and LoRA columns, ML-1M then the three Amazon panels, Llama in the appendix): UAUC of
   the LLM beside the item mean, the information-matched item mean and MF on the same rows; E-J contrast with the item mean (descriptive except H-J); item-prior share,
   non-prior share, e-share; G (registered, pooled) and G_wu beside G_CF; P1 and P1_wu for ML-1M. Every LLM number has its same-row references next to it.
2. **Findings 6.y "What fine-tuning teaches"**: FT-C retention R and its label (ML-1M, Toys), the E-F complementarity (G_{LLM|CF}, G_{CF|LLM}; H-F), the E-H strata
   (sparse versus dense, unseen versus seen; H-S), the store-name knockout (existing table, shortened) with the wording rule. ML-1M values carry the label
   "exploratory" (addendum 6), Amazon values are outcome-free; say so once in a table note.
3. **Protocol and tools**: one paragraph each on E-W (within-user estimator and repeated cross-fitting), the e-share, E-C' (deployable top-k), the new Holm families
   (E-F, E-H, E-J), the replication rule and the count rule. Keep tab:rq compact: RQ4 and RQ5 rows gain the new endpoints; shorten other cells.
4. **Appendix deviations table** (`tab:deviations`, label `app:deviations`, referenced from the introduction): the 18 rows of `docs/sigir/DEVIATIONS.md`, condensed
   to two or three short columns (what / evidence / effect). Do not drop a row; shorten wording.
5. **Related work**: tighten to 0.45 page; add the verified works (output-level studies of what LLM recommenders use: item quality and rating bias, history
   use; memorisation of ML-1M; pointwise yes/no scoring and its calibration in IR; denoising and pruning; performance prediction; popularity and position bias).
6. **Limitations** (conclusion): one compact paragraph: one LoRA recipe (r 16, 1 epoch) and 8B backbones; pointwise, sampled candidates, static exposure;
   store-name knockout only; q-hat advantaged by ratings in [T_d, t); registered thresholds of the knockout are unit-dependent; same-family reviews;
   self-hosted registration; ML-1M exploratory items.
7. Replace remaining uses of "personal evidence" as the name of e, "upper bound", "existing benchmark protocol", "C-CRP" and "Lumen"; make the abstract, the introduction's
   findings slots and the guide agree with the above.

## 5. Report back
Files changed; pages before and after (tool output); slots added, removed or renamed (with spec and test changes); the final citation list; anything in this brief you
could not do or disagree with; the open red-slot count by reason (`Paper/sigir2027/filled/UNFILLED.json` after `python scripts/sigir/fill_paper.py`). Do not delete the
`filled/` directory; regenerate it with the script.
