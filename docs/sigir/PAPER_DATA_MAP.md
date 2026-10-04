# Paper data map: result files -> the slots of Paper/sigir2027

Every number in the paper comes from a result file under `docs/sigir/results/`. Nothing is typed by hand. This page lists,
for every alias and every table, which script writes the file and which JSON field fills which cell. It also lists the
results that do not exist yet, so the GPU queue can be planned from it. The state below is from 2026-10-04 (files pulled
that day). The live state is always `Paper/sigir2027/filled/UNFILLED.json`.

## 1. Pipeline

| step | command | writes |
|---|---|---|
| pull | `powershell -NoProfile -File scripts\sigir\pull_results.ps1 [-ListOnly]` | `docs/sigir/results/<alias>/...`, `docs/sigir/results/MANIFEST.json` (path, size, sha1, server mtime; missing files listed) |
| fill | `python scripts\sigir\fill_paper.py [--check_equal]` | `Paper/sigir2027/filled/{main.tex, references.bib, sections/*.tex, UNFILLED.json, FILLED.json, CHECK_EQUAL.json}` |
| compile | in `Paper/sigir2027/filled/`: pdflatex, bibtex, pdflatex x2 | `filled/main.pdf` (unfilled slots stay red) |
| test | `python -m pytest tests\test_confrec_fillpaper.py -q` | 30 tests, CPU, about 1 minute |

- The pull is read-only on the server: one `stat` listing through `scripts\sigir\remote.ps1 -ScriptFile`, then one `scp` per
  file over the `lumen-gpu` key (BatchMode). No credential is written anywhere.
- The fill never writes into the skeleton. The same inputs give byte-identical outputs, and a file is rewritten only when its
  bytes change.
- `UNFILLED.json` has one entry per red slot: file, line, slot text, table, row, column, reason and detail. It also carries
  the gate decisions, the consistency checks and the sha1 of every result file read.
- `FILLED.json` has one entry per filled slot, with its text and the result-file fields it was read from (the audit trail).
  Its "notes" record values that are deliberately not printed. At present that is Gate-FT's own zero-shot context (section 5).
- `CHECK_EQUAL.json` (printed by `--check_equal`) lists every quantity printed in more than one place, with each place, the
  printed text and the source field.
  - A quantity is a result field. gate_ft.json fields that duplicate a Qwen ML-1M grid field, and a reference taken from
    either regime's identical rows, count as one quantity.
  - `estimates_equal = false` is an inconsistency of the paper to resolve. `printed_equal = false` is only a format
    difference (prose or table, with or without interval).
  - A fill whose estimates differ is also announced on every run.
- Prose slots are matched by slot text plus the text right before the slot (the anchor), never by their position. Removing
  or adding a slot elsewhere therefore leaves every other prose slot valid. A changed anchor reads `skeleton_changed`.

Unfilled reasons:

| reason | meaning |
|---|---|
| `result_file_missing` | the result file does not exist yet |
| `result_not_in_report` | the file exists, but the report marks the block unavailable: an arm not run or not requested yet, or excluded. The detail quotes the report's own reason |
| `branch_slot` | a `branch:` alternative; never written by the script |
| `free_text` | a sentence or verdict to be written from the filled tables |
| `field_not_produced` | no script writes the field (section 5b) |
| `ambiguous_slot` | the sentence does not fix the panel, regime or field (section 5c) |
| `not_decided` | the registered test gives a mixed outcome, so no single direction word applies |
| `interpretive` | wording that no registered test decides |
| `incomplete_regime` | a seed or run is missing or FAILED_INTEGRITY; it is never replaced |
| `below_min_n` | an interval-based claim on fewer than 150 users |
| `branch_not_taken` | the block belongs to a gate outcome that did not occur |
| `skeleton_changed` | the slot text or row layout no longer matches `TABLE_SPECS` / `PROSE_SPECS` |
| `to_be_removed` | a slot the main session drops from the skeleton (decision of 2026-10-04). It leaves the list when the edit lands; it is 0 now |

Cell formats (FILL RULE 2):

| kind | rendering |
|---|---|
| estimate with interval | `0.612{\scriptsize$\pm$.014}` (estimate, half-width of the 95% CI, three decimals) |
| LoRA cell | `0.742{\scriptsize$\pm$.019\,(.002)}` (seed mean, its CI half-width, seed s.d. (ddof 1)) |
| no interval in the file | estimate only (sel UAUCs, Gate-FT zero-shot context, temperature, lemma checks) |
| tab:exposure Gini and APLT blocks, tab:anatomy block C | estimate only (FILL RULE 2) |
| fewer than 150 users | value plus `{\scriptsize\,(descriptive)}` |
| registered cut recorded in the file | `not run` (ftprune arm `NOT_RUN`; ftmethod slot `KILLED` before the dataset) |
| seed excluded for E1 | `FAILED\_INTEGRITY` |
| labels and decisions | verbatim from the file, with underscores escaped |

## 2. Aliases

| alias | producer (script) | server path (under `/root/autodl-tmp/lumen-rec`) | local path (`docs/sigir/results/`) |
|---|---|---|---|
| sel | `src.confrec.gatefix_select dev` | `outputs/confrec/gatefix/dev/selection.json` | `sel/selection.json` |
| gate | `src.confrec.gatefix_select confirm` | `outputs/confrec/gatefix/confirm/gate.json` | `gate/gate.json` |
| gft | `src.confrec.gateft_eval` (run_gateft.sh) | `outputs/confrec/gateft/gate_ft.json` | `gft/gate_ft.json` |
| aud | `src.confrec.nextitem_audit run` / `summarize` | `outputs/confrec/nextitem_audit/{sports,toys,home,tools,summary}.json` | `aud/<d>.json`, `aud/summary.json` |
| aud2q, aud2l | `nextitem_audit restrict` (Qwen) + `run --segments single` (both backbones) | PROPOSED `outputs/confrec/nextitem_audit_z2/{qwen,llama}_<d>.json` | `aud2q/<d>.json`, `aud2l/<d>.json` |
| grid | `src.confrec.ftgrid_report` (run_ftgrid.sh stage 6) | `outputs/confrec/ftgrid/report/<d>.json` (+ `_tables.csv`); Llama `outputs/confrec/ftgrid_llama/report/<d>.json` | `grid/qwen/<d>.json`, `grid/llama/<d>.json` |
| grid (n) | `src.confrec.ftgrid_data` (run_ftgrid.sh stage 0) | `outputs/confrec/ftgrid[_llama]/panels/<d>/ftgrid_split.json` | `grid/{qwen,llama}/<d>_split.json` |
| cpu | fields of the grid report: `E_A.<regime>.UAUC_TEST.references.{q_hat,popularity,mf}` and `E_A.<regime>.mf_personal_residual_warm_pairs_only` | (the report) | (the report) |
| ko | field of the grid report: `knockout.{labels,analyses}` | (the report) | (the report) |
| corr | rated: grid report `E_A.ZS.invariance`, `E_D.ZS.corrections`; Sports: no producer | PROPOSED `outputs/confrec/ftgrid/corrections_sports.json` | `corr/sports.json` (listed as missing) |
| prn | `src.confrec.ftprune analyze` (run_ftprune.sh stage E) | `outputs/confrec/ftprune/pruning_ml1m.json` (+ `.csv`) | `prn/pruning_ml1m.json` |
| slot | `src.confrec.ftmethod_report dataset` / `slot` (run_ftmethod.sh stage 5) | `outputs/confrec/ftmethod/<d>/report.json`, `outputs/confrec/ftmethod/slot.json` | `slot/<d>.json`, `slot/slot.json` |
| mir | `scripts/sigir/pilot1_gate.py --stage3_gate` (run_gatefix_stage3.sh) | `outputs/confrec/gatefix/stage3/decision.json` | `mir/decision.json` |

Where the skeleton's PROPOSED path differs from the script, the alias follows the script (FILL RULE: "the alias follows the
script"):

- `prn` is written to `outputs/confrec/ftprune/`, not `ftgrid/`.
- `cpu` comes from the grid report's references, not from a `refs_<d>.json`. run_ftgrid.sh passes `--raw`, so the refs cache
  holds only per-pair arrays and no UAUC.
- `corr` (rated panels) comes from the grid report.

The Z2 files must be written to the PROPOSED `nextitem_audit_z2` paths, or `pull_results.ps1` and the alias must be changed
together.

## 3. Tables

Columns: `ZS` = regime zero-shot (model `zeroshot`), `LoRA` = regime FT (seeds s0-s2). `S1k` = sports segment
`events_1_1000`, `S10k` = `events_1001_10000`. Toys, Home and Tools use segment `all`. Next-item cells use question `next`
unless the row says `like`.

**tab:gate-outcomes**

| row | V0 / zero-shot | V* / LoRA | Outcome |
|---|---|---|---|
| G5 dev ML-1M | `sel.table.V0.ml1m.UAUC` | `sel.table[v_star].ml1m.UAUC` | `sel.decision` |
| G5 dev Toys | `sel.table.V0.toys.UAUC` | `sel.table[v_star].toys.UAUC` | `sel.table[v_star].E2` |
| G6 confirm | `gate.v0_context.{UAUC,ci95}` | `gate.{UAUC,ci95}` | `gate.decision` |
| LoRA seeds | | `gft.UAUC_post_T_per_seed` | |
| mean over seeds | grid ml1m `E_A.ZS.UAUC_TEST.per_model.zeroshot` (decision 1) | `gft.UAUC_post_T_mean_over_seeds`, `UAUC_post_T_seed_averaged_ci95`, `UAUC_post_T_sd_over_seeds` | `gft.decision` |
| paired delta | | grid ml1m `E_B.mean_over_seeds` with the seed s.d. (decision 1) | |
| item mean, MF | spans both columns: grid ml1m `E_A.FT` (else ZS) `references.q_hat / .mf` | | |

The caption slot is `gate.n_users`.

Decision 1 (main session, 2026-10-04): the zero-shot UAUC and the paired delta print the grid values of tab:reliability, so
both tables print 0.596 and 0.147. The slot texts `gft:...` and `grid:UAUC` / `grid:dUAUC_ft_minus_zs` are both accepted.
gate_ft.json's own `zero_shot_context` (0.595996, delta 0.14647) is a second scoring of the same prompts and is not printed.
FILLED.json "notes" records both values, read from the files, with their absolute differences.

**tab:anatomy.** Columns S1k, S10k, Toys, Home and Tools all read `aud/<d>.json` `segments.<seg>.questions.next`:

| row | field |
|---|---|
| T | top-level `temperature.next.T` |
| ECE | `C_calibration.list_normalised.ece` |
| Brier | `C_calibration.list_normalised.brier` |
| acc_low | `C_calibration.error_anatomy.unsure_correct_rate` |
| acc_high | `C_calibration.error_anatomy.tertiles[2].accuracy` |
| errors in high | `C_calibration.error_anatomy.share_errors_in_top_tertile` |
| AUROC(p_max) | `C_calibration.auroc_discrimination.top1.auroc.p_max` |
| AUROC(l) pooled | `C_calibration.pointwise.pooled_auroc` |
| UAUC | `C_calibration.pointwise.uauc` |
| NDCG@10 | `A_ranking.ranking.ndcg10` |
| block B | `E_popularity.{head_minus_tail_mean_p, bias_index.head_minus_tail, top1_head_fraction}` |
| block C (estimates) | `D_selective_serving.signals.{p_max,random}.{gain_at_50_vs_full.ndcg10, niche.niche_minus_mainstream_served_share}` |

**tab:exposure.** The same five columns in each of three blocks (Delta_head, Gini, APLT):

| row | field |
|---|---|
| Pool | `B_exposure.llm.pool_head_share` (Delta_head block), `pool_tail_share` (APLT block) |
| Target | `B_exposure.llm.target_head_share` |
| LLM next / like | `questions.{next,like}.B_exposure.llm.{delta_head, gini_exposure, tail_share_top10}` |
| baselines | `segments.<seg>.reference.<method>.B_exposure.*` |

Baseline methods: ccrp_v3, elmrec_graph, irllrec_intent, llm2rec_sasrec, llmemb, llmesr_sasrec, proex_profile,
promax_profile, rlmrec_graphcl.

**tab:reliability (Qwen) and tab:app-llama (Llama).** File `grid/<bb>/<d>.json`. A ZS cell is `per_model.zeroshot`. A LoRA
cell is the seed mean plus the s.d. of the three `per_model.s0-s2` estimates.

| row | field |
|---|---|
| UAUC | `E_A.<reg>.UAUC_TEST` |
| LoRA - ZS (span) | `E_B.mean_over_seeds` (s.d. of `E_B.per_seed`) |
| item mean / popularity / MF (span) | `E_A.ZS.UAUC_TEST.references.{q_hat,popularity,mf}` |
| ECE / Platt slope | `E_C.<reg>.{ECE, platt_slope}` |
| correct in bottom / errors in top | `E_C.<reg>.{share_correct_bottom, share_errors_top}` |
| margin AUROC / AURC | `E_C.<reg>.{AUROC_margin_correct, AURC}` |
| Users (span) | `ftgrid_split.eval.users_both_classes_test` |
| UAUC of l / pi / e on S_d | `E_D.<reg>.UAUC.{L,pi,e_hat}` |
| r8 / non-prior share | `E_D.<reg>.shares.{r8c,non_prior_share}`; "uninterpretable" when `shares_reading` says so |
| G | `E_D.<reg>.information_gain.per_model.<m>.G`, `G_mean_over_seeds` |
| G_CF | `E_D.<reg>.information_gain.G_CF` (no seed s.d.) |
| star permutation | `E_D.<reg>.star_permutation.per_model.<m>.dUAUC_L_minus_perm`, `dUAUC_mean_over_seeds` |
| popularity link of pi / l | `E_E.<reg>.partial_spearman.{pi_item, L_item}` |
| P1 (span, ML-1M) | `P1.per_seed.s0-s2`, `P1.mean_over_seeds` (est, CI, p) |
| Users in S_d (span) | `ftgrid_split.sd.users` |
| bottom reference rows | `E_A.ZS...references.{q_hat,popularity}`, `E_A.ZS.mf_personal_residual_warm_pairs_only` |

**tab:popularity.** File `grid/qwen/{toys,games}.json`:

| row | field |
|---|---|
| head-tail drop | `knockout.analyses.<m>.delta.pseudo.head_minus_tail` |
| placebo | `knockout.analyses.<m>.delta.placebo.head_minus_tail` |
| tail / overall dUAUC | `knockout.analyses.<m>.dUAUC.real_minus_pseudo.{tail,all}` |
| label | `knockout.labels.<m>.label` (LoRA cell: "s0 / s1 / s2") |

A LoRA cell is the mean of the three seed estimates with their s.d. No interval exists for that mean.

**tab:pruning.** File `prn/pruning_ml1m.json`:

| column | field |
|---|---|
| UAUC | `arms.<P>.{UAUC_seed_averaged, sd_seed}` |
| Delta vs P1 | `contrasts.<P>-P1` (P0: the exact negation of `P1-P0`) |
| >0 | count of positive `per_seed` values of that contrast |
| Delta vs P0 | `contrasts.<P>-P0` (P3-P0 is in ftprune.CONTRASTS since 2026-10-04) |
| ref row | grid ml1m `E_A.FT.references` |

`arms.P3.status = NOT_RUN` makes the P3 row read "not run".

**tab:app-sens.** Each variant row reads `sel.table[V]`:

| column | field |
|---|---|
| ML-1M, Toys | `ml1m.UAUC`, `toys.UAUC` |
| E1 | `ml1m.E1 and toys.E1` |
| E2, Eligible, Tied | `E2`, `eligible`, `tied_with_max` |

The DEV-user item-mean reference row was dropped from the skeleton (no script produces it).

**tab:corrections.** Caption and the 2nd-rated column use the domain directory of `mir.inputs.toys`. The ML-1M and 2nd-rated
columns read the grid zero-shot report:

| row | field |
|---|---|
| 0 | `E_A.ZS` UAUC / `q_hat` / `mf` |
| 1-2 | `E_A.ZS.invariance.per_model.zeroshot.{max_abs_dAUC_user, dAUC_pooled_offset_removal}` (the Sports cell of row 1 is "--") |
| 3-4 | `E_D.ZS.corrections.{dUAUC_nohist_minus_raw, dUAUC_evidence_minus_raw}.per_seed.zeroshot` |
| 5-6 | `mir.criteria.{ml1m,toys}.{dUAUC_mirror_minus_placebo, dUAUC_mirror_minus_ensemble_null}` |
| 7 | `mir.next_item_no_loss["dNDCG@10_mirror_minus_raw"].tie_exact` |

**tab:slot**

| row | field |
|---|---|
| SFT | grid `E_A.FT.UAUC_TEST` |
| stacking / offset | `slot/<d>.json slot.{post_hoc_stacking, prior_offset_LoRA}` |
| difference | `slot.difference.mean_over_seeds` plus the seed range |
| refs | grid `E_A.FT.references` |

"not run" applies when `slot.json` is `KILLED` and the dataset was never run.

**tab:app-seeds.** File grid qwen:

| row | field |
|---|---|
| domain rows | `E_A.FT.UAUC_TEST.per_model.s<k>`, `E_A.FT.secondary.UAUC_all_rows.per_model.s<k>`, ref `q_hat` |
| FT-C rows | `FT_C.seed<k>.{TEST,all_rows}.per_seed.s<k>` (permuted UAUC = `UAUC_b`; contrast = the record) |

A seed excluded for E1 reads FAILED_INTEGRITY.

**tab:app-z2.** Each cell is "aud2q / aud2l": the single segment of each file, question `next`. Rows use the same fields as
tab:anatomy and tab:exposure.

**tab:guide** is free text throughout: clauses and the admission verdicts.

## 4. Prose slots (experiments.tex, `PROSE_SPECS`)

Cross-domain sentences get per-domain lists in the registered order: next-item S10k, Toys, Home, Tools (the four family
units); rated ML-1M, Toys, Video Games, Sports. Direction words are filled only by the registered test.

| slot | rule |
|---|---|
| RQ1 `k of 4`, direction, consequence | `aud/summary.json` `S1.acc_top_minus_bottom_tertile.next.holm` (Holm over the four domains). The consequence is written only when all four agree |
| RQ2 more/as/less often | `S2.share_errors_in_top_minus_third.next.holm`, all four agreeing |
| RQ3 direction, `k of 4` | `S3.admission.llm_next` (effect_claimed and sign, or no domain excluding 0) |
| RQ3 second backbone | sign of aud2q vs aud2l `delta_head`, the same or opposite in at least 3 of 4 domains |
| RQ3 vs baselines | larger, smaller or similar only when all 9 x 4 Holm contrasts agree |
| RQ4 zero-shot UAUC, item mean, LoRA UAUC (as rewritten 2026-10-04) | per-panel lists of `E_A.ZS`, the `q_hat` reference on the same rows, and `E_A.FT`, each with its own interval; no direction word, because no paired test is registered |
| RQ4 G above / not distinguishable | `E_D.<reg>.holm_family_E_D.confirmed.G` in every domain and both regimes |
| RQ4 raises / leaves / lowers | E-B Holm over the four Qwen domains, computed from the per-domain raw p, plus `E_B.seeds.sigma_seed_rule` |
| RQ4 P1 holds / n of 3 | `P1.decision.verdict` and the per-seed signs |
| RQ5 knockout label | written only if all Qwen labels (Toys, Video Games x zeroshot, s0-s2) are identical |
| RQ6 | `prn.claim.label`: BEATS_RANDOM, WORSE_THAN_RANDOM, ABOUT_EQUAL, INCONCLUSIVE |
| slot | `slot.json holm.confirmed` and the sign of dUAUC |

## 5. State at the last pull (2026-10-04, about 17:00 UTC) and what is still needed

Pulled:

- `sel`, `gate`, `gft`;
- `aud/sports.json`;
- the six `ftgrid_split.json`;
- the Qwen ML-1M grid report `grid/qwen/ml1m.json`, written 16:50 UTC: ZS and FT complete, no PERM models yet.

Decisions:

- G5 FIX_FOUND (V* = V1);
- G6 GATE_PASS (UAUC 0.603, CI 0.594-0.612);
- G9 GATE_FT_PASS (mean 0.742);
- P1 (ML-1M) P1_HOLDS.

So the GATEFT and GATEPASS blocks stay, and the fine-tuned and stage-3 items are due.

The skeleton was edited by the main session (2026-10-04). Three removals:

- the DEV item-mean row of tab:app-sens;
- the Sports `max_abs_dNDCG_event` cell;
- the RQ4 paired-difference clause, replaced by three estimates (zero-shot UAUC, item mean on the same rows, LoRA UAUC).

The paper now has **773 slots**.

**Filled: 231 of 773**

Counted by the alias in the slot text, as the script prints them:

| alias | slots | where |
|---|---|---|
| sel | 48 | tab:app-sens, G5 rows |
| gate | 4 | G6 row, caption |
| gft | 6 | Gate-FT seeds, mean and decision, prose mean; the two `gft:`-named Gate-FT cells now print grid values (decision 1) |
| aud | 106 | sports S1k/S10k columns of tab:anatomy and tab:exposure |
| grid | 51 | ML-1M columns of tab:reliability; Users / Users in S_d; ML-1M seeds of tab:app-seeds; SFT of tab:slot |
| cpu | 10 | ML-1M reference rows, the Gate-FT reference row, the pruning reference row |
| corr | 4 | ML-1M column of tab:corrections, rows 1-4 |
| choice | 2 | "P1 holds", "3 of 3" |

**Unfilled: 542**

| reason | slots |
|---|---|
| `result_file_missing` | 471 |
| `result_not_in_report` | 8 |
| `free_text` | 33 |
| `branch_slot` | 14 |
| `ambiguous_slot` | 14 |
| `interpretive` | 2 |
| `field_not_produced`, `to_be_removed`, `skeleton_changed` | 0 |

**Duplicates (`CHECK_EQUAL.json`).**

- 16 quantities are printed more than once, 14 of them in more than one table.
- Estimates differ for 0.
- 2 differ only in format (a prose value beside its table cell).
- The 0.147 / 0.146 discrepancy found earlier is resolved by decision 1. Both tables print the grid E-B and zero-shot
  values.
- gate_ft.json's own zero-shot context (0.595996, delta 0.14647; |difference| about 1e-4; the same V1 prompts and users, a
  second scoring) is recorded in FILLED.json "notes" and is not printed.

**5a. Waiting for result files (471 slots), plus 8 slots waiting for runs inside an existing report.** Counted by the first
missing file a slot meets:

| file(s) | slots | job |
|---|---|---|
| `grid/qwen/{toys,games,sports}.json` | 71 + 58 + 48 | run_ftgrid.sh D (stages 3-6) |
| `aud/{toys,home,tools}.json` | 59 + 53 + 53 | `nextitem_audit run` per domain (scoring in progress) |
| `aud/summary.json` | 7 (RQ1-RQ3 direction words) | `nextitem_audit summarize` over the four domain files |
| `grid/llama/{ml1m,toys}.json` | 21 + 20 | run_ftgrid.sh with OUT_ROOT=outputs/confrec/ftgrid_llama |
| `aud2q/*`, `aud2l/*` | 33 + 1 | run_llama_nextitem.sh, then `restrict` (Qwen) and `run --segments single` (both), into `nextitem_audit_z2` |
| `prn/pruning_ml1m.json` | 18 | run_ftprune.sh (needs FREEZE prune) |
| `slot/slot.json` + `slot/<d>.json` | 15 | run_ftmethod.sh (needs FREEZE method) |
| `mir/decision.json` | 14 | run_gatefix_stage3.sh (due: G6 = GATE_PASS) |
| inside `grid/qwen/ml1m.json` (`result_not_in_report`) | 8 | the FT-C rows of tab:app-seeds: the permuted adapters p0 and p1 (run_ftgrid.sh ml1m with the models zeroshot,s0,s1,s2,p0,p1, then the report again) |

**5b. Fields no script writes: none now.**

- The three cases found on 2026-10-04 were dropped from the skeleton: the DEV item-mean row, the Sports
  `max_abs_dNDCG_event` cell, and the RQ4 paired difference with its two direction words and `k of 4`.
- The fourth case, tab:pruning P3 "Delta vs P0", is now produced: P3-P0 was added to ftprune.CONTRASTS.
- Should such a slot reappear, it reads `to_be_removed` (if listed as dropped) or `field_not_produced`.

**5c. Ambiguous prose slots (14).** Mechanical once the skeleton names the panel or regime:

- RQ2: the rated share of errors in the top tertile, "detects / does not detect" and the margin AUROC. The regime is not
  stated.
- RQ3:
  - `aud:B.head_share_top10_llm_minus_ref` is one slot for 36 contrasts.
  - `aud:D.gain_at_50_vs_full` and `aud:D.niche_minus_mainstream_served_share`: "relative to the random signal" names no
    field (p_max value, random value, or their difference, which is not produced).
- RQ4: `grid:G`. The regime is not stated.
- RQ5:
  - The direction word and `grid:partial_rho_pi_logpop`: the regime is not stated.
  - `ko:head_minus_tail.delta` and `ko:placebo.delta`: domain and regime.
  - Llama on Toys `ko:label, drop with the same / the opposite sign`, and Sports `ko:label or not run`: per-model labels and a
    recorded cut.
- Corrections: "each item-dependent correction is above/indistinguishable/below its control" covers several corrections.

Interpretive (2): RQ2 "mostly low-confidence / ...", RQ5 "the same / opposite / unrelated".

**5d. Written by hand from the filled tables.**

- 33 free-text slots: abstract 2, introduction 3, observation 2, method 1, experiments 18 (tab:guide), conclusion 5,
  appendix 2.
- 14 `branch:` slots: never touched by the script. `UNFILLED.json` "decisions" gives the gate outcomes that select the branch.

## 6. Choices made by the fill (each is one line in `fill_paper.py`)

Confirmed by the main session on 2026-10-04:

- LoRA cells: estimate ± CI half-width with the seed s.d. in parentheses.
- Direction words only on unanimous registered tests.
- `grid:partial_rho_L_logpop` uses the item-level partial Spearman (`L_item`), for comparability with the pi-hat row, which
  exists only at item level. The pair level is `E_E.<reg>.partial_spearman.L_pair`.
- ML-1M zero-shot values come from the grid everywhere (decision 1).

Not yet reviewed:
- cpu references come from the zero-shot regime's rows in the spanning cells, and from the FT rows in LoRA-only tables
  (gate-outcomes, pruning, slot, app-seeds). The two coincide unless a run has censored rows.
- `grid:n_users` and `grid:n_sd` come from `ftgrid_split.json` (named by the skeleton for n_sd). A check compares them with
  the report's `meta` once the report exists.
- The MIRROR next-item loss is the tie-exact dNDCG@10, the metric pilot1_gate decides on.
- In prose, next-item lists use the four family units only. S1k (the quarantine) appears in the tables.
- tab:app-z2 cells carry intervals: FILL RULE 2 exempts only the tab:exposure Gini/APLT blocks and tab:anatomy block C.

Consistency checks (`UNFILLED.json` "checks"; printed when they fail):

- grid ml1m FT seeds = gft `UAUC_post_T_per_seed`;
- report `meta.n_users_both_classes_test` = split `eval.users_both_classes_test`.

The broader duplicate check is `CHECK_EQUAL.json` (section 1). The grid E-B vs gate_ft.json zero-shot-context comparison is
no longer a check: decision 1 prints only the grid value, and the gate_ft.json values are recorded in FILLED.json "notes".
