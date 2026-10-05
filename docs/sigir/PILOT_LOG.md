# SIGIR 2027 line — pilot log (append-only, newest first)

## 2026-10-05 · Code record: the longer-training arm FT-L (addendum 9, exploratory); recorded before any 3-epoch adapter was trained

- `scripts/sigir/run_ftlen.sh` = 439f0f1749925896bf1a0b9952fc230c17565a18
- `src/confrec/ftlen_panel.py` = 76b715a35cc379b43f6e0857293cb96474a82e98
- `tests/test_confrec_ftlen.py` = afdabd75b775e0b76846f657b5e1a3852f483ae6

Implemented by the implementer of the FT-Q control on the same machinery (canonical-root guards, DRY allow-list, link sweep, E1 rerun-once rule with the same-key refinement, sticky
FAILED_INTEGRITY with a logged override); `ftlen_panel.py` imports `src/confrec/ftq_panel.py` read-only (pinned by the FT-Q record). Trains s0 and s1 with the flags of the
real adapters' recorded configuration except `--epochs 3` (ML-1M: Gate-FT's `--train`, bytes equal to the grid's TRAIN panel), scores `like` and the swap arm, links the registered
zero-shot scores, and runs `ftgrid_report` (unchanged) with `--models zeroshot,s0,s1` into `outputs/confrec/ftgrid_len3/report/<d>.json` with a NOTE saying that these are two-seed,
3-epoch, exploratory adapters. 124 of 126 tests pass (2 skipped by design); mutation checks (19 planted bugs in the script, 18 in the module) were caught. Estimated GPU time
5.3 h (ML-1M) and 6.7 h (Toys). Exploratory-root rule: `run_ftextra.sh` refuses this root; `ftgrid_extra` is run by hand with `--root_label llama` (the safe label: summarize then refuses the file
and no family or FT-C reading can use it). Not implemented as written: the paired 3-epoch minus 1-epoch contrast on identical rows (the two reports read the same panels and split, so their E-B, E-D
and E-J numbers are compared side by side). One `git status` was run by the implementer in the working folder by mistake (it can only refresh the stat cache of `.git/index`); no other git command
was run there.

## 2026-10-05 · Code record: the S6 composition diagnostics (addendum 7 section 2); recorded before the S6 arms were built

- `src/confrec/ftprune_compose.py` = d0573fc4906f870cae13ef83a00c908600dc9a93
- `tests/test_confrec_ftprune_compose.py` = 4111cf3680d7f176db032cf3e3837254db3d370a

`ftprune_compose` computes, for P0, P1 (mean over seeds), P2 and P3 from the prune manifest and the files it names (no scoring, no outcome): (a) the share of TRAIN items
that keep no example, (b) the mean absolute change of the item label rate over items with at least two TRAIN examples (weights = full-TRAIN counts; an in-scope
item that loses all its examples is counted in (a) and listed separately), (c) the share of ML-1M TEST pairs whose item keeps a TRAIN example (P0 must reproduce the
split's recorded seen share), (d) the head-tercile example share (terciles over the full TRAIN set by `stats.rank_bins`, equal counts share a tercile). It refuses a
manifest whose recorded sha1 does not match (28 refusal cases tested). 65 tests pass (19 injected bugs all caught); it imports `ftprune.py` read-only
(ftprune.py d9dd0f10... at the time, recorded with the `prune` record) and writes only to its `--out_dir`. Run after stage B:
`python -m src.confrec.ftprune_compose --manifest outputs/confrec/ftprune/prune_manifest.json --out_dir outputs/confrec/ftprune/compose`. The item-stratified arm P1s is
not implemented (conditional on P2 - P1 reading BEATS, addendum 7 section 3).

## 2026-10-05 · Amendment 3 addendum 9 (longer-training robustness arm FT-L, EXPLORATORY; written after the Toys fine-tuned results were read)

`idea-stage/PREREG_AMENDMENT_3_ADDENDUM_9.md` sha1 = 04722f5ff9bfa393e32eb94bd900b9608b00a25b. First read of Toys results (2026-10-05, after the code and wording-rule hashes of addenda 6 and 8
were recorded): Qwen3-8B on Toys: UAUC zero-shot 0.527, LoRA 0.558 (E-B +0.031 [0.006, 0.056]), item mean 0.684 (information-matched 0.633), temporal MF 0.539;
dUAUC(L - q-hat) LoRA -0.126 [-0.153, -0.098]; G (pooled) -0.005, G_wu -0.003, G_{LLM|CF} -0.0002 (all intervals include 0); P1 replication NO_EVIDENCE (mean -0.003,
p 0.67); item-prior share zero-shot 0.30 (uninterpretable, r8 0.69), LoRA 0.66; item share of MF 0.13 and of the labels 0.08; zero-shot knockout INDETERMINATE.
Toys therefore does not replicate the ML-1M personal-gain pattern. FT-L (3 epochs, seeds 0 and 1, ML-1M and Toys, like and swap arms, root
`outputs/confrec/ftgrid_len3`) is added as an exploratory answer to the objection that the registered recipe under-trains the model; it has no hypothesis and no
role in claim admission. Code to be recorded before the first adapter is trained.

## 2026-10-05 · Code record: the FT-Q control (addendum 8); recorded before any control adapter was trained or scored

- `scripts/sigir/run_ftq.sh` = f8fa2e3a2d6e19a74494e526c8a004b2a7acfbba
- `src/confrec/ftq_panel.py` = dab8d1a59021443f24988e49342cc5d06b4aed14
- `tests/test_confrec_ftq.py` = 522781a770e454eeb84daed971451be1cac8819b

Teacher panels (`train_q.jsonl`, sha1 of the byte-stable build from the method slot's TRAIN q-hat; built independently twice, by the implementer and by the reviewer
with own code, byte-identical): ML-1M 43329ebebc8ea1878be2aac612bff81304fcaf24, Toys f0ef65d1a846c5706bf30365634e6c7359aeae88, Video_Games
ea7b88dba06f3438aa666e7b16421389e494b4b7, Sports c2ac412ac25eb04cd12465aa3558051e04075c19 (teacher agreement with the real TRAIN labels 71.9%, 68.5%, 66.0%, 65.7%;
beta 0.6601, 0.6887, 0.6601, 0.6914; k = 15,412, 10,337, 10,135, 10,851 of 23,348, 15,010, 15,353, 15,695 examples). Tie direction: finite q-hat descending, then the
fixed seed-0 key ascending (as `ftprune.removal_mask`); one example sits at each cut, reversing the tie order changes 0 labels on all four. The code imports
`src/confrec/ftprune.py` (`tie_key`, `train_examples`) and `src/confrec/ftgrid_extra.py` as they are (recorded separately). Written by an implementer agent; two rounds
of independent same-family review (no blocker; one major and ten minor findings fixed and replayed as unit tests; the guard against writing outside
`outputs/confrec/ftgrid_q` was attacked with 26 path spellings and link variants). Tests: 137 pass locally (FTQ_FAST: 133 plus 4 skipped chain tests); on the
server's Linux the reviewer ran the previous test file (133 tests) in 25 s. No control adapter, score or report exists; the Toys fine-tuned job's outputs were not
opened. The jobs are `run_ftq.sh ml1m` and `run_ftq.sh toys` (required) and `run_ftq.sh games` and `sports` (unless cut on 2026-10-29), queued after the
audit that is running.

## 2026-10-05 · Wording rules record, update after editor pass 2 (recorded before any Toys, Video_Games, Sports, Llama, FT-C or FT-Q result was pulled or opened)

`scripts/sigir/fill_paper.py` sha1 = d4948d21be28deefa22402bb42ee583c0bc8582f, `docs/sigir/PAPER_DATA_MAP.md` sha1 = cb35d6d4e27d399f133d211a486e3e63c3a7fdb9, `tests/test_confrec_fillpaper.py` sha1 = 856e3bdb1025039b03581d42c3138f070d7df3ad. The previous record is
eb85eb91ef799ebb26e8f3b7723fd710f099eae4 (public commit 1767783). Changes: the `ext` slots (the extra analyses of addenda 6 and 8) now have specs and handlers
written against the real ML-1M extra file and a synthetic summary (no Amazon result existed locally); seven prose handlers (`c_ext_*`) turn the Holm families,
the robust readings and the FT-C / FT-Q wording of `summary.json` into the counts and conditions registered in addendum 6 item 9 and addendum 8 section 2;
`checks` gained a consistency test (every summarize input present locally must carry the sha1 that summarize recorded); the FT-Q rows were added. A
mechanical comparison of the 183 top-level definitions of the recorded version with the current one (git blob 1767783) shows that no existing decision
function changed; the removed definitions are the placeholder section of pass 1 (`EXT_DETAIL`, `EXT_PROSE`, `ext_pending`). The editor read no Toys,
Video_Games or Sports result and had no server access.

## 2026-10-05 · Code record: the extra-analysis module (addendum 6 items 2-8, FT-Q reading of addendum 8); recorded before any non-ML-1M statistic was computed

- `src/confrec/ftgrid_extra.py` = 8d0e5d86158b5f61c274d953d75ca63c169d44e5
- `scripts/sigir/run_ftextra.sh` = b2d3826ec2d539672b0011d64f6e8477a8eef3a4
- `tests/test_confrec_ftgrid_extra.py` = 4b99e214992318ea51b207427ee5b5630872ac6f

Written by an implementer agent, reviewed twice by an independent same-family reviewer (two rounds, no blocker; three major and eight minor findings fixed,
each replayed as a unit test; the reviewer recomputed 61 ML-1M statistics with its own code, all within 1e-15). Tests: 58 pass on the server
(CPU, with the report's 26). The module reuses the bound report code read-only and reproduces its registered numbers exactly (`registered_pooled`).
ML-1M run (exploratory, server CPU): `outputs/confrec/ftgrid/extra/ml1m.json` sha1 dfb7fc476750e52faef0cd76744d99bf830b5f68 (G_wu zero-shot +0.0001,
LoRA +0.0091; G_{LLM|CF} LoRA +0.0035 [-0.0016, +0.0092]; item share of the LoRA confidence 0.902; dUAUC(L - q-hat) LoRA -0.0161 [-0.0291, -0.0026]; deployable
top-k error rate 0.401 / 0.318 against the oracle 0.339 / 0.250 for zero-shot / LoRA). Notes for the record: (1) the module was extended after addendum 8
was written (FT-Q: the q-hat companion, the control-root scoping, `ft_wording`, `summarize --ftc/--ftq`), so addendum 8's "used as they are" holds for the
bound code only; (2) the module's readings R1-R19 (docstring, copied into every output) include choices stricter than the registered text (R13) and the
cut rule `FTEXTRA_NOT_RUN <dataset>` for a panel whose fine-tuned runs are cut at a section-10 checkpoint; (3) until this commit the server checkout held
earlier versions of the module and runner (f5fe320a, 56b0aad9), which refuse nothing relevant but have the old dry-run guard; the reviewed files are the
ones recorded here. At the time of recording the Toys fine-tuned job (`32_ft_toys`) had finished on the server (report written 2026-10-05 10:29 +0800); no
Toys, Video_Games or Sports report, score table, knockout analysis or extra-analysis output has been opened or pulled by anyone, and the server's Toys
files were only listed by name and size.

## 2026-10-05 · Wording rules record, update after editor pass 1 (recorded before any further result file was pulled)

`scripts/sigir/fill_paper.py` sha1 = eb85eb91ef799ebb26e8f3b7723fd710f099eae4 and `docs/sigir/PAPER_DATA_MAP.md` sha1 = 08354e11225cfbe5a7c7d745a98704067d6eb0be (the previous record of 2026-10-04 was b2702b7f69b8cfcb72668d860b9a821a9b32e1e9
and 01a72d00c197fea350788d13d3529436dd6a3ecb; the former version is the public commit 807cfb9). Changes: specs for the restructured tables (tab:tracks,
tab:teaches, tab:exposure summary rows, tab:deviations, the gate table), the alias `ext` (the extra-analysis files of addendum 6; every `ext:` slot reads red until
its file exists), the rename of the verbalised reranker's row label, and three new decision functions: `c_eb_count` and `c_g_count` implement the count rule of
addendum 6 item 9.2 (how many panels confirm E-B and G per regime, never one word for a mixed outcome) and `p_mir_decision` prints the registered stage-3 label
verbatim. A mechanical comparison of the 175 top-level definitions of the previous version with the current one finds no existing decision function changed (the
differences are the spec tables, the table builders, the alias list and the removed cell handlers of dropped table rows). The count functions read the ML-1M grid
report, which had already been read; the rule they implement was registered before they were written. No Toys, Video_Games, Sports, Llama, Z2, S6 or slot result
has been pulled.

## 2026-10-05 · Amendment 3 addendum 8 (the Toys permutation control withdrawn; item-only teacher control FT-Q; recorded before any control adapter existed)

`idea-stage/PREREG_AMENDMENT_3_ADDENDUM_8.md` sha1 = 85b22ba8262e5eb3a5a76ef47fb553288bfde483. The implementer of addendum 6 item 8 reported, and a CPU count on the registered TRAIN
panels (inputs only) confirmed, that a within-item permutation cannot change items that occur once: the share of TRAIN examples in items with at
least two examples is 97.7% (ML-1M), 23.8% (Toys), 64.6% (Video_Games), 21.3% (Sports), and on Toys the permutation moves 768 of 15,010 labels
(5.1%), so a high retention would be nearly mechanical. The Toys permutation run is therefore withdrawn (never run; its draft script is not
used); FT-C stays ML-1M only. FT-Q registers an item-only teacher (labels = the top round(beta n) examples by the method slot's TRAIN q-hat,
seed-0 tie key; same prompts; two adapters per dataset in the root `outputs/confrec/ftgrid_q/`; retention R_Q and labels as addendum 6 item 8) on
ML-1M and Toys (required) and Video_Games and Sports (run unless cut on 2026-10-29), with a wording rule that needs FT-C ITEM_DRIVEN on ML-1M and
FT-Q ITEM_DRIVEN on every dataset run. Recorded before any FT-C or FT-Q adapter existed or was scored and before any Toys, Video_Games or Sports
fine-tuned report, extra-analysis file or knockout analysis was read (the Toys job `32_ft_toys` was scoring its knockout arms; stage 6 had not run).

## 2026-10-05 · Amendment 3 addendum 7 (S6 wording, composition diagnostics, conditional item-stratified arm; recorded before any S6 arm was built)

`idea-stage/PREREG_AMENDMENT_3_ADDENDUM_7.md` sha1 = 832c04957ec18b40d3a5bf3f884b8e513c574713. Recorded before stages A and B of `run_ftprune.sh` ran (queue job `47_prune_AB` is
pending behind the fine-tuned grid), before any pruned adapter and before any S6 outcome. RQ6 is re-worded (no "noisy examples"; P2 =
uncertainty-selected pruning; P1 matches class counts only); composition diagnostics (items left without examples, item label-rate
distortion, TEST-item coverage, head-tercile share) are computed from the prune manifest by a new file, `ftprune_compose.py`, whose sha1 is
recorded before the `prune` record; the item-stratified random arm P1s is run only if P2 - P1 reads BEATS, and is defined completely in the
addendum so that no choice is left afterwards. No bound file changes.

## 2026-10-04 · Wording rules record (Amendment 3 addendum 6 section 9.3; recorded before any further result file was pulled)

`scripts/sigir/fill_paper.py` sha1 = b2702b7f69b8cfcb72668d860b9a821a9b32e1e9 and `docs/sigir/PAPER_DATA_MAP.md` sha1 = 01a72d00c197fea350788d13d3529436dd6a3ecb.
The decision functions of the fill script (direction words only from registered tests and Holm families; mixed outcomes stay red slots; "not
run" only for a registered cut) are the paper's wording rules from now on. The script fills 241 of 774 slots from the committed result files
(selection, gate, Gate-FT, the Sports next-item audit analysis, the ML-1M grid report and the split files); nothing from the Toys, Video_Games,
Sports, Llama, Z2, S6 or slot runs has been pulled. Extensions that only add specs for result files not yet pulled get a new line here before
those files are pulled; a change to an existing decision function after results have been read needs a dated addendum.

## 2026-10-04 · Amendment 3 addendum 6 (reviewer-driven additions; recorded before any non-ML-1M outcome of its items was read)

`idea-stage/PREREG_AMENDMENT_3_ADDENDUM_6.md` sha1 = 868f87b6819fddd03717f44a9baac63ea4db4cf3. Recorded (pushed to GitHub and pulled by the server) 2026-10-05
about 04:30 server time (2026-10-04 about 20:30 UTC). At that time the Toys fine-tuned job (`32_ft_toys`, queue) was in stage 4 (decomposition arms; the like passes of the
zero-shot model and of s0-s2 were finished) and no Toys, Video_Games or Sports report, no knockout arm, no FT-C adapter and no Llama, S6, Z2 or
method-slot result existed; the files of the running job were not opened. The addendum adds: two corrections (the non-prior share is withdrawn as
"an upper bound on the personal share"; direction words follow registered tests only), E-W (a within-user stacker estimator with repeated
cross-fitting, as sensitivity for E-D), E-F (complementarity to collaborative filtering, family E-F), E-G (item share of MF and of the labels, the
e-share), E-H (sparse/unseen strata, family E-H), E-J (the LLM against the item mean, the information-matched item mean and MF, family E-J), E-C'
(a deployable top-k decision), the FT-C reading rule and Toys FT-C (new script, no bound file changes), the admission and wording rules, and the
code to be recorded. Read before writing: Gate-FT, the ML-1M grid report, the gate-fix selection and confirmation, the Sports audit analysis, two
same-family reviewer reports on the draft, and one exploratory ML-1M recomputation with user-centred stacker features (scratch code on the server,
CPU, not a registered script): G(zero-shot) = +0.0001 [-0.0030, +0.0036], G(LoRA s0, s1, s2) = +0.0067, +0.0102, +0.0108 (mean +0.0092
[+0.0024, +0.0182]), G_CF = +0.0378 [+0.0253, +0.0498], against the registered pooled-stacker values -0.0018, +0.0096 and +0.0362: the
estimator choice moves G(zero-shot) by about +0.002 and leaves the fine-tuned mean and the sign of the regime contrast unchanged. ML-1M
values of the addendum's items are therefore exploratory. Code validation recorded here as well: the four server-only tiny-model tests of the
method slot (`tests/test_confrec_ftmethod.py -k tiny`, CPU, lumen env, commit 82ffbc9) passed (4 passed), and `ftgrid_freeze --check --stage
amendment` and `--stage core` passed on the server at that commit.

## 2026-10-04 · Amendment 1 (retroactive record; its sha1 was not entered in this log when it was written)

`idea-stage/PREREG_AMENDMENT_1.md` sha1 = fcf1a59b8a5004891c51eb16cfcb98f0b52785ea (git blob 5796c736e5b8735168d20cf52baaf007baf89b72). A methodology
review found that no line of this log carries the sha1 of Amendment 1. The file is byte-identical in its first commit,
`189c164855aaf5e0d96cd405878fbd6ad3acf58b` (branch `sigir2027`, authored 2026-10-02T12:27:08-05:00 = 17:27:08 UTC, "pre-GPU review fixes ... PREREG
amendment 1 ..."), and at the current head. Evidence that it preceded the pilot data: the GPU server's reflog records `pull --ff-only` to `189c164`
from GitHub at 2026-10-03 01:37:10 +0800 (= 2026-10-02 17:37:10 UTC), so the commit was public by then, and the earliest Pilot-1 output on the
server (the panel brand-popularity sidecar) was written at 2026-10-03 01:39:25 +0800, the earliest `pilot1_mirror` file at 01:41:22. The statement
of the draft that every registration hash was recorded before the runs it governs is therefore exact for Amendments 2 and 3 and their addenda;
for Amendment 1 it rests on the commit and pull times above, not on a line of this log.

## 2026-10-04 · Amendment 3 addendum 5 (two corrections to addendum 3; recorded before any prior-offset adapter existed)

`idea-stage/PREREG_AMENDMENT_3_ADDENDUM_5.md` sha1 = 6f3e7b0af8bfa40274825c39f835149a6c28cc49. The scorer (a bound file) does not record the Yes-mass share
on the answer token, so the train/test shift mismatch is not measured and is stated as a limitation; lifting the block of an INCOMPLETE slot dataset
needs a dated addendum and a recorded report-code change. Method-slot code (final, before the `method` record): `train_lora_offset.py`
aaae5e812637551a6430031349a418c43a78c594, `ftmethod_report.py` 8414bce88acee2a30721148404f32d3dbe0863ef, `run_ftmethod.sh`
121ad50fced444b0db67a87d9665b188d9c92539 (60 tests pass locally, 4 server-only tests pending; b's own optimizer group at lr 1e-2 is the default).

## 2026-10-04 · Gate-FT result (registered G9; `outputs/confrec/gateft/gate_ft.json`): **GATE_FT_PASS**

TALLRec-style LoRA (r 16, alpha 32, lr 1e-4, 1 epoch, seeds 0, 1, 2; prompt V1; trained on the 23,348 pre-T DEV candidates, 46 minutes per seed
at 8.5 examples/s) scored on the fresh CONFIRM users' post-T candidates (366 users with both classes): **UAUC per seed 0.7400, 0.7447, 0.7427; mean
over seeds 0.7425, 95% user-bootstrap CI of the seed-averaged UAUC [0.7232, 0.7609]**, against the registered bar 0.65 (point estimate of the
mean over seeds). Integrity holds for every run (no problems listed). Zero-shot context on the same rows (never gates): UAUC 0.5960 and a
Platt slope of 0.092 (ECE 0.344), so the paired gain is dUAUC(LoRA − zero-shot) = +0.1465 [0.1242, 0.1686]. Consequences (Amendment 3
section 0): the fine-tuned program (sections 1–3 and 5–9: the fine-tuned grid on the four rated domains, the pseudonym knockout, S6 pruning, the
Llama replication, the permutation control and the nested method slot) now runs; the Gate-FT adapters are this program's ML-1M adapters.

## 2026-10-04 · Amendment 3 addendum 4 (S6 implementation readings; recorded before the TRAIN zero-shot pass and any pruned adapter)

`idea-stage/PREREG_AMENDMENT_3_ADDENDUM_4.md` sha1 = b0bca2a7cfa269da58d25297bd9dfc3e48b43372 (class-size rounding, τ's quantile rule and the handling of
non-finite logits, P1's seeded draw, the claim-label order with BEATS winning over ABOUT_EQUAL, per-run AUC rows, and that the zero-shot
TRAIN pass is a section-0 scoring and not an S6 run). No S6 or slot outcome existed.

## 2026-10-04 · Amendment 3 addendum 3 (the optimizer of the offset b; recorded before any prior-offset adapter was trained)

`idea-stage/PREREG_AMENDMENT_3_ADDENDUM_3.md` sha1 = 8e278c8bdbff8137a67b26075d7676d39d0dfe31. The code review of section 7 found that b has no optimizer
setting and that in the LoRA group (lr 1e-4) it could move by only about 0.04 over the run, which would make the prior-offset arm SFT by
construction; b now has its own AdamW group with lr 1e-2 (fixed from the optimizer arithmetic, not tuned); INCOMPLETE is not a failure;
the train/test shift reading is stated. No slot outcome, adapter score or zero-shot panel existed when this was written.

## 2026-10-04 · Amendment 3 addendum 2 and the extended freeze tool (a bound file changed; recorded before any adapter was scored)

`idea-stage/PREREG_AMENDMENT_3_ADDENDUM_2.md` sha1 = 402ff901ba622b0848ce5aa288d41f960f77ba73 makes the `prune` and `method` freezes concrete
(manifests of the signals/subsets and of the q̂ standardisation constants; FREEZE blocks may also come from addenda). The tool
`src/confrec/ftgrid_freeze.py` (a core file) was extended accordingly: new sha1 = a9931a12bc4b184dab3db9a65f518c0eb8a6dab9
(it replaces e90c9fbddaeb9e1643a285737d3a7757c91518d4 of the full record below; nothing else of the core list changed, and no outcome
statistic of any adapter, zero-shot panel, pruning or slot run existed when this was done).

## 2026-10-04 · G7 stage 3 (the registered MIRROR arms; runs because G6 was GATE_PASS): code recorded before its first scoring

`scripts/sigir/run_gatefix_stage3.sh` (with `src/confrec/gatefix_stage3.py`, the input check) scores, under V\* = V1 on CONFIRM ML-1M and on
the first 1,500 fresh Toys users: like, dislike, like_para (V1's threshold-family strings), the 8-donor swap prior and the no-history
prior; and, under V0, the next-item no-loss check (next, like, dislike, like_para) on the first 1,000 events of the sports VALID panel
(never sports TEST events 1–1000); then `pilot_mirror` and `pilot1_gate --stage3_gate` write the decision (labels POSITIVE / NULL / NEGATIVE
/ INDETERMINATE / INCOMPLETE / GATE_FAIL_UNINTERPRETABLE). Estimated 2.7 GPU-h. Choices accepted: the correlation-matched ensemble null
is the per-user binormal prediction (the Pilot-1 decision, `pilot_mirror`), the literal within-user z-sum is context only; the Video_Games
fallback cannot trigger (Toys has 12,702 fresh users); the VALID next-item panel is scored with all four questions in one fresh run; the C-CRP
context on VALID is empty by construction (its ranks cover TEST events only; context never gates). 46 + 17 tests pass in the CPU sandbox.

| file | sha1 |
|---|---|
| `scripts/sigir/run_gatefix_stage3.sh` | 9c69a7b5a19417fb9c6123c642050325ab93a344 |
| `src/confrec/gatefix_stage3.py` | 382c519f78174aa6c19be86427caa102724d90f1 |
| `src/confrec/pilot_mirror.py` | a4a8bffb5c30035538d63dcb8013b09d7e6fa3c8 |
| `scripts/sigir/pilot1_gate.py` | 3a4c163293a7d6beebff883838fe1786a69aee59 |
| `src/confrec/gatefix_select.py` | b7f4380d872e883f331938e9ebdae50a938e08a6 |

## 2026-10-04 · Amendment 3: the two Llama-3.1-8B-Instruct split files (record for any Llama job; Amendment 3 section 0)

Built by `MODEL=<Llama-3.1-8B-Instruct> OUT_ROOT=outputs/confrec/ftgrid_llama STAGES=0 bash scripts/sigir/run_ftgrid.sh <d>` under the selected
prompt V1 (same rows, T and users as the Qwen splits; the length audit uses the Llama tokenizer: p99.5 prompt length 294 tokens on
ML-1M and 605 on Toys, share above 1,024 = 0, so micro-batch 8 × accumulation 4 and max_len 1,024 apply). ML-1M's T equals Gate-FT's
(checked). No Llama job has run.

```
outputs/confrec/ftgrid_llama/panels/ml1m/ftgrid_split.json = c75ef45b134b3427244b839a251b26dbd875ea95
outputs/confrec/ftgrid_llama/panels/toys/ftgrid_split.json = eda0ff39c0e110836844a1a679190153c53c36e4
```

## 2026-10-04 · Amendment 3 FULL RECORD (FREEZE `core`): bound code and the four `ftgrid_split.json` (before any adapter was scored)

Produced by `python -m src.confrec.ftgrid_freeze --print --stage core` on the server repo (commit 85c2ee6; the split files were built
the same day by `run_ftgrid.sh` stage 0 under the selected prompt V1: ML-1M and Toys from the gate-fix stage-0 DEV and CONFIRM panels,
Video_Games and Sports with `build_rated_panels --hist_len 20`). Every listed file is unit-tested. Gate-FT adapter training started at
20:31 server time under the amendment-stage record; scoring of any adapter and the zero-shot panels of section 4 may start now.
Split facts (V1; TRAIN = first 1,500 rows, EVAL = next min(3,000, rest) rows; none of the TRAIN prompts exceeds 1,024 tokens, so
the registered 1,024 / micro-batch 8 × accumulation 4 recipe applies everywhere): ML-1M T = 977099602.0 (equal to Gate-FT's, checked),
23,348 pre-T training examples, 1,683 EVAL users, 6,466 TEST candidates, 366 TEST users with both classes (204 in the tail), seen
share 0.966; Toys 15,010 / 3,000 / 7,463 / 943 (331) / 0.111; Video_Games 15,353 / 2,609 / 6,645 / 820 (242) / 0.214; Sports 15,695 /
3,000 / 7,654 / 997 (226) / 0.136; S_d = all TEST users with both classes in every domain (each below 1,000).

```
idea-stage/PREREG_AMENDMENT_3.md = 4db1b62f30399c4d2d06fae205cfc0a52db8a419
idea-stage/PREREG_AMENDMENT_3_ADDENDUM_1.md = 7d855492a6f6486f964a2601c485b1b89b8d58ee
src/confrec/ftgrid_data.py = cc8e1c4d07360aa75ce5e62782a4f3a89fb10716
src/confrec/ftgrid_report.py = 2604709eaefe15496c10aee259de5ad3f6f72479
src/confrec/ftgrid_freeze.py = e90c9fbddaeb9e1643a285737d3a7757c91518d4
src/confrec/lora_trainer.py = 4200b42cf1e987acbd477ffe55e248c036519b3e
src/confrec/train_lora_yesno.py = 0e3e994ff4f879c1ee8c8a0a7bac1f37ae528728
src/confrec/pyes_scorer.py = a5c45b361170b64794d8ede5721eacec17955ed4
src/confrec/prompting.py = e7e9bf7f1f998d28b6f65d73fdfde491411000e5
src/confrec/build_rated_panels.py = 3aa60f1d3541826cbfc3abbff755d06c43486e8a
src/confrec/gateft_data.py = 226835415ce0fd1c0240693b3227ffa065a70a6e
src/confrec/gateft_eval.py = 9a33c82498cd9acdbf7f92f40de3f96e42fd4125
src/confrec/forensics.py = efd1ad03df0d02be04130737f25729cd8ff32e11
src/confrec/pilot_pseudonym.py = 27737ecb459d74f83333e94c5e1b5f284c6287b2
src/confrec/pseudonymize.py = 2bcc0ce23f96c998e7c8bb3a98d916c29271f4bb
src/confrec/diag_battery.py = 01c91a3d251fb6642255b04c1752d098149c7ff7
src/confrec/metrics.py = 79564e1254b380b02321c164ed919321063bd0c7
src/confrec/stats.py = 864b2b82cfd212004afe5bab0f74e83f7250adb0
scripts/sigir/starperm_panel.py = 76da557ad7212cd534fa74c20161566a20fba4a4
scripts/sigir/run_gateft.sh = de72075f70fb3ce74f2c1b8a7166913f5509b3d8
scripts/sigir/run_ftgrid.sh = 17a55016269263e06bc93757bb675b277e1f1745
scripts/sigir/run_llama_nextitem.sh = d1199302795fecc274d8217193198e577c4ade86
outputs/confrec/ftgrid/panels/ml1m/ftgrid_split.json = e5e97ca6316e97173d72d5f3f1b476e34da3e8b6
outputs/confrec/ftgrid/panels/toys/ftgrid_split.json = e43416bb960cb86ae6fb1ed7a2ad16f063de5e28
outputs/confrec/ftgrid/panels/games/ftgrid_split.json = dfd01e9e4012d19a958c154e848c546774d62121
outputs/confrec/ftgrid/panels/sports/ftgrid_split.json = ad02a5962a63a352f74444eb050b5d5cdd3c3340
```

## 2026-10-04 · Next-item audit module: Z2 support added (default output byte-identical; Amendment 3 section 4)

`nextitem_audit.py` gained `run --segments {auto,single}`, `--test_role`, `--valid_role`, `--first_event` and a `restrict` subcommand (the
second-backbone replication on TEST events 1,001–3,000 and a 500-event VALID sample). The frozen module (sha1 8e3fa3d0…) and the new
one were run on the same synthetic inputs in nine configurations: the JSON bytes (with the `timing_s` line removed) are identical, and
the test suite pins digests of the frozen output (39 tests). `Q_ENDPOINTS` and every estimator are unchanged. New hashes:
`src/confrec/nextitem_audit.py` de08087f43acff59292b16292be74c2b071b7c7a, `docs/sigir/NEXTITEM_AUDIT_SPEC.md`
80730682855417c878e41e8c4c6a107b0e96578c. Not an outcome-driven change: no Z2 score exists yet.

## 2026-10-04 · G6 confirmatory gate result (registered stage 2; `outputs/confrec/gatefix/confirm/gate.json`): **GATE_PASS**

V\* = V1 on the 1,683 fresh ML-1M CONFIRM users (disjoint from the dev users, panel identity checked against the stage-0 manifest):
**UAUC(V1) = 0.6033, 95% user-bootstrap CI [0.5944, 0.6124]**, against the registered bar 0.60 on the point estimate; E1 holds
(censored = 2 share 0, no overlength prompt, mean Yes+No mass 0.99999). Context only (never gates): the registered prompt V0 on the
same users, UAUC 0.5756 [0.5665, 0.5848]. The gate is met on the point estimate by 0.003 and the interval extends below 0.60; both
statements are reported (the dev estimate 0.6122 carried the expected winner's curse). Consequences (Amendment 2): the zero-shot
regime under V1 is a valid regime, G7 stage 3 (the registered MIRROR arms: like, dislike, like_para, swap prior, no-history prior on
CONFIRM ML-1M and the first 1,500 fresh Toys users, plus the sports VALID next-item no-loss check) now runs, and Gate-FT trains under V1.

## 2026-10-04 · Amendment 3 recorded (enables the training of the Gate-FT adapters; scoring still waits for the full record)

`idea-stage/PREREG_AMENDMENT_3_ADDENDUM_1.md` sha1 = 7d855492a6f6486f964a2601c485b1b89b8d58ee (dated addendum 1: how the three
implementers' open choices were resolved; part of Amendment 3, recorded before any adapter was trained or scored).

`idea-stage/PREREG_AMENDMENT_3.md` sha1 = 4db1b62f30399c4d2d06fae205cfc0a52db8a419 (the fine-tuned program; section 0 of that file:
training needs this line, scoring of any adapter or of the zero-shot panels of section 4 needs in addition the sha1 of every file of
its FREEZE `core` list and of the `ftgrid_split.json` of each domain, to be written by
`python -m src.confrec.ftgrid_freeze --print --stage core`). Gate-FT training runs under the prompt of `selection.json`
(`gate_ft_prompt`, see the next entry). Written before any adapter existed; no outcome statistic of an adapter exists.

## 2026-10-04 · G3–G5 dev selection result (registered stage 1; `outputs/confrec/gatefix/dev/selection.json`, burned dev users)

**FIX_FOUND, V\* = V1** (the label-aligned threshold question "Will this user rate the candidate item 4 stars or higher (on a 1-5
scale)?"). Gate-FT is trained under V1 (`gate_ft_prompt = V1`). Dev UAUC of the `like` question, 1,500 users per panel; every variant is
eligible (E1 and E2 hold for all seven):

| variant | ML-1M dev | Toys dev |
|---|---|---|
| V0 (registered control) | 0.5875 | 0.5397 |
| **V1** | **0.6122** | 0.5442 |
| V2 (TALLRec layout) | 0.6110 | 0.5434 |
| V3 (20 events) | 0.5984 | 0.5409 |
| V4 (genre metadata) | 0.5401 | 0.5465 |
| V5 (system message) | 0.5952 | 0.5397 |
| V7 (V1+V3+V4+V5) | 0.5845 | 0.5440 |

V1 and V2 are within 0.005 of the maximum; the tie rule (higher Toys dev UAUC, then the simplicity order) selects V1. UAUC(V1) −
UAUC(V0) on ML-1M dev = +0.0247, paired one-sided user-bootstrap lower bound +0.0197 > 0 (level 0.9917, 2,000 resamples), so a fix
is found and stage 2 (the one confirmatory gate on the fresh ML-1M users: UAUC(V1) ≥ 0.60) runs. These dev numbers are on users that
were burned in Pilot 1 and carry a winner's curse; the confirmation decides.

## 2026-10-04 · Amendment-2 FREEZE record (G-round: prompt-fix, diagnosis battery; before any such GPU job)

Produced by `scripts/sigir/run_gatefix.sh` stage 0 on the server repo (`/root/autodl-tmp/lumen-rec` at commit 6da7ae4), file
`outputs/confrec/gatefix/FREEZE.txt`; the same hashes were produced by an earlier rehearsal on a snapshot of the code (apart from
the amendment, which gained the O carve-out and the G7 scale note in between). ML-1M: 3,183 eligible users, the first 1,500 lines
of the rebuilt panel hash to the Pilot-1 panel (MATCH), DEV 1,500 + CONFIRM 1,683. Toys: 14,202 eligible users, prefix MATCH,
DEV 1,500 + CONFIRM 12,702. Video_Games 4,109 and Sports 16,712 eligible users (counted, untouched).

| REQUIRED quantity | sha1 |
|---|---|
| `idea-stage/PREREG_AMENDMENT_2.md` | 46e96c763fb086e4717897acdbd70d952027233d |
| rendered prompt bank (7 variants + T0 probe, first 50 rows of each dev panel, 13,168 prompts) | 6cee435adc05c860c427e2a820dba57bc86cc0c1 |
| diagnosis battery bank (arms T0, T0_digits, T1, T2, T3, starperm; 10,956 prompts; spec baa7a763f23119f9de4c0768d29e9c820f5f2e32) | 0c7288d2d43b6225603a50739eb62de6f6cdfc4c |
| ML-1M DEV user ids | b7124d7b2d000ede32ea76078a775685a3734120 |
| ML-1M CONFIRM user ids | 9660a965bf6b0afaeb5952d3c382f280ef111d93 |
| Toys DEV user ids | 77b46ce3839d2191a3cf211ddad8e7ebfa905045 |
| Toys CONFIRM user ids | 44e673cfb2399a1a89d2ad83bad1bb9b07995589 |

Per variant (informational): V0 cc6779de2ce461076a2ce2b264e100f35041a107, V1 d135b9b1479d7a0d8dda9793efacf1d63d945cd7, V2
f8925a44b438a68c517239109c41feb618bba15d, V3 6a1f16dfc27ed632de4cbb387cbecb6879057ca3, V4 fb2a91e6dedc77a3130d23f9641e79ba15b02d3e,
V5 f8c8969d8ac94c1ef35928dd0c6e0382086e6ad0, V7 e81f23498daf842b2505100b901ad9542638e43c, T0 probe
b8eab6b78be5c4eda9606d89644130066412eb69; prompt strings cc215d598a6a6a0f6b6545df690f60d44ee6bafb; dev panels ML-1M
74b8873b23b436a62cdadb3215ae0936f4d8ffd0, Toys dde50d633416719f7cefd0af04b4f92364f82cd3. Registered V0 check: the V0 rendering
of the real Pilot-1 panels reproduces the stored prompts_sha1 (ML-1M 12e83c4f…, Toys cd47bdb8…, sports 81a2106b…) and swap
prompts_sha1 (integration agent, CPU dry run, 2026-10-04: IDENTICAL).

## 2026-10-04 · Freeze of the next-item audit analysis (Amendment 2 section N; `NEXTITEM_AUDIT_SPEC.md` section G)

Recorded before `nextitem_audit run` was executed on any LLM audit score. Provenance, stated plainly: the scoring started
2026-10-03 16:55 local under carve-out N; the detailed spec (`NEXTITEM_AUDIT_SPEC.md`) was written 22:48 the same day, while the
scoring was running, from the endpoints already declared in the judge's action 7 (2026-10-02, before any scoring). Until this
record the only audit output read was the scorer's own progress lines (chunk counters, prompts per second); the analysis module
was written and tested on synthetic data and on the C-CRP / baseline ranks, never on an LLM audit score file. Sports TEST is
fully scored on the server (100/100 chunks) but has not been analysed.

| file | sha1 |
|---|---|
| `src/confrec/nextitem_audit.py` | ea848324e8421576dd22946641da82838a3775db |
| `docs/sigir/NEXTITEM_AUDIT_SPEC.md` | 8d584f43d0f7fe637b5484a23a52e60b0b680324 |
| `scripts/sigir/export_ref_exposure.py` | 666bee91a1d33eb50679d6a338d7126c6f8fd9aa |
| `docs/sigir/ref_exposure` manifest (40 files, 34,969,901 bytes; sha1 of the lines `<relative path> <sha1>` in sorted order) | d32c79b0c1b13d3838474c605009950c84dac548 |

The registered endpoints are the entries of `Q_ENDPOINTS` in the module at this sha1. The 35 MB of exposure exports are accepted and
committed (the baselines' ranking files exist only on the workstation; the server reads them through git).

**Addendum (same day, after the first real run on sports).** The registered niche-minus-mainstream served-share difference came
back undefined: the popularity profile of the sports users is heavily tied (about half of them have every mapped history item in
the head group), `rank_bins` leaves the top quintile empty, and the bin-0 vs bin-4 difference has no estimate. The module now
compares the lowest with the highest non-empty quintile (`niche_bin`, `mainstream_bin` are reported). Made after that one field was
seen to be empty, before any niche value (served share, utility among served) had been looked at; nothing else changed
(33 tests). New hashes: `nextitem_audit.py` 8e3fa3d0f28acd4ca2843631422fe9150f03da6f, `NEXTITEM_AUDIT_SPEC.md`
79911b40bb1812c1bcd0a5a1ebf00761550c44fa (exporter and exposure manifest unchanged). The sports run is repeated with this version.
What was seen before the fix, for the record: the sports ranking, exposure, calibration, error-anatomy and selective-serving
sections of the first run (they are unaffected by the change).

## 2026-10-04 · Infrastructure smoke test of the Gate-FT path (Amendment 2 carve-out O)

Synthetic panel only (`scripts/sigir/synth_smoke_panel.py`, invented titles and random labels); no real user, item or rating
data; nothing below enters any gate, table or claim.

- **First attempt failed after 20 s.** `TrainingArguments(warmup_ratio=...)` does not exist in transformers 5.18, the
  server's version. Fixed in `train_lora_yesno.py` (`warmup_steps=0.03`, where a float below 1 is a ratio of the total steps;
  `dtype=` instead of `torch_dtype=`). Without this test Gate-FT would have failed at its scheduled start.
- **Second attempt passed.** Qwen3-8B LoRA (r 16, alpha 32, bf16, gradient checkpointing, micro-batch 8 × accumulation 4,
  last-token loss), 256 synthetic examples (longest prompt 553 tokens): `train_runtime` 66.7 s = **3.84 examples/s**, 86 s wall
  including the model load, **peak GPU memory 19,958 MiB**. Planning figure for Gate-FT: about 24k examples at 3.84/s,
  i.e. ≈ 1.7 GPU-h per seed instead of the 3 assumed (the real prompts decide; BENCH=1 on the real panel replaces this).
- **vLLM 0.30.0 loads and applies the adapter** (`pyes_scorer --lora`): mean |Δlogit| = 3.75 against the base model on
  the same 768 prompts; every row finite, censored = 2 share 0, mean Yes+No mass 0.99998.
- **Trainer loss scale.** `lora_trainer.last_token_trainer` follows the stock `sum / num_items_in_batch` rule under gradient
  accumulation. A plain micro-batch mean (the first version) made the accumulated gradient `grad_accum` times too large:
  in the control run the weights moved 0.17 further. Against the stock Trainer on a tiny LoRA model (SGD, CPU) the
  last-token trainer now agrees to 2e-7 (accumulation 1) and 7e-8 (accumulation 2): `tests/test_confrec_lora_trainer.py`.

## 2026-10-04 · CPU forensics A2–A7 on the Pilot-1 score files (EXPLORATORY: burned users, hypothesis-generating only)

File: `docs/sigir/forensics/pilot1_forensics.json` (sha1 73507b31…, produced by `src.confrec.forensics`, server CPU,
panels ml1m_rated, toys_rated, sports_next_1k; A8 needs the local C-CRP ranking file and is not in this run). Nothing here
is a gate, a selection input or a citable result; each quantity is re-measured as a pre-declared endpoint on fresh users
(Amendment 3).

| quantity (UAUC unless stated) | ML-1M | Toys |
|---|---|---|
| zero-shot like-logit (Pilot 1) | 0.587 | 0.540 |
| item prior π alone (8-donor swap) | 0.603 [0.594, 0.611] | 0.512 [0.501, 0.522] |
| no-history logit alone | 0.546 | 0.488 |
| evidence = like − π | **0.507** [0.497, 0.518] | 0.525 [0.515, 0.536] |
| π split-half r, Spearman–Brown 8-donor | 0.677, 0.807 | 0.673, 0.804 |
| prior-only item mean (ratings before the candidate time) | 0.753 [0.744, 0.761] | 0.693 [0.682, 0.703] |
| temporal biased MF, post-T pairs | 0.804 [0.784, 0.823] | 0.554 [0.513, 0.593] (38% cold items) |
| MF personal residual (MF − b_u − b_i) | 0.677 [0.654, 0.697] | 0.497 [0.458, 0.536] |
| Platt slope of the zero-shot logit; ECE raw → Platt | 0.062; 0.305 → 0.016 | 0.022; 0.364 → 0.024 |
| partial Spearman(π, log popularity \| item mean), item level | 0.361 [0.328, 0.393] | 0.248 |
| MIRROR − correlation-matched ensemble null | −0.0048 [−0.0088, −0.0009] | (CI covers 0) |

Wording flags of Amendment 2 §A (wording only): π reliability ≥ 0.7 on both panels, so the evidence collapse is
interpretable; MF personal-residual UAUC ≤ 0.55 on Toys, so "no personal signal" is stated as a property of the Amazon
(consumed-item, median 3 history events) panels, not on ML-1M (0.68); the ensemble-null rule attributes MIRROR's ML-1M edge
to two-view diversity.

## 2026-10-02 · Pilot 2 (KuaiRec MAR vs logged) result: NULL

**Setup.**
- Qwen3-8B fp16, zero-shot, 300 users × 200 small-matrix videos (60k pairs).
- Target density D = 0.349 (distinct pairs).
- Local copy: `outputs/confrec_pilot/pilot2_kuairec.json`.

**Gate.**
- UAUC_MAR at thr = 2.0 is **0.469 [0.451, 0.486]** (279 users with both classes), below the 0.58 gate.
- Pooled AUC is 0.498.

**Registered consequence (§3.2 NULL; amendment P2).** The LLM cannot read KuaiRec zero-shot:
- B1/A7, B2, A6 and B10 are deprioritized.
- S3 (echo chamber) is answered only through static exposure on the next-item panels.

**Note.** UAUC is significantly *below* 0.5. A plausible mechanism: watch_ratio ≥ 2 favours short, replayed videos, which captions do not reveal. This is unverified and is not used for any decision.

## 2026-10-02 · Gate-failure diagnosis: the task is not hard; the zero-shot yes/no channel is weak

**Method.** `src/confrec/diag_rated_baselines.py` runs on the same Pilot-1 panel pairs. Item signals use only
*other* users' ratings (leave-user-out), so no label leaks.

| UAUC | ML-1M | Toys |
|---|---|---|
| item mean rating (LOO, non-personalized) | **0.760** | **0.712** |
| −item rating variance | 0.662 | 0.687 |
| item popularity | 0.637 | 0.535 |
| LLM like-logit | 0.587 | 0.540 |
| LLM MIRROR | 0.616 | 0.537 |

**Reading.**
- Spearman correlation of the LLM like-logit with item mean rating: 0.15 on ML-1M, 0.07 on Toys.
- The same correlation with popularity is 0.20 on ML-1M and 0.07 on Toys.
- The channel is intact but weak: it misses the item-quality signal.

**Correction (ARIS judge, 2026-10-02).**
- "The task is not hard" overstates the evidence.
- `item_mean_loo` is transductive: it uses other users' *later* ratings, on labels with 3 stars dropped. 0.760/0.712 is therefore an optimistic collaborative-filtering ceiling, not proof that the task is easy.
- The right statement: zero-shot content-only yes/no is weak on a task that collaborative filtering does well.
- Literature anchor (CoLLM, TKDE'25, Table 2, ML-1M): zero-shot ICL UAUC 0.527, MF 0.636, fine-tuned TALLRec 0.682.
- Prior-only (strictly-before) leave-user-out CF references are pending (Amendment 2 artifact checks).
- These are diagnostics. Any claim must be re-validated on fresh users and domains.

## 2026-10-02 · Pilot 1 result: token-channel GATE FAIL; decision = GATE_FAIL_UNINTERPRETABLE

**Setup.**
- Qwen3-8B, fp16, thinking off.
- No censoring on any of 1.13M scored rows; Yes+No mass about 1.
- Throughput about 191 prompts/s.
- Local copies: `outputs/confrec_pilot/pilot1/*.json`.

**Gate.**
- ML-1M raw(like) UAUC = **0.587** against the registered ≥ 0.60, so the gate **FAILS**.
- Sports raw(next) NDCG@10 = 0.209 against ≥ 0.186, which passes.
- Same-event C-CRP reference: 0.231.
- Per triage §3 nothing below the gate is interpretable. The registered remedy is "fix the prompt first".

**Diagnostics** (recorded for transparency; *not* decision inputs; any use must be re-validated on fresh users):

| panel | raw/like | MIRROR | placebo (2-prompt) | evidence (like − π) | PMI no-hist |
|---|---|---|---|---|---|
| ML-1M UAUC (1500 u) | 0.587 | 0.616 | 0.598 | 0.507 | 0.547 |
| Toys UAUC (1500 u) | 0.540 | 0.537 | 0.544 | 0.525 | 0.546 |
| Sports NDCG@10 (1000 ev) | 0.209 (next) / 0.212 (like) | 0.179 | 0.219 | – | – |

- **ΔUAUC(MIRROR − placebo)** by panel:
  - ML-1M: +0.019 [0.013, 0.024].
  - Toys: −0.007 [−0.014, −0.000].
  - Sports: −0.018.
- **Sports next-item:** MIRROR loses about 0.03 NDCG@10 against raw.
- **Overconfidence (ML-1M):**
  - Raw ECE is 0.30.
  - The Platt slope is 0.062, i.e. the logits are about 16× too extreme.
  - |logit| detects errors only weakly: AUROC 0.58.
- **Popularity link (ML-1M):**
  - corr(v, log-pop) = 0.30 at the pair level and 0.42 at the item level.
  - corr(π, log-pop) = 0.43.
  - corr(a, log-pop) is about 0 at the pair level and 0.12 at the item level.
- **Popularity link (Toys):** corr(π, log-pop) = 0.28.
- Within-user acquiescence spread is large (SD_pair_df 2.3 on ML-1M, 3.5 on Toys) but is not popularity-linked at the pair level.

## 2026-10-02 · M0 closed: panels byte-identical, code fixed, Pilot 1 launched

**Panel reproduction.**
- `panel_reference.py verify` now also checks the SHA-256 of the original files recorded in the old C-CRP
  provenance.
- All four rebuilt test panels are byte-identical to the originals for both `ranking_test.jsonl` and
  `candidate_items.csv`. The domains are sports, toys, home and tools, and each covers 10000/10000 events with no
  extra, duplicate or out-of-order rows.
- So the next-item comparison against C-CRP v3 and the 8 official baselines runs on exactly the frozen data.

**Pre-GPU fixes.**
- A fix workflow split the work into 5 disjoint groups. Each group was implemented, reviewed by 2 adversarial
  reviewers, then repaired, followed by an integration step.
- That integration step included a CPU end-to-end dry run of all three pilots and of the real shell scripts in a
  sandbox.
- 148 confrec tests pass.
- Additional real defects found and fixed during review:
  - common-word store names (Kids, Toys & Games) were rewritten in every prompt;
  - pseudonym generation looped forever above 22.7k stores;
  - next-item arms were ranked on different candidate sets when a value was censored;
  - an absent primary endpoint produced a definite NULL;
  - pilot 3 was not gated on pilot 1's token channel.

**Orchestrator sign-off before data** (amendment P1.5b and the P2 addendum):
- `SD_pair_df` gates;
- all-time popularity;
- exact tie-group NDCG;
- NULL wins the NEGATIVE/NULL overlap;
- Pilot-2 POSITIVE additionally needs the lift CI above 1.

**Pilot 1 launched** (`scripts/sigir/run_pilot1_mirror.sh`, Qwen3-8B fp16). ML-1M rated panel: 3,183 eligible users,
1,500 used, 29,365 candidates, like-rate 0.660, 0 duplicates removed.

## 2026-10-01 · M0 infrastructure (server `lumen-gpu`, RTX 4090 D 24 GB, vLLM 0.30.0, torch 2.13, pandas 3.0.6)

**Panels.** Rebuilt from the Amazon-2023 HF mirror (slimmed, Range-resumable) and verified against
`docs/sigir/panel_refs`. Each domain was checked by `source_event_id`, positive item, and the sha1 of the ordered
101 candidates:

| domain | interactions (processed) | eligible users | verify |
|---|---|---|---|
| sports | 5,542,756 | — | 10000/10000 |
| toys | 5,502,054 | 992,296 | 10000/10000 |
| home | 31,541,397 | 5,056,360 | 10000/10000 |
| tools | 9,962,279 | 1,714,369 | 10000/10000 (after the arg fix below) |

- Tools first failed (0/10000). It had been built with `seed=42, max_history_len=10` (from the baseline run
  summaries), not with the `20260506 / 50` used by the other three domains.
- The pandas-3 timestamp-unit worry (review pipeline_ops#0, data_integrity#5) does not apply here: pandas 3.0.6
  reproduces the frozen ms keys exactly.

**Reference ranks.** `docs/sigir/ref_ranks/<d>/<method>.csv.gz` holds C-CRP v3 plus the 8 official baselines.
All 36 files cover the frozen event sets exactly.

**Token-channel smoke test** (20 ML-1M users, 389 candidates × {like, dislike}):
- 778 prompts in 3.8 s, about 200 prompts/s with prefix caching, so the 60 prompts/s pre-registered assumption is
  met.
- Yes+No mass ≥ 0.99998 on every prompt.
- Mean logit: like +3.66 (SD 5.66), dislike −5.39 (SD 3.74).
- This is an infrastructure check only; no endpoint was computed.

**dtype probe** (same 389 like-prompts):
- bf16 puts 57% of logit(Yes) − logit(No) values on a 0.25 grid; fp16 gives a 1/32 grid.
- bf16 vs fp16: Pearson 0.9991, Spearman 0.9989, mean |Δ| 0.17, no non-finite values.
- Decision: fp16 (amendment C0).

**Rated-panel eligibility.** All_Beauty has only 37 eligible users (≥3 likes, ≥3 dislikes, ≥3 history), so it is
replaced by Video_Games in Pilot 3 and by Sports in the full grid (amendment P3/D).

**Pre-GPU code review** (ARIS-style adversarial workflow, 5 lenses × 3 skeptics + critic):
- About 45 confirmed findings, including 3 critical:
  - KuaiRec MNAR labels are identically 0 by construction;
  - the pseudonymizer chains swaps and matches substrings, and its placebo skips history-only brands;
  - rated panels leak re-reviewed items' stars.
- 8 findings were rejected.
- Design changes: `idea-stage/PREREG_AMENDMENT_1.md`, dated before any pilot data.

## 2026-10-01 · CORRECTION to P0-CPU (raised by the Phase-4 triage reviewer, re-verified by executor)
The "+23–40% oracle complementarity" below is **mostly a max-of-two-noisy-rankers artifact, not exploitable
complementarity**. Executor re-check (20 random seeds): per-user max of the LLM with a *uniform-random* ranker
already gains **+0.029 / +0.029 / +0.037 / +0.035** NDCG@10 (sports/toys/home/tools, sd ≈ 0.001) — about half
of the LLMEmb "oracle gain". Reviewer-reported (to be re-verified before any paper use): LLM–LLMEmb per-user
NDCG correlation 0.34–0.56 (co-hit 1.7–2.0× chance); tuned rank interpolation recovers only +0.002..+0.012.
The "popularity-structured" pattern is defined on the HELD-OUT target's popularity (not observable at serving
time). ⇒ The LLM↔CF deferral/fusion cluster (A4, A5, A9, B3) is deprioritised; the winner's-curse lesson is
already published (RouteRec 2607.09908) and is at most a paragraph.

## 2026-10-01 · P0-CPU · LLM↔baseline per-user complementarity (no GPU)
- Script: `src/confrec/analysis_complementarity.py` → `outputs/confrec_pilot/complementarity.json`.
- Data: frozen Lumen 1+100 popularity-sampled panels, 10k users each (sports/toys/home/tools); LLM = C-CRP v3
  (verbalized, Qwen3-8B); 8 official baselines' full rankings (`ranking_eval_records.csv`, local).
- Event keys identical across all 9 methods (asserted).

| domain | LLM NDCG@10 | best complement | its NDCG@10 | per-user oracle | gain | base>LLM users |
|---|---|---|---|---|---|---|
| sports | 0.2329 | LLMEmb | 0.1795 | 0.3095 | +0.0766 (+33%) | 19.1% |
| toys | 0.2708 | LLMEmb | 0.2049 | 0.3353 | +0.0645 (+24%) | 15.5% |
| home | 0.1324 | LLMEmb | 0.0939 | 0.1850 | +0.0526 (+40%) | 12.3% |
| tools | 0.1661 | LLMEmb | 0.1159 | 0.2283 | +0.0622 (+37%) | 14.4% |

**Popularity structure of complementarity:** graph/CF-style baselines (IRLLRec, RLMRec, LLM-ESR) beat the
LLM almost exclusively when the target is a HEAD item (12–20% of head users vs <2% of mid/tail users);
LLMEmb's wins are spread over all groups (head 13–21%, tail 9–14%). LLM NDCG@10 is ~flat across
head/mid/tail. ⇒ headroom for a per-user LLM→baseline deferral is large and consistent (+0.05–0.08 NDCG@10,
4/4 domains), and it is popularity-structured. Open question for the GPU pilot: can LLM confidence (token
P(Yes), margin, entropy, consistency) identify the users to defer (prior cheap gate on verbalized p: corr 0.07)?
