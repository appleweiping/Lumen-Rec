"""Amendment 3 addendum 6 (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_6.md, "A3-6") items 2-8 for one domain and one backbone,
and the three A3-6 Holm families across domains (interfaces: docs/sigir/FTEXTRA_IMPL_SPEC.md).

    python -m src.confrec.ftgrid_extra build --domain toys --split outputs/confrec/ftgrid/panels/toys/ftgrid_split.json \
        --panels outputs/confrec/ftgrid/panels/toys --scores_root outputs/confrec/ftgrid/scores \
        --models zeroshot,s0,s1,s2 --raw data/raw --out outputs/confrec/ftgrid/extra/toys.json \
        [--root_label main|llama|teacher] [--pilot_log docs/sigir/PILOT_LOG.md] [--refs CACHE] [--n_boot 2000] [--seed 0]
    python -m src.confrec.ftgrid_extra check_resume <the build arguments>   # exit 0: --out is current, 1: rebuild
    python -m src.confrec.ftgrid_extra summarize --files outputs/confrec/ftgrid/extra/toys.json \
        outputs/confrec/ftgrid/extra/games.json outputs/confrec/ftgrid/extra/sports.json \
        [--ml1m outputs/confrec/ftgrid/extra/ml1m.json] [--llama outputs/confrec/ftgrid_llama/extra/*.json] \
        [--ftc outputs/confrec/ftgrid/extra/ml1m.json] [--ftq outputs/confrec/ftgrid_q/extra/*.json] [--n_boot 2000] \
        [--not_run sports --pilot_log docs/sigir/PILOT_LOG.md] --out outputs/confrec/ftgrid/extra/summary.json

A pure function of stored files: CPU only, deterministic (seed 0, 2,000 user resamples), no clock, host or path in the
output (strict JSON, stats.strict_json, allow_nan=False). `build` writes <out stem>_tables.csv (long format: block, panel,
regime, statistic, estimate, lo, hi, n_users, descriptive, holm_member, then backbone, model, stratum, p, n_pairs,
descriptive_min_n, note) and then --out: the old --out is removed first and the JSON, written last (temporary file and
rename, as the CSV), is the completion marker; it records the CSV's sha1 and the fingerprint of every input (split,
panels, every file of every scoring run directory read, the raw files, the reference cache, the code sha1, the settings and
the A3-6 item 11 record), which `check_resume` compares with the current inputs. Every block that cannot be computed is
{"available": false, "reason": ...}.

--root_label names the output root: main = outputs/confrec/ftgrid (Qwen3-8B; FT_C_reading is the registered ML-1M FT-C
control), llama = outputs/confrec/ftgrid_llama (no control), teacher = outputs/confrec/ftgrid_q (FT_C_reading is the FT-Q
item-only teacher control of addendum 8: the teacher adapters are scored as p0 and p1). FT_C_reading is computed only when
--pilot_log records the sha1 of the three X1 files (src/confrec/ftgrid_extra.py, scripts/sigir/run_ftextra.sh,
tests/test_confrec_ftgrid_extra.py; A3-6 item 11); otherwise it is {"available": false, "reason": "record missing"} and
nothing else changes.

Reuse, not re-derivation. Rows, scores, contexts, references and every registered estimator come from ftgrid_report
(imported read-only, a bound file that is never edited), loaded in the order of ftgrid_report.build: the split and the
panels (make_ctx), the runs (load_run with the E1 integrity rule: a run failing E1 is excluded exactly as in the report,
listed and never replaced), the like logits (to_pairs), the swap prior pi and its donor halves (the report's code), the CF
references (load_refs when the --refs cache identity matches, else compute_refs from --raw, written to --refs when given:
q-hat, n_prior, the temporal MF score, residual, item bias b_i and warm flag are the report's arrays). q-hat_T is never in
the cache: it is computed with forensics.prior_means (the report's function and shrinkage k = 5) on a scan of the raw
events with every candidate time replaced by T_d (references.q_hat_T and meta.q_hat_T_source say so; without --raw E-J's
q-hat_T contrast is unavailable). Raw p-values only: no Holm adjustment inside a domain file (addendum 1 item 8); `summarize` computes the
families.

Blocks (regimes ZS = zeroshot, FT = s0-s2: the seed mean with the CI of the mean, every seed listed, sigma_seed, complete):
  registered_pooled  E-D's shares (shares_block), cross-fitted stackers M0-M3 (stacker_block: pooled UAUCs, G, G_CF),
                     star permutation (starperm_block), E-B (contrast_models) and P1 (p1_block), on the report's rows:
                     the report's registered numbers, reproduced exactly (the paper prints registered and within-user
                     values from one file; summarize copies the E-B / E-D family results from them).
  E_W   (item 2)     the stackers with user-centred features (each feature minus the user's mean over the user's rows of
                     the evaluation set R = E-D's stacker rows) and K = 20 cross-fitting splits stats.user_halves(S_d, k),
                     k = 0..19 (k = 0 is E-D's split); per user each statistic is averaged over the 20 splits; user
                     bootstrap of those per-user values (the coefficients of every split held fixed). UAUC of M0-M3,
                     G_wu = dUAUC(M2 - M1), G_CF_wu = dUAUC(M3 - M0); P1_wu (p1_block's rows and decision rule on the
                     within-user G values); the reading rule (robust / estimator_dependent / ...).
  E_F   (item 3)     M4 = [q-hat, MF residual, pi, e-hat] with E-W's estimator; G_LLM_given_CF = dUAUC(M4 - M3),
                     G_CF_given_LLM = dUAUC(M4 - M2), identical users and rows.
  E_G   (item 4)     item shares: (i) L with pi (= E-D's, shares_block), (ii) the temporal MF score with its item bias b_i
                     (MF warm pairs, reliability 1), (iii) the label with q-hat (reliability 1), each the squared pooled
                     within-user correlation through shares_from_sums; the e-share [Var(L_c - pi_c) - (1 - r8c)
                     Var(pi_c)] / Var(L_c) with r8c re-estimated in every user resample (descriptive).
  E_H   (item 5)     strata sparse / dense (q-hat support n_prior < 5) and unseen / seen (no TRAIN example); per stratum
                     UAUC of L, q-hat, pi, G_prior = dUAUC(M1 - M0) and G_wu = dUAUC(M2 - M1) with E-W's predictors (fit
                     on all rows of the other fold; the stratum restricts the evaluation only).
  E_J   (item 6)     dUAUC(L - q-hat), dUAUC(L - q-hat_T) on E-A's TEST users and rows, dUAUC(L - MF) on the MF warm pairs
                     (contrast_models: the CI and p of E-B).
  E_Cprime (item 7)  E-C's top-k anatomy (topk_anatomy, anatomy_stats, E-C's resamples) with the oracle k and with the
                     deployable k_u = clip(round(r_u n_u), 1, n_u - 1) from the user's CAL like rate (descriptive).
  FT_C_reading (item 8; addendum 8)  retention R = mean_s [UAUC(p_s) - UAUC(ZS)] / [UAUC(s) - UAUC(ZS)], s = 0, 1, on
                     identical TEST users and rows, one user resample driving all UAUCs; ITEM_DRIVEN / USER_DRIVEN / MIXED /
                     NOT_DEFINED; with the UAUC of q-hat on the same rows (addendum 8's descriptive companion). FT-C on the
                     main root (ML-1M only: addendum 8 section 1 withdrew Toys), FT-Q (R_Q) on the teacher root.

`summarize`: the A3-6 Holm families (alpha = 0.05; confirmed = Holm p < 0.05, the sigma_seed rule for FT members and, for
the directional H-F and H-S, an estimate of the hypothesised sign): E-F = {G_LLM_given_CF, FT, per Qwen Amazon panel};
E-H = {G_prior on sparse rows, per Qwen Amazon panel and regime, minus panel-regimes below 150 users}; E-J = {dUAUC(L - q-hat),
FT, per Qwen Amazon panel} (two-sided, the sign reported as found). ML-1M, Llama and the other regime are never members.
Every input is validated and refused (exit 2) rather than dropped (R16). It also copies the registered E-B family (A3
section 11, Holm over the four Qwen domains; p_holm and confirmed are null unless the family is complete), the E-D
families (ftgrid_report.ed_holm_family on each file's registered_pooled E-D block), P1 (ML-1M, Qwen3-8B main root only;
the Llama P1 replication and P1_wu are kept apart), the robust readings beside the registered confirmation, and the
FT-C / FT-Q wording conditions of addendum 8 section 2 (--ftc, --ftq).

Readings. Where the registered text leaves a choice, the most conservative reading is taken; the items below are copied
into every output as `readings` (parsed from this docstring):
R1  Status (A3-6 Consequence; addendum 8). Items 2-7 are exploratory on ML-1M for either backbone (the Qwen ML-1M report
    was read; the weaker label is used for Llama ML-1M too) and registered, outcome-free on every other panel; FT_C_reading
    is registered and outcome-free: FT-C on ML-1M (main root) and FT-Q on every dataset of the teacher root; on the main
    root's other domains FT-C is not registered (addendum 8 section 1 withdrew the Toys permutation run) and the block is
    not computed; ML-1M and Llama values are never members of an A3-6 family.
R2  Arms. Only like, swap and the two star-permutation arms are loaded (ftgrid_report.load_run: E1 integrity, panel,
    variant and adapter identity exactly as the report; an excluded run is listed and never replaced); the nohist,
    pseudo and placebo arms are not read (no block here uses them).
R3  E-W rows. The evaluation set R of a regime is E-D's stacker rows (S_d TEST rows with a finite like logit and pi for
    every model of the regime that has a usable swap arm, and a finite q-hat and MF residual); "the user's mean" of a
    feature is its mean over the user's rows of R (a row without a finite feature cannot enter a mean). P1_wu uses
    p1_block's rows (finite q-hat, L and pi of zeroshot and s0-s2, no MF requirement), as P1 does.
R4  E-W splits. Fold A_k = stats.user_halves(S_d user ids, k), k = 0..19 (A_0 is E-D's fold A); fold A_k's rows are
    scored by the fit on the other fold's rows and vice versa (ftgrid_report.crossfit, logit_fit); a user enters a block
    only if its AUC is defined in every split for every stacker and model of the block (identical users); per user a
    dUAUC is the mean over the 20 splits of the per-split AUC difference and a UAUC the mean of the per-split AUCs. M0
    and M3 have no LLM feature, so their UAUCs (and G_CF_wu) are reported once per regime.
R5  E-W bootstrap. ftgrid_report.mean_draws on the per-user split averages (stats.paired_bootstrap's draws; blocks with
    the same users share the resamples; no stacker is refit in a resample); p and the minimum-n rule are
    ftgrid_report.rec's.
R6  Reading rule (A3-6 item 2). 'robust' needs both estimators (registered pooled and within-user) finite with the same
    non-zero sign, both 95% CIs excluding 0, neither descriptive by the minimum-n rule and, for a fine-tuned mean and
    for P1, all registered seeds present with every seed of both estimators of that sign; otherwise the reading is
    'estimator_dependent' (the signs differ, or exactly one CI excludes 0), 'seed_sign_disagreement' (the means agree, a
    seed does not), 'both_intervals_include_0', 'descriptive_min_n' or 'not_available'.
R7  E-F. M2, M3 and M4 use E-W's rows, splits and users (one key set for M0-M4); the p-value is ftgrid_report.boot_p of
    the paired per-user draws, the definition contrast_models uses.
R8  E-G. (i) L with pi on E-D's shares rows (ftgrid_report.shares_block, unchanged). (ii) and (iii) do not involve the LLM
    and are computed once per panel: (ii) on the S_d TEST pairs that are MF warm (user and item with training data)
    with a finite MF score and b_i; (iii) on the S_d TEST pairs with a finite q-hat. Reliability 1 is entered by giving
    shares_from_sums the item component as both donor halves (r_c = r8c = 1, share = rho^2); n = the users with 2 or
    more pairs (as shares_block). The e-share uses shares_block's rows, sums and resamples (r8c re-estimated in every
    resample). Descriptive: intervals only, no p-value.
R9  E-H strata. sparse = a finite n_prior < 5 (forensics.prior_means: other users' first ratings of the item strictly
    before the candidate), dense = a finite n_prior >= 5 (a row without a finite n_prior is in neither; counted);
    unseen = the item is not a candidate of train.jsonl (E-A secondary ii, ftgrid_report.make_ctx), seen otherwise. The
    strata restrict E-W's rows R; a stratum statistic averages per-user AUCs over the users with both classes among
    their stratum rows; the predictors are E-W's (fit on all rows of R of the other fold).
R10 E-J rows. E-A's TEST rows of the regime (uauc_models' rows: a finite like logit for every model of the regime and a
    finite q-hat, popularity, MF score and MF residual); dUAUC(L - q-hat_T) on those rows with a finite q-hat_T (the
    number without one is reported); 'MF' is the temporal MF score (E-A's reference 'mf'), on those rows that are MF
    warm.
R11 q-hat_T = forensics.prior_means(scan_events(raw), users, items, times = T_d for every pair, shrink_k = 5)
    ['mean_prior_shrunk']: other users' first ratings strictly before T_d, the global shrinkage prior also before T_d;
    q-hat recomputed from the same scan at the candidate times must equal the reference array (recorded in the output).
R12 E-C'. Rows and users are the oracle anatomy's (users with both classes among the regime's TEST rows with finite
    logits for every model), so both decisions are compared on identical rows; r_u = the like rate among all of the
    user's CAL rows (labels, whatever the model's censoring); fewer than 3 CAL rows: the pooled like rate of all CAL rows
    of the EVAL panel; round = half up, floor(r_u n_u + 1/2) (as addendum 4 item 1), in exact integer arithmetic; the
    resamples are E-C's (ftgrid_report.ec_block's cluster draws), so the oracle values equal the report's E-C anatomy.
R13 FT-C / FT-Q (DEVIATION: stricter than the registered text, for the paper's deviations table). R is defined only when
    the registered E-B contrast (s0-s2 vs ZS on TEST) is positive with a CI excluding 0 (registered) and, beyond the
    registered text, E-B is complete (all 3 seeds; addendum 1 item 7) and the FT-C rows hold at least 150 users (the
    minimum-n rule of A3 section 1 applied to R's CI-based label); otherwise R is not reported and the label is
    NOT_DEFINED. A resample with a zero denominator gives a non-finite R, dropped from the percentile CI and counted.
R14 summarize, families. H-F and H-S are directional: a member is confirmed only with an estimate > 0; H-J is two-sided.
    FT members need the sigma_seed rule (all registered seeds present, all seeds the sign of the mean, |mean| > 2
    sigma_seed). The members are the panel runs and regimes that were run: a panel whose fine-tuned runs were not run (a
    registered cut of A3 section 10: every FT model s0-s2 without a scoring run) has no FT member in E-F, E-H or E-J,
    but only with --not_run <dataset> and a pilot-log line holding the exact token 'FTEXTRA_NOT_RUN <dataset>'; without
    them such a file is refused. Its ZS regime was run, so its ZS member of E-H stays ('per Qwen Amazon panel run and
    regime'); a dataset cut as a whole has no file and no member. The registered minimum-n rule takes a member out of
    a family; a member that was run but whose statistic is unavailable (its runs excluded, e.g. by E1) stays in the
    family with p = 1 (flagged, never confirmed); an incomplete FT member keeps its p-value and is never confirmed.
R15 summarize, registered copies. The registered E-B / E-D results are copied through by applying the report's own
    functions (ftgrid_report.holm, ed_holm_family; the E-B family as fill_paper.eb_family computes it) to the files'
    registered_pooled blocks, which equal the report's blocks (tested). The E-B copy has p_holm and confirmed null unless
    ML-1M, Toys, Video_Games and Sports are all given with a complete, non-descriptive E-B (fill_paper.eb_family refuses
    otherwise). P1 is copied from the --ml1m file only (ML-1M, Qwen3-8B, main root); the Llama P1 replication goes to
    llama.P1_replication and P1_wu to sensitivity.P1_wu, never under registered.P1.
R16 summarize, inputs. Every file is validated and a bad one is refused (exit 2), never dropped: the schema; a known
    backbone (Qwen or Llama); empty input_checks.problems; n_boot = --n_boot (2,000 for a registered summary) and seed 0;
    the code sha1 of ftgrid_extra.py equal to the current file; the root and role: --files = main-root Qwen files of
    toys / games / sports eligible for the families, --ml1m and --ftc = the main-root Qwen ML-1M file, --llama =
    Llama-root files, --ftq = teacher-root Qwen files; models_requested holding the registered models of the role
    (zeroshot, s0-s2; with p0 and p1 for --ftc and --ftq); no (root, domain) twice. A build itself refuses a
    --root_label that differs from the registered root its --scores_root resolves into.
R17 robust vs registered_confirmed. `robust` is the A3-6 item 2 reading rule on the unadjusted 95% intervals of the two
    estimators, not a Holm confirmation; `registered_confirmed` beside it is the registered estimate's confirmation: P1's
    decision for P1 (ML-1M, Qwen3-8B, main root; null elsewhere), the E-D Holm family for G (null in a domain file, which
    holds raw p-values only, and filled by summarize), none for G_CF (a reference in no family).
R18 FT wording (addendum 8 section 2). fine_tuning_mostly_teaches_the_item is true iff FT-C reads ITEM_DRIVEN on ML-1M and
    FT-Q reads ITEM_DRIVEN on every dataset given (ML-1M and Toys required, addendum 8 section 3; the runner refuses a
    teacher-root dataset that was scored without its file); null with the reason when an input is missing or its block
    is unavailable (never true, and never decided, on partial input); NOT_DEFINED is a defined label that is not
    ITEM_DRIVEN. item_quality_from_item_text additionally needs FT-Q ITEM_DRIVEN on at least one Amazon dataset;
    evidence_mixed is true iff a label is USER_DRIVEN or MIXED (null on partial input).
R19 Integrity. CF references or a q-hat_T that cannot be computed although --raw is given, a q-hat recomputed from the
    raw scan that differs from the reference array, a backbone that does not match the root, or an unknown backbone is a
    problem (input_checks.problems), so summarize refuses the file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.confrec import forensics as fx
from src.confrec import ftgrid_report as fr
from src.confrec.stats import strict_json, user_halves

NAN = float("nan")
SPEC = ("idea-stage/PREREG_AMENDMENT_3_ADDENDUM_6.md items 2-8 and its Holm families (with idea-stage/PREREG_AMENDMENT_3.md "
        "sections 1, 3, 11 and addendum 1); docs/sigir/FTEXTRA_IMPL_SPEC.md")
K_SPLITS = 20                        # A3-6 item 2: K = 20 user-half splits stats.user_halves(S_d, k), k = 0..19
SPARSE_N_PRIOR = 5                   # A3-6 item 5: sparse = q-hat computed from fewer than k = 5 other-user ratings
ECP_MIN_CAL_ROWS = 3                 # A3-6 item 7: fewer than 3 CAL rows -> the pooled CAL like rate of the EVAL users
RETENTION_CUT = 0.5                  # A3-6 item 8: ITEM_DRIVEN iff lo(R) > 0.5, USER_DRIVEN iff hi(R) < 0.5
ALPHA = 0.05
ARMS = ("like", "swap", "starperm0", "starperm1")    # the scoring arms these blocks read (ftgrid_report.ARM_PANEL)
REGS = ("ZS", "FT")
STACKERS_X = {**fr.STACKERS, "M4": ("q_hat", "mf_residual", "pi", "e_hat")}
SHARED = ("M0", "M3")                # stackers without an LLM feature: one fit per split, shared by a regime's models
AMAZON = ("toys", "games", "sports")
FTC_MODELS = ("zeroshot", "s0", "s1", "p0", "p1")
REF_ARRAYS = ("mf_item_bias", "mf_residual", "mf_score", "mf_warm", "n_prior", "q_hat")   # of ftgrid_report.compute_refs
CSV_COLS = ("block", "panel", "regime", "statistic", "estimate", "lo", "hi", "n_users", "descriptive", "holm_member",
            "backbone", "model", "stratum", "p", "n_pairs", "descriptive_min_n", "note")
ROOT_LABELS = ("main", "llama", "teacher")      # outputs/confrec/ftgrid, ftgrid_llama, ftgrid_q (addendum 8)
REGISTERED_ROOTS = {"main": "outputs/confrec/ftgrid", "llama": "outputs/confrec/ftgrid_llama",
                    "teacher": "outputs/confrec/ftgrid_q"}
FT_MODELS = ("s0", "s1", "s2")
ROLE_MODELS = {"files": ("zeroshot",) + FT_MODELS, "ml1m": ("zeroshot",) + FT_MODELS, "llama": ("zeroshot",) + FT_MODELS,
               "ftc": ("zeroshot",) + FT_MODELS + ("p0", "p1"), "ftq": ("zeroshot",) + FT_MODELS + ("p0", "p1")}
NOT_RUN_TOKEN = "FTEXTRA_NOT_RUN"                # pilot-log record of a registered cut of a dataset's FT runs
X1_FILES = ("src/confrec/ftgrid_extra.py", "scripts/sigir/run_ftextra.sh", "tests/test_confrec_ftgrid_extra.py")
CODE_FILES = ("src/confrec/ftgrid_extra.py", "src/confrec/ftgrid_report.py", "src/confrec/forensics.py",
              "src/confrec/stats.py", "src/confrec/metrics.py")
REGISTERED_N_BOOT = fr.N_BOOT                    # A3 section 3: 2,000 user resamples, seed 0
P1_WU_VERDICT = {"P1_HOLDS": "P1_WU_HOLDS", "NO_EVIDENCE": "P1_WU_DOES_NOT_HOLD", "INCOMPLETE": "P1_WU_INCOMPLETE"}
ROBUST_DESCRIPTION = ("robust = the A3-6 item 2 reading rule on the UNADJUSTED 95% intervals of the registered pooled and "
                      "the within-user estimator (same sign, both intervals excluding 0, all seeds agreeing for a fine-tuned "
                      "mean); it is not a Holm confirmation. registered_confirmed = the registered estimate's confirmation: "
                      "P1's decision for P1 (ML-1M, Qwen3-8B, main root), the E-D Holm family for G (decided by summarize: "
                      "a domain file holds raw p-values only), none for G_CF")

def _readings(doc: str | None) -> tuple:
    """The R<n> items of the module docstring's Readings section, one string each (label kept)."""
    if not doc or "\nReadings." not in doc:
        return ("see the module docstring of src/confrec/ftgrid_extra.py",)
    items = []
    for line in doc.split("\nReadings.", 1)[1].splitlines():
        if re.match(r"^R\d+\s", line):
            items.append(" ".join(line.split()))
        elif items and line.startswith("    ") and line.strip():
            items[-1] += " " + " ".join(line.split())
    return tuple(items)


READINGS = _readings(__doc__)


# ---------------------------------------------------------------- small helpers
def _num(x) -> float:
    return fr._num(x)


def _na(reason: str) -> dict:
    return {"available": False, "reason": reason}


def _is_qwen(backbone) -> bool:
    return bool(backbone) and "qwen" in str(backbone).lower()


def _sha1_file(path: Path) -> str | None:
    return fx.file_sha1(path) if path.is_file() else None


def _family_of(backbone) -> str | None:
    """'qwen' / 'llama' for the two registered backbones, None for a missing or unknown one."""
    b = str(backbone or "").lower()
    return "qwen" if "qwen" in b else "llama" if "llama" in b else None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canon(p) -> Path:
    """Canonical path: symlinks and junctions resolved, '..' removed, case folded where the filesystem folds it."""
    return Path(os.path.normcase(os.path.realpath(str(p))))


def root_label_of(scores_root) -> str | None:
    """The label of the registered root whose directory holds `scores_root` (canonical paths), None elsewhere."""
    s = _canon(scores_root)
    for label, rel in REGISTERED_ROOTS.items():
        r = _canon(repo_root() / rel)
        if s == r or r in s.parents:
            return label
    return None


def check_root_label(a) -> None:
    """A build reading a registered root's scores must carry that root's label (a teacher build labelled main, or the
    converse, is refused before anything is read); scores outside the registered roots (rehearsals) take any label."""
    lab = root_label_of(a.scores_root)
    if lab is not None and lab != a.root_label:
        print(f"refused: --root_label {a.root_label}, but --scores_root lies in the {lab} root ({REGISTERED_ROOTS[lab]})",
              file=sys.stderr)
        raise SystemExit(2)


def code_sha1() -> dict:
    """sha1 of this module and of the bound modules it computes with."""
    root = repo_root()
    return {Path(rel).name: _sha1_file(root / rel) for rel in CODE_FILES}


def x1_record(pilot_log) -> dict:
    """A3-6 item 11: are the sha1 of the three X1 files (this module, its runner, its unit tests) in the pilot log?
    (Case-insensitive substring, as ftgrid_freeze checks.) complete is False without --pilot_log."""
    root = repo_root()
    sha = {rel: _sha1_file(root / rel) for rel in X1_FILES}
    if not pilot_log:
        return {"pilot_log_given": False, "sha1": sha, "recorded": {rel: False for rel in X1_FILES}, "complete": False}
    try:
        log = Path(pilot_log).read_text(encoding="utf-8").lower()
    except OSError:
        log = None
    rec = {rel: bool(log is not None and sha[rel] and sha[rel].lower() in log) for rel in X1_FILES}
    return {"pilot_log_given": True, "pilot_log_readable": log is not None, "sha1": sha, "recorded": rec,
            "complete": all(rec.values())}


def _source_of(panels: Path, domain: str) -> str:
    """The panel's `source` (first row of eval.jsonl), as ftgrid_report.build reads it."""
    try:
        with open(Path(panels) / fr.ARM_PANEL["like"], encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return str(json.loads(line).get("source") or domain)
    except (OSError, ValueError):
        pass
    return domain


def raw_files(raw, source: str) -> list:
    """The raw files that forensics.load_raw_events and compute_refs' description lengths read for this source."""
    raw = Path(raw)
    if source == "ml1m":
        d = raw if (raw / "ratings.dat").exists() else raw / "ml-1m"
        return [d / "movies.dat", d / "ratings.dat"]
    try:
        from src.confrec.build_rated_panels import amazon_paths
        return list(amazon_paths(raw, source))
    except KeyError:
        return []


def input_fingerprint(a) -> dict:
    """Everything a build reads, by content: the settings, the code sha1, the split, the panels, every file of every
    scoring run directory of the requested models and arms, the raw files, the reference cache and the A3-6 item 11
    record. A stored build is resumed only when its fingerprint equals the current one (check_resume)."""
    split = Path(a.split)
    panels = Path(a.panels)
    models = [m for m in a.models.split(",") if m]
    try:
        sd_name = (fr.read_json(split).get("sd") or {}).get("user_ids_path") or "sd_users.txt"
    except (OSError, ValueError, AttributeError):
        sd_name = "sd_users.txt"
    names = sorted({fr.ARM_PANEL[x] for x in ARMS} | {"train.jsonl", "sd_users.txt", "eval_users.txt", sd_name})
    sdir = Path(a.scores_root) / a.domain
    sdir = sdir if sdir.is_dir() else Path(a.scores_root)
    runs = {}
    for m in models:
        for arm in ARMS:
            d = sdir / m / arm
            runs[f"{m}/{arm}"] = ({p.name: fx.file_sha1(p) for p in sorted(d.iterdir()) if p.is_file()}
                                  if d.is_dir() else None)
    source = _source_of(panels, a.domain)
    raw = {}
    if a.raw:
        for p in raw_files(a.raw, source):
            try:
                key = p.relative_to(Path(a.raw)).as_posix()
            except ValueError:
                key = p.name
            raw[key] = _sha1_file(p)
    rec = x1_record(getattr(a, "pilot_log", None))
    return {"domain": a.domain, "root_label": getattr(a, "root_label", "main"), "models": models,
            "n_boot": int(a.n_boot), "seed": int(a.seed), "code_sha1": code_sha1(), "split_sha1": _sha1_file(split),
            "panels": {n: _sha1_file(panels / n) for n in names}, "runs": runs, "raw_source": source, "raw": raw,
            "refs_cache_sha1": _sha1_file(Path(a.refs)) if getattr(a, "refs", None) else None,
            "x1_record": {"recorded": rec["recorded"], "complete": rec["complete"], "sha1": rec["sha1"]}}


# ---------------------------------------------------------------- inputs (the order of ftgrid_report.build)
def load_inputs(a) -> SimpleNamespace:
    split = fr.read_json(a.split)
    T = float(split["T"])
    panels = Path(a.panels)
    models = [m for m in a.models.split(",") if m]
    unknown = [m for m in models if m not in fr.MODELS]
    if unknown:
        raise SystemExit(f"--models: unknown {unknown} (choose from {list(fr.MODELS)})")
    if len(set(models)) != len(models):
        raise SystemExit("--models repeats a model")
    eval_path = panels / fr.ARM_PANEL["like"]
    if not eval_path.is_file():
        raise SystemExit("--panels: no eval.jsonl")
    eval_rows = fx.read_jsonl(eval_path)
    files = split.get("files") or {}
    problems, checks, panel_sha = [], {}, {}
    for name in sorted({fr.ARM_PANEL[x] for x in ARMS} | {"train.jsonl", "sd_users.txt", "eval_users.txt"}):
        p = panels / name
        panel_sha[name] = fx.file_sha1(p) if p.is_file() else None
        if name in files and panel_sha[name] is not None and files[name] != panel_sha[name]:
            problems.append(f"{name}: sha1 {panel_sha[name]} differs from ftgrid_split.json files ({files[name]})")
    eval_ids = [str(r["user_id"]) for r in eval_rows]
    checks["eval_user_ids_sha1_match"] = (None if not (split.get("eval") or {}).get("user_ids_sha1")
                                          else fr.sha1_text_lines(eval_ids) == split["eval"]["user_ids_sha1"])
    sd_path = panels / ((split.get("sd") or {}).get("user_ids_path") or "sd_users.txt")
    sd_test_path = panels / fr.ARM_PANEL["swap"]
    if sd_path.is_file():
        sd_ids = [x for x in sd_path.read_bytes().decode("utf-8").split("\n") if x]
        checks["sd_user_ids_sha1_match"] = (None if not (split.get("sd") or {}).get("user_ids_sha1")
                                            else fx.file_sha1(sd_path) == split["sd"]["user_ids_sha1"])
    elif sd_test_path.is_file():
        sd_ids = [str(r["user_id"]) for r in fx.read_jsonl(sd_test_path)]
        checks["sd_user_ids_source"] = "eval_sd_test.jsonl (sd_users.txt absent)"
    else:
        sd_ids = []
        problems.append("no sd_users.txt and no eval_sd_test.jsonl: the S_d blocks cannot run")
    train_path = panels / "train.jsonl"
    train_items = ({str(i) for r in fx.read_jsonl(train_path) for i in r["candidate_item_ids"]}
                   if train_path.is_file() else None)
    cx = fr.make_ctx(eval_rows, T, sd_ids, train_items)
    if cx.n_duplicate_pairs:
        problems.append(f"eval.jsonl repeats {cx.n_duplicate_pairs} (user_id, item_id) pairs")
    if (~(cx.test | cx.cal)).any():
        problems.append(f"{int((~(cx.test | cx.cal)).sum())} EVAL pairs without a finite candidate timestamp")
    if sd_test_path.is_file():
        sdp = {(str(r["user_id"]), str(i)) for r in fx.read_jsonl(sd_test_path) for i in r["candidate_item_ids"]}
        mine = {(u, i) for u, i, s in zip(cx.users.tolist(), cx.item.tolist(), cx.sd_test.tolist()) if s}
        checks["eval_sd_test_pairs_equal_sd_test_rows"] = sdp == mine
        if sdp != mine:
            problems.append("eval_sd_test.jsonl pairs differ from S_d's TEST rows of eval.jsonl")
    both = set(fr.user_aucs(np.zeros(cx.n), cx.y, cx.users, cx.test))
    checks["sd_users_are_eval_users_with_both_classes_in_test"] = all(u in both for u in sd_ids)
    if not checks["sd_users_are_eval_users_with_both_classes_in_test"]:
        problems.append("an S_d user is not an EVAL user with both classes among the TEST rows")

    # ---- runs (E1 integrity and the identity checks of ftgrid_report.load_run)
    sdir = Path(a.scores_root) / a.domain
    sdir = sdir if sdir.is_dir() else Path(a.scores_root)
    expected = {"like": np.ones(cx.n, bool), "swap": cx.sd_test, "starperm0": cx.sd_test, "starperm1": cx.sd_test}
    runs, excluded = {}, []
    for m in models:
        runs[m] = {}
        for arm in ARMS:
            r = fr.load_run(sdir / m / arm, f"{m}/{arm}", panel_sha.get(fr.ARM_PANEL[arm]), m, split.get("variant"),
                            need_swap=(arm == "swap"))
            runs[m][arm] = r
            if r["status"] not in ("OK", "ABSENT"):
                excluded.append({"model": m, "arm": arm, "status": r["status"], "reason": r["reason"]})
    backbones = sorted({r.get("backbone") for m in models for r in runs[m].values()
                        if r["status"] == "OK" and r.get("backbone")})
    if len(backbones) > 1:
        problems.append(f"runs of several backbones in one file: {backbones}")
    adapter_of = {}
    for m in models:
        ids = {r.get("adapter_id") for r in runs[m].values() if r["status"] == "OK"}
        if len(ids) > 1:
            problems.append(f"{m}: its arms were scored with different adapters (report.json lora)")
        adapter_of[m] = sorted(i for i in ids if i)
    shared = Counter(i for v in adapter_of.values() for i in v)
    checks["adapters_distinct_across_models"] = not any(c > 1 for c in shared.values())
    if not checks["adapters_distinct_across_models"]:
        problems.append("two models were scored with the same adapter (report.json lora)")

    # ---- per-model vectors (ftgrid_report.build's code)
    L, PI, PA, PB, LP, joins, consistency = {}, {}, {}, {}, {}, defaultdict(dict), {}
    for m in models:
        rl = runs[m]["like"]
        if rl["status"] == "OK":
            L[m], joins[m]["like"] = fr.to_pairs(rl["sc"], cx, expected["like"])
        rs = runs[m]["swap"]
        if rs["status"] == "OK" and m in L:
            pos = rs["swap"]["pos"]
            per_item = {i: (fr._fmean(v), fr._fmean(v[fx.HALF_A]), fr._fmean(v[fx.HALF_B])) for i, v in pos.items() if v}
            got = [per_item.get(i, (NAN, NAN, NAN)) for i in cx.item.tolist()]
            PI[m], PA[m], PB[m] = (np.where(cx.sd_test, np.array([g[k] for g in got], float), NAN) for k in range(3))
            own, joins[m]["swap"] = fr.to_pairs(rs["sc"], cx, expected["swap"])
            ok = np.isfinite(own) & np.isfinite(L[m])
            consistency[m] = {"n_pairs": int(ok.sum()),
                              "max_abs_diff": float(np.abs(own[ok] - L[m][ok]).max()) if ok.any() else NAN}
        if all(runs[m][k]["status"] == "OK" for k in fr.STARPERM_ARMS):
            LP[m] = []
            for k in fr.STARPERM_ARMS:
                v, joins[m][k] = fr.to_pairs(runs[m][k]["sc"], cx, expected[k])
                LP[m].append(v)
    present = set(L)
    for m in models:
        for r in runs[m].values():
            r.pop("sc", None)
            r.pop("swap", None)

    # ---- CF references (ftgrid_report.build's rule: the cache when its identity matches, else --raw)
    source = str(eval_rows[0].get("source") or a.domain) if eval_rows else a.domain
    ident = fr.refs_identity(a.domain, panel_sha.get("eval.jsonl"), T, source, a.n_boot, a.seed)
    refs, refs_info = None, {}
    # --refs follows ftgrid_report: a mismatching cache is rewritten from --raw (the bound report's behaviour); the
    # runner (run_ftextra.sh) never passes --refs
    rp = Path(a.refs) if a.refs else None
    if rp is not None and rp.is_file():
        refs, why = fr.load_refs(rp, ident, cx)
        lacking = sorted(set(REF_ARRAYS) - set((refs or {}).get("arrays") or {})) if refs is not None else []
        if lacking:                    # a needed array not in the cache: computed with the report's function instead
            refs, why = None, f"the cache lacks {lacking}"
        refs_info = {"source": "cache (--refs)"} if refs is not None else {"cache_rejected": why}
    if refs is None and a.raw:
        try:
            refs = fr.compute_refs(cx, a.raw, source, T, a.n_boot, a.seed)
            refs_info["source"] = "computed from --raw (ftgrid_report.compute_refs)"
            if rp is not None:
                fr.save_refs(rp, ident, cx, refs)
                refs_info["cache_written"] = True
        except (OSError, KeyError, ValueError) as e:
            refs_info["error"] = f"{type(e).__name__}: {e}"
    if refs is None:
        refs_info.setdefault("reason", "no CF references: give --raw (or a matching --refs cache)")
    else:
        refs_info["arrays_used"] = list(REF_ARRAYS)
    q_T, qT_info = None, {}
    if refs is None:
        qT_info = {"available": False, "reason": "no CF references"}
    elif not a.raw:
        qT_info = {"available": False, "reason": "q-hat_T is not in the reference cache and needs --raw"}
    else:
        try:
            q_T, qT_info = q_hat_T(cx, a.raw, source, T, refs["arrays"]["q_hat"])
        except (OSError, KeyError, ValueError) as e:
            qT_info = {"available": False, "reason": f"{type(e).__name__}: {e}"}
    if refs is None and a.raw:
        problems.append("the CF references could not be computed although --raw is given: "
                        f"{refs_info.get('error') or refs_info.get('cache_rejected') or refs_info.get('reason')}")
    if q_T is not None and qT_info.get("q_hat_recomputed_equals_reference") is not True:
        problems.append("q-hat recomputed from the raw scan differs from the reference array (max abs diff "
                        f"{qT_info.get('q_hat_recomputed_max_abs_diff')}): the references are not the report's")
    if refs is not None and a.raw and q_T is None:
        problems.append(f"q-hat_T could not be computed although --raw is given: {qT_info.get('reason')}")
    bb = backbones[0] if len(backbones) == 1 else None
    root_label = getattr(a, "root_label", "main")
    fam = _family_of(bb)
    if bb is None or fam is None:
        problems.append(f"backbone missing or unknown ({backbones}): neither Qwen nor Llama")
    elif (root_label == "llama") != (fam == "llama"):
        problems.append(f"backbone {bb} does not belong to the {root_label} root")
    role = ("P1 confirmatory (ML-1M, Qwen3-8B)" if a.domain == "ml1m" and _is_qwen(bb) else
            "replication of P1's sign (Amendment 2 F)" if bb is not None else "backbone unknown")
    return SimpleNamespace(a=a, split=split, T=T, models=models, cx=cx, sd_ids=list(sd_ids), runs=runs,
                           excluded=excluded, problems=problems, checks=checks, panel_sha=panel_sha, L=L, PI=PI, PA=PA,
                           PB=PB, LP=LP, joins={m: dict(v) for m, v in joins.items()}, consistency=consistency,
                           present=present, refs=refs, refs_info=refs_info, q_T=q_T, qT_info=qT_info, backbone=bb,
                           backbones=backbones, role=role, n_boot=a.n_boot, seed=a.seed, root_label=root_label,
                           domain=a.domain, x1=x1_record(getattr(a, "pilot_log", None)))


def q_hat_T(cx, raw, source: str, T: float, q_ref) -> tuple[np.ndarray, dict]:
    """A3-6 item 6: the information-matched item mean, forensics.prior_means with every candidate time replaced by T_d
    (k = 5, the report's shrinkage); q-hat at the candidate times from the same scan is compared with the reference."""
    events = fx.load_raw_events(raw, source)
    users, items = cx.E["user"].tolist(), cx.E["item"].tolist()
    scan = fx.scan_events(events, set(items), set(zip(users, items)))
    pm_T = fx.prior_means(scan, users, items, np.full(len(users), float(T)), fr.SHRINK_K)
    pm_t = fx.prior_means(scan, users, items, cx.ts, fr.SHRINK_K)
    a, b = np.asarray(pm_t["mean_prior_shrunk"], float), np.asarray(q_ref, float)
    both = np.isfinite(a) & np.isfinite(b)
    info = {"available": True, "source": "computed from --raw: forensics.prior_means(times = T_d, shrink_k = 5); "
                                         "not in the reference cache",
            "n_pairs_finite": int(np.isfinite(pm_T["mean_prior_shrunk"]).sum()),
            "n_test_pairs_finite": int((np.isfinite(pm_T["mean_prior_shrunk"]) & cx.test).sum()),
            "q_hat_recomputed_equals_reference": bool(np.array_equal(np.isfinite(a), np.isfinite(b))
                                                      and (not both.any() or np.abs(a[both] - b[both]).max() <= 1e-12)),
            "q_hat_recomputed_max_abs_diff": float(np.abs(a[both] - b[both]).max()) if both.any() else NAN}
    return np.asarray(pm_T["mean_prior_shrunk"], float), info


# ---------------------------------------------------------------- shared machinery
def ed_rows(cx, ms: list, msw: list, L: dict, PI: dict) -> np.ndarray:
    """E-D's identical rows of a regime (ftgrid_report.build): S_d TEST rows with L (and pi, when the regime has swap
    arms) finite for every model."""
    D = cx.sd_test.copy()
    for m in msw or ms:
        D &= np.isfinite(L[m]) & (np.isfinite(PI[m]) if msw else True)
    return D


def split_folds(cx, k_splits: int = K_SPLITS) -> list:
    """Fold A_k = stats.user_halves(S_d, k) per pair, k = 0..K-1 (A_0 is make_ctx's fold_a, E-D's split)."""
    users = cx.users.tolist()
    out = []
    for k in range(k_splits):
        ids = user_halves(cx.sd_ids, k)
        out.append(np.array([u in ids for u in users], bool))
    return out


def wu_fit(cx, feats: dict, stackers: dict, rows, folds: list) -> dict:
    """E-W's cross-fitted linear predictors: {stacker: [eta_k, k = 0..K-1]}; every feature centred over the user's
    `rows` (ftgrid_report.user_centred), each split's fold A_k scored by the fit on the other fold and vice versa
    (ftgrid_report.crossfit: logit_fit with intercept, standardised features, L2 1e-4 on the slopes)."""
    rows = np.asarray(rows, bool)
    used = sorted({c for cols in stackers.values() for c in cols})
    C = {f: fr.user_centred(feats[f], cx.uc, rows) for f in used}
    return {s: [fr.crossfit(np.column_stack([C[c] for c in cols]), cx.y, fa, rows) for fa in folds]
            for s, cols in stackers.items()}


def regime_etas(cx, models: list, L: dict, PI: dict, q, mfres, rows, folds: list,
                own: tuple = ("M1", "M2", "M4"), shared: tuple = SHARED) -> tuple[dict, dict]:
    """(shared {stacker: [eta_k]}, {model: {stacker: [eta_k]}}) for a regime's models on `rows`."""
    common = {"q_hat": q, "mf_residual": mfres}
    sh = wu_fit(cx, common, {s: STACKERS_X[s] for s in shared}, rows, folds) if shared else {}
    per = {}
    for m in models:
        f = {**common, "pi": PI[m], "e_hat": L[m] - PI[m]}
        per[m] = wu_fit(cx, f, {s: STACKERS_X[s] for s in own}, rows, folds)
    return sh, per


def auc_table(scores: dict, y, users, rows) -> tuple[list, dict]:
    """{key: [score array, ...]} -> (users, {key: (n_users, len(list)) per-user AUCs over `rows`}); users = those with
    an AUC for every score of every key (both classes among their rows; every score is finite on the rows)."""
    per = {k: [fr.user_aucs(s, y, users, rows) for s in lst] for k, lst in scores.items()}
    sets = [set(d) for lst in per.values() for d in lst]
    keys = sorted(set.intersection(*sets)) if sets else []
    return keys, {k: np.array([[d[u] for d in lst] for u in keys], float).reshape(len(keys), len(lst))
                  for k, lst in per.items()}


class Cols:
    """Per-user columns of one block, bootstrapped together (shared draws: ftgrid_report.mean_draws)."""

    def __init__(self, n: int):
        self.n, self.cols, self.idx = n, [], {}

    def add(self, key, vec) -> None:
        self.idx[key] = len(self.cols)
        self.cols.append(np.asarray(vec, float).reshape(self.n))

    def draws(self, n_boot: int, seed: int) -> None:
        self.V = np.column_stack(self.cols) if self.cols else np.zeros((self.n, 0))
        self.V = self.V.reshape(self.n, len(self.cols))
        self.D = fr.mean_draws(self.V, n_boot, seed)

    def one(self, key, n_pairs, contrast: bool = False, **extra) -> dict:
        j = self.idx[key]
        return fr.rec(self.V[:, j].mean() if self.n else NAN, self.D[:, j], self.n, n_pairs, contrast=contrast, **extra)

    def models(self, keys: dict, n_pairs, n_reg: int, contrast: bool) -> dict:
        """keys = {model: column key}: per model, the mean over the models (seeds) and sigma_seed."""
        names = list(keys)
        per = {m: self.one(keys[m], n_pairs, contrast) for m in names}
        j = [self.idx[keys[m]] for m in names]
        mean = fr.rec(self.V[:, j].mean(1).mean() if self.n and j else NAN, self.D[:, j].mean(1) if j else None,
                      self.n, n_pairs, contrast=contrast)
        return {"per_model": per, "mean_over_seeds": mean,
                "seeds": fr.seed_summary([per[m]["est"] for m in names], n_reg, contrast, mean["est"])}


def robust_reading(reg: dict | None, wu: dict | None, reg_seeds: list | None = None, wu_seeds: list | None = None,
                   fine_tuned: bool = False, n_registered: int = 3, registered_confirmed: bool | None = None,
                   confirmed_source: str = "") -> dict:
    """A3-6 item 2 reading rule (most conservative reading, R6), with the registered confirmation beside it (R17)."""
    conf = {"registered_confirmed": registered_confirmed, "registered_confirmed_source": confirmed_source,
            "description": ROBUST_DESCRIPTION}
    if not isinstance(reg, dict) or not isinstance(wu, dict) or "est" not in reg or "est" not in wu:
        return {"reading": "not_available", "robust": False, **conf}
    e1, e2 = _num(reg.get("est")), _num(wu.get("est"))
    s1 = int(np.sign(e1)) if math.isfinite(e1) else 0
    s2 = int(np.sign(e2)) if math.isfinite(e2) else 0
    ex1, ex2 = reg.get("ci_excludes_0") is True, wu.get("ci_excludes_0") is True
    desc = bool(reg.get("descriptive_min_n") or wu.get("descriptive_min_n"))
    same = s1 != 0 and s1 == s2
    out = {"registered_est": e1, "within_user_est": e2, "same_sign": same, "registered_ci_excludes_0": ex1,
           "within_user_ci_excludes_0": ex2, "descriptive_min_n": desc}
    seeds_ok = True
    if fine_tuned:
        v1, v2 = [_num(x) for x in (reg_seeds or [])], [_num(x) for x in (wu_seeds or [])]
        seeds_ok = bool(len(v1) == n_registered and len(v2) == n_registered and s2 != 0
                        and all(math.isfinite(x) and int(np.sign(x)) == s2 for x in v1 + v2))
        out["all_seeds_same_sign_both_estimators"] = seeds_ok
    robust = (not desc) and same and ex1 and ex2 and seeds_ok
    if desc:
        reading = "descriptive_min_n"
    elif robust:
        reading = "robust"
    elif not ex1 and not ex2:
        reading = "both_intervals_include_0"
    elif ex1 and ex2 and same:
        reading = "seed_sign_disagreement"
    else:
        reading = "estimator_dependent"
    out.update(robust=bool(robust), reading=reading, **conf)
    return out


# ---------------------------------------------------------------- registered_pooled (the equality block)
def registered_pooled(X, regimes: dict) -> dict:
    """E-D's shares, stackers and star permutation, E-B and P1 through the report's own functions on the report's rows."""
    cx, L, PI, PA, PB, LP, refs = X.cx, X.L, X.PI, X.PA, X.PB, X.LP, X.refs
    boot = (X.n_boot, X.seed)
    out = {"definition": "the registered E-D numbers (ftgrid_report.shares_block, stacker_block, starperm_block), E-B "
                         "(contrast_models) and P1 (p1_block) re-derived from the same files with the report's functions, "
                         "rows and resamples: equal to the report's E_D, E_B and P1 blocks; no Holm adjustment here "
                         "(summarize applies ftgrid_report.ed_holm_family)", "E_D": {}}
    for reg in REGS:
        info = regimes[reg]
        ms, nreg = info["models"], info["n_registered"]
        if not ms:
            out["E_D"][reg] = _na(f"no usable like arm for the {reg} models {list(fr.REGIMES[reg])}"
                                  if info["missing_or_excluded"] else f"{reg} models not requested")
            continue
        ed = {"models": ms, "missing_or_excluded": info["missing_or_excluded"], "complete": len(ms) == nreg}
        msw = [m for m in ms if m in PI]
        if not X.sd_ids:
            ed.update(_na("no S_d"))
        else:
            ed["swap_models"] = msw
            D = ed_rows(cx, ms, msw, L, PI)
            if msw:
                ed["shares"] = fr.shares_block(cx.uc, D, {m: (PA[m], PB[m], L[m], PI[m]) for m in msw}, *boot, nreg)
                ed["information_gain"] = (fr.stacker_block(cx, msw, L, PI, refs["arrays"]["q_hat"],
                                                           refs["arrays"]["mf_residual"], D, *boot, nreg)
                                          if refs is not None else _na("no q-hat / MF residual (needs --raw or --refs)"))
            else:
                for k in ("shares", "information_gain"):
                    ed[k] = _na("no usable swap arm for the regime's models")
            mlp = [m for m in ms if m in LP]
            ed["star_permutation"] = (fr.starperm_block(cx, mlp, L, LP, cx.sd_test, *boot, nreg) if mlp
                                      else _na("no usable starperm0 and starperm1 arms"))
            ed["models_per_sub_block"] = {"swap": msw, "star_permutation": mlp}
        out["E_D"][reg] = ed
    ft = regimes["FT"]["models"]
    if "zeroshot" in X.present and ft:
        out["E_B"] = {"models": ft, "missing_or_excluded": regimes["FT"]["missing_or_excluded"],
                      "complete": len(ft) == 3, "holm_family": "E-B {dUAUC per Qwen domain} (A3 section 11)",
                      **fr.contrast_models({s: (L[s], L["zeroshot"]) for s in ft}, cx.y, cx.uc, cx.test, *boot,
                                           n_registered=3)}
    else:
        out["E_B"] = _na("needs the zeroshot like arm and at least one FT seed")
    out["P1"] = fr.p1_block(cx, L, PI, refs, {m for m in X.present if m in PI}, X.role, *boot)
    return out


# ---------------------------------------------------------------- E-W, E-F, E-H (one fit per regime and split)
def wu_regime(X, reg: str, info: dict, folds: list, strata: dict, rp: dict) -> tuple[dict, dict, dict]:
    """E_W[reg], E_F[reg], E_H[reg] from one set of cross-fitted predictors (E-W's estimator)."""
    cx, L, PI, refs = X.cx, X.L, X.PI, X.refs
    ms, nreg = info["models"], info["n_registered"]
    if not ms:
        na = _na(f"no usable like arm for the {reg} models {list(fr.REGIMES[reg])}" if info["missing_or_excluded"]
                 else f"{reg} models not requested")
        return na, na, na
    msw = [m for m in ms if m in PI]
    if not X.sd_ids:
        na = _na("no S_d")
        return na, na, na
    if not msw:
        na = _na("no usable swap arm (pi) for the regime's models")
        return na, na, na
    if refs is None:
        na = _na("no q-hat / MF residual (needs --raw or --refs)")
        return na, na, na
    q, mfres = refs["arrays"]["q_hat"], refs["arrays"]["mf_residual"]
    D = ed_rows(cx, ms, msw, L, PI)
    R = D & np.isfinite(q) & np.isfinite(mfres)              # = ftgrid_report.stacker_block's rows
    hdr = {"models": msw, "missing_or_excluded": list(info["missing_or_excluded"]) + [m for m in ms if m not in msw],
           "complete": len(msw) == nreg}
    sh, per = regime_etas(cx, msw, L, PI, q, mfres, R, folds)
    # ---- E-W and E-F on R
    sc = {("s", s): sh[s] for s in SHARED}
    sc.update({(m, s): per[m][s] for m in msw for s in ("M1", "M2", "M4")})
    keys, A = auc_table(sc, cx.y, cx.uc, R)
    nu, npairs = len(keys), fr._rows_of_users(cx.uc, R, keys)
    c = Cols(nu)
    for s in SHARED:
        c.add(("UAUC", s), A[("s", s)].mean(1))
    c.add("G_CF_wu", (A[("s", "M3")] - A[("s", "M0")]).mean(1))
    for m in msw:
        for s in ("M1", "M2", "M4"):
            c.add(("UAUC", s, m), A[(m, s)].mean(1))
        c.add(("G_wu", m), (A[(m, "M2")] - A[(m, "M1")]).mean(1))
        c.add(("G_LLM_given_CF", m), (A[(m, "M4")] - A[("s", "M3")]).mean(1))
        c.add(("G_CF_given_LLM", m), (A[(m, "M4")] - A[(m, "M2")]).mean(1))
    c.draws(X.n_boot, X.seed)
    rows = {"n_pairs": npairs, "n_users": nu, "K": len(folds),
            "fold_a_users_k0": int(len(set(cx.uc[R & folds[0]].tolist()))) if folds else 0,
            "fold_b_users_k0": int(len(set(cx.uc[R & ~folds[0]].tolist()))) if folds else 0}
    ew = {**hdr, "rows": rows,
          "UAUC": {"M0": c.one(("UAUC", "M0"), npairs), "M3": c.one(("UAUC", "M3"), npairs),
                   **{s: c.models({m: ("UAUC", s, m) for m in msw}, npairs, nreg, False) for s in ("M1", "M2")}},
          "G_wu": c.models({m: ("G_wu", m) for m in msw}, npairs, nreg, True),
          "G_CF_wu": c.one("G_CF_wu", npairs, contrast=True)}
    ig = ((rp.get("E_D") or {}).get(reg) or {}).get("information_gain") or {}
    ft = reg == "FT"
    ew["reading_G"] = robust_reading(ig.get("G_mean_over_seeds"), ew["G_wu"]["mean_over_seeds"],
                                     (ig.get("seeds") or {}).get("per_seed"), ew["G_wu"]["seeds"]["per_seed"], ft,
                                     nreg, None, "the E-D Holm family of A3 section 11 (G per domain and regime): set by "
                                                 "summarize (a domain file holds raw p-values only)")
    ew["reading_G_CF"] = robust_reading(ig.get("G_CF"), ew["G_CF_wu"], registered_confirmed=None,
                                        confirmed_source="none: G_CF is a reference in no registered family")
    ef = {**hdr, "rows": rows,
          "UAUC_M4": c.models({m: ("UAUC", "M4", m) for m in msw}, npairs, nreg, False),
          "G_LLM_given_CF": c.models({m: ("G_LLM_given_CF", m) for m in msw}, npairs, nreg, True),
          "G_CF_given_LLM": c.models({m: ("G_CF_given_LLM", m) for m in msw}, npairs, nreg, True)}
    # ---- E-H: the strata restrict the evaluation of the same predictors
    eh = {**hdr, "strata": {}, "partition": strata_partition(R, strata)}
    for name, S in strata.items():
        eh["strata"][name] = stratum_block(X, msw, nreg, R & S, sh, per)
    return ew, ef, eh


def strata_partition(R, strata: dict) -> dict:
    out = {"n_pairs_R": int(R.sum())}
    for name, S in strata.items():
        out[f"n_pairs_{name}"] = int((R & S).sum())
    if "sparse" in strata:
        out["sparse_and_dense_disjoint"] = not bool((strata["sparse"] & strata["dense"] & R).any())
        out["n_pairs_R_without_n_prior"] = int((R & ~(strata["sparse"] | strata["dense"])).sum())
    if "seen" in strata:
        out["seen_and_unseen_partition_R"] = bool(not (strata["seen"] & strata["unseen"] & R).any()
                                                  and ((strata["seen"] | strata["unseen"]) & R).sum() == R.sum())
    return out


def stratum_block(X, msw: list, nreg: int, rows, sh: dict, per: dict) -> dict:
    """One stratum: UAUC of L, q-hat and pi; G_prior = dUAUC(M1 - M0) and G_wu = dUAUC(M2 - M1) with E-W's predictors,
    evaluated on the stratum rows of users with both classes there."""
    cx, L, PI, q = X.cx, X.L, X.PI, X.refs["arrays"]["q_hat"]
    sc = {("s", "M0"): sh["M0"], ("lvl", "q_hat"): [q]}
    for m in msw:
        sc[(m, "M1")], sc[(m, "M2")] = per[m]["M1"], per[m]["M2"]
        sc[("lvl", "L", m)], sc[("lvl", "pi", m)] = [L[m]], [PI[m]]
    keys, A = auc_table(sc, cx.y, cx.uc, rows)
    nu, npairs = len(keys), fr._rows_of_users(cx.uc, rows, keys)
    c = Cols(nu)
    c.add("q_hat", A[("lvl", "q_hat")][:, 0])
    for m in msw:
        c.add(("L", m), A[("lvl", "L", m)][:, 0])
        c.add(("pi", m), A[("lvl", "pi", m)][:, 0])
        c.add(("G_prior", m), (A[(m, "M1")] - A[("s", "M0")]).mean(1))
        c.add(("G_wu", m), (A[(m, "M2")] - A[(m, "M1")]).mean(1))
    c.draws(X.n_boot, X.seed)
    return {"n_users_both_classes": nu, "n_pairs": npairs, "n_pairs_in_stratum": int(np.asarray(rows, bool).sum()),
            "descriptive_min_n": nu < fr.MIN_N,
            "UAUC_L": c.models({m: ("L", m) for m in msw}, npairs, nreg, False),
            "UAUC_q_hat": c.one("q_hat", npairs),
            "UAUC_pi": c.models({m: ("pi", m) for m in msw}, npairs, nreg, False),
            "G_prior": c.models({m: ("G_prior", m) for m in msw}, npairs, nreg, True),
            "G_wu": c.models({m: ("G_wu", m) for m in msw}, npairs, nreg, True)}


def p1_wu_block(X, folds: list, rp: dict) -> dict:
    """P1_wu: p1_block's rows and decision rule (ftgrid_report.p1_decision) on the within-user G values."""
    cx, L, PI, refs = X.cx, X.L, X.PI, X.refs
    need = ["zeroshot", "s0", "s1", "s2"]
    out = {"role": "within-user sensitivity of P1 (A3-6 item 2); P1 itself stays exactly as registered (registered_pooled)",
           "definition": "per user G_wu,m = mean over the K splits of AUC(M2_m,k) - AUC(M1_m,k) with user-centred "
                         "features on p1_block's rows (S_d TEST rows finite in q-hat, L and pi of zeroshot and s0-s2); "
                         "contrast_s = mean over users of G_wu,s - G_wu,ZS; mean over seeds with the P1 conditions"}
    swap_ok = {m for m in X.present if m in PI}
    miss = [m for m in need if m not in swap_ok]
    if miss or refs is None:
        reason = (f"missing E-D inputs (like and swap arms) for {miss}" if miss else "no q-hat (needs --raw or --refs)")
        out.update(available=False, reason=reason,
                   decision={"verdict": "P1_WU_INCOMPLETE", "holds": False, "reason": reason})
        return out
    q = refs["arrays"]["q_hat"]
    R = cx.sd_test & np.isfinite(q)
    for m in need:
        R &= np.isfinite(L[m]) & np.isfinite(PI[m])
    _, per = regime_etas(cx, need, L, PI, q, refs["arrays"]["mf_residual"], R, folds, own=("M1", "M2"), shared=())
    keys, A = auc_table({(m, s): per[m][s] for m in need for s in ("M1", "M2")}, cx.y, cx.uc, R)
    nu, npairs = len(keys), fr._rows_of_users(cx.uc, R, keys)
    G = {m: (A[(m, "M2")] - A[(m, "M1")]).mean(1) for m in need}
    c = Cols(nu)
    for s in need[1:]:
        c.add(s, G[s] - G["zeroshot"])
    c.draws(X.n_boot, X.seed)
    per_seed = {s: c.one(s, npairs, contrast=True, G_FT=float(G[s].mean()) if nu else NAN) for s in need[1:]}
    j = [c.idx[s] for s in need[1:]]
    mean = fr.rec(c.V[:, j].mean(1).mean() if nu else NAN, c.D[:, j].mean(1), nu, npairs, contrast=True)
    d = fr.p1_decision([per_seed[s]["est"] for s in need[1:]], mean)        # P1's conditions, applied as they are
    decision = {"verdict": P1_WU_VERDICT[d["verdict"]], "holds": d["holds"], "conditions": d["conditions"],
                "mean": d["mean"], "p": d["p"], "per_seed": d["per_seed"], "sigma_seed": d["sigma_seed"],
                "rule": "the P1 conditions of A3 section 3 (3 seeds, not descriptive, mean > 0 with p < 0.05, all seeds "
                        "positive, mean > 2 sigma_seed) applied to the within-user values; a sensitivity estimator, not "
                        "P1 (registered_pooled.P1 is P1)"}
    out.update(available=True, rows={"n_pairs": npairs, "n_users": nu, "K": len(folds)},
               G_ZS=float(G["zeroshot"].mean()) if nu else NAN, per_seed=per_seed, mean_over_seeds=mean,
               decision=decision, mean_sign=None if not math.isfinite(mean["est"]) else int(np.sign(mean["est"])))
    p1 = rp.get("P1") or {}
    confirmatory = (getattr(X, "root_label", "main") == "main" and p1.get("available") is True
                    and str(p1.get("role") or "").startswith("P1 confirmatory"))
    out["reading_P1"] = robust_reading(
        p1.get("mean_over_seeds"), mean,
        [(p1.get("per_seed") or {}).get(s, {}).get("est") for s in need[1:]] if p1.get("available") else None,
        [per_seed[s]["est"] for s in need[1:]], True, 3,
        bool((p1.get("decision") or {}).get("verdict") == "P1_HOLDS") if confirmatory else None,
        "P1's registered decision (a single test, A3 section 3)" if confirmatory else
        "none: P1 is ML-1M Qwen3-8B's test on the main root; here P1's quantity is a replication of its sign")
    return out


# ---------------------------------------------------------------- E-G
def rel1_share(users, rows, s, b, n_boot: int, seed: int) -> dict:
    """Item share of a score s with its item component b at reliability 1 (A3-6 item 4 ii, iii): the squared pooled
    within-user Pearson correlation of user-centred s and b through ftgrid_report.shares_from_sums, b standing for both
    donor halves (r_c = r8c = 1, share = rho^2); resamples as ftgrid_report.shares_block."""
    s, b = np.asarray(s, float), np.asarray(b, float)
    R = np.asarray(rows, bool) & np.isfinite(s) & np.isfinite(b)
    keys, S = fr.within_user_sums(users, R, [b, b, s, b], fr.SHARE_PRODS)
    nu = len(keys)
    nu2 = sum(c >= 2 for c in Counter(np.asarray(users)[R].tolist()).values())
    e = fr.shares_from_sums(S.sum(0)) if nu else None
    D = fr.mean_draws(S, n_boot, seed) * nu if nu else np.full((n_boot, 6), NAN)
    d = fr.shares_from_sums(D)
    n = int(R.sum())

    def est(k):
        return float(e[k][0]) if e is not None else NAN
    return {"rows": {"n_pairs": n, "n_users": nu, "n_users_with_2_or_more_pairs": int(nu2)}, "reliability": 1.0,
            "item_share": fr.rec(est("item_prior_share"), d["item_prior_share"], nu2, n),
            "rho": fr.rec(est("rho"), d["rho"], nu2, n),
            "non_item_share": fr.rec(est("non_prior_share"), d["non_prior_share"], nu2, n)}


def e_share_from_sums(S) -> np.ndarray:
    """[Var(L_c - pi_c) - (1 - r8c) Var(pi_c)] / Var(L_c) from the SHARE_PRODS sums (pooled within-user variances: the
    common divisor cancels); r8c from the same sums."""
    s = np.asarray(S, float).reshape(-1, 6)
    r8 = fr.shares_from_sums(s)["r8c"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (s[:, 4] - 2 * s[:, 3] + s[:, 5] - (1 - r8) * s[:, 5]) / s[:, 4]


def eshare_block(users, rows, per_model: dict, n_boot: int, seed: int, n_reg: int) -> dict:
    """The e-share per model on ftgrid_report.shares_block's rows, sums and resamples (identical draws)."""
    models = list(per_model)
    R = np.asarray(rows, bool).copy()
    for cols in per_model.values():
        for x in cols:
            R &= np.isfinite(x)
    st = {m: fr.within_user_sums(users, R, list(per_model[m]), fr.SHARE_PRODS) for m in models}
    keys = st[models[0]][0] if models else []
    S = np.column_stack([st[m][1] for m in models]) if models else np.zeros((0, 0))
    nu = len(keys)
    nu2 = sum(c >= 2 for c in Counter(np.asarray(users)[R].tolist()).values())
    D = fr.mean_draws(S, n_boot, seed) * nu if nu else np.full((n_boot, 6 * len(models)), NAN)
    n = int(R.sum())
    per, est, drw = {}, {}, {}
    for j, m in enumerate(models):
        tot = S[:, 6 * j:6 * j + 6].sum(0)
        est[m] = float(e_share_from_sums(tot)[0]) if nu else NAN
        drw[m] = e_share_from_sums(D[:, 6 * j:6 * j + 6])
        with np.errstate(divide="ignore", invalid="ignore"):
            per[m] = {"e_share": fr.rec(est[m], drw[m], nu2, n),
                      "var_e_over_var_L_uncorrected": float((tot[4] - 2 * tot[3] + tot[5]) / tot[4]) if nu else NAN,
                      "non_prior_share": float(fr.shares_from_sums(tot)["non_prior_share"][0]) if nu else NAN}
    mean = fr.rec(float(np.mean([est[m] for m in models])) if models else NAN,
                  np.mean([drw[m] for m in models], 0) if models else None, nu2, n)
    return {"rows": {"n_pairs": n, "n_users": nu, "n_users_with_2_or_more_pairs": int(nu2)}, "per_model": per,
            "mean_over_seeds": mean, "seeds": fr.seed_summary([per[m]["e_share"]["est"] for m in models], n_reg)}


def eg_block(X, regimes: dict) -> dict:
    cx, refs, boot = X.cx, X.refs, (X.n_boot, X.seed)
    out = {"definition": "item share = squared pooled within-user Pearson correlation of the user-centred score and its "
                         "user-centred item component, divided by the component's reliability (A3-6 item 4): (i) L with "
                         "pi (E-D's item-prior share, shares_block), (ii) the temporal MF score with its item bias b_i on "
                         "MF warm S_d TEST pairs (reliability 1), (iii) the label with q-hat (reliability 1; a lower "
                         "bound); e-share = [Var(L_c - pi_c) - (1 - r8c) Var(pi_c)] / Var(L_c) beside E-D's non-prior "
                         "share", "descriptive": True}
    if not X.sd_ids:
        out.update(_na("no S_d"))
        return out
    if refs is None:
        out["MF_score_item_bias"] = out["label_q_hat"] = _na("no CF references (needs --raw or --refs)")
    else:
        arr = refs["arrays"]
        out["MF_score_item_bias"] = rel1_share(cx.uc, cx.sd_test & (arr["mf_warm"] > 0), arr["mf_score"],
                                               arr["mf_item_bias"], *boot)
        out["label_q_hat"] = rel1_share(cx.uc, cx.sd_test, cx.y.astype(float), arr["q_hat"], *boot)
    for reg in REGS:
        info = regimes[reg]
        ms, nreg = info["models"], info["n_registered"]
        msw = [m for m in ms if m in X.PI]
        if not msw:
            out[reg] = _na("no usable like and swap arms for the regime's models" if ms else f"{reg}: no usable model")
            continue
        D = ed_rows(cx, ms, msw, X.L, X.PI)
        pm = {m: (X.PA[m], X.PB[m], X.L[m], X.PI[m]) for m in msw}
        out[reg] = {"models": msw, "complete": len(msw) == nreg,
                    "item_share_L_pi": fr.shares_block(cx.uc, D, pm, *boot, nreg),
                    "e_share": eshare_block(cx.uc, D, pm, *boot, nreg)}
    return out


# ---------------------------------------------------------------- E-J
def ej_block(X, regimes: dict) -> dict:
    cx, L, refs, boot = X.cx, X.L, X.refs, (X.n_boot, X.seed)
    out = {"definition": "paired on E-A's TEST users and rows (contrast_models: user-bootstrap CI and two-sided p as E-B): "
                         "dUAUC(L - q-hat), dUAUC(L - q-hat_T) (q-hat_T = forensics.prior_means at T_d, k = 5), "
                         "dUAUC(L - MF) on the MF warm pairs (the temporal MF score, descriptive)",
           "q_hat_T": X.qT_info}
    if refs is None:
        out.update(_na("no CF references (needs --raw or --refs)"))
        return out
    arr = refs["arrays"]
    q, mf, warm = arr["q_hat"], arr["mf_score"], arr["mf_warm"] > 0
    xr = {"q_hat": q, "popularity": cx.pop, "mf": mf, "mf_personal_residual": arr["mf_residual"]}
    for reg in REGS:
        info = regimes[reg]
        ms, nreg = info["models"], info["n_registered"]
        if not ms:
            out[reg] = _na(f"no usable like arm for the {reg} models")
            continue
        R = cx.test.copy()                                     # = uauc_models' rows of E-A's UAUC_TEST
        for m in ms:
            R &= np.isfinite(L[m])
        for v in xr.values():
            R &= np.isfinite(np.asarray(v, float))
        both = sorted(fr.user_aucs(np.zeros(cx.n), cx.y, cx.uc, R))
        blk = {"models": ms, "missing_or_excluded": info["missing_or_excluded"], "complete": len(ms) == nreg,
               "rows_E_A": {"n_pairs": fr._rows_of_users(cx.uc, R, both), "n_users": len(both)},
               "dUAUC_L_minus_q_hat": fr.contrast_models({m: (L[m], q) for m in ms}, cx.y, cx.uc, R, *boot,
                                                         n_registered=nreg)}
        if X.q_T is not None:
            blk["dUAUC_L_minus_q_hat_T"] = fr.contrast_models({m: (L[m], X.q_T) for m in ms}, cx.y, cx.uc, R, *boot,
                                                              n_registered=nreg)
            blk["dUAUC_L_minus_q_hat_T"]["n_E_A_pairs_without_q_hat_T"] = int((R & ~np.isfinite(X.q_T)).sum())
        else:
            blk["dUAUC_L_minus_q_hat_T"] = _na(X.qT_info.get("reason") or "no q-hat_T")
        blk["dUAUC_L_minus_MF_warm"] = {
            "descriptive": True,
            **fr.contrast_models({m: (L[m], mf) for m in ms}, cx.y, cx.uc, R & warm, *boot, n_registered=nreg)}
        out[reg] = blk
    return out


# ---------------------------------------------------------------- E-C'
def cal_rate_k(cx, rows) -> tuple[np.ndarray, dict]:
    """A3-6 item 7: per row of `rows`, the user's k_u = clip(round(r_u n_u), 1, n_u - 1); n_u = the user's rows in
    `rows`, r_u = the user's like rate among the user's CAL rows, or the pooled CAL like rate of the EVAL panel when the
    user has fewer than 3 CAL rows; round half up in exact integer arithmetic. -1 where undefined."""
    uc, y = np.asarray(cx.uc), np.asarray(cx.y, np.int64)
    cal, rows = np.asarray(cx.cal, bool), np.asarray(rows, bool)
    n_all = int(uc.max()) + 1 if len(uc) else 0
    cal_n = np.bincount(uc[cal], minlength=n_all)
    cal_pos = np.bincount(uc[cal], weights=y[cal].astype(float), minlength=n_all).round().astype(np.int64)
    pool_n, pool_pos = int(cal.sum()), int(y[cal].sum())
    n_u = np.bincount(uc[rows], minlength=n_all)
    k_user = np.full(n_all, -1, np.int64)
    fallback = np.zeros(n_all, bool)
    for u in np.flatnonzero(n_u > 0):
        if cal_n[u] >= ECP_MIN_CAL_ROWS:
            num, den = int(cal_pos[u]), int(cal_n[u])
        else:
            num, den = pool_pos, pool_n
            fallback[u] = True
        nu = int(n_u[u])
        if den <= 0 or nu < 2:
            continue
        k = (2 * num * nu + den) // (2 * den)                 # floor(r_u n_u + 1/2), exact
        k_user[u] = min(max(k, 1), nu - 1)
    users = np.flatnonzero(n_u > 0)
    info = {"n_users": int(len(users)), "n_users_pooled_rate_fallback": int(fallback[users].sum()),
            "pooled_cal_like_rate": pool_pos / pool_n if pool_n else NAN, "min_cal_rows": ECP_MIN_CAL_ROWS,
            "rounding": "half up: floor(r_u n_u + 1/2)"}
    return np.where(rows, k_user[uc], -1), info


def topk_anatomy_at(L, y, users, rows, k_row) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ftgrid_report.topk_anatomy with the user's k given (k_row: the user's k on each of its rows) instead of the
    user's TEST likes: c_u = midpoint of the k-th and (k+1)-th largest L, decision = 1[L >= c_u]; a user whose k is not
    in 1..n_u - 1 is skipped."""
    L, y = np.asarray(L, float), np.asarray(y, int)
    margin, correct, cu = np.full(len(L), NAN), np.full(len(L), NAN), np.full(len(L), NAN)
    ok = np.asarray(rows, bool) & np.isfinite(L)
    okidx = np.flatnonzero(ok)
    for idx in fx._groups(np.asarray(users)[ok]).values():
        j = okidx[idx]
        k = int(k_row[j[0]])
        if k < 1 or k > len(j) - 1:
            continue
        s = np.sort(L[j])[::-1]
        c = (s[k - 1] + s[k]) / 2
        cu[j] = c
        margin[j] = np.abs(L[j] - c)
        correct[j] = ((L[j] >= c).astype(int) == y[j]).astype(float)
    return margin, correct, cu


def ecprime_regime(X, ms: list, nreg: int) -> dict:
    """E-C's anatomy rows and resamples (ftgrid_report.ec_block): the oracle k and the deployable k_u side by side."""
    cx, L = X.cx, X.L
    R = (cx.cal | cx.test).copy()
    for m in ms:
        R &= np.isfinite(L[m])
    anat = {m: fr.topk_anatomy(L[m], cx.y, cx.uc, R & cx.test) for m in ms}
    aok = R & cx.test & np.isfinite(anat[ms[0]][0])
    k_row, kinfo = cal_rate_k(cx, aok)
    dep = {m: topk_anatomy_at(L[m], cx.y, cx.uc, aok, k_row) for m in ms}
    ok2 = aok & np.isfinite(dep[ms[0]][0])
    ridx = np.flatnonzero(R)
    cl = fr.Clusters(cx.uc[ridx])
    nk = len(fr.ANATOMY_KEYS)

    def stat(an, ok, rows):
        ar = rows[ok[rows]]
        return fr.anatomy_stats(an[0][ar], an[1][ar])
    est = np.array([[stat(anat[m], aok, ridx), stat(dep[m], ok2, ridx)] for m in ms]).reshape(len(ms), 2, nk)
    draws = np.full((X.n_boot, len(ms), 2, nk), NAN)
    for b, p in enumerate(cl.picks(X.n_boot, X.seed)):
        rows = ridx[cl.rows(p)]
        draws[b] = [[stat(anat[m], aok, rows), stat(dep[m], ok2, rows)] for m in ms]
    n_users, n_pairs = len(np.unique(cx.uc[aok])), int(aok.sum())
    aidx = np.flatnonzero(aok)
    diff = np.array([abs(int(cx.y[aidx[idx]].sum()) - int(k_row[aidx[idx[0]]]))
                     for idx in fx._groups(cx.uc[aok]).values()], float)
    kinfo.update(n_users_anatomy=n_users, n_users_deployable=len(np.unique(cx.uc[ok2])),
                 share_users_k_equals_oracle=float((diff == 0).mean()) if len(diff) else NAN,
                 mean_abs_k_minus_oracle=float(diff.mean()) if len(diff) else NAN)
    out = {"rows": {"n_users": n_users, "n_pairs": n_pairs, "n_pairs_deployable": int(ok2.sum())}, "k_rule": kinfo}
    for v, name in enumerate(("oracle", "deployable")):
        per = {m: {k: fr.rec(est[j, v, i], draws[:, j, v, i], n_users, n_pairs) for i, k in enumerate(fr.ANATOMY_KEYS)}
               for j, m in enumerate(ms)}
        mean = {k: fr.rec(est[:, v, i].mean(), draws[:, :, v, i].mean(1), n_users, n_pairs)
                for i, k in enumerate(fr.ANATOMY_KEYS)}
        out[name] = {"per_model": per, "mean_over_seeds": mean,
                     "seeds": {k: fr.seed_summary([per[m][k]["est"] for m in ms], nreg)
                               for k in ("topk_error_rate", "AUROC_margin_correct")}}
    dd = draws[:, :, 1, :] - draws[:, :, 0, :]
    de = est[:, 1, :] - est[:, 0, :]
    out["deployable_minus_oracle"] = {
        "per_model": {m: {k: fr.rec(de[j, i], dd[:, j, i], n_users, n_pairs) for i, k in enumerate(fr.ANATOMY_KEYS)}
                      for j, m in enumerate(ms)},
        "mean_over_seeds": {k: fr.rec(de[:, i].mean(), dd[:, :, i].mean(1), n_users, n_pairs)
                            for i, k in enumerate(fr.ANATOMY_KEYS)}}
    return out


def ecprime_block(X, regimes: dict) -> dict:
    out = {"definition": "E-C's per-user top-k anatomy (ftgrid_report.topk_anatomy, anatomy_stats; E-C's user "
                         "resamples) with the oracle k = the user's TEST likes and with the deployable k_u = clip(round("
                         "r_u n_u), 1, n_u - 1), r_u = the user's CAL like rate (fewer than 3 CAL rows: the pooled CAL "
                         "like rate of the EVAL users); 'the oracle top-k decision' vs E-C' (A3-6 item 7)",
           "descriptive": True}
    for reg in REGS:
        info = regimes[reg]
        ms = info["models"]
        if not ms:
            out[reg] = _na(f"no usable like arm for the {reg} models")
            continue
        out[reg] = {"models": ms, "missing_or_excluded": info["missing_or_excluded"],
                    "complete": len(ms) == info["n_registered"], **ecprime_regime(X, ms, info["n_registered"])}
    return out


# ---------------------------------------------------------------- FT-C reading (item 8)
def retention(U) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R_0, R_1, R) from UAUCs in FTC_MODELS order (last axis): R_s = [UAUC(p_s) - UAUC(ZS)] / [UAUC(s) - UAUC(ZS)]."""
    U = np.asarray(U, float)
    zs, s0, s1, p0, p1 = (U[..., j] for j in range(5))
    with np.errstate(divide="ignore", invalid="ignore"):
        r0, r1 = (p0 - zs) / (s0 - zs), (p1 - zs) / (s1 - zs)
    return r0, r1, (r0 + r1) / 2


def ftc_label(r: dict | None) -> str:
    if not isinstance(r, dict) or not (math.isfinite(_num(r.get("lo"))) and math.isfinite(_num(r.get("hi")))):
        return "NOT_DEFINED"
    if r["lo"] > RETENTION_CUT:
        return "ITEM_DRIVEN"
    if r["hi"] < RETENTION_CUT:
        return "USER_DRIVEN"
    return "MIXED"


def ftc_reading(X, eb: dict) -> dict:
    cx, L = X.cx, X.L
    root, domain = getattr(X, "root_label", "main"), getattr(X, "domain", None)
    control = "FT-Q" if root == "teacher" else "FT-C"
    out = {"definition": "retention R = mean over s in {0, 1} of [UAUC(p_s) - UAUC(ZS)] / [UAUC(s) - UAUC(ZS)] on "
                         "identical TEST users and rows; defined only when the E-B contrast is positive with a CI "
                         "excluding 0 (and, stricter than registered, E-B complete and at least 150 users: R13); CI: one "
                         "user resample drives all UAUCs; ITEM_DRIVEN iff lo > 0.5, USER_DRIVEN iff hi < 0.5, MIXED "
                         "otherwise, NOT_DEFINED when R is not defined (A3-6 item 8; addendum 8 section 2 for FT-Q, R_Q); "
                         "the UAUC of q-hat on the same rows is addendum 8's descriptive companion",
           "control": control, "descriptive": True,
           "status": ("registered, outcome-free: FT-Q, the item-only teacher control (addendum 8 section 2)"
                      if control == "FT-Q" else "registered, outcome-free: FT-C on ML-1M (A3 section 9; addendum 8)")}
    x1 = getattr(X, "x1", None) or {"complete": False, "recorded": {}}
    if not x1.get("complete"):
        missing = [rel for rel, ok in (x1.get("recorded") or {}).items() if not ok] or list(X1_FILES)
        out.update(_na("record missing: A3-6 item 11 requires the sha1 of " + ", ".join(missing) + " in the pilot log "
                       "(--pilot_log) before this registered, outcome-free block is computed"), label="NOT_DEFINED")
        return out
    if root == "llama":
        out.update(_na("no FT-C or FT-Q control for the Llama backbone (A3 section 9, addendum 8 section 3: Qwen3-8B "
                       "only)"), label="NOT_DEFINED")
        return out
    if root == "main" and domain != "ml1m":
        out.update(_na("FT-C is the registered ML-1M control only: addendum 8 section 1 withdrew the Toys permutation "
                       "run (the FT-Q teacher control runs on the teacher root)"), label="NOT_DEFINED")
        return out
    miss = [m for m in FTC_MODELS if m not in X.present]
    if miss:
        out.update(_na(f"needs usable like arms of {list(FTC_MODELS)}; missing or excluded: {miss}"),
                   label="NOT_DEFINED")
        return out
    if not isinstance(eb, dict) or not isinstance(eb.get("mean_over_seeds"), dict):
        out.update(_na("the E-B contrast is not available (needs zeroshot and the FT seeds)"), label="NOT_DEFINED")
        return out
    R = cx.test.copy()
    for m in FTC_MODELS:
        R &= np.isfinite(L[m])
    pu = {m: fr.user_aucs(L[m], cx.y, cx.uc, R) for m in FTC_MODELS}
    keys = sorted(set.intersection(*(set(d) for d in pu.values())))
    nu, npairs = len(keys), fr._rows_of_users(cx.uc, R, keys)
    V = np.array([[pu[m][u] for m in FTC_MODELS] for u in keys], float).reshape(nu, len(FTC_MODELS))
    D = fr.mean_draws(V, X.n_boot, X.seed)
    U = V.mean(0) if nu else np.full(len(FTC_MODELS), NAN)
    em = eb["mean_over_seeds"]
    cond = {"E_B_complete": eb.get("complete") is True, "E_B_mean_gt_0": bool(_num(em.get("est")) > 0),
            "E_B_ci_excludes_0": em.get("ci_excludes_0") is True, "n_users_ge_150": nu >= fr.MIN_N}
    q = (X.refs or {}).get("arrays", {}).get("q_hat") if getattr(X, "refs", None) is not None else None
    if q is None:
        uq = _na("no q-hat (needs --raw or --refs)")
    elif not np.isfinite(np.asarray(q, float)[R]).all():
        uq = _na("q-hat is not finite on every row of the reading")
    else:
        pq = fr.user_aucs(q, cx.y, cx.uc, R)                    # the same rows and users (every user has both classes)
        vq = np.array([pq[u] for u in keys], float)
        uq = fr.rec(vq.mean() if nu else NAN, fr.mean_draws(vq, X.n_boot, X.seed)[:, 0], nu, npairs)
    out.update(available=True, rows={"n_users": nu, "n_pairs": npairs},
               UAUC={m: fr.rec(U[j], D[:, j], nu, npairs) for j, m in enumerate(FTC_MODELS)}, UAUC_q_hat=uq,
               E_B_condition=cond, E_B_mean_over_seeds={k: em.get(k) for k in ("est", "lo", "hi", "p", "n_users")})
    if not all(cond.values()):
        why = "R is not defined: " + ", ".join(k for k, v in cond.items() if not v) + " fails"
        out.update(defined=False, R=_na(why), R_per_seed=_na(why), label="NOT_DEFINED", reason=why)
        return out
    r0, r1, r = retention(U)
    d0, d1, d = retention(D)
    out.update(defined=True,
               R=fr.rec(float(r), d, nu, npairs, n_draws_nonfinite=int((~np.isfinite(d)).sum())),
               R_per_seed={"s0": fr.rec(float(r0), d0, nu, npairs), "s1": fr.rec(float(r1), d1, nu, npairs)})
    out["label"] = ftc_label(out["R"])
    return out


# ---------------------------------------------------------------- driver
def status_of(domain: str, backbone, root_label: str = "main") -> dict:
    ftc = ("registered_outcome_free (FT-Q, addendum 8)" if root_label == "teacher" else
           "registered_outcome_free (FT-C, ML-1M)" if root_label == "main" and domain == "ml1m" else
           "not registered on this root and domain (not computed)")
    return {"root_label": root_label,
            "items_2_to_7": "exploratory" if domain == "ml1m" else "registered_outcome_free",
            "FT_C_reading": ftc,
            "family_eligible_panel": bool(root_label == "main" and domain in AMAZON and _is_qwen(backbone)),
            "rule": "A3-6 Consequence and addendum 8: items 2-7 are exploratory on ML-1M (its report was read) and "
                    "registered, outcome-free on every other panel; FT-C is registered on ML-1M (main root) and FT-Q on "
                    "the teacher root; the families E-F, E-H and E-J hold main-root Qwen Amazon panels only (summarize)"}


def build(a) -> dict:
    check_root_label(a)
    fingerprint = input_fingerprint(a)          # before anything is read: the inputs as this build reads them
    X = load_inputs(a)
    regimes = {}
    for reg in REGS:
        ok, miss = fr._regime(X.models, X.present, reg)
        regimes[reg] = {"models": ok, "missing_or_excluded": miss, "n_registered": len(fr.REGIMES[reg])}
    rp = registered_pooled(X, regimes)
    folds = split_folds(X.cx) if X.sd_ids else []
    strata = {}
    if X.refs is not None:
        n_prior = np.asarray(X.refs["arrays"]["n_prior"], float)
        fin = np.isfinite(n_prior)
        strata = {"sparse": fin & (n_prior < SPARSE_N_PRIOR), "dense": fin & (n_prior >= SPARSE_N_PRIOR)}
    if X.cx.seen is not None:
        strata.update(unseen=~X.cx.seen, seen=X.cx.seen.copy())
    E_W = {"definition": "E-D's stackers M0-M3 with user-centred features (each feature minus the user's mean over the "
                         "user's rows of R = E-D's stacker rows) and K = 20 cross-fitting splits stats.user_halves(S_d, "
                         "k), k = 0..19; per user every statistic averaged over the splits; user bootstrap with the "
                         "coefficients of every split held fixed (A3-6 item 2)", "K": K_SPLITS}
    E_F = {"definition": "M4 = [q-hat, MF personal residual, pi, e-hat] with E-W's estimator; G_LLM_given_CF = dUAUC(M4 - "
                         "M3), G_CF_given_LLM = dUAUC(M4 - M2) on identical users and rows (A3-6 item 3); H-F: G_LLM_given_CF "
                         "> 0 in FT (family E-F in summarize)"}
    E_H = {"definition": "strata of the S_d TEST rows: sparse = q-hat support n_prior < 5, dense otherwise; unseen = the "
                         "item has no TRAIN example, seen otherwise; UAUC of L, q-hat and pi, G_prior = dUAUC(M1 - M0) "
                         "and G_wu = dUAUC(M2 - M1) with E-W's predictors (fit on all rows of the other fold; the stratum "
                         "restricts the evaluation); fewer than 150 users: descriptive (A3-6 item 5)",
           "sparse_rule": f"n_prior < {SPARSE_N_PRIOR}"}
    if not strata:
        E_H.update(_na("no strata: needs the CF references (n_prior) or train.jsonl"))
    for reg in REGS:
        ew, ef, eh = wu_regime(X, reg, regimes[reg], folds, strata, rp)
        E_W[reg], E_F[reg] = ew, ef
        if strata:
            E_H[reg] = eh
    E_W["P1_wu"] = (p1_wu_block(X, folds, rp) if folds
                    else {**_na("no S_d"), "decision": {"verdict": "P1_WU_INCOMPLETE", "holds": False}})
    res = {"registered_pooled": rp, "E_W": E_W, "E_F": E_F, "E_G": eg_block(X, regimes), "E_H": E_H,
           "E_J": ej_block(X, regimes), "E_Cprime": ecprime_block(X, regimes),
           "FT_C_reading": ftc_reading(X, rp.get("E_B") or {})}
    cx = X.cx
    meta = {"domain": a.domain, "backbone": X.backbone, "backbones_seen": X.backbones, "T": X.T,
            "variant": X.split.get("variant"), "split_sha1": fx.file_sha1(a.split), "panel_sha1": X.panel_sha,
            "models_requested": X.models, "regimes": regimes, "n_boot": a.n_boot, "seed": a.seed, "min_n": fr.MIN_N,
            "K_splits": K_SPLITS, "fold_k0_equals_E_D_fold_a": bool(folds and np.array_equal(folds[0], cx.fold_a)),
            "n_eval_pairs": cx.n, "n_test_pairs": int(cx.test.sum()), "n_cal_pairs": int(cx.cal.sum()),
            "n_sd_users": len(X.sd_ids), "n_sd_test_pairs": int(cx.sd_test.sum()), "role_of_P1": X.role,
            "references_source": X.refs_info.get("source") or X.refs_info.get("reason"),
            "q_hat_T_source": X.qT_info.get("source") or X.qT_info.get("reason"), "code_sha1": code_sha1(),
            "root_label": X.root_label, "x1_record": X.x1, "fingerprint": fingerprint}
    return {"spec": SPEC, "status": status_of(a.domain, X.backbone, X.root_label), "meta": meta,
            "input_checks": {"problems": X.problems, **X.checks}, "excluded_runs": X.excluded, "runs": X.runs,
            "joins": X.joins, "swap_arm_like_consistency": X.consistency,
            "references": {**X.refs_info, "q_hat_T": X.qT_info}, "readings": list(READINGS), **res}


# ---------------------------------------------------------------- tables
FAMILY_STATS = {("E-F", "FT", "G_LLM_given_CF"), ("E-H", "ZS", "G_prior:sparse"), ("E-H", "FT", "G_prior:sparse"),
                ("E-J", "FT", "dUAUC_L_minus_q_hat")}


def table_rows(res: dict) -> list:
    """Long-format rows. holm_member names the Holm family a row is a member candidate of (A3-6: E-F, E-H, E-J on
    main-root Qwen Amazon panels; A3 section 11: E-D for G and E-B on main-root Qwen panels); confirmation is decided
    by summarize only, so no row carries it. descriptive: a registered row keeps the minimum-n flag; an A3-6 row is
    descriptive unless it is a family member candidate."""
    d, bb = res["meta"]["domain"], res["meta"]["backbone"] or ""
    elig = res["status"]["family_eligible_panel"]
    main_qwen = res["status"].get("root_label", "main") == "main" and _is_qwen(bb)
    rows = []

    def add(block, regime, model, stat, r, stratum="", note="", family_key=None, holm=""):
        if not isinstance(r, dict) or "est" not in r:
            return
        dmin = r.get("descriptive_min_n")
        if block == "registered_pooled":
            hm = holm if (holm and main_qwen and not dmin) else ""
            desc = bool(dmin)
        else:
            hm = (family_key[0] if (elig and model == "mean_over_seeds" and family_key in FAMILY_STATS and not dmin)
                  else "")
            desc = bool(dmin) or not hm
        rows.append({"block": block, "panel": d, "regime": regime, "statistic": stat, "estimate": r.get("est"),
                     "lo": r.get("lo"), "hi": r.get("hi"), "n_users": r.get("n_users"), "descriptive": desc,
                     "holm_member": hm, "backbone": bb, "model": model, "stratum": stratum, "p": r.get("p"),
                     "n_pairs": r.get("n_pairs"), "descriptive_min_n": dmin, "note": note})

    def models(block, regime, stat, blk, stratum="", note="", fam=None):
        if not isinstance(blk, dict):
            return
        for m, r in (blk.get("per_model") or {}).items():
            add(block, regime, m, stat, r, stratum, note)
        add(block, regime, "mean_over_seeds", stat, blk.get("mean_over_seeds"), stratum, note, fam)

    rp = res["registered_pooled"]
    for reg, ed in rp["E_D"].items():
        ig = ed.get("information_gain") or {}
        for m, blk in (ig.get("per_model") or {}).items():
            add("registered_pooled", reg, m, "G", blk.get("G"), note="registered (A3 section 3)")
        add("registered_pooled", reg, "mean_over_seeds", "G", ig.get("G_mean_over_seeds"), note="registered",
            holm="E-D")
        add("registered_pooled", reg, "reference", "G_CF", ig.get("G_CF"), note="registered")
        sh = ed.get("shares") or {}
        for k in ("item_prior_share", "non_prior_share", "r8c"):
            add("registered_pooled", reg, "mean_over_seeds", k, (sh.get("mean_over_seeds") or {}).get(k),
                note="registered")
    eb = rp.get("E_B") or {}
    add("registered_pooled", "FT", "mean_over_seeds", "E_B_dUAUC_FT_minus_ZS", eb.get("mean_over_seeds"),
        note="registered", holm="E-B")
    p1 = rp.get("P1") or {}
    if p1.get("available"):
        add("registered_pooled", "FT-ZS", "mean_over_seeds", "P1_G_FT_minus_G_ZS", p1["mean_over_seeds"],
            note=f"{p1['decision']['verdict']} (registered)")
    for reg in REGS:
        ew = res["E_W"].get(reg) or {}
        if ew.get("available") is False:
            continue
        for s in ("M0", "M3"):
            add("E-W", reg, "shared", f"UAUC_{s}", ew["UAUC"][s])
        for s in ("M1", "M2"):
            models("E-W", reg, f"UAUC_{s}", ew["UAUC"][s])
        models("E-W", reg, "G_wu", ew["G_wu"], note=f"reading={ew['reading_G']['reading']}")
        add("E-W", reg, "shared", "G_CF_wu", ew["G_CF_wu"], note=f"reading={ew['reading_G_CF']['reading']}")
        ef = res["E_F"][reg]
        models("E-F", reg, "UAUC_M4", ef["UAUC_M4"])
        models("E-F", reg, "G_LLM_given_CF", ef["G_LLM_given_CF"], fam=("E-F", reg, "G_LLM_given_CF"))
        models("E-F", reg, "G_CF_given_LLM", ef["G_CF_given_LLM"])
    pw = res["E_W"].get("P1_wu") or {}
    if pw.get("available"):
        for s, r in pw["per_seed"].items():
            add("E-W", "FT-ZS", s, "P1_wu_G_FT_minus_G_ZS", r)
        add("E-W", "FT-ZS", "mean_over_seeds", "P1_wu_G_FT_minus_G_ZS", pw["mean_over_seeds"],
            note=f"{pw['decision']['verdict']}; reading={pw['reading_P1']['reading']}")
    eg = res["E_G"]
    for k in ("MF_score_item_bias", "label_q_hat"):
        add("E-G", "", "reference", f"item_share_{k}", (eg.get(k) or {}).get("item_share"), note="reliability 1")
    for reg in REGS:
        g = eg.get(reg) or {}
        if g.get("available") is False or not g:
            continue
        sh = g["item_share_L_pi"]
        for m, blk in sh["per_model"].items():
            add("E-G", reg, m, "item_share_L_pi", blk["item_prior_share"])
            add("E-G", reg, m, "non_prior_share", blk["non_prior_share"])
        add("E-G", reg, "mean_over_seeds", "item_share_L_pi", sh["mean_over_seeds"].get("item_prior_share"))
        add("E-G", reg, "mean_over_seeds", "non_prior_share", sh["mean_over_seeds"].get("non_prior_share"))
        es = g["e_share"]
        for m, blk in es["per_model"].items():
            add("E-G", reg, m, "e_share", blk["e_share"])
        add("E-G", reg, "mean_over_seeds", "e_share", es["mean_over_seeds"])
    for reg in REGS:
        eh = res["E_H"].get(reg) or {}
        for name, st in (eh.get("strata") or {}).items():
            models("E-H", reg, "UAUC_L", st["UAUC_L"], name)
            add("E-H", reg, "reference", "UAUC_q_hat", st["UAUC_q_hat"], name)
            models("E-H", reg, "UAUC_pi", st["UAUC_pi"], name)
            models("E-H", reg, "G_prior", st["G_prior"], name, fam=("E-H", reg, f"G_prior:{name}"))
            models("E-H", reg, "G_wu", st["G_wu"], name)
    for reg in REGS:
        ej = res["E_J"].get(reg) or {}
        for k in ("dUAUC_L_minus_q_hat", "dUAUC_L_minus_q_hat_T", "dUAUC_L_minus_MF_warm"):
            blk = ej.get(k) or {}
            for m, r in (blk.get("per_seed") or {}).items():
                add("E-J", reg, m, k, r)
            add("E-J", reg, "mean_over_seeds", k, blk.get("mean_over_seeds"), family_key=("E-J", reg, k))
    for reg in REGS:
        ec = res["E_Cprime"].get(reg) or {}
        for v in ("oracle", "deployable", "deployable_minus_oracle"):
            blk = ec.get(v) or {}
            for m, d_ in (blk.get("per_model") or {}).items():
                for k in ("topk_error_rate", "AUROC_margin_correct", "share_errors_top", "AURC"):
                    add("E-C'", reg, m, f"{v}:{k}", d_.get(k))
            for k in ("topk_error_rate", "AUROC_margin_correct", "share_errors_top", "AURC"):
                add("E-C'", reg, "mean_over_seeds", f"{v}:{k}", (blk.get("mean_over_seeds") or {}).get(k))
    ftc = res["FT_C_reading"]
    ctl = ftc.get("control", "FT-C")
    if ftc.get("available") is not False:
        for m, r in (ftc.get("UAUC") or {}).items():
            add(ctl, "FT-PERM", m, "UAUC_TEST", r)
        add(ctl, "FT-PERM", "reference", "UAUC_TEST_q_hat", ftc.get("UAUC_q_hat"), note="descriptive companion")
    if ftc.get("defined"):
        for s, r in ftc["R_per_seed"].items():
            add(ctl, "FT-PERM", s, "retention_R", r)
        add(ctl, "FT-PERM", "mean_over_seeds", "retention_R", ftc["R"], note=ftc["label"])
    return rows


def tables_text(rows: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLS)
    for r in rows:
        w.writerow([fr._cell(r.get(c)) for c in CSV_COLS])
    return buf.getvalue()


def tables_path(out: Path) -> Path:
    return out.with_name(out.stem + "_tables.csv")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    tmp.replace(path)


def write_build(out: Path, obj: dict) -> dict:
    """The JSON is the completion marker: the old JSON is removed first, the CSV is written next and the JSON (holding the
    CSV's sha1) last, each through a temporary file and a rename, so a JSON on disk always sits beside its own CSV."""
    res = strict_json(obj)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or out.is_symlink():
        out.unlink()
    text = tables_text(table_rows(res))
    _atomic_write(tables_path(out), text)
    res["meta"]["tables_csv_sha1"] = hashlib.sha1(text.encode("utf-8")).hexdigest()
    _atomic_write(out, json.dumps(res, indent=2, allow_nan=False) + "\n")
    return res


def _strict_load(path) -> dict:
    def bad(x):
        raise ValueError(f"non-strict JSON constant {x}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=bad)


def _diff_keys(a, b, prefix: str = "", depth: int = 2) -> list:
    if not (isinstance(a, dict) and isinstance(b, dict)) or depth == 0:
        return [prefix or "<root>"] if a != b else []
    out = []
    for k in sorted(set(a) | set(b), key=str):
        if a.get(k) != b.get(k):
            out += _diff_keys(a.get(k), b.get(k), f"{prefix}.{k}" if prefix else str(k), depth - 1)
    return out


def check_resume(a) -> tuple[bool, str]:
    """(True, why) iff --out is a complete build of exactly the current inputs, settings and code: its stored
    fingerprint equals input_fingerprint(a) and the CSV beside it is the one written with it."""
    check_root_label(a)
    out = Path(a.out)
    if not out.is_file():
        return False, "no output yet"
    try:
        doc = _strict_load(out)
    except (OSError, ValueError) as e:
        return False, f"the output is unreadable ({type(e).__name__})"
    meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
    stored = meta.get("fingerprint")
    cur = json.loads(json.dumps(strict_json(input_fingerprint(a))))
    if stored != cur:
        return False, "the stored build differs in " + ", ".join(_diff_keys(stored or {}, cur)[:12])
    tp = tables_path(out)
    if not tp.is_file() or _sha1_file(tp) != meta.get("tables_csv_sha1"):
        return False, "the tables CSV is missing or is not the one written with the JSON"
    return True, "the stored build matches the current inputs, settings and code"


# ---------------------------------------------------------------- summarize (the A3-6 families)
DOC_KEYS = ("spec", "status", "meta", "input_checks", "registered_pooled", "E_W", "E_F", "E_G", "E_H", "E_J",
            "E_Cprime", "FT_C_reading")
ROLE_WANTS = {"files": ("main", "qwen"), "ml1m": ("main", "qwen"), "ftc": ("main", "qwen"), "llama": ("llama", "llama"),
              "ftq": ("teacher", "qwen")}


def _refuse(msg: str):
    print(f"summarize refused: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_checked(path: str, role: str, n_boot: int, cur_sha1: str | None) -> dict:
    """A domain file, validated for its role (R16); anything else is refused (exit 2), never dropped."""
    name = Path(path).name
    try:
        doc = _strict_load(path)
    except (OSError, ValueError) as e:
        _refuse(f"{name}: unreadable ({type(e).__name__}: {e})")
    if not isinstance(doc, dict):
        _refuse(f"{name}: not an ftgrid_extra domain file")
    missing = [k for k in DOC_KEYS if k not in doc]
    if missing:
        _refuse(f"{name}: not an ftgrid_extra domain file (missing {missing})")
    m = doc["meta"] if isinstance(doc["meta"], dict) else {}
    probs = (doc.get("input_checks") or {}).get("problems")
    if probs or probs is None:
        _refuse(f"{name}: input_checks.problems is not empty: {probs}")
    fam = _family_of(m.get("backbone"))
    if fam is None:
        _refuse(f"{name}: meta.backbone {m.get('backbone')!r} is missing or unknown")
    if m.get("n_boot") != n_boot or m.get("seed") != 0:
        _refuse(f"{name}: built with n_boot {m.get('n_boot')} and seed {m.get('seed')}; this summary needs n_boot "
                f"{n_boot} and seed 0")
    built = (m.get("code_sha1") or {}).get("ftgrid_extra.py")
    if built != cur_sha1:
        _refuse(f"{name}: built by ftgrid_extra.py {built}, not by the current file {cur_sha1}")
    root, dom = m.get("root_label"), m.get("domain")
    want = ROLE_WANTS[role]
    if (root, fam) != want:
        _refuse(f"{name}: a {root} root file with a {fam} backbone cannot be given as --{role} (needs the {want[0]} "
                f"root and a {want[1]} backbone)")
    if role == "files" and not (dom in AMAZON and (doc.get("status") or {}).get("family_eligible_panel") is True):
        _refuse(f"{name}: --files takes the main-root Qwen files of toys, games and sports only (got {dom})")
    if role in ("ml1m", "ftc") and dom != "ml1m":
        _refuse(f"{name}: --{role} takes the main-root Qwen ML-1M file (got {dom})")
    if role == "llama" and dom not in ("ml1m", "toys"):
        _refuse(f"{name}: the Llama program runs on ML-1M and Toys only (got {dom})")
    if role == "ftq" and dom not in ("ml1m",) + AMAZON:
        _refuse(f"{name}: unknown FT-Q dataset {dom}")
    lacking = [x for x in ROLE_MODELS[role] if x not in (m.get("models_requested") or [])]
    if lacking:
        _refuse(f"{name}: models_requested lacks the registered models {lacking} of its root and role (--{role})")
    doc["_file"] = {"name": name, "sha1": fx.file_sha1(path), "role": role}
    return doc


def ft_status(doc: dict) -> str:
    """'not_run' iff every FT model (s0-s2) has no scoring directory for any arm read (load_run status ABSENT): its
    fine-tuned runs were not run; 'run' otherwise (a run that was excluded, e.g. by E1, was run)."""
    runs = doc.get("runs") or {}
    per = [runs.get(m) for m in FT_MODELS]
    if not all(isinstance(r, dict) and r for r in per):
        return "unknown"
    absent = all((arm or {}).get("status") == "ABSENT" for r in per for arm in r.values())
    return "not_run" if absent else "run"


def not_run_recorded(pilot_log, dataset: str) -> bool:
    """A line of the pilot log carries the exact token `FTEXTRA_NOT_RUN <dataset>` (the record of a registered cut)."""
    if not pilot_log:
        return False
    try:
        text = Path(pilot_log).read_text(encoding="utf-8")
    except OSError:
        return False
    pat = re.compile(rf"(?<![\w-]){NOT_RUN_TOKEN} {re.escape(dataset)}(?![\w-])")
    return any(pat.search(line) for line in text.splitlines())


def _member(stat: dict | None, seeds: dict | None, fine_tuned: bool, label: str) -> dict:
    """A listed family member (R14): the minimum-n rule alone takes it out of the family; a member whose statistic is
    unavailable stays in with p = 1 and is never confirmed."""
    if not isinstance(stat, dict) or "est" not in stat:
        out = {"est": None, "p": None, "missing": True, "p_for_holm": 1.0, "in_family": True,
               "reason": f"{label}: statistic not available; counted in the family with p = 1 (R14)"}
        if fine_tuned:
            out.update(sigma_seed_rule=None, complete=None, per_seed=None, sigma_seed=None)
        return out
    out = {"est": stat.get("est"), "lo": stat.get("lo"), "hi": stat.get("hi"), "p": stat.get("p"),
           "n_users": stat.get("n_users"), "descriptive_min_n": stat.get("descriptive_min_n"), "missing": False}
    if fine_tuned:
        s = seeds or {}
        out.update(sigma_seed_rule=s.get("sigma_seed_rule"), complete=s.get("complete"), per_seed=s.get("per_seed"),
                   sigma_seed=s.get("sigma_seed"))
    if stat.get("descriptive_min_n"):
        out.update(in_family=False, reason=f"{label}: fewer than {fr.MIN_N} users (descriptive, A3 section 1)")
    elif not fr._fin(stat.get("p")):
        out.update(in_family=True, missing=True, p_for_holm=1.0,
                   reason=f"{label}: no p-value; counted in the family with p = 1 (R14)")
    else:
        out.update(in_family=True, p_for_holm=float(stat["p"]))
    return out


def holm_family(cands: dict, fine_tuned: dict, direction: int | None, hypothesis: str) -> dict:
    """Holm over the listed members (R14); confirmed = Holm p < 0.05, the sigma_seed rule for FT members and, for a
    directional hypothesis, an estimate of the hypothesised sign; a missing member is never confirmed."""
    members = {k: v for k, v in cands.items() if v.get("in_family")}
    adj = fr.holm({k: v["p_for_holm"] for k, v in members.items()})
    out_m = {}
    for k, v in members.items():
        est = _num(v.get("est"))
        sign = int(np.sign(est)) if math.isfinite(est) else None
        rule_ok = (v.get("sigma_seed_rule") is True) if fine_tuned[k] else True
        dir_ok = True if direction is None else sign == direction
        out_m[k] = {**v, "p_holm": adj[k], "sign": sign, "fine_tuned": fine_tuned[k], "direction_ok": dir_ok,
                    "confirmed": bool(not v.get("missing") and adj[k] is not None and adj[k] < ALPHA and rule_ok
                                      and dir_ok)}
    return {"hypothesis": hypothesis, "alpha": ALPHA, "m": len(members), "members": out_m,
            "outside_family": {k: v for k, v in cands.items() if not v.get("in_family")},
            "n_missing_members": sum(bool(v.get("missing")) for v in out_m.values()),
            "n_confirmed": sum(v["confirmed"] for v in out_m.values()),
            "rule": "Holm over the members the family lists (a member is outside only by the minimum-n rule; a member "
                    "whose statistic is unavailable stays in with p = 1, R14); confirmed = Holm p < 0.05, the sigma_seed "
                    "rule for FT members (all registered seeds present, all seeds the sign of the mean, |mean| > 2 "
                    "sigma_seed)" + ("" if direction is None else " and an estimate of the hypothesised sign")}


def eb_family_copy(by_domain: dict) -> dict:
    """A3 section 11 E-B family over the four Qwen domains, as fill_paper.eb_family computes it; p_holm and confirmed
    are null unless all four files are given with a complete, non-descriptive E-B (R15)."""
    four = ("ml1m", "toys", "games", "sports")
    members, why = {}, []
    for d in four:
        doc = by_domain.get(d)
        if doc is None:
            why.append(f"{d}: file not given")
            continue
        eb = doc["registered_pooled"].get("E_B") or {}
        mean = eb.get("mean_over_seeds") if isinstance(eb.get("mean_over_seeds"), dict) else None
        if mean is None:
            why.append(f"{d}: E-B not available")
            continue
        s = eb.get("seeds") or {}
        members[d] = {"est": mean.get("est"), "lo": mean.get("lo"), "hi": mean.get("hi"), "p": mean.get("p"),
                      "n_users": mean.get("n_users"), "descriptive_min_n": mean.get("descriptive_min_n"),
                      "complete": eb.get("complete"), "sigma_seed_rule": s.get("sigma_seed_rule")}
        if eb.get("complete") is not True:
            why.append(f"{d}: E-B incomplete (a seed missing or excluded)")
        if mean.get("descriptive_min_n"):
            why.append(f"{d}: E-B on fewer than {fr.MIN_N} users")
        elif not fr._fin(mean.get("p")):
            why.append(f"{d}: E-B has no p-value")
    complete = not why
    adj = fr.holm({d: v["p"] for d, v in members.items()}) if complete else {}
    for d, v in members.items():
        v["p_holm"] = adj.get(d) if complete else None
        v["confirmed"] = (bool(adj.get(d) is not None and adj[d] < ALPHA and v["sigma_seed_rule"] is True)
                          if complete else None)
    return {"hypothesis": "E-B (A3 section 11): Holm over the four Qwen domains of the seed-averaged dUAUC(FT - ZS); "
                          "confirmed = Holm p < 0.05 and the sigma_seed rule",
            "complete_family": complete, "reason": None if complete else "; ".join(why),
            "m": len(members) if complete else None, "members": members,
            "rule": "p_holm and confirmed are null unless ML-1M, Toys, Video_Games and Sports are all given with a "
                    "complete, non-descriptive E-B (fill_paper.eb_family refuses otherwise)"}


def _ed_copy(doc: dict) -> dict:
    eds = doc["registered_pooled"].get("E_D") or {}
    return {reg: (fr.ed_holm_family(eds[reg], reg) if isinstance(eds.get(reg), dict)
                  and eds[reg].get("available") is not False else _na("E-D not available")) for reg in REGS}


def _p1_copy(doc: dict) -> dict:
    p1 = doc["registered_pooled"].get("P1") or {}
    return {"role": p1.get("role"), "available": p1.get("available"), "decision": p1.get("decision"),
            "mean_over_seeds": p1.get("mean_over_seeds"), "per_seed": p1.get("per_seed")}


def ft_wording(ftc: dict | None, ftq: list) -> dict:
    """Addendum 8 section 2, R18: the FT-C / FT-Q labels per dataset and control and the wording conditions."""
    out = {"rule": "addendum 8 section 2: 'fine-tuning mostly teaches the item' only if FT-C reads ITEM_DRIVEN on "
                   "ML-1M and FT-Q reads ITEM_DRIVEN on every dataset on which it was run; a USER_DRIVEN or MIXED label "
                   "means the paper reports the labels per dataset and control and calls the evidence mixed; 'the "
                   "adapter learns item quality from item text' additionally needs FT-Q ITEM_DRIVEN on at least one "
                   "Amazon dataset (R18)",
           "labels": {"FT-C": {}, "FT-Q": {}}}
    missing = []
    if ftc is None:
        missing.append("FT-C: the main-root ML-1M file (--ftc) is not given")
    else:
        b = ftc.get("FT_C_reading") or {}
        if b.get("available") is False or b.get("control") != "FT-C":
            missing.append(f"FT-C ML-1M: {b.get('reason') or 'not an FT-C reading'}")
        else:
            out["labels"]["FT-C"]["ml1m"] = b.get("label")
    given = {}
    for doc in ftq:
        dom = doc["meta"]["domain"]
        b = doc.get("FT_C_reading") or {}
        given[dom] = True
        if b.get("available") is False or b.get("control") != "FT-Q":
            missing.append(f"FT-Q {dom}: {b.get('reason') or 'not an FT-Q reading'}")
        else:
            out["labels"]["FT-Q"][dom] = b.get("label")
    for req in ("ml1m", "toys"):                       # addendum 8 section 3: ML-1M and Toys are required
        if req not in given:
            missing.append(f"FT-Q {req}: file not given (required by addendum 8 section 3)")
    labels = list(out["labels"]["FT-C"].values()) + list(out["labels"]["FT-Q"].values())
    if missing:
        out.update(fine_tuning_mostly_teaches_the_item=None, item_quality_from_item_text=None, evidence_mixed=None,
                   complete=False, reason="; ".join(missing))
        return out
    ok = all(lab == "ITEM_DRIVEN" for lab in labels)
    out.update(complete=True, reason=None, fine_tuning_mostly_teaches_the_item=ok,
               item_quality_from_item_text=bool(ok and any(out["labels"]["FT-Q"].get(d) == "ITEM_DRIVEN"
                                                           for d in AMAZON)),
               evidence_mixed=any(lab in ("USER_DRIVEN", "MIXED") for lab in labels))
    return out


def summarize(a) -> dict:
    cur = code_sha1().get("ftgrid_extra.py")
    files = [load_checked(p, "files", a.n_boot, cur) for p in a.files]
    ml = load_checked(a.ml1m, "ml1m", a.n_boot, cur) if a.ml1m else None
    ftc = load_checked(a.ftc, "ftc", a.n_boot, cur) if a.ftc else None
    llama = [load_checked(p, "llama", a.n_boot, cur) for p in (a.llama or [])]
    ftq = [load_checked(p, "ftq", a.n_boot, cur) for p in (a.ftq or [])]
    if ml is not None and ftc is not None and ml["_file"]["sha1"] != ftc["_file"]["sha1"]:
        _refuse("--ml1m and --ftc must be the same main-root ML-1M file")
    seen = set()
    for doc in files + ([ml] if ml else []) + llama + ftq:
        key = (doc["meta"]["root_label"], doc["meta"]["domain"])
        if key in seen:
            _refuse(f"two files for root {key[0]}, domain {key[1]}")
        seen.add(key)
    # a panel whose fine-tuned runs were not run (a registered cut, A3 section 10) is outside the FT members of every
    # family, only with --not_run and its pilot-log record; its ZS regime was run and stays a member of E-H (R14)
    not_run, by_dom = {}, {doc["meta"]["domain"]: doc for doc in files}
    for d in a.not_run or []:
        if d in not_run:
            _refuse(f"--not_run {d} is given twice")
        if not not_run_recorded(a.pilot_log, d):
            _refuse(f"--not_run {d}: the pilot log has no line with the token '{NOT_RUN_TOKEN} {d}' (the record of a "
                    "registered cut; give the log with --pilot_log)")
        if d not in by_dom:
            _refuse(f"--not_run {d}: no --files file of {d} is given")
        not_run[d] = None
    for doc in files:
        d, st = doc["meta"]["domain"], ft_status(doc)
        if st == "unknown":
            _refuse(f"{doc['_file']['name']}: the run record of the FT models s0-s2 is missing")
        if st == "not_run" and d not in not_run:
            _refuse(f"{doc['_file']['name']}: the fine-tuned runs of {d} were not run (every FT model is absent); pass "
                    f"--not_run {d} once the cut is recorded in the pilot log as '{NOT_RUN_TOKEN} {d}' (a panel "
                    "without FT runs never enters a family with p = 1)")
        if d in not_run and st != "not_run":
            _refuse(f"--not_run {d}, but {doc['_file']['name']} holds fine-tuned runs of {d}")
        if d in not_run:
            not_run[d] = {"reason": f"registered cut recorded in the pilot log ('{NOT_RUN_TOKEN} {d}'); no FT model "
                                    f"of {d} has a scoring run",
                          "outside": [f"E_F:{d}", f"E_H:{d}:FT", f"E_J:{d}"],
                          "kept": [f"E_H:{d}:ZS (the ZS regime was run: a member per panel run and regime)"]}
    inputs = [{"file": d["_file"]["name"], "sha1": d["_file"]["sha1"], "role": d["_file"]["role"],
               "root_label": d["meta"]["root_label"], "domain": d["meta"]["domain"], "backbone": d["meta"]["backbone"],
               "code_sha1": d["meta"].get("code_sha1")} for d in files + ([ml] if ml else []) + ([ftc] if ftc else [])
              + llama + ftq]
    ef, eh, ej, ft_f, ft_h, ft_j = {}, {}, {}, {}, {}, {}
    for doc in files:
        d = doc["meta"]["domain"]
        cut = d in not_run                                  # the FT members of a cut panel do not exist
        blk = (doc["E_F"].get("FT") or {}).get("G_LLM_given_CF") if isinstance(doc["E_F"].get("FT"), dict) else None
        if not cut:
            ef[d], ft_f[d] = _member((blk or {}).get("mean_over_seeds"), (blk or {}).get("seeds"), True,
                                     f"{d} E-F FT"), True
        for reg in REGS:
            if cut and reg == "FT":
                continue
            st = (((doc["E_H"].get(reg) or {}).get("strata") or {}).get("sparse") or {}).get("G_prior")
            eh[f"{d}:{reg}"] = _member((st or {}).get("mean_over_seeds"), (st or {}).get("seeds"), reg == "FT",
                                       f"{d} E-H {reg} sparse")
            ft_h[f"{d}:{reg}"] = reg == "FT"
        blk = (doc["E_J"].get("FT") or {}).get("dUAUC_L_minus_q_hat") if isinstance(doc["E_J"].get("FT"), dict) else None
        if not cut:
            ej[d], ft_j[d] = _member((blk or {}).get("mean_over_seeds"), (blk or {}).get("seeds"), True,
                                     f"{d} E-J FT"), True
    fams = {"E_F": holm_family(ef, ft_f, +1, "H-F (directional): in the FT regime G_LLM_given_CF = dUAUC(M4 - M3) > 0; "
                                             "family E-F = {G_LLM_given_CF, FT, per Qwen Amazon panel run}"),
            "E_H": holm_family(eh, ft_h, +1, "H-S (directional): on sparse rows G_prior = dUAUC(M1 - M0) > 0; family E-H = "
                                             "{G_prior on sparse rows, per Qwen Amazon panel run and regime (ZS, FT)}; "
                                             "a panel-regime below 150 users is outside"),
            "E_J": holm_family(ej, ft_j, None, "H-J (two-sided): in the FT regime dUAUC(L - q-hat) is not 0; family E-J = "
                                               "{dUAUC(L - q-hat), FT, per Qwen Amazon panel run}; the sign of a "
                                               "confirmed member is reported as found")}
    main_docs = files + ([ml] if ml else [])
    ed_f = {f"{d['meta']['domain']}:{d['meta']['backbone']}": _ed_copy(d) for d in main_docs}
    robust = {}
    for doc in main_docs:
        d = doc["meta"]["domain"]
        fam_d = ed_f[f"{d}:{doc['meta']['backbone']}"]
        for reg in REGS:
            ew = doc["E_W"].get(reg) or {}
            rg = ew.get("reading_G")
            if not isinstance(rg, dict):
                continue
            conf = (fam_d[reg].get("confirmed") or {}).get("G") if fam_d[reg].get("available") is not False else None
            robust[f"{d}:{reg}"] = {
                "G": {**rg, "registered_confirmed": conf,
                      "registered_confirmed_source": "the E-D Holm family (ftgrid_report.ed_holm_family on the "
                                                     "registered_pooled block)"},
                "G_CF": ew.get("reading_G_CF")}
    sensitivity = {}
    if ml is not None:
        pw = ml["E_W"].get("P1_wu") or {}
        robust["ml1m:P1"] = pw.get("reading_P1") or _na(pw.get("reason") or "P1_wu not available")
        sensitivity["P1_wu"] = {"decision": pw.get("decision"), "mean_over_seeds": pw.get("mean_over_seeds"),
                                "per_seed": pw.get("per_seed"),
                                "note": "a within-user sensitivity estimator of P1's quantity (A3-6 item 2); not P1"}
    llama_out = {}
    for doc in llama:
        d = doc["meta"]["domain"]
        llama_out[d] = {"backbone": doc["meta"]["backbone"], "E_D": _ed_copy(doc),
                        "P1_replication": (_p1_copy(doc) if d == "ml1m" else _na("P1's replication is ML-1M's")),
                        "note": "the Llama program (A3 section 8): replication and descriptive values only; never a "
                                "member of an A3-6 family and never under registered.P1"}
    outside = []
    for doc in ([ml] if ml else []) + llama:
        m = doc["meta"]
        why = ("ML-1M (exploratory; never a member)" if m["root_label"] == "main" else
               "the Llama backbone (descriptive; never a member)")
        vals = {}
        for reg in REGS:
            f_ = doc["E_F"].get(reg) or {}
            h_ = (((doc["E_H"].get(reg) or {}).get("strata") or {}).get("sparse") or {})
            j_ = doc["E_J"].get(reg) or {}
            vals[reg] = {"G_LLM_given_CF": ((f_.get("G_LLM_given_CF") or {}).get("mean_over_seeds")),
                         "G_prior_sparse": ((h_.get("G_prior") or {}).get("mean_over_seeds")),
                         "dUAUC_L_minus_q_hat": ((j_.get("dUAUC_L_minus_q_hat") or {}).get("mean_over_seeds"))}
        outside.append({"file": doc["_file"]["name"], "root_label": m["root_label"], "domain": m["domain"],
                        "backbone": m["backbone"], "why": why, "values": vals})
    for doc in files:
        zs = {"G_LLM_given_CF": (((doc["E_F"].get("ZS") or {}).get("G_LLM_given_CF") or {}).get("mean_over_seeds")),
              "dUAUC_L_minus_q_hat": (((doc["E_J"].get("ZS") or {}).get("dUAUC_L_minus_q_hat") or {})
                                      .get("mean_over_seeds"))}
        outside.append({"file": doc["_file"]["name"], "root_label": "main", "domain": doc["meta"]["domain"],
                        "backbone": doc["meta"]["backbone"],
                        "why": "the ZS regime of E-F and E-J (descriptive; never a member)", "values": {"ZS": zs}})
    out = {"spec": SPEC, "n_boot": a.n_boot, "inputs": inputs, "families": fams, "not_run": not_run,
           "registered": {"E_B": eb_family_copy({d["meta"]["domain"]: d for d in main_docs}), "E_D": ed_f,
                          "P1": (_p1_copy(ml) if ml is not None else
                                 _na("no --ml1m file: P1 is the single test of ML-1M, Qwen3-8B, main root")),
                          "note": "copied through from the domain files' registered_pooled blocks with the report's "
                                  "functions (R15); the per-domain reports stay the registered record"},
           "robust_readings": robust, "sensitivity": sensitivity, "llama": llama_out,
           "descriptive_outside_families": outside,
           "readings": [r for r in READINGS if r.startswith(("R1 ", "R13 ", "R14 ", "R15 ", "R16 ", "R17 ", "R18 "))]}
    out["ft_wording"] = (ft_wording(ftc, ftq) if (ftc is not None or ftq)
                         else _na("not requested (give --ftc and --ftq)"))
    return out


# ---------------------------------------------------------------- CLI
def _build_args(b) -> None:
    b.add_argument("--domain", required=True)
    b.add_argument("--split", required=True, help="panels/<d>/ftgrid_split.json")
    b.add_argument("--panels", required=True, help="panels/<d>/")
    b.add_argument("--scores_root", required=True, help="scores/ (or scores/<d>/)")
    b.add_argument("--models", default="zeroshot,s0,s1,s2", help="comma list of " + ",".join(fr.MODELS))
    b.add_argument("--raw", default=None, help="raw data root (data/raw): CF references and q-hat_T")
    b.add_argument("--out", required=True, help="extra/<d>.json; the tables go to <stem>_tables.csv next to it")
    b.add_argument("--refs", default=None, help="ftgrid_report's CF reference cache (read if it matches, else written)")
    b.add_argument("--root_label", choices=ROOT_LABELS, default="main",
                   help="main = outputs/confrec/ftgrid, llama = ftgrid_llama, teacher = ftgrid_q (addendum 8)")
    b.add_argument("--pilot_log", default=None, help="the pilot log: FT_C_reading needs the A3-6 item 11 record in it")
    b.add_argument("--n_boot", type=int, default=fr.N_BOOT)
    b.add_argument("--seed", type=int, default=fr.SEED)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    _build_args(sub.add_parser("build", help="items 2-8 for one domain and backbone"))
    _build_args(sub.add_parser("check_resume", help="exit 0 iff --out is a complete build of the current inputs"))
    s = sub.add_parser("summarize", help="the A3-6 Holm families across the Qwen Amazon domain files")
    s.add_argument("--files", nargs="+", required=True, help="main-root extra/<d>.json of toys, games, sports")
    s.add_argument("--ml1m", default=None, help="main-root extra/ml1m.json (never a family member: E-B, E-D, P1 copies)")
    s.add_argument("--llama", nargs="*", default=None, help="Llama-root extra files (descriptive; P1 replication)")
    s.add_argument("--ftc", default=None, help="main-root extra/ml1m.json: the FT-C condition of the FT wording")
    s.add_argument("--ftq", nargs="*", default=None, help="teacher-root extra files: the FT-Q conditions (addendum 8)")
    s.add_argument("--n_boot", type=int, default=REGISTERED_N_BOOT, help="the n_boot every file must have")
    s.add_argument("--not_run", action="append", choices=AMAZON, default=None,
                   help="a dataset whose fine-tuned runs were cut (needs the pilot-log line 'FTEXTRA_NOT_RUN <d>')")
    s.add_argument("--pilot_log", default=None, help="the pilot log holding the FTEXTRA_NOT_RUN records")
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd in ("build", "check_resume") and a.n_boot < 0:
        ap.error("--n_boot must be >= 0")
    return a


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out)
    if a.cmd == "check_resume":
        ok, why = check_resume(a)
        print(("resume: " if ok else "rebuild: ") + why)
        return 0 if ok else 1
    if a.cmd == "build":
        res = write_build(out, build(a))
        print(f"wrote {tables_path(out)} and {out}")
        return res
    res = strict_json(summarize(a))
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(out, json.dumps(res, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}")
    for k, f in res["families"].items():
        print(f"{k}: m = {f['m']}, confirmed = {f['n_confirmed']}, missing members = {f['n_missing_members']}")
    return res


if __name__ == "__main__":
    r = main()
    sys.exit(r if isinstance(r, int) else 0)
