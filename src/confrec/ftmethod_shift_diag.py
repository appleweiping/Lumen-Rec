"""Train/test-shift diagnostic of the method slot (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_10.md section 2; read with addendum 3
item 3 and addendum 5 item 1, which state the mismatch as a limitation). Outcome-free and descriptive: no hypothesis, no family.

The prior-offset LoRA shifts, during training, the logit of the single answer token tok("Yes") (id 9454 for the Qwen3 tokenizer)
by b z(q-hat) before the full-vocabulary softmax. The registered test-time score is the scorer's logit(Yes) - logit(No), summed
over ALL yes ids and all no ids, plus b z(q-hat). The two agree when the answer token carries nearly all of the yes mass. For
every dataset, before its slot report is built, this module records, for each prior-offset adapter o0-o2, on the FIRST 1,000
rows (in panel order) of the dataset's EVAL CAL candidates (the candidates of eval.jsonl with timestamp < T_d, T_d from
ftgrid_split.json), the per-id top-50 log-probabilities of the next token under the `like` question exactly as stage 4 of
run_ftmethod.sh scores it, and reports

    s   = P(answer token) / P(all yes ids)        the share of the yes mass on the answer token, per row
    R   = logit(Yes) - logit(No) + b z            the registered score (read_yes_no of the scorer, b from offset.json, z from
                                                  eval_qhat.csv.gz with the manifest constants, as ftmethod_report builds it)
    C   = R + log(s + (1 - s) exp(-b z))          the training-consistent score
    (C is the logit of the summed yes mass over the summed no mass when only the answer token is shifted by b z:
     log((e^{bz} s Y + (1 - s) Y) / N) = log(Y / N) + bz + log(s + (1 - s) e^{-bz}) with Y = the yes mass and N = the no mass)

the mean, median and 1st percentile of s per adapter and pooled, the number of within-user pairs whose order differs between R
and C, and the UAUC of R and of C on those rows (CAL labels exist).
Reading rule (addendum 10 section 2): if the 1st percentile of s is at least 0.99 the registered score is read as equivalent to
the training-consistent one for ranking; if it is lower, a FAIL of that dataset is reported as inconclusive for the method (the
test-time rule then works against it) and counts as a failure for the kill rule exactly as before, while a PASS stands (the
mismatch only works against the method). Operationalisation: "the 1st percentile" is numpy's linear percentile of the s of the
usable rows of ONE adapter; the dataset reads EQUIVALENT only if every one of the three adapters has p1 >= 0.99 (compared as
p1 >= 0.99 - 1e-12, the tolerance of addendum 10 item 3), NOT_EQUIVALENT if one has less, UNDETERMINED if the diagnostic is not OK.
The pooled p1 of the three adapters is reported beside it and never decides.

    python -m src.confrec.ftmethod_shift_diag score  --domain ml1m --seed 0 --split SPLIT --panels PANELS --method_dir M \
        --model <Qwen3-8B> --variant V [--n_rows 1000] [--dry_run]
    python -m src.confrec.ftmethod_shift_diag report --domain ml1m --split SPLIT --panels PANELS --method_dir M [--dry_run]
    python -m src.confrec.ftmethod_shift_diag verify --domain ml1m --split SPLIT --panels PANELS --method_dir M [--dry_run]
    python -m src.confrec.ftmethod_shift_diag record --pilot_log docs/sigir/PILOT_LOG.md --files <files> [--print] [--append]

score    (GPU, one adapter per process, as stage 4 scores one adapter per process) the prompts of the selected rows, rendered by
         the scorer's own code (pyes_scorer.record_requests: the variant, the registered history window, the `like` question),
         sent through pyes_scorer.load_vllm and pyes_scorer.score_ids with a read callback that keeps the whole top-50 dict next
         to prompting.read_yes_no's reading, so the loading, the arguments (float16, max_model_len 4096, top-50 logprobs, seed 0,
         vLLM LoRA of M/adapters/o<seed>) and the extraction are the scorer's by import; no vLLM call is written here. Writes
         M/shift_diag/o<seed>/top50.jsonl.gz (one line per row: ids and [token id, logprob] pairs) and, last, meta.json (the
         completion marker: panel, split, selection, adapter-weights and manifest sha1, yes / no / answer ids, E1 facts, run key);
         a finished run with the same run key is skipped. --dry_run replaces the model and tokenizer by a CPU stand-in (several yes
         ids; --dry_share and --dry_low_frac steer s) and marks everything dry_run.
report   (CPU) M/shift_diag/report.json and report_tables.csv from the three scoring directories and the inputs of ftmethod_report
         (load_eval_qhat, load_offset: the same checks and constants): status OK, INCOMPLETE (an adapter not scored) or INVALID
         (an input problem: sha1, selection, weights, manifest, ids, censoring). Strict JSON, no wall-clock, host or path.
verify   (CPU) recomputes the report from the files and requires the one on disk to be byte-identical (a marker or a stored
         number proves nothing about a file that was replaced afterwards), status OK, the same dry-run mode as --dry_run says and,
         for a real run, the registered 1,000 rows. Exit 0 or 1; run_ftmethod.sh stage 5 refuses (exit 4) without it.
record   0 iff the pilot log holds the sha1 of every listed file (the diagnostic's own record); --print writes the lines.
Exit codes: 0 done; 1 error or an inconsistent input; 2 refused input; 4 refused by a registered gate (record).
torch, transformers, peft and vllm are never imported by this module except through pyes_scorer.load_vllm in `score`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.confrec import forensics as fx
from src.confrec import ftgrid_report as fr
from src.confrec import ftmethod_report as fm
from src.confrec import pyes_scorer as ps
from src.confrec import train_lora_offset as tlo
from src.confrec.prompting import (chat_ids, panel_kind_of, read_yes_no, resolve_hist_len, short_history_error,
                                   yes_no_ids)
from src.confrec.stats import strict_json

SPEC = ("idea-stage/PREREG_AMENDMENT_3_ADDENDUM_10.md section 2 (train/test-shift diagnostic); PREREG_AMENDMENT_3.md section 7; "
        "PREREG_AMENDMENT_3_ADDENDUM_3.md item 3; PREREG_AMENDMENT_3_ADDENDUM_5.md item 1")
DATASETS = fm.DATASETS
SEEDS = fm.SEEDS                           # the prior-offset adapters o0-o2
N_ROWS = 1000                              # the first 1,000 CAL rows in panel order
TOPK = 50                                  # per-id top-50 log-probabilities
P1_MIN, P1_TOL = 0.99, 1e-12               # the reading rule: the 1st percentile of s >= 0.99 (same tolerance as addendum 10 item 3)
ANSWER = "Yes"                             # the answer token of training (YesNoSet: "Yes" for a like)
QUESTION = "like"
DIR = "shift_diag"                         # everything below <method_dir>/shift_diag/
TOP_FILE, META_FILE, REPORT_FILE, TABLE_FILE = "top50.jsonl.gz", "meta.json", "report.json", "report_tables.csv"
SCORE_FORMAT, REPORT_FORMAT = "ftmethod_shift_diag_scores_v1", "ftmethod_shift_diag_v1"
CODE_FILES = ("ftmethod_shift_diag.py", "ftmethod_report.py", "train_lora_offset.py", "pyes_scorer.py", "prompting.py",
              "ftgrid_report.py", "forensics.py")
REPORT_COLS = ("dataset", "adapter", "n_rows", "n_used", "b", "s_mean", "s_median", "s_p1", "s_min", "share_s_below_0.99",
               "abs_delta_max", "UAUC_R", "UAUC_C", "dUAUC_C_minus_R", "n_users", "n_pairs", "n_pairs_label_discordant",
               "n_order_differs", "n_order_differs_label_discordant", "reading")
COMMANDS = ("score", "report", "verify", "record")
DRY_SHARE, DRY_LOW_FRAC = 0.9995, 0.0      # the stand-in's defaults: s ~ 0.9995 on every row, no low-share tail

OPERATIONALIZATIONS = (
    "rows: the pairs (user_id, item_id) of eval.jsonl in panel order (row order, then candidate order) whose candidate "
    "timestamp is < T (ftgrid_split.json): the first 1,000 of them (all, if fewer); the last user of the selection may be cut "
    "(n_users_cut); selection_sha1 is the sha1 of their 'user_id TAB item_id' lines",
    "prompts: pyes_scorer.record_requests(row, ['like'], hist_len, variant, 'rated', 'yesno') for the selected candidates, "
    "hist_len = the variant's registered window, the adapter's train_config.json variant and window checked as the scorer "
    "checks them; one process (one vLLM engine with the adapter) per adapter",
    "R = prompting.read_yes_no (lp_yes - lp_no over all yes / no ids present in the top-50, a missing side imputed by the "
    "scorer's floor rule) + b z, b from offset.json (ftmethod_report.load_offset: the registered optimizer group), z from "
    "eval_qhat.csv.gz with the manifest constants (ftmethod_report.load_eval_qhat): the score of stage 4 up to the six "
    "decimals of scores.csv.gz",
    "s = exp(logprob(answer token) - logsumexp(logprob of the yes ids present in the top-50)); the answer token absent from the "
    "top-50 while another yes id is present counts as s = 0; no yes id present: s undefined (the row is excluded from s, C and "
    "every comparison and counted: n_yes_side_absent); C = R + logaddexp(log s, log(1 - s) - b z)",
    "usable rows: R and s finite (censored 0 or 1 with a yes id present); every statistic below is on the same rows",
    "UAUC = the mean over the users with both classes among the usable rows of the per-user AUC (ties 1/2, "
    "ftgrid_report.user_aucs); pairs = unordered pairs of usable rows of one user; 'order differs' = sign(R_i - R_j) != "
    "sign(C_i - C_j) (a strict reversal is counted apart); label-discordant pairs are those with different labels (the pairs "
    "UAUC counts)",
    "s statistics: mean, median, numpy's linear 1st percentile (and the 5th), minimum and the share below 0.99, per adapter and "
    "pooled over the three adapters; reading: EQUIVALENT iff p1 >= 0.99 - 1e-12 for every adapter",
)


class DiagError(Exception):
    """A refused or inconsistent input: the message goes to stderr and `code` is the exit status."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------- the answer token and the shares
def answer_id(tok, word: str, allowed) -> int:
    """The id of the answer token of training (train_lora_yesno.answer_id: tok(word) is one id and a yes / no id of the
    scorer); derived from the tokenizer, never hard-coded."""
    ids = tok(word, add_special_tokens=False)["input_ids"]
    if len(ids) != 1 or ids[0] not in allowed:
        raise DiagError(f"{word!r} tokenizes to {ids}, not one id from the scorer's set {sorted(allowed)}")
    return int(ids[0])


def _logsumexp(v) -> float:
    m = max(v)
    return m + math.log(sum(math.exp(x - m) for x in v))


def row_reading(top, yes_ids, no_ids, ans_id: int) -> dict:
    """One row from its top-k list [(token id, logprob), ...]: the scorer's reading (lp_yes, lp_no, mass, censored) and the
    answer-token share s (None where no yes id is in the top-k)."""
    obj = {int(t): SimpleNamespace(logprob=float(v)) for t, v in top}
    lp_yes, lp_no, mass, cens = read_yes_no(obj, yes_ids, no_ids)
    present = [o.logprob for t, o in obj.items() if t in yes_ids and math.isfinite(o.logprob)]
    s = None
    if present:
        tok = obj.get(ans_id)
        s = math.exp(tok.logprob - _logsumexp(present)) if tok is not None and math.isfinite(tok.logprob) else 0.0
        s = min(1.0, max(0.0, s))
    return {"lp_yes": lp_yes, "lp_no": lp_no, "mass": mass, "censored": cens, "s": s}


def consistent_shift(R, s, bz) -> np.ndarray:
    """C = R + log(s + (1 - s) exp(-b z)) = R + logaddexp(log s, log(1 - s) - b z), NaN where R or s is."""
    R, s, bz = (np.asarray(x, float) for x in (R, s, bz))
    with np.errstate(divide="ignore", invalid="ignore"):
        return R + np.logaddexp(np.log(s), np.log1p(-s) - bz)


def share_stats(s) -> dict:
    """n, mean, median, 1st and 5th percentile (numpy linear), minimum and the share below 0.99 of the finite s."""
    s = np.asarray(s, float)
    s = s[np.isfinite(s)]
    if not len(s):
        return {"n": 0, "mean": None, "median": None, "p1": None, "p5": None, "min": None, "share_below_0.99": None}
    return {"n": int(len(s)), "mean": float(s.mean()), "median": float(np.median(s)), "p1": float(np.percentile(s, 1)),
            "p5": float(np.percentile(s, 5)), "min": float(s.min()), "share_below_0.99": float((s < P1_MIN).mean())}


def pair_counts(users, label, R, C, ok) -> dict:
    """Within-user pairs of the usable rows: all, label-discordant, those whose order differs between R and C (sign
    differs; a strict reversal apart) and the label-discordant ones among them."""
    users, label, R, C = np.asarray(users), np.asarray(label, int), np.asarray(R, float), np.asarray(C, float)
    ok = np.asarray(ok, bool)
    n = disc = differs = differs_disc = reversals = 0
    for idx in fx._groups(users[ok]).values():
        rows = np.flatnonzero(ok)[idx]
        if len(rows) < 2:
            continue
        i, j = np.triu_indices(len(rows), 1)
        dR, dC = (R[rows][:, None] - R[rows][None, :])[i, j], (C[rows][:, None] - C[rows][None, :])[i, j]
        diff_lab = label[rows][i] != label[rows][j]
        changed = np.sign(dR) != np.sign(dC)
        n += len(i)
        disc += int(diff_lab.sum())
        differs += int(changed.sum())
        differs_disc += int((changed & diff_lab).sum())
        reversals += int((dR * dC < 0).sum())
    return {"n_pairs": n, "n_pairs_label_discordant": disc, "n_order_differs": differs,
            "n_order_differs_label_discordant": differs_disc, "n_strict_reversals": reversals,
            "share_order_differs": (differs / n) if n else None,
            "share_order_differs_label_discordant": (differs_disc / disc) if disc else None}


def uauc_of(score, label, users, ok) -> tuple:
    """(UAUC, number of users with both classes) of a score on the rows in `ok`."""
    per = fr.user_aucs(score, label, users, ok)
    return (float(np.mean(list(per.values()))) if per else None), len(per)


def adapter_stats(tops, stored_cens, yes_ids, no_ids, ans_id: int, z, b: float, label, users) -> tuple:
    """(the report block of one adapter, the arrays R, C, s and the usable-row mask) from the stored top-k lists
    ([] = no list: an overlength prompt)."""
    n = len(tops)
    lp_yes, lp_no, mass = (np.full(n, np.nan) for _ in range(3))
    s, cens = np.full(n, np.nan), np.array(stored_cens, int)
    mismatch = 0
    for i, top in enumerate(tops):
        if not len(top):
            continue
        r = row_reading(top, yes_ids, no_ids, ans_id)
        lp_yes[i], lp_no[i], mass[i] = r["lp_yes"], r["lp_no"], r["mass"]
        if r["censored"] != cens[i]:
            mismatch += 1
        if r["s"] is not None:
            s[i] = r["s"]
    z = np.asarray(z, float)
    R = lp_yes - lp_no + float(b) * z
    C = consistent_shift(R, s, float(b) * z)
    ok = np.isfinite(R) & np.isfinite(s)
    label, users = np.asarray(label, int), np.asarray(users)
    u_r, n_users = uauc_of(R, label, users, ok)
    u_c, _ = uauc_of(C, label, users, ok)
    delta = (C - R)[ok]
    n_main = int(n)
    out = {"b": float(b), "n_rows": n_main, "n_used": int(ok.sum()),
           "censored": {str(c): int((cens == c).sum()) for c in range(4)},
           "n_yes_side_absent": int((np.isfinite(R) & ~np.isfinite(s)).sum()), "censoring_mismatch_rows": int(mismatch),
           # the scorer's mean_yes_no_mass: the mass of every row, 0 for a row it did not score (censored 2 and 3 carry 0)
           "mean_yes_no_mass": float(np.where(np.isfinite(mass), mass, 0.0).mean()) if n_main else None,
           "s": share_stats(s[ok]),
           "delta_C_minus_R": {"mean": float(delta.mean()) if len(delta) else None,
                               "abs_max": float(np.abs(delta).max()) if len(delta) else None,
                               "abs_mean": float(np.abs(delta).mean()) if len(delta) else None},
           "bz": {"sd": float(np.std((float(b) * z)[ok])) if ok.any() else None,
                  "abs_max": float(np.abs(float(b) * z)[ok].max()) if ok.any() else None},
           "UAUC": {"R": u_r, "C": u_c, "C_minus_R": (u_c - u_r) if (u_r is not None and u_c is not None) else None,
                    "n_users": n_users},
           "pairs": pair_counts(users, label, R, C, ok)}
    out["E1"] = {"censored2_share": float(out["censored"]["2"] / n_main) if n_main else None,
                 "overlength": out["censored"]["3"], "mean_yes_no_mass": out["mean_yes_no_mass"],
                 "ok": bool(n_main > 0 and out["censored"]["2"] <= 0.005 * n_main and out["censored"]["3"] == 0
                            and out["mean_yes_no_mass"] is not None and out["mean_yes_no_mass"] >= 0.95)}
    return out, {"R": R, "C": C, "s": s, "ok": ok}


def reading(p1_by_adapter: dict, status: str) -> dict:
    """The reading rule of addendum 10 section 2 from the 1st percentile of s of each adapter."""
    vals = [p1_by_adapter.get(f"o{k}") for k in SEEDS]
    rule = ("p1 of s >= 0.99 for every adapter: the registered score is read as equivalent to the training-consistent one for "
            "ranking; lower: a FAIL of the dataset is reported as inconclusive for the method (the test-time rule then works "
            "against it) and counts as a failure for the kill rule exactly as before; a PASS stands")
    if status != "OK" or any(v is None or not math.isfinite(v) for v in vals):
        label = "UNDETERMINED"
    elif all(v >= P1_MIN - P1_TOL for v in vals):
        label = "EQUIVALENT"
    else:
        label = "NOT_EQUIVALENT"
    return {"rule": rule, "p1_threshold": P1_MIN, "tolerance": P1_TOL, "p1_by_adapter": p1_by_adapter,
            "p1_min": min([v for v in vals if v is not None and math.isfinite(v)], default=None), "label": label,
            "equivalent_for_ranking": {"EQUIVALENT": True, "NOT_EQUIVALENT": False}.get(label),
            "fail_reading": {"EQUIVALENT": "a FAIL stands as a failure of the method",
                             "NOT_EQUIVALENT": "a FAIL is inconclusive for the method (and still counts as a failure for the "
                                               "kill rule)", "UNDETERMINED": "not determined"}[label],
            "pass_reading": "a PASS stands (the mismatch only works against the method)"}


# ---------------------------------------------------------------- inputs: panel, selection, prompts
def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def sha1_lines(lines) -> str:
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()


def select_rows(cx, n_rows: int = N_ROWS) -> np.ndarray:
    """Indices (into the EVAL pairs, panel order) of the first n_rows CAL pairs: candidate timestamp < T."""
    return np.flatnonzero(cx.cal)[:n_rows]


class Panel:
    """eval.jsonl as the diagnostic uses it: the rows, the pair context of ftgrid_report, the CAL selection and the prompts."""

    def __init__(self, panels_dir, split_path, n_rows: int):
        self.split = fr.read_json(split_path)
        self.T, self.variant = float(self.split["T"]), self.split.get("variant")
        self.eval_p = Path(panels_dir) / "eval.jsonl"
        if not self.eval_p.is_file():
            raise DiagError(f"{self.eval_p}: no eval.jsonl", 2)
        self.rows = fx.read_jsonl(self.eval_p)
        self.eval_sha, self.split_sha = fx.file_sha1(self.eval_p), fx.file_sha1(split_path)
        self.cx = fr.make_ctx(self.rows, self.T, [], None)
        self.sel = select_rows(self.cx, n_rows)
        counts = np.array([len(r["candidate_item_ids"]) for r in self.rows], int)
        self.row_of = np.repeat(np.arange(len(self.rows)), counts)
        self.cand_of = self.cx.E["cand"]
        self.n_rows_requested = int(n_rows)
        self.keys = [f"{self.cx.users[p]}\t{self.cx.item[p]}" for p in self.sel]
        self.selection_sha1 = sha1_lines(self.keys)
        self.n_cal_rows = int(self.cx.cal.sum())
        kinds = sorted({panel_kind_of(r) for r in self.rows})
        if kinds != ["rated"]:
            raise DiagError(f"{self.eval_p}: the diagnostic reads rated panels, this one has the kinds {kinds}", 2)
        self.kind = "rated"

    def users_cut(self) -> int:
        """Users of the selection that have CAL rows beyond it (the cut falls inside the last user's rows)."""
        if not len(self.sel):
            return 0
        last = self.cx.users[self.sel[-1]]
        return int(int(((self.cx.users == last) & self.cx.cal).sum()) > int((self.cx.users[self.sel] == last).sum()))

    def prompts(self, variant: str, hist_len: int) -> list:
        cache, out = {}, []
        for p in self.sel:
            r = int(self.row_of[p])
            if r not in cache:
                cache[r] = {c: pr for c, _, pr in ps.record_requests(self.rows[r], [QUESTION], hist_len, variant, self.kind,
                                                                    "yesno")}
            out.append(cache[r][int(self.cand_of[p])])
        return out

    def check_variant(self, variant: str, lora) -> int:
        """The scorer's panel and adapter checks (resolve_hist_len, short_history_error, check_lora_variant); the window."""
        if variant != self.variant:
            raise DiagError(f"--variant {variant} is not the variant {self.variant} the split was built under", 2)
        try:
            hist_len = resolve_hist_len(variant, self.kind, None)
        except ValueError as e:
            raise DiagError(str(e), 2) from None
        short = short_history_error(variant, self.kind, hist_len, max(len(r.get("history") or []) for r in self.rows))
        if short:
            raise DiagError(f"{self.eval_p}: --{short}", 2)
        try:
            ps.check_lora_variant(lora, variant, hist_len)
        except SystemExit as e:
            raise DiagError(str(e), 2) from None
        return hist_len


# ---------------------------------------------------------------- the scoring directory of one adapter
def sdir(method_dir, k: int) -> Path:
    return Path(method_dir) / DIR / f"o{k}"


def adapter_dir(method_dir, k: int) -> Path:
    return Path(method_dir) / "adapters" / f"o{k}"


def run_key(panel: Panel, variant: str, backbone: str, weights_sha1: str, k: int, dry: bool, dtype: str,
            max_model_len: int, vllm_seed: int) -> str:
    return sha1_lines([panel.eval_sha, panel.split_sha, panel.selection_sha1, str(panel.n_rows_requested), variant,
                       backbone, weights_sha1, str(k), "dry" if dry else "real", f"top{TOPK}", dtype, str(max_model_len),
                       str(vllm_seed), QUESTION, ANSWER])


def top_lines(panel: Panel, results) -> bytes:
    """top50.jsonl.gz: one JSON line per selected row (panel pair index, event, user, item, candidate, label, censored, top-50)."""
    lines = []
    for p, res in zip(panel.sel, results):
        top = res[4] if isinstance(res[4], list) else []
        lines.append(json.dumps({"p": int(p), "ev": str(panel.cx.E["ev"][p]), "u": str(panel.cx.users[p]),
                                 "i": str(panel.cx.item[p]), "c": int(panel.cand_of[p]), "y": int(panel.cx.y[p]),
                                 "cens": int(res[3]), "top": [[int(t), float(v)] for t, v in top if math.isfinite(v)]},
                                allow_nan=False))
    return tlo.gz_bytes("\n".join(lines) + "\n")


def read_top_file(path) -> list:
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


# ---------------------------------------------------------------- the stand-in model (CPU, DRY_RUN)
class DryTok:
    """Stand-in tokenizer: ids 1-10 are the yes / no spellings (Yes No ' Yes' ' No' yes no ' yes' ' no' YES NO: five yes ids
    and five no ids, "Yes" is id 1), every other space-separated piece of a prompt gets the id 100 + crc32(piece) mod 10^6, so
    that the ids of a prompt do not depend on which prompts were encoded before it (the scorer and this module see the same ids
    for the same prompt). The chat template wraps the messages and, with thinking off, ends in the closed empty think block
    (Amendment 2 G0)."""
    FIXED = ["", "Yes", "No", " Yes", " No", "yes", "no", " yes", " no", "YES", "NO"]
    pad_token_id, eos_token = 0, "<|end|>"

    def __init__(self):
        self.ids, self.text = {t: i for i, t in enumerate(self.FIXED) if t}, dict(enumerate(self.FIXED))

    def __len__(self):
        return len(self.FIXED)

    def decode(self, ids):
        return "".join(self.text.get(i, f"w{i}") for i in ids)

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, enable_thinking=True, **kw):
        out = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in msg) + "<|assistant|>"
        return out + ("" if enable_thinking else "<think>\n\n</think>\n\n")

    def __call__(self, text, add_special_tokens=False):
        if text in self.ids:
            return {"input_ids": [self.ids[text]]}
        out = []
        for w in text.split(" "):
            if w not in self.ids:
                self.ids[w] = 100 + zlib.crc32(w.encode("utf-8")) % 1_000_000
                self.text[self.ids[w]] = w
            out.append(self.ids[w])
        return {"input_ids": out}


class DryLLM:
    """Stand-in for the vLLM engine: generate() returns, per prompt, a top-50 dict {token id: object with .logprob} of the
    ten yes / no ids and 40 filler ids, deterministic in (adapter tag, prompt ids). The yes mass is split over the five yes ids
    with a share s on id 1 ('Yes'): s ~ share on every row, and on a fraction low_frac of the rows s ~ U(0.5, 0.9)."""
    YES_REST, NO_REST = (3, 5, 7, 9), (4, 6, 8, 10)

    def __init__(self, tag: str, share: float = DRY_SHARE, low_frac: float = DRY_LOW_FRAC, mass: float = 0.995):
        self.tag, self.share, self.low_frac, self.mass = tag, share, low_frac, mass

    def generate(self, prompts, sp, use_tqdm=False, **kw):
        res = []
        for p in prompts:
            ids = p["prompt_token_ids"]
            h = hashlib.sha1((self.tag + "\x1e" + ",".join(map(str, ids))).encode("utf-8")).digest()
            rng = random.Random(int.from_bytes(h[:8], "big"))
            p_yes = self.mass / (1.0 + math.exp(-rng.gauss(0.0, 1.5)))
            p_no = self.mass - p_yes
            low = rng.random() < self.low_frac
            s = rng.uniform(0.5, 0.9) if low else self.share + (1.0 - self.share) * 0.8 * rng.random()
            probs = {1: p_yes * s, 2: p_no * 0.9}
            for t, w in zip(self.YES_REST, (0.4, 0.3, 0.2, 0.1)):
                probs[t] = p_yes * (1.0 - s) * w
            for t, w in zip(self.NO_REST, (0.4, 0.3, 0.2, 0.1)):
                probs[t] = p_no * 0.1 * w
            norm = sum(0.8 ** k for k in range(40))
            for k in range(40):
                probs[200 + k] = (1.0 - self.mass) * 0.8 ** k / norm
            top = {t: SimpleNamespace(logprob=math.log(v)) for t, v in probs.items()}
            res.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=[top])]))
        return res


def dry_model(a) -> SimpleNamespace:
    """What pyes_scorer.load_vllm returns, without a model: a stand-in engine, tokenizer and the TokensPrompt dict form."""
    tag = f"{Path(str(a.model)).name}\x1f{a.lora}"
    return SimpleNamespace(llm=DryLLM(tag, a.dry_share, a.dry_low_frac), tok=DryTok(), gen_kw={},
                           to_prompt=lambda ids: {"prompt_token_ids": ids}, sp=None, version="dryrun-fake")


def vllm_args(a) -> SimpleNamespace:
    """The attributes pyes_scorer.load_vllm reads, with the scorer's own conventions (run_ftmethod.sh stage 4: float16,
    top-50 logprobs, max_model_len 4096, seed 0, the scorer's default gpu_mem)."""
    return SimpleNamespace(model=a.model, lora=a.lora, gpu_mem=a.gpu_mem, max_model_len=a.max_model_len, dtype=a.dtype,
                           topk_logprobs=TOPK, seed=a.vllm_seed)


# ---------------------------------------------------------------- score (GPU, one adapter)
def score_adapter(a, load_model=None) -> dict | None:
    mdir, k = Path(a.method_dir), a.seed
    panel = Panel(a.panels, a.split, a.n_rows)
    if not len(panel.sel):
        raise DiagError(f"{panel.eval_p}: no CAL candidate (timestamp < T = {panel.T})", 2)
    lora = adapter_dir(mdir, k)
    if not lora.is_dir() or not sorted(lora.glob("adapter_model.*")):
        raise DiagError(f"{lora}: no adapter (run_ftmethod.sh stage 3)", 1)
    a.lora = str(a.lora_arg or lora).replace("\\", "/")
    hist_len = panel.check_variant(a.variant, lora)
    weights = fm.adapter_weights_sha1(lora)
    backbone = Path(str(a.model)).name
    key = run_key(panel, a.variant, backbone, weights, k, bool(a.dry_run), a.dtype, a.max_model_len, a.vllm_seed)
    out = sdir(mdir, k)
    meta_p, top_p = out / META_FILE, out / TOP_FILE
    if meta_p.is_file() and top_p.is_file():
        old = json.loads(meta_p.read_text(encoding="utf-8"))
        if old.get("run_key") == key and old.get("scoring_sha1") == fx.file_sha1(top_p):
            print(f"[skip] {out}: scored (run key {key[:12]})")
            return None
    out.mkdir(parents=True, exist_ok=True)
    prompts = panel.prompts(a.variant, hist_len)
    m = (load_model or (dry_model if a.dry_run else (lambda x: ps.load_vllm(vllm_args(x)))))(a)
    yes_ids, no_ids = yes_no_ids(m.tok)
    if not yes_ids or not no_ids:
        raise DiagError(f"tokenizer has no yes ids ({sorted(yes_ids)}) or no ids ({sorted(no_ids)})")
    ans = answer_id(m.tok, ANSWER, yes_ids)
    print(f"yes ids {sorted(yes_ids)}  no ids {sorted(no_ids)}  answer token {ANSWER!r} = {ans}  vllm {m.version}", flush=True)
    id_lists = [chat_ids(m.tok, p) for p in prompts]

    def read(top):
        entries = sorted(((int(t), float(o.logprob)) for t, o in top.items()), key=lambda e: (-e[1], e[0]))
        return (*read_yes_no(top, yes_ids, no_ids), entries)

    res = ps.score_ids(m.llm, m.sp, id_lists, a.max_model_len, read, m.to_prompt, m.gen_kw)
    data = top_lines(panel, res)
    tlo.write_if_changed(top_p, data)
    cens = [int(r[3]) for r in res]
    meta = {"format": SCORE_FORMAT, "spec": SPEC, "domain": a.domain, "seed": k, "run_key": key, "dry_run": bool(a.dry_run),
            "backbone": backbone, "variant": a.variant, "hist_len": hist_len, "question": QUESTION, "topk": TOPK,
            "dtype": a.dtype, "max_model_len": a.max_model_len, "vllm_seed": a.vllm_seed, "vllm_version": m.version,
            "T": panel.T, "eval_sha1": panel.eval_sha, "split_sha1": panel.split_sha, "n_rows_requested": panel.n_rows_requested,
            "n_rows": int(len(panel.sel)), "selection_sha1": panel.selection_sha1, "adapter_weights_sha1": weights,
            "yes_ids": sorted(yes_ids), "no_ids": sorted(no_ids), "answer_token": ANSWER, "answer_token_id": ans,
            "censored": {str(c): cens.count(c) for c in range(4)}, "scoring_sha1": sha1_bytes(data),
            "scores_file": TOP_FILE}
    tlo.write_if_changed(meta_p, tlo.json_bytes(meta))
    print(f"[shift_diag score] {a.domain} o{k}: {len(panel.sel)} rows, censored {meta['censored']}, wrote {out}")
    return meta


# ---------------------------------------------------------------- report (CPU)
def load_scoring(panel: Panel, mdir, k: int, off: dict, qh: dict, dry: bool) -> dict:
    """{status OK | ABSENT | INVALID, reason, meta, rows} of o<k>'s scoring directory against the current inputs."""
    d = sdir(mdir, k)
    mp, tp = d / META_FILE, d / TOP_FILE
    if not mp.is_file() or not tp.is_file():
        return {"status": "ABSENT", "reason": f"o{k}: no {META_FILE} / {TOP_FILE} in {DIR}/o{k} (score)"}
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
        rows = read_top_file(tp)
    except (OSError, ValueError, EOFError) as e:
        return {"status": "INVALID", "reason": f"o{k}: unreadable ({type(e).__name__})"}
    probs = []
    for key, want in (("format", SCORE_FORMAT), ("domain", panel.split.get("domain")), ("seed", k),
                      ("eval_sha1", panel.eval_sha), ("split_sha1", panel.split_sha), ("selection_sha1", panel.selection_sha1),
                      ("n_rows_requested", panel.n_rows_requested), ("variant", panel.variant), ("topk", TOPK),
                      ("question", QUESTION), ("answer_token", ANSWER), ("dry_run", bool(dry))):
        if meta.get(key) != want:
            probs.append(f"{key} {meta.get(key)!r} is not {want!r}")
    if meta.get("scoring_sha1") != fx.file_sha1(tp):
        probs.append(f"{TOP_FILE} differs from the sha1 {meta.get('scoring_sha1')} that {META_FILE} records")
    if off.get("status") == "OK" and meta.get("adapter_weights_sha1") != off.get("weights_sha1"):
        probs.append("it scored another adapter than the one in adapters/ (weights sha1)")
    if meta.get("n_rows") != len(panel.sel) or len(rows) != len(panel.sel):
        probs.append(f"{len(rows)} rows for a selection of {len(panel.sel)}")
    else:
        for r, p in zip(rows, panel.sel):
            if (r.get("p"), r.get("u"), r.get("i"), r.get("c"), r.get("y")) != (
                    int(p), str(panel.cx.users[p]), str(panel.cx.item[p]), int(panel.cand_of[p]), int(panel.cx.y[p])):
                probs.append("a row is not the selected CAL pair at its place")
                break
    return {"status": "INVALID" if probs else "OK", "reason": f"o{k}: " + "; ".join(probs) if probs else None,
            "meta": meta, "rows": rows}


def build_report(a) -> dict:
    mdir = Path(a.method_dir)
    panel = Panel(a.panels, a.split, a.n_rows)
    problems = []
    if panel.split.get("domain") != a.domain:
        problems.append(f"--split is the split of {panel.split.get('domain')!r}, not of {a.domain!r}")
    rec = (panel.split.get("files") or {}).get("eval.jsonl")
    if rec is not None and rec != panel.eval_sha:
        problems.append(f"eval.jsonl: sha1 {panel.eval_sha} differs from ftgrid_split.json files ({rec})")
    if panel.cx.n_duplicate_pairs:
        problems.append(f"eval.jsonl repeats {panel.cx.n_duplicate_pairs} (user_id, item_id) pairs")
    qh = fm.load_eval_qhat(mdir, panel.cx, panel.eval_sha, panel.T, a.domain, problems)
    sel = panel.sel
    z = qh["z"][sel] if len(sel) else np.array([])
    label, users = panel.cx.y[sel], panel.cx.users[sel]
    adapters, arrays, excluded, yes_sets = {}, {}, [], {}
    for k in SEEDS:
        off = fm.load_offset(adapter_dir(mdir, k), k, qh, a.domain)
        sc = load_scoring(panel, mdir, k, off, qh, bool(a.dry_run))
        entry = {"offset_status": off["status"], "scoring_status": sc["status"], "b": off.get("b"),
                 "weights_sha1": off.get("weights_sha1"), "adapter_dry_run": bool(off.get("dry_run"))}
        if off["status"] == "INVALID":
            problems.append(off["reason"])
        if sc["status"] == "INVALID":
            problems.append(sc["reason"])
        if off["status"] == "OK" and sc["status"] == "OK":
            meta = sc["meta"]
            if bool(off.get("dry_run")) != bool(a.dry_run):
                problems.append(f"o{k}: the adapter is {'a DRY_RUN stand-in' if off.get('dry_run') else 'a real adapter'} but "
                                f"the diagnostic is {'a rehearsal' if a.dry_run else 'a real run'}")
            tops = [[(t, v) for t, v in r["top"]] for r in sc["rows"]]
            yes_ids, no_ids = frozenset(meta["yes_ids"]), frozenset(meta["no_ids"])
            stats, arr = adapter_stats(tops, [r["cens"] for r in sc["rows"]], yes_ids, no_ids, int(meta["answer_token_id"]),
                                       z, off["b"], label, users)
            if stats["censoring_mismatch_rows"]:
                problems.append(f"o{k}: the censoring of {stats['censoring_mismatch_rows']} rows differs from the recomputation")
            if int(meta["answer_token_id"]) not in yes_ids:
                problems.append(f"o{k}: the answer token {meta['answer_token_id']} is not a yes id of the scorer")
            entry.update(scoring_sha1=meta["scoring_sha1"], run_key=meta["run_key"], backbone=meta.get("backbone"), **stats)
            arrays[k] = arr
            yes_sets[k] = (sorted(yes_ids), sorted(no_ids), int(meta["answer_token_id"]), meta.get("backbone"),
                           meta.get("vllm_version"))
        else:
            reason = off.get("reason") if off["status"] != "OK" else sc.get("reason")
            excluded.append({"adapter": f"o{k}", "offset": off["status"], "scoring": sc["status"], "reason": reason})
        adapters[f"o{k}"] = entry
    if len({json.dumps(v[:4]) for v in yes_sets.values()}) > 1:
        problems.append(f"the adapters were scored with different yes / no / answer ids or backbones: {yes_sets}")
    done = [k for k in SEEDS if k in arrays]
    status = "INVALID" if problems else "OK" if len(done) == len(SEEDS) else "INCOMPLETE"
    # pooled over the adapters present: s statistics, pairs, UAUC of the pooled rows are not defined across adapters, so the
    # pooled block holds the s distribution, the pair counts and the mean UAUC of the adapters
    pooled = {}
    if done:
        s_all = np.concatenate([arrays[k]["s"][arrays[k]["ok"]] for k in done])
        pooled = {"adapters": [f"o{k}" for k in done], "s": share_stats(s_all),
                  "n_pairs": sum(adapters[f"o{k}"]["pairs"]["n_pairs"] for k in done),
                  "n_pairs_label_discordant": sum(adapters[f"o{k}"]["pairs"]["n_pairs_label_discordant"] for k in done),
                  "n_order_differs": sum(adapters[f"o{k}"]["pairs"]["n_order_differs"] for k in done),
                  "n_order_differs_label_discordant": sum(adapters[f"o{k}"]["pairs"]["n_order_differs_label_discordant"]
                                                         for k in done),
                  "mean_UAUC_R": _mean([adapters[f"o{k}"]["UAUC"]["R"] for k in done]),
                  "mean_UAUC_C": _mean([adapters[f"o{k}"]["UAUC"]["C"] for k in done])}
    rd = reading({f"o{k}": adapters[f"o{k}"]["s"]["p1"] for k in done}, status)
    first = next(iter(yes_sets.values()), None)
    meta = {"domain": a.domain, "backbone": first[3] if first else None, "single_backbone": True, "T": panel.T,
            "variant": panel.variant, "question": QUESTION, "answer_token": ANSWER,
            "answer_token_id": first[2] if first else None, "yes_ids": first[0] if first else None,
            "no_ids": first[1] if first else None, "topk": TOPK, "n_rows_requested": panel.n_rows_requested,
            "n_rows_registered": N_ROWS, "n_rows_selected": int(len(sel)), "n_cal_rows": panel.n_cal_rows,
            "n_users_selected": int(len(set(users.tolist()))), "n_users_cut": panel.users_cut(),
            "selection_sha1": panel.selection_sha1, "eval_sha1": panel.eval_sha, "split_sha1": panel.split_sha,
            "qhat_manifest_sha1": qh["manifest_sha1"], "standardisation": (qh["manifest"] or {}).get("standardisation"),
            "dry_run": bool(a.dry_run), "registered": bool(panel.n_rows_requested == N_ROWS and not a.dry_run),
            "n_pairs_without_qhat_row": qh["n_pairs_without_qhat_row"]}
    code = {n: fx.file_sha1(Path(__file__).resolve().parent / n) for n in CODE_FILES}
    return {"spec": SPEC, "format": REPORT_FORMAT, "alias": "shift_diag", "status": status, "meta": meta,
            "input_checks": {"problems": problems, "excluded_adapters": excluded}, "adapters": adapters, "pooled": pooled,
            "reading": rd, "code_sha1": code, "operationalizations": list(OPERATIONALIZATIONS)}


def _mean(v) -> float | None:
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def table_rows(res: dict) -> list:
    d = res["meta"]["domain"]
    rows = []
    for name, e in res["adapters"].items():
        if "s" not in e:
            rows.append({"dataset": d, "adapter": name, "reading": "not scored"})
            continue
        rows.append({"dataset": d, "adapter": name, "n_rows": e["n_rows"], "n_used": e["n_used"], "b": e["b"],
                     "s_mean": e["s"]["mean"], "s_median": e["s"]["median"], "s_p1": e["s"]["p1"], "s_min": e["s"]["min"],
                     "share_s_below_0.99": e["s"]["share_below_0.99"], "abs_delta_max": e["delta_C_minus_R"]["abs_max"],
                     "UAUC_R": e["UAUC"]["R"], "UAUC_C": e["UAUC"]["C"], "dUAUC_C_minus_R": e["UAUC"]["C_minus_R"],
                     "n_users": e["UAUC"]["n_users"], "n_pairs": e["pairs"]["n_pairs"],
                     "n_pairs_label_discordant": e["pairs"]["n_pairs_label_discordant"],
                     "n_order_differs": e["pairs"]["n_order_differs"],
                     "n_order_differs_label_discordant": e["pairs"]["n_order_differs_label_discordant"], "reading": ""})
    p = res["pooled"]
    if p:
        rows.append({"dataset": d, "adapter": "pooled", "n_used": p["s"]["n"], "s_mean": p["s"]["mean"],
                     "s_median": p["s"]["median"], "s_p1": p["s"]["p1"], "s_min": p["s"]["min"],
                     "share_s_below_0.99": p["s"]["share_below_0.99"], "UAUC_R": p["mean_UAUC_R"], "UAUC_C": p["mean_UAUC_C"],
                     "n_pairs": p["n_pairs"], "n_pairs_label_discordant": p["n_pairs_label_discordant"],
                     "n_order_differs": p["n_order_differs"],
                     "n_order_differs_label_discordant": p["n_order_differs_label_discordant"], "reading": ""})
    rows.append({"dataset": d, "adapter": "reading", "s_p1": res["reading"]["p1_min"], "reading": res["reading"]["label"]})
    return rows


def report_files(res: dict) -> dict:
    """{file name: bytes} of report.json and report_tables.csv for a report dict."""
    res = strict_json(res)
    return {REPORT_FILE: (json.dumps(res, indent=2, allow_nan=False) + "\n").encode("utf-8"),
            TABLE_FILE: fm.csv_bytes(REPORT_COLS, table_rows(res))}


def write_report(a) -> dict:
    res = strict_json(build_report(a))
    out = Path(a.method_dir) / DIR
    out.mkdir(parents=True, exist_ok=True)
    for name, data in report_files(res).items():
        tlo.write_if_changed(out / name, data)
    rd = res["reading"]
    print(f"[shift_diag report] {a.domain}: {res['status']}; reading {rd['label']} (p1 of s by adapter {rd['p1_by_adapter']}); "
          f"wrote {out / REPORT_FILE}" + (f"; problems: {res['input_checks']['problems']}" if res["input_checks"]["problems"]
                                          else ""))
    return res


def verify(a) -> None:
    """Exit 0 iff report.json on disk is the recomputation, OK, of the mode --dry_run says (and registered when real)."""
    out = Path(a.method_dir) / DIR
    for name in (REPORT_FILE, TABLE_FILE):
        if not (out / name).is_file():
            raise DiagError(f"the shift diagnostic of {a.domain} is not recorded: {out / name} is missing", 1)
    res = strict_json(build_report(a))
    for name, data in report_files(res).items():
        if (out / name).read_bytes() != data:
            raise DiagError(f"{out / name} is not the recomputation of the report from its files and inputs (a file, an adapter, "
                            "the panel, the manifest or the code changed after it was written): run the report step again", 1)
    if res["status"] != "OK":
        raise DiagError(f"the shift diagnostic of {a.domain} reads {res['status']}: {res['input_checks']}", 1)
    if res["meta"]["dry_run"] != bool(a.dry_run):
        raise DiagError(f"the shift diagnostic of {a.domain} is {'a DRY_RUN rehearsal' if res['meta']['dry_run'] else 'a real run'}"
                        f" but --dry_run is {'set' if a.dry_run else 'not set'}", 1)
    if not a.dry_run and not res["meta"]["registered"]:
        raise DiagError(f"the shift diagnostic of {a.domain} was not made on the registered {N_ROWS} rows", 1)
    print(f"shift diagnostic of {a.domain} verified: {res['reading']['label']} (p1 of s {res['reading']['p1_by_adapter']})")


# ---------------------------------------------------------------- the pilot-log record of the diagnostic's own files
def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _file_sha1(root: Path, rel: str) -> str:
    try:
        return fx.file_sha1(root / rel)
    except OSError as e:
        raise DiagError(f"cannot read {rel} for the record: {e}", 1) from None


def record_lines(files, root=None) -> list:
    root = Path(root) if root else default_root()
    return [f"{rel} = {_file_sha1(root, rel)}" for rel in files]


def record_missing(pilot_log, files, root=None) -> list:
    """The files whose sha1 is not in the pilot log (a case-insensitive substring test, as ftgrid_freeze)."""
    if not Path(pilot_log).is_file():
        raise DiagError(f"pilot log {pilot_log} does not exist", 4)
    root = Path(root) if root else default_root()
    log = Path(pilot_log).read_text(encoding="utf-8").lower()
    return [rel for rel in files if _file_sha1(root, rel).lower() not in log]


def cmd_record(a) -> int:
    if not a.files or (not a.pilot_log and not a.do_print):
        raise DiagError("record needs --files and, unless --print, --pilot_log", 2)
    if a.do_print:
        print("\n".join(record_lines(a.files, a.root)))
        return 0
    missing = record_missing(a.pilot_log, a.files, a.root)
    if missing and a.append:
        lines = record_lines(missing, a.root)
        with open(a.pilot_log, "a", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        missing = []
    if missing:
        print(f"shift diagnostic record (addendum 10 section 2): the sha1 of these files is not in {a.pilot_log}: {missing}; "
              "run `python -m src.confrec.ftmethod_shift_diag record --print --files ...`, record the lines in the pilot log "
              "and push it", file=sys.stderr)
        return 4
    print(f"shift diagnostic record OK: every listed sha1 is in {a.pilot_log}")
    return 0


# ---------------------------------------------------------------- command line
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, scoring: bool):
        p.add_argument("--domain", required=True, choices=list(DATASETS))
        p.add_argument("--split", required=True, help="the ftgrid panels/<d>/ftgrid_split.json")
        p.add_argument("--panels", required=True, help="the ftgrid panels/<d>/ (eval.jsonl)")
        p.add_argument("--method_dir", required=True, help="outputs/confrec/ftmethod/<d>/ (adapters, q-hat; writes shift_diag/)")
        p.add_argument("--n_rows", type=int, default=N_ROWS,
                       help=f"the first n CAL rows (registered: {N_ROWS}; another value is only a rehearsal's)")
        p.add_argument("--dry_run", action="store_true", help="CPU stand-in for the model and tokenizer: a rehearsal, marked dry_run")
        if scoring:
            p.add_argument("--dry_share", type=float, default=DRY_SHARE, help="--dry_run: the stand-in's s on a normal row")
            p.add_argument("--dry_low_frac", type=float, default=DRY_LOW_FRAC,
                           help="--dry_run: the share of rows on which the stand-in's s is U(0.5, 0.9)")

    s = sub.add_parser("score", help="GPU: the top-50 log-probabilities of one prior-offset adapter on the selected CAL rows")
    common(s, True)
    s.add_argument("--seed", type=int, required=True, choices=list(SEEDS), help="the adapter o<seed>")
    s.add_argument("--model", required=True, help="the Qwen3-8B directory (dryrun/Qwen3-8B under --dry_run)")
    s.add_argument("--variant", required=True, help="the selected prompt (selection.json gate_ft_prompt)")
    s.add_argument("--lora_arg", default=None, help="the adapter path as vLLM is given it (default: <method_dir>/adapters/o<seed>)")
    s.add_argument("--dtype", default="float16")
    s.add_argument("--max_model_len", type=int, default=4096)
    s.add_argument("--gpu_mem", type=float, default=0.88)
    s.add_argument("--vllm_seed", type=int, default=0)
    r = sub.add_parser("report", help="CPU: report.json and report_tables.csv from the scoring directories")
    common(r, False)
    v = sub.add_parser("verify", help="CPU: report.json is the recomputation, OK, of the right mode (exit 0 or 1)")
    common(v, False)
    t = sub.add_parser("record", help="the pilot-log record of the diagnostic's own files")
    t.add_argument("--pilot_log", default=None, help="docs/sigir/PILOT_LOG.md")
    t.add_argument("--files", nargs="+", default=None, help="repo-relative files whose sha1 the log must hold")
    t.add_argument("--root", default=None, help="the repo root (default: this checkout)")
    t.add_argument("--print", dest="do_print", action="store_true", help="print the lines for the pilot log")
    t.add_argument("--append", action="store_true",
                   help="DRY_RUN only, the human step of the rehearsal: append the missing lines to the (temporary) pilot log")
    a = ap.parse_args(argv)
    if a.cmd != "record" and a.n_rows < 1:
        ap.error("--n_rows must be >= 1")
    if a.cmd == "score" and (a.dry_share < 0 or a.dry_share > 1 or a.dry_low_frac < 0 or a.dry_low_frac > 1):
        ap.error("--dry_share and --dry_low_frac are fractions")
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        if a.cmd == "record":
            return cmd_record(a)
        if a.cmd == "score":
            score_adapter(a)
        elif a.cmd == "report":
            write_report(a)
        else:
            verify(a)
        return 0
    except DiagError as e:
        print(str(e), file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
