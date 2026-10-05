"""S6 composition diagnostics (FT-P): idea-stage/PREREG_AMENDMENT_3_ADDENDUM_7.md section 2 (with PREREG_AMENDMENT_3.md
sections 1 and 6 and PREREG_AMENDMENT_3_ADDENDUM_4.md). Descriptive and CPU: a pure function of files (the prune manifest and
what it names, ftgrid_data's EVAL panel and split record); no model, no score, no outcome, no clock. It says how the pruned arms
differ in item composition from P0 and from each other (the methodology review: P1 matches class counts only), never whether
pruning helped.

    python -m src.confrec.ftprune_compose --manifest outputs/confrec/ftprune/prune_manifest.json \
        --out_dir outputs/confrec/ftprune/compose \
        [--eval_panel outputs/confrec/ftgrid/panels/ml1m/eval.jsonl] \
        [--ftgrid_split outputs/confrec/ftgrid/panels/ml1m/ftgrid_split.json] [--train outputs/confrec/gateft/train.jsonl]

Inputs. Every one is checked before anything is computed; a mismatch is a refusal (exit 2, nothing written):
  * the manifest (format ftprune_manifest_v1, domain ml1m) and, relative to its directory, every file it lists under `files`: the signals file,
    the 20 subset index files and the 17 pruned TRAIN files must exist with their recorded sha1 (ftprune.verify's rule, run on
    this manifest; the sha1 of all 38 are recorded in the output) and the manifest's own records must agree (signals.sha1,
    subsets.<tag>.file / sha1 / n_keep / n_removed, classes);
  * the TRAIN rows: the signals file the manifest names (one row per train.jsonl example in train.jsonl order: key = user::item,
    item_id, label; the signals themselves are not used); its counts must equal inputs.train_jsonl (examples, positives,
    negatives). --train (optional) re-reads Gate-FT's train.jsonl, checks its sha1 against inputs.train_jsonl and that its
    examples are the signals rows one for one;
  * the arms' examples: subsets/<arm>_s<seed>.txt (the KEPT keys) of P0-P3, seeds 0-4, whose per-class counts must equal the
    manifest's records;
  * the TEST pairs (statistic c): ftgrid_data's panels/ml1m/eval.jsonl (the EVAL users' rows) and its ftgrid_split.json. The split's
    recorded sha1 of eval.jsonl must match the file, its T must equal the Gate-FT T recorded in the manifest, its recorded sha1 of
    train.jsonl must equal inputs.train_jsonl.sha1 (ML-1M's TRAIN set is the G9 set), its recorded TEST count must equal the pairs
    counted here and its recorded share_test_pairs_item_seen_in_train must equal what P0 gives here.

Statistics, per arm and seed. K = the number of distinct TRAIN items (the item ids of the signals rows); n_i = the TRAIN examples
of item i and P_i their positives; m_i and p_i = the examples and positives of item i that the arm keeps (the subset file).
  a_no_example_item_share      #{i : m_i = 0} / K.
  b_mean_abs_item_rate_change  sum over the items with n_i >= 2 and m_i >= 1 of n_i |p_i / m_i - P_i / n_i|, divided by the sum of
                               n_i over the same items: the mean absolute change of the item label rate, weighted by the items'
                               number of TRAIN examples.
  c_test_pairs_seen_share      the share of the EVAL users' TEST pairs (every candidate of eval.jsonl with ts >= T, ftgrid_data's
                               rule and the split's T) whose item has m_i >= 1; a pair whose item is not a TRAIN item is unseen.
  d_head_tercile_example_share the share of the arm's kept examples whose item is in the head tercile of the items by TRAIN
                               popularity (n_i over the full TRAIN set, never recomputed on an arm).
Readings chosen where the text left one (all stated in the output's `definitions`):
  1. (b) weights = the TRAIN counts n_i, the same for every arm; the scope (n_i >= 2) is fixed by the full TRAIN set, so singletons
     are out of scope whether kept or not. An in-scope item that keeps no example has no label rate in the arm: it is left out of
     the numerator and of the denominator (the mean is over the items whose rate exists) and is counted in
     b_items_in_scope_lost / b_weight_in_scope_lost beside the statistic, and in (a).
  2. (d) popularity = the item's number of TRAIN examples (= its distinct TRAIN users: a user-item pair is one example); the terciles
     are over items, cut on average ranks with stats.rank_bins(n, 3) (the function the amendment's popularity bins use): the head
     is bin 2. Ties: items with equal n_i have equal average rank and share one tercile, so a tie group is never split by a random
     key and the head can hold more or fewer than K / 3 items (terciles.n_items_head is reported); when every item has the same
     n_i the head is empty and (d) is 0.
  3. (c) a TEST pair is one candidate of one EVAL user (counted per pair, not per user or per distinct item).
  4. every arm is summarised by the mean over its seeds 0-4 of the per-seed values (P1's seeds differ; P0, P2 and P3 do not depend
     on the seed, `subsets_identical_across_seeds`); the per-seed values and their SD (ddof 1) are listed.
  5. all sums and means are exact rationals converted to float once, so no summation order or platform changes a digit.
Outputs (--out_dir, written only when their bytes change): ftprune_compose.json (strict JSON, allow_nan=False, no clock, host or
path; the sha1 of every input; the code sha1 of this file and of ftprune.py) and ftprune_compose.csv. Exit 0, or 2 on a refusal.
ftprune is imported read-only (its functions and constants are read, never assigned); the only files written are the two above.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from src.confrec import ftprune as fp
from src.confrec.stats import rank_bins

SPEC = ("idea-stage/PREREG_AMENDMENT_3_ADDENDUM_7.md section 2 (with PREREG_AMENDMENT_3.md sections 1 and 6, "
        "PREREG_AMENDMENT_3_ADDENDUM_4.md)")
FORMAT = "ftprune_compose_v1"
DOMAIN = "ml1m"
OUT_JSON = "ftprune_compose.json"
OUT_CSV = "ftprune_compose.csv"
DEFAULT_EVAL = "outputs/confrec/ftgrid/panels/ml1m/eval.jsonl"
DEFAULT_SPLIT = "outputs/confrec/ftgrid/panels/ml1m/ftgrid_split.json"
N_TERCILES = 3
HEAD_BIN = N_TERCILES - 1
TAIL_BIN = 0
MIN_SCOPE = 2                            # (b): the items with at least two TRAIN examples
SEEN_TOL = 1e-12                         # P0's seen share against ftgrid_split.json's recorded float
STATS = ("a_no_example_item_share", "b_mean_abs_item_rate_change", "c_test_pairs_seen_share",
         "d_head_tercile_example_share")
COUNT_KEYS = ("n_kept", "n_kept_positives", "n_kept_negatives", "items_without_example", "b_items_in_scope_lost",
              "b_weight_in_scope_lost", "c_test_pairs_seen", "d_head_examples")
CSV_COLS = ("arm", "seed") + STATS + ("n_kept", "items_without_example", "b_items_in_scope_lost")
DEFINITIONS = (
    "TRAIN rows = the signals file the manifest names (one row per train.jsonl example, key = user_id + '::' + item_id); K = the "
    "number of distinct item ids; n_i / P_i = the TRAIN examples / positives of item i; m_i / p_i = the examples / positives of "
    "item i kept by the arm (subsets/<arm>_s<seed>.txt lists the kept keys)",
    "a_no_example_item_share = #{i : m_i = 0} / K",
    "b_mean_abs_item_rate_change = sum over the items with n_i >= 2 and m_i >= 1 of n_i * |p_i / m_i - P_i / n_i|, divided by the "
    "sum of n_i over the same items (weights = the TRAIN counts n_i, the same for every arm; the scope n_i >= 2 is fixed by the full "
    "TRAIN set); an in-scope item with m_i = 0 has no label rate in the arm and is left out of numerator and denominator, and "
    "counted in b_items_in_scope_lost / b_weight_in_scope_lost (and in a_no_example_item_share); null when no item is left",
    "c_test_pairs_seen_share = #{TEST pairs whose item has m_i >= 1} / #{TEST pairs}; TEST pairs = every candidate of every row of "
    "eval.jsonl (ftgrid_data's EVAL users) with float(timestamp) >= T of ftgrid_split.json; a pair whose item is not a TRAIN item "
    "is unseen; for P0 this is ftgrid_split.json's share_test_pairs_item_seen_in_train",
    "d_head_tercile_example_share = sum of m_i over the head items / sum of m_i; the head = the items with "
    "stats.rank_bins(n, 3) == 2, n = the TRAIN examples per item over the full TRAIN set (the terciles are over items, cut on "
    "average ranks; items with equal n_i share one tercile, no random tie key, the head may hold more or fewer than K / 3 items; "
    "empty when all n_i are equal, then the statistic is 0); null when the arm keeps no example",
    "an arm is summarised by the mean over seeds 0-4 of the per-seed values (exact rationals converted to float once); sd_seed = "
    "the sample SD (ddof 1) of the per-seed values; P0, P2 and P3 do not depend on the seed (subsets_identical_across_seeds)",
)


class Refusal(Exception):
    """Inputs that are not the registered ones: nothing is computed or written (exit 2)."""

    def __init__(self, problems):
        self.problems = [problems] if isinstance(problems, str) else [str(p) for p in problems]
        super().__init__("; ".join(self.problems))


# ---------------------------------------------------------------- reading and checking the inputs
def load_json(path, what: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise Refusal(f"{what} {p.name}: no such file")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refusal(f"{what} {p.name} is not readable JSON ({type(e).__name__}: {e})") from None
    if not isinstance(obj, dict):
        raise Refusal(f"{what} {p.name} is not a JSON object")
    return obj


def dig(obj, *path, what: str = "the manifest"):
    """obj[path[0]][path[1]]...; a Refusal when a key is missing or a level is not an object."""
    cur = obj
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise Refusal(f"{what} lacks {'.'.join(path)}")
        cur = cur[k]
    return cur


def registered_files() -> set:
    """The files a prune manifest lists under `files` (ftprune.verify's expected set)."""
    return ({fp.SIGNALS_FILE} | {fp.subset_rel(a, s) for a in fp.ARMS for s in fp.SEEDS}
            | {fp.train_rel(a, s) for a in fp.ARMS for s in fp.SEEDS if fp.has_train_file(a, s)})


def verify_tree(man: dict, base) -> list:
    """Problems (empty = every file the manifest lists exists with its recorded sha1; only the registered names are read)."""
    base = Path(base)
    files = fp.manifest_files(man)
    need = registered_files()
    probs = []
    if set(files) != need:
        probs.append(f"the manifest lists {sorted(set(files) ^ need)} unexpectedly (or lacks them)")
    for rel in sorted(need & set(files)):
        p = base / rel
        if not p.is_file():
            probs.append(f"{rel}: missing")
            continue
        got = fp.sha1_file(p)
        if got != files[rel]:
            probs.append(f"{rel}: sha1 {got} is not the recorded {files[rel]}")
    return probs


def class_counts(y, mask) -> dict:
    y, mask = np.asarray(y, int), np.asarray(mask, bool)
    return {"1": int(((y == 1) & mask).sum()), "0": int(((y == 0) & mask).sum())}


def train_table(sig: dict) -> dict:
    """The TRAIN examples of the signals file (key, item, label) and their per-item counts over the sorted item ids."""
    keys = [str(k) for k in sig["key"]]
    users = [str(u) for u in sig["user_id"]]
    items = [str(i) for i in sig["item_id"]]
    y = np.asarray(sig["label"], dtype=np.int64)
    if not keys:
        raise Refusal("the signals file holds no TRAIN example")
    probs = []
    if len(set(keys)) != len(keys):
        probs.append("the signals file holds a key twice")
    if sum(1 for k, u, i in zip(keys, users, items) if k != u + fp.SEP + i):
        probs.append("a signals key is not user_id + '::' + item_id")
    if not np.isin(y, (0, 1)).all():
        probs.append("a signals label is not 0/1")
    if probs:
        raise Refusal(probs)
    uniq = sorted(set(items))
    index = {it: j for j, it in enumerate(uniq)}
    inv = np.array([index[i] for i in items], dtype=np.int64)
    k = len(uniq)
    return {"keys": keys, "items_of": items, "y": y, "inv": inv, "items": uniq, "index": index,
            "n": np.bincount(inv, minlength=k).astype(np.int64),
            "P": np.bincount(inv[y == 1], minlength=k).astype(np.int64),
            "key_index": {key: j for j, key in enumerate(keys)}}


def check_signals_record(man: dict, tt: dict) -> None:
    """The signals file against the manifest's records of it and of the TRAIN set."""
    rec = dig(man, "inputs", "train_jsonl")
    probs = []
    if dig(man, "signals", "file") != fp.SIGNALS_FILE:
        probs.append(f"signals.file is not {fp.SIGNALS_FILE}")
    if dig(man, "signals", "sha1") != fp.manifest_files(man).get(fp.SIGNALS_FILE):
        probs.append("signals.sha1 differs from the files record of the signals file")
    got = {"examples": len(tt["keys"]), "positives": int(tt["y"].sum()), "negatives": int((tt["y"] == 0).sum())}
    for k, v in got.items():
        if rec.get(k) != v:
            probs.append(f"the signals file holds {v} {k}, the manifest's inputs.train_jsonl records {rec.get(k)}")
    classes = dig(man, "classes")
    for c, name in (("1", "positives"), ("0", "negatives")):
        if dig(classes, c, "n", what="the manifest's classes") != got[name]:
            probs.append(f"classes.{c}.n is {classes[c].get('n')}, the signals file holds {got[name]} {name}")
    if probs:
        raise Refusal(probs)


def read_train(man: dict, base) -> dict:
    """The TRAIN table from the signals file the manifest names, checked against the manifest's records."""
    try:
        sig = fp.read_signals(Path(base) / fp.SIGNALS_FILE)
    except (OSError, KeyError, ValueError, EOFError, csv.Error) as e:
        raise Refusal(f"{fp.SIGNALS_FILE} is not a readable signals file ({type(e).__name__}: {e})") from None
    tt = train_table(sig)
    check_signals_record(man, tt)
    return tt


def load_keep(man: dict, base, tt: dict) -> dict:
    """{(arm, seed): kept mask over the TRAIN examples} from the subset index files; refuses keys that are not TRAIN examples
    and per-class counts that differ from the manifest's records (subsets.<tag>, classes)."""
    base = Path(base)
    files = fp.manifest_files(man)
    y = tt["y"]
    tot = class_counts(y, np.ones(len(y), bool))
    classes = dig(man, "classes")
    probs, out = [], {}
    for arm in fp.ARMS:
        for s in fp.SEEDS:
            tag, rel = fp.run_tag(arm, s), fp.subset_rel(arm, s)
            rec = dig(man, "subsets", tag)
            if not isinstance(rec, dict):
                probs.append(f"{tag}: the manifest's subsets record is not an object")
                continue
            if rec.get("file") != rel or rec.get("sha1") != files.get(rel):
                probs.append(f"{tag}: the manifest's subsets record does not name {rel} with its files sha1")
            text = (base / rel).read_bytes().decode("utf-8")
            idx = [tt["key_index"].get(k, -1) for k in (text.split("\n") if text else [])]
            if min(idx, default=0) < 0:
                probs.append(f"{rel}: {sum(1 for i in idx if i < 0)} keys are not TRAIN examples")
                continue
            if len(set(idx)) != len(idx):
                probs.append(f"{rel}: a key twice")
                continue
            keep = np.zeros(len(y), bool)
            keep[np.array(idx, dtype=np.int64)] = True
            kept = class_counts(y, keep)
            removed = {c: tot[c] - kept[c] for c in ("1", "0")}
            for name, got in (("n_keep", kept), ("n_removed", removed)):
                rec_c = rec.get(name)
                if not isinstance(rec_c, dict) or {c: rec_c.get(c) for c in ("1", "0")} != got:
                    probs.append(f"{tag}: {name} {got} differs from the manifest's {rec_c}")
            want = {c: (tot[c] if arm == "P0" else dig(classes, c, "n_keep", what="the manifest's classes"))
                    for c in ("1", "0")}
            if kept != want:
                probs.append(f"{tag}: keeps {kept} per class, the manifest's classes record says {want}")
            out[(arm, s)] = keep
    if probs:
        raise Refusal(probs)
    return out


def check_train_file(path, man: dict, tt: dict) -> None:
    """--train: Gate-FT's train.jsonl has the recorded sha1 and its examples are the signals rows one for one."""
    p = Path(path)
    if not p.is_file():
        raise Refusal(f"--train {p.name}: no such file")
    want = dig(man, "inputs", "train_jsonl", "sha1")
    got = fp.sha1_file(p)
    if got != want:
        raise Refusal(f"--train {p.name}: sha1 {got} is not the manifest's recorded train.jsonl sha1 {want}")
    try:
        ex = fp.train_examples(fp.read_rows(p))
    except SystemExit as e:
        raise Refusal(f"--train {p.name}: {e.code}") from None
    if ex["key"] != tt["keys"] or ex["item"] != tt["items_of"] or ex["y"].tolist() != tt["y"].tolist():
        raise Refusal(f"--train {p.name}: its examples are not the signals file's rows (key, item, label, order)")


def read_eval(path, T: float) -> dict:
    """The EVAL rows of eval.jsonl: {"rows", "candidates", "test_items"} (the item ids, as str, of the TEST pairs: ts >= T)."""
    p = Path(path)
    rows = pairs = 0
    test_items = []
    with open(p, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line.decode("utf-8"))
                ids, ts = r["candidate_item_ids"], r["candidate_timestamps"]
                if not isinstance(ids, list) or not isinstance(ts, list) or len(ids) != len(ts):
                    raise ValueError("candidate_item_ids and candidate_timestamps are not aligned lists")
                stamps = [float(t) for t in ts]
                if not all(math.isfinite(t) for t in stamps):
                    raise ValueError("a non-finite timestamp")
            except (ValueError, KeyError, TypeError) as e:
                raise Refusal(f"{p.name} row {rows}: not an EVAL panel row ({type(e).__name__}: {e})") from None
            test_items += [str(i) for i, t in zip(ids, stamps) if t >= T]
            pairs += len(ids)
            rows += 1
    return {"rows": rows, "candidates": pairs, "test_items": test_items}


def split_record(split: dict) -> dict:
    """The fields of ftgrid_split.json this module needs (ftgrid_data's schema); refuses a file without them."""
    try:
        rec = {"T": float(split["T"]), "eval_sha1": split["files"]["eval.jsonl"],
               "train_sha1": split["files"]["train.jsonl"], "test_candidates": int(split["eval"]["test_candidates"]),
               "seen_share": split["eval"]["share_test_pairs_item_seen_in_train"]}
    except (KeyError, TypeError, ValueError):
        raise Refusal("ftgrid_split.json lacks T, files.eval.jsonl, files.train.jsonl, eval.test_candidates or "
                      "eval.share_test_pairs_item_seen_in_train: not ftgrid_data's output") from None
    if not math.isfinite(rec["T"]):
        raise Refusal("ftgrid_split.json records a non-finite T")
    return rec


def check_split_inputs(man: dict, eval_panel, ftgrid_split) -> dict:
    """The EVAL panel and the split record against each other and against the manifest (sha1s, T)."""
    spl, ev = Path(ftgrid_split), Path(eval_panel)
    rec = split_record(load_json(spl, "ftgrid_split.json"))
    if not ev.is_file():
        raise Refusal(f"--eval_panel {ev.name}: no such file")
    probs = []
    ev_sha1 = fp.sha1_file(ev)
    if ev_sha1 != rec["eval_sha1"]:
        probs.append(f"{ev.name}: sha1 {ev_sha1} is not the ftgrid split's recorded eval.jsonl sha1 {rec['eval_sha1']}")
    train_sha1 = dig(man, "inputs", "train_jsonl", "sha1")
    if rec["train_sha1"] != train_sha1:
        probs.append(f"the ftgrid split records train.jsonl sha1 {rec['train_sha1']}, the manifest's TRAIN set is {train_sha1}")
    gs = dig(man, "inputs", "gateft_split")
    g_t = gs.get("T") if isinstance(gs, dict) else None
    if g_t is None:
        probs.append("the manifest records no Gate-FT T (inputs.gateft_split.T): stage B ran without --split_report")
    else:
        try:
            same = float(g_t) == rec["T"]
        except (TypeError, ValueError):
            same = False
        if not same:
            probs.append(f"the ftgrid split's T {rec['T']!r} is not the Gate-FT T {g_t!r} the manifest records")
    if probs:
        raise Refusal(probs)
    return {"rec": rec, "eval_path": ev, "eval_sha1": ev_sha1, "split_sha1": fp.sha1_file(spl), "gateft": gs}


# ---------------------------------------------------------------- the statistics
def tercile_bins(n) -> np.ndarray:
    """0 = tail, 1 = middle, 2 = head tercile of the items by TRAIN popularity n (stats.rank_bins: average ranks, equal n
    share a bin)."""
    n = np.asarray(n, float)
    return rank_bins(n, N_TERCILES) if len(n) else np.zeros(0, int)


def exact_mean(values):
    """Exact mean of Fractions (None when any is missing or there is none)."""
    if not values or any(v is None for v in values):
        return None
    return sum(values, Fraction(0)) / len(values)


def exact_sd(values):
    """Sample SD (ddof 1) of Fractions as a float: the variance is exact, so equal values give exactly 0.0."""
    if len(values) < 2 or any(v is None for v in values):
        return None
    mu = exact_mean(values)
    var = sum(((v - mu) ** 2 for v in values), Fraction(0)) / (len(values) - 1)
    return math.sqrt(float(var))


def as_float(x):
    return None if x is None else float(x)


def arm_stats(tt: dict, keep, test_count, n_test: int, head) -> dict:
    """The four statistics (exact Fractions, None where undefined) and their counts for one kept mask. test_count[i] = the TEST
    pairs whose item is TRAIN item i; n_test = all TEST pairs; head = the head-tercile mask over the items."""
    k = len(tt["items"])
    inv, y, n, big_p = tt["inv"], tt["y"], tt["n"], tt["P"]
    m = np.bincount(inv[keep], minlength=k)
    p = np.bincount(inv[keep & (y == 1)], minlength=k)
    lost = m == 0
    scope = n >= MIN_SCOPE
    ret = np.flatnonzero(scope & ~lost)
    acc = defaultdict(int)          # per m: the sum of |n p - P m|, since n |p / m - P / n| = |n p - P m| / m
    for i in ret.tolist():
        acc[int(m[i])] += abs(int(n[i]) * int(p[i]) - int(big_p[i]) * int(m[i]))
    weight = int(n[ret].sum())
    b = sum((Fraction(v, mm) for mm, v in sorted(acc.items())), Fraction(0)) / weight if weight else None
    n_kept = int(keep.sum())
    seen = int(test_count[~lost].sum())
    d_num = int(m[head].sum())
    return {
        "a_no_example_item_share": Fraction(int(lost.sum()), k),
        "b_mean_abs_item_rate_change": b,
        "c_test_pairs_seen_share": Fraction(seen, n_test) if n_test else None,
        "d_head_tercile_example_share": Fraction(d_num, n_kept) if n_kept else None,
        "n_kept": n_kept, "n_kept_positives": int(keep[y == 1].sum()), "n_kept_negatives": int(keep[y == 0].sum()),
        "items_without_example": int(lost.sum()), "b_items_in_scope_lost": int((scope & lost).sum()),
        "b_weight_in_scope_lost": int(n[scope & lost].sum()), "c_test_pairs_seen": seen, "d_head_examples": d_num,
    }


def seed_entry(st: dict) -> dict:
    out = {k: as_float(st[k]) for k in STATS}
    out.update({k: st[k] for k in COUNT_KEYS})
    return out


def arm_entry(per_seed: dict, files: dict, arm: str) -> dict:
    """One arm over its seeds: the exact mean and SD of each statistic and the per-seed values."""
    seeds = list(fp.SEEDS)
    cols = {k: [per_seed[(arm, s)][k] for s in seeds] for k in STATS}
    shas = {files[fp.subset_rel(arm, s)] for s in seeds}
    return {"seeds": seeds,
            "mean": {k: as_float(exact_mean(cols[k])) for k in STATS},
            "sd_seed": {k: exact_sd(cols[k]) for k in STATS},
            "statistics_identical_across_seeds": all(len(set(cols[k])) == 1 for k in STATS),
            "subsets_identical_across_seeds": len(shas) == 1,
            "per_seed": {f"s{s}": seed_entry(per_seed[(arm, s)]) for s in seeds}}


# ---------------------------------------------------------------- the whole computation
def compose(manifest, eval_panel=DEFAULT_EVAL, ftgrid_split=DEFAULT_SPLIT, train=None) -> dict:
    """The result dict (see the module docstring); raises Refusal on any input that is not the registered one."""
    mp = Path(manifest)
    man = load_json(mp, "manifest")
    if man.get("format") != fp.MANIFEST_FORMAT:
        raise Refusal(f"{mp.name}: format {man.get('format')!r} is not {fp.MANIFEST_FORMAT}")
    if man.get("domain", DOMAIN) != DOMAIN:
        raise Refusal(f"{mp.name}: domain {man.get('domain')!r} is not {DOMAIN!r} (S6 is ML-1M only)")
    base = mp.parent
    probs = verify_tree(man, base)
    if probs:
        raise Refusal(probs)
    files = fp.manifest_files(man)
    tt = read_train(man, base)
    keep = load_keep(man, base, tt)
    if train:
        check_train_file(train, man, tt)
    chk = check_split_inputs(man, eval_panel, ftgrid_split)
    rec = chk["rec"]
    ev = read_eval(chk["eval_path"], rec["T"])
    n_test = len(ev["test_items"])
    if not n_test:
        raise Refusal("no TEST pair (timestamp >= T) in the EVAL panel")
    if n_test != rec["test_candidates"]:
        raise Refusal(f"{n_test} TEST pairs found, the ftgrid split records {rec['test_candidates']}")
    k, y, n = len(tt["items"]), tt["y"], tt["n"]
    t_idx = np.array([tt["index"].get(i, -1) for i in ev["test_items"]], dtype=np.int64)
    test_count = np.bincount(t_idx[t_idx >= 0], minlength=k)
    bins = tercile_bins(n)
    head = bins == HEAD_BIN

    per_seed = {(arm, s): arm_stats(tt, keep[(arm, s)], test_count, n_test, head) for arm in fp.ARMS for s in fp.SEEDS}
    p0_seen = per_seed[("P0", 0)]["c_test_pairs_seen_share"]
    try:
        seen_ok = abs(float(p0_seen) - float(rec["seen_share"])) <= SEEN_TOL
    except (TypeError, ValueError):
        seen_ok = False
    if not seen_ok:
        raise Refusal(f"P0's seen share {float(p0_seen)!r} is not the ftgrid split's recorded "
                      f"share_test_pairs_item_seen_in_train {rec['seen_share']!r}: the TRAIN items and the TEST pairs are "
                      "not the split's")

    arms = {a: arm_entry(per_seed, files, a) for a in fp.ARMS}
    in_scope = n >= MIN_SCOPE
    n_head, n_tail = int(head.sum()), int((bins == TAIL_BIN).sum())
    return {
        "format": FORMAT, "spec": SPEC, "domain": DOMAIN, "variant": man.get("variant"),
        "code_sha1": {"ftprune_compose.py": fp.sha1_file(__file__), "ftprune.py": fp.sha1_file(fp.__file__)},
        "inputs": {
            "manifest": {"sha1": fp.sha1_file(mp), "format": man["format"], "files_verified": len(files)},
            "signals": {"file": fp.SIGNALS_FILE, "sha1": files[fp.SIGNALS_FILE], "rows": len(tt["keys"])},
            "subsets": {fp.run_tag(a, s): {"file": fp.subset_rel(a, s), "sha1": files[fp.subset_rel(a, s)]}
                        for a in fp.ARMS for s in fp.SEEDS},
            "pruned_train_files": {fp.run_tag(a, s): {"file": fp.train_rel(a, s), "sha1": files[fp.train_rel(a, s)]}
                                   for a in fp.ARMS for s in fp.SEEDS if fp.has_train_file(a, s)},
            "train_jsonl": {"sha1": dig(man, "inputs", "train_jsonl", "sha1"), "recorded_by": "the prune manifest",
                            "equals_ftgrid_split_record": True, "file_checked_here": bool(train)},
            "gateft_split": {"sha1": chk["gateft"].get("sha1"), "T": chk["gateft"].get("T"),
                             "recorded_by": "the prune manifest (not re-read here)"},
            "eval_panel": {"sha1": chk["eval_sha1"], "rows": ev["rows"], "candidates": ev["candidates"]},
            "ftgrid_split": {"sha1": chk["split_sha1"], "T": rec["T"]}},
        "train": {"examples": len(tt["keys"]), "positives": int(y.sum()), "negatives": int((y == 0).sum()), "items": k,
                  "items_singleton": int((n == 1).sum()), "items_in_scope_b": int(in_scope.sum()),
                  "examples_in_scope_b": int(n[in_scope].sum())},
        "test": {"T": rec["T"], "pairs": n_test, "pairs_item_in_train": int(test_count.sum()),
                 "pairs_item_not_in_train": int(n_test - int(test_count.sum()))},
        "terciles": {"rule": "stats.rank_bins(n, 3) == 2, n = TRAIN examples per item (average ranks: equal n share a tercile)",
                     "n_items_head": n_head, "n_items_middle": int(k - n_head - n_tail), "n_items_tail": n_tail,
                     "head_min_popularity": int(n[head].min()) if n_head else None,
                     "head_example_share_P0": as_float(per_seed[("P0", 0)]["d_head_tercile_example_share"])},
        "arms": arms,
        "checks": {
            "manifest_tree_verified": True, "signals_counts_match_manifest": True, "subset_counts_match_manifest": True,
            "ftgrid_split_T_equals_manifest_T": True, "ftgrid_train_sha1_equals_manifest": True,
            "eval_sha1_equals_ftgrid_split_record": True, "eval_test_pairs_equal_ftgrid_split_record": True,
            "P0_seen_share_equals_ftgrid_split_record": True,
            "train_file_equals_signals_rows": True if train else None,
            "P2_P3_subsets_identical_across_seeds": bool(arms["P2"]["subsets_identical_across_seeds"]
                                                         and arms["P3"]["subsets_identical_across_seeds"]),
            "P1_subsets_differ_across_seeds": not arms["P1"]["subsets_identical_across_seeds"]},
        "definitions": list(DEFINITIONS),
    }


# ---------------------------------------------------------------- output
def _num(x) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return str(int(x))
    if isinstance(x, int):
        return str(x)
    return repr(float(x))


def _mean_count(values: list) -> str:
    """The exact mean of integer counts: an integer when it is one, else the float."""
    m = Fraction(sum(values), len(values))
    return str(m.numerator) if m.denominator == 1 else repr(float(m))


def result_csv(res: dict) -> str:
    """One row per arm (seed = mean) and one per arm and seed: the four statistics, then three counts."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLS)
    for arm, a in res["arms"].items():
        seeds = [a["per_seed"][f"s{s}"] for s in a["seeds"]]
        w.writerow([arm, "mean"] + [_num(a["mean"][k]) for k in STATS]
                   + [_mean_count([v[c] for v in seeds]) for c in ("n_kept", "items_without_example",
                                                                   "b_items_in_scope_lost")])
        for s, v in zip(a["seeds"], seeds):
            w.writerow([arm, s] + [_num(v[k]) for k in STATS]
                       + [_num(v["n_kept"]), _num(v["items_without_example"]), _num(v["b_items_in_scope_lost"])])
    return buf.getvalue()


def summary_lines(res: dict) -> list:
    t, te = res["train"], res["test"]
    lines = [f"S6 composition diagnostics ({res['domain']}): {t['examples']} TRAIN examples, {t['items']} items "
             f"({res['terciles']['n_items_head']} in the head tercile), {te['pairs']} TEST pairs",
             "  arm  " + "  ".join(f"{k[:1]}={k[2:]}" for k in STATS)]
    for arm, a in res["arms"].items():
        vals = "  ".join("n/a" if a["mean"][k] is None else f"{a['mean'][k]:.6f}" for k in STATS)
        lines.append(f"  {arm}   {vals}   (mean over seeds {a['seeds'][0]}-{a['seeds'][-1]})")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S6 composition diagnostics (A3 addendum 7 section 2)")
    ap.add_argument("--manifest", required=True, help="outputs/confrec/ftprune/prune_manifest.json")
    ap.add_argument("--out_dir", required=True, help="the only directory written (ftprune_compose.json / .csv)")
    ap.add_argument("--eval_panel", default=DEFAULT_EVAL, help="ftgrid_data's ML-1M eval.jsonl (the EVAL users' rows)")
    ap.add_argument("--ftgrid_split", default=DEFAULT_SPLIT, help="ftgrid_data's ML-1M ftgrid_split.json (T, sha1 records)")
    ap.add_argument("--train", default=None,
                    help="optional: Gate-FT's train.jsonl, re-read and checked against the signals rows")
    a = ap.parse_args(argv)
    try:
        res = compose(a.manifest, a.eval_panel, a.ftgrid_split, a.train)
    except Refusal as e:
        for p in e.problems:
            print("REFUSED:", p, file=sys.stderr)
        print(f"ftprune_compose: refused ({len(e.problems)} problem(s)); nothing was written", file=sys.stderr)
        return 2
    out = Path(a.out_dir)
    fp.put(out / OUT_JSON, fp.json_bytes(res))
    fp.put(out / OUT_CSV, result_csv(res).encode("utf-8"))
    for line in summary_lines(res):
        print(line)
    print(f"wrote {out / OUT_JSON} and {out / OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
