"""Data roles of the Amendment-3 fine-tuned program (idea-stage/PREREG_AMENDMENT_3.md section 1, and the FT-C file of
section 9; interfaces: docs/sigir/FTGRID_IMPL_SPEC.md).

    python -m src.confrec.ftgrid_data --domain ml1m --panel_all PANEL_ALL --out_dir outputs/confrec/ftgrid/panels/ml1m \
        [--variant V0] [--tokenizer MODEL_DIR] [--n_train 1500] [--n_eval_max 3000] [--quantile 0.8] [--s_max 1000] \
        [--train_cap 24000] [--seed 0] [--gateft_split outputs/confrec/gateft/gateft_split.json] [--ml1m_dev_users_sha1 SHA]

Input: the hist_len-20 rated panel of ALL eligible users of the domain (build_rated_panels, seed 0) in its row order, the
seed-0 shuffle. "Position" is that row order, never a sorted user-id list. Every rule below is checked here (SystemExit on
a bad input, AssertionError on a broken invariant; neither is stripped by python -O).
  * TRAIN = the first n_train rows (1,500); EVAL = the next min(n_eval_max, rows - n_train) rows (3,000 cap). TRAIN and EVAL
    are disjoint by user_id, and neither holds a user twice. --ml1m_dev_users_sha1 (alias --dev_users_sha1): the TRAIN user
    ids, sorted as Python strings and joined by '\\n' (the DEV list format of build_confirm_panels.py), must hash to it.
  * T = gateft_data.split_threshold(TRAIN, EVAL, quantile): the numpy-linear quantile of every candidate timestamp of
    TRAIN u EVAL (timestamps only). --gateft_split (ml1m only): T must equal Gate-FT's T, and train.jsonl must be Gate-FT's
    TRAIN file byte for byte when gateft_split.json records its sha1 ("ML-1M is the G9 set").
  * train.jsonl = gateft_data.build_train(TRAIN, T): each TRAIN row keeps its candidates with ts < T (every candidate_* list
    filtered together, the history untouched, a row left without a candidate dropped). Above --train_cap examples a
    uniform subset of exactly train_cap examples is kept: random.Random(seed).sample over the examples in row / candidate
    order (rows left empty are dropped). A panel row has no history timestamps; its source_event_id ends in its first
    candidate's timestamp, which its history precedes (build_rated_panels): that timestamp must be <= every candidate
    timestamp of the TRAIN row, so a training example's history precedes T.
  * eval.jsonl = the EVAL rows, byte for byte. CAL = EVAL candidates with ts < T, TEST = EVAL candidates with ts >= T.
  * Power facts (labels, timestamps and popularity only): EVAL users with both classes among their TEST candidates; among
    their TEST candidates in the popularity tail (stats.rank_bins(candidate_popularity over all EVAL candidates, 5) == 0);
    over all their candidates; and the share of TEST pairs whose item_id has a TRAIN example (an example of train.jsonl).
  * S_d = the first s_max (1,000) EVAL users in row order with both classes among their TEST candidates (all, if fewer);
    eval_sd_test.jsonl = their rows restricted to TEST candidates.
  * TRAIN length rule: every train.jsonl example's `like` prompt under --variant is rendered and tokenised as the trainer
    does it (prompting.render_record + chat_ids, the variant's registered history window); length = prompt ids + 1 answer
    token. share_above_1024 = share of lengths > 1024 (the trainer's skip rule) and p995 = the numpy-linear 99.5th
    percentile are recorded; max_len_used = 1024 if the share <= 2%, else p995 rounded up to a multiple of 256; micro-batch
    x accumulation = 8 x 4 at max_len_used <= 1024, else 4 x 8 (effective batch 32). Without a tokenizer (--tokenizer, or
    one passed to main()) the share and p995 are null and max_len_used is 1024; the prompts are still rendered (checks).
    The trainer's / scorer's panel rules are applied first: rated rows only, the variant's history window must exist.
  * train_perm.jsonl (--domain ml1m, always written there; A3 section 9): within-item label permutation of train.jsonl. For
    every item with >= 2 examples the labels of its examples are permuted uniformly at random (one random.Random(seed)
    stream, items in sorted item-id order, each item's examples in train.jsonl order); a candidate's star rating moves with
    its label (rows with candidate_ratings), so label == (rating >= 4) still holds; items with one example are unchanged.
    Asserted: every item's label sum and rating multiset are unchanged and every other field is identical.
Outputs (--out_dir): train.jsonl, [train_perm.jsonl], eval.jsonl, eval_sd_test.jsonl, train_users.txt, eval_users.txt,
sd_users.txt (one id per line in panel row order, joined by '\\n', no trailing newline, UTF-8: user_ids_sha1 = sha1 of the
file) and ftgrid_split.json (the schema of the spec; strict JSON, allow_nan=False; no wall-clock, host or path, so a rerun
on the same inputs is byte-identical; `files` = the sha1 of every file written here). Everything is staged as temp files and
committed together, the json last; a file whose bytes do not change is left untouched (build_rated_panels.commit), and a
train_perm.jsonl that this run did not write (another domain) is removed, so `files` lists every data file of the directory
that this builder owns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.confrec import build_rated_panels as brp
from src.confrec import gateft_data as gd
from src.confrec.prompting import (GATE_VARIANTS, chat_ids, panel_kind_of, render_record, resolve_hist_len,
                                   short_history_error)
from src.confrec.split_panel import filter_candidates
from src.confrec.stats import rank_bins, strict_json

MAX_LEN_REGISTERED = 1024        # G9's max_len (skip-and-count)
OVERLENGTH_PCT = 2               # A3 section 1: above 2% the domain trains with max_len = p99.5 rounded up to 256
LEN_MULTIPLE = 256
BATCH_SHORT, BATCH_LONG = (8, 4), (4, 8)   # (micro-batch, accumulation): effective batch 32 either way
TAIL_BINS = 5                    # popularity tail = rank_bins(pop, 5) == 0
MIN_N = 150                      # A3 section 1 minimum n (endpoints on fewer users are descriptive)
PERM_DOMAIN = "ml1m"             # A3 section 9: FT-C on the ML-1M TRAIN examples
SPLIT_JSON = "ftgrid_split.json"
SD_USERS = "sd_users.txt"
REQUIRED = ("candidate_item_ids", "candidate_labels", "candidate_timestamps", "candidate_popularity")
MOVED = ("candidate_labels", "candidate_ratings")   # what the FT-C permutation moves


def _check(cond, msg: str) -> None:
    """An invariant of this module (kept under python -O)."""
    if not cond:
        raise AssertionError(msg)


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def ids_bytes(ids) -> bytes:
    """A user-id list file: one id per line, joined by '\\n', no trailing newline, UTF-8."""
    return "\n".join(str(u) for u in ids).encode("utf-8")


def row_bytes(row: dict) -> bytes:
    """One panel line exactly as build_rated_panels and gateft_data write it."""
    return brp.row_line(row).encode("utf-8")


def n_examples(rows: list) -> int:
    return sum(len(r["candidate_labels"]) for r in rows)


def both_classes(labels) -> bool:
    return 0 < sum(labels) < len(labels)


def read_panel(path, n_parse: int) -> tuple[str, int, list]:
    """(sha1 of the file bytes, number of rows, [(line bytes, row)] of the first n_parse rows). Blank lines are not rows
    (as gateft_data.read_rows); a last line without its newline gets one; rows after the first n_parse are only counted."""
    h, n, head = hashlib.sha1(), 0, []
    with open(path, "rb") as f:
        for line in f:
            h.update(line)
            if not line.strip():
                continue
            if n < n_parse:
                head.append((line if line.endswith(b"\n") else line + b"\n", json.loads(line)))
            n += 1
    return h.hexdigest(), n, head


def check_rows(rows: list, role: str) -> None:
    """Rated-panel rows with aligned candidate lists, binary labels and finite timestamps / popularity."""
    for k, r in enumerate(rows):
        where = f"{role} row {k} (user {r.get('user_id')!r})"
        if "user_id" not in r:
            raise SystemExit(f"{where}: no user_id")
        if panel_kind_of(r) != "rated":
            raise SystemExit(f"{where}: not a rated-panel row (needs candidate_labels and history_ratings)")
        missing = [f for f in REQUIRED if f not in r]
        if missing:
            raise SystemExit(f"{where}: missing {missing} (rebuild the panel with build_rated_panels)")
        n = len(r["candidate_labels"])
        if n == 0:
            raise SystemExit(f"{where}: no candidate")
        bad = {f: len(v) for f, v in r.items() if f.startswith("candidate_") and isinstance(v, list) and len(v) != n}
        if bad:
            raise SystemExit(f"{where}: candidate lists misaligned with candidate_labels ({n}): {bad}")
        if any(y not in (0, 1) for y in r["candidate_labels"]):
            raise SystemExit(f"{where}: candidate_labels must be 0/1")
        for f in ("candidate_timestamps", "candidate_popularity"):
            if not np.isfinite(np.asarray(r[f], float)).all():
                raise SystemExit(f"{where}: non-finite {f}")


def check_roles(train_rows: list, eval_rows: list) -> tuple[list, list]:
    """(TRAIN user ids, EVAL user ids) in row order; refuses shared or repeated users."""
    tu, eu = [str(r["user_id"]) for r in train_rows], [str(r["user_id"]) for r in eval_rows]
    shared = set(tu) & set(eu)
    if shared:
        raise SystemExit(f"TRAIN and EVAL share {len(shared)} users (e.g. {sorted(shared)[:3]}): A3 section 1 requires "
                         "disjoint user sets")
    for name, ids in (("TRAIN", tu), ("EVAL", eu)):
        if len(set(ids)) != len(ids):
            dup = sorted({u for u in ids if ids.count(u) > 1})
            raise SystemExit(f"{name} holds a user twice (e.g. {dup[:3]}): one row per user")
    return tu, eu


def check_dev_users(train_ids: list, sha: str) -> None:
    """The TRAIN users are the registered DEV list (build_confirm_panels format: ids sorted as Python strings, '\\n')."""
    got = sha1_bytes("\n".join(sorted(train_ids)).encode("utf-8"))
    if got != sha.strip().lower():
        raise SystemExit(f"the TRAIN users (the first {len(train_ids)} rows) are not the DEV list: sha1 of the sorted ids "
                         f"{got} != {sha} (A3 section 1: TRAIN = the Pilot-1 users)")


def first_candidate_ts(row: dict) -> int:
    """The timestamp that ends a build_rated_panels source_event_id ('<user>::<ts of the first candidate>')."""
    sid = str(row.get("source_event_id", ""))
    _, sep, tail = sid.rpartition("::")
    if not sep or not re.fullmatch(r"-?\d+", tail):
        raise SystemExit(f"user {row.get('user_id')!r}: source_event_id {sid!r} does not end in '::<first candidate "
                         "timestamp>' (build_rated_panels), so the history-before-T rule cannot be checked")
    return int(tail)


def check_history_precedes(rows: list) -> None:
    """A3 section 1: a training example's history precedes all of that user's candidates (hence T)."""
    for r in rows:
        t0 = first_candidate_ts(r)
        if t0 > min(float(t) for t in r["candidate_timestamps"]):
            raise SystemExit(f"TRAIN user {r['user_id']!r}: source_event_id timestamp {t0} is after a candidate "
                             "timestamp: the history would not precede every candidate")


def check_variant(rows: list, variant: str, name: str) -> None:
    """The scorer's / trainer's rule: the variant's registered history window must exist in the panel."""
    hist = resolve_hist_len(variant, "rated")
    max_hist = max((len(r.get("history") or []) for r in rows), default=0)
    err = short_history_error(variant, "rated", hist, max_hist)
    if err:
        raise SystemExit(f"{name}: {err}")


def cap_examples(rows: list, cap: int, seed: int) -> tuple[list, bool]:
    """(rows holding a seeded uniform subset of exactly `cap` examples, True) above the cap, else (rows, False)."""
    n = n_examples(rows)
    if n <= cap:
        return rows, False
    keep = set(random.Random(seed).sample(range(n), cap))
    out, k = [], 0
    for r in rows:
        m = len(r["candidate_labels"])
        flags = [k + j in keep for j in range(m)]
        k += m
        if any(flags):
            out.append(filter_candidates(r, flags))
    return out, True


def train_prompt_lengths(rows: list, variant: str, tok=None) -> list | None:
    """Prompt ids + 1 (the answer token) of every example's `like` prompt under `variant`, tokenised as
    train_lora_yesno.YesNoSet does it; without a tokenizer the prompts are only rendered (field checks): None."""
    lengths = []
    for r in rows:
        try:
            prompts = render_record(r, ["like"], variant)
        except ValueError as e:
            raise SystemExit(f"train.jsonl under variant {variant}: {e}") from None
        if tok is not None:
            lengths += [len(chat_ids(tok, user, system)) + 1 for _, _, system, user in prompts]
    return lengths if tok is not None else None


def length_rule(lengths) -> dict:
    """A3 section 1 TRAIN length rule from the example lengths (None = no tokenizer)."""
    if lengths is None:
        share = p995 = None
        max_len = MAX_LEN_REGISTERED
    else:
        if not len(lengths):
            raise ValueError("no TRAIN example to measure")
        x = np.asarray(lengths, float)
        n_above = int((x > MAX_LEN_REGISTERED).sum())
        share = n_above / len(x)
        p995 = float(np.percentile(x, 99.5))
        if 100 * n_above <= OVERLENGTH_PCT * len(x):
            max_len = MAX_LEN_REGISTERED
        else:
            max_len = LEN_MULTIPLE * math.ceil(p995 / LEN_MULTIPLE)
            _check(max_len >= MAX_LEN_REGISTERED, f"p99.5 = {p995} with {share:.2%} above {MAX_LEN_REGISTERED}")
    micro, accum = BATCH_SHORT if max_len <= MAX_LEN_REGISTERED else BATCH_LONG
    return {"max_len_registered": MAX_LEN_REGISTERED, "share_above_1024": share, "p995": p995, "max_len_used": max_len,
            "micro_bsz": micro, "grad_accum": accum}


def eval_facts(rows: list, T: float, seen_items: set, s_max: int) -> tuple[dict, list]:
    """(the EVAL counts and power facts of the split json, S_d rows in row order)."""
    pop = np.concatenate([np.asarray(r["candidate_popularity"], float) for r in rows])
    tail = rank_bins(pop, TAIL_BINS) == 0
    f = {"candidates": 0, "cal_candidates": 0, "test_candidates": 0, "users_both_classes_test": 0,
         "users_both_classes_test_tail": 0, "users_both_classes_all_rows": 0}
    n_seen, k, sd = 0, 0, []
    for r in rows:
        lab = [int(y) for y in r["candidate_labels"]]
        test = [j for j, t in enumerate(r["candidate_timestamps"]) if float(t) >= T]
        y_test = [lab[j] for j in test]
        f["candidates"] += len(lab)
        f["test_candidates"] += len(test)
        f["cal_candidates"] += len(lab) - len(test)
        f["users_both_classes_test"] += both_classes(y_test)
        f["users_both_classes_test_tail"] += both_classes([lab[j] for j in test if tail[k + j]])
        f["users_both_classes_all_rows"] += both_classes(lab)
        n_seen += sum(str(r["candidate_item_ids"][j]) in seen_items for j in test)
        if both_classes(y_test) and len(sd) < s_max:
            sd.append(r)
        k += len(lab)
    f["share_test_pairs_item_seen_in_train"] = n_seen / f["test_candidates"] if f["test_candidates"] else float("nan")
    return f, sd


def permute_within_item(rows: list, seed: int) -> tuple[list, int]:
    """(train_perm rows, number of labels changed): A3 section 9, see the module docstring."""
    has_r = ["candidate_ratings" in r for r in rows]
    if any(has_r) and not all(has_r):
        raise SystemExit("candidate_ratings is present in some TRAIN rows only")
    moved = MOVED if all(has_r) else MOVED[:1]
    where = defaultdict(list)                       # item -> [(row, candidate)] in train.jsonl order
    for k, r in enumerate(rows):
        for j, it in enumerate(r["candidate_item_ids"]):
            where[str(it)].append((k, j))
    out = [dict(r, **{f: list(r[f]) for f in moved}) for r in rows]
    rng = random.Random(seed)
    for item in sorted(where):
        pos = where[item]
        if len(pos) < 2:
            continue
        src = list(range(len(pos)))
        rng.shuffle(src)
        for (k, j), s in zip(pos, src):
            sk, sj = pos[s]
            for f in moved:
                out[k][f][j] = rows[sk][f][sj]
    # the registered assertions: every item keeps its label sum (and rating multiset), nothing else changes
    for r, p in zip(rows, out):
        _check(list(r) == list(p) and all(r[f] == p[f] for f in r if f not in moved), "FT-C changed a non-label field")
        _check(all(len(p[f]) == len(r[f]) for f in moved), "FT-C changed a candidate list length")
    for item, pos in where.items():
        _check(sum(int(rows[k]["candidate_labels"][j]) for k, j in pos)
               == sum(int(out[k]["candidate_labels"][j]) for k, j in pos), f"item {item}: label sum not preserved")
        for f in moved:
            _check(sorted(rows[k][f][j] for k, j in pos) == sorted(out[k][f][j] for k, j in pos),
                   f"item {item}: {f} multiset not preserved")
        if len(pos) == 1:
            k, j = pos[0]
            _check(all(out[k][f][j] == rows[k][f][j] for f in moved), f"item {item}: a single example changed")
    changed = sum(int(r["candidate_labels"][j] != p["candidate_labels"][j])
                  for r, p in zip(rows, out) for j in range(len(r["candidate_labels"])))
    return out, changed


def check_gateft(path, T: float, train_sha1: str) -> bool:
    """A3 section 1: the ML-1M T equals Gate-FT's T, and (when recorded) train.jsonl is Gate-FT's TRAIN file."""
    g = json.loads(Path(path).read_text(encoding="utf-8"))
    if g.get("T") is None:
        raise SystemExit(f"{path} records no T")
    if float(g["T"]) != T:
        raise SystemExit(f"T = {T!r} differs from Gate-FT's T = {g['T']!r} ({path}): A3 section 1 requires the ML-1M T "
                         "to equal Gate-FT's")
    gs = (g.get("train") or {}).get("sha1")
    if gs is not None and gs != train_sha1:
        raise SystemExit(f"train.jsonl (sha1 {train_sha1}) is not Gate-FT's TRAIN file (sha1 {gs}, {path}): A3 section 1 "
                         "says the ML-1M TRAIN examples are the G9 set")
    return True


def stage(out: Path, name: str, chunks, staged: dict) -> str:
    """Write the byte chunks to <out>/<name>.tmp (committed later); their sha1."""
    tmp = out / f"{name}.tmp"
    staged[name] = tmp
    h = hashlib.sha1()
    with open(tmp, "wb") as f:
        for b in chunks:
            f.write(b)
            h.update(b)
    return h.hexdigest()


def load_tokenizer(path: str):
    """The training model's tokenizer; transformers is imported here only, never at module import."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)


def build(a, tok=None) -> dict:
    out = Path(a.out_dir)
    panel_sha1, n_rows, head = read_panel(a.panel_all, a.n_train + a.n_eval_max)
    if n_rows <= a.n_train:
        raise SystemExit(f"{a.panel_all}: {n_rows} rows; TRAIN takes the first {a.n_train}, so no EVAL row is left")
    n_eval = min(a.n_eval_max, n_rows - a.n_train)
    eval_part = head[a.n_train:a.n_train + n_eval]
    train_rows, eval_rows = [r for _, r in head[:a.n_train]], [r for _, r in eval_part]
    check_rows(train_rows, "TRAIN")
    check_rows(eval_rows, "EVAL")
    train_ids, eval_ids = check_roles(train_rows, eval_rows)
    if a.dev_users_sha1:
        check_dev_users(train_ids, a.dev_users_sha1)
    check_history_precedes(train_rows)

    # the time split, computed once (Gate-FT's function)
    T = gd.split_threshold(train_rows, eval_rows, a.quantile)
    pre = gd.build_train(train_rows, T)
    n_pre = n_examples(pre)
    if not n_pre:
        raise SystemExit(f"no TRAIN candidate precedes T = {T}")
    train, cap_applied = cap_examples(pre, a.train_cap, a.seed)
    tu = {str(r["user_id"]) for r in train}
    _check(tu <= set(train_ids) and not tu & set(eval_ids), "train.jsonl holds a non-TRAIN user")
    _check(all(float(t) < T for r in train for t in r["candidate_timestamps"]), "train.jsonl holds an event at/after T")
    _check(all(first_candidate_ts(r) < T for r in train), "a training example's history does not precede T")
    _check(n_examples(train) == min(n_pre, a.train_cap), "the cap kept the wrong number of examples")
    check_rows(train, "train.jsonl")

    # prompts: the scorer's / trainer's panel rules, then the TRAIN length rule
    check_variant(train, a.variant, "train.jsonl")
    check_variant(eval_rows, a.variant, "EVAL")
    overlength = length_rule(train_prompt_lengths(train, a.variant, tok))

    seen = {str(it) for r in train for it in r["candidate_item_ids"]}
    facts, sd_rows = eval_facts(eval_rows, T, seen, a.s_max)
    sd_test = [filter_candidates(r, [float(t) >= T for t in r["candidate_timestamps"]]) for r in sd_rows]
    sd_ids = [str(r["user_id"]) for r in sd_rows]
    perm = None
    if a.domain == PERM_DOMAIN:
        perm, n_changed = permute_within_item(train, a.seed)
        if not n_changed:
            print("WARNING: the within-item permutation changed no label (no item has examples of both classes)",
                  file=sys.stderr)

    out.mkdir(parents=True, exist_ok=True)
    staged: dict = {}
    try:
        sha = {"train.jsonl": stage(out, "train.jsonl", map(row_bytes, train), staged)}
        if perm is not None:
            sha["train_perm.jsonl"] = stage(out, "train_perm.jsonl", map(row_bytes, perm), staged)
        sha["eval.jsonl"] = stage(out, "eval.jsonl", (line for line, _ in eval_part), staged)
        sha["eval_sd_test.jsonl"] = stage(out, "eval_sd_test.jsonl", map(row_bytes, sd_test), staged)
        for name, ids in (("train_users.txt", train_ids), ("eval_users.txt", eval_ids), (SD_USERS, sd_ids)):
            sha[name] = stage(out, name, [ids_bytes(ids)], staged)
        gate = check_gateft(a.gateft_split, T, sha["train.jsonl"]) if a.gateft_split else None
        rep = strict_json({
            "domain": a.domain, "panel_all_sha1": panel_sha1, "n_rows": n_rows, "variant": a.variant,
            "quantile": a.quantile, "T": T,
            "train": {"users": len(train_ids), "user_ids_sha1": sha["train_users.txt"],
                      "candidates_total": n_examples(train_rows), "examples_pre_T": n_pre, "cap": a.train_cap,
                      "cap_applied": cap_applied, "examples_written": n_examples(train), "overlength": overlength},
            "eval": {"users": len(eval_ids), "user_ids_sha1": sha["eval_users.txt"], **facts},
            "sd": {"users": len(sd_ids), "user_ids_sha1": sha[SD_USERS], "user_ids_path": SD_USERS},
            "files": dict(sorted(sha.items())),
            "code_sha1": {"ftgrid_data.py": gd.sha1_file(__file__), "gateft_data.py": gd.sha1_file(gd.__file__),
                          "build_rated_panels.py": gd.sha1_file(brp.__file__)},
            "args": {"seed": a.seed, "n_train": a.n_train, "n_eval_max": a.n_eval_max, "s_max": a.s_max,
                     "train_cap": a.train_cap, "tokenizer_used": tok is not None,
                     "tokenizer": Path(a.tokenizer).name if a.tokenizer else None,
                     "dev_users_sha1_checked": bool(a.dev_users_sha1), "gateft_split_checked": bool(a.gateft_split)},
            "gateft_T_match": gate})
        _check(facts["cal_candidates"] + facts["test_candidates"] == facts["candidates"], "CAL + TEST != EVAL")
        text = json.dumps(rep, indent=2, allow_nan=False) + "\n"
        stage(out, SPLIT_JSON, [text.encode("utf-8")], staged)
        for name in [n for n in staged if n != SPLIT_JSON] + [SPLIT_JSON]:   # the json last
            brp.commit(staged.pop(name), out / name)
        if perm is None:   # a permutation file this run did not write must not linger next to its json
            (out / "train_perm.jsonl").unlink(missing_ok=True)
    finally:
        for tmp in staged.values():   # staged but not committed (an error)
            tmp.unlink(missing_ok=True)
    for key in ("users_both_classes_test", "users_both_classes_test_tail"):
        if facts[key] < MIN_N:
            print(f"NOTE {a.domain}: {key} = {facts[key]} < {MIN_N}: endpoints on these users are descriptive (A3 "
                  "section 1 minimum n)", file=sys.stderr)
    print(json.dumps(rep, allow_nan=False))
    return rep


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--domain", required=True, help="ml1m, toys, games or sports (ml1m also writes train_perm.jsonl)")
    ap.add_argument("--panel_all", required=True, help="hist_len-20 rated panel of all eligible users, in row order")
    ap.add_argument("--out_dir", required=True, help="panels/<domain>/")
    ap.add_argument("--variant", default="V0", choices=list(GATE_VARIANTS), help="selection.json gate_ft_prompt")
    ap.add_argument("--tokenizer", default=None, help="model dir whose tokenizer measures the TRAIN prompts")
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--n_eval_max", type=int, default=3000)
    ap.add_argument("--quantile", type=float, default=0.8)
    ap.add_argument("--s_max", type=int, default=1000)
    ap.add_argument("--train_cap", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gateft_split", default=None, help="Gate-FT's gateft_split.json (ml1m only)")
    ap.add_argument("--ml1m_dev_users_sha1", "--dev_users_sha1", dest="dev_users_sha1", default=None,
                    help="sha1 of the DEV user-id list (build_confirm_panels manifest, sorted ids)")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.domain):
        ap.error(f"--domain {a.domain!r}: letters, digits, '_' or '-' only")
    if not 0.0 < a.quantile < 1.0:
        ap.error("--quantile must be in (0, 1)")
    for k in ("n_train", "n_eval_max", "s_max", "train_cap"):
        if getattr(a, k) < 1:
            ap.error(f"--{k} must be >= 1")
    if a.gateft_split and a.domain != PERM_DOMAIN:
        ap.error("--gateft_split is Gate-FT's ML-1M split: only with --domain ml1m")
    for k in ("panel_all", "gateft_split"):
        if getattr(a, k) and not Path(getattr(a, k)).is_file():
            ap.error(f"--{k} {getattr(a, k)}: no such file")
    return a


def main(argv=None, tokenizer=None) -> dict:
    """The split dict (as written to ftgrid_split.json). `tokenizer` overrides --tokenizer (tests pass a fake one)."""
    a = parse_args(argv)
    if tokenizer is None and a.tokenizer:
        tokenizer = load_tokenizer(a.tokenizer)
    return build(a, tokenizer)


if __name__ == "__main__":
    main()
