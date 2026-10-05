"""Tests of the S6 composition diagnostics (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_7.md section 2):
src/confrec/ftprune_compose.py. CPU only, deterministic, no network, no GPU, no model; torch / transformers never imported.

  * each statistic (a no-example item share, b mean absolute item-rate change, c TEST-pair seen share, d head-tercile example
    share) against a direct brute-force computation (pure Python and Fractions, a different formulation from the module's: per-item
    rationals, and the head tercile by an exact integer rule on counts instead of rank_bins' float ranks) on
      - a hand-built world whose numbers were worked out by hand: an item that loses all its examples, singleton items, a tie group
        of popularity exactly on the tercile cut, a TEST pair at ts == T, CAL pairs that must not count, P1's five different draws;
      - worlds with every item tied in popularity (the head is empty), with no item of two examples (b undefined), with an arm
        identical to P0 (classes of one example: n_remove = 0);
      - random worlds (skewed popularity, many ties, unseen TEST items);
      - the REAL prune pipeline: tests/test_confrec_ftprune.py's world (raw ML-1M -> panels -> Gate-FT data -> planted zero-shot
        pass -> ftprune signals) with a real ftgrid_data run on top (EVAL panel, ftgrid_split.json), and the exact server command;
  * the head tercile is cut over the full TRAIN set, never on an arm's kept examples; tercile_bins = the exact integer rule;
    exact_mean / exact_sd against `statistics`;
  * the refusals (exit 2, nothing written): a recorded sha1 that does not match the file (signals, each arm's subset, a pruned
    TRAIN file, the manifest's own record, the EVAL panel, --train), a missing file, and inputs that are consistent in sha1 but are
    not the registered ones (keys that are not TRAIN examples, class counts, T, TRAIN sha1, TEST count, P0's seen share, a split
    without ftgrid_data's fields, no TEST pair, a malformed EVAL row);
  * determinism: byte-identical reruns (in place without touching a file, in a fresh directory, on a copy of the world at another
    path, in processes with other PYTHONHASHSEEDs), strict JSON, no clock / host / path;
  * the module imports ftprune read-only (AST: no assignment to its attributes; run: its namespace and the bound files unchanged)
    and writes nowhere but --out_dir (AST: the only writes are the two fp.put calls on out_dir; run: a snapshot of the world, the
    working directory and the bound files before and after), and its import needs no torch / transformers.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import io
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.confrec import ftgrid_data as fd
from src.confrec import ftprune as fp
from src.confrec import ftprune_compose as fc

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "confrec" / "ftprune_compose.py"
BOUND = (ROOT / "src" / "confrec" / "ftprune.py", ROOT / "scripts" / "sigir" / "run_ftprune.sh",
         ROOT / "tests" / "test_confrec_ftprune.py")
STATS = fc.STATS
A, B, C, D = STATS


def _load(name: str):
    """A sibling test module as a library (its world builders; it is never edited here)."""
    spec = importlib.util.spec_from_file_location(f"{name}_for_compose", ROOT / "tests" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PT = _load("test_confrec_ftprune")        # build_world / build: the synthetic prune world


# ---------------------------------------------------------------- small helpers
def sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def sha1_file(p) -> str:
    return sha1(Path(p).read_bytes())


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj) -> None:
    Path(p).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")


def edit_json(p, fn) -> None:
    obj = read_json(p)
    fn(obj)
    write_json(p, obj)


def jsonl(p) -> list:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def jsonl_bytes(rows: list) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def class_counts(y, mask) -> dict:
    y, mask = np.asarray(y, int), np.asarray(mask, bool)
    return {"1": int(((y == 1) & mask).sum()), "0": int(((y == 0) & mask).sum())}


# ---------------------------------------------------------------- hand-built worlds in ftprune's own formats
def rows_from(examples: list) -> list:
    """[(user, item, label)] -> train.jsonl rows, one per user in order of first appearance (candidates in example order)."""
    by_user = {}
    for j, (u, i, y) in enumerate(examples):
        by_user.setdefault(u, []).append((i, y, 10 + j))
    return [{"user_id": u, "source_event_id": f"{u}::{min(t for _, _, t in c)}",
             "candidate_item_ids": [i for i, _, _ in c], "candidate_labels": [y for _, y, _ in c],
             "candidate_timestamps": [t for _, _, t in c]} for u, c in by_user.items()]


def write_prune_tree(out: Path, rows: list, removed: dict, T: float, train_sha1: str) -> dict:
    """A prune output directory (signals, 20 subset files, 17 pruned TRAIN files, manifest) written with ftprune's own writers
    for hand-designed subsets: removed[(arm, seed)] = the keys that arm removes (missing = nothing)."""
    ex = fp.train_examples(rows)
    y, n = ex["y"], ex["n"]
    train_bytes = jsonl_bytes(rows)
    nan = np.full(n, np.nan)
    data = fp.gz_bytes(fp.signals_text(ex, nan, nan, nan, nan, [fp.tie_key(k) for k in ex["key"]]))
    fp.put(out / fp.SIGNALS_FILE, data)
    files, sub, trn = {fp.SIGNALS_FILE: sha1(data)}, {}, {}
    for arm in fp.ARMS:
        for s in fp.SEEDS:
            rm = np.array([k in removed.get((arm, s), ()) for k in ex["key"]], bool)
            keep = ~rm
            rel = fp.subset_rel(arm, s)
            data = "\n".join(k for k, kp in zip(ex["key"], keep) if kp).encode("utf-8")
            fp.put(out / rel, data)
            files[rel] = sha1(data)
            sub[fp.run_tag(arm, s)] = {"file": rel, "sha1": files[rel], "n_keep": class_counts(y, keep),
                                       "n_removed": class_counts(y, rm)}
            if fp.has_train_file(arm, s):
                rel = fp.train_rel(arm, s)
                if arm == "P0":
                    data, n_rows, n_ex, n_pos = train_bytes, len(rows), n, int(y.sum())
                else:
                    txt, n_rows, n_ex, n_pos = fp.pruned_text(rows, ex, keep)
                    data = txt.encode("utf-8")
                fp.put(out / rel, data)
                files[rel] = sha1(data)
                trn[fp.run_tag(arm, s)] = {"file": rel, "sha1": files[rel], "rows": n_rows, "examples": n_ex,
                                           "positives": n_pos, "negatives": n_ex - n_pos}
    pos, neg = int(y.sum()), int(n - y.sum())
    man = {"format": fp.MANIFEST_FORMAT, "domain": "ml1m", "variant": "V1",
           "inputs": {"train_jsonl": {"sha1": train_sha1, "rows": len(rows), "examples": int(n), "positives": pos,
                                      "negatives": neg},
                      "gateft_split": {"sha1": sha1(b"gateft_split.json of the world"), "T": T, "train_rows": len(rows),
                                       "train_candidates": int(n), "train_positive_rate": pos / n}},
           "signals": {"file": fp.SIGNALS_FILE, "sha1": files[fp.SIGNALS_FILE]},
           "classes": {"1": {"n": pos, "n_remove": fp.n_remove(pos), "n_keep": pos - fp.n_remove(pos)},
                       "0": {"n": neg, "n_remove": fp.n_remove(neg), "n_keep": neg - fp.n_remove(neg)}},
           "subsets": sub, "train_files": trn, "files": dict(sorted(files.items()))}
    fp.put(out / fp.MANIFEST, fp.json_bytes(man))
    return man


def layout(base: Path) -> SimpleNamespace:
    return SimpleNamespace(base=base, prune=base / "ftprune", manifest=base / "ftprune" / "prune_manifest.json",
                           ftgrid=base / "ftgrid", eval=base / "ftgrid" / "eval.jsonl",
                           split=base / "ftgrid" / "ftgrid_split.json", train=base / "train.jsonl", out=base / "compose")


def make_world(base: Path, examples: list, removed: dict, eval_rows: list, T: float = 1000.0) -> SimpleNamespace:
    """A manifest tree, Gate-FT's train.jsonl, ftgrid_data's eval.jsonl and an ftgrid_split.json holding the facts the module
    cross-checks (the TEST count and P0's seen share come from the brute force below)."""
    w = layout(base)
    rows = rows_from(examples)
    train_bytes = jsonl_bytes(rows)
    w.train.parent.mkdir(parents=True, exist_ok=True)
    w.train.write_bytes(train_bytes)
    man = write_prune_tree(w.prune, rows, removed, T, sha1(train_bytes))
    w.ftgrid.mkdir(parents=True, exist_ok=True)
    ev_bytes = jsonl_bytes(eval_rows)
    w.eval.write_bytes(ev_bytes)
    all_keys = {f"{r['user_id']}::{i}" for r in rows for i in r["candidate_item_ids"]}
    p0 = brute(rows, all_keys, eval_rows, T)
    write_json(w.split, {"T": T, "eval": {"users": len(eval_rows),
                                          "candidates": sum(len(r["candidate_item_ids"]) for r in eval_rows),
                                          "test_candidates": p0["n_test"], "share_test_pairs_item_seen_in_train": float(p0[C])},
                         "files": {"eval.jsonl": sha1(ev_bytes), "train.jsonl": sha1(train_bytes)}})
    w.rows, w.eval_rows, w.T, w.man, w.removed = rows, eval_rows, T, man, removed
    return w


def cli_args(w, out=None, **over) -> list:
    """The command line of a world; `over` replaces or adds an option (None drops it)."""
    opts = {"manifest": w.manifest, "out_dir": out or w.out, "eval_panel": w.eval, "ftgrid_split": w.split}
    opts.update(over)
    return [x for k, v in opts.items() if v is not None for x in (f"--{k}", str(v))]


def compose_world(w, **kw) -> dict:
    return fc.compose(w.manifest, w.eval, w.split, **kw)


# ---------------------------------------------------------------- the brute force: a different formulation
def brute(rows: list, kept: set, eval_rows: list, T: float) -> dict:
    """The statistics of one arm straight from the definitions, per-item rationals and an exact integer tercile rule: the item
    is in the head iff 6 * #{items with a smaller count} + 3 * #{items with an equal count (itself included)} >= 4 K, which is
    stats.rank_bins(n, 3) == 2 on average ranks (r = lo + (g + 1) / 2 and 3 (r - 1/2) / K >= 2)."""
    ex = [(f"{r['user_id']}::{i}", str(i), int(y)) for r in rows for i, y in zip(r["candidate_item_ids"],
                                                                                 r["candidate_labels"])]
    items = sorted({i for _, i, _ in ex})
    k = len(items)
    n = {i: 0 for i in items}
    pos, m, p = dict(n), dict(n), dict(n)
    for key, i, y in ex:
        n[i] += 1
        pos[i] += y
        if key in kept:
            m[i] += 1
            p[i] += y
    num, den = Fraction(0), 0
    for i in items:
        if n[i] >= 2 and m[i] >= 1:
            num += n[i] * abs(Fraction(p[i], m[i]) - Fraction(pos[i], n[i]))
            den += n[i]
    test = [str(i) for r in eval_rows for i, t in zip(r["candidate_item_ids"], r["candidate_timestamps"]) if float(t) >= T]
    head = {i for i in items
            if 6 * sum(1 for j in items if n[j] < n[i]) + 3 * sum(1 for j in items if n[j] == n[i]) >= 4 * k}
    kept_n = sum(m.values())
    seen = sum(1 for i in test if m.get(i, 0) >= 1)
    return {A: Fraction(sum(1 for i in items if m[i] == 0), k), B: num / den if den else None,
            C: Fraction(seen, len(test)) if test else None,
            D: Fraction(sum(m[i] for i in head), kept_n) if kept_n else None,
            "n_kept": kept_n, "items_without_example": sum(1 for i in items if m[i] == 0),
            "b_items_in_scope_lost": sum(1 for i in items if n[i] >= 2 and m[i] == 0),
            "b_weight_in_scope_lost": sum(n[i] for i in items if n[i] >= 2 and m[i] == 0),
            "c_test_pairs_seen": seen, "d_head_examples": sum(m[i] for i in head), "n_test": len(test),
            "head": head, "n_head": len(head)}


def kept_keys(rows: list, removed: dict, arm: str, seed: int) -> set:
    return {f"{r['user_id']}::{i}" for r in rows for i in r["candidate_item_ids"]} - set(removed.get((arm, seed), ()))


def fl(x):
    return None if x is None else float(x)


def assert_matches_brute(res: dict, rows: list, removed: dict, eval_rows: list, T: float) -> None:
    """Every (arm, seed) value, every count and every arm mean of the result against the brute force (exact floats: both are
    one correctly rounded conversion of the same rational)."""
    for arm in fp.ARMS:
        refs = []
        for s in fp.SEEDS:
            ref = brute(rows, kept_keys(rows, removed, arm, s), eval_rows, T)
            refs.append(ref)
            got = res["arms"][arm]["per_seed"][f"s{s}"]
            for k in STATS:
                assert got[k] == fl(ref[k]), (arm, s, k, got[k], ref[k])
            for k in ("n_kept", "items_without_example", "b_items_in_scope_lost", "b_weight_in_scope_lost",
                      "c_test_pairs_seen", "d_head_examples"):
                assert got[k] == ref[k], (arm, s, k)
        for k in STATS:
            vals = [r[k] for r in refs]
            want = None if any(v is None for v in vals) else float(sum(vals, Fraction(0)) / len(vals))
            assert res["arms"][arm]["mean"][k] == want, (arm, k)


# ---------------------------------------------------------------- the worked world H (numbers derived by hand)
# items: A 4 examples (+ + - -), B 3 (+ + +), C 2 (+ -), D 1 (+), E 1 (-), F 3 (- - +): K = 6, 8 positives, 6 negatives, so every
# pruned arm removes floor(8/4 + 1/2) = 2 positives and floor(6/4 + 1/2) = 2 negatives.
H_EXAMPLES = [("a1", "A", 1), ("a2", "A", 1), ("a3", "A", 0), ("a4", "A", 0), ("b1", "B", 1), ("b2", "B", 1), ("b3", "B", 1),
              ("c1", "C", 1), ("c2", "C", 0), ("d1", "D", 1), ("e1", "E", 0), ("f1", "F", 0), ("f2", "F", 0), ("f3", "F", 1)]
H_P2 = {"a1::A", "c1::C", "c2::C", "e1::E"}          # +: a1 c1; -: c2 e1 -> C and E keep nothing, A keeps + - -
H_P3 = {"b1::B", "b2::B", "c2::C", "e1::E"}          # +: b1 b2; -: c2 e1 -> B keeps b3 only, C keeps c1, E keeps nothing
H_P1 = [{"a1::A", "a2::A", "a3::A", "a4::A"}, {"b1::B", "d1::D", "e1::E", "f1::F"},
        {"c1::C", "f3::F", "a3::A", "c2::C"}, {"a1::A", "b3::B", "f1::F", "f2::F"}, {"d1::D", "c1::C", "e1::E", "a4::A"}]
H_REMOVED = {**{("P1", s): H_P1[s] for s in fp.SEEDS}, **{("P2", s): H_P2 for s in fp.SEEDS},
             **{("P3", s): H_P3 for s in fp.SEEDS}}
H_T = 1000.0
# EVAL rows: TEST pairs (ts >= T): A (ts == T), E, G (not a TRAIN item), D, D; CAL pairs (ts < T) C and B must not count
H_EVAL = [{"user_id": "x1", "candidate_item_ids": ["A", "C", "E", "G"], "candidate_timestamps": [1000, 999, 1001, 1500],
           "candidate_labels": [1, 0, 1, 0]},
          {"user_id": "x2", "candidate_item_ids": ["D", "B"], "candidate_timestamps": [2000, 10], "candidate_labels": [1, 0]},
          {"user_id": "x3", "candidate_item_ids": ["D"], "candidate_timestamps": [1200], "candidate_labels": [0]}]


@pytest.fixture(scope="module")
def hand(tmp_path_factory):
    w = make_world(tmp_path_factory.mktemp("hand"), H_EXAMPLES, H_REMOVED, H_EVAL, H_T)
    w.res = compose_world(w)
    return w


def test_worked_world_P0_P2_P3_equal_the_numbers_derived_by_hand(hand):
    arms = hand.res["arms"]
    F = Fraction
    # P0: nothing removed. a = 0, b = 0; TEST items A E G D D with A, E, D seen: 4/5; head {A, B, F} holds 4 + 3 + 3 of 14 examples
    assert [arms["P0"]["mean"][k] for k in STATS] == [0.0, 0.0, float(F(4, 5)), float(F(5, 7))]
    # P2: C and E lose everything (a = 2/6); A keeps + - - (rate 1/3 against 1/2: |change| 1/6, weight 4), B and F keep their
    # rates; C (n = 2, in scope) is lost and left out: b = (4 / 6) / (4 + 3 + 3) = 1/15; TEST A D D seen: 3/5; kept 10 examples,
    # 9 of them (A 3, B 3, F 3) in the head: 9/10
    assert [arms["P2"]["mean"][k] for k in STATS] == [float(F(1, 3)), float(F(1, 15)), float(F(3, 5)), float(F(9, 10))]
    p2 = arms["P2"]["per_seed"]["s0"]
    assert (p2["items_without_example"], p2["b_items_in_scope_lost"], p2["b_weight_in_scope_lost"]) == (2, 1, 2)
    assert (p2["n_kept"], p2["n_kept_positives"], p2["n_kept_negatives"], p2["c_test_pairs_seen"],
            p2["d_head_examples"]) == (10, 6, 4, 3, 9)
    # P3: only E is lost (a = 1/6); B keeps b3 (rate 1 = 1), C keeps c1 (rate 1 against 1/2: change 1/2, weight 2):
    # b = (2 * 1/2) / (4 + 3 + 2 + 3) = 1/12; TEST A D D seen: 3/5. The head is {A, B, F} of the FULL TRAIN counts: the 10 kept
    # examples hold A 4 + B 1 + F 3 = 8 in it: 4/5 (counts recomputed on the kept examples would make B a tail item: 7/10)
    assert [arms["P3"]["mean"][k] for k in STATS] == [float(F(1, 6)), float(F(1, 12)), float(F(3, 5)), float(F(4, 5))]
    assert arms["P3"]["per_seed"]["s4"]["d_head_examples"] == 8 and arms["P3"]["per_seed"]["s4"]["n_kept"] == 10


def test_worked_world_P1_draws_and_their_seed_mean(hand):
    p1 = hand.res["arms"]["P1"]
    s0 = p1["per_seed"]["s0"]          # A loses all four examples: a = 1/6, b = 0 over B, C, F, TEST E D D seen, head B 3 + F 3
    assert [s0[k] for k in STATS] == [float(Fraction(1, 6)), 0.0, float(Fraction(3, 5)), float(Fraction(3, 5))]
    assert (s0["items_without_example"], s0["b_items_in_scope_lost"], s0["b_weight_in_scope_lost"]) == (1, 1, 4)
    s1 = p1["per_seed"]["s1"]          # D (a singleton) and E are lost, B keeps b2 b3, F keeps - +: a = 1/3, D is not in scope
    assert s1[A] == float(Fraction(1, 3)) and (s1["items_without_example"], s1["b_items_in_scope_lost"]) == (2, 0)
    assert len({tuple(p1["per_seed"][f"s{s}"][k] for k in STATS) for s in fp.SEEDS}) == 5      # five different draws
    # a: (1/6 + 1/3 + 1/6 + 0 + 1/3) / 5 = 1/5, the exact mean (a mean of the rounded floats is 0.19999999999999998)
    assert p1["mean"][A] == float(Fraction(1, 5)) and p1["mean"][A] == 0.2
    for k in STATS:                                   # (the exact means against the brute force: assert_matches_brute)
        vals = [p1["per_seed"][f"s{s}"][k] for s in fp.SEEDS]
        assert p1["mean"][k] == pytest.approx(sum(vals) / 5, abs=1e-15)
        assert p1["sd_seed"][k] == pytest.approx(statistics.stdev(vals), rel=1e-9)
    assert not p1["subsets_identical_across_seeds"] and not p1["statistics_identical_across_seeds"]


def test_worked_world_counts_the_scope_the_terciles_and_the_test_pairs(hand):
    res = hand.res
    assert res["train"] == {"examples": 14, "positives": 8, "negatives": 6, "items": 6, "items_singleton": 2,
                            "items_in_scope_b": 4, "examples_in_scope_b": 12}
    # TRAIN counts A 4, B 3, F 3, C 2, D 1, E 1: average ranks 6, 4.5, 4.5, 3, 1.5, 1.5; the tie group B, F lies exactly on the cut
    # (3 (4.5 - 1/2) / 6 = 2): a whole tie group is in the head, so the head is {A, B, F}
    assert res["terciles"]["n_items_head"] == 3 and res["terciles"]["n_items_middle"] == 1
    assert res["terciles"]["n_items_tail"] == 2 and res["terciles"]["head_min_popularity"] == 3
    assert list(fc.tercile_bins([4, 3, 2, 1, 1, 3])) == [2, 2, 1, 0, 0, 2]
    # 5 TEST pairs (ts >= T, the pair at ts == T included; the two CAL pairs left out), one on an item that is not in TRAIN
    assert res["test"] == {"T": 1000.0, "pairs": 5, "pairs_item_in_train": 4, "pairs_item_not_in_train": 1}
    assert res["inputs"]["eval_panel"]["rows"] == 3 and res["inputs"]["eval_panel"]["candidates"] == 7
    assert all(res["checks"][k] is True for k in res["checks"] if k not in ("train_file_equals_signals_rows",))
    assert_matches_brute(res, hand.rows, hand.removed, hand.eval_rows, hand.T)


# ---------------------------------------------------------------- the other designed worlds
def test_every_item_tied_in_popularity_has_an_empty_head_and_d_is_zero(tmp_path):
    ex = [(f"p{k}", f"X{k}", 1) for k in range(1, 5)] + [(f"q{k}", f"X{k}", 0) for k in range(1, 5)]
    rm = {**{("P2", s): {"p1::X1", "q2::X2"} for s in fp.SEEDS}, **{("P3", s): {"p3::X3", "q4::X4"} for s in fp.SEEDS},
          **{("P1", s): {f"p{s % 4 + 1}::X{s % 4 + 1}", f"q{(s + 1) % 4 + 1}::X{(s + 1) % 4 + 1}"} for s in fp.SEEDS}}
    ev = [{"user_id": "e1", "candidate_item_ids": ["X1", "X2", "X5"], "candidate_timestamps": [5, 5, 5],
           "candidate_labels": [1, 0, 1]}]
    w = make_world(tmp_path, ex, rm, ev, T=5.0)
    res = compose_world(w)
    assert res["terciles"]["n_items_head"] == 0 and res["terciles"]["head_min_popularity"] is None
    assert res["terciles"]["n_items_middle"] == 4 and res["terciles"]["head_example_share_P0"] == 0.0
    for arm in fp.ARMS:
        assert res["arms"][arm]["mean"][D] == 0.0
    # P2: X1 keeps q1 (rate 0 against 1/2), X2 keeps p2 (1 against 1/2); no item is lost: b = (2 * 1/2 + 2 * 1/2) / 8 = 1/4
    assert res["arms"]["P2"]["mean"][A] == 0.0 and res["arms"]["P2"]["mean"][B] == 0.25
    assert res["arms"]["P2"]["mean"][C] == res["arms"]["P0"]["mean"][C] == pytest.approx(2 / 3)
    assert_matches_brute(res, w.rows, rm, ev, w.T)


def test_an_arm_identical_to_P0_has_P0s_statistics(tmp_path):
    ex = [("u1", "I1", 1), ("u2", "I1", 0)]              # one positive and one negative: n_remove(1) = 0, no arm removes anything
    ev = [{"user_id": "e1", "candidate_item_ids": ["I1", "Z"], "candidate_timestamps": [7, 8], "candidate_labels": [1, 0]}]
    w = make_world(tmp_path, ex, {}, ev, T=7.0)
    assert fp.n_remove(1) == 0
    res = compose_world(w)
    p0 = res["arms"]["P0"]
    assert [p0["mean"][k] for k in STATS] == [0.0, 0.0, 0.5, 0.0]
    for arm in ("P1", "P2", "P3"):
        a = res["arms"][arm]
        assert a["mean"] == p0["mean"] and a["sd_seed"] == p0["sd_seed"] and a["per_seed"] == p0["per_seed"]
        assert a["statistics_identical_across_seeds"] and a["subsets_identical_across_seeds"]
    # P1 not differing across seeds is recorded, not refused
    assert res["checks"]["P1_subsets_differ_across_seeds"] is False and res["checks"]["P2_P3_subsets_identical_across_seeds"]
    assert p0["sd_seed"] == {k: 0.0 for k in STATS}


def test_no_item_with_two_examples_leaves_b_undefined(tmp_path, capsys):
    ex = [(f"p{k}", f"S{k}", 1) for k in range(1, 5)] + [(f"q{k}", f"S{k + 4}", 0) for k in range(1, 5)]
    rm = {**{("P2", s): {"p1::S1", "q1::S5"} for s in fp.SEEDS}, **{("P3", s): {"p4::S4", "q4::S8"} for s in fp.SEEDS},
          **{("P1", s): {f"p{s % 4 + 1}::S{s % 4 + 1}", f"q{(s + 2) % 4 + 1}::S{(s + 2) % 4 + 5}"} for s in fp.SEEDS}}
    ev = [{"user_id": "e1", "candidate_item_ids": ["S1", "S5", "S2"], "candidate_timestamps": [9, 9, 9],
           "candidate_labels": [1, 0, 1]}]
    w = make_world(tmp_path, ex, rm, ev, T=9.0)
    res = compose_world(w)
    assert res["train"]["items_in_scope_b"] == 0 and res["train"]["items_singleton"] == 8
    for arm in fp.ARMS:
        assert res["arms"][arm]["mean"][B] is None and res["arms"][arm]["sd_seed"][B] is None
    assert res["arms"]["P2"]["mean"][A] == 0.25                    # two of eight singleton items lose their only example
    assert fc.main(cli_args(w)) == 0
    text = (w.out / fc.OUT_CSV).read_text(encoding="utf-8")
    row = next(r for r in csv.DictReader(io.StringIO(text)) if r["arm"] == "P2" and r["seed"] == "mean")
    assert row[B] == "" and row[A] == "0.25"                       # undefined = an empty CSV cell, null in the JSON
    assert read_json(w.out / fc.OUT_JSON)["arms"]["P2"]["mean"][B] is None
    assert "n/a" in capsys.readouterr().out
    assert_matches_brute(res, w.rows, rm, ev, w.T)


def test_arm_stats_on_the_extremes_and_the_exact_helpers():
    tt = fc.train_table({"key": ["u1::A", "u2::A", "u3::B"], "user_id": ["u1", "u2", "u3"], "item_id": ["A", "A", "B"],
                         "label": [1, 0, 1]})
    head = fc.tercile_bins(tt["n"]) == 2
    none = fc.arm_stats(tt, np.zeros(3, bool), np.array([1, 0]), 1, head)       # an arm that keeps nothing
    assert none[A] == 1 and none[B] is None and none[D] is None and none[C] == 0
    allk = fc.arm_stats(tt, np.ones(3, bool), np.array([1, 0]), 1, head)
    assert allk[A] == 0 and allk[B] == 0 and allk[C] == 1
    assert fc.exact_mean([]) is None and fc.exact_mean([Fraction(1, 3), None]) is None and fc.exact_sd([Fraction(1)]) is None
    vals = [Fraction(1, 10), Fraction(3, 10), Fraction(1, 7)]
    assert fc.exact_mean(vals) == sum(vals) / 3
    assert fc.exact_sd(vals) == pytest.approx(statistics.stdev([float(v) for v in vals]), rel=1e-12)
    assert fc.exact_sd([Fraction(1, 10)] * 5) == 0.0                              # exactly 0.0, not 1e-17


def test_the_head_is_cut_on_average_ranks_by_the_exact_integer_rule():
    """tercile_bins (stats.rank_bins on average ranks) = lo / g integer rule, on many tie patterns; equal counts share a bin."""
    rng = random.Random(11)
    for _ in range(300):
        k = rng.randint(1, 40)
        n = [rng.choice([1, 1, 2, 2, 3, 5, 8, 13]) if rng.random() < 0.7 else rng.randint(1, 400) for _ in range(k)]
        bins = fc.tercile_bins(n)
        for i in range(k):
            lo = sum(1 for j in n if j < n[i])
            g = sum(1 for j in n if j == n[i])
            want = 2 if 6 * lo + 3 * g >= 4 * k else (0 if 6 * lo + 3 * g < 2 * k else 1)
            assert bins[i] == want, (n, i)
        assert all(len({int(bins[i]) for i in range(k) if n[i] == v}) == 1 for v in set(n))
    assert len(fc.tercile_bins([])) == 0


# ---------------------------------------------------------------- random worlds against the brute force
def random_world(seed: int):
    rng = random.Random(seed)
    n_items = rng.randint(4, 14)
    weights = [1.0 / (r + 1) ** rng.choice([0.0, 1.0, 1.5]) for r in range(n_items)]      # flat to skewed: ties everywhere
    examples = []
    for u in range(rng.randint(10, 26)):
        its = []
        for it in rng.choices(range(n_items), weights, k=rng.randint(1, 5)):
            if it not in its:
                its.append(it)
        examples += [(f"u{u}", f"i{it}", int(rng.random() < 0.25 + 0.5 * ((it * 7919) % 5) / 4)) for it in its]
    keys_by_class = {c: [f"{u}::{i}" for u, i, y in examples if y == c] for c in (0, 1)}

    def draw():
        out = set()
        for c in (0, 1):
            out |= set(rng.sample(keys_by_class[c], fp.n_remove(len(keys_by_class[c]))))
        return out
    p2, p3 = draw(), draw()
    removed = {**{("P1", s): draw() for s in fp.SEEDS}, **{("P2", s): p2 for s in fp.SEEDS}, **{("P3", s): p3 for s in fp.SEEDS}}
    T = 100.0
    ev = []
    for e in range(rng.randint(4, 10)):
        its = rng.sample(range(n_items + 3), rng.randint(1, 4))          # item ids above n_items are not in TRAIN
        ts = [rng.choice([T - 5, T - 1, T, T, T + 1, T + 50]) for _ in its]
        if e == 0:
            ts[0] = T
        ev.append({"user_id": f"x{e}", "candidate_item_ids": [f"i{i}" for i in its], "candidate_timestamps": ts,
                   "candidate_labels": [rng.randint(0, 1) for _ in its]})
    return examples, removed, ev, T


@pytest.mark.parametrize("seed", range(14))
def test_random_worlds_match_the_brute_force(tmp_path, seed):
    examples, removed, ev, T = random_world(seed)
    w = make_world(tmp_path, examples, removed, ev, T)
    res = compose_world(w)
    assert_matches_brute(res, w.rows, removed, ev, T)
    assert res["checks"]["P2_P3_subsets_identical_across_seeds"] is True
    assert res["train"]["items"] == len({i for _, i, _ in examples})


@pytest.mark.parametrize("seed", (3, 8))
def test_relabelling_the_items_changes_no_statistic_so_no_id_order_breaks_a_tie(tmp_path, seed):
    """Popularity ties are settled by average ranks, not by item id or position: reversing the order of the item ids (and
    renaming them) gives the same numbers, the same tercile sizes and the same counts."""
    examples, removed, ev, T = random_world(seed)
    top = max(int(i[1:]) for _, i, _ in examples) + 5
    ren = {f"i{k}": f"z{top - k:04d}" for k in range(top + 1)}
    ex2 = [(u, ren[i], y) for u, i, y in examples]
    rm2 = {k: {f"{key.split('::')[0]}::{ren[key.split('::')[1]]}" for key in v} for k, v in removed.items()}
    ev2 = [dict(r, candidate_item_ids=[ren[i] for i in r["candidate_item_ids"]]) for r in ev]
    a = compose_world(make_world(tmp_path / "a", examples, removed, ev, T))
    b = compose_world(make_world(tmp_path / "b", ex2, rm2, ev2, T))
    assert a["arms"] == b["arms"] and a["terciles"] == b["terciles"] and a["train"] == b["train"] and a["test"] == b["test"]


# ---------------------------------------------------------------- the real prune pipeline and a real ftgrid_data run
@pytest.fixture(scope="module")
def real(tmp_path_factory):
    """tests/test_confrec_ftprune.py's synthetic ML-1M world through the REAL ftprune signals step, then the REAL ftgrid_data on
    the same panels (TRAIN = the 25 DEV rows, EVAL = the rest), copied into the registered output layout."""
    base = tmp_path_factory.mktemp("real")
    w = PT.build_world(base)
    out = base / "out"
    man = PT.build(w, out)
    panel_all = base / "panel_all.jsonl"
    panel_all.write_bytes((base / "panels" / "ml1m_dev_h20.jsonl").read_bytes()
                          + (base / "panels" / "ml1m_confirm_h20.jsonl").read_bytes())
    fd.main(["--domain", "ml1m", "--panel_all", str(panel_all), "--out_dir", str(base / "ftgrid"), "--variant", PT.VARIANT,
             "--n_train", "25", "--gateft_split", str(w["split"])])
    reg = base / "reg"
    shutil.copytree(out, reg / "outputs" / "confrec" / "ftprune")
    shutil.copytree(base / "ftgrid", reg / "outputs" / "confrec" / "ftgrid" / "panels" / "ml1m")
    (reg / "outputs" / "confrec" / "gateft").mkdir(parents=True)
    shutil.copy2(w["train"], reg / "outputs" / "confrec" / "gateft" / "train.jsonl")
    r = layout(reg / "outputs" / "confrec")
    r.manifest = reg / "outputs" / "confrec" / "ftprune" / "prune_manifest.json"
    r.eval = reg / "outputs" / "confrec" / "ftgrid" / "panels" / "ml1m" / "eval.jsonl"
    r.split = r.eval.with_name("ftgrid_split.json")
    r.train = reg / "outputs" / "confrec" / "gateft" / "train.jsonl"
    r.out = reg / "outputs" / "confrec" / "ftprune" / "compose"
    r.reg, r.world, r.man, r.prune = reg, w, man, reg / "outputs" / "confrec" / "ftprune"
    r.res = fc.compose(r.manifest, r.eval, r.split, train=r.train)
    return r


def test_real_prune_world_matches_the_brute_force_from_the_raw_files(real):
    """Independent of ftprune and of the module: train.jsonl, the subset files and eval.jsonl read as plain files."""
    rows, ev = jsonl(real.train), jsonl(real.eval)
    T = float(read_json(real.split)["T"])
    removed = {}
    for arm in fp.ARMS:
        for s in fp.SEEDS:
            kept = (real.prune / "subsets" / f"{arm}_s{s}.txt").read_text(encoding="utf-8").split("\n")
            removed[(arm, s)] = {f"{r['user_id']}::{i}" for r in rows for i in r["candidate_item_ids"]} - set(kept)
    assert_matches_brute(real.res, rows, removed, ev, T)
    n_items = len({i for r in rows for i in r["candidate_item_ids"]})
    counts = {}
    for r in rows:
        for i in r["candidate_item_ids"]:
            counts[i] = counts.get(i, 0) + 1
    assert real.res["train"]["items"] == n_items and real.res["train"]["items_singleton"] == sum(v == 1 for v in counts.values())
    # the world exercises the edge cases: tied counts, singleton items, and in-scope items that lose every example
    assert len(set(counts.values())) < n_items and real.res["train"]["items_singleton"] > 0
    assert max(real.res["arms"][a]["per_seed"][f"s{s}"]["b_items_in_scope_lost"] for a in ("P1", "P2", "P3")
               for s in fp.SEEDS) > 0


def test_real_world_P0_reproduces_ftgrid_datas_seen_share_and_test_count(real):
    split = read_json(real.split)
    assert real.res["arms"]["P0"]["mean"][C] == split["eval"]["share_test_pairs_item_seen_in_train"]
    assert real.res["test"]["pairs"] == split["eval"]["test_candidates"] and real.res["test"]["T"] == split["T"]
    assert real.res["inputs"]["train_jsonl"]["sha1"] == split["files"]["train.jsonl"] == real.man["inputs"]["train_jsonl"]["sha1"]
    assert real.res["inputs"]["train_jsonl"]["file_checked_here"] is True and real.res["checks"]["train_file_equals_signals_rows"]
    assert real.res["variant"] == PT.VARIANT == real.man["variant"]
    for arm in ("P1", "P2", "P3"):                 # the registered rule: every pruned arm keeps 75% of each class
        assert real.res["arms"][arm]["per_seed"]["s0"]["n_kept_positives"] == real.man["classes"]["1"]["n_keep"]
        assert real.res["arms"][arm]["per_seed"]["s0"]["n_kept_negatives"] == real.man["classes"]["0"]["n_keep"]


def test_the_exact_server_command_runs_in_the_registered_layout(real, monkeypatch, capsys):
    monkeypatch.chdir(real.reg)
    rc = fc.main(["--manifest", "outputs/confrec/ftprune/prune_manifest.json", "--out_dir", "outputs/confrec/ftprune/compose"])
    out = capsys.readouterr().out
    assert rc == 0 and "S6 composition diagnostics" in out and "wrote" in out
    got = (real.reg / "outputs" / "confrec" / "ftprune" / "compose" / fc.OUT_JSON).read_bytes()
    want = fp.json_bytes(fc.compose(real.manifest, real.eval, real.split))     # without --train: the same bytes except that flag
    assert json.loads(got)["arms"] == json.loads(want)["arms"]
    assert json.loads(got)["inputs"]["train_jsonl"]["file_checked_here"] is False
    assert sorted(p.name for p in (real.reg / "outputs" / "confrec" / "ftprune" / "compose").iterdir()) == [fc.OUT_CSV,
                                                                                                               fc.OUT_JSON]
    assert fc.DEFAULT_EVAL == "outputs/confrec/ftgrid/panels/ml1m/eval.jsonl"
    assert fc.DEFAULT_SPLIT == "outputs/confrec/ftgrid/panels/ml1m/ftgrid_split.json"
    with pytest.raises(SystemExit) as e:
        fc.main(["--manifest", "x"])                                        # --out_dir is required: argparse exits 2
    assert e.value.code == 2


# ---------------------------------------------------------------- outputs: schema, csv, strict JSON, no clock / host / path
def test_outputs_have_the_documented_schema_and_the_csv_matches_the_json(hand, tmp_path):
    out = tmp_path / "o"
    assert fc.main(cli_args(hand, out)) == 0
    text = (out / fc.OUT_JSON).read_text(encoding="utf-8")
    res = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    assert list(res) == ["format", "spec", "domain", "variant", "code_sha1", "inputs", "train", "test", "terciles", "arms",
                         "checks", "definitions"]
    assert res["format"] == fc.FORMAT == "ftprune_compose_v1" and res["domain"] == "ml1m"
    assert list(res["arms"]) == list(fp.ARMS)
    for a in res["arms"].values():
        assert list(a) == ["seeds", "mean", "sd_seed", "statistics_identical_across_seeds", "subsets_identical_across_seeds",
                           "per_seed"]
        assert list(a["mean"]) == list(STATS) and list(a["per_seed"]) == [f"s{s}" for s in fp.SEEDS]
        assert list(a["per_seed"]["s0"]) == list(STATS) + list(fc.COUNT_KEYS)
    assert (out / fc.OUT_JSON).read_bytes() == fp.json_bytes(res) and text.endswith("}\n")
    rows = list(csv.DictReader(io.StringIO((out / fc.OUT_CSV).read_text(encoding="utf-8"))))
    assert len(rows) == 4 * 6 and list(rows[0]) == list(fc.CSV_COLS)
    for r in rows:
        src = res["arms"][r["arm"]]["mean" if r["seed"] == "mean" else "per_seed"]
        src = src if r["seed"] == "mean" else src[f"s{r['seed']}"]
        assert [float(r[k]) for k in STATS] == [src[k] for k in STATS]
    assert [r["seed"] for r in rows if r["arm"] == "P1"] == ["mean", "0", "1", "2", "3", "4"]
    assert not (tmp_path / "o" / "ftprune_compose.json.tmp").exists()


def test_every_input_sha1_is_recorded_and_is_the_files(hand):
    inp = hand.res["inputs"]
    assert inp["manifest"]["sha1"] == sha1_file(hand.manifest) and inp["manifest"]["format"] == fp.MANIFEST_FORMAT
    assert inp["signals"] == {"file": fp.SIGNALS_FILE, "sha1": sha1_file(hand.prune / fp.SIGNALS_FILE), "rows": 14}
    assert len(inp["subsets"]) == 20 and inp["manifest"]["files_verified"] == 38 == len(hand.man["files"])
    assert len(inp["pruned_train_files"]) == 17
    for tag, rec in inp["pruned_train_files"].items():
        assert rec["file"] == f"train/{tag}.jsonl" and rec["sha1"] == sha1_file(hand.prune / rec["file"])
    assert "P0_s0" not in inp["pruned_train_files"] and "P0_s3" in inp["pruned_train_files"]      # P0 s0-s2: Gate-FT's adapters
    for tag, rec in inp["subsets"].items():
        assert rec["file"] == f"subsets/{tag}.txt" and rec["sha1"] == sha1_file(hand.prune / rec["file"])
    assert inp["eval_panel"]["sha1"] == sha1_file(hand.eval) and inp["ftgrid_split"]["sha1"] == sha1_file(hand.split)
    assert inp["train_jsonl"]["sha1"] == sha1_file(hand.train) == hand.man["inputs"]["train_jsonl"]["sha1"]
    assert inp["gateft_split"]["T"] == 1000.0 == inp["ftgrid_split"]["T"]
    assert hand.res["code_sha1"] == {"ftprune_compose.py": sha1_file(MODULE), "ftprune.py": sha1_file(BOUND[0])}


def json_keys(o) -> list:
    """Every key at every depth of a parsed JSON value."""
    if isinstance(o, dict):
        return [k for key, v in o.items() for k in [key] + json_keys(v)]
    if isinstance(o, list):
        return [k for v in o for k in json_keys(v)]
    return []


def test_output_is_strict_json_without_clock_host_or_path(hand, real):
    for w, base in ((hand, hand.base), (real, real.base)):
        text = fp.json_bytes(w.res).decode("utf-8")
        for s in (str(base), base.as_posix(), "\\", str(Path.home()), str(ROOT)):
            assert s not in text, s
        json.loads(text, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
        assert "NaN" not in text and "Infinity" not in text
        keys = json_keys(json.loads(text))
        assert not [k for k in keys if re.search(r"(^|_)(time|timestamp|date|mtime|host|hostname|path|dir|cwd)(_|$)", k, re.I)]
    assert "T" in hand.res["test"] and hand.res["spec"].startswith("idea-stage/PREREG_AMENDMENT_3_ADDENDUM_7.md section 2")
    assert len(hand.res["definitions"]) == 6 and hand.res["definitions"][2].startswith("b_mean_abs_item_rate_change")


# ---------------------------------------------------------------- refusals: a recorded sha1 that does not match, and more
def copy_world(w, dest: Path) -> SimpleNamespace:
    shutil.copytree(w.base, dest)
    c = layout(dest)
    c.rows, c.eval_rows, c.T = w.rows, w.eval_rows, w.T
    return c


def rewrite_subset(c, tag: str, text: str) -> None:
    """Change a subset file and keep the manifest's sha1 records consistent with it (so only the content is wrong)."""
    rel = f"subsets/{tag}.txt"
    (c.prune / rel).write_bytes(text.encode("utf-8"))
    new = sha1(text.encode("utf-8"))

    def fix(m):
        m["files"][rel] = new
        m["subsets"][tag]["sha1"] = new
    edit_json(c.manifest, fix)


def keys_of(c, tag: str) -> list:
    return (c.prune / "subsets" / f"{tag}.txt").read_text(encoding="utf-8").split("\n")


def set_eval(c, rows: list) -> None:
    """Replace the EVAL panel and keep the split's sha1 record consistent with it."""
    c.eval.write_bytes(jsonl_bytes(rows))
    edit_json(c.split, lambda s: s["files"].__setitem__("eval.jsonl", sha1_file(c.eval)))


def append(path: Path, data: bytes) -> None:
    path.write_bytes(path.read_bytes() + data)


def put_text(path: Path, text: str) -> None:
    Path(path).write_bytes(text.encode("utf-8"))


def drop_first_key(c, tag: str) -> None:
    """A valid but different subset file: the sha1 recorded in the manifest no longer matches."""
    put_text(c.prune / "subsets" / f"{tag}.txt", "\n".join(keys_of(c, tag)[1:]))


def list_extra_file(c) -> None:
    edit_json(c.manifest, lambda m: m["files"].__setitem__("../outside.txt", "0" * 40))


SHA1_CASES = [
    ("signals", lambda c: append(c.prune / fp.SIGNALS_FILE, b"x"), r"signals/ml1m_signals\.csv\.gz: sha1"),
    ("subset_P0", lambda c: drop_first_key(c, "P0_s0"), r"subsets/P0_s0\.txt: sha1"),
    ("subset_P1", lambda c: drop_first_key(c, "P1_s3"), r"subsets/P1_s3\.txt: sha1"),
    ("subset_P2", lambda c: drop_first_key(c, "P2_s4"), r"subsets/P2_s4\.txt: sha1"),
    ("subset_P3", lambda c: drop_first_key(c, "P3_s0"), r"subsets/P3_s0\.txt: sha1"),
    ("pruned_train_file", lambda c: append(c.prune / "train" / "P2_s1.jsonl", b"\n"), r"train/P2_s1\.jsonl: sha1"),
    ("missing_subset", lambda c: (c.prune / "subsets" / "P3_s2.txt").unlink(), r"subsets/P3_s2\.txt: missing"),
    ("manifest_record", lambda c: edit_json(c.manifest, lambda m: m["files"].__setitem__("subsets/P1_s0.txt", "0" * 40)),
     r"subsets/P1_s0\.txt: sha1"),
    ("unexpected_listed_file", list_extra_file, r"unexpectedly"),
    ("eval_panel", lambda c: append(c.eval, b"\n"), r"eval\.jsonl: sha1 .* recorded eval\.jsonl sha1"),
    ("wrong_format", lambda c: edit_json(c.manifest, lambda m: m.__setitem__("format", "ftprune_manifest_v0")), r"format"),
    ("wrong_domain", lambda c: edit_json(c.manifest, lambda m: m.__setitem__("domain", "toys")),
     r"domain 'toys' is not 'ml1m' \(S6 is ML-1M only\)"),
]


@pytest.mark.parametrize("name, mutate, pattern", SHA1_CASES, ids=[c[0] for c in SHA1_CASES])
def test_a_recorded_sha1_that_does_not_match_is_refused_with_exit_2_and_nothing_written(hand, tmp_path, capsys, name, mutate,
                                                                                         pattern):
    c = copy_world(hand, tmp_path / "w")
    mutate(c)
    with pytest.raises(fc.Refusal, match=pattern):
        compose_world(c)
    assert fc.main(cli_args(c)) == 2
    err = capsys.readouterr().err
    assert "REFUSED:" in err and "nothing was written" in err
    assert not c.out.exists()


def test_the_train_file_is_checked_against_the_recorded_sha1_and_the_signals_rows(hand, tmp_path, capsys):
    c = copy_world(hand, tmp_path / "w")
    res = compose_world(c, train=c.train)
    assert res["inputs"]["train_jsonl"]["file_checked_here"] is True and res["checks"]["train_file_equals_signals_rows"] is True
    assert fp.json_bytes({k: v for k, v in res.items() if k not in ("inputs", "checks")}) == fp.json_bytes(
        {k: v for k, v in compose_world(c).items() if k not in ("inputs", "checks")})
    append(c.train, b"\n")
    with pytest.raises(fc.Refusal, match=r"--train train\.jsonl: sha1 .* recorded train\.jsonl sha1"):
        compose_world(c, train=c.train)
    assert fc.main(cli_args(c, train=c.train)) == 2 and not c.out.exists()
    capsys.readouterr()
    # the right sha1 but other rows: the manifest is made to record the changed file's sha1, the signals file stays
    c2 = copy_world(hand, tmp_path / "w2")
    rows = jsonl(c2.train)
    rows[0]["candidate_labels"] = [1 - rows[0]["candidate_labels"][0]]
    c2.train.write_bytes(jsonl_bytes(rows))
    edit_json(c2.manifest, lambda m: m["inputs"]["train_jsonl"].__setitem__("sha1", sha1_file(c2.train)))
    edit_json(c2.split, lambda s: s["files"].__setitem__("train.jsonl", sha1_file(c2.train)))
    with pytest.raises(fc.Refusal, match="not the signals file's rows"):
        compose_world(c2, train=c2.train)
    assert compose_world(c2)["checks"]["train_file_equals_signals_rows"] is None        # without --train nothing re-reads it


def with_extra_key(c) -> None:
    rewrite_subset(c, "P2_s0", "\n".join(keys_of(c, "P2_s0") + ["ghost::Z"]))


def with_a_key_twice(c) -> None:
    keys = keys_of(c, "P1_s1")
    rewrite_subset(c, "P1_s1", "\n".join(keys + keys[:1]))


def one_kept_example_fewer(c) -> None:
    """The file and its sha1 records agree, but the manifest's recorded kept counts no longer do."""
    rewrite_subset(c, "P3_s2", "\n".join(keys_of(c, "P3_s2")[:-1]))


def not_the_registered_class_counts(c) -> None:
    """The subset file, its sha1 records and its recorded counts all agree (7 + 4 kept), but the manifest's `classes` says
    that a pruned arm keeps 6 positives and 4 negatives."""
    rewrite_subset(c, "P2_s0", "\n".join(keys_of(c, "P2_s0") + ["a1::A"]))

    def fix(m):
        m["subsets"]["P2_s0"]["n_keep"] = {"1": 7, "0": 4}
        m["subsets"]["P2_s0"]["n_removed"] = {"1": 1, "0": 2}
    edit_json(c.manifest, fix)


def no_test_pair_at_all(c) -> None:
    edit_json(c.split, lambda s: s.__setitem__("T", 10 ** 9))
    edit_json(c.manifest, lambda m: m["inputs"]["gateft_split"].__setitem__("T", 10 ** 9))


CONSISTENT_CASES = [
    ("key_not_in_train", with_extra_key, r"P2_s0\.txt: 1 keys are not TRAIN examples"),
    ("key_twice", with_a_key_twice, r"P1_s1\.txt: a key twice"),
    ("kept_count_differs_from_the_record", one_kept_example_fewer, r"P3_s2: n_keep .* differs from the manifest's"),
    ("not_the_registered_class_counts", not_the_registered_class_counts,
     r"P2_s0: keeps .* per class, the manifest's classes record says"),
    ("signals_count_vs_manifest", lambda c: edit_json(c.manifest, lambda m: m["inputs"]["train_jsonl"].__setitem__("examples", 15)),
     r"the signals file holds 14 examples, the manifest's inputs\.train_jsonl records 15"),
    ("classes_record", lambda c: edit_json(c.manifest, lambda m: m["classes"]["1"].__setitem__("n", 9)), r"classes\.1\.n is 9"),
    ("T_differs_from_gateft", lambda c: edit_json(c.split, lambda s: s.__setitem__("T", 1000.5)), r"not the Gate-FT T"),
    ("manifest_without_gateft_T", lambda c: edit_json(c.manifest, lambda m: m["inputs"].__setitem__("gateft_split", None)),
     r"records no Gate-FT T"),
    ("ftgrid_train_sha1", lambda c: edit_json(c.split, lambda s: s["files"].__setitem__("train.jsonl", "1" * 40)),
     r"ftgrid split records train\.jsonl sha1"),
    ("test_count", lambda c: edit_json(c.split, lambda s: s["eval"].__setitem__("test_candidates", 6)),
     r"5 TEST pairs found, the ftgrid split records 6"),
    ("p0_seen_share", lambda c: edit_json(c.split, lambda s: s["eval"].__setitem__("share_test_pairs_item_seen_in_train", 0.75)),
     r"P0's seen share 0\.8 is not the ftgrid split's recorded"),
    ("split_without_fields", lambda c: edit_json(c.split, lambda s: s.pop("eval")), r"not ftgrid_data's output"),
    ("split_non_finite_T", lambda c: edit_json(c.split, lambda s: s.__setitem__("T", float("inf"))), r"non-finite T"),
    ("no_test_pair", no_test_pair_at_all, r"no TEST pair"),
    ("malformed_eval_row", lambda c: set_eval(c, [{"user_id": "x", "candidate_item_ids": ["A", "B"],
                                                   "candidate_timestamps": [1]}]), r"not an EVAL panel row"),
    ("non_finite_timestamp", lambda c: set_eval(c, [{"user_id": "x", "candidate_item_ids": ["A"],
                                                    "candidate_timestamps": [float("inf")]}]), r"non-finite timestamp"),
]


@pytest.mark.parametrize("name, mutate, pattern", CONSISTENT_CASES, ids=[c[0] for c in CONSISTENT_CASES])
def test_inputs_that_agree_in_sha1_but_are_not_the_registered_ones_are_refused(hand, tmp_path, name, mutate, pattern):
    c = copy_world(hand, tmp_path / "w")
    mutate(c)
    with pytest.raises(fc.Refusal, match=pattern):
        compose_world(c)
    assert fc.main(cli_args(c)) == 2 and not c.out.exists()


def test_missing_inputs_are_refused_not_crashed(hand, tmp_path, capsys):
    c = copy_world(hand, tmp_path / "w")
    nope = tmp_path / "nope.json"
    for over, pattern in ((dict(manifest=nope), "manifest nope.json: no such file"),
                          (dict(eval_panel=nope), "--eval_panel nope.json: no such file"),
                          (dict(ftgrid_split=nope), "ftgrid_split.json nope.json: no such file"),
                          (dict(train=nope), "--train nope.json: no such file")):
        assert fc.main(cli_args(c, **over)) == 2, over
        assert pattern in capsys.readouterr().err and not c.out.exists()
    (c.prune / "prune_manifest.json").write_text("{not json", encoding="utf-8")
    assert fc.main(cli_args(c)) == 2 and "not readable JSON" in capsys.readouterr().err


# ---------------------------------------------------------------- determinism
def run_cli(args, cwd=ROOT, hashseed="0", module_args=None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(PYTHONHASHSEED=hashseed, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               PYTHONPATH=str(ROOT))
    return subprocess.run([sys.executable, "-m", "src.confrec.ftprune_compose", *map(str, args)], cwd=cwd, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)


def test_reruns_are_byte_identical_in_place_fresh_elsewhere_and_across_hash_seeds(real, tmp_path):
    first = tmp_path / "first"
    assert fc.main(cli_args(real, first)) == 0
    want = {n: (first / n).read_bytes() for n in (fc.OUT_JSON, fc.OUT_CSV)}
    mt = {n: (first / n).stat().st_mtime_ns for n in want}
    assert fc.main(cli_args(real, first)) == 0                       # in place: not a byte, not an mtime changes
    assert {n: (first / n).read_bytes() for n in want} == want and {n: (first / n).stat().st_mtime_ns for n in want} == mt
    fresh = tmp_path / "fresh"
    assert fc.main(cli_args(real, fresh)) == 0                       # a fresh directory
    assert {n: (fresh / n).read_bytes() for n in want} == want
    # the whole world copied to another (deeper) path: no path may reach the output
    moved = tmp_path / "some" / "other" / "place"
    shutil.copytree(real.reg, moved)
    out2 = tmp_path / "out2"
    assert fc.main(["--manifest", str(moved / "outputs/confrec/ftprune/prune_manifest.json"), "--out_dir", str(out2),
                    "--eval_panel", str(moved / "outputs/confrec/ftgrid/panels/ml1m/eval.jsonl"),
                    "--ftgrid_split", str(moved / "outputs/confrec/ftgrid/panels/ml1m/ftgrid_split.json")]) == 0
    assert {n: (out2 / n).read_bytes() for n in want} == want
    # processes with other PYTHONHASHSEEDs (set / dict order of strings differs between them), the real command line
    for seed, cwd in (("1", tmp_path), ("4242", ROOT)):
        out = tmp_path / f"proc{seed}"
        r = run_cli(cli_args(real, out), cwd=cwd, hashseed=seed)
        assert r.returncode == 0, r.stderr[-2000:]
        assert {n: (out / n).read_bytes() for n in want} == want, seed


# ---------------------------------------------------------------- read-only import of ftprune and writes only to --out_dir
def source_tree():
    return ast.parse(MODULE.read_text(encoding="utf-8"))


def test_the_source_imports_ftprune_read_only_and_writes_only_through_two_put_calls_on_out_dir():
    tree = source_tree()
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert not mods & {"os", "shutil", "subprocess", "tempfile", "socket", "urllib", "requests", "http", "pickle", "sqlite3",
                       "importlib", "builtins"}
    imports = [(n.module, a.name, a.asname) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names]
    assert ("src.confrec", "ftprune", "fp") in imports and sum(1 for i in imports if i[1] == "ftprune") == 1

    def root(n):
        while isinstance(n, (ast.Attribute, ast.Subscript)):
            n = n.value
        return n.id if isinstance(n, ast.Name) else None
    for node in ast.walk(tree):                      # nothing of ftprune is ever assigned, deleted, mutated or monkeypatched
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = node.targets
        for t in targets:
            for sub in ast.walk(t):
                assert not (isinstance(sub, (ast.Attribute, ast.Subscript)) and root(sub) == "fp"), ast.dump(sub)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("setattr", "delattr", "exec", "eval"):
            raise AssertionError(f"{node.func.id} call")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and root(node.func) == "fp":
            assert node.func.attr not in ("append", "extend", "update", "pop", "clear", "add", "remove", "insert", "setdefault",
                                          "__setitem__", "__delitem__"), node.func.attr
    banned = {"write_text", "write_bytes", "unlink", "rmdir", "mkdir", "touch", "rename", "symlink_to", "hardlink_to", "chmod",
              "truncate", "write", "writelines"}
    assert not [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in banned]
    for node in ast.walk(tree):                      # open() only to read bytes
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            assert len(node.args) == 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value == "rb"
    puts = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "put"
            and root(n.func) == "fp"]
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    in_main = [n for n in ast.walk(main) if n in puts]
    assert len(puts) == 2 == len(in_main)
    for call, name in zip(sorted(puts, key=lambda c: c.lineno), ("OUT_JSON", "OUT_CSV")):
        first = call.args[0]
        assert isinstance(first, ast.BinOp) and isinstance(first.op, ast.Div) and first.left.id == "out" \
            and first.right.id == name


def prune_state() -> dict:
    return {k: (id(v), repr(v)) for k, v in vars(fp).items()
            if isinstance(v, (tuple, list, dict, set, str, int, float, bool, type(None)))}


def file_state(paths) -> dict:
    return {str(p): (sha1_file(p), p.stat().st_mtime_ns) for p in paths}


def tree_state(root: Path, skip=()) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and not any(s in p.parts for s in skip):
            st = p.stat()
            out[p.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
    return out


def test_running_changes_neither_ftprunes_namespace_nor_any_bound_file(real, tmp_path):
    before_ns, before_files = prune_state(), file_state(BOUND)
    mod_before = sys.modules["src.confrec.ftprune"]
    assert fc.fp is mod_before
    assert fc.main(cli_args(real, tmp_path / "o")) == 0
    assert sys.modules["src.confrec.ftprune"] is mod_before and prune_state() == before_ns
    assert file_state(BOUND) == before_files


def test_a_run_writes_nowhere_but_out_dir_and_imports_no_torch(real, tmp_path):
    probe = tmp_path / "cwd_probe"
    probe.mkdir()
    out = tmp_path / "o"
    watched = list(BOUND) + [MODULE, Path(__file__)]       # fixed files, so a concurrent edit elsewhere cannot flake this
    repo = file_state(watched)
    world = tree_state(real.base, skip=("compose",))
    r = run_cli(cli_args(real, out), cwd=probe, hashseed="7")
    assert r.returncode == 0, r.stderr[-2000:]
    assert sorted(p.name for p in out.iterdir()) == [fc.OUT_CSV, fc.OUT_JSON]       # and no .tmp left behind
    assert list(probe.iterdir()) == []                                              # nothing relative to the cwd
    assert tree_state(real.base, skip=("compose",)) == world                        # the world (inputs) untouched
    assert file_state(watched) == repo                                              # nor ftprune, its script, this code
    code = ("import sys; import src.confrec.ftprune_compose; "
            "bad = [m for m in ('torch', 'transformers', 'vllm', 'peft') if m in sys.modules]; print(bad); "
            "sys.exit(1 if bad else 0)")
    imp = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=600,
                         env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT)})
    assert imp.returncode == 0, imp.stdout + imp.stderr[-1000:]


def test_a_refusal_in_a_real_process_exits_2_and_writes_nothing(hand, tmp_path):
    c = copy_world(hand, tmp_path / "w")
    append(c.prune / "subsets" / "P2_s0.txt", b"x")
    out = tmp_path / "o"
    r = run_cli(cli_args(c, out), cwd=tmp_path)
    assert r.returncode == 2 and "REFUSED:" in r.stderr and "subsets/P2_s0.txt: sha1" in r.stderr
    assert not out.exists() and r.stdout == ""
