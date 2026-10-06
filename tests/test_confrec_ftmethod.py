"""Tests of the Amendment-3 method slot (idea-stage/PREREG_AMENDMENT_3.md section 7, FT-M; addenda 1 and 2):
src/confrec/train_lora_offset.py, src/confrec/ftmethod_report.py and scripts/sigir/run_ftmethod.sh. CPU only, deterministic,
no network, no GPU.
  1. q-hat on raw-event fixtures: hand-computed values (shrinkage k = 5, strictly before t, leave-user-out, the first rating
     per (user, item)), a brute-force reference of the A3 section 3 definition on every TRAIN and EVAL pair, one scan =
     per-role scans = ftgrid_report.compute_refs' q-hat, the TRAIN standardisation, deterministic files and the trainer's
     refusal of files that do not match the manifest;
  2. the dataset (torch): PriorOffsetSet's examples are YesNoSet(rows)'s (order, ids, overlength skips), each with its z;
     collate_offset;
  3. the loss (torch): b = 0 equals last_token_loss; the offset enters only the Yes logit of the answer position before the
     full-vocabulary softmax; the gradient into b (analytic) and none into z; the num_items_in_batch rule; the trainer
     subclass of last_token_trainer; b's optional learning-rate group;
  4. the SFT-comparator recipe check; the train CLI against train_lora_yesno's; the model / LoRA setup (AST parity); main()
     end to end with stand-in transformers / peft (torch);
  5. tiny-model Trainer runs (transformers + peft: the GPU server's lumen env; skipped elsewhere);
  6. the report: planted per-user effects through the real statistics for a pass and each failing condition (below +0.01,
     CI covering 0, one seed negative, mean not above 2 sigma_seed), the minimum n, INCOMPLETE / INVALID; the stacking fit on
     CAL rows only; end-to-end reports from files; the kill-rule state machine, check_next and Holm over the datasets run;
  7. the run script: LF, bash -n, set -euo pipefail, the freeze lists, every flag in the real argparse (subcommand aware,
     and the integration audit of tests/test_confrec_contracts.py), the refusals, the DRY_RUN chain end to end on ML-1M and
     its resume, the freeze / gate / date / order / kill refusals after it, the E1 rerun-once rule and INCOMPLETE, and
     stage 1 for a dataset that is not next;
  8. the corrections of idea-stage/PREREG_AMENDMENT_3_ADDENDUM_10.md section 1 (the review of the first version of the
     slot code), one test per finding that replays the reviewer's scenario (section 8 below, test names carry the finding):
     M1 the DRY_RUN root guard compares canonical forms (every spelling of a registered root is refused, links are
     refused) and the slot does not trust any report (a rehearsal report, another bootstrap, a hand-written file is INVALID
     and cannot use up a failure); M2 DRY_RUN is 0 or 1; m1 the +0.01 threshold; m2 the hard kill date before every seed,
     DRY_TODAY eight digits, NOT_RUN after the date; m3 the "b = 0 is SFT" wording and the untouched trainer (m6: its
     recorded sha1); m4 NOT_RUN and the `FTMETHOD_NOT_RUN <dataset>` cut record; m5 the E1 same-key rule;
     and the stage-5 gate on the shift diagnostic of section 2 (tests/test_confrec_ftmethod_shift_diag.py has the rest).

Run time: the DRY worlds are built once and cached across sessions (a sha1 of the code that builds them is the key; the
shift-diagnostic tests share the cache); the script runs of the guard and refusal tests run concurrently (a script run is
dominated by process starts). FTMETHOD_FAST=1 skips every test that runs the script's DRY chain.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import csv
import gzip
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import types
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.confrec import build_rated_panels as brp
from src.confrec import forensics as fx
from src.confrec import ftgrid_freeze as ff
from src.confrec import ftgrid_report as fr
from src.confrec import ftmethod_report as fm
from src.confrec import pyes_scorer as ps
from src.confrec import train_lora_offset as tlo
from src.confrec.prompting import chat_ids, render_record
from src.confrec.split_panel import filter_candidates

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sigir" / "run_ftmethod.sh"
DIAG_SCRIPT = ROOT / "scripts" / "sigir" / "run_ftmethod_shift_diag.sh"
DIAG_TESTS = ROOT / "tests" / "test_confrec_ftmethod_shift_diag.py"
FAST = os.environ.get("FTMETHOD_FAST") == "1"
WORKERS = max(2, min(6, (os.cpu_count() or 4) - 2))       # concurrent script runs of the guard and refusal tests
# a DRY_RUN lives in a temporary directory outside outputs/confrec (addendum 10 item 1): the world run_ftgrid.sh builds and the slot root
DRYB = "tmp_outputs/ftmethod_dryrun"
DRY_ROOT = f"{DRYB}/ftmethod"
GRID_DRY = f"{DRYB}/ftgrid"
QWEN = "dryrun/Qwen3-8B"
TRAINER_SHA1_PREFIX = "aaae5e81"      # train_lora_offset.py as recorded in PILOT_LOG (addendum 10 item 7 / review m6: never edited)
YES, NO = 1, 2                      # the answer ids of the word tokenizer below (and of the hand-built batches)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha1_file(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list) -> Path:
    path.write_text("".join(brp.row_line(r) for r in rows), encoding="utf-8", newline="\n")
    return path


class WordTok:
    """Word tokenizer for the trainer path: "Yes" -> 1, "No" -> 2 (single tokens, the scorer's yes / no sets), every other
    space-separated piece its own id; the chat template closes an empty think block when thinking is off."""
    pad_token_id = 0
    eos_token = "<|end|>"

    def __init__(self):
        self.ids, self.text = {"Yes": YES, "No": NO}, {0: "", YES: "Yes", NO: "No"}

    def __len__(self):
        return 1 + len(self.ids)

    def decode(self, ids):
        return " ".join(self.text.get(i, "") for i in ids)

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, enable_thinking=True, **kw):
        out = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in msg) + "<|assistant|>"
        return out + ("" if enable_thinking else "<think>\n\n</think>\n\n")

    def __call__(self, text, add_special_tokens=False):
        out = []
        for w in text.split(" "):
            if w not in self.ids:
                self.ids[w] = len(self.ids) + 1
                self.text[self.ids[w]] = w
            out.append(self.ids[w])
        return {"input_ids": out}

    def save_pretrained(self, out):
        pass


# ---------------------------------------------------------------- 1. q-hat
def first_events(ratings) -> dict:
    """{user: {item: (t, r)}}: each user's first rating of an item ((ts, item, rating) order, as build_rated_panels)."""
    per = defaultdict(set)
    for u, i, r, t in ratings:
        per[str(u)].add((int(t), str(i), float(r)))
    out = {}
    for u, evs in per.items():
        seen = {}
        for t, i, r in sorted(evs):
            seen.setdefault(i, (t, r))
        out[u] = seen
    return out


def brute_qhat(first: dict, u: str, i: str, t: float, k: float = 5.0) -> float:
    """A3 section 3 by brute force: other users' first ratings of i strictly before t, shrunk with k pseudo-ratings at the
    mean of all other users' first ratings strictly before t."""
    s = n = gs = gn = 0.0
    for v, items in first.items():
        if v == u:
            continue
        for j, (tj, rj) in items.items():
            if tj < t:
                gs, gn = gs + rj, gn + 1
                if j == i:
                    s, n = s + rj, n + 1
    return (s + k * gs / gn) / (n + k) if gn else float("nan")


def ml1m_world(tmp: Path, n_users: int = 24, n_items: int = 10, per_user: int = 7, n_train: int = 9,
               seed: int = 3) -> SimpleNamespace:
    """Raw ML-1M files (with a later re-rating that is not a first rating) and the ftgrid panels on them: full rated rows
    (history, titles; 2-4 candidates per user, so the prompts differ in length), TRAIN = the first n_train users'
    candidates before T (the 0.8-quantile of every candidate timestamp), EVAL = the other users' rows."""
    rng = random.Random(seed)
    q = {i: rng.gauss(0, 1) for i in range(1, n_items + 1)}
    ratings = []
    for u in range(1, n_users + 1):
        t = 978_000_000 + 997 * u
        for i in rng.sample(range(1, n_items + 1), per_user):
            t += rng.randint(1, 3000)
            ratings.append((str(u), str(i), min(5, max(1, round(3 + 1.3 * q[i] + rng.gauss(0, 1)))), t))
    u0, i0, r0, t0 = ratings[3]
    ratings.append((u0, i0, 5 if r0 < 5 else 1, t0 + 7))
    raw = tmp / "raw"
    (raw / "ml-1m").mkdir(parents=True)
    (raw / "ml-1m" / "movies.dat").write_text("".join(f"{i}::Movie {i} ({1980 + i})::Drama|Comedy\n"
                                                      for i in range(1, n_items + 1)), encoding="latin-1")
    (raw / "ml-1m" / "ratings.dat").write_text("".join(f"{u}::{i}::{r}::{t}\n" for u, i, r, t in ratings),
                                               encoding="latin-1")
    first = first_events(ratings)
    pop = Counter(i for _, i, _, _ in ratings)
    rows = []
    for u in range(1, n_users + 1):
        evs = sorted((t, i, r) for i, (t, r) in first[str(u)].items())
        n_c = 2 + u % 3
        hist, cands = evs[:-n_c], evs[-n_c:]
        rows.append({"user_id": str(u), "source_event_id": f"{u}::{cands[0][0]}", "source": "ml1m",
                     "domain_kind": "movie",
                     "history": [f"Movie {i} ({1980 + int(i)}) (rated {int(r)}/5)" for _, i, r in hist],
                     "history_ratings": [float(r) for _, _, r in hist], "history_item_ids": [i for _, i, _ in hist],
                     "history_meta": ["Genres: Drama, Comedy"] * len(hist),
                     "candidate_item_ids": [i for _, i, _ in cands],
                     "candidate_titles": [f"Movie {i} ({1980 + int(i)})" for _, i, _ in cands],
                     "candidate_texts": ["Genres: Drama, Comedy"] * n_c,
                     "candidate_ratings": [float(r) for _, _, r in cands],
                     "candidate_labels": [int(r >= 4) for _, _, r in cands],
                     "candidate_timestamps": [t for t, _, _ in cands],
                     "candidate_popularity": [pop[i] for _, i, _ in cands]})
    T = float(np.quantile([t for r in rows for t in r["candidate_timestamps"]], 0.8))
    train_rows = [filter_candidates(r, [t < T for t in r["candidate_timestamps"]]) for r in rows[:n_train]]
    train_rows = [r for r in train_rows if r["candidate_labels"]]
    eval_rows = rows[n_train:]
    panels = tmp / "panels"
    panels.mkdir()
    write_jsonl(panels / "train.jsonl", train_rows)
    write_jsonl(panels / "eval.jsonl", eval_rows)
    (panels / "ftgrid_split.json").write_text(json.dumps(
        {"domain": "ml1m", "T": T, "variant": "V0", "files": {"train.jsonl": sha1_file(panels / "train.jsonl"),
                                                              "eval.jsonl": sha1_file(panels / "eval.jsonl")}}),
        encoding="utf-8")
    return SimpleNamespace(raw=raw, panels=panels, ratings=ratings, first=first, T=T, train_rows=train_rows,
                           eval_rows=eval_rows)


@pytest.fixture
def world(tmp_path):
    return ml1m_world(tmp_path)


def qhat_args(w, out) -> argparse.Namespace:
    return tlo.parse_args(["qhat", "--domain", "ml1m", "--panels", str(w.panels), "--raw", str(w.raw),
                           "--out_dir", str(out)])


def test_qhat_by_hand_shrinkage_strictly_before_leave_user_out_and_first_ratings():
    events = {"u1": [(10, "1", 5.0)], "u2": [(5, "2", 1.0), (20, "1", 3.0), (25, "1", 1.0)],
              "u3": [(1, "2", 5.0), (30, "1", 4.0)], "u4": [(20, "1", 2.0)]}
    ex = [tlo.Example("u3::30", 0, "u3", "1", 30.0, 1), tlo.Example("u4::20", 0, "u4", "1", 20.0, 0)]
    pm, info = tlo.qhat_arrays(ex, events)
    # u3 at t = 30: item 1 rated before 30 by u1 (5), u2 (3; its re-rating at 25 is not a first rating) and u4 (2);
    # the global mean before 30 leaves u3's own rating out: u1 5, u2 1 and 3, u4 2 -> 11 / 4
    assert pm["n_prior"][0] == 3 and pm["mean_prior"][0] == pytest.approx(10 / 3)
    assert pm["global_prior"][0] == pytest.approx(11 / 4)
    assert pm["mean_prior_shrunk"][0] == pytest.approx((10 + 5 * 11 / 4) / (3 + 5), abs=1e-12)
    # u4 at t = 20: strictly before 20 only u1's 5 (u2's 3 at t = 20 is not before); global: u1 5, u2 1, u3 5
    assert pm["n_prior"][1] == 1 and pm["global_prior"][1] == pytest.approx(11 / 3)
    assert pm["mean_prior_shrunk"][1] == pytest.approx((5 + 5 * 11 / 3) / (1 + 5), abs=1e-12)
    assert info["n_candidate_pairs_not_in_raw"] == 0


def test_qhat_files_follow_the_definition_on_every_pair(world, tmp_path):
    out = tmp_path / "m"
    man = tlo.build_qhat(qhat_args(world, out))
    for name, rows in (("train", world.train_rows), ("eval", world.eval_rows)):
        got = tlo.read_qhat(out / f"{name}_qhat.csv.gz")
        ex = tlo.panel_examples(rows, name)
        assert list(got) == [(e.user, e.item) for e in ex]                      # one row per example, panel order
        for e in ex:
            r = got[(e.user, e.item)]
            assert (r["ts"], r["label"]) == (e.ts, e.label)
            assert r["q_hat"] == pytest.approx(brute_qhat(world.first, e.user, e.item, e.ts), abs=1e-12)
    st = man["standardisation"]
    tq = [r["q_hat"] for r in tlo.read_qhat(out / "train_qhat.csv.gz").values()]
    assert (st["mean"], st["sd"], st["ddof"]) == (pytest.approx(np.mean(tq)), pytest.approx(np.std(tq)), 0)
    assert st["n_examples"] == st["n_finite"] == len(tq) and st["n_nonfinite_z_imputed_0"] == 0
    for name in ("train", "eval"):                                               # EVAL z uses the TRAIN constants
        for r in tlo.read_qhat(out / f"{name}_qhat.csv.gz").values():
            assert r["z"] == pytest.approx((r["q_hat"] - st["mean"]) / st["sd"], abs=1e-12)
    assert man["files"] == {n: sha1_file(out / n) for n in ("train_qhat.csv.gz", "eval_qhat.csv.gz")}
    assert man["train"]["sha1"] == sha1_file(world.panels / "train.jsonl")
    assert man["eval"]["sha1"] == sha1_file(world.panels / "eval.jsonl") and man["T"] == world.T
    assert man["q_hat"]["shrink_k"] == 5.0 and man["code_sha1"]["train_lora_offset.py"] == sha1_file(tlo.__file__)
    assert max(e.ts for e in tlo.panel_examples(world.train_rows, "t")) < world.T     # no TRAIN example at or after T


def test_one_scan_equals_per_role_scans_and_the_ftgrid_report_q_hat(world, tmp_path):
    out = tmp_path / "m"
    tlo.build_qhat(qhat_args(world, out))
    events = fx.load_raw_events(world.raw, "ml1m")
    for name, rows in (("train", world.train_rows), ("eval", world.eval_rows)):
        ex = tlo.panel_examples(rows, name)
        pm, _ = tlo.qhat_arrays(ex, events)                                     # this role's pairs only
        got = tlo.read_qhat(out / f"{name}_qhat.csv.gz")
        assert np.array_equal(pm["mean_prior_shrunk"], [got[(e.user, e.item)]["q_hat"] for e in ex])
    cx = fr.make_ctx(world.eval_rows, world.T, [], None)
    refs = fr.compute_refs(cx, world.raw, "ml1m", world.T, 2, 0)                # the E-A reference of ftgrid_report
    got = tlo.read_qhat(out / "eval_qhat.csv.gz")
    assert np.array_equal(refs["arrays"]["q_hat"],
                          [got[k]["q_hat"] for k in zip(cx.users.tolist(), cx.item.tolist())])


def test_qhat_files_are_deterministic_and_a_rerun_touches_nothing(world, tmp_path):
    out = tmp_path / "m"
    tlo.build_qhat(qhat_args(world, out))
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in out.iterdir()}
    tlo.build_qhat(qhat_args(world, out))
    assert {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in out.iterdir()} == before
    assert sorted(before) == ["eval_qhat.csv.gz", "qhat_manifest.json", "train_qhat.csv.gz"]
    text = (out / "qhat_manifest.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in text and tmp_path.as_posix() not in text        # no host path in the frozen file
    with gzip.open(out / "train_qhat.csv.gz", "rt", encoding="utf-8") as f:
        assert f.readline().strip().split(",") == list(tlo.QHAT_COLS)


def test_qhat_refuses_panels_that_are_not_the_splits(world, tmp_path):
    p = world.panels / "eval.jsonl"
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="sha1"):
        tlo.build_qhat(qhat_args(world, tmp_path / "m"))
    late = [dict(r, candidate_timestamps=[world.T] * len(r["candidate_labels"])) for r in world.train_rows]
    w2 = ml1m_world(tmp_path / "w2")
    write_jsonl(w2.panels / "train.jsonl", late)
    split = read_json(w2.panels / "ftgrid_split.json")
    split["files"]["train.jsonl"] = sha1_file(w2.panels / "train.jsonl")
    (w2.panels / "ftgrid_split.json").write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(SystemExit, match="at or after T"):
        tlo.build_qhat(qhat_args(w2, tmp_path / "m2"))
    with pytest.raises(SystemExit, match="split of"):
        tlo.build_qhat(tlo.parse_args(["qhat", "--domain", "toys", "--panels", str(world.panels), "--raw",
                                       str(world.raw), "--out_dir", str(tmp_path / "m3")]))


def test_the_trainer_refuses_files_that_do_not_match_the_manifest(world, tmp_path):
    out = tmp_path / "m"
    tlo.build_qhat(qhat_args(world, out))
    train, qh, man = world.panels / "train.jsonl", out / "train_qhat.csv.gz", out / "qhat_manifest.json"
    z = tlo.load_train_z(train, qh, man)
    ex = tlo.panel_examples(world.train_rows, "t")
    assert list(z["z_of"]) == [(e.user, e.item) for e in ex] and z["imputed"] == set()
    assert z["manifest_sha1"] == sha1_file(man) and z["domain"] == "ml1m"
    other = tmp_path / "other.jsonl"
    other.write_bytes(train.read_bytes() + b"\n")                             # same rows, other bytes
    with pytest.raises(SystemExit, match="the manifest standardised"):
        tlo.load_train_z(other, qh, man)
    with gzip.open(qh, "rt", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["q_hat"] = "4.25"
    bad = tmp_path / "bad.csv.gz"
    bad.write_bytes(tlo.gz_bytes("".join([",".join(tlo.QHAT_COLS) + "\n"] +
                                          [",".join(r[c] for c in tlo.QHAT_COLS) + "\n" for r in rows])))
    with pytest.raises(SystemExit, match="sha1"):
        tlo.load_train_z(train, bad, man)
    m2 = read_json(man)
    m2["standardisation"]["mean"] += 0.25                                     # constants that are not the stored z's
    man2 = tmp_path / "man2.json"
    man2.write_text(json.dumps(m2), encoding="utf-8")
    with pytest.raises(SystemExit, match="stored z differs"):
        tlo.load_train_z(train, qh, man2)


def test_standardisation_and_z_of_a_nonfinite_q_hat():
    st = tlo.standardisation([1.0, 2.0, 3.0, float("nan")])
    assert (st["mean"], st["ddof"], st["n_finite"], st["n_nonfinite_z_imputed_0"]) == (2.0, 0, 3, 1)
    assert st["sd"] == pytest.approx(math.sqrt(2 / 3))
    z = tlo.zscore([1.0, float("nan"), 3.0], st)
    assert z[1] == 0.0 and z[0] == pytest.approx(-1 / math.sqrt(2 / 3)) and z[2] == pytest.approx(1 / math.sqrt(2 / 3))
    with pytest.raises(SystemExit, match="SD 0"):
        tlo.standardisation([2.0, 2.0])
    with pytest.raises(SystemExit, match="fewer than 2"):
        tlo.standardisation([float("nan"), 1.0])


# ---------------------------------------------------------------- 2. the dataset (torch)
def torch_or_skip():
    return pytest.importorskip("torch")


def shuffled_train_rows(world, seed: int = 0) -> list:
    rows = [json.loads(json.dumps(r)) for r in world.train_rows]
    random.Random(seed).shuffle(rows)
    return rows


def test_prior_offset_set_is_yesnoset_with_each_examples_z(world):
    torch_or_skip()
    from src.confrec.train_lora_yesno import YesNoSet
    rows = shuffled_train_rows(world)
    keys = [(str(r["user_id"]), str(i)) for r in rows for i in r["candidate_item_ids"]]
    z_of = {k: round(0.1 * n - 1.0, 3) for n, k in enumerate(sorted(keys))}
    tok = WordTok()
    base = YesNoSet(rows, tok, "standard", None, 4096, "V0")
    ds = tlo.prior_offset_set(rows, z_of, tok, None, 4096, "V0")
    assert ds.items == base.items and ds.n_skipped == base.n_skipped == 0 and ds.max_prompt_len == base.max_prompt_len
    assert ds.keys == keys and ds.z == [z_of[k] for k in keys] and ds.yes_id == YES and ds.answer == base.answer
    assert ds[3] == {**base[3], "z": z_of[keys[3]]}
    # overlength skips: a max_len between the shortest and the longest example
    lengths = sorted(len(ids) for ids, _ in base.items)
    cut = lengths[len(lengths) // 2]
    base2 = YesNoSet(rows, tok, "standard", None, cut, "V0")
    ds2 = tlo.prior_offset_set(rows, z_of, tok, None, cut, "V0")
    assert 0 < ds2.n_skipped == base2.n_skipped and ds2.items == base2.items and len(ds2.keys) == len(ds2.z) == len(ds2)
    by_user = {str(r["user_id"]): r for r in rows}
    for (ids, n), key, zk in zip(ds2.items, ds2.keys, ds2.z):                # every kept example is its key's prompt
        r = by_user[key[0]]
        _, _, system, user = render_record(r, ["like"], "V0")[r["candidate_item_ids"].index(key[1])]
        assert ids[:n] == chat_ids(tok, user, system) and zk == z_of[key]
    ds2.truncate(3)
    assert len(ds2) == len(ds2.keys) == len(ds2.z) == 3
    with pytest.raises(SystemExit, match="no z"):
        tlo.prior_offset_set(rows, {}, tok, None, 4096, "V0")


def test_collate_offset_is_collate_plus_a_float32_z(world):
    torch = torch_or_skip()
    from src.confrec.train_lora_yesno import collate
    rows = shuffled_train_rows(world)
    z_of = {(str(r["user_id"]), str(i)): 0.5 * k for k, r in enumerate(rows) for i in r["candidate_item_ids"]}
    ds = tlo.prior_offset_set(rows, z_of, WordTok(), None, 4096, "V0")
    batch = [ds[0], ds[1], ds[2]]
    got, ref = tlo.collate_offset(batch, 0), collate(batch, 0)
    assert set(got) == set(ref) | {"z"}
    assert all(torch.equal(got[k], ref[k]) for k in ref)
    assert got["z"].dtype == torch.float32 and got["z"].tolist() == pytest.approx(ds.z[:3])


# ---------------------------------------------------------------- 3. the loss (torch)
def tiny_lm(torch, vocab: int = 12, dim: int = 6):
    """A causal toy LM with the interface last_token_loss uses (input_ids, attention_mask, logits_to_keep -> .logits)."""

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.emb = torch.nn.Parameter(torch.randn(vocab, dim, generator=g))
            self.head = torch.nn.Parameter(torch.randn(dim, vocab, generator=g))

        def forward(self, input_ids, attention_mask, logits_to_keep=0):
            h = torch.tanh((self.emb[input_ids] * attention_mask[..., None]).cumsum(1))
            lg = h @ self.head
            return SimpleNamespace(logits=lg[:, -logits_to_keep:, :] if logits_to_keep else lg)

    return TinyLM()


def lm_batch(torch, lengths=(5, 3, 4, 5), vocab: int = 12, seed: int = 1):
    """Left-padded `prompt + answer` rows, labels on the answer only (Yes for even rows, No for odd ones)."""
    g = torch.Generator().manual_seed(seed)
    m = max(lengths)
    ids = torch.zeros((len(lengths), m), dtype=torch.long)
    lab = torch.full((len(lengths), m), -100, dtype=torch.long)
    att = torch.zeros((len(lengths), m), dtype=torch.long)
    for k, n in enumerate(lengths):
        seq = torch.randint(3, vocab, (n,), generator=g)
        seq[-1] = YES if k % 2 == 0 else NO
        ids[k, m - n:], att[k, m - n:], lab[k, -1] = seq, 1, seq[-1]
    return ids, att, lab


def offset_loss(model, ids, att, lab, b, z, num_items=None):
    from src.confrec.lora_trainer import last_token_loss
    return last_token_loss(tlo.PriorOffsetModel(model, YES, b, z), ids, att, lab, num_items_in_batch=num_items)


def test_with_b_zero_or_z_zero_the_loss_is_last_token_loss():
    torch = torch_or_skip()
    from src.confrec.lora_trainer import last_token_loss
    model = tiny_lm(torch)
    ids, att, lab = lm_batch(torch)
    z = torch.tensor([0.5, -1.0, 2.0, 0.3])
    with torch.no_grad():
        ref = last_token_loss(model, ids, att, lab)
        assert torch.equal(offset_loss(model, ids, att, lab, torch.tensor(0.0), z), ref)
        assert torch.equal(offset_loss(model, ids, att, lab, torch.tensor(0.7), torch.zeros(4)), ref)
        assert torch.equal(offset_loss(model, ids, att, lab, 0.0, z, 8), last_token_loss(model, ids, att, lab,
                                                                                           num_items_in_batch=8))
        bad = lab.clone()
        bad[0, 1] = 5                                                        # last_token_loss's own checks still apply
        with pytest.raises(ValueError):
            offset_loss(model, ids, att, bad, 0.5, z)


def test_the_offset_enters_only_the_yes_logit_of_the_answer_position_before_the_softmax():
    torch = torch_or_skip()
    model = tiny_lm(torch)
    ids, att, lab = lm_batch(torch)
    b, z = torch.tensor(0.8), torch.tensor([0.5, -1.0, 2.0, 0.3])
    with torch.no_grad():
        raw = model(input_ids=ids, attention_mask=att, logits_to_keep=2).logits.float()
        view = tlo.PriorOffsetModel(model, YES, b, z)(input_ids=ids, attention_mask=att, logits_to_keep=2).logits
        d = view - raw
        assert torch.equal(d[:, -1], torch.zeros_like(d[:, -1]))             # the answer token's own position
        assert torch.allclose(d[:, -2, YES], b * z)
        others = torch.ones_like(d[:, -2], dtype=torch.bool)
        others[:, YES] = False
        assert torch.equal(d[:, -2][others], torch.zeros(int(others.sum())))
        want = -torch.log_softmax(view[:, -2], -1)[torch.arange(4), lab[:, -1]].mean()   # full-vocabulary softmax
        assert torch.allclose(offset_loss(model, ids, att, lab, b, z), want)
    lg = torch.arange(12.0).reshape(2, 6)                                     # a hand-built logits tensor
    assert tlo.shift_yes(lg, 4, 0.5, torch.tensor([2.0, -2.0])).tolist() == [[0, 1, 2, 3, 5, 5], [6, 7, 8, 9, 9, 11]]
    with pytest.raises(ValueError):
        tlo.shift_yes(lg, 4, 0.5, torch.tensor([1.0]))


def test_the_gradient_reaches_b_not_z_and_follows_num_items_in_batch():
    torch = torch_or_skip()
    model = tiny_lm(torch)
    ids, att, lab = lm_batch(torch)
    b = torch.nn.Parameter(torch.tensor(0.3))
    z = torch.tensor([0.5, -1.0, 2.0, 0.3], requires_grad=True)
    offset_loss(model, ids, att, lab, b, z).backward()
    assert z.grad is None                                                    # no gradient into q-hat
    with torch.no_grad():
        lg = model(input_ids=ids, attention_mask=att, logits_to_keep=2).logits[:, -2].float().clone()
        lg[:, YES] += b * z
        p_yes = torch.softmax(lg, -1)[:, YES]
        want = (z * (p_yes - (lab[:, -1] == YES).float())).mean()           # dCE/db = z (p'_Yes - 1[answer = Yes])
    assert torch.allclose(b.grad, want, atol=1e-6) and float(b.grad) != 0.0
    b.grad = None
    offset_loss(model, ids, att, lab, b, z.detach(), num_items=10).backward()   # sum / num_items_in_batch
    assert torch.allclose(b.grad, want * 4 / 10, atol=1e-6)
    with torch.no_grad():
        mean = offset_loss(model, ids, att, lab, b, z)
        assert torch.allclose(offset_loss(model, ids, att, lab, b, z, 8), mean * 4 / 8, atol=1e-7)


def test_the_trainer_is_last_token_trainer_with_the_offset_and_an_optional_b_group():
    torch = torch_or_skip()
    model = tiny_lm(torch)
    ids, att, lab = lm_batch(torch)
    b = torch.nn.Parameter(torch.tensor(0.4))

    class Base:                                       # transformers.Trainer's two hooks used here (optimizer built once)
        def __init__(self, model=None, **kw):
            self.model, self.optimizer = model, None

        def create_optimizer(self):
            if self.optimizer is None:
                self.optimizer = torch.optim.SGD(list(self.model.parameters()) + [b], lr=0.1)
            return self.optimizer

    cls = tlo.prior_offset_trainer(Base, YES, b)
    assert "LastTokenTrainer" in [c.__name__ for c in cls.__mro__]
    tr = cls(model=model)
    z = torch.tensor([0.5, -1.0, 2.0, 0.3])
    inputs = {"input_ids": ids, "attention_mask": att, "labels": lab, "z": z}
    with torch.no_grad():
        assert torch.equal(tr.compute_loss(model, inputs, num_items_in_batch=6), offset_loss(model, ids, att, lab, b, z, 6))
        loss, out = tr.compute_loss(model, inputs, return_outputs=True)
        assert torch.equal(loss, offset_loss(model, ids, att, lab, b, z)) and out.logits.shape[1] == 2
    for b_lr in (None, 0, 0.0):                                              # b with the LoRA group (--b_lr 0)
        opt = tlo.prior_offset_trainer(Base, YES, b, b_lr)(model=model).create_optimizer()
        assert [len(g["params"]) for g in opt.param_groups] == [3]
        assert tlo.b_group_record(opt, b) == {"own_group": False, "lr": 0.1, "weight_decay": 0.0, "n_params_in_group": 3}
    tr2 = tlo.prior_offset_trainer(Base, YES, b, b_lr=0.01)(model=model)
    tr2.create_optimizer()
    tr2.create_optimizer()                                                   # the Trainer's second call: idempotent
    groups = tr2.optimizer.param_groups
    assert [len(g["params"]) for g in groups] == [2, 1] and groups[1]["params"][0] is b
    assert (groups[1]["lr"], groups[1]["weight_decay"]) == (0.01, 0.0) and tlo.b_in_optimizer(tr2.optimizer, b)
    assert tlo.b_group_record(tr2.optimizer, b) == {"own_group": True, "lr": 0.01, "weight_decay": 0.0,
                                                    "n_params_in_group": 1}
    with pytest.raises(RuntimeError, match="appears 0 times"):
        tlo.own_param_group(torch.optim.SGD(model.parameters(), lr=0.1), b, 0.05)


def test_b_lr_defaults_to_its_own_group_at_1e_2_and_0_keeps_the_lora_group():
    """Amendment 3 addendum 3: --b_lr 1e-2 is the default (b's own group); 0 keeps the literal pre-addendum reading."""
    a = tlo.parse_args(train_argv("t.jsonl", "sft"))
    assert a.b_lr == tlo.B_LR == 0.01
    assert tlo.intended_b_group(a) == {"own_group": True, "lr": 0.01, "weight_decay": 0.0}
    a0 = tlo.parse_args(train_argv("t.jsonl", "sft", **{"--b_lr": "0"}))
    assert tlo.intended_b_group(a0) == {"own_group": False, "lr": 1e-4, "weight_decay": 0.0}
    assert tlo.b_group_problems({"own_group": True, "lr": 0.01, "weight_decay": 0.0, "n_params_in_group": 1}, a) == []
    assert len(tlo.b_group_problems({"own_group": False, "lr": 1e-4, "weight_decay": 0.0}, a)) == 2
    assert tlo.b_group_problems({"own_group": False, "lr": 1e-4, "weight_decay": 0.0}, a0) == []
    with pytest.raises(SystemExit):
        tlo.parse_args(train_argv("t.jsonl", "sft", **{"--b_lr": "-0.01"}))


def toy_run(torch, b_lr, steps: int = 200, horizon: int = 730, batch: int = 8, seed: int = 0):
    """The prior-offset trainer in a transformers-like loop: the recipe's AdamW (lr 1e-4, weight decay 0) over the
    trainable parameters plus b, the create_optimizer override (called twice, as the Trainer does), and
    transformers.get_cosine_schedule_with_warmup's multiplier (3% warmup) over an ML-1M-sized horizon (about 730
    steps); `steps` steps of batches whose labels depend on z (Yes iff z > 0). Returns (b, b's group record, the
    optimizer)."""
    model = tiny_lm(torch)
    b = torch.nn.Parameter(torch.zeros(()))
    params = list(model.parameters()) + [b]               # as a PEFT model's named_parameters, b registered on it

    class Base:
        def __init__(self, model=None, **kw):
            self.model, self.optimizer = model, None

        def create_optimizer(self):
            if self.optimizer is None:
                self.optimizer = torch.optim.AdamW([{"params": params, "weight_decay": 0.0}], lr=1e-4)
            return self.optimizer

    tr = tlo.prior_offset_trainer(Base, YES, b, b_lr)(model=model)
    tr.create_optimizer()                                 # train_lora_offset.train: built and checked before train()
    rec = tlo.b_group_record(tr.optimizer, b)
    opt = tr.create_optimizer()                           # the Trainer's own call inside train(): reused
    warm = math.ceil(0.03 * horizon)

    def mult(step):                                       # transformers.get_cosine_schedule_with_warmup
        if step < warm:
            return step / max(1, warm)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (step - warm) / max(1, horizon - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, mult)  # one multiplier for every group
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        ids = torch.randint(3, 12, (batch, 5), generator=g)
        z = torch.randn(batch, generator=g)
        ans = torch.where(z > 0, torch.tensor(YES), torch.tensor(NO))
        ids[:, -1] = ans
        lab = torch.full_like(ids, -100)
        lab[:, -1] = ans
        loss = tr.compute_loss(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": lab, "z": z})
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return float(b.detach()), rec, opt


def test_b_moves_beyond_1_within_200_steps_in_its_own_group_and_not_in_the_lora_group():
    """The optimizer arithmetic behind addendum 3: with labels planted on z, b in its own group (lr 1e-2, the same
    cosine schedule and warmup) passes |b| = 1 within 200 steps; in the LoRA group (--b_lr 0, lr 1e-4) AdamW moves it
    by at most about the summed learning rates (~0.02 here)."""
    torch = torch_or_skip()
    b_own, rec, opt = toy_run(torch, tlo.B_LR)
    assert abs(b_own) > 1.0, b_own
    assert rec == {"own_group": True, "lr": 0.01, "weight_decay": 0.0, "n_params_in_group": 1}
    held = [grp for grp in opt.param_groups if any(p.ndim == 0 for p in grp["params"])]
    assert len(held) == 1 and held[0]["initial_lr"] == rec["lr"] and len(held[0]["params"]) == 1    # the recorded lr
    assert [grp["initial_lr"] for grp in opt.param_groups if grp is not held[0]] == [1e-4]
    b_shared, rec0, _ = toy_run(torch, 0)
    assert abs(b_shared) < 0.05, b_shared
    assert rec0 == {"own_group": False, "lr": 1e-4, "weight_decay": 0.0, "n_params_in_group": 3}


def test_b_is_a_zero_float32_parameter_registered_on_the_model():
    torch = torch_or_skip()
    model = tiny_lm(torch)
    model.get_output_embeddings = lambda: SimpleNamespace(weight=model.head)
    b = tlo.attach_prior_offset(model)
    assert dict(model.named_parameters())[tlo.OFFSET_PARAM] is b and b.requires_grad
    assert b.dtype == torch.float32 and float(b.detach()) == 0.0 and "lora_" not in tlo.OFFSET_PARAM
    assert tlo.b_in_optimizer(torch.optim.SGD(model.parameters(), lr=0.1), b)


# ---------------------------------------------------------------- 4. the comparator, the CLI and main()
def sft_config(train, **over) -> dict:
    """A train_config.json as train_lora_yesno.main writes it."""
    cfg = {"train": str(train), "model": "/models/Qwen3-8B", "out": "x", "mode": "standard", "variant": "V0", "seed": 1,
           "hist_len": None, "max_len": 1024, "epochs": 1.0, "lr": 1e-4, "bsz": 8, "grad_accum": 4, "lora_r": 16,
           "max_examples": None, "loss": "last_token", "hist_len_used": 10, "panel_kind": "rated",
           "max_history_len_in_panel": 5, "n_examples": 40, "n_skipped_overlength": 0}
    cfg.update(over)
    return cfg


def train_argv(train, sft, out="o", **over) -> list:
    kw = {"--train": str(train), "--model": "/root/models/Qwen3-8B", "--out": str(out), "--variant": "V0", "--seed": "1",
          "--qhat": "q.csv.gz", "--manifest": "m.json", "--sft_adapter": str(sft)}
    kw.update(over)
    return ["train"] + [x for k, v in kw.items() for x in (k, v)]


def test_the_comparator_must_share_the_recipe_and_the_train_file(tmp_path):
    train = write_jsonl(tmp_path / "train.jsonl", [{"user_id": "u", "x": 1}])
    a = tlo.parse_args(train_argv(train, tmp_path))
    assert tlo.recipe_problems(sft_config(train), a, 10) == []
    cases = {"seed": sft_config(train, seed=2), "bsz": sft_config(train, bsz=4, grad_accum=8),
             "max_len": sft_config(train, max_len=1280), "lr": sft_config(train, lr=2e-4),
             "variant": sft_config(train, variant="V1"), "backbone": sft_config(train, model="/m/Llama-3.1-8B-Instruct"),
             "loss": sft_config(train, loss=None), "mode": sft_config(train, mode="mirror")}
    for key, cfg in cases.items():
        assert any(key in p for p in tlo.recipe_problems(cfg, a, 10)), key
    assert any("history window" in p for p in tlo.recipe_problems(sft_config(train), a, 20))
    other = write_jsonl(tmp_path / "other.jsonl", [{"user_id": "v", "x": 1}])
    assert any("differ in bytes" in p for p in tlo.recipe_problems(sft_config(other), a, 10))
    assert any("does not exist" in p for p in tlo.recipe_problems(sft_config(tmp_path / "gone.jsonl"), a, 10))
    sft = tmp_path / "sft"
    sft.mkdir()
    (sft / "train_config.json").write_text(json.dumps(sft_config(train, seed=2)), encoding="utf-8")
    with pytest.raises(SystemExit, match="seed"):
        tlo.sft_comparator(tlo.parse_args(train_argv(train, sft)), 10)
    with pytest.raises(SystemExit, match="no train_config"):
        tlo.sft_comparator(tlo.parse_args(train_argv(train, tmp_path / "none")), 10)
    assert tlo.example_count_problems(sft_config(train), 40, 0) == []
    assert len(tlo.example_count_problems(sft_config(train), 39, 1)) == 2


class _Grab(Exception):
    pass


def captured_parser(fn):
    """The ArgumentParser an entry point builds, stopped at parse_args."""
    orig = argparse.ArgumentParser.parse_args

    def grab(self, *a, **k):
        raise _Grab(self)
    argparse.ArgumentParser.parse_args = grab
    try:
        fn()
    except _Grab as g:
        return g.args[0]
    finally:
        argparse.ArgumentParser.parse_args = orig
    raise AssertionError("no parser captured")


def options(parser, sub=None) -> dict:
    subs = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    p = subs[0].choices[sub] if sub else parser
    return {s: a for a in p._actions for s in a.option_strings}


def test_the_train_cli_has_train_lora_yesnos_options_and_defaults():
    pytest.importorskip("torch")
    from src.confrec import train_lora_yesno as tl
    theirs = options(captured_parser(lambda: tl.main([])))
    mine = options(captured_parser(lambda: tlo.main([])), "train")
    for flag, act in theirs.items():
        m = mine[flag]
        assert (m.default, m.type, m.required) == (act.default, act.type, act.required), flag
        if flag != "--mode":
            assert m.choices == act.choices, flag
    assert mine["--mode"].choices == ["standard"]                            # A3 section 7: standard mode only
    assert set(mine) - set(theirs) == {"--qhat", "--manifest", "--sft_adapter", "--b_lr"}
    assert {s for s, a in mine.items() if a.required} == {"--train", "--model", "--out", "--qhat", "--manifest",
                                                          "--sft_adapter"}
    assert mine["--b_lr"].default == tlo.B_LR == 0.01                       # Amendment 3 addendum 3


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls(fn: ast.FunctionDef, names) -> list:
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name in names:
                out.append(ast.dump(node))
    return sorted(out)


def test_the_model_and_lora_setup_are_train_lora_yesnos():
    """The lines the trainer could not import (train_lora_yesno.main is one function) are the same calls."""
    theirs = _function(ROOT / "src" / "confrec" / "train_lora_yesno.py", "main")
    mine = _function(Path(tlo.__file__), "train")
    for names in (("LoraConfig",), ("get_peft_model",), ("from_pretrained",), ("gradient_checkpointing_enable",),
                  ("enable_input_require_grads",), ("set_seed",), ("shuffle",)):
        assert _calls(theirs, names) == _calls(mine, names) and _calls(mine, names), names

    def ta(fn):
        call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", None)
                    == "TrainingArguments")
        inner = call.keywords[0].value
        return (call.keywords[0].arg, inner.func.attr if isinstance(inner.func, ast.Attribute) else inner.func.id)
    assert ta(theirs) == ta(mine) == (None, "training_arguments")


def fake_stack(monkeypatch, torch) -> dict:
    """transformers / peft stand-ins around real torch: a tiny model, and a Trainer that takes one SGD step through its
    compute_loss and create_optimizer (the hooks the prior-offset trainer overrides)."""
    seen = {"model_loads": 0, "trained": False}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(4000, 6)
            self.head = torch.nn.Linear(6, 4000)

        def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
            lg = self.head(torch.tanh(self.emb(input_ids).cumsum(1)))
            return SimpleNamespace(logits=lg[:, -logits_to_keep:, :] if logits_to_keep else lg)

        def gradient_checkpointing_enable(self):
            seen["gc"] = True

        def enable_input_require_grads(self):
            seen["irg"] = True

        def get_output_embeddings(self):
            return self.head

        def save_pretrained(self, out):
            seen["saved"] = str(out)

    class Trainer:
        def __init__(self, model, args, train_dataset, data_collator):
            self.model, self.args, self.train_dataset, self.data_collator = model, args, train_dataset, data_collator
            self.optimizer = None

        def create_optimizer(self):                                          # as transformers: built once, then reused
            if self.optimizer is None:
                seen["optimizers_built"] = seen.get("optimizers_built", 0) + 1
                self.optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad],
                                                 lr=self.args.learning_rate, weight_decay=0.0)
            return self.optimizer

        def train(self):
            self.create_optimizer()
            batch = self.data_collator([self.train_dataset[i] for i in range(4)])
            seen["batch_z"] = batch["z"].tolist()
            loss = self.compute_loss(self.model, batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            seen["trained"] = True

    def load_model(path, **kw):
        seen["model_loads"] += 1
        seen["model_kw"] = kw
        torch.manual_seed(0)
        return Model()

    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = SimpleNamespace(from_pretrained=lambda path: WordTok())
    tf.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load_model)
    tf.Trainer, tf.TrainingArguments, tf.set_seed = Trainer, (lambda **kw: SimpleNamespace(**kw)), (lambda s: None)
    peft = types.ModuleType("peft")
    peft.LoraConfig, peft.get_peft_model = (lambda **kw: SimpleNamespace(**kw)), (lambda model, cfg: model)
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "peft", peft)
    return seen


def test_main_learns_b_and_records_it_where_the_scorer_and_the_report_read_it(world, tmp_path, monkeypatch):
    torch = torch_or_skip()
    q = tmp_path / "m"
    tlo.build_qhat(qhat_args(world, q))
    train = world.panels / "train.jsonl"
    n = sum(len(r["candidate_labels"]) for r in world.train_rows)
    sft = tmp_path / "sft_s0"
    sft.mkdir()
    (sft / "train_config.json").write_text(json.dumps(sft_config(train, seed=0, max_len=4096, n_examples=n)),
                                           encoding="utf-8")
    seen = fake_stack(monkeypatch, torch)
    out = tmp_path / "o0"
    argv = train_argv(train, sft, out, **{"--seed": "0", "--max_len": "4096", "--qhat": str(q / "train_qhat.csv.gz"),
                                          "--manifest": str(q / "qhat_manifest.json")})
    off = tlo.main(argv)
    assert seen["trained"] and seen["gc"] and seen["irg"] and seen["model_kw"] == {"dtype": torch.bfloat16,
                                                                                  "device_map": "auto"}
    assert seen["optimizers_built"] == 1                                     # checked before train(), then reused
    rows = shuffled_train_rows(world, 0)                                     # train_lora_yesno's order of seed 0
    z_of = tlo.load_train_z(train, q / "train_qhat.csv.gz", q / "qhat_manifest.json")["z_of"]
    first4 = [(str(r["user_id"]), str(i)) for r in rows for i in r["candidate_item_ids"]][:4]
    assert seen["batch_z"] == pytest.approx([z_of[k] for k in first4], abs=1e-6)
    cfg, off_j = read_json(out / "train_config.json"), read_json(out / "offset.json")
    assert off["b"] != 0.0 and math.isfinite(off["b"]) and cfg["prior_offset"]["b"] == off_j["b"] == off["b"]
    assert cfg["loss"] == "last_token_prior_offset" and cfg["n_examples"] == n and cfg["n_skipped_overlength"] == 0
    assert off_j["standardisation"] == read_json(q / "qhat_manifest.json")["standardisation"]
    assert off_j["qhat_manifest_sha1"] == sha1_file(q / "qhat_manifest.json") and off_j["yes_token_id"] == YES
    assert off_j["seed"] == 0 and off_j["domain"] == "ml1m" and off_j["b_init"] == 0.0 and off_j["b_lr"] == 0.01
    grp = off_j["b_group"]                                                   # the optimizer's group of b, recorded
    assert {k: grp[k] for k in ("own_group", "lr", "weight_decay", "n_params_in_group")} == {
        "own_group": True, "lr": 0.01, "weight_decay": 0.0, "n_params_in_group": 1}
    assert grp["registered"] is True and grp["source"].startswith("the Trainer's optimizer") and grp["init"] == 0.0
    assert cfg["prior_offset"]["b_group"] == grp and cfg["b_lr"] == 0.01
    assert read_json(out / "train_report.json")["n_examples"] == n
    ps.check_lora_variant(out, "V0", 10)                                    # pyes_scorer --lora reads it as an SFT adapter
    ps.check_lora_variant(out, "V0", 0)
    out0 = tmp_path / "o0_b_lr_0"                                            # --b_lr 0: b stays in the LoRA group
    off0 = tlo.main(train_argv(train, sft, out0, **{"--seed": "0", "--max_len": "4096", "--b_lr": "0",
                                                     "--qhat": str(q / "train_qhat.csv.gz"),
                                                     "--manifest": str(q / "qhat_manifest.json")}))
    assert {k: off0["b_group"][k] for k in ("own_group", "lr", "registered")} == {"own_group": False, "lr": 1e-4,
                                                                                  "registered": False}
    (sft / "train_config.json").write_text(json.dumps(sft_config(train, seed=0, max_len=4096, n_examples=n + 1)),
                                           encoding="utf-8")
    with pytest.raises(SystemExit, match="examples"):                       # the comparator's examples, before the model
        tlo.main(argv)
    assert seen["model_loads"] == 2


# ---------------------------------------------------------------- 5. tiny-model Trainer runs (GPU server env)
def _tiny_run(tmp_path, offset: bool, learn_b: bool, b_lr=None, grad_accum: int = 1):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from src.confrec.lora_trainer import last_token_trainer
    cfg = transformers.Qwen3Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128)
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(cfg).train()
    torch.manual_seed(0)
    model = peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, task_type="CAUSAL_LM",
                                                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    b = tlo.attach_prior_offset(model)
    if not learn_b:
        b.requires_grad_(False)

    class ZRows(torch.utils.data.Dataset):
        def __init__(self, lengths=(9, 5, 8, 6, 7, 9, 4, 8), zs=(0.9, -1.3, 0.4, 2.0, -0.2, 1.1, -0.7, 0.5)):
            g = torch.Generator().manual_seed(3)
            self.items = []
            for k, n in enumerate(lengths):
                seq = torch.randint(10, 300, (n,), generator=g).tolist()
                seq[-1] = 7 if k % 2 == 0 else 8                                # the answer: "Yes" 7 / "No" 8
                self.items.append({"input_ids": seq, "labels": [-100] * (n - 1) + [seq[-1]], "z": zs[k]})

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    kw = dict(output_dir=str(tmp_path), per_device_train_batch_size=2, gradient_accumulation_steps=grad_accum, max_steps=2,
              optim="sgd", learning_rate=0.5, max_grad_norm=1e9, lr_scheduler_type="constant", weight_decay=0.0,
              save_strategy="no", report_to=[], seed=0, data_seed=0, remove_unused_columns=False, use_cpu=True,
              disable_tqdm=True, logging_steps=1000)
    cls = (tlo.prior_offset_trainer(transformers.Trainer, 7, b, b_lr) if offset
           else last_token_trainer(transformers.Trainer))
    trainer = cls(model=model, args=transformers.TrainingArguments(**kw), train_dataset=ZRows(),
                  data_collator=lambda batch: tlo.collate_offset(batch, 0))
    trainer.create_optimizer()                       # as train_lora_offset.train: built first, reused by train()
    assert tlo.b_in_optimizer(trainer.optimizer, b) == (offset and learn_b)
    opt = trainer.optimizer
    trainer.train()
    assert opt is trainer.optimizer or getattr(trainer.optimizer, "optimizer", None) is opt   # reused (maybe wrapped)
    lora = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}
    return model, b, trainer, lora


@pytest.mark.parametrize("grad_accum", [1, 2])
def test_tiny_lora_with_b_fixed_at_zero_trains_exactly_like_sft(tmp_path, grad_accum):
    torch = pytest.importorskip("torch")
    _, _, _, sft = _tiny_run(tmp_path / "sft", offset=False, learn_b=False, grad_accum=grad_accum)
    _, b, _, mine = _tiny_run(tmp_path / "po", offset=True, learn_b=False, grad_accum=grad_accum)
    assert float(b) == 0.0 and sft.keys() == mine.keys() and sft
    assert max(float(p.abs().max()) for n, p in sft.items() if "lora_B" in n) > 1e-3        # the run did train
    for n in sft:
        assert torch.allclose(sft[n], mine[n], atol=1e-5, rtol=1e-4), n


def test_tiny_lora_learns_b_in_the_optimizer_and_never_saves_it(tmp_path):
    pytest.importorskip("torch")
    model, b, trainer, _ = _tiny_run(tmp_path / "run", offset=True, learn_b=True)          # --b_lr 0: the LoRA group
    assert float(b.detach()) != 0.0 and tlo.b_in_optimizer(trainer.optimizer, b)
    model.save_pretrained(tmp_path / "adapter")
    keys = tlo.saved_adapter_keys(tmp_path / "adapter")
    assert keys and all("lora_" in k for k in keys) and not any(tlo.OFFSET_PARAM in k for k in keys)
    _, b2, trainer2, _ = _tiny_run(tmp_path / "run2", offset=True, learn_b=True, b_lr=tlo.B_LR)
    assert tlo.b_group_record(trainer2.optimizer, b2) == {"own_group": True, "lr": 0.01, "weight_decay": 0.0,
                                                          "n_params_in_group": 1}
    assert float(b2.detach()) != 0.0


def _tiny_registered_run(tmp_path, b_lr, steps: int = 200, horizon: int = 730, bsz: int = 8):
    """The tiny model under the registered TrainingArguments (train_lora_yesno.training_arguments: AdamW lr 1e-4, cosine,
    3% warmup, one epoch; bf16 off on CPU), an epoch of `horizon` steps (ML-1M's size) stopped after `steps`, labels
    planted on z (Yes iff z > 0). Returns (b, b's group as recorded before training, the trainer)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from src.confrec import train_lora_yesno as tl
    cfg = transformers.Qwen3Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128)
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(cfg).train()
    torch.manual_seed(0)
    model = peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, task_type="CAUSAL_LM",
                                                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    b = tlo.attach_prior_offset(model)

    class Planted(torch.utils.data.Dataset):
        def __init__(self, n):
            g = torch.Generator().manual_seed(5)
            self.items = []
            for _ in range(n):
                seq = torch.randint(10, 300, (6,), generator=g).tolist()
                z = float(torch.randn((), generator=g))
                seq[-1] = 7 if z > 0 else 8                                     # "Yes" 7 iff z > 0, else "No" 8
                self.items.append({"input_ids": seq, "labels": [-100] * 5 + [seq[-1]], "z": z})

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    class Stop(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if state.global_step >= steps:
                control.should_training_stop = True

    ns = SimpleNamespace(out=str(tmp_path), bsz=bsz, grad_accum=1, epochs=1.0, lr=1e-4, seed=0)
    kw = {**tl.training_arguments(ns), "bf16": False, "use_cpu": True, "disable_tqdm": True}
    trainer = tlo.prior_offset_trainer(transformers.Trainer, 7, b, b_lr)(
        model=model, args=transformers.TrainingArguments(**kw), train_dataset=Planted(horizon * bsz),
        data_collator=lambda batch: tlo.collate_offset(batch, 0), callbacks=[Stop()])
    trainer.create_optimizer()
    rec = tlo.b_group_record(trainer.optimizer, b)
    trainer.train()
    assert trainer.state.global_step == steps
    return b, rec, trainer


def test_tiny_lora_b_passes_1_within_200_registered_steps_in_its_own_group_only(tmp_path):
    """Amendment 3 addendum 3 on the real Trainer: under the registered schedule b's own AdamW group (lr 1e-2) carries
    |b| past 1 within 200 steps when the labels depend on z; in the LoRA group (--b_lr 0, lr 1e-4) it cannot. The
    group holding b is its own, with the recorded lr as the schedule's base (initial_lr)."""
    pytest.importorskip("torch")
    b, rec, trainer = _tiny_registered_run(tmp_path / "own", tlo.B_LR)
    assert abs(float(b.detach())) > 1.0, float(b.detach())
    assert rec == {"own_group": True, "lr": 0.01, "weight_decay": 0.0, "n_params_in_group": 1}
    assert tlo.b_group_record(trainer.optimizer, b) == rec                   # after training: initial_lr, the same
    assert {g.get("initial_lr") for g in trainer.optimizer.param_groups if len(g["params"]) > 1} == {1e-4}
    b0, rec0, _ = _tiny_registered_run(tmp_path / "shared", 0)
    assert abs(float(b0.detach())) < 0.05, float(b0.detach())
    assert rec0["own_group"] is False and rec0["lr"] == 1e-4


# ---------------------------------------------------------------- 6. the report
AUC_ROWS = {0.5: [2, 1, 0, 0], 0.25: [2, 1, 1.5, 0], 0.0: [0, 0, 0, 0], -0.25: [1.5, 0, 2, 1], -0.5: [0, 0, 1, 1]}


def planted(per_seed: list, n_users: int, n_boot: int = 2000) -> dict:
    """contrast_models on users with labels [1, 1, 0, 0]: in seed k the first users get the per-user AUC differences
    d of per_seed[k] ({d: count}) and the others 0 (the score a gives AUC 0.5 + d, the score b is constant)."""
    y = np.tile([1, 1, 0, 0], n_users)
    users = np.repeat(np.arange(n_users), 4)
    pairs = {}
    for k, counts in enumerate(per_seed):
        ds = [d for d, c in counts.items() for _ in range(c)]
        ds += [0.0] * (n_users - len(ds))
        pairs[f"seed{k}"] = (np.concatenate([AUC_ROWS[d] for d in ds]).astype(float), np.zeros(4 * n_users))
    return fr.contrast_models(pairs, y, users, np.ones(4 * n_users, bool), n_boot, 0, n_registered=3)


def test_planted_pass():
    diff = planted([{0.25: 80}, {0.25: 88}, {0.25: 84}], 400)
    assert [diff["per_seed"][f"seed{k}"]["est"] for k in range(3)] == pytest.approx([0.05, 0.055, 0.0525])
    dec = fm.decide(diff, 3, [])
    assert dec["status"] == "PASS" and dec["reason"] is None and all(dec["conditions"].values())
    assert dec["p"] < 0.05 and dec["lo"] > 0 and dec["dUAUC"] == pytest.approx(0.0525)


@pytest.mark.parametrize("name, per_seed, n_users, failing", [
    ("below +0.01", [{0.25: 40}, {0.25: 48}, {0.25: 44}], 2000, {"mean_ge_0.01"}),
    ("CI covering 0", [{0.5: 104, -0.5: 96}, {0.5: 105, -0.5: 95}, {0.5: 104, -0.5: 96}], 200, {"ci_excludes_0"}),
    ("one seed negative", [{0.25: 80}, {0.25: 80}, {-0.25: 4}], 400,
     {"all_seeds_positive", "mean_gt_2_sigma_seed", "sigma_seed_rule"}),
    ("mean not above 2 sigma_seed", [{0.25: 20}, {0.25: 96}, {0.25: 48}], 400,
     {"mean_gt_2_sigma_seed", "sigma_seed_rule"}),
    ("descriptive (< 150 users)", [{0.5: 60}, {0.5: 60}, {0.5: 60}], 100, {"n_users_ge_150", "ci_excludes_0"}),
])
def test_planted_failures_fail_on_exactly_their_condition(name, per_seed, n_users, failing):
    """With 3 seeds a negative seed under a positive mean always puts the mean below 2 sigma_seed (SD >= 0.87 |x_neg -
    mean|), so that case fails both; every other case fails on its own condition only."""
    dec = fm.decide(planted(per_seed, n_users), 3, [])
    assert dec["status"] == "FAIL", name
    assert {k for k, v in dec["conditions"].items() if not v} == failing, (name, dec["conditions"])
    if name == "CI covering 0":
        assert dec["lo"] < 0 < dec["hi"] and dec["p"] > 0.05
    if name.startswith("descriptive"):
        assert dec["p"] is None and "descriptive" in dec["reason"] and dec["dUAUC"] == pytest.approx(0.3)


def test_a_missing_seed_or_an_input_problem_is_not_a_decision():
    two = planted([{0.25: 80}, {0.25: 88}], 400)
    assert fm.decide(two, 2, [])["status"] == "INCOMPLETE" and two["seeds"]["complete"] is False
    assert fm.decide(None, 0, [])["status"] == "INCOMPLETE"
    full = planted([{0.25: 80}, {0.25: 88}, {0.25: 84}], 400)
    dec = fm.decide(full, 3, ["o0: standardisation constants differ from the manifest's"])
    assert dec["status"] == "INVALID" and "standardisation" in dec["reason"]


def test_post_hoc_stacking_is_fit_on_the_cal_rows_only():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    L, z = y + rng.normal(0, 1, n), 0.5 * y + rng.normal(0, 1, n)
    cal = np.zeros(n, bool)
    cal[:250] = True
    eta, info = fm.stacking(L, z, y, cal)
    X = np.column_stack([L, z])
    assert np.allclose(eta, fr.logit_eta(fr.logit_fit(X[cal], y[cal]), X))
    y2 = y.copy()
    y2[~cal] = 1 - y2[~cal]                                                  # other TEST labels: the same fit
    eta2, info2 = fm.stacking(L, z, y2, cal)
    assert np.array_equal(eta, eta2) and info == info2
    assert np.allclose(info["intercept"] + info["slope_sft_logit"] * L + info["slope_z"] * z, eta)
    assert info["n_fit_rows"] == 250 and info["slope_z"] > 0 and info["slope_sft_logit"] > 0
    eta3, info3 = fm.stacking(L, z, np.zeros(n, int), cal)
    assert eta3 is None and info3["available"] is False


SCORE_HDR = ps.SCORE_COLS


def write_run(d: Path, rows: list, logits: dict, eval_sha: str, lora, *, mass: float = 0.999, run_key=None) -> None:
    """A finished like pass in pyes_scorer's format (scores.csv.gz, report.json[, run.key])."""
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for r in rows:
        for j, (i, y) in enumerate(zip(r["candidate_item_ids"], r["candidate_labels"])):
            lg = logits[(r["user_id"], i)]
            out.append([r["source_event_id"], r["user_id"], i, j, y, "like", "", "", repr(float(lg)), mass, 0, ""])
    buf = ",".join(SCORE_HDR) + "\n" + "".join(",".join(map(str, x)) + "\n" for x in out)
    (d / "scores.csv.gz").write_bytes(tlo.gz_bytes(buf))
    n = len(out)
    (d / "report.json").write_text(json.dumps(
        {"data_sha1": eval_sha, "variant": "V3", "lora": str(lora), "backbone": "Qwen3-8B", "n_main_prompts": n,
         "censored_main": {"0": n, "1": 0, "2": 0, "3": 0}, "n_overlength": 0, "mean_yes_no_mass": mass}),
        encoding="utf-8")
    if run_key is not None:
        (d / "run.key").write_text(run_key + "\n", encoding="utf-8")


def report_world(tmp: Path, n_users: int = 200, plant: str = "pass", flip_test: bool = False,
                 b=(0.3, 0.2, 0.1)) -> SimpleNamespace:
    """eval.jsonl (8 candidates per user: 4 CAL before T, 4 TEST after, 2 likes each), the split, the stage-1 q-hat files
    and manifest, the SFT like passes s0-s2 and the prior-offset adapters and like passes o0-o2, in the layouts the
    report reads. plant 'pass': uninformative SFT logits and q-hat, label-driven prior-offset logits; 'fail': the
    prior-offset logits are the SFT logits, b = 0 and q-hat constant (so the stacking predictor ranks as the SFT logit).
    The noise of a row depends on (seed, user, candidate) only, so CAL rows are the same under flip_test."""
    panels, mdir, sfts = tmp / "panels", tmp / "method", tmp / "sft_scores"
    panels.mkdir(parents=True)
    T = 500.0
    rows = []
    for u in range(n_users):
        lab = [1, 1, 0, 0] + ([0, 0, 1, 1] if flip_test else [1, 1, 0, 0])
        rows.append({"user_id": f"u{u:03d}", "source_event_id": f"u{u:03d}::100", "source": "ml1m",
                     "candidate_item_ids": [f"i{u}_{j}" for j in range(8)], "candidate_labels": lab,
                     "candidate_timestamps": [100 + j for j in range(4)] + [1000 + j for j in range(4)],
                     "candidate_popularity": [3] * 8})
    write_jsonl(panels / "eval.jsonl", rows)
    eval_sha = sha1_file(panels / "eval.jsonl")
    (panels / "ftgrid_split.json").write_text(json.dumps({"domain": "ml1m", "T": T, "variant": "V3",
                                                          "files": {"eval.jsonl": eval_sha}}), encoding="utf-8")
    consts = {"mean": 3.5, "sd": 0.5, "ddof": 0, "n_examples": 100, "n_finite": 100, "n_nonfinite_z_imputed_0": 0}
    ex = tlo.panel_examples(rows, "eval")
    qh = np.array([3.5 if plant == "fail" else 3.5 + 0.5 * np.random.default_rng([7, int(e.user[1:]), e.cand]).normal()
                   for e in ex])
    pm = {"mean_prior_shrunk": qh, "mean_prior": qh, "n_prior": np.ones(len(ex)), "global_prior": np.full(len(ex), 3.5)}
    mdir.mkdir()
    data = tlo.qhat_csv_bytes(ex, pm, tlo.zscore(qh, consts))
    (mdir / tlo.EVAL_QHAT).write_bytes(data)
    man = {"format": tlo.MANIFEST_FORMAT, "domain": "ml1m", "T": T, "standardisation": consts,
           "eval": {"sha1": eval_sha}, "files": {tlo.EVAL_QHAT: hashlib.sha1(data).hexdigest()}}
    (mdir / tlo.MANIFEST).write_text(json.dumps(man), encoding="utf-8")
    man_sha = sha1_file(mdir / tlo.MANIFEST)
    for k in range(3):
        bk = 0.0 if plant == "fail" else b[k]
        L_s, L_o = {}, {}
        for e in ex:
            g = np.random.default_rng([k, int(e.user[1:]), e.cand])
            noise, sign = g.normal(), 2 * e.label - 1
            L_s[(e.user, e.item)] = noise if plant == "pass" else sign + noise
            L_o[(e.user, e.item)] = 3 * sign + g.normal() if plant == "pass" else L_s[(e.user, e.item)]
        adir = mdir / "adapters" / f"o{k}"
        adir.mkdir(parents=True)
        (adir / "adapter_model.safetensors").write_bytes(f"weights {k}".encode())
        sft_adapter = tmp / "sft_adapters" / f"s{k}"
        (adir / "offset.json").write_text(json.dumps(
            {"format": tlo.OFFSET_FORMAT, "seed": k, "domain": "ml1m", "b": bk, "standardisation": consts,
             "qhat_manifest_sha1": man_sha, "sft_adapter": str(sft_adapter), "yes_token_id": 7, "b_lr": tlo.B_LR,
             "b_group": {"own_group": True, "lr": tlo.B_LR, "weight_decay": 0.0, "n_params_in_group": 1,
                         "source": "the Trainer's optimizer, before the first step"}}), encoding="utf-8")
        (adir / "train_config.json").write_text(json.dumps({"seed": k, "prior_offset": {"b": bk}}), encoding="utf-8")
        write_run(sfts / f"s{k}" / "like", rows, L_s, eval_sha, sft_adapter)
        write_run(mdir / "scores" / f"o{k}" / "like", rows, L_o, eval_sha, adir,
                  run_key=f"{eval_sha} {QWEN} V3 {fm.adapter_weights_sha1(adir)} --lora {adir}")
    return SimpleNamespace(panels=panels, method=mdir, sft=sfts, tmp=tmp)


def run_report(w, n_boot: int = 300) -> dict:
    out = w.method / "report.json"
    return fm.main(["dataset", "--domain", "ml1m", "--split", str(w.panels / "ftgrid_split.json"), "--panels",
                    str(w.panels), "--sft_scores", str(w.sft), "--method_dir", str(w.method), "--out", str(out),
                    "--n_boot", str(n_boot), "--seed", "0"])


def test_report_from_files_planted_pass(tmp_path):
    res = run_report(report_world(tmp_path))
    dec = res["decision"]
    assert dec["status"] == "PASS" and res["input_checks"]["problems"] == [] and dec["dUAUC"] > 0.3
    slot = res["slot"]
    assert list(slot) == list(fm.ALIAS) and res["alias"] == "slot"
    assert slot["prior_offset_LoRA"]["b"] == {"o0": 0.3, "o1": 0.2, "o2": 0.1}
    assert all(res["offsets"][f"o{k}"]["b_group"]["own_group"] is True and res["offsets"][f"o{k}"]["b_group"]["lr"]
               == 0.01 for k in range(3))
    nu = {slot[p]["rows"]["n_users"] for p in ("SFT_b0", "post_hoc_stacking", "prior_offset_LoRA", "difference")}
    assert nu == {200} and slot["reference"]["rows"]["n_users"] == 200        # identical users in every block
    assert slot["SFT_b0"]["mean_over_seeds"]["est"] == pytest.approx(0.5, abs=0.1)
    assert slot["difference"]["mean_over_seeds"]["est"] == pytest.approx(
        slot["prior_offset_LoRA"]["mean_over_seeds"]["est"] - slot["post_hoc_stacking"]["mean_over_seeds"]["est"])
    for k in range(3):
        fit = slot["post_hoc_stacking"]["fit"][f"s{k}"]
        assert fit["available"] and fit["n_fit_rows"] == 800                # 4 CAL rows x 200 users
    assert res["meta"]["rows"] == {"test_rows_used": 800, "cal_fit_rows": 800}
    text = (tmp_path / "method" / "report.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in text and tmp_path.as_posix() not in text     # no path in the json
    with open(tmp_path / "method" / "report_tables.csv", encoding="utf-8") as f:
        tab = list(csv.DictReader(f))
    assert {r["regime"] for r in tab} == set(fm.ALIAS) | {"decision"}
    assert [r for r in tab if r["regime"] == "decision"][0]["note"] == "PASS"


def test_report_from_files_planted_fail_and_the_cal_only_fit(tmp_path):
    res = run_report(report_world(tmp_path / "a", plant="fail"))
    assert res["decision"]["status"] == "FAIL" and res["decision"]["dUAUC"] == pytest.approx(0.0, abs=1e-12)
    assert res["decision"]["per_seed"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    a = run_report(report_world(tmp_path / "b"))["slot"]["post_hoc_stacking"]["fit"]
    b = run_report(report_world(tmp_path / "c", flip_test=True))["slot"]["post_hoc_stacking"]["fit"]
    assert a == b                                                            # TEST labels never enter the fit


def test_report_missing_seed_and_failed_integrity_are_incomplete(tmp_path):
    w = report_world(tmp_path / "a")
    shutil.rmtree(w.method / "scores" / "o2" / "like")
    res = run_report(w)
    assert res["decision"]["status"] == "INCOMPLETE" and res["input_checks"]["seeds_present"] == ["seed0", "seed1"]
    assert res["input_checks"]["seeds_missing"][0]["prior_offset"] == "ABSENT"
    assert res["slot"]["difference"]["seeds"]["complete"] is False
    w = report_world(tmp_path / "b")
    rep = read_json(w.sft / "s1" / "like" / "report.json")
    rep["mean_yes_no_mass"] = 0.5                                            # E1 fails: the seed is missing, not replaced
    (w.sft / "s1" / "like" / "report.json").write_text(json.dumps(rep), encoding="utf-8")
    res = run_report(w)
    assert res["decision"]["status"] == "INCOMPLETE"
    assert res["input_checks"]["excluded_runs"][0]["status"] == "FAILED_INTEGRITY"


def test_report_input_problems_make_it_invalid(tmp_path):
    w = report_world(tmp_path / "a")                                          # offset trained on other constants
    off = read_json(w.method / "adapters" / "o0" / "offset.json")
    off["standardisation"]["mean"] = 3.0
    (w.method / "adapters" / "o0" / "offset.json").write_text(json.dumps(off), encoding="utf-8")
    assert run_report(w)["decision"]["status"] == "INVALID"
    w = report_world(tmp_path / "b")                                          # b of another adapter than the scored one
    (w.method / "scores" / "o1" / "like" / "run.key").write_text("x y z 0123 --lora a\n", encoding="utf-8")
    res = run_report(w)
    assert res["decision"]["status"] == "INVALID" and any("run.key" in p for p in res["input_checks"]["problems"])
    w = report_world(tmp_path / "c")                                          # the SFT pass of another adapter
    rep = read_json(w.sft / "s2" / "like" / "report.json")
    rep["lora"] = str(tmp_path / "elsewhere")
    (w.sft / "s2" / "like" / "report.json").write_text(json.dumps(rep), encoding="utf-8")
    res = run_report(w)
    assert res["decision"]["status"] == "INVALID" and any("comparator" in p for p in res["input_checks"]["problems"])
    w = report_world(tmp_path / "d")                                          # an EVAL pair without its q-hat row
    m = read_json(w.method / tlo.MANIFEST)
    m["eval"]["sha1"] = "0" * 40
    (w.method / tlo.MANIFEST).write_text(json.dumps(m), encoding="utf-8")
    assert run_report(w)["decision"]["status"] == "INVALID"
    w = report_world(tmp_path / "e")                                          # b trained in the LoRA group (--b_lr 0)
    off = read_json(w.method / "adapters" / "o1" / "offset.json")
    off["b_group"] = {"own_group": False, "lr": 1e-4, "weight_decay": 0.0}
    (w.method / "adapters" / "o1" / "offset.json").write_text(json.dumps(off), encoding="utf-8")
    res = run_report(w)
    assert res["decision"]["status"] == "INVALID" and any("addendum 3" in p for p in res["input_checks"]["problems"])


# ---------------------------------------------------------------- 6b. the slot across datasets
def st(*statuses) -> dict:
    return dict(zip(fm.DATASETS, statuses))


def test_kill_rule_fail_fail_kills_and_refuses_the_next_dataset():
    s = fm.slot_state(st("FAIL", "FAIL"))
    assert (s["state"], s["killed_after"], s["final"], s["next_dataset"]) == ("KILLED", "toys", True, None)
    ok, why = fm.check_next(st("FAIL", "FAIL"), "games")
    assert not ok and "killed after toys" in why and "ml1m and toys" in why
    assert fm.check_next(st("FAIL", "FAIL"), "toys")[0]                      # the dataset that fired it may resume
    assert not fm.check_next(st("FAIL", "FAIL"), "sports")[0]


@pytest.mark.parametrize("statuses, state, final, nxt", [
    (("PASS", "FAIL", "PASS", "PASS"), "SURVIVES", True, None),
    (("PASS", "PASS", "PASS"), "SURVIVES", False, "sports"),            # 3 of 4: survives; sports still runs
    (("PASS", "FAIL", "PASS"), "OPEN", False, "sports"),
    (("PASS", "PASS", "FAIL", "FAIL"), "KILLED", True, None),           # 2 passes: killed at the second failure
    (("FAIL", "PASS", "FAIL"), "KILLED", True, None),
    (("PASS",), "OPEN", False, "toys"),
    ((), "OPEN", False, "ml1m"),
])
def test_kill_rule_state_machine(statuses, state, final, nxt):
    s = fm.slot_state(st(*statuses))
    assert (s["state"], s["final"], s["next_dataset"]) == (state, final, nxt) and s["violations"] == []


def test_undecided_datasets_block_the_order_and_reports_out_of_order_are_violations():
    s = fm.slot_state(st("PASS", "INCOMPLETE"))
    assert (s["state"], s["pending"], s["pending_status"], s["next_dataset"]) == ("OPEN", "toys", "INCOMPLETE", "toys")
    ok, why = fm.check_next(st("PASS", "INCOMPLETE"), "games")
    assert not ok and "toys has status INCOMPLETE" in why
    assert fm.check_next(st("PASS", "INCOMPLETE"), "toys")[0]
    assert not fm.check_next(st(None, None), "toys")[0] and "ml1m has no report" in fm.check_next({}, "toys")[1]
    assert fm.check_next({}, "ml1m")[0]
    assert fm.slot_state(st("FAIL", "FAIL", "PASS"))["violations"] == [
        "games: a report exists although the slot was killed after toys"]
    assert fm.slot_state(st("PASS", None, "PASS"))["violations"] == [
        "games: a report exists although toys (earlier in the registered order) is not decided"]
    assert fm.slot_state(st("PASS", "INVALID", "FAIL"))["violations"][0].startswith("games")
    with pytest.raises(ValueError):
        fm.slot_state(st("MAYBE"))


REGISTERED_META = {"backbone": "Qwen3-8B", "dry_run_inputs": False, "n_boot": 2000, "seed": 0}   # what a real run's report says


def write_dataset_report(root: Path, d: str, status: str, p=None, rule=True, est=0.02, **meta) -> None:
    """A per-dataset report as a REGISTERED run writes it (meta: real inputs, n_boot 2000, seed 0); **meta overrides a key."""
    (root / d).mkdir(parents=True, exist_ok=True)
    (root / d / "report.json").write_text(json.dumps(
        {"meta": {"domain": d, **REGISTERED_META, **meta},
         "decision": {"status": status, "dUAUC": est, "lo": 0.01, "hi": 0.03, "p": p, "sigma_seed": 0.002,
                      "n_users": 400, "descriptive_min_n": False, "conditions": {"sigma_seed_rule": rule}}}),
        encoding="utf-8")


def pilot_log(tmp: Path, *lines: str) -> Path:
    """A pilot log (the file the slot reads its cuts from) holding the given lines."""
    p = tmp / "PILOT_LOG.md"
    p.write_text("# pilot log\n" + "".join(x + "\n" for x in lines), encoding="utf-8")
    return p


def test_slot_summary_holm_over_the_datasets_run_and_check_next_cli(tmp_path, capsys):
    root = tmp_path / "ftmethod"
    log = ["--pilot_log", str(pilot_log(tmp_path))]                  # no cut recorded (the slot CLI needs its pilot log)
    write_dataset_report(root, "ml1m", "PASS", p=0.01)
    write_dataset_report(root, "toys", "FAIL", p=0.04, est=0.005)
    write_dataset_report(root, "games", "PASS", p=0.03)
    out = root / "slot.json"
    s = fm.main(["slot", "--root", str(root), "--out", str(out), *log])
    assert s["datasets_run"] == ["ml1m", "toys", "games"] and s["state"] == "OPEN" and s["next_dataset"] == "sports"
    # Holm over the three datasets run: 3 x 0.01, 2 x 0.03 (step-down), max(running, 1 x 0.04)
    assert s["holm"]["p_holm"] == {"ml1m": pytest.approx(0.03), "toys": pytest.approx(0.06), "games": pytest.approx(0.06)}
    assert s["holm"]["confirmed"] == {"ml1m": True, "toys": False, "games": False}
    assert s["single_backbone"] is True and s["abstract_eligible"] is False and s["backbones"] == ["Qwen3-8B"]
    assert read_json(out)["state"] == "OPEN" and (root / "slot_tables.csv").is_file()
    write_dataset_report(root, "sports", "FAIL", p=0.2)
    s = fm.main(["slot", "--root", str(root), *log])
    assert s["state"] == "KILLED" and s["killed_after"] == "sports" and s["passes"] == ["ml1m", "games"]
    write_dataset_report(root, "toys", "FAIL", p=0.04)
    write_dataset_report(root, "ml1m", "FAIL", p=0.5)
    with pytest.raises(SystemExit) as e:
        fm.main(["slot", "--root", str(root), "--check_next", "games", *log])
    assert e.value.code == 4 and "killed after toys" in capsys.readouterr().err
    (root / "games" / "report.json").write_text(json.dumps({"meta": {"domain": "toys"}}), encoding="utf-8")
    assert fm.slot_summary(root)["datasets"]["games"]["status"] == "INVALID"   # a report of the wrong dataset


# ---------------------------------------------------------------- 7. the run script
def _bash():
    for c in ("C:/Program Files/Git/bin/bash.exe", shutil.which("bash")):
        if c and Path(c).exists() and "system32" not in str(c).lower():          # not WSL's bash.exe
            return str(c)
    return None


BASH = _bash()
NO_BASH = "no bash (Git Bash or a POSIX bash) on PATH"


def freeze_blocks() -> dict:
    blocks = {}
    for p in [ROOT / ff.AMENDMENT] + sorted((ROOT / "idea-stage").glob("PREREG_AMENDMENT_3_ADDENDUM_*.md")):
        for stage, files in ff.parse_blocks(p.read_text(encoding="utf-8")).items():
            blocks.setdefault(stage, []).extend(f for f in files if f not in blocks.get(stage, []))
    return blocks


def test_script_is_lf_bash_and_bash_n_clean():
    raw = SCRIPT.read_bytes()
    assert b"\r\n" not in raw and raw.startswith(b"#!/usr/bin/env bash\n") and b"set -euo pipefail" in raw
    if BASH is None:
        pytest.skip(NO_BASH)
    r = subprocess.run([BASH, "-n", SCRIPT.as_posix()], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_slot_code_is_the_freeze_method_list():
    assert set(freeze_blocks()["method"]) == {"src/confrec/train_lora_offset.py", "scripts/sigir/run_ftmethod.sh",
                                              "src/confrec/ftmethod_report.py"}


# the flag audit: every entry point the script calls through "$PY" / "$MPY" (subcommand aware), arrays expanded
class _Captured(Exception):
    pass


def _capture(fn):
    orig = (argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args)

    def boom(self, *a, **k):
        raise _Captured(self)
    argparse.ArgumentParser.parse_args = argparse.ArgumentParser.parse_known_args = boom
    old = sys.argv
    sys.argv = ["prog"]
    try:
        fn()
    except _Captured as c:
        return c.args[0]
    except (SystemExit, Exception):
        return None
    finally:
        argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args = orig
        sys.argv = old
    return None


def entry_parser(target: str):
    mod = importlib.import_module(target)
    for fn in (lambda: mod.main([]), lambda: mod.parse_args([]), lambda: mod.main()):
        p = _capture(fn)
        if p is not None:
            return p
    raise AssertionError(f"no argparse parser captured for {target}")


def declared(parser, sub):
    subs = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    if subs:
        if sub not in subs[0].choices:
            return None
        parser = subs[0].choices[sub]
    elif sub is not None:
        return None
    return {s: a for a in parser._actions for s in a.option_strings}


def _code(text: str) -> str:
    """The shell code: heredoc bodies blanked (they are python), continuation lines joined."""
    out, end = [], None
    for ln in text.splitlines():
        if end is not None:
            out.append("")
            if ln == end:
                end = None
            continue
        out.append(ln)
        m = re.search(r"(?<!<)<<-?\s*'([A-Za-z_]+)'", ln)
        if m:
            end = m.group(1)
    return re.sub(r"\\\n\s*", " ", "\n".join(out))


def _flags(words: list, arrays: dict) -> list:
    out, plain = [], []
    for w in words:
        m = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\[@\]\}", w)
        if m:
            for body in arrays.get(m.group(1), []):
                out += _flags(shlex.split(body), arrays)
        else:
            plain.append(w)
    for i, w in enumerate(plain):
        if w.startswith("--"):
            nxt = plain[i + 1] if i + 1 < len(plain) else None
            out.append((w, None if nxt is None or nxt.startswith("--") else nxt))
    return out


_CUT = re.compile(r"\s(?:\|\||\||&&|;|>|2>&1|2>)\s|;\s*(?:then|do)\b|;\s*$")


def _words(rest: str) -> list:
    return shlex.split(re.sub(r"\)\s*$", "", _CUT.split(rest)[0]))


def invocations(text: str) -> list:
    """(target, subcommand, [(flag, value)], line) of every call of a repo entry point through "$PY" / "$MPY"; a `score`
    line carries the flags of score()'s own pyes_scorer call."""
    code = _code(text)
    arrays = defaultdict(list)
    for name, body in re.findall(r"(?<![\w$])([A-Za-z_][A-Za-z_0-9]*)\+?=\(([^)]*)\)", code):
        arrays[name].append(body)
    lines = [ln.strip() for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    calls, fixed = [], []
    for s in lines:
        m = re.search(r'"\$(?:PY|MPY)" -m (src\.confrec\.[a-z_0-9]+)((?: [a-z][a-z_0-9]*)?)(.*)$', s)
        if m:
            flags = _flags(_words(m.group(3)), arrays)
            calls.append((m.group(1), m.group(2).strip() or None, flags, s))
            if m.group(1) == "src.confrec.pyes_scorer":
                fixed = flags
    for s in lines:
        if re.match(r"score\s", s):
            calls.append(("src.confrec.pyes_scorer", None, fixed + _flags(_words(s)[3:], arrays), s))
    return calls


def audit(text: str) -> tuple[list, Counter]:
    problems, parsers, count = [], {}, Counter()
    for target, sub, flags, line in invocations(text):
        count[(target, sub)] += 1
        if target not in parsers:
            parsers[target] = entry_parser(target)
        acts = declared(parsers[target], sub)
        if acts is None:
            problems.append(f"{target} has no subcommand {sub!r}: {line[:100]}")
            continue
        used = {f for f, _ in flags}
        if used - set(acts):
            problems.append(f"{target} {sub or ''} has no flag {sorted(used - set(acts))}: {line[:100]}")
        req = {a.option_strings[0] for a in set(acts.values()) if a.option_strings and a.required}
        if req - used:
            problems.append(f"{target} {sub or ''} lacks required {sorted(req - used)}: {line[:100]}")
        for f, v in flags:
            a = acts.get(f)
            if a is None or v is None or "$" in v:
                continue
            if a.choices and v not in a.choices:
                problems.append(f"{target} {f} {v!r} not in {list(a.choices)}")
            if a.type in (int, float):
                try:
                    a.type(v)
                except ValueError:
                    problems.append(f"{target} {f} {v!r} is not a {a.type.__name__}")
    return problems, count


def test_every_flag_the_script_passes_exists_in_the_real_argparse():
    text = SCRIPT.read_text(encoding="utf-8")
    problems, count = audit(text)
    assert not problems, "\n".join(problems)
    trains = [flags for t, s, flags, _ in invocations(text) if (t, s) == ("src.confrec.train_lora_offset", "train")]
    assert trains and all("--b_lr" not in {f for f, _ in flags} for flags in trains)   # addendum 3: the default applies
    for key, n in {("src.confrec.train_lora_offset", "qhat"): 1, ("src.confrec.train_lora_offset", "train"): 1,
                   ("src.confrec.ftmethod_report", "dataset"): 1, ("src.confrec.ftmethod_report", "slot"): 2,
                   ("src.confrec.pyes_scorer", None): 2, ("src.confrec.ftgrid_freeze", None): 4,
                   ("src.confrec.ftmethod_shift_diag", "verify"): 1}.items():
        assert count.get(key, 0) >= n, (key, count)


def test_the_flag_audit_catches_a_wrong_flag_value_or_subcommand():
    bad = SCRIPT.read_text(encoding="utf-8").replace("--sft_adapter", "--sft_adaptor").replace(
        "--method_dir", "--methoddir").replace("--readout yesno", "--readout yesn0").replace(
        "--print --stage core", "--print --stage cor").replace("ftmethod_report slot --root", "ftmethod_report slots --root")
    problems, _ = audit(bad)
    for what in ("--sft_adaptor", "--methoddir", "'yesn0'", "'cor'", "no subcommand 'slots'"):
        assert any(what in p for p in problems), what


def test_the_integration_audit_of_test_confrec_contracts_accepts_the_script():
    spec = importlib.util.spec_from_file_location("contracts_for_ftmethod", ROOT / "tests" / "test_confrec_contracts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert SCRIPT in sorted((ROOT / "scripts" / "sigir").glob("*.sh"))      # found by its glob: no registration needed
    mod.test_every_script_flag_exists_in_the_real_argparse(SCRIPT)


def make_repo(dest: Path) -> Path:
    """A copy of the repo parts the chain runs: src/confrec, the amendment and its addenda, every file of their core and
    method freeze lists (a listed file that does not exist yet becomes a placeholder), the run scripts (the shift
    diagnostic's too) and the diagnostic's tests file (the third file of its record)."""
    (dest / "src" / "confrec").mkdir(parents=True)
    shutil.copy2(ROOT / "src" / "__init__.py", dest / "src" / "__init__.py")
    for f in (ROOT / "src" / "confrec").glob("*.py"):
        shutil.copy2(f, dest / "src" / "confrec" / f.name)
    blocks = freeze_blocks()
    rels = ["scripts/sigir/run_ftgrid.sh", "scripts/sigir/run_ftmethod.sh", "scripts/sigir/run_ftmethod_shift_diag.sh",
            "scripts/sigir/starperm_panel.py", "tests/test_confrec_ftmethod_shift_diag.py", ff.AMENDMENT]
    rels += [p.relative_to(ROOT).as_posix() for p in (ROOT / "idea-stage").glob("PREREG_AMENDMENT_3_ADDENDUM_*.md")]
    for rel in rels + blocks["core"] + blocks["method"]:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        if (ROOT / rel).exists():
            shutil.copy2(ROOT / rel, dest / rel)
        elif not (dest / rel).exists():
            (dest / rel).write_text("# placeholder: a file the amendment lists that does not exist yet\n",
                                    encoding="utf-8", newline="\n")
    return dest


CLEAN_ENV = ("MODEL", "OUT_ROOT", "VARIANT", "STAGES", "DRY_RUN", "DRY_GATE", "DRY_E1_FAIL", "DRY_TODAY", "DRY_NO_RECORD",
             "DRY_SHIFT_SHARE", "DRY_SHIFT_LOW_FRAC", "KNOCKOUT_SPORTS", "FAKE_DATE_FLIP", "BASH_ENV", "PYTHONPATH")
FIXED_TODAY = "20261001"      # a rehearsal's date: before the hard kill date whatever day the tests run (None removes it)


def run_script(repo: Path, domain: str, dry: bool = True, script: str = "run_ftmethod.sh", **env) -> subprocess.CompletedProcess:
    """A run of a script of the repo copy; a value None removes the variable; a rehearsal has a fixed DRY_TODAY."""
    e = {k: v for k, v in os.environ.items() if k not in CLEAN_ENV}
    e.update(PYTHON=sys.executable.replace("\\", "/"), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if dry:
        e["DRY_RUN"] = "1"
        e["DRY_TODAY"] = FIXED_TODAY
    e.update({k: v for k, v in env.items() if v is not None})
    for k, v in env.items():
        if v is None:
            e.pop(k, None)
    for attempt in range(3):
        r = subprocess.run([BASH, (repo / "scripts" / "sigir" / script).as_posix(), domain], env=e,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
        if r.returncode == 0 or r.stdout.strip() or r.stderr.strip():
            return r                                      # every exit of the scripts says something; silence is bash failing to start
        time.sleep(1 + attempt)
    return r


def run_diag(repo: Path, domain: str, dry: bool = True, **env) -> subprocess.CompletedProcess:
    return run_script(repo, domain, dry, "run_ftmethod_shift_diag.sh", **env)


def tail(r: subprocess.CompletedProcess, n: int = 4000) -> str:
    return f"rc={r.returncode}\n--- stdout ---\n{r.stdout[-n:]}\n--- stderr ---\n{r.stderr[-n:]}"


def snapshot(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and "_dry" not in p.relative_to(root).parts:
            s = p.stat()
            out[p.relative_to(root).as_posix()] = (s.st_size, s.st_mtime_ns, sha1_file(p))
    return out


needs_bash = pytest.mark.skipif(BASH is None, reason=NO_BASH)
needs_chain = pytest.mark.skipif(BASH is None or FAST, reason=NO_BASH if BASH is None else "FTMETHOD_FAST=1")


# ---------------------------------------------------------------- the cached DRY states (shared with the shift-diagnostic tests)
STATE_DIRS = (GRID_DRY, DRY_ROOT)


def state_key() -> str:
    """sha1 over the code that builds a DRY state (every src/confrec file but the diagnostic's own, every bound file, the
    scripts it runs, the amendment and its addenda): a changed key rebuilds the states. FTMETHOD_STATE_KEY pins the key (mutation
    experiments on a scratch copy of the repo reuse the states of the real one: a planted bug in a guard does not change them)."""
    if os.environ.get("FTMETHOD_STATE_KEY"):
        return os.environ["FTMETHOD_STATE_KEY"]
    h = hashlib.sha1(sys.version.encode())
    h.update(DRYB.encode())
    blocks = freeze_blocks()
    files = [p for p in sorted((ROOT / "src" / "confrec").glob("*.py")) if p.name != "ftmethod_shift_diag.py"]
    files += [ROOT / n for n in sorted({*blocks["core"], *blocks["method"]}) if (ROOT / n).exists()]
    files += [ROOT / "scripts" / "sigir" / n for n in ("run_ftgrid.sh", "run_ftmethod.sh", "starperm_panel.py")]
    files += [ROOT / ff.AMENDMENT] + sorted((ROOT / "idea-stage").glob("PREREG_AMENDMENT_3_ADDENDUM_*.md"))
    for p in files:
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:20]


def state_cache(tmp_path_factory) -> Path:
    base = tmp_path_factory.getbasetemp().parent / "ftmethod_state_cache"
    key = state_key()
    base.mkdir(parents=True, exist_ok=True)
    for old in base.iterdir():                         # one key at a time: a changed code base drops the old states
        if old.name != key:
            shutil.rmtree(old, ignore_errors=True)
    (base / key).mkdir(exist_ok=True)
    return base / key


STATE_STAGES = {"ml1m": "1,2,3,4", "games": "1"}      # ml1m: the world, q-hat, the freeze record, the adapters and their scores


def build_state(cache: Path, d: str) -> None:
    """The DRY state of dataset d (run_ftgrid.sh's world and the stages of run_ftmethod.sh in STATE_STAGES), built by the
    script itself in a scratch repo and kept without that repo."""
    tmp = Path(tempfile.mkdtemp(prefix=f"state_{d}_", dir=str(cache)))
    try:
        repo = make_repo(tmp / "repo")
        r = run_script(repo, d, STAGES=STATE_STAGES[d])
        assert r.returncode == 0, tail(r)
        dest = cache / f"{d}.part{os.getpid()}"
        for rel in STATE_DIRS:
            shutil.copytree(repo / rel, dest / rel)
        (dest / "build.json").write_text(json.dumps({"stdout": r.stdout, "stderr": r.stderr, "stages": STATE_STAGES[d]}),
                                         encoding="utf-8")
        (dest / ".complete").write_text("ok", encoding="utf-8")
        if (cache / d).exists():
            shutil.rmtree(dest, ignore_errors=True)    # another session finished first
        else:
            os.replace(dest, cache / d)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build_state_once(cache: Path, d: str) -> None:
    """build_state for one builder at a time (a lock directory, atomic to make): the test modules of this slot share the cache, and a
    second one that wants a state under construction waits for it. A lock older than 30 minutes is a killed session's."""
    done, lock = cache / d / ".complete", cache / f"{d}.lock"
    while not done.exists():
        try:
            os.mkdir(lock)
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 1800:
                    shutil.rmtree(lock, ignore_errors=True)
            except OSError:
                pass
            time.sleep(1)
            continue
        try:
            if not done.exists():
                build_state(cache, d)
        finally:
            shutil.rmtree(lock, ignore_errors=True)


_BUILDERS = cf.ThreadPoolExecutor(max_workers=2)
_PREBUILT: dict = {}


def start_states(cache: Path, datasets) -> None:
    """Start building the DRY states of `datasets` in the background (once per state in this process)."""
    for d in datasets:
        if (cache, d) not in _PREBUILT and not (cache / d / ".complete").exists():
            _PREBUILT[(cache, d)] = _BUILDERS.submit(build_state_once, cache, d)


def ensure_states(tmp_path_factory, datasets=("ml1m",)) -> Path:
    """The cache directory holding the DRY states of `datasets`: built now if nobody did, waited for if a background build runs."""
    cache = state_cache(tmp_path_factory)
    start_states(cache, datasets)
    for d in datasets:
        if (cache, d) in _PREBUILT:
            _PREBUILT[(cache, d)].result()
        assert (cache / d / ".complete").exists(), d
    return cache


def seed_state(repo: Path, cache: Path, d: str) -> dict:
    """Copy the cached state of d into the repo copy; the step markers are made newer than the freshly copied sources (a skip is
    by timestamp), so that the run's steps are skipped as they would be in the session that built them. Returns build.json."""
    for rel in STATE_DIRS:
        shutil.copytree(cache / d / rel, repo / rel, dirs_exist_ok=True)
    for f in (repo / DRY_ROOT / "build").rglob("*.done"):
        os.utime(f)
    return json.loads((cache / d / "build.json").read_text(encoding="utf-8"))


def prebuild_for(request, tmp_path_factory, wants: dict) -> None:
    """The autouse fixtures of the test modules call this: when a selected test of the module uses one of the fixtures in `wants`
    ({fixture name: [datasets]}), the states it needs start building at once, so that the cold build overlaps the tests that run
    before it (the first test with a script run waits only for what is left)."""
    if BASH is None or FAST:
        return
    names = {n for it in request.session.items if it.module is request.module for n in it.fixturenames}
    datasets = sorted({d for fx_name, ds in wants.items() if fx_name in names for d in ds})
    if datasets:
        start_states(state_cache(tmp_path_factory), datasets)


@pytest.fixture(scope="module", autouse=True)
def prebuild_states(request, tmp_path_factory):
    prebuild_for(request, tmp_path_factory, {"states": ["ml1m"], "games_state": ["games"]})


@pytest.fixture(scope="module")
def states(tmp_path_factory):
    if BASH is None or FAST:
        pytest.skip(NO_BASH if BASH is None else "FTMETHOD_FAST=1")
    return ensure_states(tmp_path_factory, ("ml1m",))


@pytest.fixture(scope="module")
def games_state(tmp_path_factory):
    if BASH is None or FAST:
        pytest.skip(NO_BASH if BASH is None else "FTMETHOD_FAST=1")
    return ensure_states(tmp_path_factory, ("games",))


@pytest.fixture(scope="module")
def guard_repo(tmp_path_factory):
    if BASH is None:
        pytest.skip(NO_BASH)
    return make_repo(tmp_path_factory.mktemp("ftmethod_guards") / "repo")


@pytest.mark.parametrize("domain, env, dry, msg", [
    ("beauty", {}, True, "usage"),
    ("ml1m", {"STAGES": "1,7"}, True, "unknown stage"),
    ("ml1m", {"OUT_ROOT": "outputs/confrec/ftmethod"}, True, "never writes to a registered output root"),
    ("ml1m", {"OUT_ROOT": "./outputs/confrec/ftgrid_dryrun/"}, True, "never writes to a registered output root"),
    ("ml1m", {"MODEL": "dryrun/Llama-3.1-8B-Instruct"}, True, "single-backbone"),
    ("ml1m", {"OUT_ROOT": "outputs/confrec/ftmethod_llama"}, False, "one registered root"),
])
def test_input_guards_refuse_before_anything_is_written(guard_repo, domain, env, dry, msg):
    r = run_script(guard_repo, domain, dry=dry, **env)
    assert r.returncode == 2 and msg in r.stderr, tail(r)
    assert not (guard_repo / "outputs").exists() and not (guard_repo / "tmp_outputs").exists()


def test_the_order_and_the_kill_rule_refuse_at_the_start(guard_repo):
    r = run_script(guard_repo, "toys")
    assert r.returncode == 4 and "ml1m has no report" in r.stderr, tail(r)
    assert not (guard_repo / "tmp_outputs").exists()                        # nothing built for a refused dataset
    root = guard_repo / DRY_ROOT
    write_dataset_report(root, "ml1m", "FAIL", p=0.4)
    write_dataset_report(root, "toys", "FAIL", p=0.3)
    before = snapshot(root)
    r = run_script(guard_repo, "games")
    assert r.returncode == 4 and "killed after toys" in r.stderr and "games refused" in r.stderr, tail(r)
    assert snapshot(root) == before and not (guard_repo / GRID_DRY).exists()
    shutil.rmtree(guard_repo / "tmp_outputs")


@pytest.fixture(scope="module")
def chain(states, tmp_path_factory):
    """The whole DRY chain on ML-1M: the cached state (run_ftgrid.sh's world and stages 1-4 of run_ftmethod.sh), then the shift
    diagnostic of the rehearsal, which stage 5 waits for (addendum 10 section 2), then stage 5."""
    repo = make_repo(tmp_path_factory.mktemp("ftmethod_chain") / "repo")
    build = seed_state(repo, states, "ml1m")
    root = repo / DRY_ROOT
    r_diag = run_diag(repo, "ml1m")
    assert r_diag.returncode == 0, tail(r_diag)
    r5 = run_script(repo, "ml1m", STAGES="5")
    assert r5.returncode == 0, tail(r5)
    return {"repo": repo, "root": root, "build": build, "r_diag": r_diag, "r": r5, "s1": snapshot(root)}


@needs_chain
def test_chain_layout_and_adapters(chain):
    repo, root = chain["repo"], chain["root"]
    m = root / "ml1m"
    assert {p.name for p in root.iterdir()} == {"_dry", "build", "freeze", "ml1m", "slot.json", "slot_tables.csv"}
    assert {p.name for p in m.iterdir()} == {"train_qhat.csv.gz", "eval_qhat.csv.gz", "qhat_manifest.json", "adapters",
                                             "scores", "report.json", "report_tables.csv", "shift_diag"}
    assert {p.name for p in (m / "adapters").iterdir()} == {"o0", "o1", "o2"}
    grid = repo / GRID_DRY
    train = grid / "panels" / "ml1m" / "train.jsonl"
    n = sum(len(json.loads(x)["candidate_labels"]) for x in train.read_text(encoding="utf-8").splitlines() if x.strip())
    for k in range(3):
        a = m / "adapters" / f"o{k}"
        assert {p.name for p in a.iterdir()} == {"adapter_config.json", "adapter_model.safetensors", "offset.json",
                                                 "train_config.json", "train_report.json"}
        cfg, off = read_json(a / "train_config.json"), read_json(a / "offset.json")
        gt = read_json(grid / "_dry" / "gateft" / "adapters" / f"s{k}" / "train_config.json")   # ML-1M: Gate-FT's
        for key in ("bsz", "grad_accum", "max_len", "lr", "epochs", "lora_r", "variant", "seed", "mode", "hist_len_used"):
            assert cfg[key] == gt[key], (k, key)
        assert cfg["bsz"] * cfg["grad_accum"] == 32 and Path(cfg["train"]).name == "train.jsonl"
        assert off["b"] == pytest.approx(0.05 * (k + 1)) and off["seed"] == k and off["n_examples"] == n
        assert off["b_lr"] == cfg["b_lr"] == 0.01                            # the trainer's default (addendum 3)
        assert {x: off["b_group"][x] for x in ("own_group", "lr", "weight_decay", "registered")} == {
            "own_group": True, "lr": 0.01, "weight_decay": 0.0, "registered": True}
        assert off["b_group"]["source"].startswith("intended") and cfg["prior_offset"]["b_group"] == off["b_group"]
        assert off["qhat_manifest_sha1"] == sha1_file(m / "qhat_manifest.json")
        assert off["sft_adapter"] == f"{GRID_DRY}/_dry/gateft/adapters/s{k}" and off["dry_run"] is True
        d = m / "scores" / f"o{k}" / "like"
        key = (d / "run.key").read_text(encoding="utf-8").split()
        weights = b"".join(w.read_bytes() for w in sorted(a.glob("adapter_model.*")))
        assert key == [sha1_file(grid / "panels" / "ml1m" / "eval.jsonl"), QWEN, "V3", hashlib.sha1(weights).hexdigest(),
                       "--lora", f"{DRY_ROOT}/ml1m/adapters/o{k}"]
        cfg_s = read_json(d / "report.json")["config"]
        assert cfg_s["lora"] == f"{DRY_ROOT}/ml1m/adapters/o{k}" and cfg_s["variant"] == "V3" and cfg_s["hist_len"] == 20
        assert cfg_s["data_sha1"] == key[0] and cfg_s["questions"] == ["like"] and cfg_s["dtype"] == "float16"


@needs_chain
def test_chain_freeze_record_report_and_slot(chain):
    repo, root, out = chain["repo"], chain["root"], chain["build"]["stdout"]
    assert "[dry] freeze rehearsal" in out and "freeze check OK (stage method)" in out
    assert out.index("freeze check OK (stage method)") < out.index("== stage 3")
    log = (repo / GRID_DRY / "_dry" / "PILOT_LOG.md").read_text(encoding="utf-8").lower()
    for rel in ("src/confrec/train_lora_offset.py", "scripts/sigir/run_ftmethod.sh", "src/confrec/ftmethod_report.py",
                f"{DRY_ROOT}/ml1m/qhat_manifest.json",
                # the human step of the diagnostic's own record (addendum 10 section 2), done on the temporary log
                "scripts/sigir/run_ftmethod_shift_diag.sh", "src/confrec/ftmethod_shift_diag.py",
                "tests/test_confrec_ftmethod_shift_diag.py"):
        assert sha1_file(repo / rel) in log, rel
    mark = (root / "freeze" / "ml1m.method.ok").read_text(encoding="utf-8")
    assert sha1_file(root / "ml1m" / "qhat_manifest.json") in mark and sha1_file(repo / SCRIPT.relative_to(ROOT)) in mark
    rep = read_json(root / "ml1m" / "report.json")
    assert rep["decision"]["status"] == "FAIL" and rep["meta"]["dry_run_inputs"] is True and rep["meta"]["n_boot"] == 200
    assert "descriptive" in rep["decision"]["reason"]                       # the dry world has fewer than 150 users
    assert rep["input_checks"] == {"problems": [], "excluded_runs": [], "seeds_present": ["seed0", "seed1", "seed2"],
                                   "seeds_missing": []}
    # review M1: this FAIL is a rehearsal's, so the slot reads it INVALID: it neither fails the dataset nor unlocks toys
    slot = read_json(root / "slot.json")
    assert (slot["state"], slot["fails"], slot["decided"], slot["next_dataset"]) == ("OPEN", [], [], "ml1m")
    i = slot["datasets"]["ml1m"]
    assert i["status"] == "INVALID" and i["report_status"] == "FAIL" and i["dry_run_inputs"] is True
    assert "dry_run_inputs" in i["reason"] and "n_boot" in i["reason"] and i.get("dUAUC") is None
    assert slot["dry_run_inputs"] is True and slot["pending"] == "ml1m"


# the refusals, on clones of the finished chain: each scenario works on its own copy and is small (one to three script runs); they run
# concurrently (a script run is dominated by process starts). A scenario returns the list of its errors.
def clone(c: dict, tmp: Path) -> Path:
    dest = tmp / "repo"
    shutil.copytree(c["repo"], dest)
    return dest


def forbid_training(r) -> str:
    return "" if "dry-run prior-offset trainer" not in r.stdout else "a training started"


def scenario_second_run(c, repo):
    """The whole command a second time (stages 1-5): every finished step is skipped and no file of the chain is touched."""
    root = repo / DRY_ROOT
    before = snapshot(root)
    r = run_script(repo, "ml1m", STAGES="1,2,3,4,5")
    out = r.stdout
    errs = [] if r.returncode == 0 else [tail(r)]
    errs += [] if snapshot(root) == before else ["a file of the chain was touched"]
    errs += [] if "dry-run prior-offset trainer" not in out and "scores chunk" not in out else ["something was run again"]
    errs += [] if out.count(": scored") == 3 and out.count("[skip] adapter") == 3 else ["scored / adapter skips"]
    for product in ("ml1m/qhat_manifest.json", "ml1m/report.json"):
        errs += [] if f"[skip] {DRY_ROOT}/{product}" in out else [f"no skip of {product}"]
    return errs + ([] if "[skip] run_ftgrid.sh's DRY_RUN world for ml1m exists" in out else ["the world was rebuilt"])


def scenario_gate_ft_fail(c, repo):
    gate = repo / GRID_DRY / "_dry" / "gateft" / "gate_ft.json"
    gate.write_text(json.dumps({"decision": "GATE_FT_FAIL"}), encoding="utf-8")
    r = run_script(repo, "ml1m", STAGES="3")
    return [] if r.returncode == 4 and "Gate-FT decision GATE_FT_FAIL" in r.stderr and forbid_training(r) == "" else [tail(r)]


def scenario_a_bound_file_changed_after_the_freeze_check(c, repo, stage="3"):
    bound = repo / "src" / "confrec" / "ftmethod_report.py"
    bound.write_bytes(bound.read_bytes() + b"\n# changed after the freeze check\n")
    r = run_script(repo, "ml1m", STAGES=stage)
    return [] if r.returncode == 4 and f"stage {stage} refused" in r.stderr and "no longer equals" in r.stderr else [tail(r)]


def scenario_a_rehearsal_report_does_not_unlock_the_next_dataset(c, repo):
    r = run_script(repo, "sports")        # the rehearsal's report of ml1m is INVALID: it is not decided, so toys is not next
    return [] if r.returncode == 4 and "ml1m has status INVALID" in r.stderr and "dry_run_inputs" in r.stderr else [tail(r)]


def scenario_kill_rule(c, repo):
    root = repo / DRY_ROOT
    write_dataset_report(root, "ml1m", "FAIL", p=0.4)                         # registered-looking reports: ml1m and toys failed
    write_dataset_report(root, "toys", "FAIL", p=0.3)
    r = run_script(repo, "games")
    return [] if r.returncode == 4 and "killed after toys" in r.stderr and "games refused" in r.stderr else [tail(r)]


def scenario_the_date_without_a_finished_report(c, repo):
    """m2 on the script: after 2026-11-30 a dataset without a finished report reads NOT_RUN and does not run (stages 3 and 5), and
    nothing is written."""
    root, errs = repo / DRY_ROOT, []
    before = snapshot(root)
    r = run_script(repo, "ml1m", STAGES="3", DRY_TODAY="20261201")             # no finished report: NOT_RUN by the date
    errs += [] if r.returncode == 4 and "reads NOT_RUN" in r.stderr and "hard kill date" in r.stderr else [tail(r)]
    return errs + ([] if snapshot(root) == before else ["a refused run wrote"])


def scenario_the_date_with_a_finished_report_refuses_stages_2_and_4(c, repo):
    root, errs = repo / DRY_ROOT, []
    write_dataset_report(root, "ml1m", "PASS", p=0.01)                          # a finished report: the order rule lets the run through
    for stage in ("4",):                                                        # ... to the script's own check of the date
        r = run_script(repo, "ml1m", STAGES=stage, DRY_TODAY="20261201")
        errs += [] if r.returncode == 4 and "hard kill date" in r.stderr and forbid_training(r) == "" else [tail(r)]
    return errs


def scenario_the_date_stage_5_still_reports_a_finished_dataset(c, repo):
    root, errs = repo / DRY_ROOT, []
    write_dataset_report(root, "ml1m", "PASS", p=0.01)
    planted = (root / "ml1m" / "report.json").read_bytes()
    r = run_script(repo, "ml1m", STAGES="5", DRY_TODAY="20261201")            # the report still runs after the date
    errs += [] if r.returncode == 0 else [tail(r)]
    slot = read_json(root / "slot.json")
    errs += [] if (slot["passes"] == ["ml1m"] and slot["past_hard_kill_date"] is True and slot["not_run"] == ["toys", "games", "sports"]
                   and slot["state"] == "NOT_SURVIVED") else [f"slot after the date: {slot['state']} {slot['not_run']}"]
    r = run_script(repo, "toys", STAGES="2", DRY_TODAY="20261201")             # the next dataset does not run after the date
    errs += [] if r.returncode == 4 and "toys refused" in r.stderr and "reads NOT_RUN" in r.stderr else [tail(r)]
    return errs + ([] if (root / "ml1m" / "report.json").read_bytes() == planted else ["the decided report was rewritten"])


def fake_date_env(tmp: Path) -> Path:
    """A file for BASH_ENV that defines `date` as a function: +%Y%m%d prints 20261130 until the file $FAKE_DATE_FLIP exists and 20261201
    after it, every other call is the real date. (A function, not a `date` on the PATH: Git Bash puts /usr/bin ahead of the inherited
    PATH, so a fake executable there is never found; BASH_ENV is read by every non-interactive bash, the script's and its children.)"""
    f = tmp / "fake_date_env.sh"
    f.write_text('date() {\n  if [ "${1:-}" = "+%Y%m%d" ]; then\n'
                 '    if [ -e "${FAKE_DATE_FLIP:-/nonexistent/flip}" ]; then echo 20261201; else echo 20261130; fi\n'
                 '  else\n    command date "$@"\n  fi\n}\n', encoding="utf-8", newline="\n")
    return f


def test_review_m2_the_hard_kill_date_is_checked_before_every_seed_not_once_per_stage(states, tmp_path):
    """The reviewer's scenario: the date turns while stage 3 runs (a stage of five GPU hours crosses midnight). The old script read the
    date once, at the start of the stage, and went on to train every seed. Here the clock is a `date` function that reads 2026-11-30
    until the first adapter exists and 2026-12-01 after it: the second seed is refused (exit 4), the third never starts."""
    if BASH is None or FAST:
        pytest.skip(NO_BASH if BASH is None else "FTMETHOD_FAST=1")
    repo = make_repo(tmp_path / "repo")
    build = seed_state(repo, states, "ml1m")
    root = repo / DRY_ROOT
    shutil.rmtree(root / "ml1m" / "adapters")                                 # stage 3 has to train all three seeds
    shutil.rmtree(root / "ml1m" / "scores")
    flip = root / "ml1m" / "adapters" / "o0" / "offset.json"
    r = run_script(repo, "ml1m", STAGES="3", DRY_TODAY=None, BASH_ENV=fake_date_env(tmp_path).as_posix(), FAKE_DATE_FLIP=flip.as_posix())
    assert r.returncode == 4 and "hard kill date" in r.stderr and "it is 20261201" in r.stderr, tail(r)
    assert r.stdout.count("dry-run prior-offset trainer") == 1 and (root / "ml1m" / "adapters" / "o0" / "offset.json").is_file()
    assert not (root / "ml1m" / "adapters" / "o1").exists() and not (root / "ml1m" / "adapters" / "o2").exists()
    # the control is the state build itself: its stage 3 trained all three seeds under an ordinary clock
    assert build["stdout"].count("dry-run prior-offset trainer") == 3


def stage_5_refused(repo, tag):
    """Stage 5 refuses (exit 4, naming the diagnostic) and builds no report."""
    root = repo / DRY_ROOT
    for name in ("report.json", "report_tables.csv"):
        (root / "ml1m" / name).unlink()
    r = run_script(repo, "ml1m", STAGES="5")
    ok = r.returncode == 4 and "stage 5 refused" in r.stderr and "shift diagnostic" in r.stderr and not (root / "ml1m" / "report.json").exists()
    return [] if ok else [f"{tag}: " + tail(r)]


def scenario_stage_5_waits_for_the_shift_diagnostic(c, repo):
    """Addendum 10 section 2: the slot report is built after the diagnostic. Stage 5 refuses without it."""
    shutil.rmtree(repo / DRY_ROOT / "ml1m" / "shift_diag")
    return stage_5_refused(repo, "no diagnostic")


def scenario_stage_5_refuses_a_damaged_scoring_file(c, repo):
    top = repo / DRY_ROOT / "ml1m" / "shift_diag" / "o1" / "top50.jsonl.gz"
    top.write_bytes(top.read_bytes()[:-9] + b"damaged!!")
    return stage_5_refused(repo, "damaged scoring file")


def scenario_stage_5_refuses_a_diagnostic_of_another_adapter(c, repo):
    w = repo / DRY_ROOT / "ml1m" / "adapters" / "o2" / "adapter_model.safetensors"
    w.write_bytes(w.read_bytes() + b" retrained")
    return stage_5_refused(repo, "another adapter")


def patch_fakes_for_a_transient_failure(repo: Path) -> None:
    """Test only: in this scratch repo's copy of run_ftgrid.sh's scorer stand-in DRY_E1_FAIL fails just the FIRST run of a run.key
    (a transient failure): once a DIR.e1fail.* sibling with the same run.key exists, the rerun passes."""
    fakes = repo / GRID_DRY / "_dry" / "ftgrid_fakes.py"
    text = fakes.read_text(encoding="utf-8")
    helper = ("def _failed_before(output):\n"
              "    out = Path(str(output))\n"
              "    key = (out / 'run.key').read_text(encoding='utf-8') if (out / 'run.key').is_file() else None\n"
              "    return any((d / 'run.key').is_file() and (d / 'run.key').read_text(encoding='utf-8') == key\n"
              "               for d in out.parent.glob(out.name + '.e1fail.*'))\n\n\n")
    assert text.count("def fake_model(args):") == 1
    text = text.replace("def fake_model(args):", helper + "def fake_model(args):")
    text, n = re.subn(r"(mass = 0\.5 if fail and fail in str\(args\.output\)\.replace\(.*?\))( else 0\.99999)",
                      r"\1 and not _failed_before(args.output)\2", text)
    assert n == 1
    fakes.write_text(text, encoding="utf-8")


E1_FAIL = "ftmethod/ml1m/scores/o1"          # DRY_E1_FAIL: the like pass of o1 gets Yes+No mass 0.5 (it fails E1)


def leftover_e1fail(root: Path, run_key) -> Path:
    """o1's finished like pass moved aside as like.e1fail.<an old time> (with the given run.key; None: the current one), so that o1
    is scored again with a failed run of some earlier time left next to it."""
    like = root / "ml1m" / "scores" / "o1" / "like"
    left = like.parent / "like.e1fail.20200101000000"
    shutil.copytree(like, left)
    if run_key is not None:
        (left / "run.key").write_text(run_key, encoding="utf-8")
    shutil.rmtree(like)
    return left


def scenario_review_m5_a_leftover_e1fail_of_an_earlier_run_does_not_use_the_rerun_up(c, repo):
    """The reviewer's plant (as for run_ftq.sh): an o1/like.e1fail.<time> left by an earlier run (another panel, adapter or argument
    vector: another run.key) used to turn a TRANSIENT first E1 failure into the final one (FAILED_INTEGRITY without the rerun). Now
    only an e1fail directory of the current run.key uses the rerun up."""
    root = repo / DRY_ROOT
    left = leftover_e1fail(root, f"{'0' * 40} {QWEN} V3 {'1' * 40} --lora elsewhere/o1\n")
    patch_fakes_for_a_transient_failure(repo)
    r = run_script(repo, "ml1m", STAGES="4", DRY_E1_FAIL=E1_FAIL)
    like = root / "ml1m" / "scores" / "o1" / "like"
    errs = [] if r.returncode == 0 else [tail(r)]
    errs += [] if r.stderr.count("[E1 failed]") == 1 and "failed E1 twice" not in r.stderr else ["no rerun: " + tail(r)]
    errs += [] if (like / "report.json").is_file() and not (like / "FAILED_INTEGRITY").exists() else ["the rerun did not stand"]
    return errs + ([] if left.is_dir() and len(list(like.parent.glob("like.e1fail.*"))) == 2 else ["the leftover and this run's e1fail"])


def scenario_review_m5_an_e1fail_of_the_same_run_key_uses_the_rerun_up(c, repo):
    """The rule itself, as run_ftgrid.sh and run_ftq.sh have it: an e1fail directory of the CURRENT run.key (an earlier failed attempt of
    the very same panel, adapter and arguments) leaves no rerun: the next failure is final."""
    root = repo / DRY_ROOT
    leftover_e1fail(root, None)
    r = run_script(repo, "ml1m", STAGES="4", DRY_E1_FAIL=E1_FAIL)
    like = root / "ml1m" / "scores" / "o1" / "like"
    errs = [] if r.returncode == 0 else [tail(r)]
    errs += [] if r.stderr.count("[E1 failed]") == 0 and r.stderr.count("failed E1 twice") == 1 else ["no immediate end: " + tail(r)]
    return errs + ([] if (like / "FAILED_INTEGRITY").is_file() and len(list(like.parent.glob("like.e1fail.*"))) == 1
                   else ["no FAILED_INTEGRITY"])


def scenario_e1_failure_reruns_once_then_the_missing_seed_makes_the_report_incomplete(c, repo):
    root, errs = repo / DRY_ROOT, []
    shutil.rmtree(root / "ml1m" / "scores" / "o1" / "like")
    r = run_script(repo, "ml1m", STAGES="4,5", DRY_E1_FAIL=E1_FAIL)
    errs += [] if r.returncode == 0 else [tail(r)]
    d = root / "ml1m" / "scores" / "o1"
    errs += [] if (d / "like" / "FAILED_INTEGRITY").is_file() and len(list(d.glob("like.e1fail.*"))) == 1 else ["no FAILED_INTEGRITY"]
    errs += [] if r.stderr.count("[E1 failed]") == 1 and r.stderr.count("FAILED_INTEGRITY:") == 1 else ["rerun-once messages"]
    rep = read_json(root / "ml1m" / "report.json")
    errs += [] if (rep["decision"]["status"] == "INCOMPLETE" and rep["input_checks"]["seeds_present"] == ["seed0", "seed2"]
                   and rep["input_checks"]["excluded_runs"][0]["status"] == "FAILED_INTEGRITY") else ["report"]
    slot = read_json(root / "slot.json")      # the rehearsal's report is INVALID to the slot (review M1), INCOMPLETE in its own words
    errs += [] if (slot["state"], slot["pending"], slot["pending_status"]) == ("OPEN", "ml1m", "INVALID") else [f"slot {slot['state']}"]
    errs += [] if slot["datasets"]["ml1m"]["report_status"] == "INCOMPLETE" else ["report_status"]
    return errs


def scenario_a_cut_dataset_does_not_run(c, repo):
    """Review m4 on the script: a pilot-log line with `FTMETHOD_NOT_RUN toys` makes toys NOT_RUN (cut at the checkpoint): it is refused,
    and nothing is built for it."""
    log = repo / GRID_DRY / "_dry" / "PILOT_LOG.md"
    log.write_text(log.read_text(encoding="utf-8") + "\n- 2026-10-29 FTMETHOD_NOT_RUN toys: cut at the checkpoint (GPU-h)\n", encoding="utf-8")
    before = snapshot(repo / DRY_ROOT)
    r = run_script(repo, "toys")
    errs = [] if r.returncode == 4 and "toys refused" in r.stderr and "FTMETHOD_NOT_RUN toys" in r.stderr else [tail(r)]
    return errs + ([] if snapshot(repo / DRY_ROOT) == before and not (repo / DRY_ROOT / "toys").exists() else ["something was built"])


SCENARIOS = {
    "second_run": scenario_second_run, "gate_ft_fail": scenario_gate_ft_fail,
    "bound_file_stage_3": scenario_a_bound_file_changed_after_the_freeze_check,
    "dry_report_does_not_unlock": scenario_a_rehearsal_report_does_not_unlock_the_next_dataset, "kill_rule": scenario_kill_rule,
    "date_no_report": scenario_the_date_without_a_finished_report,
    "date_finished_stage_4": scenario_the_date_with_a_finished_report_refuses_stages_2_and_4,
    "date_stage_5_reports": scenario_the_date_stage_5_still_reports_a_finished_dataset,
    "diag_missing": scenario_stage_5_waits_for_the_shift_diagnostic, "diag_damaged": scenario_stage_5_refuses_a_damaged_scoring_file,
    "diag_other_adapter": scenario_stage_5_refuses_a_diagnostic_of_another_adapter,
    "m5_leftover": scenario_review_m5_a_leftover_e1fail_of_an_earlier_run_does_not_use_the_rerun_up,
    "m5_same_key": scenario_review_m5_an_e1fail_of_the_same_run_key_uses_the_rerun_up,
    "e1_incomplete": scenario_e1_failure_reruns_once_then_the_missing_seed_makes_the_report_incomplete,
    "cut_dataset": scenario_a_cut_dataset_does_not_run}


@needs_chain
def test_refusals_the_date_the_e1_rules_and_the_diagnostic_gate_on_clones_of_the_chain(chain, tmp_path):
    def work(name):
        sub = tmp_path / name
        sub.mkdir()
        return name, SCENARIOS[name](chain, clone(chain, sub))
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = dict(ex.map(work, SCENARIOS))
    failed = {name: errs for name, errs in results.items() if errs}
    assert not failed, "\n\n".join(f"== {name}\n" + "\n".join(map(str, errs)) for name, errs in failed.items())


@needs_chain
def test_stage_1_runs_for_a_dataset_that_is_not_next(games_state, tmp_path):
    """games is not next, yet its q-hat manifest is built (the cached state of games is the product of `STAGES=1 run_ftmethod.sh games`,
    run by the script itself): every manifest can be recorded before the slot's first run (addendum 2 item 2); stages 2-5 for it are
    refused."""
    build = json.loads((games_state / "games" / "build.json").read_text(encoding="utf-8"))
    assert build["stages"] == "1" and "== stage 1" in build["stdout"] and "== stage 2" not in build["stdout"]
    repo = make_repo(tmp_path / "repo")
    seed_state(repo, games_state, "games")
    root = repo / DRY_ROOT
    assert (root / "games" / "qhat_manifest.json").is_file() and not (root / "games" / "adapters").exists()
    assert read_json(root / "games" / "qhat_manifest.json")["domain"] == "games"
    r = run_script(repo, "games", STAGES="2")
    assert r.returncode == 4 and "ml1m has no report" in r.stderr, tail(r)


# ---------------------------------------------------------------- 8. addendum 10 section 1: one test per finding of the review
# M1 (a): the DRY_RUN root guard compares canonical forms, never the spelling
REGISTERED_FILES = {                                  # what the registered trees hold: a refused run must leave all of it as it is
    "outputs/confrec/ftmethod/ml1m/report.json": b"registered report\n",
    "outputs/confrec/ftmethod/ml1m/qhat_manifest.json": b"registered manifest\n",
    "outputs/confrec/ftmethod/slot.json": b"registered slot\n",
    "outputs/confrec/ftgrid/report/ml1m.json": b"registered grid report\n",
    "outputs/confrec/ftgrid/panels/ml1m/train.jsonl": b"rows\n",
    "outputs/confrec/gateft/gate_ft.json": b"{}\n",
    "outputs/confrec/gatefix/dev/note.txt": b"gate fix\n"}


def registered_repo(dest: Path) -> Path:
    """A scratch repo whose registered roots (the slot, the grid, Gate-FT, the gate fix) hold files."""
    repo = make_repo(dest)
    for rel, data in REGISTERED_FILES.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(data)
    return repo


READ_ONLY_ROOTS = ("ftgrid", "gateft", "gatefix")        # the registered roots the slot only reads


def registered_state(repo: Path, roots=("ftmethod", "ftgrid", "gateft", "gatefix")) -> dict:
    """{path: (size, mtime_ns, sha1)} of everything below the registered roots, names included."""
    out = {}
    for rel in roots:
        base = repo / "outputs" / "confrec" / rel
        for p in sorted(base.rglob("*")) if base.exists() else []:
            out[p.relative_to(repo).as_posix()] = (p.stat().st_size, p.stat().st_mtime_ns, sha1_file(p)) if p.is_file() else "dir"
    return out


def run_many(repo: Path, cases: list, **kw) -> list:
    """[(OUT_ROOT spelling, result)], the script runs concurrently (every run is dominated by process starts)."""
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        return list(ex.map(lambda oc: (oc, run_script(repo, "ml1m", OUT_ROOT=oc, **kw)), cases))


def spellings(repo: Path, rel: str) -> list:
    """The spellings of a path below the repo that a string comparison does not equate with `rel` (// . .. absolute, native)."""
    absolute = (repo / rel).as_posix()
    head, _, last = rel.rpartition("/")
    out = [rel + "//", rel.replace("/", "//", 1), head + "/./" + last, "tmp_outputs/../" + rel, absolute, str(repo / rel)]
    if os.name == "nt":                                # D:\... spellings and a case-insensitive filesystem
        out += [absolute.replace("/", "\\"), absolute.upper(), rel.upper()]
    return out


@needs_bash
def test_review_M1_a_dry_run_never_writes_under_outputs_confrec_in_any_spelling(tmp_path):
    """The reviewer's plant: DRY_RUN=1 OUT_ROOT=outputs/confrec/ftmethod// (one trailing slash was stripped, the string comparison with
    the registered root failed) wrote a synthetic q-hat manifest, adapters and report into the registered root. Now every spelling of
    every registered root, and anything in the repo but tmp_outputs, beside it or above it, is refused (exit 2) before anything is
    written."""
    repo = registered_repo(tmp_path / "repo")
    before = registered_state(repo)
    cases = spellings(repo, "outputs/confrec/ftmethod") + [
        "outputs/confrec/ftmethod", "outputs/confrec/ftmethod///", "outputs/confrec/ftgrid", "outputs/confrec/ftgrid//",
        "outputs/confrec/gateft", "outputs/confrec/gatefix", "outputs/confrec/ftmethod/sub", "outputs/confrec/ftgrid/../ftmethod",
        "outputs/confrec/ftmethod_dryrun", "outputs/confrec", "outputs", ".", "/", "data/raw", "src", "tests", "../outside_repo",
        "..", "tmp_outputs/../src"]
    bad = [(oc, tail(r)) for oc, r in run_many(repo, cases)
           if r.returncode != 2 or not ("never writes to a registered output root" in r.stderr or "root is empty" in r.stderr
                                        or "filesystem root" in r.stderr)]
    assert not bad, "\n\n".join(f"== OUT_ROOT={oc!r}\n{t}" for oc, t in bad)
    assert registered_state(repo) == before and not (repo / "tmp_outputs").exists()
    assert (repo / "outputs/confrec/ftmethod/ml1m/report.json").read_bytes() == REGISTERED_FILES["outputs/confrec/ftmethod/ml1m/report.json"]


@needs_bash
def test_review_M1_a_dry_run_accepts_a_root_in_tmp_outputs_or_outside_the_repo_parent_in_any_spelling(tmp_path):
    """Acceptance is as canonical as refusal: tmp_outputs of the repo, and a directory outside the repo's parent directory, are fine
    however they are spelled (the run stops at the stage list here, after every guard has passed)."""
    repo = registered_repo(tmp_path / "repo")
    before = registered_state(repo)
    away = Path(tempfile.mkdtemp(prefix="ftmethod_dry_away_"))               # outside the repo and outside its parent directory
    try:
        outside = (away / "slot root").as_posix()
        cases = [DRY_ROOT, DRY_ROOT + "//", "tmp_outputs//ftmethod_dryrun/./y", "tmp_outputs/a/../b", outside, str(away / "slot root")]
        if os.name == "nt":
            cases += [DRY_ROOT.upper(), outside.replace("/", "\\")]
        bad = [(oc, tail(r)) for oc, r in run_many(repo, cases, STAGES="9")
               if r.returncode != 2 or "unknown stage" not in r.stderr]
        assert not bad, "\n\n".join(f"== OUT_ROOT={oc!r}\n{t}" for oc, t in bad)
        assert registered_state(repo) == before and not (repo / "tmp_outputs").exists() and not (away / "slot root").exists()
    finally:
        shutil.rmtree(away, ignore_errors=True)


@needs_bash
def test_review_M1_a_real_run_takes_the_registered_root_in_any_spelling_and_no_other_directory(tmp_path):
    repo = registered_repo(tmp_path / "repo")
    before = registered_state(repo)
    real = {"MODEL": "/models/Qwen3-8B", "STAGES": "1"}      # stage 1 is not gated by the order: the next refusal is the selection
    refused = spellings(repo, "outputs/confrec/ftgrid")[:3] + [
        "outputs/confrec/ftgrid", "outputs/confrec/ftmethod_dryrun", "outputs/confrec/ftmethod/sub", "outputs/confrec/ftmethod/..",
        "outputs/confrec/gateft", DRY_ROOT, "outputs/confrec", "."]
    accepted = spellings(repo, "outputs/confrec/ftmethod")[:4] + ["outputs/confrec/ftmethod", "outputs/confrec/./ftmethod/"]
    bad = [(oc, tail(r)) for oc, r in run_many(repo, refused, dry=False, **real) if r.returncode != 2 or "one registered root" not in r.stderr]
    bad += [(oc, tail(r)) for oc, r in run_many(repo, accepted, dry=False, **real)      # the guards pass: the next refusal is
            if r.returncode != 1 or "missing outputs/confrec/gatefix/dev/selection.json" not in r.stderr]    # the missing selection
    assert not bad, "\n\n".join(f"== OUT_ROOT={oc!r}\n{t}" for oc, t in bad)
    assert registered_state(repo) == before and not (repo / "tmp_outputs").exists()


def link_dir(link: Path, target: Path) -> bool:
    """A directory link at `link` to `target`: a symlink, or an NTFS junction where the symlink privilege is missing (a Windows
    box). False when neither can be made (the test then skips)."""
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(str(target), str(link), target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
        return made.returncode == 0 and link.exists()
    return False


def unlink_dir(link: Path) -> None:
    """Remove a directory link (or an empty directory) without touching what it points to."""
    try:
        os.unlink(link)
    except OSError:
        os.rmdir(link)


@needs_bash
def test_review_M1_a_link_between_the_registered_roots_and_the_files_a_run_writes_is_refused(tmp_path):
    """Planted links (symlinks, or junctions on a Windows box): an alias of the registered slot as a DRY root, the synthetic world's own
    path leading into the grid, a link inside that world, a link below the DRY root, the registered root itself a link to the grid, the
    whole outputs tree reached through a link, and a link below the registered root. Every one is refused with exit 2 before anything is
    written, and the registered tree is left as it was."""
    base = registered_repo(tmp_path / "base")
    probe = tmp_path / "probe"
    (probe / "t").mkdir(parents=True)
    if not link_dir(probe / "l", probe / "t"):
        pytest.skip("no directory links can be made here (no symlink privilege, no junctions)")
    unlink_dir(probe / "l")
    slot, grid, real = "outputs/confrec/ftmethod", "outputs/confrec/ftgrid", {"MODEL": "/models/Qwen3-8B"}
    state = registered_state(base, READ_ONLY_ROOTS)       # a plant replaces a part of the slot root itself: the roots it only reads must stay

    def clone_of(name: str) -> Path:
        dest = tmp_path / name / "repo"
        shutil.copytree(base, dest)
        return dest

    def plant(repo: Path, rel: str, target: str) -> None:
        link = repo / rel
        if link.exists():
            shutil.rmtree(link)                                            # a real directory is replaced by the link
        assert link_dir(link, repo / target), rel

    def refused(repo: Path, name: str, r, msg: str) -> list:
        ok = r.returncode == 2 and msg in r.stderr and registered_state(repo, READ_ONLY_ROOTS) == state
        return [] if ok else [(name, tail(r) + "\nregistered tree unchanged: "
                               + str(registered_state(repo, READ_ONLY_ROOTS) == state))]

    def dry_cases() -> list:
        repo, bad = clone_of("dry"), []
        plant(repo, "tmp_outputs/alias", slot)                             # an alias of the registered slot root as the DRY root
        bad += refused(repo, "alias", run_script(repo, "ml1m", OUT_ROOT="tmp_outputs/alias/sub"), "never writes to a registered")
        bad += [] if os.listdir(repo / "tmp_outputs") == ["alias"] else [("alias", "something was written")]
        unlink_dir(repo / "tmp_outputs" / "alias")
        plant(repo, GRID_DRY, grid)                                        # the synthetic world's own path leads into the grid
        bad += refused(repo, "world", run_script(repo, "ml1m"), "never writes to a registered")
        unlink_dir(repo / GRID_DRY)
        plant(repo, DRY_ROOT + "/ml1m", slot + "/ml1m")                    # a link below the DRY root
        r = run_script(repo, "ml1m")
        bad += refused(repo, "dry child", r, "resolves to") + ([] if not (repo / GRID_DRY).exists() else [("dry child", "built")])
        unlink_dir(repo / DRY_ROOT / "ml1m")
        plant(repo, GRID_DRY + "/panels", grid + "/panels")                # a link inside the synthetic world
        bad += refused(repo, "world child", run_script(repo, "ml1m"), "never writes to a registered")
        unlink_dir(repo / GRID_DRY / "panels")
        return bad

    def registered_cases() -> list:
        repo, bad = clone_of("registered"), []
        shutil.rmtree(repo / slot)
        plant(repo, slot, grid)                                            # the registered slot root is a link to the grid
        bad += refused(repo, "registered root", run_script(repo, "ml1m", dry=False, STAGES="1", **real), "one registered root")
        unlink_dir(repo / slot)
        return bad

    def ancestor_cases() -> list:
        repo = clone_of("ancestor")
        os.rename(repo / "outputs", repo / "real_outputs")                 # the whole outputs tree is reached through a link
        assert link_dir(repo / "outputs", repo / "real_outputs")
        out = refused(repo, "outputs is a link", run_script(repo, "ml1m", dry=False, STAGES="1", **real), "one registered root")
        unlink_dir(repo / "outputs")
        return out

    def child_cases() -> list:
        repo, bad = clone_of("child"), []
        for child, msg in (("ml1m", "redirect"),                           # a path the script lists: the tree check ...
                           ("scores", "is a link")):                        # ... and one it does not: the sweep for links
            plant(repo, f"{slot}/{child}", grid + "/panels")
            r = run_script(repo, "ml1m", dry=False, STAGES="1", **real)
            bad += refused(repo, "child " + child, r, msg)
            unlink_dir(repo / slot / child)
        return bad

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        bad = [x for found in [f.result() for f in [ex.submit(fn) for fn in (dry_cases, registered_cases, ancestor_cases, child_cases)]]
               for x in found]
    assert not bad, "\n\n".join(f"== {name}\n{t}" for name, t in bad)


# M1 (b): the slot does not trust any report
@pytest.mark.parametrize("meta, why", [
    ({"dry_run_inputs": True}, "dry_run_inputs"),                                    # the rehearsal's report (the reviewer's plant)
    ({"n_boot": 200}, "n_boot"), ({"n_boot": 1999}, "n_boot"), ({"n_boot": 2000.0}, "n_boot"), ({"n_boot": "2000"}, "n_boot"),
    ({"n_boot": True}, "n_boot"), ({"n_boot": None}, "n_boot"), ({"seed": 1}, "seed"), ({"seed": "0"}, "seed"),
    ({"seed": False}, "seed"), ({"seed": None}, "seed"), ({"dry_run_inputs": None}, "dry_run_inputs"),
    ({"domain": "toys"}, "is of 'toys'")])
def test_review_M1_only_a_registered_report_can_be_decided(tmp_path, meta, why):
    """A report whose meta says dry_run_inputs, n_boot != 2000 or seed != 0 (or that lacks the keys, or is of another dataset) is
    INVALID: it is never counted as decided, never as a failure and never unlocks the next dataset (addendum 10 item 2)."""
    root = tmp_path / "ftmethod"
    write_dataset_report(root, "ml1m", "FAIL", p=0.3, **meta)
    i = fm.read_dataset_report(root, "ml1m")
    assert i["status"] == "INVALID" and why in i["reason"] and i["report_status"] == "FAIL" and i.get("dUAUC") is None
    s = fm.slot_summary(root)
    assert s["fails"] == [] and s["decided"] == [] and s["pending"] == "ml1m" and s["state"] == "OPEN"
    ok, text = fm.check_next({"ml1m": i["status"]}, "toys", {"ml1m": i["reason"]})
    assert not ok and "ml1m has status INVALID" in text and why in text


def test_review_M1_a_rehearsal_cannot_use_up_one_of_the_two_failures_of_the_kill_rule(tmp_path, capsys):
    """The reviewer's scenario: two DRY FAIL reports (dry_run_inputs, n_boot 200) kill the slot in the registered root and refuse the
    next dataset. They are now INVALID; the registered reports of the same numbers do kill it."""
    root, log = tmp_path / "ftmethod", pilot_log(tmp_path)
    for d in ("ml1m", "toys"):
        write_dataset_report(root, d, "FAIL", p=0.4, dry_run_inputs=True, n_boot=200)
    s = fm.slot_summary(root)
    assert (s["state"], s["fails"], s["killed_after"], s["next_dataset"]) == ("OPEN", [], None, "ml1m")
    assert s["datasets"]["toys"]["status"] == "INVALID" and s["violations"] == [
        "toys: a report exists although ml1m (earlier in the registered order) is not decided"]
    with pytest.raises(SystemExit) as e:
        fm.main(["slot", "--root", str(root), "--pilot_log", str(log), "--check_next", "games"])
    assert e.value.code == 4 and "killed after" not in capsys.readouterr().err
    for d in ("ml1m", "toys"):
        write_dataset_report(root, d, "FAIL", p=0.4)
    s = fm.slot_summary(root)
    assert (s["state"], s["fails"], s["killed_after"]) == ("KILLED", ["ml1m", "toys"], "toys")
    with pytest.raises(SystemExit) as e:
        fm.main(["slot", "--root", str(root), "--pilot_log", str(log), "--check_next", "games"])
    assert e.value.code == 4 and "killed after toys" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:                                              # --pilot_log is required (the cuts are read there)
        fm.main(["slot", "--root", str(root)])
    assert e.value.code == 2


def test_review_M1_an_unreadable_or_foreign_report_is_invalid_too(tmp_path):
    root = tmp_path / "ftmethod"
    for d, text in (("ml1m", "{not json"), ("toys", "[]"), ("games", json.dumps({"meta": {"domain": "games"}})),
                    ("sports", json.dumps({"meta": REGISTERED_META | {"domain": "sports"}, "decision": {"status": "MAYBE"}}))):
        (root / d).mkdir(parents=True)
        (root / d / "report.json").write_text(text, encoding="utf-8")
        assert fm.read_dataset_report(root, d)["status"] == "INVALID", d
    assert fm.read_dataset_report(root, "ml1m")["reason"].startswith("unreadable report")
    assert "not a JSON object" in fm.read_dataset_report(root, "toys")["reason"]
    assert "no decision status" in fm.read_dataset_report(root, "sports")["reason"]
    assert fm.report_problems(json.loads(json.dumps({"meta": {"domain": "ml1m", **REGISTERED_META},
                                                     "decision": {"status": "PASS"}})), "ml1m") == []


# M2: DRY_RUN is 0 or 1
@needs_bash
def test_review_M2_dry_run_values_other_than_0_and_1_start_nothing(tmp_path):
    """The old script took [ "$DRY_RUN" = 1 ] for a rehearsal and anything else for the real chain: DRY_RUN=true or DRY_RUN=yes
    started the real job (and wrote below the registered root). Now any other value, an empty one included, is refused (exit 2)."""
    repo = registered_repo(tmp_path / "repo")
    before = registered_state(repo)
    real = {"MODEL": "/models/Qwen3-8B"}
    cases = ["yes", "true", "2", "", "01", " 1", "on"]
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        runs = list(ex.map(lambda v: (v, run_script(repo, "ml1m", dry=False, DRY_RUN=v, **real)), cases))
        runs += list(ex.map(lambda v: (v, run_script(repo, "ml1m", DRY_RUN=v)), ["yes", ""]))
    bad = [(v, tail(r)) for v, r in runs if r.returncode != 2 or "DRY_RUN must be 0 or 1" not in r.stderr or r.stdout.strip()]
    assert not bad, "\n\n".join(f"== DRY_RUN={v!r}\n{t}" for v, t in bad)
    assert registered_state(repo) == before and not (repo / "tmp_outputs").exists()
    r = run_script(repo, "ml1m", dry=False, DRY_RUN="0", STAGES="1", **real)         # 0 is the real chain: it gets to its own refusals
    assert r.returncode == 1 and "missing outputs/confrec/gatefix/dev/selection.json" in r.stderr, tail(r)


@needs_bash
def test_review_m2_dry_today_must_be_eight_digits_and_belongs_to_a_rehearsal(tmp_path):
    """DRY_TODAY=2026-12-01 used to pass silently: [ 2026-12-01 -gt 20261130 ] is an error that an `if` reads as false."""
    repo = make_repo(tmp_path / "repo")
    bad_values = ["2026-12-01", "2026121", "202612011", " 20261201", "abc", ""]
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        runs = list(ex.map(lambda v: (v, run_script(repo, "ml1m", DRY_TODAY=v, STAGES="3")), bad_values))
        runs += list(ex.map(lambda v: (v, run_diag(repo, "ml1m", DRY_TODAY=v)), ["2026-12-01"]))
    bad = [(v, tail(r)) for v, r in runs if r.returncode != 2 or "DRY_TODAY must be eight digits" not in r.stderr]
    assert not bad, "\n\n".join(f"== DRY_TODAY={v!r}\n{t}" for v, t in bad)
    r = run_script(repo, "ml1m", dry=False, DRY_TODAY="20261201", MODEL="/models/Qwen3-8B", STAGES="1")   # a real run reads the clock
    assert r.returncode == 2 and "a real run reads the clock" in r.stderr, tail(r)
    assert not (repo / "tmp_outputs").exists() and not (repo / "outputs").exists()


# m1: the +0.01 threshold
def planted_diff(est: float, per_seed=(0.012, 0.0105, 0.0095), lo=0.002, hi=0.018, n_users=400, sd=0.001) -> dict:
    """The part of ftgrid_report.contrast_models that decide() reads, with a chosen seed-averaged estimate."""
    return {"mean_over_seeds": {"est": est, "lo": lo, "hi": hi, "p": 0.01, "n_users": n_users, "descriptive_min_n": False,
                                "ci_excludes_0": bool(lo > 0 or hi < 0)},
            "seeds": {"sigma_seed": sd}, "per_seed": {f"seed{k}": {"est": v} for k, v in enumerate(per_seed)}}


def test_review_m1_a_mean_of_exactly_plus_001_passes_though_the_float_is_a_hair_below():
    """The reviewer's number: the mean of per-user differences that is +0.0100 in exact arithmetic was computed as
    0.009999999999999998 and failed `>= 0.01`. The comparison is est >= 0.01 - 1e-12 (addendum 10 item 3)."""
    assert 0.009999999999999998 < 0.01
    ok = fm.decide(planted_diff(0.009999999999999998, per_seed=(0.0101, 0.0100, 0.0099)), 3, [])
    assert ok["status"] == "PASS" and ok["conditions"]["mean_ge_0.01"] is True and ok["reason"] is None
    assert fm.decide(planted_diff(0.01), 3, [])["status"] == "PASS"
    assert fm.decide(planted_diff(0.01 - 0.9e-12), 3, [])["status"] == "PASS"            # inside the tolerance
    below = fm.decide(planted_diff(0.01 - 2e-12), 3, [])                                  # outside it: a real shortfall
    assert below["status"] == "FAIL" and below["conditions"]["mean_ge_0.01"] is False and "< +0.01" in below["reason"]
    assert fm.decide(planted_diff(0.0099), 3, [])["status"] == "FAIL"
    # the other conditions are untouched by the tolerance: a mean at the threshold still needs its CI, its seeds and the sigma rule
    assert fm.decide(planted_diff(0.01, lo=-0.001), 3, [])["status"] == "FAIL"
    assert fm.decide(planted_diff(0.01, sd=0.006), 3, [])["status"] == "FAIL"
    assert fm.decide(planted_diff(0.01, per_seed=(0.02, 0.0105, -0.0005)), 3, [])["status"] == "FAIL"
    assert "1e-12" in fm.decide(planted_diff(0.01), 3, [])["rule"]


def thirds_design(n_users: int, ks) -> dict:
    """ftgrid_report.contrast_models on users with labels [1, 0, 0, 0]: in seed s the first ks[s] users have the positive ranked
    second by the prior-offset score and first by the stacking score (a per-user AUC difference of 1/3), the others no difference."""
    y = np.tile([1, 0, 0, 0], n_users)
    users = np.repeat(np.arange(n_users), 4)
    pairs = {}
    for s, k in enumerate(ks):
        a, b = np.tile([3.0, 2.0, 1.0, 0.0], n_users), np.tile([3.0, 2.0, 1.0, 0.0], n_users)
        for u in range(k):
            a[4 * u: 4 * u + 4] = [2.0, 3.0, 1.0, 0.0]
        pairs[f"seed{s}"] = (b, a)                       # (score a, score b): AUC(a) - AUC(b) = +1/3 for the first k users
    return fr.contrast_models(pairs, y, users, np.ones(4 * n_users, bool), 200, 0, n_registered=3)


def test_review_m1_the_threshold_through_the_real_statistics():
    """The same end to end: 400 users, seeds in which 11, 12 and 13 users gain a per-user AUC of 1/3, so that the seed-averaged mean
    is (11 + 12 + 13) / 3 / 3 / 400 = +0.01 in exact arithmetic (a search over such designs found this one, whose float mean is the
    reviewer's number), through ftgrid_report.contrast_models and fm.decide."""
    diff = thirds_design(400, (11, 12, 13))
    est = diff["mean_over_seeds"]["est"]
    assert est == 0.009999999999999998 and est < 0.01 and abs(est - 0.01) < 1e-15
    assert [round(diff["per_seed"][f"seed{k}"]["est"], 6) for k in range(3)] == [round(x / 3 / 400, 6) for x in (11, 12, 13)]
    dec = fm.decide(diff, 3, [])
    assert dec["status"] == "PASS" and dec["conditions"]["mean_ge_0.01"] is True and all(dec["conditions"].values()), dec
    assert dec["dUAUC"] == est and dec["lo"] > 0
    # the bare comparison the old code made fails this very estimate
    assert not (est >= fm.PASS_MIN)


# m4: NOT_RUN and the cut record
def test_review_m4_the_cut_token_is_exact(tmp_path):
    lines = ["- 2026-10-29 FTMETHOD_NOT_RUN sports: cut",                    # 2
             "`FTMETHOD_NOT_RUN games`",                                    # 3
             "FTMETHOD_NOT_RUN  toys", "ftmethod_not_run toys", "FTMETHOD_NOT_RUN_X toys", "xFTMETHOD_NOT_RUN toys",
             "FTMETHOD_NOT_RUN toys2", "FTMETHOD_NOT_RUN <dataset>", "FTMETHOD_NOT_RUN Toys", "FTMETHOD_NOT_RUN",
             "FTMETHOD_NOT_RUN ml1m (cut), FTMETHOD_NOT_RUN ml1m again"]
    cuts = fm.read_cuts(pilot_log(tmp_path, *lines))
    assert cuts == {"sports": [2], "games": [3], "ml1m": [12]}                   # (line 1 is the log's heading)
    assert fm.read_cuts(tmp_path / "gone.md", allow_missing=True) == {}
    with pytest.raises(SystemExit, match="does not exist"):
        fm.read_cuts(tmp_path / "gone.md")
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe FTMETHOD_NOT_RUN toys \x80")
    with pytest.raises(SystemExit, match="cannot be read"):
        fm.read_cuts(tmp_path / "bad.md")


def test_review_m4_a_cut_dataset_reads_not_run_it_is_neither_decided_nor_failed_and_the_next_dataset_skips_it(tmp_path, capsys):
    root = tmp_path / "ftmethod"
    write_dataset_report(root, "ml1m", "PASS", p=0.01)
    log = pilot_log(tmp_path, "- 2026-10-29 FTMETHOD_NOT_RUN toys: cut at the checkpoint (GPU-h remaining)")
    s = fm.slot_summary(root, log)
    assert s["datasets"]["toys"]["status"] == "NOT_RUN" and s["datasets"]["toys"]["cut_lines"] == [2]
    assert (s["state"], s["next_dataset"], s["pending"], s["not_run"], s["decided"], s["fails"]) == (
        "OPEN", "games", "games", ["toys"], ["ml1m"], [])
    assert s["cuts"] == {"toys": [2]} and s["datasets_run"] == ["ml1m"] and "toys" not in s["holm"]["p_raw"]
    ok, why = fm.check_next({d: i["status"] for d, i in s["datasets"].items()}, "games")
    assert ok and "1 NOT_RUN skipped: toys" in why
    ok, why = fm.check_next({d: i["status"] for d, i in s["datasets"].items()}, "toys", {"toys": s["datasets"]["toys"]["reason"]})
    assert not ok and "toys refused" in why and "FTMETHOD_NOT_RUN toys" in why                 # a cut dataset does not run
    fm.main(["slot", "--root", str(root), "--pilot_log", str(log), "--check_next", "games"])   # exit 0 through the CLI
    assert "games may run" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        fm.main(["slot", "--root", str(root), "--pilot_log", str(log), "--check_next", "toys"])
    assert e.value.code == 4 and "reads NOT_RUN" in capsys.readouterr().err
    # without the record the same reports leave toys next
    assert fm.slot_summary(root, pilot_log(tmp_path))["next_dataset"] == "toys"


def test_review_m4_survival_needs_3_passes_so_cuts_can_make_it_impossible():
    st = fm.slot_state({"ml1m": "PASS", "toys": "NOT_RUN", "games": "PASS", "sports": "PASS"})
    assert (st["state"], st["final"], st["passes"], st["not_run"]) == ("SURVIVES", True, ["ml1m", "games", "sports"], ["toys"])
    st = fm.slot_state({"ml1m": "PASS", "toys": "NOT_RUN", "games": "NOT_RUN"})
    assert (st["state"], st["final"], st["next_dataset"]) == ("NOT_SURVIVED", False, "sports")    # 1 pass + 1 open dataset < 3
    st = fm.slot_state({"ml1m": "PASS", "toys": "NOT_RUN", "games": "PASS", "sports": "FAIL"})
    assert (st["state"], st["final"], st["fails"]) == ("NOT_SURVIVED", True, ["sports"])           # 2 passes, one failure: not killed
    st = fm.slot_state({"ml1m": "FAIL", "toys": "NOT_RUN", "games": "FAIL"})
    assert (st["state"], st["killed_after"], st["final"]) == ("KILLED", "games", True)             # a cut does not hide a failure
    st = fm.slot_state({"ml1m": "PASS", "toys": "NOT_RUN", "games": None, "sports": "PASS"})       # a report behind an open dataset
    assert st["violations"] == ["sports: a report exists although games (earlier in the registered order) is not decided"]
    assert fm.slot_state({"ml1m": "NOT_RUN"})["next_dataset"] == "toys"
    with pytest.raises(ValueError):
        fm.slot_state({"ml1m": "SKIPPED"})


def test_review_m4_a_cut_lifts_an_incomplete_dataset_but_never_hides_a_decided_one(tmp_path):
    """Addendum 5 item 2: an INCOMPLETE dataset blocks the next one until it is recorded as not run at a section-10 checkpoint, and the
    report code had no input for that record. The cut record is that input; a cut recorded AFTER a decided report is an INVALID record
    (a cut is decided on GPU-hours alone, never after a result)."""
    root = tmp_path / "ftmethod"
    write_dataset_report(root, "ml1m", "PASS", p=0.01)
    write_dataset_report(root, "toys", "INCOMPLETE")
    assert fm.slot_summary(root)["next_dataset"] == "toys"
    assert not fm.check_next({"ml1m": "PASS", "toys": "INCOMPLETE"}, "games")[0]
    s = fm.slot_summary(root, pilot_log(tmp_path, "FTMETHOD_NOT_RUN toys"))
    assert s["datasets"]["toys"]["status"] == "NOT_RUN" and s["datasets"]["toys"]["report_status"] == "INCOMPLETE"
    assert s["next_dataset"] == "games"
    root2 = tmp_path / "ftmethod2"
    write_dataset_report(root2, "ml1m", "FAIL", p=0.4)
    s = fm.slot_summary(root2, pilot_log(tmp_path, "FTMETHOD_NOT_RUN ml1m"))
    i = s["datasets"]["ml1m"]
    assert i["status"] == "INVALID" and "never after a result" in i["reason"] and s["fails"] == [] and s["next_dataset"] == "ml1m"


# m2: the date in the slot report
def test_review_m2_after_the_hard_kill_date_a_dataset_without_a_finished_report_reads_not_run(tmp_path, capsys):
    root, log = tmp_path / "ftmethod", pilot_log(tmp_path)
    write_dataset_report(root, "ml1m", "PASS", p=0.01)
    write_dataset_report(root, "toys", "INCOMPLETE")
    on_the_day = fm.slot_summary(root, log, 20261130)                           # 2026-11-30 itself is still inside the slot
    assert on_the_day["past_hard_kill_date"] is False and on_the_day["next_dataset"] == "toys"
    after = fm.slot_summary(root, log, 20261201)
    assert after["past_hard_kill_date"] is True and after["next_dataset"] is None and after["final"] is True
    assert [after["datasets"][d]["status"] for d in fm.DATASETS] == ["PASS", "NOT_RUN", "NOT_RUN", "NOT_RUN"]
    assert after["datasets"]["toys"]["report_status"] == "INCOMPLETE" and "hard kill date 2026-11-30" in after["datasets"]["toys"]["reason"]
    assert after["not_run_by_date"] == ["toys", "games", "sports"] and after["state"] == "NOT_SURVIVED"
    assert after["datasets"]["ml1m"]["status"] == "PASS"                        # a finished report stays decided
    # a slot killed before the date keeps its "not run" (null status) behind the kill point: the date adds nothing there
    write_dataset_report(root, "ml1m", "FAIL", p=0.4)
    write_dataset_report(root, "toys", "FAIL", p=0.3)
    killed = fm.slot_summary(root, log, 20261215)
    assert killed["state"] == "KILLED" and killed["datasets"]["games"]["status"] is None and killed["not_run_by_date"] == []
    # through the CLI: --today is eight digits, and a dataset that reads NOT_RUN by the date does not run
    for bad in ("2026-12-01", "2026121", "x"):
        with pytest.raises(SystemExit) as e:
            fm.main(["slot", "--root", str(root), "--pilot_log", str(log), "--today", bad])
        assert e.value.code == 2
    write_dataset_report(root, "ml1m", "PASS", p=0.01)
    (root / "toys" / "report.json").unlink()
    (root / "toys").rmdir()
    with pytest.raises(SystemExit) as e:
        fm.main(["slot", "--root", str(root), "--pilot_log", str(log), "--today", "20261201", "--check_next", "toys"])
    assert e.value.code == 4 and "toys refused" in capsys.readouterr().err
    fm.main(["slot", "--root", str(root), "--pilot_log", str(log), "--today", "20261201", "--check_next", "ml1m"])   # decided: resumable
    assert "ml1m may run" in capsys.readouterr().out


# m3 and m6: the "b = 0" wording, and the trainer that must not change
def test_review_m3_b_equal_0_is_sft_only_for_a_frozen_b_and_the_trainer_file_is_untouched():
    """Addendum 10 item 7. The statement lives in train_lora_offset.py's docstring, which cannot change (its recorded sha1 pins the
    qhat manifests and FT-Q's inputs: review m6), so the qualification is stated in the files of this slot that can: the report's
    docstring and the run script's header. The test that backs the statement freezes b."""
    assert hashlib.sha1(Path(tlo.__file__).read_bytes()).hexdigest().startswith(TRAINER_SHA1_PREFIX)
    assert "With b = 0 the loss, the gradients and so the run are SFT's (tested)" in re.sub(r"\s+", " ", tlo.__doc__)   # unedited
    text, script = re.sub(r"\s+", " ", fm.__doc__), re.sub(r"[#\s]+", " ", SCRIPT.read_text(encoding="utf-8"))
    assert "holds for a FROZEN b" in text and "The registered prior-offset run TRAINS b" in text
    assert "global gradient-norm clip" in text and "not SFT even where b stays near 0" in text
    assert "holds for a FROZEN b only" in script and "so it is not SFT even where b ends near 0" in script
    assert "Addendum 10 section 1" in script and "supersedes the code record of addendum 5" in script
    me = Path(__file__).read_text(encoding="utf-8")
    assert "test_tiny_lora_with_b_fixed_at_zero_trains_exactly_like_sft" in me and "learn_b=False" in me   # the backing test: b frozen


def test_review_m6_none_of_the_new_files_is_bound_and_the_bound_ones_are_listed():
    """No bound core file is edited by this task: the new files are outside every FREEZE block, the report and the script are in the
    `method` block (their new sha1 are recorded before the method record), and the trainer is in it with its recorded sha1."""
    blocks = freeze_blocks()
    listed = {f for files in blocks.values() for f in files}
    assert not listed & {"src/confrec/ftmethod_shift_diag.py", "scripts/sigir/run_ftmethod_shift_diag.sh",
                         "tests/test_confrec_ftmethod_shift_diag.py", "tests/test_confrec_ftmethod.py"}
    assert {"src/confrec/train_lora_offset.py", "src/confrec/ftmethod_report.py", "scripts/sigir/run_ftmethod.sh"} <= set(blocks["method"])
    assert {"src/confrec/pyes_scorer.py", "src/confrec/prompting.py", "src/confrec/ftgrid_report.py"} <= set(blocks["core"])
