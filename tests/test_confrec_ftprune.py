"""Tests of S6, pruning by uncertainty (idea-stage/PREREG_AMENDMENT_3.md section 6): src/confrec/ftprune.py and
scripts/sigir/run_ftprune.sh. CPU only, deterministic, no network, no GPU, no model; torch / transformers never imported.

  * signals on a planted toy (Gate-FT's data builder on synthetic ML-1M raw files, the REAL pyes_scorer with a planted fake
    model): u prunes the examples nearest tau, exactly 25% of each class is removed (round half up), class balance is
    preserved, ties go by the seed-0 key, the seeds change only P1;
  * C's sign and the q-hat plumbing: a brute-force leave-user-out prior on the toy and a hand-computed raw-event fixture;
  * the manifest: byte-identical rebuild (fresh directory and in place, nothing touched), no path, no wall-clock;
  * subsets vs pruned train files: kept keys equal the index file, every candidate_* list aligned, rows without a kept
    candidate dropped, P0 = a byte copy of train.jsonl; verify catches a changed file; the refusals of the signals step;
  * analyze on planted effects: every label (BEATS_RANDOM, WORSE_THAN_RANDOM, ABOUT_EQUAL, INCONCLUSIVE, INCOMPLETE,
    DESCRIPTIVE_MIN_N), the 2 x SD clause, the all-5-positive clause (implied by 2 x SD with five seeds, so tested on
    ten), p symmetry, the user bootstrap = stats.paired_bootstrap, the per-seed UAUC = Gate-FT's G9 statistic, the
    exclusion codes, a cut P3;
  * the script: LF / bash -n, the flag audit against the real argparse (ftprune subcommands, pyes_scorer,
    train_lora_yesno, ftgrid_freeze) and tests/test_confrec_contracts.py's audit, the DRY_RUN stand-ins, the refusals (no
    GATE_FT_PASS, no freeze record, stage order, files changed after the freeze, input guards), DRY_RUN end to end with
    P3 cut, then P3 added, then a resume run that changes nothing, and the E1 rerun-once rule reading INCOMPLETE.
The chain tests need a POSIX bash (Git Bash on Windows; WSL's bash.exe is not used) and are skipped without one.
"""
from __future__ import annotations

import argparse
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
import types
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.confrec import ftgrid_freeze as ff
from src.confrec import ftprune as fp
from src.confrec import gateft_data as gd
from src.confrec import gateft_eval as ge
from src.confrec import pyes_scorer as ps
from src.confrec.metrics import auroc
from src.confrec.prompting import prompt_key
from src.confrec.stats import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sigir" / "run_ftprune.sh"
DRY_ROOT = "outputs/confrec/ftprune_dryrun"
VARIANT = "V1"
THINK = "<think>\n\n</think>\n\n"


# ---------------------------------------------------------------- small helpers
def sha1_file(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path) -> list:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return path


def hx(*parts) -> str:
    return hashlib.sha1(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def subset_keys(out: Path, arm: str, seed: int) -> list:
    t = (out / "subsets" / f"{arm}_s{seed}.txt").read_text(encoding="utf-8")
    return t.split("\n") if t else []


class CharTok:
    """Character tokenizer (ids 1 = Yes, 2 = No); the chat template wraps each message and closes the think block."""

    def __init__(self):
        self.v, self.r = {}, {1: "Yes", 2: "No"}

    def __len__(self):
        return 3 + len(self.v)

    def decode(self, ids):
        return "".join(self.r.get(i, "") for i in ids)

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, enable_thinking=True, **kw):
        out = "".join(f"<{m['role'][0]}>{m['content']}</{m['role'][0]}>" for m in msg) + "<a>"
        return out + (THINK if not enable_thinking else "")

    def __call__(self, text, add_special_tokens=False):
        for c in text:
            if c not in self.v:
                self.v[c] = len(self.v) + 3
                self.r[self.v[c]] = c
        return {"input_ids": [self.v[c] for c in text]}


def fake_loader(planted: dict, mass: float = 0.99999):
    """A fake vLLM model for pyes_scorer.run: the logit of a prompt is planted[prompt_key]."""
    def load(_a):
        tok = CharTok()

        class LLM:
            def generate(self, prompts, sp, use_tqdm=False, **kw):
                out = []
                for pr in prompts:
                    t = tok.decode(pr["prompt_token_ids"])
                    assert t.endswith(THINK)
                    m = re.match(r"^(?:<s>(.*?)</s>)?<u>(.*)</u><a>$", t[: -len(THINK)], re.S)
                    key = m.group(2) if m.group(1) is None else m.group(1) + "\x1d" + m.group(2)
                    py = mass / (1 + math.exp(-planted[key]))
                    top = {-k: SimpleNamespace(logprob=math.log(1e-7) - k / 100) for k in range(1, 49)}
                    top[1] = SimpleNamespace(logprob=math.log(py))
                    top[2] = SimpleNamespace(logprob=math.log(mass - py))
                    out.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=[top])]))
                return out
        return SimpleNamespace(llm=LLM(), tok=tok, gen_kw={}, to_prompt=lambda ids: {"prompt_token_ids": ids}, sp=None,
                               version="fake")
    return load


def planted_scores(panel: Path, out: Path, value, variant: str = VARIANT, mass: float = 0.99999, lora=None) -> Path:
    """The REAL pyes_scorer.run (the run_gateft.sh arguments) on `panel`, the like logit of (user, item) = value(u, i)."""
    rows = read_jsonl(panel)
    hist = ps.resolve_hist_len(variant, "rated", None)
    planted = {}
    for r in rows:
        for i, _q, p in ps.record_requests(r, ["like"], hist, variant, "rated", "yesno"):
            k = prompt_key(p)
            assert k not in planted, "two prompts are identical: the planted logit would be ambiguous"
            planted[k] = value(str(r["user_id"]), str(r["candidate_item_ids"][i]))
    argv = ["--data", str(panel), "--output", str(out), "--model", "/models/Qwen3-8B", "--dtype", "float16",
            "--topk_logprobs", "50", "--max_model_len", "4096", "--chunk_users", "100", "--variant", variant,
            "--readout", "yesno", "--questions", "like"]
    if lora:
        argv += ["--lora", str(lora)]
    ps.run(ps.parse_args(argv), load_model=fake_loader(planted, mass))
    return out


def write_ml1m_raw(d: Path, n_users: int = 40, n_items: int = 60, per_user: int = 30, seed: int = 3) -> Path:
    """ML-1M files (movies.dat, ratings.dat) with distinct timestamps per user."""
    rng = random.Random(seed)
    d.mkdir(parents=True, exist_ok=True)
    (d / "movies.dat").write_text("".join(f"{m}::Movie {m} ({1990 + m % 9})::Drama|Comedy\n"
                                          for m in range(1, n_items + 1)), encoding="latin-1", newline="\n")
    q = {m: rng.gauss(0, 1) for m in range(1, n_items + 1)}
    lines = []
    for u in range(1, n_users + 1):
        bias = rng.gauss(0, 0.3)
        items = rng.sample(range(1, n_items + 1), per_user)
        times = sorted(rng.sample(range(0, 10 ** 6), per_user))
        for m, t in zip(items, times):
            r = min(5, max(1, round(3 + 1.2 * q[m] + bias + rng.gauss(0, 1.0))))
            lines.append(f"{u}::{m}::{r}::{978_000_000 + t}\n")
    (d / "ratings.dat").write_text("".join(lines), encoding="latin-1", newline="\n")
    return d


def planted_logit(u: str, i: str) -> float:
    """Multiples of 1/8 in [-1.5, 1.5] (exact after the scorer's 6-decimal logit): many exact ties."""
    return ((int(hx("L", u, i), 16) % 25) - 12) / 8


def build_world(base: Path, value=planted_logit) -> dict:
    """raw ML-1M -> build_rated_panels h20 panels (DEV = first 25 rows) -> gateft_data (train.jsonl, gateft_split.json)
    -> the zero-shot like pass of train.jsonl (real scorer, planted logits)."""
    from src.confrec import build_rated_panels as brp
    raw = base / "raw"
    write_ml1m_raw(raw / "ml-1m")
    items, events = brp.load_ml1m(raw / "ml-1m")
    rows, _ = brp.build(items, events, n_users=10 ** 6, n_cands=12, hist_len=20, min_hist=3, min_like=2,
                        min_dislike=2, seed=0, source="ml1m")
    assert len(rows) > 30
    dev = base / "panels" / "ml1m_dev_h20.jsonl"
    conf = base / "panels" / "ml1m_confirm_h20.jsonl"
    dev.parent.mkdir(parents=True, exist_ok=True)
    dev.write_text("".join(brp.row_line(r) for r in rows[:25]), encoding="utf-8", newline="\n")
    conf.write_text("".join(brp.row_line(r) for r in rows[25:]), encoding="utf-8", newline="\n")
    gt = base / "gateft"
    gd.main(["--dev_panel", str(dev), "--confirm_panel", str(conf), "--out_dir", str(gt)])
    zs = planted_scores(gt / "train.jsonl", base / "zs_train", value)
    return {"base": base, "raw": raw, "train": gt / "train.jsonl", "split": gt / "gateft_split.json", "zs": zs}


def build(w: dict, out: Path) -> dict:
    return fp.build_signals(w["train"], w["zs"], w["raw"], VARIANT, out, w["split"])


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    base = tmp_path_factory.mktemp("toy")
    w = build_world(base)
    out = base / "out"
    w["man"] = build(w, out)
    w["out"] = out
    w["sig"] = fp.read_signals(out / fp.SIGNALS_FILE)
    return w


# ---------------------------------------------------------------- 1. signals on the planted toy
def test_n_remove_is_25_percent_rounded_half_up():
    assert [fp.n_remove(n) for n in (0, 1, 2, 3, 4, 8, 9, 10, 11, 12)] == [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
    assert fp.n_remove(15412) == 3853 and fp.n_remove(7936) == 1984       # the registered ML-1M classes: exact 25%


def test_u_signal_prunes_the_examples_nearest_tau_with_seed0_ties(toy):
    sig, man, out = toy["sig"], toy["man"], toy["out"]
    L, y = np.array(sig["L_ZS"]), np.array(sig["label"])
    keys = sig["key"]
    planted = np.array([planted_logit(*k.split("::")) for k in keys])
    assert np.array_equal(L, planted)                                    # the scorer's logits are the planted ones
    beta = y.mean()
    tau = float(np.quantile(L, 1 - beta))                                # numpy linear (A3 section 1's convention)
    assert man["signals"]["label_mean_beta"] == pytest.approx(beta) and man["signals"]["tau"] == pytest.approx(tau)
    assert np.allclose(sig["u"], -np.abs(L - tau))
    removed = set(keys) - set(subset_keys(out, "P2", 0))
    straddle = False
    for c in (0, 1):
        idx = [j for j in range(len(keys)) if y[j] == c]
        order = sorted(idx, key=lambda j: (abs(L[j] - tau), hx("tie", 0, keys[j])))
        k = fp.n_remove(len(idx))
        assert removed & {keys[j] for j in idx} == {keys[j] for j in order[:k]}
        assert abs(L[order[k - 1]] - tau) <= min(abs(L[j] - tau) for j in order[k:])     # nearest tau first
        straddle |= abs(L[order[k - 1]] - tau) == abs(L[order[k]] - tau)
        # the planted logits make the most uncertain examples sit right at the boundary
        assert np.mean([abs(L[j] - tau) for j in order[:k]]) < np.mean([abs(L[j] - tau) for j in order[k:]])
    assert straddle, "the planted ties should straddle the 25% cut in a class (the seed-0 key then decides)"


def test_exactly_25_percent_of_each_class_is_removed_and_balance_is_preserved(toy):
    sig, man, out = toy["sig"], toy["man"], toy["out"]
    y = dict(zip(sig["key"], sig["label"]))
    n = {c: sum(1 for v in y.values() if v == c) for c in (0, 1)}
    for arm in fp.ARMS:
        for s in fp.SEEDS:
            kept = set(subset_keys(out, arm, s))
            for c in (0, 1):
                rm = sum(1 for k, v in y.items() if v == c and k not in kept)
                assert rm == (0 if arm == "P0" else fp.n_remove(n[c])), (arm, s, c)
                assert man["subsets"][f"{arm}_s{s}"]["n_removed"][str(c)] == rm
            if arm != "P0":
                share = sum(y[k] for k in kept) / len(kept)
                assert abs(share - n[1] / (n[0] + n[1])) < 2 / len(kept)          # class balance preserved
    assert man["classes"]["1"]["n"] == n[1] and man["classes"]["0"]["n"] == n[0]


def test_the_seeds_change_only_P1(toy):
    out, keys = toy["out"], toy["sig"]["key"]
    y = dict(zip(keys, toy["sig"]["label"]))
    for arm in ("P2", "P3"):
        files = {(out / "subsets" / f"{arm}_s{s}.txt").read_bytes() for s in fp.SEEDS}
        assert len(files) == 1, arm
        assert len({(out / "train" / f"{arm}_s{s}.jsonl").read_bytes() for s in fp.SEEDS}) == 1
    p1 = [frozenset(subset_keys(out, "P1", s)) for s in fp.SEEDS]
    assert len(set(p1)) == 5                                             # every seed draws its own subset
    for s in fp.SEEDS:                                                   # the draw: smallest sha1("P1:<s>:<key>")
        for c in (0, 1):
            idx = sorted((k for k in keys if y[k] == c), key=lambda k: hx("P1", s, k))
            assert set(idx[: fp.n_remove(len(idx))]).isdisjoint(p1[s])
            assert set(idx[fp.n_remove(len(idx)):]) <= p1[s]
        assert subset_keys(out, "P0", s) == keys                         # P0 removes nothing
    assert toy["man"]["checks"]["P1_differs_across_seeds"] and toy["man"]["checks"]["P2_P3_identical_across_seeds"]


def brute_q_hat(ratings: Path, users, items, times, k: float = 5.0) -> np.ndarray:
    """Independent reference: other users' first ratings of the item strictly before t, shrunk to the mean of every
    other user's first ratings strictly before t, (s + k g) / (n + k)."""
    first = {}
    for line in ratings.read_text(encoding="latin-1").splitlines():
        u, i, r, t = line.split("::")
        key = (u, i)
        if key not in first or (int(t), i) < (first[key][0], i):
            first[key] = (int(t), float(r))
    out = []
    for u, i, t in zip(users, items, times):
        s = n = gs = gn = 0.0
        for (v, j), (tt, r) in first.items():
            if v == u or tt >= t:
                continue
            gs += r
            gn += 1
            if j == i:
                s += r
                n += 1
        g = gs / gn
        out.append((s + k * g) / (n + k))
    return np.array(out)


def test_q_hat_is_the_prior_only_leave_user_out_item_mean_and_C_its_signed_deviation(toy):
    sig = toy["sig"]
    users = [k.split("::")[0] for k in sig["key"]]
    q = np.array(sig["q_hat"])
    ref = brute_q_hat(toy["raw"] / "ml-1m" / "ratings.dat", users, sig["item_id"], sig["timestamp"])
    assert np.allclose(q, ref, atol=1e-12)
    y = np.array(sig["label"])
    assert np.allclose(sig["C"], (2 * y - 1) * (q - q.mean()))
    removed = set(sig["key"]) - set(subset_keys(toy["out"], "P3", 0))
    for c in (0, 1):                                   # P3: positives with the highest q-hat, negatives with the lowest
        idx = [j for j in range(len(y)) if y[j] == c]
        order = sorted(idx, key=lambda j: (-sig["C"][j], hx("tie", 0, sig["key"][j])))
        k = fp.n_remove(len(idx))
        assert {sig["key"][j] for j in order[:k]} == removed & {sig["key"][j] for j in idx}
        qr = np.mean([q[j] for j in order[:k]])
        qk = np.mean([q[j] for j in order[k:]])
        assert (qr > qk) if c == 1 else (qr < qk)
    assert toy["man"]["inputs"]["raw"]["n_candidate_pairs_not_in_raw"] == 0
    assert toy["man"]["inputs"]["raw"]["shrink_k"] == 5.0


def test_q_hat_on_a_hand_computed_raw_fixture(tmp_path):
    """u1 rates i1 5 @100 and i2 1 @300 (candidate), i3 4 @350 (candidate); u2: i2 4 @150, i1 2 @250 (candidate);
    u3: i3 3 @50, i2 2 @200; u4: i2 5 @300 (same second as u1's candidate: not before it); u5: i2 3 @400 (after)."""
    d = tmp_path / "raw" / "ml-1m"
    d.mkdir(parents=True)
    (d / "movies.dat").write_text("1::A (1990)::Drama\n2::B (1991)::Drama\n3::C (1992)::Drama\n", encoding="latin-1")
    (d / "ratings.dat").write_text(
        "1::1::5::100\n1::2::1::300\n1::3::4::350\n2::2::4::150\n2::1::2::250\n3::3::3::50\n3::2::2::200\n"
        "4::2::5::300\n5::2::3::400\n", encoding="latin-1")
    rows = [{"user_id": "1", "source_event_id": "1::300", "candidate_item_ids": ["2", "3"],
             "candidate_timestamps": [300, 350], "candidate_labels": [0, 1]},
            {"user_id": "2", "source_event_id": "2::250", "candidate_item_ids": ["1"],
             "candidate_timestamps": [250], "candidate_labels": [0]}]
    ex = fp.train_examples(rows)
    q, rec = fp.q_hat_of(ex, tmp_path / "raw")
    # (u1, i2, 300): others' i2 before 300 = 4, 2; others' ratings before 300 = 4, 2, 2, 3 (u4's 300 is not before)
    # (u1, i3, 350): others' i3 = 3; others before 350 = 4, 2, 2, 3, 5;  (u2, i1, 250): others' i1 = 5; others = 5, 2, 3
    assert np.allclose(q, [(6 + 5 * 11 / 4) / 7, (3 + 5 * 16 / 5) / 6, (5 + 5 * 10 / 3) / 6])
    assert rec["n_candidate_pairs_not_in_raw"] == 0 and rec["n_nonfinite_q_hat"] == 0
    C, mq = fp.c_signal(q, ex["y"])
    assert mq == pytest.approx(q.mean())
    assert C[0] == pytest.approx(-(q[0] - mq)) and C[1] == pytest.approx(q[1] - mq)       # sign (2y - 1)
    assert C[2] == pytest.approx(-(q[2] - mq)) and C[2] < 0 < C[0]      # dislikes: below the mean congruent, above not
    rows[1]["candidate_timestamps"] = [251]                            # not u2's own rating time: wrong source
    with pytest.raises(SystemExit, match="not the panel's source"):
        fp.q_hat_of(fp.train_examples(rows), tmp_path / "raw")


# ---------------------------------------------------------------- 2. manifest, files, verify, refusals
def all_files(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def test_manifest_and_every_file_are_byte_identical_on_a_rebuild_and_hold_no_path_or_time(toy, tmp_path):
    out = toy["out"]
    again = build(toy, tmp_path / "again")
    assert all_files(tmp_path / "again") == all_files(out)              # a fresh directory: the same bytes
    assert json.loads(fp.json_bytes(again)) == read_json(out / fp.MANIFEST)
    mt = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    build(toy, out)                                                      # in place: nothing is rewritten
    assert {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()} == mt
    text = (out / fp.MANIFEST).read_text(encoding="utf-8")
    for s in (str(toy["base"]), toy["base"].as_posix(), "\\", str(Path.home())):
        assert s not in text
    keys = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                keys.append(k)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(read_json(out / fp.MANIFEST))
    assert not [k for k in keys if re.search(r"(^|_)(time|timestamp|date|mtime|host|hostname|path|dir|cwd)(_|$)", k,
                                             re.I)], keys
    man = read_json(out / fp.MANIFEST)
    assert man["code_sha1"]["ftprune.py"] == sha1_file(ROOT / "src" / "confrec" / "ftprune.py")
    assert man["signals"]["sha1"] == sha1_file(out / fp.SIGNALS_FILE)
    assert set(man["files"]) == {fp.SIGNALS_FILE} | {f"subsets/{a}_s{s}.txt" for a in fp.ARMS for s in fp.SEEDS} | \
        {f"train/{a}_s{s}.jsonl" for a in fp.ARMS for s in fp.SEEDS if a != "P0" or s >= 3}
    for rel, sha in man["files"].items():
        assert sha1_file(out / rel) == sha
    with gzip.open(out / fp.SIGNALS_FILE, "rb") as f:
        assert f.readline().decode().strip().split(",") == list(fp.SIGNAL_COLS)
    raw = (out / fp.SIGNALS_FILE).read_bytes()
    assert raw[4:8] == b"\x00\x00\x00\x00"                               # gzip mtime 0


def test_a_changed_input_changes_the_manifest(toy, tmp_path):
    w = dict(toy)
    w["zs"] = planted_scores(toy["train"], tmp_path / "zs2", lambda u, i: 2 * planted_logit(u, i))
    man = build(w, tmp_path / "o")
    assert man["signals"]["sha1"] != toy["man"]["signals"]["sha1"]
    assert man["inputs"]["zero_shot_like_pass"]["scores_sha1"] != toy["man"]["inputs"]["zero_shot_like_pass"]["scores_sha1"]


def test_subset_index_files_and_pruned_train_files_agree(toy):
    out, train = toy["out"], read_jsonl(toy["train"])
    by_user = {str(r["user_id"]): r for r in train}
    assert (out / "train" / "P0_s3.jsonl").read_bytes() == toy["train"].read_bytes()
    assert (out / "train" / "P0_s4.jsonl").read_bytes() == toy["train"].read_bytes()
    assert not any((out / "train" / f"P0_s{s}.jsonl").exists() for s in (0, 1, 2))   # the Gate-FT adapters' data
    assert toy["man"]["checks"]["pruned_writer_reproduces_train_jsonl_bytes"] is True
    for arm in ("P1", "P2", "P3"):
        for s in fp.SEEDS:
            kept = subset_keys(out, arm, s)
            rows = read_jsonl(out / "train" / f"{arm}_s{s}.jsonl")
            got = [f"{r['user_id']}::{i}" for r in rows for i in r["candidate_item_ids"]]
            assert got == kept                                           # same keys, same (train.jsonl) order
            users = [str(r["user_id"]) for r in rows]
            assert users == [u for u in by_user if any(k.startswith(u + "::") for k in kept)]   # empty rows dropped
            for r in rows:
                o = by_user[str(r["user_id"])]
                n = len(r["candidate_item_ids"])
                assert n >= 1
                keep = [i in set(r["candidate_item_ids"]) for i in o["candidate_item_ids"]]
                for k, v in o.items():
                    if k.startswith("candidate_") and isinstance(v, list):
                        assert len(r[k]) == n and r[k] == [x for x, m in zip(v, keep) if m], (arm, s, k)
                    else:
                        assert r[k] == v, (arm, s, k)                    # history and row fields untouched
            rec = toy["man"]["train_files"][f"{arm}_s{s}"]
            assert rec["examples"] == len(kept) and rec["rows"] == len(rows)
            assert rec["positives"] == sum(sum(r["candidate_labels"]) for r in rows)


def test_verify_catches_a_changed_or_missing_file(toy, tmp_path):
    out = tmp_path / "o"
    shutil.copytree(toy["out"], out)
    assert fp.verify(out) == []
    p = out / "train" / "P2_s1.jsonl"
    p.write_bytes(p.read_bytes() + b"\n")
    (out / "subsets" / "P3_s4.txt").unlink()
    probs = fp.verify(out)
    assert any("train/P2_s1.jsonl: sha1" in x for x in probs) and any("subsets/P3_s4.txt: missing" in x for x in probs)
    assert fp.main(["verify", "--out_dir", str(out)]) == 1


def test_signals_refuses_inputs_that_are_not_the_registered_ones(toy, tmp_path):
    w = toy
    with pytest.raises(SystemExit, match="not the selected"):
        fp.build_signals(w["train"], w["zs"], w["raw"], "V3", tmp_path / "a", w["split"])
    lz = planted_scores(w["train"], tmp_path / "zs_lora", planted_logit, lora=tmp_path / "some_adapter")
    with pytest.raises(SystemExit, match="scored an adapter"):
        fp.build_signals(w["train"], lz, w["raw"], VARIANT, tmp_path / "b", w["split"])
    bad = planted_scores(w["train"], tmp_path / "zs_e1", planted_logit, mass=0.5)
    with pytest.raises(SystemExit, match="E1 fails"):
        fp.build_signals(w["train"], bad, w["raw"], VARIANT, tmp_path / "c", w["split"])
    shutil.copytree(w["zs"], tmp_path / "zs_fi")
    (tmp_path / "zs_fi" / "FAILED_INTEGRITY").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="FAILED_INTEGRITY"):
        fp.build_signals(w["train"], tmp_path / "zs_fi", w["raw"], VARIANT, tmp_path / "d", w["split"])
    sp = read_json(w["split"])
    sp["train"]["sha1"] = "0" * 40
    (tmp_path / "split.json").write_text(json.dumps(sp), encoding="utf-8")
    with pytest.raises(SystemExit, match="not Gate-FT's TRAIN set"):
        fp.build_signals(w["train"], w["zs"], w["raw"], VARIANT, tmp_path / "e", tmp_path / "split.json")
    other = write_jsonl(tmp_path / "other.jsonl", read_jsonl(w["train"])[:-1])          # not the scored file
    with pytest.raises(SystemExit, match="data_sha1"):
        fp.build_signals(other, w["zs"], w["raw"], VARIANT, tmp_path / "f")
    (tmp_path / "raw2" / "ml-1m").mkdir(parents=True)
    (tmp_path / "raw2" / "ml-1m" / "movies.dat").write_text("1::A (1990)::Drama\n", encoding="latin-1")
    (tmp_path / "raw2" / "ml-1m" / "ratings.dat").write_text("1::1::5::100\n", encoding="latin-1")
    with pytest.raises(SystemExit, match="not the panel's source"):
        fp.build_signals(w["train"], w["zs"], tmp_path / "raw2", VARIANT, tmp_path / "g", w["split"])
    assert not any((tmp_path / x).exists() for x in "abcdefg")          # a refusal writes nothing


def test_signals_cli(toy, tmp_path, capsys):
    rc = fp.main(["signals", "--train", str(toy["train"]), "--zs_dir", str(toy["zs"]), "--raw", str(toy["raw"]),
                  "--variant", VARIANT, "--split_report", str(toy["split"]), "--out_dir", str(tmp_path / "o")])
    assert rc == 0 and "removed per class" in capsys.readouterr().out
    assert all_files(tmp_path / "o") == all_files(toy["out"])


# ---------------------------------------------------------------- 3. analyze on planted effects
A_T = 1000.0


def analysis_world(base: Path, deltas: dict, n_users: int = 200, noise: dict | None = None, seed: int = 0) -> dict:
    """A synthetic CONFIRM panel (per user 2 CAL rows and 4 TEST rows, 2 likes among them) and the 20 scored runs of the
    registered layout (P0 s0-s2 under the Gate-FT dirs); logit = delta * label + N(0, 1) noise drawn from the run's noise
    key (default: its own), so two runs with one key and one delta score identically."""
    base.mkdir(parents=True, exist_ok=True)
    rows = [{"user_id": f"u{k:04d}", "source_event_id": f"u{k:04d}::1",
             "candidate_item_ids": [f"i{k}_{c}" for c in range(6)],
             "candidate_timestamps": [500, 501, 1000, 1001, 1002, 1003], "candidate_labels": [1, 0, 1, 1, 0, 0]}
            for k in range(n_users)]
    panel = write_jsonl(base / "confirm.jsonl", rows)
    (base / "gateft_split.json").write_text(json.dumps({"T": A_T}), encoding="utf-8")
    w = {"panel": panel, "split": base / "gateft_split.json", "scores": base / "scores", "gt_scores": base / "gt_scores",
         "adapters": base / "adapters", "gt_adapters": base / "gt_adapters", "rows": rows}
    for (arm, s), (d, a, tname) in layout(w).items():
        dl = deltas[arm][s] if isinstance(deltas[arm], (list, tuple)) else deltas[arm]
        key = (noise or {}).get((arm, s), f"{arm}/{s}")
        rng = np.random.default_rng(int(hx(seed, key)[:12], 16))
        write_run(d, a, rows, [dl * y + float(rng.normal()) for r in rows for y in r["candidate_labels"]], tname, s,
                  sha1_file(panel))
    return w


def layout(w: dict) -> dict:
    return fp.expected_layout(w["scores"], w["gt_scores"], w["adapters"], w["gt_adapters"])


def write_run(d: Path, adapter: Path, rows: list, logits: list, train_name: str, seed: int, panel_sha1: str,
              mass: float = 0.99, variant: str = VARIANT) -> None:
    d.mkdir(parents=True, exist_ok=True)
    lines, j = [",".join(ps.SCORE_COLS)], 0
    for r in rows:
        for c, (i, y) in enumerate(zip(r["candidate_item_ids"], r["candidate_labels"])):
            lg = logits[j]
            j += 1
            lines.append(",".join(str(x) for x in (r["source_event_id"], r["user_id"], i, c, y, "like", f"{lg:.6f}",
                                                   "0.000000", f"{lg:.6f}", f"{mass:.6f}", 0, "nan")))
    with gzip.open(d / "scores.csv.gz", "wt", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")
    (d / "report.json").write_text(json.dumps(
        {"variant": variant, "lora": str(adapter), "model": "/models/Qwen3-8B", "backbone": "Qwen3-8B", "hist_len": 10,
         "data_sha1": panel_sha1, "n_overlength": 0, "mean_yes_no_mass": mass, "n_main_prompts": j,
         "censored_main": {"0": j}, "questions": ["like"], "scorer": ps.SCORER}), encoding="utf-8")
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "train_config.json").write_text(json.dumps(
        {"seed": seed, "variant": variant, "train": f"somewhere/{train_name}"}), encoding="utf-8")


def run_analyze(w: dict, arms=fp.ARMS, n_boot: int = 2000) -> dict:
    return fp.analyze(w["panel"], w["split"], w["scores"], w["gt_scores"], w["adapters"], w["gt_adapters"], VARIANT,
                      arms, None, n_boot, 0)


BASE = {"P0": 0.0, "P1": 0.0, "P2": 0.0, "P3": 0.0}


@pytest.mark.parametrize("name, deltas, noise, label", [
    ("beats", {**BASE, "P2": 1.5}, None, "BEATS_RANDOM"),
    ("worse", {**BASE, "P1": 1.5}, None, "WORSE_THAN_RANDOM"),
    ("equal", {**BASE, "P1": 0.8, "P2": 0.8}, {("P2", s): f"P1/{s}" for s in fp.SEEDS}, "ABOUT_EQUAL"),
    ("inconclusive", {**BASE, "P1": 0.5, "P2": 0.5}, None, "INCONCLUSIVE"),
])
def test_planted_effects_get_their_registered_label(tmp_path, name, deltas, noise, label):
    res = run_analyze(analysis_world(tmp_path / name, deltas, noise=noise))
    c = res["contrasts"]["P2-P1"]
    assert res["claim"]["label"] == c["label"] == label, (c["est"], c["lo"], c["hi"], c["per_seed"])
    assert c["n_users"] == 200 and c["complete"] and c["status"] == "OK" and c["seeds_paired"] == list(fp.SEEDS)
    d = np.array(list(c["per_seed"].values()))
    assert c["est"] == pytest.approx(d.mean()) and c["sd_seed"] == pytest.approx(np.std(d, ddof=1))
    cond = c["conditions"]
    if label == "BEATS_RANDOM":
        assert c["lo"] > 0 and (d > 0).all() and c["est"] > 2 * np.std(d, ddof=1) and c["p"] < 0.05
    elif label == "WORSE_THAN_RANDOM":
        assert c["hi"] < 0 and (d < 0).all() and cond["worse"] and not cond["beats"]
    elif label == "ABOUT_EQUAL":
        assert c["lo"] == c["hi"] == 0 and cond["ci_within_equivalence"] and not cond["beats"]
    else:
        assert c["lo"] < 0 < c["hi"] and not cond["ci_within_equivalence"]
    assert res["claim"]["wording"] == fp.WORDING[label]
    assert all(res["arms"][a]["complete"] for a in fp.ARMS)
    assert [res["arms"]["P0"]["seeds"][f"s{s}"]["source"] for s in fp.SEEDS] == ["gateft"] * 3 + ["ftprune"] * 2
    for nm in ("P3-P1", "P1-P0", "P2-P0"):
        assert res["contrasts"][nm]["label"] is None and res["contrasts"][nm]["role"] == "descriptive"


def test_the_two_sd_clause_alone_blocks_beats(tmp_path):
    """Every seed positive and the CI above 0, but one seed carries the effect: mean < 2 SD -> not BEATS."""
    res = run_analyze(analysis_world(tmp_path / "w", {**BASE, "P2": [0.4, 0.4, 0.4, 0.4, 4.0]}, n_users=400))
    c = res["contrasts"]["P2-P1"]
    cond = c["conditions"]
    assert cond["ci_lo_gt_0"] and cond["all_positive"] and not cond["mean_gt_2sd"], c["per_seed"]
    assert c["label"] == "INCONCLUSIVE"


def test_the_all_positive_clause_is_checked_and_implied_by_two_sd_with_five_seeds():
    # five seeds: one negative difference makes mean > 2 SD impossible (Samuelson: |d_s - mean| <= 4/sqrt(5) SD)
    d5 = [0.1, 0.1, 0.1, 0.1, -0.001]
    c5 = fp.conditions(float(np.mean(d5)), 0.01, 0.2, d5)
    assert not c5["all_positive"] and not c5["mean_gt_2sd"] and not c5["beats"]
    rng = np.random.default_rng(0)
    for _ in range(20000):
        d = rng.normal(rng.uniform(-1, 1), rng.uniform(0.01, 1), 5)
        if d.mean() > 2 * d.std(ddof=1):
            assert (d > 0).all()
    # the clause itself bites with more values: ten differences, one negative, mean > 2 SD and lo > 0
    d10 = [1.0] * 9 + [-0.1]
    c10 = fp.conditions(float(np.mean(d10)), 0.5, 1.2, d10)
    assert c10["ci_lo_gt_0"] and c10["mean_gt_2sd"] and not c10["all_positive"] and not c10["beats"]
    assert fp.conditions(float(np.mean(d10[:9])), 0.5, 1.2, d10[:9])["beats"]
    w10 = fp.conditions(-float(np.mean(d10)), -1.2, -0.5, [-x for x in d10])
    assert w10["ci_hi_lt_0"] and w10["mean_lt_minus_2sd"] and not w10["all_negative"] and not w10["worse"]


def test_label_precedence_and_overlap_of_beats_and_about_equal():
    tight = fp.conditions(0.005, 0.002, 0.008, [0.005, 0.0049, 0.0051, 0.005, 0.005])
    assert tight["beats"] and tight["about_equal"] and tight["beats_and_about_equal_overlap"]
    assert fp.claim_label(True, 366, tight) == "BEATS_RANDOM"            # listed order: beats first
    edge = fp.conditions(0.0, -0.01, 0.01, [0.0, 0.001, -0.001, 0.0, 0.0])
    assert edge["about_equal"] and fp.claim_label(True, 366, edge) == "ABOUT_EQUAL"     # within +-0.01, inclusive
    out = fp.conditions(0.0, -0.0101, 0.01, [0.0, 0.001, -0.001, 0.0, 0.0])
    assert fp.claim_label(True, 366, out) == "INCONCLUSIVE"
    assert fp.claim_label(False, 366, tight) == "INCOMPLETE" and fp.claim_label(True, 149, tight) == "DESCRIPTIVE_MIN_N"
    nan = fp.conditions(float("nan"), 0.1, 0.2, [0.1] * 5)
    assert not any(nan[k] for k in ("beats", "worse", "about_equal"))


def test_p_value_and_ci_are_symmetric_and_use_the_paired_user_bootstrap():
    rng = np.random.default_rng(1)
    users = [f"u{k}" for k in range(120)]
    A = {s: {u: float(rng.uniform(0.3, 1)) for u in users} for s in fp.SEEDS}
    B = {s: {u: float(rng.uniform(0.2, 0.95)) for u in users} for s in fp.SEEDS}
    ab, ba = fp.contrast(A, B), fp.contrast(B, A)
    assert ab["est"] == pytest.approx(-ba["est"]) and ab["p"] == ba["p"]
    assert ab["lo"] == pytest.approx(-ba["hi"]) and ab["hi"] == pytest.approx(-ba["lo"])
    ref = paired_bootstrap({u: np.mean([A[s][u] for s in fp.SEEDS]) for u in users},
                           {u: np.mean([B[s][u] for s in fp.SEEDS]) for u in users}, 2000, 0)
    assert (ab["est"], ab["lo"], ab["hi"]) == pytest.approx((ref["est"], ref["lo"], ref["hi"]))
    # p = 2 min(P*(d <= 0), P*(d >= 0)) with +1/(B + 1) smoothing: a sure effect has the floor 2/2001
    C = {s: {u: A[s][u] + 1.0 for u in users} for s in fp.SEEDS}
    assert fp.contrast(C, A)["p"] == pytest.approx(2 / 2001)
    same = fp.contrast(A, A)
    assert same["p"] == 1.0 and same["lo"] == same["hi"] == 0.0


def test_per_seed_uauc_is_gate_fts_g9_statistic(tmp_path):
    w = analysis_world(tmp_path / "w", {**BASE, "P2": 1.0})
    res = run_analyze(w)
    ts = ge.panel_timestamps(w["panel"])
    for (arm, s), (d, _a, _t) in layout(w).items():
        st = ge.run_stats(d, ts, A_T)
        got = res["arms"][arm]["seeds"][f"s{s}"]
        assert got["UAUC"] == pytest.approx(np.mean(list(st["auc_post"].values())))
        # brute force: per user the AUC over the TEST rows (ts >= T), ties 1/2
        rows = {}
        with gzip.open(d / "scores.csv.gz", "rt", encoding="utf-8") as f:
            next(f)
            for line in f:
                ev, u, i, c, y, *_rest = line.split(",")
                lg = float(_rest[3])
                if w["rows"][0]["candidate_timestamps"][int(c)] >= A_T:
                    rows.setdefault(u, []).append((lg, int(y)))
        ref = np.mean([auroc([x for x, _ in v], [y for _, y in v]) for v in rows.values()])
        assert got["UAUC"] == pytest.approx(ref) and got["n_users"] == 200


def test_missing_or_excluded_seeds_read_incomplete_and_are_never_replaced(tmp_path):
    w = analysis_world(tmp_path / "w", {**BASE, "P2": 1.5})
    lay = layout(w)
    shutil.rmtree(lay[("P2", 4)][0])                                     # ABSENT
    (lay[("P1", 1)][0] / "FAILED_INTEGRITY").write_text("", encoding="utf-8")
    d, a, t = lay[("P1", 2)]
    write_run(d, a, w["rows"], [0.0] * 1200, t, 2, sha1_file(w["panel"]), mass=0.5)      # E1: Yes+No mass 0.5
    rep = read_json(lay[("P3", 0)][0] / "report.json")
    (lay[("P3", 0)][0] / "report.json").write_text(json.dumps({**rep, "data_sha1": "0" * 40}), encoding="utf-8")
    rep = read_json(lay[("P3", 1)][0] / "report.json")
    (lay[("P3", 1)][0] / "report.json").write_text(json.dumps({**rep, "variant": "V3"}), encoding="utf-8")
    rep = read_json(lay[("P3", 2)][0] / "report.json")
    (lay[("P3", 2)][0] / "report.json").write_text(json.dumps({**rep, "lora": str(lay[("P3", 3)][1])}), encoding="utf-8")
    cfg = lay[("P0", 3)][1] / "train_config.json"
    cfg.write_text(json.dumps({**read_json(cfg), "seed": 4}), encoding="utf-8")
    (lay[("P0", 4)][0] / "scores.csv.gz").unlink()
    with gzip.open(lay[("P3", 4)][0] / "scores.csv.gz", "at", encoding="utf-8") as f:
        f.write("stranger::1,stranger,x,0,1,like,0.1,0,0.1,0.99,0,nan\n")
    res = run_analyze(w)
    st = {f"{a}/s{s}": res["arms"][a]["seeds"][f"s{s}"]["status"] for a in fp.ARMS for s in fp.SEEDS}
    assert st["P2/s4"] == "ABSENT" and st["P1/s1"] == "FAILED_INTEGRITY" and st["P1/s2"] == "FAILED_INTEGRITY"
    assert st["P3/s0"] == "PANEL_MISMATCH" and st["P3/s1"] == "VARIANT_MISMATCH" and st["P3/s2"] == "ADAPTER_MISMATCH"
    assert st["P0/s3"] == "ADAPTER_MISMATCH" and st["P0/s4"] == "INCOMPLETE" and st["P3/s4"] == "PANEL_MISMATCH"
    c = res["contrasts"]["P2-P1"]
    assert c["label"] == res["claim"]["label"] == "INCOMPLETE" and not c["complete"] and c["status"] == "INCOMPLETE"
    assert c["seeds_paired"] == [0, 3] and c["seeds_missing"] == [1, 2, 4]
    assert c["excluded_runs"] == {"P1/s1": "FAILED_INTEGRITY", "P1/s2": "FAILED_INTEGRITY", "P2/s4": "ABSENT"}
    assert not res["arms"]["P1"]["complete"] and res["arms"]["P1"]["seeds_ok"] == [0, 3, 4]
    assert math.isnan(res["arms"]["P2"]["UAUC_per_seed"][4])            # never replaced by another seed
    assert fp.main(["analyze", "--confirm_panel", str(w["panel"]), "--split_report", str(w["split"]),
                    "--scores_root", str(w["scores"]), "--gateft_scores", str(w["gt_scores"]), "--adapters_root",
                    str(w["adapters"]), "--gateft_adapters", str(w["gt_adapters"]), "--variant", VARIANT, "--out",
                    str(tmp_path / "r.json")]) == 2


def test_fewer_than_150_users_is_descriptive_without_p(tmp_path):
    res = run_analyze(analysis_world(tmp_path / "w", {**BASE, "P2": 1.5}, n_users=100))
    c = res["contrasts"]["P2-P1"]
    assert c["label"] == "DESCRIPTIVE_MIN_N" and c["descriptive_min_n"] and c["p"] is None and c["p_lt_0.05"] is None
    assert c["lo"] > 0 and c["conditions"]["beats"]                     # the interval stays, as a description


def test_a_cut_p3_is_not_run_and_leaves_the_confirmatory_test_alone(tmp_path):
    w = analysis_world(tmp_path / "w", {**BASE, "P2": 1.5})
    for s in fp.SEEDS:
        shutil.rmtree(layout(w)[("P3", s)][0])
    res = run_analyze(w, arms=("P0", "P1", "P2"))
    assert res["arms"]["P3"]["status"] == "NOT_RUN" and res["contrasts"]["P3-P1"]["status"] == "NOT_RUN"
    assert res["claim"]["label"] == "BEATS_RANDOM" and res["inputs"]["arms_run"] == ["P0", "P1", "P2"]
    full = run_analyze(w)                                                # P3 listed but absent: incomplete, descriptive
    assert full["contrasts"]["P3-P1"]["status"] == "INCOMPLETE" and full["claim"]["label"] == "BEATS_RANDOM"
    with pytest.raises(SystemExit, match="always run"):
        run_analyze(w, arms=("P0", "P2"))


def test_analyze_cli_writes_strict_json_and_csv_deterministically(tmp_path, capsys):
    w = analysis_world(tmp_path / "w", {**BASE, "P2": 1.5})
    out = tmp_path / "res" / "pruning_ml1m.json"
    argv = ["analyze", "--confirm_panel", str(w["panel"]), "--split_report", str(w["split"]), "--scores_root",
            str(w["scores"]), "--gateft_scores", str(w["gt_scores"]), "--adapters_root", str(w["adapters"]),
            "--gateft_adapters", str(w["gt_adapters"]), "--variant", VARIANT, "--out", str(out)]
    assert fp.main(argv) == 0 and "LABEL: BEATS_RANDOM" in capsys.readouterr().out
    text = out.read_text(encoding="utf-8")
    res = json.loads(text)                                               # strict: no NaN / Infinity tokens
    assert "NaN" not in text and "Infinity" not in text and res["alias"] == "prn" and res["registered_resampling"]
    assert res["claim"]["label"] == "BEATS_RANDOM" and res["contrasts"]["P2-P1"]["role"] == "confirmatory"
    csv_text = out.with_suffix(".csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].split(",") == list(fp.CSV_COLS)
    assert "contrast,P2-P1,confirmatory,OK," in csv_text and csv_text.rstrip().count("\n") == 4 * 6 + len(fp.CONTRASTS)
    mt = (out.stat().st_mtime_ns, out.with_suffix(".csv").stat().st_mtime_ns)
    assert fp.main(argv) == 0
    assert (out.stat().st_mtime_ns, out.with_suffix(".csv").stat().st_mtime_ns) == mt     # identical rerun: untouched


# ---------------------------------------------------------------- 4. the script: text, flags, stand-ins
def _bash():
    for c in ("C:/Program Files/Git/bin/bash.exe", shutil.which("bash")):
        if c and Path(c).exists() and "system32" not in str(c).lower():   # not WSL's bash.exe
            return str(c)
    return None


BASH = _bash()
NO_BASH = "no bash (Git Bash or a POSIX bash) on PATH"


def test_script_is_lf_bash_and_bash_n_clean():
    raw = SCRIPT.read_bytes()
    assert b"\r\n" not in raw and raw.startswith(b"#!/usr/bin/env bash\n") and b"set -euo pipefail" in raw
    assert b"\r\n" not in (ROOT / "src" / "confrec" / "ftprune.py").read_bytes()
    if BASH is None:
        pytest.skip(NO_BASH)
    r = subprocess.run([BASH, "-n", SCRIPT.as_posix()], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_both_files_are_the_amendments_prune_freeze_list():
    blocks = ff.parse_blocks((ROOT / ff.AMENDMENT).read_text(encoding="utf-8"))
    assert blocks["prune"] == ["src/confrec/ftprune.py", "scripts/sigir/run_ftprune.sh"]
    labels = [rel for rel, _ in ff.required("prune", ROOT, splits=[ff.AMENDMENT])]
    assert "src/confrec/ftprune.py" in labels and "scripts/sigir/run_ftprune.sh" in labels


class _Captured(Exception):
    pass


def _capture(calls):
    orig = (argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args)

    def boom(self, *a, **k):
        raise _Captured(self)
    argparse.ArgumentParser.parse_args = argparse.ArgumentParser.parse_known_args = boom
    old = sys.argv
    sys.argv = ["prog"]
    try:
        for call in calls:
            try:
                call()
            except _Captured as c:
                return c.args[0]
            except (SystemExit, Exception):
                continue
    finally:
        argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args = orig
        sys.argv = old
    return None


@contextmanager
def stub_torch():
    """train_lora_yesno imports torch at module level; its argparse parser needs none of it. Inside the block torch
    resolves to stand-in modules and train_lora_yesno / lora_trainer are imported afresh; restored afterwards."""
    import src.confrec as pkg
    mods = ("src.confrec.train_lora_yesno", "src.confrec.lora_trainer")
    torch_names = ("torch", "torch.utils", "torch.utils.data", "torch.nn", "torch.nn.functional")
    saved = {k: sys.modules.get(k) for k in mods + torch_names}
    attrs = {m.rsplit(".", 1)[1]: getattr(pkg, m.rsplit(".", 1)[1], None) for m in mods}
    try:
        for k in torch_names:
            sys.modules[k] = types.ModuleType(k)
        sys.modules["torch.utils.data"].Dataset = object
        for k in mods:
            sys.modules.pop(k, None)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for a, v in attrs.items():
            if v is not None:
                setattr(pkg, a, v)
            elif hasattr(pkg, a):
                delattr(pkg, a)


def _parser(target: str, sub: str | None):
    """The real argparse parser of 'src.confrec.x' (its subcommand's parser when sub is given)."""
    if target == "src.confrec.train_lora_yesno":
        with stub_torch():
            mod = importlib.import_module(target)
            p = _capture([lambda: mod.main()])
    else:
        mod = importlib.import_module(target)
        p = _capture([lambda: mod.main(), lambda: mod.parse_args(), lambda: mod.main([])])
    assert p is not None, f"no argparse parser captured for {target}"
    subs = [a for a in p._actions if isinstance(a, argparse._SubParsersAction)]
    if subs:
        assert sub in subs[0].choices, f"{target} needs a subcommand from {sorted(subs[0].choices)}, got {sub!r}"
        return subs[0].choices[sub]
    assert sub is None, f"{target} has no subcommand {sub!r}"
    return p


def _code(text: str) -> str:
    """The shell code: heredoc bodies blanked (python), continuation lines joined."""
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


_CUT = re.compile(r"\s(?:\|\||\||&&|;;|;|>>|>|2>&1|2>)\s|;;?\s*$|;\s*(?:then|do)\b")


def _words(rest: str) -> list:
    return shlex.split(re.sub(r"\)\s*$", "", _CUT.split(rest)[0]))


def invocations(text: str) -> list:
    """(target, subcommand, [(flag, value)], line) of every "$PY" / "$MPY" -m src.confrec.X call (arrays expanded from
    every definition); a `score` line carries the flags of score()'s pyes_scorer call."""
    code = _code(text)
    arrays = defaultdict(list)
    for name, body in re.findall(r"(?<![\w$])([A-Za-z_][A-Za-z_0-9]*)\+?=\(([^)]*)\)", code, re.S):
        arrays[name].append(body)
    calls, fixed = [], []
    lines = [ln.strip() for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    for s in lines:
        m = re.search(r'"\$(?:PY|MPY)" -m (src\.confrec\.[a-z_0-9]+)(?: ([a-z]+)(?=\s|$))?(.*)$', s)
        if m:
            target, sub, flags = m.group(1), m.group(2), _flags(_words(m.group(3)), arrays)
            calls.append((target, sub, flags, s))
            if target == "src.confrec.pyes_scorer":
                fixed = flags
    for s in lines:
        if re.match(r"score\s", s):
            calls.append(("src.confrec.pyes_scorer", None, fixed + _flags(_words(s)[3:], arrays), s))
    return calls


def audit(text: str) -> tuple:
    problems, parsers, count = [], {}, defaultdict(int)
    for target, sub, flags, line in invocations(text):
        count[f"{target} {sub or ''}".strip()] += 1
        if (target, sub) not in parsers:
            parsers[(target, sub)] = _parser(target, sub)
        acts = {s: a for a in parsers[(target, sub)]._actions for s in a.option_strings}
        used = {f for f, _ in flags}
        if used - set(acts):
            problems.append(f"{target} {sub or ''} has no flag {sorted(used - set(acts))}: {line[:120]}")
        req = {a.option_strings[0] for a in set(acts.values()) if a.option_strings and a.required}
        if req - used:
            problems.append(f"{target} {sub or ''} lacks required {sorted(req - used)}: {line[:120]}")
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
    return problems, dict(count)


def test_every_flag_the_script_passes_exists_in_the_real_argparse():
    problems, count = audit(SCRIPT.read_text(encoding="utf-8"))
    assert not problems, "\n".join(problems)
    for target, n in {"src.confrec.ftprune signals": 1, "src.confrec.ftprune verify": 2, "src.confrec.ftprune analyze": 1,
                      "src.confrec.pyes_scorer": 3, "src.confrec.train_lora_yesno": 1,
                      "src.confrec.ftgrid_freeze": 8}.items():
        assert count.get(target, 0) >= n, (target, count)               # an audit that finds nothing cannot pass


def test_the_flag_audit_catches_a_wrong_flag_a_wrong_value_and_a_missing_subcommand_flag():
    text = SCRIPT.read_text(encoding="utf-8")
    bad = text.replace("--zs_dir", "--zsdir").replace("--readout yesno", "--readout yesn").replace(
        "--n_boot 2000", "--n_boot many").replace('--max_len "$MAXLEN"', '--maxlen "$MAXLEN"').replace(
        "--stage prune --pilot_log", "--stage prunes --pilot_log")
    problems, _ = audit(bad)
    for frag in ("--zsdir", "'yesn'", "'many'", "--maxlen", "'prunes'", "lacks required ['--zs_dir']"):
        assert any(frag in p for p in problems), (frag, problems)


def test_the_integration_audit_of_test_confrec_contracts_accepts_the_script():
    spec = importlib.util.spec_from_file_location("contracts_for_ftprune", ROOT / "tests" / "test_confrec_contracts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert SCRIPT in sorted((ROOT / "scripts" / "sigir").glob("*.sh"))   # found by its glob: no registration needed
    with stub_torch():
        mod.test_every_script_flag_exists_in_the_real_argparse(SCRIPT)
    mod.test_every_script_flag_exists_in_the_real_argparse(SCRIPT)      # and without torch stand-ins


def fakes_source() -> str:
    return re.search(r"<<'PYFAKES'\n(.*?)\nPYFAKES\n", SCRIPT.read_text(encoding="utf-8"), re.S).group(1) + "\n"


def test_the_dry_run_stand_ins_are_the_real_scorer_and_the_real_trainer_argparse(tmp_path):
    (tmp_path / "f.py").write_text(fakes_source(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ftprune_fakes_under_test", tmp_path / "f.py")
    fk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fk)
    base = tmp_path / "w"
    from src.confrec import build_rated_panels as brp
    write_ml1m_raw(base / "raw" / "ml-1m", n_users=12, n_items=40, per_user=25)
    items, events = brp.load_ml1m(base / "raw" / "ml-1m")
    rows, _ = brp.build(items, events, n_users=10 ** 6, n_cands=8, hist_len=20, min_hist=3, min_like=2, min_dislike=2,
                        seed=0, source="ml1m")
    panel = base / "p.jsonl"
    panel.write_text("".join(brp.row_line(r) for r in rows[:5]), encoding="utf-8", newline="\n")
    argv =["--data", str(panel), "--model", "dryrun/Qwen3-8B", "--dtype", "float16", "--topk_logprobs", "50",
            "--max_model_len", "4096", "--chunk_users", "2", "--variant", VARIANT, "--readout", "yesno",
            "--questions", "like"]
    fk.scorer(argv + ["--output", str(tmp_path / "a")])
    fk.scorer(argv + ["--output", str(tmp_path / "b")])
    rep = read_json(tmp_path / "a" / "report.json")
    assert rep["scorer"] == ps.SCORER and rep["variant"] == VARIANT and rep["hist_len"] == 10 and rep["n_overlength"] == 0
    with gzip.open(tmp_path / "a" / "scores.csv.gz", "rt") as fa, gzip.open(tmp_path / "b" / "scores.csv.gz", "rt") as fb:
        assert fa.read() == fb.read()                                    # deterministic logits
    block = ("import importlib.abc, runpy, sys\n"
             "class NoTorch(importlib.abc.MetaPathFinder):\n"
             "    def find_spec(self, name, path=None, target=None):\n"
             "        if name == 'torch' or name.startswith('torch.'):\n"
             "            raise ImportError('torch is blocked by the test')\n"
             "sys.meta_path.insert(0, NoTorch())\n"
             "sys.argv = sys.argv[1:]\n"
             "runpy.run_path(sys.argv[0], run_name='__main__')\n")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    base_cmd = [sys.executable, "-c", block, str(tmp_path / "f.py"), "trainer", "--train", str(panel), "--model",
                "dryrun/Qwen3-8B"]
    ok = subprocess.run(base_cmd + ["--out", str(tmp_path / "ad"), "--variant", VARIANT, "--seed", "3", "--bsz", "8",
                                    "--grad_accum", "4", "--max_len", "1024"], capture_output=True, text=True, env=env,
                        cwd=ROOT)
    assert ok.returncode == 0, ok.stderr
    cfg = read_json(tmp_path / "ad" / "train_config.json")
    assert (cfg["variant"], cfg["seed"], cfg["hist_len_used"], cfg["bsz"], cfg["grad_accum"], cfg["max_len"]) == \
        (VARIANT, 3, 10, 8, 4, 1024) and cfg["dry_run"] is True
    ps.check_lora_variant(tmp_path / "ad", VARIANT, 10)
    bad = subprocess.run(base_cmd + ["--out", str(tmp_path / "bad"), "--bogus", "1"], capture_output=True, text=True,
                         env=env, cwd=ROOT)
    assert bad.returncode == 2 and "--bogus" in bad.stderr and not (tmp_path / "bad").exists()


# ---------------------------------------------------------------- 5. the script: refusals and the DRY_RUN chain
def make_repo(dest: Path) -> Path:
    """A copy of the repo parts the chain runs: src/confrec, run_ftprune.sh, the amendment with its addenda and every
    file of the core and prune freeze lists."""
    (dest / "src" / "confrec").mkdir(parents=True)
    shutil.copy2(ROOT / "src" / "__init__.py", dest / "src" / "__init__.py")
    for f in (ROOT / "src" / "confrec").glob("*.py"):
        shutil.copy2(f, dest / "src" / "confrec" / f.name)
    blocks = ff.parse_blocks((ROOT / ff.AMENDMENT).read_text(encoding="utf-8"))
    addenda = [p.relative_to(ROOT).as_posix() for p in (ROOT / "idea-stage").glob("PREREG_AMENDMENT_3_ADDENDUM_*.md")]
    for rel in [ff.AMENDMENT, "scripts/sigir/run_ftprune.sh"] + addenda + blocks["core"] + blocks["prune"]:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        if (ROOT / rel).exists():
            shutil.copy2(ROOT / rel, dest / rel)
        elif not (dest / rel).exists():
            (dest / rel).write_text("# placeholder: a listed file that does not exist yet\n", encoding="utf-8",
                                    newline="\n")
    return dest


ENV_DROP = ("MODEL", "OUT_ROOT", "VARIANT", "STAGES", "DRY_RUN", "DRY_GATE", "DRY_E1_FAIL", "RUN_P3", "PYTHONPATH")


def run_script(repo: Path, dry: bool = True, **env) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in ENV_DROP}
    e.update(PYTHON=sys.executable.replace("\\", "/"), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if dry:
        e["DRY_RUN"] = "1"
    e.update(env)
    return subprocess.run([BASH, (repo / "scripts" / "sigir" / "run_ftprune.sh").as_posix()], env=e, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=3600)


def tail(r: subprocess.CompletedProcess, n: int = 4000) -> str:
    return f"rc={r.returncode}\n--- stdout ---\n{r.stdout[-n:]}\n--- stderr ---\n{r.stderr[-n:]}"


def snapshot(root: Path) -> dict:
    """{relative path: (size, mtime_ns, sha1)} of every file under root except the DRY_RUN inputs (_dry)."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and "_dry" not in p.relative_to(root).parts:
            st = p.stat()
            out[p.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns, hashlib.sha1(p.read_bytes()).hexdigest())
    return out


def trained(r: subprocess.CompletedProcess) -> list:
    return re.findall(r"^dry-run trainer: \S*/adapters/(P\d)/s(\d) ", r.stdout, re.M)


needs_bash = pytest.mark.skipif(BASH is None, reason=NO_BASH)


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """DRY_RUN=1 end to end: first with P3 cut (RUN_P3=0), then P3 added (the default), then the same command again."""
    if BASH is None:
        pytest.skip(NO_BASH)
    repo = make_repo(tmp_path_factory.mktemp("chain") / "repo")
    root = repo / DRY_ROOT
    c = {"repo": repo, "root": root}
    c["r1"] = run_script(repo, RUN_P3="0")
    assert c["r1"].returncode == 0, tail(c["r1"])
    c["res1"] = read_json(root / "pruning_ml1m.json")
    c["s1"] = snapshot(root)
    c["r2"] = run_script(repo)
    assert c["r2"].returncode == 0, tail(c["r2"])
    c["res2"] = read_json(root / "pruning_ml1m.json")
    c["s2"] = snapshot(root)
    c["r3"] = run_script(repo)
    assert c["r3"].returncode == 0, tail(c["r3"])
    c["s3"] = snapshot(root)
    return c


def test_chain_layout(chain):
    root = chain["root"]
    names = set(chain["s3"])
    assert {f"subsets/{a}_s{s}.txt" for a in fp.ARMS for s in fp.SEEDS} <= names
    trains = {n for n in names if n.startswith("train/")}
    assert trains == {f"train/{a}_s{s}.jsonl" for a in fp.ARMS for s in fp.SEEDS if fp.has_train_file(a, s)}
    runs = {(a, s) for a in fp.ARMS for s in fp.SEEDS if fp.has_train_file(a, s)}
    assert {(p.parent.name, int(p.name[1:])) for p in (root / "adapters").glob("*/s*")} == runs
    assert {(p.parent.name, int(p.name[1:])) for p in (root / "scores").glob("*/s*")} == runs   # no stale / e1fail
    for a, s in runs:
        assert {"run.key", "report.json", "scores.csv.gz"} <= {p.name for p in (root / "scores" / a / f"s{s}").iterdir()}
    assert {"zs_train/report.json", "zs_train/scores.csv.gz", "zs_train/run.key", "prune_manifest.json",
            "signals/ml1m_signals.csv.gz", "freeze/prune.ok", "pruning_ml1m.json", "pruning_ml1m.csv"} <= names
    assert not [n for n in names if n.endswith(".tmp")]


def test_chain_trains_every_arm_on_its_file_with_the_gate_ft_recipe_and_scores_it_like_run_gateft(chain):
    repo, root = chain["repo"], chain["root"]
    gt = read_json(root / "_dry" / "gateft" / "adapters" / "s0" / "train_config.json")
    conf = root / "_dry" / "gatefix" / "panels" / "ml1m_confirm_h20.jsonl"
    for a in fp.ARMS:
        for s in fp.SEEDS:
            if not fp.has_train_file(a, s):
                continue
            cfg = read_json(root / "adapters" / a / f"s{s}" / "train_config.json")
            assert cfg["train"] == f"{DRY_ROOT}/train/{a}_s{s}.jsonl" and cfg["seed"] == s and cfg["variant"] == VARIANT
            assert (cfg["bsz"], cfg["grad_accum"], cfg["max_len"], cfg["lr"], cfg["lora_r"], cfg["mode"]) == \
                (gt["bsz"], gt["grad_accum"], gt["max_len"], gt["lr"], gt["lora_r"], gt["mode"]) == (8, 4, 1024, 1e-4, 16,
                                                                                                    "standard")
            d = root / "scores" / a / f"s{s}"
            lora = f"{DRY_ROOT}/adapters/{a}/s{s}"
            weights = b"".join(p.read_bytes() for p in sorted((repo / lora).glob("adapter_model.*")))
            assert (d / "run.key").read_text(encoding="utf-8").split() == [
                sha1_file(conf), "dryrun/Qwen3-8B", VARIANT, hashlib.sha1(weights).hexdigest(), "--lora", lora]
            c = read_json(d / "report.json")["config"]
            assert (c["data_sha1"], c["variant"], c["questions"], c["dtype"], c["topk_logprobs"], c["max_model_len"],
                    c["lora"], c["hist_len"], c["readout"]) == (sha1_file(conf), VARIANT, ["like"], "float16", 50, 4096,
                                                                lora, 10, "yesno")
    zs = root / "zs_train"
    train = root / "_dry" / "gateft" / "train.jsonl"
    assert (zs / "run.key").read_text(encoding="utf-8").split() == [sha1_file(train), "dryrun/Qwen3-8B", VARIANT, "-"]
    rep = read_json(zs / "report.json")
    assert rep["lora"] is None and rep["config"]["data_sha1"] == sha1_file(train) and rep["variant"] == VARIANT


def test_chain_order_freeze_before_any_s6_run(chain):
    r1, r2 = chain["r1"], chain["r2"]
    out = r1.stdout
    assert trained(r1) == [("P1", "0"), ("P2", "0"), ("P1", "1"), ("P2", "1"), ("P1", "2"), ("P2", "2"), ("P1", "3"),
                           ("P2", "3"), ("P0", "3"), ("P1", "4"), ("P2", "4"), ("P0", "4")]
    assert trained(r2) == [("P3", str(s)) for s in fp.SEEDS] and trained(chain["r3"]) == []
    assert "P3 not run: cut by RUN_P3=0" in out and "P3 not run" not in r2.stdout
    a = out.index("== stage A")
    assert a < out.index("freeze check OK (stage core)", a) < out.index("[scores chunk", a)    # scoring waits for core
    first = out.index("/adapters/P1/s0 ")
    assert out.index("[dry] freeze rehearsal") < out.index("freeze check OK (stage prune)") < first
    assert out.index("verify OK") < first


def test_chain_freeze_record_and_marker(chain):
    repo, root = chain["repo"], chain["root"]
    log = (root / "_dry" / "PILOT_LOG.md").read_text(encoding="utf-8").lower()
    for rel in [ff.AMENDMENT, "src/confrec/ftprune.py", "scripts/sigir/run_ftprune.sh", "src/confrec/ftgrid_freeze.py",
                f"{DRY_ROOT}/prune_manifest.json", f"{DRY_ROOT}/_dry/ftgrid/panels/ml1m/ftgrid_split.json"]:
        assert sha1_file(repo / rel) in log, rel
    mark = (root / "freeze" / "prune.ok").read_text(encoding="utf-8")
    assert f"{DRY_ROOT}/prune_manifest.json = {sha1_file(root / 'prune_manifest.json')}" in mark
    assert f"scripts/sigir/run_ftprune.sh = {sha1_file(repo / 'scripts/sigir/run_ftprune.sh')}" in mark
    man = read_json(root / "prune_manifest.json")
    for rel, sha in man["files"].items():
        assert sha1_file(root / rel) == sha


def test_chain_analysis_with_p3_cut_then_added(chain):
    res1, res2, root = chain["res1"], chain["res2"], chain["root"]
    assert res1["inputs"]["arms_run"] == ["P0", "P1", "P2"] and res1["arms"]["P3"]["status"] == "NOT_RUN"
    assert res1["contrasts"]["P3-P1"]["status"] == "NOT_RUN"
    assert res2["inputs"]["arms_run"] == list(fp.ARMS)
    for res, arms in ((res1, ("P0", "P1", "P2")), (res2, fp.ARMS)):
        for a in arms:
            assert all(res["arms"][a]["seeds"][f"s{s}"]["status"] == "OK" for s in fp.SEEDS), a
        n = res["contrasts"]["P2-P1"]["n_users"]
        assert 0 < n < 150 and res["claim"]["label"] == "DESCRIPTIVE_MIN_N" and res["claim"]["p"] is None
        assert res["inputs"]["manifest"]["sha1"] == sha1_file(root / "prune_manifest.json")
    assert all(res2["contrasts"][k]["status"] == "OK" for k in ("P2-P1", "P3-P1", "P1-P0", "P2-P0"))
    assert res1["contrasts"]["P2-P1"] == res2["contrasts"]["P2-P1"]    # adding P3 changes no other result
    gt = root / "_dry" / "gateft"
    ts = ge.panel_timestamps(root / "_dry" / "gatefix" / "panels" / "ml1m_confirm_h20.jsonl")
    T = read_json(gt / "gateft_split.json")["T"]
    for s in (0, 1, 2):                                                   # P0 s0-s2 = the Gate-FT passes (G9)
        st = ge.run_stats(gt / "scores" / f"s{s}", ts, T)
        assert res2["arms"]["P0"]["seeds"][f"s{s}"]["UAUC"] == pytest.approx(np.mean(list(st["auc_post"].values())))
        assert res2["arms"]["P0"]["seeds"][f"s{s}"]["source"] == "gateft"


def test_chain_resume_changes_nothing(chain):
    s1, s2, s3 = chain["s1"], chain["s2"], chain["s3"]
    assert s2 == s3
    out = chain["r3"].stdout
    assert "dry-run trainer" not in out and "scores chunk" not in out and "already complete" not in out
    assert out.count("[skip] adapter") == 17 and out.count(": scored") == 18
    assert f"[skip] {DRY_ROOT}/prune_manifest.json" in out
    changed = {k for k in s1 if s1[k] != s2.get(k)}
    assert changed == {"pruning_ml1m.json", "pruning_ml1m.csv"}          # adding P3 touched no earlier file
    assert all(k.startswith(("adapters/P3/", "scores/P3/")) for k in set(s2) - set(s1))


def test_chain_signals_equal_an_in_process_rebuild(chain, tmp_path):
    root = chain["root"]
    dry = root / "_dry"
    man = fp.build_signals(dry / "gateft" / "train.jsonl", root / "zs_train", dry / "raw", VARIANT, tmp_path / "o",
                           dry / "gateft" / "gateft_split.json")
    assert fp.json_bytes(man) == (root / "prune_manifest.json").read_bytes()
    assert all_files(tmp_path / "o")["signals/ml1m_signals.csv.gz"] == (root / fp.SIGNALS_FILE).read_bytes()


def copy_chain(chain, dest: Path) -> Path:
    shutil.copytree(chain["repo"], dest)
    return dest


def test_chain_e1_failure_is_rerun_once_then_failed_integrity_and_the_analysis_reads_incomplete(chain, tmp_path):
    repo = copy_chain(chain, tmp_path / "repo")
    root = repo / DRY_ROOT
    shutil.rmtree(root / "scores" / "P2" / "s4")
    r = run_script(repo, STAGES="D,E", DRY_E1_FAIL="scores/P2/s4")
    assert r.returncode == 2, tail(r)
    assert (root / "scores" / "P2" / "s4" / "FAILED_INTEGRITY").is_file()
    assert len(list((root / "scores" / "P2").glob("s4.e1fail.*"))) == 1
    assert r.stderr.count("[E1 failed]") == 1 and r.stderr.count("FAILED_INTEGRITY:") == 1
    res = read_json(root / "pruning_ml1m.json")
    assert res["claim"]["label"] == "INCOMPLETE"
    assert res["contrasts"]["P2-P1"]["excluded_runs"] == {"P2/s4": "FAILED_INTEGRITY"}
    assert trained(r) == []                                              # nothing is retrained to replace it
    r = run_script(repo, STAGES="D")                                     # a rerun keeps the failure (never replaced)
    assert r.returncode == 0 and f"[skip] {DRY_ROOT}/scores/P2/s4: scored" in r.stdout, tail(r)
    assert (root / "scores" / "P2" / "s4" / "FAILED_INTEGRITY").is_file() and "scores chunk" not in r.stdout


def test_chain_files_changed_after_the_freeze_are_refused(chain, tmp_path):
    repo = copy_chain(chain, tmp_path / "repo")
    root = repo / DRY_ROOT
    p = root / "train" / "P2_s1.jsonl"
    orig = p.read_bytes()
    p.write_bytes(orig + b"\n")
    r = run_script(repo, STAGES="D")
    assert r.returncode == 4 and "no longer has its frozen sha1" in r.stderr, tail(r)
    p.write_bytes(orig)
    code = repo / "src" / "confrec" / "ftprune.py"
    code.write_text(code.read_text(encoding="utf-8") + "\n# changed after the freeze\n", encoding="utf-8", newline="\n")
    for st in ("D", "E"):
        r = run_script(repo, STAGES=st)
        assert r.returncode == 4 and f"stage {st} refused" in r.stderr and "no longer equals" in r.stderr, tail(r)


@needs_bash
def test_no_gate_ft_pass_and_no_freeze_record_are_refused_in_dry_run(tmp_path):
    repo = make_repo(tmp_path / "repo")
    root = repo / DRY_ROOT
    r = run_script(repo, DRY_GATE="GATE_FT_FAIL")
    assert r.returncode == 4 and "S6 refused: Gate-FT decision GATE_FT_FAIL" in r.stderr, tail(r)
    assert {p.name for p in root.iterdir()} == {"_dry"}                  # nothing but the dry inputs
    (root / "_dry" / "gateft" / "gate_ft.json").unlink()                 # the dry world now records GATE_FT_PASS
    for st in ("D", "E"):
        r = run_script(repo, STAGES=st)
        assert r.returncode == 4 and f"stage {st} refused" in r.stderr, tail(r)
    r = run_script(repo, STAGES="C")
    assert r.returncode == 1 and "missing" in r.stderr and "stage B" in r.stderr, tail(r)
    assert {p.name for p in root.iterdir()} == {"_dry"}


def real_context(repo: Path, gate: str = "GATE_FT_PASS") -> None:
    """The registered layout's Gate-FT context, without data: selection, gate_ft.json and three complete adapters."""
    g = repo / "outputs" / "confrec"
    (g / "gatefix" / "dev").mkdir(parents=True)
    (g / "gatefix" / "dev" / "selection.json").write_text(json.dumps({"gate_ft_prompt": VARIANT}), encoding="utf-8")
    (g / "gateft").mkdir(parents=True)
    (g / "gateft" / "gate_ft.json").write_text(json.dumps({"decision": gate}), encoding="utf-8")
    for s in (0, 1, 2):
        a = g / "gateft" / "adapters" / f"s{s}"
        a.mkdir(parents=True)
        (a / "train_config.json").write_text("{}", encoding="utf-8")
        (a / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    (g / "gateft" / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (g / "gateft" / "gateft_split.json").write_text("{}", encoding="utf-8")
    (repo / "docs" / "sigir").mkdir(parents=True, exist_ok=True)
    (repo / "docs" / "sigir" / "PILOT_LOG.md").write_text("# pilot log without the record\n", encoding="utf-8")


@needs_bash
def test_real_mode_refusals(tmp_path):
    repo = make_repo(tmp_path / "repo")
    r = run_script(repo, dry=False)
    assert r.returncode == 4 and "S6 refused: Gate-FT decision missing" in r.stderr, tail(r)
    real_context(repo, gate="GATE_FT_FAIL")
    r = run_script(repo, dry=False)
    assert r.returncode == 4 and "GATE_FT_FAIL" in r.stderr, tail(r)
    (repo / "outputs" / "confrec" / "gateft" / "gate_ft.json").write_text(json.dumps({"decision": "GATE_FT_PASS"}),
                                                                         encoding="utf-8")
    r = run_script(repo, dry=False, STAGES="A")
    assert r.returncode == 4 and "stage A refused" in r.stderr, tail(r)      # no core record: no scoring
    r = run_script(repo, dry=False, STAGES="D")
    assert r.returncode == 4 and "stage D refused" in r.stderr, tail(r)
    r = run_script(repo, dry=False, VARIANT="V3")
    assert r.returncode == 2 and "section 2 registers no other variant" in r.stderr, tail(r)
    assert not (repo / "outputs" / "confrec" / "ftprune").exists()


@needs_bash
@pytest.mark.parametrize("dry, env, msg", [
    (True, {"STAGES": "A,F"}, "unknown stage"),
    (True, {"RUN_P3": "2"}, "RUN_P3 must be"),
    (True, {"OUT_ROOT": "outputs/confrec/ftprune"}, "never writes to the registered output root"),
    (True, {"OUT_ROOT": "./outputs/confrec/ftprune/"}, "never writes to the registered output root"),
    (False, {"OUT_ROOT": "outputs/confrec/ftprune_other"}, "OUT_ROOT is for DRY_RUN"),
    (False, {"MODEL": "/models/Llama-3.1-8B-Instruct"}, "Gate-FT backbone Qwen3-8B only"),
])
def test_input_guards_refuse_before_anything_is_written(tmp_path, dry, env, msg):
    repo = make_repo(tmp_path / "repo")
    r = run_script(repo, dry=dry, **env)
    assert r.returncode == 2 and msg in r.stderr, tail(r)
    assert not (repo / "outputs").exists()
