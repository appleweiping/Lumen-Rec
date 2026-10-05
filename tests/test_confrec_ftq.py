"""Tests of FT-Q, the item-only teacher control (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_8.md section 2):
src/confrec/ftq_panel.py and scripts/sigir/run_ftq.sh. CPU only, deterministic, no network, no GPU, no model, no torch.

  * the teacher is a pure function of the q-hat and the tie key of the examples (shuffle-invariant, no label in reach, the same order
    as ftprune.removal_mask, two implementations), its label rate equals beta exactly, and the teacher panel differs from train.jsonl
    in candidate_labels only; the stage-1 q-hat files are checked against their manifest and aligned one to one with the TRAIN
    examples, and every misalignment is refused;
  * the training and scoring argument vectors of p0 / p1 equal the real adapters' recorded ones except --train, --out, --seed (and
    --lora / --output), for the Amazon layout and for ML-1M, whose real adapters are Gate-FT's; the recorded-input refusals;
  * the links of the real scores into the FT-Q root (symlinks, never copies; a copy only in DRY_RUN without symlinks), each asserted to
    resolve to its target, and a scores directory that is not a directory of its own refused;
  * the output roots (review finding M1): a DRY_RUN refuses every spelling (// . .. absolute, native, letter case) of the registered
    roots and of anything at, under or above outputs/confrec before it writes anything, a real run takes the registered FT-Q root in
    any spelling and no other directory, no link may lie between the repo root and the files a run writes (symlinks; NTFS junctions
    on a Windows box without the symlink privilege), and DRY_RUN is 0 or 1;
  * the teacher panel on disk is recomputed before any training or scoring and must equal train_q.jsonl and its manifest byte for
    byte (m2: the real labels swapped in with the manifest re-tagged are refused); the E1 rerun-once rule counts only the failures of
    the same run.key (m1); the sidecar report/NOTE_FTQ.txt (m3); the record of three files and the printed teacher sha1 (m5);
  * the script is LF, `bash -n` clean, passes the flag audit, mirrors run_ftgrid.sh's helpers and E1 rule, refuses (wrong dataset,
    model or root, no Gate-FT PASS, a missing real adapter, a missing or stale freeze or FT-Q record, a wrong recorded input);
  * run_ftq.sh DRY_RUN=1 end to end on run_ftgrid.sh's own synthetic worlds of Toys (the Amazon layout) and ML-1M (Gate-FT's adapters),
    real code everywhere except the trainer and scorer stand-ins: the teacher panel, p0 and p1 trained with the recorded recipe and
    scored with the recorded arguments, the links, ftgrid_report with the six models in the FT-Q root, the registered grid files
    (and the nested slot's) never written, a rerun that changes nothing, the E1 rerun-once rule, the stage-order refusals.

Run time: the DRY worlds are built once and cached across sessions (a sha1 of the code that builds them is the key); the two chains
run together, and so do the script runs of the refusal scenarios (every script run is dominated by process starts); the DRY_RUN report
bootstraps 20 resamples, not 200. FTQ_FAST=1 skips every test that runs the script's DRY chain.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.confrec import ftgrid_data as fd
from src.confrec import ftprune
from src.confrec import ftq_panel as fq
from src.confrec import pyes_scorer as ps
from src.confrec import train_lora_offset as tlo

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sigir" / "run_ftq.sh"
PANEL = ROOT / "src" / "confrec" / "ftq_panel.py"
FAST = os.environ.get("FTQ_FAST") == "1"


def _load(name: str):
    """A sibling test module as a library (its helpers; it is never edited here)."""
    spec = importlib.util.spec_from_file_location(f"{name}_for_ftq", ROOT / "tests" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FR = _load("test_confrec_ftgrid_run")      # run_ftgrid.sh's chain helpers, its flag audit, the stand-ins' source
FD = _load("test_confrec_ftgrid_data")     # synthetic rated panels and ftgrid_data's runner
MODEL, VARIANT = "dryrun/Qwen3-8B", "V3"
# the two layouts of the real adapters s0-s2: the grid's (Amazon) and Gate-FT's (ML-1M)
LAYOUTS = {
    "toys": {"adapters": "outputs/confrec/ftgrid/adapters/toys", "train_ref": "outputs/confrec/ftgrid/panels/toys/train.jsonl",
             "split_recipe": True},
    "ml1m": {"adapters": "outputs/confrec/gateft/adapters", "train_ref": "outputs/confrec/gateft/train.jsonl",
             "split_recipe": False}}
QROOT = "outputs/confrec/ftgrid_q/adapters/toys"
TQ_PATH = "outputs/confrec/ftgrid_q/ftq/toys/train_q.jsonl"
SPLIT_RECIPE = {"train": {"overlength": {"micro_bsz": 8, "grad_accum": 4, "max_len_used": 1024}, "examples_written": 15010}}
needs_bash = pytest.mark.skipif(FR.BASH is None, reason=FR.NO_BASH)
needs_chain = pytest.mark.skipif(FR.BASH is None or FAST, reason=FR.NO_BASH if FR.BASH is None else "FTQ_FAST=1")


def sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def sha1_file(p) -> str:
    return sha1(Path(p).read_bytes())


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jsonl(path) -> list:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


# ================================================================ 1. the teacher (a pure function)
def test_the_tie_key_is_ftprunes_seed_zero_key():
    assert ftprune.TIE_SEED == fq.TIE_SEED == 0
    assert ftprune.tie_key("u::i") == hashlib.sha1(b"tie:0:u::i").hexdigest() == ftprune.hkey("tie", 0, "u::i")


def test_the_teacher_labels_the_k_largest_q_hat_one_and_breaks_ties_by_the_tie_key():
    q = [3.0, 4.5, 4.5, 2.0, 4.5, 3.0]
    tie = ["d", "b", "a", "z", "c", "e"]
    assert fq.teacher_order(q, tie) == [2, 1, 4, 0, 5, 3]                # 4.5: a, b, c; 3.0: d, e; 2.0: z
    assert fq.teacher_labels(q, tie, 2) == [0, 1, 1, 0, 0, 0]
    assert fq.teacher_labels(q, tie, 4) == [1, 1, 1, 0, 1, 0]
    assert fq.teacher_labels(q, tie, 0) == [0] * 6 and fq.teacher_labels(q, tie, 6) == [1] * 6
    nan = float("nan")                                                    # a non-finite q-hat sorts after every finite one
    assert fq.teacher_order([nan, 1.0, 2.0, nan], ["a", "b", "c", "d"]) == [2, 1, 0, 3]
    assert fq.teacher_labels([nan, 1.0, 2.0, nan], ["a", "b", "c", "d"], 3) == [1, 1, 1, 0]
    with pytest.raises(fq.FtqError, match="cannot label 7 of 6"):
        fq.teacher_labels(q, tie, 7)
    with pytest.raises(fq.FtqError, match="tie keys"):
        fq.teacher_labels(q, tie[:5], 2)


def random_case(rng: random.Random):
    n = rng.randint(1, 80)
    q = [round(rng.uniform(1, 5), rng.choice([0, 1, 2])) if rng.random() > 0.1 else float("nan") for _ in range(n)]   # many ties
    tie = [ftprune.tie_key(f"u{rng.randrange(10 ** 6)}::i{j}") for j in range(n)]
    return q, tie, rng.randint(0, n)


def test_the_teacher_is_a_function_of_q_hat_and_the_tie_key_only():
    """Nothing but the q-hat and the tie key of each example reaches the teacher: no label, user, item or time, and not the
    example's position (shuffle-invariance), and two implementations agree."""
    assert list(inspect.signature(fq.teacher_labels).parameters) == ["q_hat", "tie", "k"]
    rng = random.Random(7)
    for _ in range(80):
        q, tie, k = random_case(rng)
        n, base = len(q), fq.teacher_labels(q, tie, k)
        perm = rng.sample(range(n), n)
        shuffled = fq.teacher_labels([q[j] for j in perm], [tie[j] for j in perm], k)
        back = [0] * n
        for i, j in enumerate(perm):
            back[j] = shuffled[i]
        assert back == base and sum(base) == k                            # the label follows the (q-hat, tie key) pair
        assert fq.teacher_labels_lexsort(q, tie, k) == base
        order = fq.teacher_order(q, tie)
        first = set(order[:k])
        fin = [j for j in range(n) if math.isfinite(q[j])]
        for a in first:                                                   # every 1 precedes every 0 in (q-hat, tie key)
            for b in set(range(n)) - first:
                if math.isfinite(q[a]) and math.isfinite(q[b]):
                    assert (-q[a], tie[a]) < (-q[b], tie[b])
                else:
                    assert math.isfinite(q[a]) or not math.isfinite(q[b])
        assert len(fin) == sum(math.isfinite(v) for v in q)


def test_the_teacher_order_is_ftprunes_removal_order_for_one_class():
    rng = random.Random(3)
    for _ in range(30):
        q, tie, _ = random_case(rng)
        n = len(q)
        removed = ftprune.removal_mask(np.array(q, float), np.ones(n, int), tie)
        assert {j for j in range(n) if removed[j]} == set(fq.teacher_order(q, tie)[:ftprune.n_remove(n)])


def test_the_teacher_rate_equals_beta_exactly_for_every_class_balance():
    rng = random.Random(1)
    for n in (1, 2, 3, 7, 100, 1013, 15010, 15353, 15695, 23348, 100003):
        counts = range(n + 1) if n <= 1013 else sorted({0, 1, n // 3, n // 2, n - 1, n} | {rng.randrange(n + 1) for _ in range(300)})
        for positives in counts:
            beta = positives / n
            k = fq.teacher_k(beta, n)
            assert k == positives and k / n == beta


# ================================================================ 2. the teacher panel (build)
def toys_panel(tmp_path: Path, name: str = "p"):
    """A Toys-shaped split of ftgrid_data on a synthetic rated panel (items repeat, so q-hat values tie): (report, out dir)."""
    rows = FD.random_panel(40, n_cands=8, n_items=12, seed=3)
    return FD.run(tmp_path, rows, "--n_train", 30, domain="toys", name=name)


def q_of(examples, bucket: int = 8) -> list:
    """A q-hat that depends on the item only (coarse levels, so there are many ties)."""
    return [3.0 + 0.25 * (int(hashlib.sha1(e.item.encode()).hexdigest()[:2], 16) % bucket) for e in examples]


def write_stage1(dest: Path, train: Path, split: Path | None, domain: str = "toys", q: list | None = None,
                 nan_every: int = 0) -> tuple:
    """The nested slot's stage-1 files of a train.jsonl in the real format (train_lora_offset's own writers and manifest keys)."""
    dest.mkdir(parents=True, exist_ok=True)
    ex = tlo.panel_examples(tlo.read_rows(train), "train.jsonl")
    q = list(q_of(ex) if q is None else q)
    if nan_every:
        for j in range(0, len(q), nan_every):
            q[j] = float("nan")
    pm ={"mean_prior_shrunk": q, "mean_prior": q, "n_prior": [4.0] * len(q), "global_prior": [3.5] * len(q)}
    consts = tlo.standardisation(q)
    data = tlo.qhat_csv_bytes(ex, pm, tlo.zscore(q, consts))
    (dest / "train_qhat.csv.gz").write_bytes(data)
    man = {"format": tlo.MANIFEST_FORMAT, "domain": domain, "standardisation": consts,
           "train": {"file": "train.jsonl", "sha1": sha1_file(train), "n_examples": len(ex)},
           "files": {"train_qhat.csv.gz": sha1(data)}, "split_sha1": sha1_file(split) if split else None}
    (dest / "qhat_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return dest / "train_qhat.csv.gz", dest / "qhat_manifest.json"


def retag(qman: Path, qcsv: Path, data: bytes) -> None:
    """Replace the q-hat file's bytes and record their sha1 in the manifest (so only the content is wrong, not the hash)."""
    qcsv.write_bytes(data)
    m = read_json(qman)
    m["files"]["train_qhat.csv.gz"] = sha1(data)
    qman.write_text(json.dumps(m), encoding="utf-8")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ftq_build")
    rep, out = toys_panel(tmp)
    qcsv, qman = write_stage1(tmp / "qh", out / "train.jsonl", out / "ftgrid_split.json")
    man = fq.build(out / "train.jsonl", qcsv, qman, tmp / "ftq", domain="toys", split=out / "ftgrid_split.json")
    return {"tmp": tmp, "out": out, "qcsv": qcsv, "qman": qman, "ftq": tmp / "ftq", "man": man, "rep": rep}


def teacher_inputs(b: dict) -> tuple:
    rows = jsonl(b["out"] / "train.jsonl")
    ex = ftprune.train_examples(rows)
    q = tlo.read_qhat(b["qcsv"])
    return rows, ex, [row["q_hat"] for row in q.values()], [ftprune.tie_key(k) for k in ex["key"]]


def test_the_teacher_panel_changes_only_candidate_labels_and_is_the_registered_teacher(built):
    rows, ex, q, tie = teacher_inputs(built)
    teacher = jsonl(built["ftq"] / "train_q.jsonl")
    assert len(rows) == len(teacher) > 20
    flat = []
    for r, t in zip(rows, teacher):
        assert list(r) == list(t)                                         # the same keys in the same order
        assert all(r[f] == t[f] for f in r if f != "candidate_labels")    # histories, ratings, ids, times, order: unchanged
        assert all(isinstance(y, int) and y in (0, 1) for y in t["candidate_labels"])
        flat += t["candidate_labels"]
    real = [int(y) for r in rows for y in r["candidate_labels"]]
    n, k = len(real), sum(real)
    assert flat == fq.teacher_labels(q, tie, k) and sum(flat) == k and sum(flat) / n == k / n   # the rate is beta exactly
    assert flat != real                                                   # the teacher is not the real label
    assert (built["ftq"] / "train_q.jsonl").read_bytes() != (built["out"] / "train.jsonl").read_bytes()
    man = built["man"]
    assert (man["n"], man["k"], man["beta"], man["n_real_positives"], man["n_teacher_positives"]) == (n, k, k / n, k, k)
    assert man["train_q"]["sha1"] == sha1_file(built["ftq"] / "train_q.jsonl")


def test_the_manifest_records_the_hashes_the_rate_the_checks_and_the_code(built):
    man = read_json(built["ftq"] / "train_q.manifest.json")
    assert man == built["man"] and man["format"] == fq.MANIFEST_FORMAT and man["domain"] == "toys"
    assert man["train"]["sha1"] == sha1_file(built["out"] / "train.jsonl") == built["rep"]["files"]["train.jsonl"]
    assert man["qhat"]["sha1"] == sha1_file(built["qcsv"]) and man["qhat"]["manifest_sha1"] == sha1_file(built["qman"])
    assert man["qhat"]["manifest_train_sha1"] == man["train"]["sha1"]     # the nested slot's manifest names this train.jsonl
    assert man["teacher"]["tie_seed"] == 0 and man["n_qhat_nonfinite"] == 0
    assert man["code_sha1"] == {"ftq_panel.py": sha1_file(PANEL), "ftprune.py": sha1_file(ftprune.__file__),
                                "train_lora_offset.py": sha1_file(tlo.__file__),
                                "ftgrid_data.py": sha1_file(fd.__file__)}
    assert man["split"]["T"] == built["rep"]["T"] and man["split"]["train_sha1_recorded"] == man["train"]["sha1"]
    assert all(v is True for v in man["checks"].values())
    assert 0 <= man["agreement_with_real_labels"]["share_teacher_equals_real_label"] <= 1
    raw = (built["ftq"] / "train_q.manifest.json").read_bytes()
    assert b"\r" not in raw and b"NaN" not in raw and b"\\" not in raw and b":/" not in raw     # strict JSON, LF, no path


def test_a_rerun_is_byte_identical_and_leaves_the_files_untouched(built, tmp_path):
    again = tmp_path / "again"
    fq.build(built["out"] / "train.jsonl", built["qcsv"], built["qman"], again, domain="toys",
             split=built["out"] / "ftgrid_split.json")
    for name in ("train_q.jsonl", "train_q.manifest.json"):
        assert (again / name).read_bytes() == (built["ftq"] / name).read_bytes()
    before = {n: (built["ftq"] / n).stat().st_mtime_ns for n in ("train_q.jsonl", "train_q.manifest.json")}
    fq.build(built["out"] / "train.jsonl", built["qcsv"], built["qman"], built["ftq"], domain="toys",
             split=built["out"] / "ftgrid_split.json")                    # onto the existing files
    assert {n: (built["ftq"] / n).stat().st_mtime_ns for n in before} == before     # untouched: the skip rules hold
    assert not list(built["ftq"].glob("*.tmp"))


def permuted_real_labels(out: Path, dest: Path, seed: int) -> Path:
    """train.jsonl with its real labels shuffled among all examples (the same count of positives), in ftgrid_data's format."""
    rows = jsonl(out / "train.jsonl")
    flat = [y for r in rows for y in r["candidate_labels"]]
    random.Random(seed).shuffle(flat)
    pos, new = 0, []
    for r in rows:
        m = len(r["candidate_labels"])
        new.append(dict(r, candidate_labels=flat[pos:pos + m]))
        pos += m
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "train.jsonl").write_bytes(b"".join(fd.row_bytes(r) for r in new))
    return dest / "train.jsonl"


def test_the_teacher_does_not_use_the_label_of_its_own_example(built, tmp_path):
    """Other real labels (the same number of positives), the same q-hat and tie keys: the identical teacher; one more real
    positive: k grows by one and the teacher gains exactly the next example in its order."""
    base = [y for r in jsonl(built["ftq"] / "train_q.jsonl") for y in r["candidate_labels"]]
    other = permuted_real_labels(built["out"], tmp_path / "o1", seed=5)
    qcsv, qman = write_stage1(tmp_path / "q1", other, None)
    man = fq.build(other, qcsv, qman, tmp_path / "f1", domain="toys")
    got = [y for r in jsonl(tmp_path / "f1" / "train_q.jsonl") for y in r["candidate_labels"]]
    real_other = [y for r in jsonl(other) for y in r["candidate_labels"]]
    real_base = [y for r in jsonl(built["out"] / "train.jsonl") for y in r["candidate_labels"]]
    assert got == base and real_other != real_base and man["k"] == built["man"]["k"]
    rows = jsonl(built["out"] / "train.jsonl")                            # one more real positive
    j = real_base.index(0)
    flipped = real_base.copy()
    flipped[j] = 1
    pos, new = 0, []
    for r in rows:
        m = len(r["candidate_labels"])
        new.append(dict(r, candidate_labels=flipped[pos:pos + m]))
        pos += m
    (tmp_path / "o2").mkdir()
    (tmp_path / "o2" / "train.jsonl").write_bytes(b"".join(fd.row_bytes(r) for r in new))
    qcsv, qman = write_stage1(tmp_path / "q2", tmp_path / "o2" / "train.jsonl", None)
    man2 = fq.build(tmp_path / "o2" / "train.jsonl", qcsv, qman, tmp_path / "f2", domain="toys")
    more = [y for r in jsonl(tmp_path / "f2" / "train_q.jsonl") for y in r["candidate_labels"]]
    assert man2["k"] == built["man"]["k"] + 1 and sum(more) == sum(base) + 1
    assert all(m >= b for m, b in zip(more, base)) and [i for i in range(len(base)) if more[i] != base[i]] == [
        fq.teacher_order(*teacher_inputs(built)[2:4])[built["man"]["k"]]]


def test_non_finite_q_hat_values_sort_after_every_finite_one(tmp_path):
    rep, out = toys_panel(tmp_path)
    qcsv, qman = write_stage1(tmp_path / "q", out / "train.jsonl", None, nan_every=7)
    man = fq.build(out / "train.jsonl", qcsv, qman, tmp_path / "f", domain="toys")
    labels = [y for r in jsonl(tmp_path / "f" / "train_q.jsonl") for y in r["candidate_labels"]]
    q = [row["q_hat"] for row in tlo.read_qhat(qcsv).values()]
    assert man["n_qhat_nonfinite"] == sum(not math.isfinite(v) for v in q) > 0
    n_finite = sum(math.isfinite(v) for v in q)
    if man["k"] <= n_finite:
        assert all(labels[j] == 0 for j in range(len(q)) if not math.isfinite(q[j]))


def test_every_misalignment_of_the_stage_1_files_is_refused(built, tmp_path):
    out, split = built["out"], built["out"] / "ftgrid_split.json"
    train = out / "train.jsonl"

    def fresh_stage1(name: str):
        d = tmp_path / name
        d.mkdir()
        shutil.copy2(built["qcsv"], d / "train_qhat.csv.gz")
        shutil.copy2(built["qman"], d / "qhat_manifest.json")
        return d / "train_qhat.csv.gz", d / "qhat_manifest.json"

    def csv_rows(path: Path):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            return f.read().split("\n")

    def attempt(qcsv, qman, **kw):
        return fq.build(train, qcsv, qman, tmp_path / "x", domain=kw.pop("domain", "toys"), **kw)

    qcsv, qman = fresh_stage1("swap")                                    # two rows swapped, the manifest re-tagged
    lines = csv_rows(qcsv)
    lines[1], lines[2] = lines[2], lines[1]
    retag(qman, qcsv, tlo.gz_bytes("\n".join(lines)))
    with pytest.raises(fq.FtqError, match="align one to one"):
        attempt(qcsv, qman)
    qcsv, qman = fresh_stage1("sha")                                     # a changed file, the manifest not re-tagged
    qcsv.write_bytes(tlo.gz_bytes("\n".join(csv_rows(qcsv)) + "\n"))      # one more line: other bytes, the old sha1 recorded
    with pytest.raises(fq.FtqError, match="sha1"):
        attempt(qcsv, qman)
    qcsv, qman = fresh_stage1("trainsha")                                # a manifest that standardised another train.jsonl
    m = read_json(qman)
    m["train"]["sha1"] = "0" * 40
    qman.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(fq.FtqError, match="train.jsonl the manifest standardised"):
        attempt(qcsv, qman)
    qcsv, qman = fresh_stage1("label")                                   # a q-hat row with another label
    lines = csv_rows(qcsv)
    cells = lines[3].split(",")
    cells[5] = str(1 - int(cells[5]))
    lines[3] = ",".join(cells)
    retag(qman, qcsv, tlo.gz_bytes("\n".join(lines)))
    with pytest.raises(fq.FtqError, match="another timestamp / label"):
        attempt(qcsv, qman)
    qcsv, qman = fresh_stage1("short")                                   # a row missing
    lines = [x for x in csv_rows(qcsv) if x]
    retag(qman, qcsv, tlo.gz_bytes("\n".join(lines[:-1]) + "\n"))
    with pytest.raises(fq.FtqError, match="rows for"):
        attempt(qcsv, qman)
    qcsv, qman = fresh_stage1("misc")
    with pytest.raises(fq.FtqError, match="manifest of 'toys', not 'ml1m'"):
        attempt(qcsv, qman, domain="ml1m")
    m = read_json(qman)
    m["split_sha1"] = "1" * 40
    qman.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(fq.FtqError, match="another ftgrid_split.json"):
        attempt(qcsv, qman, split=split)
    other = {**read_json(split), "domain": "ml1m"}
    (tmp_path / "split_ml1m.json").write_text(json.dumps(other), encoding="utf-8")
    qcsv, qman = write_stage1(tmp_path / "okq", train, None)
    with pytest.raises(fq.FtqError, match="split of 'ml1m'"):
        attempt(qcsv, qman, split=tmp_path / "split_ml1m.json")
    other = {**read_json(split), "files": {"train.jsonl": "2" * 40}}
    (tmp_path / "split_sha.json").write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(fq.FtqError, match="records for train.jsonl"):
        attempt(qcsv, qman, split=tmp_path / "split_sha.json")
    assert not (tmp_path / "x").exists()                                 # a refused build writes nothing


def test_a_train_file_that_is_not_the_registered_one_is_refused(built, tmp_path):
    train = jsonl(built["out"] / "train.jsonl")
    odd = [dict(r) for r in train]
    odd[0]["candidate_titles"] = ["Café " + x for x in odd[0]["candidate_titles"]]
    (tmp_path / "odd.jsonl").write_bytes(b"".join((json.dumps(r) + "\n").encode("utf-8") for r in odd))   # ASCII-escaped
    qcsv, qman = built["qcsv"], built["qman"]
    with pytest.raises(fq.FtqError, match="not in the format ftgrid_data writes"):
        fq.build(tmp_path / "odd.jsonl", qcsv, qman, tmp_path / "o1", domain="toys")
    (tmp_path / "blank.jsonl").write_bytes((built["out"] / "train.jsonl").read_bytes() + b"\n\n")
    with pytest.raises(fq.FtqError, match="blank lines"):
        fq.build(tmp_path / "blank.jsonl", qcsv, qman, tmp_path / "o2", domain="toys")
    (tmp_path / "empty.jsonl").write_bytes(b"")
    with pytest.raises(fq.FtqError, match="no row"):
        fq.build(tmp_path / "empty.jsonl", qcsv, qman, tmp_path / "o3", domain="toys")
    bad = [dict(r) for r in train]
    del bad[0]["candidate_labels"]
    (tmp_path / "bad.jsonl").write_bytes(b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in bad))
    with pytest.raises(fq.FtqError, match="rated-panel row"):
        fq.build(tmp_path / "bad.jsonl", qcsv, qman, tmp_path / "o4", domain="toys")
    assert fq.main(["build", "--domain", "toys", "--train", str(tmp_path / "nope.jsonl"), "--qhat", str(qcsv), "--qmanifest",
                    str(qman), "--out_dir", str(tmp_path / "o5")]) == 1           # a missing input is an error (exit 1)
    assert fq.main(["build", "--domain", "beauty", "--train", "a", "--qhat", "b", "--qmanifest", "c", "--out_dir", "d"]) == 2
    assert fq.main(["build", "--domain", "toys"]) == 2                            # usage
    assert not any((tmp_path / f"o{k}").exists() for k in range(1, 6))


def test_the_verification_catches_every_kind_of_corruption(built):
    rows, ex, q, tie = teacher_inputs(built)
    labels = [y for r in jsonl(built["ftq"] / "train_q.jsonl") for y in r["candidate_labels"]]
    good = jsonl(built["ftq"] / "train_q.jsonl")
    assert fq.verify_teacher_rows(rows, good, labels)["examples"] == len(labels)

    def corrupt(fn):
        bad = json.loads(json.dumps(good))
        fn(bad)
        return bad

    cases = (("another label", lambda b: b[0]["candidate_labels"].__setitem__(0, 1 - b[0]["candidate_labels"][0]),
              "not the teacher's"),
             ("a rating", lambda b: b[1]["candidate_ratings"].__setitem__(0, 9.0), r"changed \(only candidate_labels"),
             ("a title", lambda b: b[2]["candidate_titles"].__setitem__(0, "x"), r"changed \(only candidate_labels"),
             ("a key", lambda b: b[3].__setitem__("extra", 1), "other keys"),
             ("a length", lambda b: b[4]["candidate_labels"].pop(), "changed its length|not the teacher"),
             ("a row", lambda b: b.pop(), "rows"))
    for name, fn, msg in cases:
        with pytest.raises(fq.FtqError, match=msg):
            fq.verify_teacher_rows(rows, corrupt(fn), labels)
        assert name


def test_verify_teacher_recomputes_the_teacher_and_requires_byte_equality(built, tmp_path, capsys):
    """The reviewer's plant at the module level: train_q.jsonl replaced by the REAL-label train.jsonl with its manifest re-tagged.
    No step marker and no manifest can vouch for a panel that is not the teacher recomputed from train.jsonl and the q-hat file."""
    out, split = built["out"], built["out"] / "ftgrid_split.json"
    args = (out / "train.jsonl", built["qcsv"], built["qman"])

    def fresh(name: str) -> Path:
        d = tmp_path / name
        shutil.copytree(built["ftq"], d)
        return d

    def verify(d: Path):
        return fq.verify_teacher(*args, d, domain="toys", split=split)

    def swap_panel(d: Path, data: bytes) -> None:                              # the panel replaced and its manifest re-tagged
        (d / "train_q.jsonl").write_bytes(data)
        m = read_json(d / "train_q.manifest.json")
        m["train_q"]["sha1"] = sha1(data)
        (d / "train_q.manifest.json").write_bytes(fq.json_bytes(m))

    assert verify(built["ftq"]) == built["man"]                                # what build wrote is the recomputed teacher
    assert fq.compute_teacher(*args, domain="toys", split=split)[1] == built["man"]
    d = fresh("real_labels")                                                   # the plant: the real labels as the teacher panel
    swap_panel(d, (out / "train.jsonl").read_bytes())
    with pytest.raises(fq.FtqError, match="train_q.jsonl is not the teacher recomputed"):
        verify(d)
    d = fresh("one_label")                                                     # one label flipped
    rows = jsonl(d / "train_q.jsonl")
    rows[0]["candidate_labels"][0] = 1 - rows[0]["candidate_labels"][0]
    swap_panel(d, b"".join(fd.row_bytes(r) for r in rows))
    with pytest.raises(fq.FtqError, match="train_q.jsonl is not the teacher recomputed"):
        verify(d)
    d = fresh("a_byte")                                                        # one more byte
    swap_panel(d, (built["ftq"] / "train_q.jsonl").read_bytes() + b"\n")
    with pytest.raises(fq.FtqError, match="is not the teacher recomputed"):
        verify(d)
    for name, edit in (("beta", lambda m: m.update(beta=0.5)), ("code", lambda m: m["code_sha1"].update({"ftq_panel.py": "0" * 40})),
                       ("key", lambda m: m.pop("checks")), ("hash", lambda m: m["qhat"].update(sha1="1" * 40))):
        d = fresh("manifest_" + name)                                          # a manifest that says something else
        m = read_json(d / "train_q.manifest.json")
        edit(m)
        (d / "train_q.manifest.json").write_bytes(fq.json_bytes(m))
        with pytest.raises(fq.FtqError, match="train_q.manifest.json is not the teacher recomputed"):
            verify(d)
    for name in ("train_q.jsonl", "train_q.manifest.json"):
        d = fresh("missing_" + name)
        (d / name).unlink()
        with pytest.raises(fq.FtqError, match=r"missing \(stage 1"):
            verify(d)
    rows_q = tlo.panel_examples(tlo.read_rows(out / "train.jsonl"), "train.jsonl")  # another, consistent q-hat file: the stored teacher
    qcsv, qman = write_stage1(tmp_path / "q_other", out / "train.jsonl", split, q=list(reversed(q_of(rows_q))))   # is not its teacher
    with pytest.raises(fq.FtqError, match="is not the teacher recomputed"):
        fq.verify_teacher(out / "train.jsonl", qcsv, qman, built["ftq"], domain="toys", split=split)
    base = ["verify_teacher", "--domain", "toys", "--train", str(out / "train.jsonl"), "--qhat", str(built["qcsv"]), "--qmanifest",
            str(built["qman"]), "--split", str(split), "--out_dir"]
    capsys.readouterr()
    assert fq.main(base + [str(built["ftq"])]) == 0 and "byte for byte the teacher recomputed" in capsys.readouterr().out
    assert fq.main(base + [str(tmp_path / "real_labels")]) == 1 and "is not the teacher recomputed" in capsys.readouterr().err
    assert fq.main(["verify_teacher", "--domain", "toys"]) == 2                       # usage
    assert fq.main(["verify_teacher", "--domain", "beauty", "--train", "a", "--qhat", "b", "--qmanifest", "c", "--out_dir", "d"]) == 2
    assert fq.main(base[:-1] + ["--out_dir", str(tmp_path / "nowhere")]) == 1       # no panel: stage 1 has not run


def test_the_teacher_is_described_as_using_no_label_of_the_example_and_no_user_preference_information(built):
    """q-hat leaves the example's own user out of the global prior (a user-free recomputation moves 2 of 23,348 ML-1M labels), so the
    claim is 'no label of the example and no user preference information', not 'no user information'."""
    phrase = "no label of the example and no user preference information"
    flat = lambda text: re.sub(r"\s+", " ", re.sub(r"(?m)^\s*#", "", text))      # a comment or docstring, rewrapped
    for name, text in (("module", PANEL.read_text(encoding="utf-8")), ("script", SCRIPT.read_text(encoding="utf-8"))):
        assert phrase in flat(text), name
        assert "no user " + "information" not in flat(text), name
    assert phrase in built["man"]["teacher"]["information"] and "no user " + "information" not in json.dumps(built["man"])
    assert "no label of the example" in inspect.getdoc(fq.teacher_labels) and "no user preference information" in flat(
        inspect.getdoc(fq.teacher_labels))


def test_the_header_names_what_is_used_as_it_is_and_the_files_the_record_needs():
    head = re.sub(r"\s+", " ", re.sub(r"(?m)^\s*#", "", SCRIPT.read_text(encoding="utf-8").split("set -euo pipefail")[0]))
    assert "the sha1 of this script, of src/confrec/ftq_panel.py and of tests/test_confrec_ftq.py are in docs/sigir/PILOT_LOG.md" in head
    assert "Imported or used as they are, and recorded separately (not part of that record): src/confrec/ftprune.py (tie_key, " \
           "train_examples" in head and "src/confrec/ftgrid_extra.py" in head
    assert "DRY_RUN 0 or 1 (any other value is refused with exit 2)" in head and "NOTE_FTQ.txt" in head
    assert "tmp_outputs/ftq_dryrun" in head
    # review round 3: the DRY_RUN guard is an allow-list on lower-case canonical forms; a relative OUT_ROOT resolves from the repo root
    assert "The DRY_RUN guard is an allow-list" in head and "lie under tmp_outputs of the repo, or outside the repo's parent directory" in head
    assert "A relative OUT_ROOT resolves from the repo root" in head and "every comparison is made on lower-case canonical forms" in head
    # m1: what the sweep and the removal of stale temporary files cover, and what they do not
    assert "find OUT_ROOT -type l" in head and "report/D.json.tmp" in head and "like/*.tmp" in head and "are not enumerated" in head
    # m4: the sticky FAILED_INTEGRITY and its logged override
    assert "FAILED_INTEGRITY is sticky per seed" in head and "FTQ_ALLOW_RETRY" in head and "NOTE_FTQ_overrides.txt" in head


def test_the_ftq_scores_directory_is_a_directory_of_its_own_and_every_link_is_asserted(tmp_path, monkeypatch):
    real = make_real_scores(tmp_path / "grid" / "scores" / "toys")
    for q in (real, real / "sub", tmp_path / "grid" / "scores", tmp_path / "grid"):     # the registered scores, below it, above it
        with pytest.raises(fq.FtqError, match="root of its own"):
            fq.link_models(real, q, ["s0"])
    assert not (real / "sub").exists() and not (real / "s0" / "s0").exists()               # nothing was made
    assert fq.main(["link", "--real_scores", str(real), "--q_scores", str(real)]) == 1
    probe = tmp_path / "probe"
    (probe / "t").mkdir(parents=True)
    if link_dir(probe / "l", probe / "t"):                                                # where directory links can be made:
        unlink_dir(probe / "l")
        assert link_dir(tmp_path / "alias", real)                                         # an alias of the registered scores ...
        with pytest.raises(fq.FtqError, match="root of its own"):
            fq.link_models(real, tmp_path / "alias", ["s0"])
        with pytest.raises(fq.FtqError, match="root of its own"):
            fq.link_models(real, tmp_path / "alias" / "sub", ["s0"])
        unlink_dir(tmp_path / "alias")
        (tmp_path / "elsewhere").mkdir()
        assert link_dir(tmp_path / "q_link", tmp_path / "elsewhere")                      # ... and a scores directory that is a link
        with pytest.raises(fq.FtqError, match="is itself a link"):
            fq.link_models(real, tmp_path / "q_link", ["s0"])
        unlink_dir(tmp_path / "q_link")
    monkeypatch.setattr(os, "symlink", lambda src, dst, target_is_directory=False: Path(dst).mkdir())   # a link that is none
    with pytest.raises(fq.FtqError, match="was made but resolves to"):
        fq.link_models(real, tmp_path / "q" / "scores" / "toys", ["s0"])


def test_the_build_command_prints_the_manifest_and_the_module_needs_no_torch(built, tmp_path, capsys):
    argv = ["build", "--domain", "toys", "--train", str(built["out"] / "train.jsonl"), "--qhat", str(built["qcsv"]),
            "--qmanifest", str(built["qman"]), "--split", str(built["out"] / "ftgrid_split.json"), "--out_dir",
            str(tmp_path / "ftq")]
    assert fq.main(argv) == 0
    cap = capsys.readouterr()
    assert json.loads(cap.out) == built["man"]
    assert f"labels {built['man']['k']} of {built['man']['n']} TRAIN examples 1" in cap.err
    code = ("import sys; import src.confrec.ftq_panel; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'transformers', 'vllm', 'peft', 'numpy')]; "
            "assert not bad, bad")                                        # the light commands need not even numpy
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stderr


# ================================================================ 3. the recorded recipe (both layouts)
@pytest.fixture(scope="module")
def trainer_parser():
    """The real argparse parser of train_lora_yesno (torch replaced by stand-in modules for the import)."""
    with FR.stub_torch():
        parser = FR._parser("src.confrec.train_lora_yesno")
    return parser


def sft_config(parser, k: int, layout: str, *, bsz: int = 8, accum: int = 4, max_len: int = 1024, model: str = MODEL,
               variant: str = VARIANT, n_examples: int = 15010) -> dict:
    """The train_config.json train_lora_yesno writes for a real adapter s<k> of the layout (the flags of run_ftgrid.sh's
    train_adapter, and of run_gateft.sh for ML-1M: --train --model --out --variant --seed --max_len --bsz --grad_accum)."""
    lay = LAYOUTS[layout]
    argv = ["--train", lay["train_ref"], "--model", model, "--out", f"{lay['adapters']}/s{k}", "--variant", variant, "--seed",
            str(k), "--max_len", str(max_len), "--bsz", str(bsz), "--grad_accum", str(accum)]
    cfg = {**vars(parser.parse_args(argv)), "loss": "last_token", "hist_len_used": 20, "panel_kind": "rated",
           "max_history_len_in_panel": 20, "n_examples": n_examples, "n_skipped_overlength": 0}
    return json.loads(json.dumps(cfg))


def flag_pairs(argv: list) -> dict:
    assert len(argv) % 2 == 0 and all(a.startswith("--") for a in argv[::2]), argv   # every flag takes one value
    assert len(set(argv[::2])) == len(argv) // 2, argv                               # no flag twice
    return dict(zip(argv[::2], argv[1::2]))


def recipe(cfgs, layout, split=SPLIT_RECIPE, **kw):
    lay = LAYOUTS[layout]
    args = dict(adapters=lay["adapters"], model=MODEL, variant=VARIANT, train_ref=lay["train_ref"], split=split,
                split_recipe=lay["split_recipe"])
    args.update(kw)
    return fq.check_recipe(cfgs, **args)


def good_cfgs(parser, layout):
    return {k: sft_config(parser, k, layout) for k in (0, 1, 2)}


def test_train_keys_are_exactly_the_trainers_recorded_arguments(trainer_parser):
    dests = {a.dest for a in trainer_parser._actions if a.option_strings and a.dest != "help"}
    assert dests == set(fq.TRAIN_KEYS)
    cfg = sft_config(trainer_parser, 0, "toys")
    assert set(fq.TRAIN_KEYS) <= set(cfg) and set(fq.TRAIN_FACTS) <= set(cfg)


@pytest.mark.parametrize("layout, k, bsz, accum, max_len", [("toys", 0, 8, 4, 1024), ("toys", 1, 4, 8, 1280),
                                                            ("ml1m", 0, 8, 4, 1024), ("ml1m", 1, 8, 4, 1024)])
def test_the_ftq_training_vector_equals_the_recorded_vector_except_the_three_flags(trainer_parser, layout, k, bsz, accum,
                                                                                    max_len):
    """p_k is trained with the arguments of the real adapters' train_config.json (the grid's, or Gate-FT's for ML-1M); only --train,
    --out and --seed are replaced. Parsed by the real trainer parser the two vectors agree on every other argument."""
    cfgs = {i: sft_config(trainer_parser, i, layout, bsz=bsz, accum=accum, max_len=max_len) for i in (0, 1, 2)}
    split = {"train": {"overlength": {"micro_bsz": bsz, "grad_accum": accum, "max_len_used": max_len}, "examples_written": 15010}}
    s0 = recipe(cfgs, layout, split=split)
    assert s0 == cfgs[0]
    out = f"{QROOT}/p{k}"
    argv = fq.train_argv(s0, train=TQ_PATH, out=out, seed=k)
    got = vars(trainer_parser.parse_args(argv))
    lay = LAYOUTS[layout]
    recorded = vars(trainer_parser.parse_args(                            # the vector of s0's train call, re-parsed
        ["--train", lay["train_ref"], "--model", MODEL, "--out", f"{lay['adapters']}/s0", "--variant", VARIANT, "--seed", "0",
         "--max_len", str(max_len), "--bsz", str(bsz), "--grad_accum", str(accum)]))
    keep = [x for x in fq.TRAIN_KEYS if x not in fq.REPLACED]
    assert set(got) == set(fq.TRAIN_KEYS)
    assert {x: got[x] for x in keep} == {x: recorded[x] for x in keep} == {x: s0[x] for x in keep}
    assert {x: got[x] for x in fq.REPLACED} == {"train": TQ_PATH, "out": out, "seed": k}
    pairs = flag_pairs(argv)                                              # the token level: every recorded flag once
    assert set(pairs) == {f"--{x}" for x in fq.TRAIN_KEYS if s0[x] is not None}
    assert fq.recipe_flags(s0) == [t for f, v in pairs.items() if f[2:] not in fq.REPLACED for t in (f, v)]
    assert (pairs["--bsz"], pairs["--grad_accum"], pairs["--max_len"]) == (str(bsz), str(accum), str(max_len))


def test_unset_and_set_optional_flags_round_trip(trainer_parser):
    cfg = sft_config(trainer_parser, 0, "toys")
    cfg.update(hist_len=20, max_examples=100, epochs=0.5, lr=2.5e-05)
    got = vars(trainer_parser.parse_args(fq.train_argv(cfg, train="a", out="b", seed=3)))
    assert got == {**{x: cfg[x] for x in fq.TRAIN_KEYS}, "train": "a", "out": "b", "seed": 3}
    cfg.update(hist_len=None, max_examples=None)
    pairs = flag_pairs(fq.train_argv(cfg, train="a", out="b", seed=3))
    assert "--hist_len" not in pairs and "--max_examples" not in pairs
    with pytest.raises(fq.FtqError, match="no record of"):
        fq.train_argv({k: v for k, v in cfg.items() if k != "lora_r"}, train="a", out="b", seed=0)


@pytest.mark.parametrize("layout", ["toys", "ml1m"])
def test_a_valid_recipe_is_accepted_and_its_flags_are_the_run_ftgrid_ones(trainer_parser, layout):
    cfgs = good_cfgs(trainer_parser, layout)
    s0 = recipe(cfgs, layout)
    assert fq.recipe_flags(s0) == ["--model", MODEL, "--mode", "standard", "--variant", VARIANT, "--max_len", "1024",
                                   "--epochs", "1.0", "--lr", "0.0001", "--bsz", "8", "--grad_accum", "4", "--lora_r", "16"]
    assert recipe(cfgs, layout, split=None) == s0


def mutate(cfgs: dict, k: int, **kw) -> dict:
    out = {i: dict(c) for i, c in cfgs.items()}
    out[k].update(kw)
    return out


@pytest.mark.parametrize("layout", ["toys", "ml1m"])
@pytest.mark.parametrize("k, change, msg", [
    (0, {"seed": 7}, "seed is 7"), (1, {"out": "elsewhere/s1"}, "out is"), (0, {"train": "other/train.jsonl"}, "train is"),
    (2, {"model": "/m/Llama"}, "model is"), (1, {"variant": "V0"}, "variant is"),
    (1, {"bsz": 4, "grad_accum": 8}, "not trained with one recipe"), (2, {"max_len": 1280}, "not trained with one recipe"),
    (2, {"n_examples": 15009}, "not trained with one recipe")])
def test_adapters_that_do_not_share_one_recipe_are_refused(trainer_parser, layout, k, change, msg):
    with pytest.raises(fq.FtqError, match=msg):
        recipe(mutate(good_cfgs(trainer_parser, layout), k, **change), layout)


@pytest.mark.parametrize("change", [{"epochs": 2.0}, {"lr": 5e-05}, {"lora_r": 8}, {"mode": "mirror"}, {"hist_len": 20},
                                    {"max_examples": 5}])
def test_a_recipe_that_agrees_with_itself_but_is_not_the_registered_one_is_refused(trainer_parser, change):
    cfgs = {i: {**c, **change} for i, c in good_cfgs(trainer_parser, "ml1m").items()}
    with pytest.raises(fq.FtqError, match="registered recipe"):
        recipe(cfgs, "ml1m")


def test_the_recipe_is_checked_against_the_batch_the_split_and_the_examples(trainer_parser):
    cfgs = {k: sft_config(trainer_parser, k, "toys", bsz=8, accum=8) for k in (0, 1, 2)}
    with pytest.raises(fq.FtqError, match="effective batch"):
        recipe(cfgs, "toys", split=None)
    cfgs = good_cfgs(trainer_parser, "toys")
    bad = {"train": {"overlength": {"micro_bsz": 4, "grad_accum": 8, "max_len_used": 1280}, "examples_written": 15010}}
    with pytest.raises(fq.FtqError, match=r"differs from ftgrid_split.json's"):
        recipe(cfgs, "toys", split=bad)                                   # the Amazon layout compares the split's recipe
    assert recipe(good_cfgs(trainer_parser, "ml1m"), "ml1m", split=bad)   # ML-1M (Gate-FT's) does not: only the examples
    with pytest.raises(fq.FtqError, match="not the registered TRAIN file"):
        recipe(good_cfgs(trainer_parser, "ml1m"), "ml1m", split={"train": {**SPLIT_RECIPE["train"], "examples_written": 20000}})
    with pytest.raises(fq.FtqError, match="needs the configs of the adapters s0-s2"):
        recipe({0: cfgs[0], 1: cfgs[1]}, "toys")


def test_the_train_file_the_real_adapters_recorded_must_hold_the_registered_bytes(tmp_path):
    a, b, c = tmp_path / "gateft_train.jsonl", tmp_path / "panel_train.jsonl", tmp_path / "other.jsonl"
    a.write_bytes(b"rows\n")
    b.write_bytes(b"rows\n")
    c.write_bytes(b"other rows\n")
    fq.check_train_file(str(a), str(a))                                   # the same path: nothing to compare
    fq.check_train_file(str(a), str(b))                                   # another path, the same bytes (ML-1M's Gate-FT file)
    with pytest.raises(fq.FtqError, match="differs in bytes"):
        fq.check_train_file(str(a), str(c))
    with pytest.raises(fq.FtqError, match="no such file"):
        fq.check_train_file(str(tmp_path / "gone.jsonl"), str(b))


def test_a_missing_real_adapter_is_named(tmp_path, trainer_parser):
    for k in (0, 1):
        (tmp_path / f"s{k}").mkdir()
        (tmp_path / f"s{k}" / "train_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(fq.FtqError, match=r"missing the real adapter s2"):
        fq.sft_configs(str(tmp_path))
    assert fq.main(["recipe", "--adapters", str(tmp_path), "--model", MODEL, "--variant", VARIANT, "--train_ref", "x"]) == 1
    assert fq.main(["recipe", "--adapters", str(tmp_path)]) == 2          # a usage refusal


@pytest.mark.parametrize("layout", ["toys", "ml1m"])
def test_a_trained_ftq_adapter_must_record_the_real_adapters_arguments(trainer_parser, layout):
    s0 = sft_config(trainer_parser, 0, layout)
    p0 = {**vars(trainer_parser.parse_args(fq.train_argv(s0, train=TQ_PATH, out=f"{QROOT}/p0", seed=0))),
          **{k: s0[k] for k in fq.TRAIN_FACTS}}
    fq.check_same_training(s0, p0, train=TQ_PATH, out=f"{QROOT}/p0", seed=0)
    for change, msg in (({"lr": 2e-4}, "lr"), ({"max_len": 512}, "max_len"), ({"bsz": 4}, "bsz"),
                        ({"n_examples": 15000}, "n_examples"), ({"variant": "V0"}, "variant"),
                        ({"train": LAYOUTS[layout]["train_ref"]}, "train is"), ({"seed": 1}, "seed is"),
                        ({"out": f"{QROOT}/p1"}, "out is")):
        with pytest.raises(fq.FtqError, match=msg):
            fq.check_same_training(s0, {**p0, **change}, train=TQ_PATH, out=f"{QROOT}/p0", seed=0)


# ================================================================ 4. the recorded like pass (both layouts)
@pytest.fixture(scope="module")
def fakes(tmp_path_factory):
    """run_ftgrid.sh's DRY_RUN stand-ins, imported from the source the script writes (the real scorer, a fake model)."""
    d = tmp_path_factory.mktemp("ftq_fakes")
    (d / "ftgrid_fakes.py").write_text(FR.fakes_source(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ftgrid_fakes_for_ftq", d / "ftgrid_fakes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_argv(data, out, lora) -> list:
    """The pyes_scorer call of run_ftgrid.sh's score() for a like pass of an adapter."""
    return ["--data", str(data), "--output", str(out), "--model", MODEL, "--dtype", "float16", "--topk_logprobs", "50",
            "--max_model_len", "4096", "--chunk_users", "100", "--variant", VARIANT, "--readout", "yesno",
            "--questions", "like", "--lora", lora]


@pytest.fixture(scope="module", params=["toys", "ml1m"])
def like_world(request, tmp_path_factory, fakes):
    """A scored like pass of a real adapter s0 on a tiny rated panel, recorded as run_ftgrid.sh records it (run.key and the real
    scorer's report.json), under the adapters directory of the layout (the grid's, or Gate-FT's for ML-1M)."""
    root = tmp_path_factory.mktemp(f"ftq_like_{request.param}")
    panel = FR.write_rows(root / "eval.jsonl", FR.rated_rows(n=6, hist=20))
    adapters = (root / ("gateft/adapters" if request.param == "ml1m" else "adapters")).as_posix()
    scores = (root / "scores").as_posix()
    qadapters = (root / "q/adapters").as_posix()
    for d, name in ((adapters, "s0"), (qadapters, "p0")):
        (Path(d) / name).mkdir(parents=True)
        (Path(d) / name / "adapter_model.safetensors").write_bytes(f"weights of {name}".encode())
    fakes.scorer(score_argv(panel, f"{scores}/s0/like", f"{adapters}/s0"))
    key = f"{sha1_file(panel)} {MODEL} {VARIANT} {fq.weights_sha1(Path(adapters) / 's0')} --lora {adapters}/s0"
    (root / "scores" / "s0" / "like" / "run.key").write_text(key + "\n", encoding="utf-8")
    return {"root": root, "panel": panel, "adapters": adapters, "scores": scores, "key": key, "qadapters": qadapters,
            "layout": request.param}


def scoring_args(w, **kw) -> argparse.Namespace:
    d = dict(command="scoring", scores=w["scores"], adapters=w["adapters"], data=str(w["panel"]), model=MODEL, variant=VARIANT)
    d.update(kw)
    return argparse.Namespace(**d)


def test_the_ftq_scoring_vector_equals_the_recorded_like_pass_except_lora_and_output(like_world, fakes):
    w = like_world
    flags = fq.cmd_scoring(scoring_args(w))
    mine = score_argv(w["panel"], f"{w['scores']}/p0/like", f"{w['qadapters']}/p0")
    sibling = score_argv(w["panel"], f"{w['scores']}/s0/like", f"{w['adapters']}/s0")
    # flags + the literal --data/--output/--model/--lora of the script's score() = the real like pass's vector
    assert flag_pairs(flags) == {f: v for f, v in flag_pairs(sibling).items()
                                 if f not in ("--data", "--output", "--model", "--lora")}
    a = vars(ps.parse_args(["--data", str(w["panel"]), "--output", f"{w['scores']}/p0/like", "--model", MODEL] + flags
                           + ["--lora", f"{w['qadapters']}/p0"]))
    b, c = vars(ps.parse_args(mine)), vars(ps.parse_args(sibling))
    assert a == b and {k: v for k, v in a.items() if k not in ("output", "lora")} == {
        k: v for k, v in c.items() if k not in ("output", "lora")}
    # and what the scorer then records for p0 equals what it recorded for s0 except the adapter
    fakes.scorer(["--data", str(w["panel"]), "--output", f"{w['scores']}/p0/like", "--model", MODEL] + flags
                 + ["--lora", f"{w['qadapters']}/p0"])
    ref = fq.report_config(read_json(f"{w['scores']}/s0/like/report.json"), "s0")
    run = fq.report_config(read_json(f"{w['scores']}/p0/like/report.json"), "p0")
    fq.check_same_scoring(ref, run, lora=f"{w['qadapters']}/p0", where="p0")
    assert ref["lora"] != run["lora"] and {k for k in ref if ref[k] != run[k]} == {"lora"}
    assert fq.main(["verify_scores", "--scores", w["scores"], "--ref_scores", w["scores"], "--adapters", w["qadapters"],
                    "--p_seeds", "0"]) == 0


def test_a_p_run_scored_with_other_arguments_is_caught(like_world, fakes):
    w = like_world
    ref = fq.report_config(read_json(f"{w['scores']}/s0/like/report.json"), "s0")
    for flag, value, key in (("--topk_logprobs", "40", "topk_logprobs"), ("--chunk_users", "50", "chunk_users"),
                             ("--dtype", "bfloat16", "dtype"), ("--swap_k", "2", "swap_k"), ("--hist_len", "0", "hist_len"),
                             ("--seed", "1", "seed")):
        out = f"{w['scores']}/bad_{key}/like"
        fakes.scorer(score_argv(w["panel"], out, f"{w['qadapters']}/p0") + [flag, value])
        run = fq.report_config(read_json(f"{out}/report.json"), "bad")
        with pytest.raises(fq.FtqError, match=key):
            fq.check_same_scoring(ref, run, lora=f"{w['qadapters']}/p0", where="bad")
    with pytest.raises(fq.FtqError, match="scored with"):
        fq.check_same_scoring(ref, {**ref, "lora": "x"}, lora=f"{w['qadapters']}/p0", where="p0")


def record_args(w, **kw) -> dict:
    d = dict(data_sha1=sha1_file(w["panel"]), model=MODEL, variant=VARIANT, lora=f"{w['adapters']}/s0",
             lora_weights_sha1=fq.weights_sha1(Path(w["adapters"]) / "s0"))
    d.update(kw)
    return d


def test_the_recorded_like_pass_is_validated_against_the_registered_one(like_world):
    w = like_world
    cfg = read_json(f"{w['scores']}/s0/like/report.json")["config"]
    assert fq.check_scoring_record(w["key"], cfg, **record_args(w)) == [
        "--dtype", "float16", "--topk_logprobs", "50", "--max_model_len", "4096", "--chunk_users", "100", "--variant",
        VARIANT, "--readout", "yesno", "--questions", "like"]
    key = w["key"].split()
    bad_keys = {"panel": " ".join(["0" * 40] + key[1:]), "model": " ".join([key[0], "/m/Llama"] + key[2:]),
                "variant": " ".join(key[:2] + ["V0"] + key[3:]), "weights": " ".join(key[:3] + ["1" * 40] + key[4:]),
                "extra args": w["key"] + " --swap_k 8", "other adapter": w["key"].replace("/s0", "/s1")}
    for name, text in bad_keys.items():
        with pytest.raises(fq.FtqError, match="run.key"):
            fq.check_scoring_record(text, cfg, **record_args(w))
        assert name


@pytest.mark.parametrize("change, msg", [
    ({"dtype": "bfloat16"}, "dtype"), ({"topk_logprobs": 20}, "topk_logprobs"), ({"max_model_len": 2048}, "max_model_len"),
    ({"chunk_users": 50}, "chunk_users"), ({"readout": "digits"}, "readout"), ({"questions": ["like", "dislike"]}, "questions"),
    ({"swap_k": 8}, "swap_k"), ({"seed": 1}, "seed"), ({"n_users": 10}, "n_users"), ({"chunk_items": 10}, "chunk_items"),
    ({"hist_len": 0}, "hist_len"), ({"data_sha1": "0" * 40}, "data_sha1"), ({"model": "/m/Llama"}, "model"),
    ({"variant": "V0"}, "variant"), ({"lora": "elsewhere/s0"}, "lora"), ({"panel_kind": None}, "panel_kind|not in")])
def test_a_like_pass_that_is_not_the_registered_one_is_refused(like_world, change, msg):
    w = like_world
    cfg = {**read_json(f"{w['scores']}/s0/like/report.json")["config"], **change}
    with pytest.raises(fq.FtqError, match=msg):
        fq.check_scoring_record(w["key"], cfg, **record_args(w))


def test_the_scoring_command_refuses_a_missing_or_stale_like_pass(like_world, tmp_path):
    w = like_world
    assert fq.cmd_scoring(scoring_args(w))
    with pytest.raises(fq.FtqError, match="missing the real like pass of s0"):
        fq.cmd_scoring(scoring_args(w, scores=str(tmp_path / "none")))
    other = FR.write_rows(tmp_path / "eval2.jsonl", FR.rated_rows(n=7, hist=20))        # eval.jsonl changed since the pass
    with pytest.raises(fq.FtqError, match="panel sha1"):
        fq.cmd_scoring(scoring_args(w, data=str(other)))
    root = tmp_path / "copy"
    shutil.copytree(w["root"], root)
    a2, s2 = (root / Path(w["adapters"]).relative_to(w["root"])).as_posix(), (root / "scores").as_posix()
    (root / "scores" / "s0" / "like" / "run.key").write_text(w["key"].replace(w["adapters"], a2) + "\n", encoding="utf-8")
    rep = read_json(root / "scores" / "s0" / "like" / "report.json")
    rep["config"]["lora"] = f"{a2}/s0"
    (root / "scores" / "s0" / "like" / "report.json").write_text(json.dumps(rep), encoding="utf-8")
    assert fq.cmd_scoring(scoring_args(w, scores=s2, adapters=a2))                       # consistent copy: accepted
    (Path(a2) / "s0" / "adapter_model.safetensors").write_bytes(b"retrained")            # the adapter changed since the pass
    with pytest.raises(fq.FtqError, match="adapter weights sha1"):
        fq.cmd_scoring(scoring_args(w, scores=s2, adapters=a2))
    assert fq.main(["scoring", "--scores", s2, "--adapters", a2, "--data", str(w["panel"]), "--model", MODEL,
                    "--variant", VARIANT]) == 1


# ================================================================ 5. provenance, the links, the record, info
def test_adapter_provenance_binds_an_adapter_to_the_teacher_panel_for_both_seeds(trainer_parser, tmp_path, capsys):
    real, q = tmp_path / "real", (tmp_path / "q").as_posix()
    teacher, manifest = tmp_path / "train_q.jsonl", tmp_path / "train_q.manifest.json"
    teacher.write_bytes(b"teacher rows")
    manifest.write_text(json.dumps({"train_q": {"sha1": sha1(b"teacher rows")}}), encoding="utf-8")
    s0 = sft_config(trainer_parser, 0, "toys")
    (real / "s0").mkdir(parents=True)
    (real / "s0" / "train_config.json").write_text(json.dumps(s0), encoding="utf-8")
    cfgs = {}
    for k in (0, 1):
        (Path(q) / f"p{k}").mkdir(parents=True)
        cfgs[k] = {**vars(trainer_parser.parse_args(fq.train_argv(s0, train=str(teacher), out=f"{q}/p{k}", seed=k))),
                   **{x: s0[x] for x in fq.TRAIN_FACTS}}
        (Path(q) / f"p{k}" / "train_config.json").write_text(json.dumps(cfgs[k]), encoding="utf-8")
    args = ["verify_adapter", "--adapters", q, "--ref_adapters", str(real), "--p_seeds", "0,1", "--train", str(teacher),
            "--teacher_manifest", str(manifest)]
    assert fq.main(args) == 1 and "has no ftq.json" in capsys.readouterr().err        # trained by hand: refused
    assert fq.main(args + ["--write"]) == 0
    prov = read_json(Path(q) / "p1" / "ftq.json")
    assert prov["train_q_sha1"] == sha1(b"teacher rows") and prov["adapter"] == "p1" and prov["seed"] == 1
    assert prov["args"] == {x: cfgs[1][x] for x in fq.TRAIN_KEYS}
    assert fq.main(args) == 0                                                          # an existing adapter is accepted
    prov["train_q_sha1"] = "0" * 40                                                    # trained on another teacher panel
    (Path(q) / "p1" / "ftq.json").write_text(json.dumps(prov), encoding="utf-8")
    assert fq.main(args) == 1 and "was trained on a teacher panel" in capsys.readouterr().err
    fq.write_provenance(Path(q) / "p1", p_seed=1, train_q_sha1=sha1(b"teacher rows"), manifest_sha1="x", args={})
    teacher.write_bytes(b"another teacher")                                            # the panel on disk changed
    assert fq.main(args) == 1 and "is not the file" in capsys.readouterr().err
    teacher.write_bytes(b"teacher rows")
    (Path(q) / "p0" / "train_config.json").write_text(json.dumps({**cfgs[0], "lr": 3e-4}), encoding="utf-8")
    assert fq.main(args) == 1 and "lr" in capsys.readouterr().err                      # arguments other than the real s0's
    assert fq.main(args[:-6] + ["--p_seeds", "2", "--train", str(teacher), "--teacher_manifest", str(manifest)]) == 2


def make_real_scores(root: Path, models=fq.LINKED_MODELS) -> Path:
    for m in models:
        d = root / m / "like"
        d.mkdir(parents=True)
        (d / "report.json").write_text(json.dumps({"config": {"lora": m}}), encoding="utf-8")
        (d / "run.key").write_text(f"key of {m}\n", encoding="utf-8")
    return root


def can_symlink(tmp: Path) -> bool:
    try:
        (tmp / "t").mkdir()
        os.symlink(str(tmp / "t"), str(tmp / "l"), target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        return False


def test_the_real_scores_are_linked_into_the_ftq_root_never_copied(tmp_path):
    real = make_real_scores(tmp_path / "grid" / "scores" / "toys")
    q = tmp_path / "q" / "scores" / "toys"
    (tmp_path / "probe").mkdir()
    if can_symlink(tmp_path / "probe"):
        notes = fq.link_models(real, q, fq.LINKED_MODELS)
        assert len(notes) == 4 and all(n.startswith("[link]") for n in notes)
        for m in fq.LINKED_MODELS:
            assert os.path.islink(q / m) and os.path.realpath(q / m) == os.path.realpath(real / m)
            assert not os.path.isabs(os.readlink(q / m))                       # a relative link: the tree can move
            assert (q / m / "like" / "report.json").read_bytes() == (real / m / "like" / "report.json").read_bytes()
        assert all(n.startswith("[link]") for n in fq.link_models(real, q, fq.LINKED_MODELS))     # idempotent
        other = make_real_scores(tmp_path / "grid2" / "scores" / "toys")
        with pytest.raises(fq.FtqError, match="is a link to"):
            fq.link_models(other, q, ["s0"])                                    # never relinked to another directory
        (tmp_path / "q2" / "toys").mkdir(parents=True)
        (tmp_path / "q2" / "toys" / "s0").mkdir()
        with pytest.raises(fq.FtqError, match="never copied"):
            fq.link_models(real, tmp_path / "q2" / "toys", ["s0"])             # an existing real directory is not replaced
    else:
        with pytest.raises(fq.FtqError, match="never copied"):                  # no symlinks (Windows): refused ...
            fq.link_models(real, q, fq.LINKED_MODELS)
        notes = fq.link_models(real, q, fq.LINKED_MODELS, allow_copy=True)      # ... and copied only when DRY_RUN allows it
        assert len(notes) == 4 and all(n.startswith("[copy]") for n in notes)
        assert all(n.startswith("[copy]") for n in fq.link_models(real, q, fq.LINKED_MODELS, allow_copy=True))
        (real / "s0" / "like" / "run.key").write_text("changed\n", encoding="utf-8")
        with pytest.raises(fq.FtqError, match="never copied"):                  # a stale copy is not accepted
            fq.link_models(real, q, ["s0"], allow_copy=True)
    with pytest.raises(fq.FtqError, match="missing the real like pass"):
        fq.link_models(tmp_path / "nowhere", tmp_path / "q3", ["s0"])
    assert fq.main(["link", "--real_scores", str(tmp_path / "nowhere"), "--q_scores", str(tmp_path / "q3")]) == 1
    assert fq.main(["link", "--q_scores", str(tmp_path / "q3")]) == 2
    assert not (tmp_path / "q3").exists()


def test_the_ftq_record_gate_needs_the_sha1_of_the_files_in_the_pilot_log(tmp_path, capsys):
    (tmp_path / "a.sh").write_text("script a\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("module b\n", encoding="utf-8")
    log = tmp_path / "PILOT_LOG.md"
    log.write_text("# log\n", encoding="utf-8")
    base = ["record", "--pilot_log", str(log), "--files", "a.sh", "b.py", "--root", str(tmp_path)]
    assert fq.main(base) == 4 and "a.sh" in capsys.readouterr().err                   # not recorded: refused (exit 4)
    assert fq.main(base + ["--print"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"a.sh = {sha1_file(tmp_path / 'a.sh')}", f"b.py = {sha1_file(tmp_path / 'b.py')}"]
    log.write_text("# log\n" + lines[0].upper() + "\n" + lines[1] + "\n", encoding="utf-8")   # case-insensitive, as freeze
    assert fq.main(base) == 0 and "FT-Q record OK" in capsys.readouterr().out
    (tmp_path / "b.py").write_text("module b, changed after the record\n", encoding="utf-8")
    assert fq.main(base) == 4 and "b.py" in capsys.readouterr().err                   # a changed file needs a new record
    assert fq.main(base + ["--append"]) == 0                                          # DRY_RUN's human step on a temporary log
    out = capsys.readouterr().out
    assert "[dry] recorded b.py" in out and f"b.py = {sha1_file(tmp_path / 'b.py')}" in log.read_text(encoding="utf-8")
    log.write_bytes(b"# log without a final newline")
    assert fq.main(base + ["--append"]) == 0 and len(log.read_text(encoding="utf-8").splitlines()) == 3
    assert fq.main(["record", "--pilot_log", str(tmp_path / "none.md"), "--files", "a.sh", "--root", str(tmp_path)]) == 4
    assert fq.main(["record", "--files", "a.sh"]) == 2                                 # usage
    assert fq.record_lines(list(FTQ_LINES)) == [
        f"scripts/sigir/run_ftq.sh = {sha1_file(SCRIPT)}", f"src/confrec/ftq_panel.py = {sha1_file(PANEL)}",
        f"tests/test_confrec_ftq.py = {sha1_file(__file__)}"]                  # the record the script needs: three files
    assert "FTQ_FILES=(scripts/sigir/run_ftq.sh src/confrec/ftq_panel.py tests/test_confrec_ftq.py)" in SCRIPT.read_text(encoding="utf-8")


def test_info_reads_the_decisions_the_gates_use(tmp_path, capsys):
    sel, split, gate = tmp_path / "selection.json", tmp_path / "ftgrid_split.json", tmp_path / "gate_ft.json"
    sel.write_text(json.dumps({"decision": "FIX_FOUND", "gate_ft_prompt": "V1"}), encoding="utf-8")
    base = ["info", "--selection", str(sel), "--split", str(split), "--gate", str(gate)]
    assert fq.main(base) == 0 and capsys.readouterr().out.splitlines() == ["FIX_FOUND", "V1", "", "missing"]
    split.write_text(json.dumps({"variant": "V1"}), encoding="utf-8")
    gate.write_text(json.dumps({"decision": "GATE_FT_PASS"}), encoding="utf-8")
    assert fq.main(base) == 0 and capsys.readouterr().out.splitlines() == ["FIX_FOUND", "V1", "V1", "GATE_FT_PASS"]
    assert fq.main(["info", "--selection", str(tmp_path / "gone.json"), "--split", str(split), "--gate", str(gate)]) == 1
    assert fq.main(["info", "--selection", str(sel)]) == 2


def test_the_command_line_is_one_flat_parser_with_every_command_and_exit_codes(capsys):
    assert set(fq.COMMANDS) == {"build", "verify_teacher", "recipe", "scoring", "verify_adapter", "verify_scores", "link", "record",
                                "info"}
    for cmd in fq.COMMANDS:
        assert fq.main([cmd]) == 2 and "needs" in capsys.readouterr().err        # a missing flag is a usage refusal
    with pytest.raises(SystemExit):
        fq.main(["nonsense"])
    p = subprocess.run([sys.executable, "-m", "src.confrec.ftq_panel", "record", "--files", "x"], cwd=ROOT,
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert p.returncode == 2 and "needs --pilot_log" in p.stderr


# ================================================================ 6. the script file, its flags and its input guards
def test_script_is_lf_bash_and_bash_n_clean():
    raw = SCRIPT.read_bytes()
    assert b"\r\n" not in raw and raw.startswith(b"#!/usr/bin/env bash\n") and b"set -euo pipefail" in raw
    assert b"\r\n" not in PANEL.read_bytes()
    if FR.BASH is None:
        pytest.skip(FR.NO_BASH)
    r = subprocess.run([FR.BASH, "-n", SCRIPT.as_posix()], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_every_flag_the_script_passes_exists_in_the_real_argparse(trainer_parser):
    """The stricter audit of test_confrec_ftgrid_run (heredocs ignored, arrays expanded, literal values checked against choices and
    types), with the RECIPE and SARGS flags filled in from ftq_panel's own output, so the flags the script takes from the recorded
    configs are audited too. The one call with a subcommand (train_lora_offset qhat) is audited here against that subparser."""
    text = SCRIPT.read_text(encoding="utf-8")
    cfg = {"dtype": "float16", "topk_logprobs": 50, "max_model_len": 4096, "chunk_users": 100, "variant": VARIANT,
           "readout": "yesno", "questions": ["like"]}
    arrays = (f"\nRECIPE=({' '.join(fq.recipe_flags(sft_config(trainer_parser, 0, 'toys')))})"
              f"\nSARGS=({' '.join(fq.score_flags(cfg))})\n")
    qhat_call = re.search(r'"\$PY" -m src\.confrec\.train_lora_offset qhat ([^\\\n]*)', text).group(1)
    audited = text.replace('"$PY" -m src.confrec.train_lora_offset qhat', ": train_lora_offset qhat")
    problems, count = FR.audit(audited + arrays)
    assert not problems, "\n".join(problems)
    for target, n in {"src.confrec.pyes_scorer": 2, "src.confrec.train_lora_yesno": 1, "src.confrec.ftgrid_freeze": 2,
                      "src.confrec.ftgrid_report": 1, "src.confrec.ftq_panel": 9}.items():
        assert count.get(target, 0) >= n, (target, count)
    top = FR._parser("src.confrec.train_lora_offset")
    sub = [a for a in top._actions if isinstance(a, argparse._SubParsersAction)][0].choices["qhat"]
    flags = set(re.findall(r"(--[a-z_]+)", qhat_call))
    assert flags and flags <= {s for a in sub._actions for s in a.option_strings}
    bad = text.replace("--p_seeds", "--pseeds").replace("--teacher_manifest", "--teacher_manifes").replace(
        "--scores_root", "--scoresroot")
    problems, _ = FR.audit(bad.replace('"$PY" -m src.confrec.train_lora_offset qhat', ": q") + arrays)
    assert any("--pseeds" in p for p in problems) and any("--scoresroot" in p for p in problems)


def test_the_script_never_names_an_output_below_the_registered_grid_root():
    """Nothing is written below the grid root or the nested slot's root: no output flag, redirect, rm or mv aims at $GRID, $P, $RS,
    $QD, $RA or $REG, and the FT-Q root is the only root a run writes."""
    text = re.sub(r"# ---- DRY_RUN: run_ftgrid.*?# ---- the prompt", "# ---- the prompt", SCRIPT.read_text(encoding="utf-8"),
                  flags=re.S)                          # the DRY world block builds the world it reads: the one place that may
    code = re.sub(r"\\\n\s*", " ", "\n".join(x.split("#", 1)[0] for x in text.splitlines()))
    for pat in (r'--out_dir "\$(P|RS|QD|RA|GRID|REG)', r'--output "\$(P|RS|QD|RA|GRID|REG|QH)', r'--out "\$(P|RS|QD|RA|GRID|REG|QH)',
                r'> *"\$(P|RS|QD|RA|GRID|REG|QH|REGREP)', r'(rm|mv|cp|touch|mkdir)[^;|&\n]*"\$(P|RS|QD|RA|GRID|REG|QH|REGREP)[/"]',
                r'ln -s', r'git '):
        assert not re.search(pat, code), pat
    for needle in ('REGQ=outputs/confrec/ftgrid_q ', 'FTQ="$OUT_ROOT/ftq/$D"', 'QA="$OUT_ROOT/adapters/$D"',
                   'QS="$OUT_ROOT/scores/$D"', 'REP="$OUT_ROOT/report/$D.json"', '--scores_root "$OUT_ROOT/scores"',
                   "--models zeroshot,s0,s1,s2,p0,p1", "ftq_panel link", "ftq_panel build"):
        assert needle in code, needle


def shell_function(text: str, name: str) -> str:
    """The text of `name() {...}` in a shell script: its one line, or up to the closing brace at the start of a line."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"{name}() {{"))
    if lines[start].rstrip().endswith("}"):
        return lines[start]
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start:end + 1])


def test_the_helpers_mirrored_from_run_ftgrid_are_its_text_and_the_e1_rule_is_the_same():
    mine = SCRIPT.read_text(encoding="utf-8")
    theirs = (ROOT / "scripts" / "sigir" / "run_ftgrid.sh").read_text(encoding="utf-8")
    for name in ("fresh", "step", "adapter_done", "e1_ok"):
        assert shell_function(mine, name) == shell_function(theirs, name), name
    needles = ('key="$(', 'wsha=$(', "run.key", "$dir.stale", "$dir.e1fail", "FAILED_INTEGRITY", 'e1_ok "$dir"',
               'score "$data" "$dir" "$@"')
    keep = lambda text: [ln for ln in shell_function(text, "score").splitlines()
                         if any(n in ln for n in needles) and "compgen" not in ln and "e1_failed_before" not in ln]
    assert keep(mine) == keep(theirs) and len(keep(mine)) >= 8                 # key, skip, stale, E1 rerun-once, FAILED
    # ... except the ONE deliberate deviation (review finding m1): run_ftgrid.sh counts every DIR.e1fail.*, so an e1fail left by an
    # earlier run (another panel, adapter or arguments) turns a transient first failure into the final one; here the rerun is used
    # up only by an e1fail directory of the same run.key
    assert 'if compgen -G "$dir.e1fail.*" > /dev/null; then' in shell_function(theirs, "score")
    assert 'if e1_failed_before "$dir" "$key"; then' in shell_function(mine, "score") and "compgen" not in shell_function(mine, "score")
    helper = shell_function(mine, "e1_failed_before")
    assert 'for d in "$1".e1fail.*; do' in helper and '[ "$(cat "$d/run.key")" = "$2" ]' in helper
    for line in ('if [ "$SEL_DECISION" = FIX_FOUND ] && [ ! -f "$G/confirm/gate.json" ]; then',
                 'SEL="$G/dev/selection.json"', 'VARIANT="${VARIANT:-$SEL_PROMPT}"'):
        assert line in theirs and line in mine, line
    for exit_rule in ("exit 4", "exit 2", "exit 1"):
        assert exit_rule in mine and exit_rule in theirs


def test_the_registered_text_of_addendum_8_is_what_the_script_and_the_module_implement():
    a8 = (ROOT / "idea-stage" / "PREREG_AMENDMENT_3_ADDENDUM_8.md").read_text(encoding="utf-8")
    for needle in ("outputs/confrec/ftgrid_q/", "scripts/sigir/run_ftq.sh", "src/confrec/ftq_panel.py", "train_qhat.csv.gz",
                   "round(beta n)", "seed-0 random key", "candidate_labels", "p0 and p1", "Qwen3-8B only"):
        assert needle in a8, needle
    assert fq.TIE_SEED == 0 and fq.CONTROL_SEEDS == (0, 1) and set(fq.DOMAINS) == {"ml1m", "toys", "games", "sports"}
    code = SCRIPT.read_text(encoding="utf-8")
    assert code.count("for seed in 0 1;") >= 3 and code.count("p0, p1") >= 3


def test_the_ftq_files_are_new_files_outside_every_freeze_block():
    """No bound file changes: the files of this task are not in any FREEZE_FILES block of Amendment 3 or its addenda, and the
    bound files it imports or mirrors are."""
    ff = FR.ff
    texts = [ROOT / ff.AMENDMENT] + sorted((ROOT / "idea-stage").glob("PREREG_AMENDMENT_3_ADDENDUM_*.md"))
    listed = {f for p in texts for files in ff.parse_blocks(p.read_text(encoding="utf-8")).values() for f in files}
    assert listed and not listed & {"scripts/sigir/run_ftq.sh", "src/confrec/ftq_panel.py", "tests/test_confrec_ftq.py"}
    assert {"src/confrec/ftgrid_data.py", "scripts/sigir/run_ftgrid.sh", "src/confrec/pyes_scorer.py",
            "src/confrec/train_lora_yesno.py"} <= listed


@pytest.fixture(scope="module")
def guard_repo(tmp_path_factory):
    if FR.BASH is None:
        pytest.skip(FR.NO_BASH)
    return make_ftq_repo(tmp_path_factory.mktemp("ftq_guards") / "repo")


def make_ftq_repo(dest: Path) -> Path:
    """The repo parts the chain runs (run_ftgrid.sh's copy with the real ftgrid_report) plus this script."""
    FR.make_repo(dest, real_f2=True)
    shutil.copy2(SCRIPT, dest / "scripts" / "sigir" / "run_ftq.sh")
    (dest / "tests").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), dest / "tests" / "test_confrec_ftq.py")        # the third file of the FT-Q record
    return dest


def run_ftq(repo: Path, *args, dry: bool = True, **env) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in (
        "MODEL", "OUT_ROOT", "VARIANT", "STAGES", "DRY_RUN", "DRY_GATE", "DRY_E1_FAIL", "DRY_NO_RECORD", "DRY_NO_WORLD", "FTQ_ALLOW_RETRY", "FTQ_RETRY_REASON", "PYTHONPATH")}
    e.update(PYTHON=sys.executable.replace("\\", "/"), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if dry:
        e["DRY_RUN"] = "1"
    e.update(env)
    script = (repo / "scripts" / "sigir" / "run_ftq.sh").as_posix()
    return subprocess.run([FR.BASH, script, *args], env=e, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=3600)


LLAMA_PATH = "/models/Llama-3.1-8B-Instruct"


@needs_bash
@pytest.mark.parametrize("args, dry, env, rc, msg", [
    (["beauty"], True, {}, 2, "usage"), ([], True, {}, 2, "usage"), (["toys"], True, {"STAGES": "0,7"}, 2, "unknown stage"),
    (["toys"], True, {"STAGES": "5"}, 2, "unknown stage"),
    (["toys"], True, {"OUT_ROOT": "outputs/confrec/ftgrid_q"}, 2, "never writes to a registered output root"),
    (["toys"], True, {"OUT_ROOT": "./outputs/confrec/ftgrid/"}, 2, "never writes to a registered output root"),
    (["toys"], True, {"MODEL": LLAMA_PATH}, 2, "is not Qwen3-8B"), (["games"], False, {"MODEL": LLAMA_PATH}, 2, "is not Qwen3-8B"),
    (["toys"], False, {"MODEL": "/models/Qwen3-8B", "OUT_ROOT": "outputs/confrec/ftgrid_q_dryrun"}, 2, "one registered root"),
    (["toys"], False, {"MODEL": "/models/Qwen3-8B", "OUT_ROOT": "outputs/confrec/ftgrid"}, 2, "one registered root"),
    (["ml1m"], False, {"MODEL": "/models/Qwen3-8B"}, 1, "missing outputs/confrec/gatefix/dev/selection.json"),
    (["sports"], False, {"MODEL": "/models/Qwen3-8B"}, 1, "missing outputs/confrec/gatefix/dev/selection.json"),
    (["sports"], False, {"MODEL": "/models/Qwen3-8B", "DRY_RUN": "0"}, 1, "missing outputs/confrec/gatefix/dev/selection.json"),
    # the switches are 0 or 1: DRY_RUN=yes must not become a real run (nor an empty DRY_RUN a rehearsal)
    (["toys"], False, {"DRY_RUN": "yes"}, 2, "DRY_RUN must be 0 or 1"), (["toys"], False, {"DRY_RUN": "true"}, 2, "DRY_RUN must be 0 or 1"),
    (["toys"], False, {"DRY_RUN": "2"}, 2, "DRY_RUN must be 0 or 1"), (["toys"], False, {"DRY_RUN": ""}, 2, "DRY_RUN must be 0 or 1"),
    (["toys"], False, {"DRY_RUN": "01"}, 2, "DRY_RUN must be 0 or 1"), (["toys"], False, {"DRY_RUN": " 1"}, 2, "DRY_RUN must be 0 or 1"),
    (["toys"], True, {"DRY_NO_RECORD": "yes"}, 2, "DRY_NO_RECORD must be 0 or 1"),
    (["toys"], True, {"DRY_NO_WORLD": "true"}, 2, "DRY_NO_WORLD must be 0 or 1"),
    # the override of the sticky FAILED_INTEGRITY names a seed and needs a reason, before anything starts
    (["toys"], True, {"FTQ_ALLOW_RETRY": "p0"}, 2, "needs a non-empty FTQ_RETRY_REASON"),
    (["toys"], True, {"FTQ_ALLOW_RETRY": "p0", "FTQ_RETRY_REASON": ""}, 2, "needs a non-empty FTQ_RETRY_REASON"),
    (["toys"], True, {"FTQ_ALLOW_RETRY": "p2", "FTQ_RETRY_REASON": "why"}, 2, "FTQ_ALLOW_RETRY must be p0, p1 or p0,p1"),
    (["toys"], True, {"FTQ_ALLOW_RETRY": "all", "FTQ_RETRY_REASON": "why"}, 2, "FTQ_ALLOW_RETRY must be p0, p1 or p0,p1")])
def test_input_guards_refuse_before_anything_is_written(guard_repo, args, dry, env, rc, msg):
    r = run_ftq(guard_repo, *args, dry=dry, **env)
    assert r.returncode == rc and msg in r.stderr, FR.tail(r)
    assert not (guard_repo / "outputs").exists() and not (guard_repo / "tmp_outputs").exists()


# ---------------------------------------------------------------- the output roots: canonical forms decide, never the spelling
REGISTERED_FILES = {                                  # what a registered tree holds: a refused run must leave all of it as it is
    "outputs/confrec/ftgrid/report/toys.json": b"registered report\n",
    "outputs/confrec/ftgrid/adapters/toys/s0/train_config.json": b"{}\n",
    "outputs/confrec/ftgrid/scores/toys/s0/like/report.json": b"{}\n",
    "outputs/confrec/ftgrid/panels/toys/train.jsonl": b"rows\n",
    "outputs/confrec/ftgrid_q/report/toys.json": b"registered FT-Q report\n",
    "outputs/confrec/ftmethod/toys/train_qhat.csv.gz": b"q-hat\n",
    "outputs/confrec/gateft/gate_ft.json": b"{}\n",
    "outputs/confrec/gatefix/dev/note.txt": b"gate fix\n"}


def registered_repo(dest: Path) -> Path:
    """A scratch repo whose registered roots (the grid, the FT-Q root, the nested slot, Gate-FT, the gate fix) hold files."""
    repo = make_ftq_repo(dest)
    for rel, data in REGISTERED_FILES.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(data)
    return repo


def registered_state(repo: Path) -> dict:
    """{path: (size, mtime_ns, sha1)} of everything below the registered roots other than the FT-Q root, names included."""
    out = {}
    for rel in ("ftgrid", "ftmethod", "gateft", "gatefix"):
        base = repo / "outputs" / "confrec" / rel
        for p in sorted(base.rglob("*")) if base.exists() else []:
            out[p.relative_to(repo).as_posix()] = (p.stat().st_size, p.stat().st_mtime_ns, sha1_file(p)) if p.is_file() else "dir"
    return out


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


def run_many(repo: Path, cases: list, *, dry: bool, **env) -> list:
    """[(OUT_ROOT spelling, result)], the script runs concurrently (every run is dominated by process starts)."""
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        return list(ex.map(lambda oc: (oc, run_ftq(repo, "toys", dry=dry, OUT_ROOT=oc, **env)), cases))


def spellings(repo: Path, rel: str) -> list:
    """The spellings of a path below the repo that a string comparison does not equate with `rel` (// . .. absolute, native)."""
    absolute = (repo / rel).as_posix()
    head, _, tail = rel.rpartition("/")
    out = [rel + "//", rel.replace("/", "//", 1), head + "/./" + tail, "tmp_outputs/../" + rel, absolute, str(repo / rel)]
    if os.name == "nt":                                # D:\... spellings and a case-insensitive filesystem
        out += [absolute.replace("/", "\\"), absolute.upper(), rel.upper()]
    return out


@needs_bash
def test_a_dry_run_never_writes_under_outputs_confrec_in_any_spelling(tmp_path):
    """The reviewer's plant: DRY_RUN=1 OUT_ROOT=outputs/confrec/ftgrid// (and its relatives) used to write placeholder adapters, links
    and a marker under the registered root and to overwrite its report. Now every spelling of every registered root, and
    anything at, under or above outputs/confrec, is refused (exit 2) before anything is written."""
    repo = registered_repo(tmp_path / "repo")
    before = registered_state(repo)
    cases = spellings(repo, "outputs/confrec/ftgrid") + spellings(repo, "outputs/confrec/ftgrid_q") + [
        "outputs/confrec/ftgrid", "outputs/confrec/ftgrid_q", "outputs/confrec/gateft", "outputs/confrec/ftgrid/sub",
        "outputs/confrec/ftmethod", "outputs/confrec/gatefix", "outputs/confrec/ftgrid/../ftgrid_q", "outputs/confrec/ftgrid_dryrun",
        "outputs/confrec/ftgrid_q_dryrun", "outputs/confrec/ftmethod_dryrun", "outputs/confrec", "outputs/confrec/other",
        "outputs", ".", "/",
        # the allow-list (review round 3, m3): nothing in the repo but tmp_outputs, nothing beside it, nothing above it
        "outputs/summary", "outputs/baselines/x", "data/raw", "data", "src", "scripts", "docs", "tests", "idea-stage", "Paper",
        "../outside_repo", "../repo/src", "..", "tmp_outputs/../src", "tmp_outputs/../../outside_repo"]
    bad = [(oc, FR.tail(r)) for oc, r in run_many(repo, cases, dry=True)
           if r.returncode != 2 or not ("never writes to a registered output root" in r.stderr or "root is empty" in r.stderr
                                        or "filesystem root" in r.stderr)]
    assert not bad, "\n\n".join(f"== OUT_ROOT={oc!r}\n{tail}" for oc, tail in bad)
    assert registered_state(repo) == before and not (repo / "tmp_outputs").exists()
    assert (repo / "outputs/confrec/ftgrid/report/toys.json").read_bytes() == REGISTERED_FILES["outputs/confrec/ftgrid/report/toys.json"]
    assert not (repo / "outputs/confrec/ftgrid/adapters/toys/p0").exists() and not (repo / "adapters").exists()


@needs_bash
def test_a_dry_run_accepts_a_root_outside_outputs_confrec_in_any_spelling(tmp_path):
    """Acceptance is as canonical as refusal: tmp_outputs of the repo, and a directory outside the repo's parent, are fine however they
    are spelled (the run stops at the stage list here, after every guard has passed)."""
    repo = registered_repo(tmp_path / "repo")
    before = registered_state(repo)
    away = Path(tempfile.mkdtemp(prefix="ftq_dry_away_"))                    # outside the repo and outside its parent directory
    try:
        outside = (away / "ftq root").as_posix()
        cases = [DRYB + "/ftgrid_q", DRYB + "/ftgrid_q//", "./" + DRYB + "/x", "tmp_outputs//ftq_dryrun/./y", "tmp_outputs/a/../b",
                 outside, outside + "/", str(away / "ftq root")]
        if os.name == "nt":                                                  # D:\... and a case-insensitive filesystem
            cases += [DRYB.upper() + "/z", outside.replace("/", "\\")]
        bad = [(oc, FR.tail(r)) for oc, r in run_many(repo, cases, dry=True, STAGES="9")
               if r.returncode != 2 or "unknown stage" not in r.stderr]
        assert not bad, "\n\n".join(f"== OUT_ROOT={oc!r}\n{tail}" for oc, tail in bad)
        assert registered_state(repo) == before and not (repo / "tmp_outputs").exists() and not (away / "ftq root").exists()
    finally:
        shutil.rmtree(away, ignore_errors=True)


@needs_bash
def test_a_real_run_takes_the_registered_ftq_root_in_any_spelling_and_no_other_directory(tmp_path):
    repo = registered_repo(tmp_path / "repo")
    before = registered_state(repo)
    real = {"MODEL": "/models/Qwen3-8B"}
    refused = spellings(repo, "outputs/confrec/ftgrid") + [
        "outputs/confrec/ftgrid", "outputs/confrec/ftgrid_q_dryrun", "outputs/confrec/ftgrid_q/sub", "outputs/confrec/ftgrid_q/..",
        "outputs/confrec/ftgrid/../ftgrid", "outputs/confrec/gateft", "outputs/confrec/ftmethod", DRYB + "/ftgrid_q", "tmp_outputs/x",
        "outputs/confrec", "."]
    accepted = spellings(repo, "outputs/confrec/ftgrid_q") + [
        "outputs/confrec/ftgrid_q", "./outputs/confrec/ftgrid_q", "outputs/confrec/./ftgrid_q/", "outputs/confrec/ftgrid/../ftgrid_q"]
    bad = [(oc, FR.tail(r)) for oc, r in run_many(repo, refused, dry=False, **real)
           if r.returncode != 2 or "one registered root" not in r.stderr]
    bad += [(oc, FR.tail(r)) for oc, r in run_many(repo, accepted, dry=False, **real)       # the guards pass: the next refusal is
            if r.returncode != 1 or "missing outputs/confrec/gatefix/dev/selection.json" not in r.stderr]    # the missing selection
    assert not bad, "\n\n".join(f"== OUT_ROOT={oc!r}\n{tail}" for oc, tail in bad)
    assert registered_state(repo) == before and not (repo / "tmp_outputs").exists()


@needs_bash
def test_a_link_between_the_registered_roots_and_the_files_a_run_writes_is_refused(tmp_path):
    """Planted links (symlinks, or junctions on a Windows box): an alias of the registered grid as a DRY root, the synthetic world's
    own path leading into the grid, a link below the DRY root, the registered FT-Q root itself a link to the grid, and a link below
    the registered FT-Q root that would carry the adapters, the scores or the report into the grid. Every one is refused with
    exit 2 before anything is written, and the registered tree is left as it was."""
    base = registered_repo(tmp_path / "base")
    probe = tmp_path / "probe"
    (probe / "t").mkdir(parents=True)
    if not link_dir(probe / "l", probe / "t"):
        pytest.skip("no directory links can be made here (no symlink privilege, no junctions)")
    unlink_dir(probe / "l")
    grid, state, real = "outputs/confrec/ftgrid", registered_state(base), {"MODEL": "/models/Qwen3-8B"}

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
        ok = r.returncode == 2 and msg in r.stderr and registered_state(repo) == state
        return [] if ok else [(name, FR.tail(r) + "\nregistered tree unchanged: " + str(registered_state(repo) == state))]

    def dry_cases() -> list:
        repo, bad = clone_of("dry"), []
        plant(repo, "tmp_outputs/alias", grid)                             # an alias of the registered grid root as the DRY root
        bad += refused(repo, "alias", run_ftq(repo, "toys", OUT_ROOT="tmp_outputs/alias/sub"), "never writes to a registered")
        bad += [] if os.listdir(repo / "tmp_outputs") == ["alias"] else [("alias", "something was written")]
        unlink_dir(repo / "tmp_outputs" / "alias")
        plant(repo, DRYB + "/ftgrid", grid)                                # the synthetic world's own path leads into the grid
        bad += refused(repo, "world", run_ftq(repo, "toys"), "never writes to a registered")
        unlink_dir(repo / DRYB / "ftgrid")
        plant(repo, DRYB + "/ftgrid_q/scores", grid + "/scores")           # a link below the DRY root
        r = run_ftq(repo, "toys")
        bad += refused(repo, "dry child", r, "resolves to") + ([] if not (repo / DRYB / "ftgrid").exists() else [("dry child", "built")])
        unlink_dir(repo / DRYB / "ftgrid_q" / "scores")
        plant(repo, DRYB + "/ftgrid/panels", grid + "/panels")             # a link inside the synthetic world
        bad += refused(repo, "world child", run_ftq(repo, "toys"), "never writes to a registered")
        unlink_dir(repo / DRYB / "ftgrid" / "panels")
        return bad

    def registered_cases() -> list:
        repo = clone_of("registered")
        shutil.rmtree(repo / "outputs/confrec/ftgrid_q")
        plant(repo, "outputs/confrec/ftgrid_q", grid)                      # the registered FT-Q root is a link to the grid
        r = run_ftq(repo, "toys", dry=False, **real)
        out = refused(repo, "registered root", r, "one registered root")
        unlink_dir(repo / "outputs/confrec/ftgrid_q")
        return out

    def ancestor_cases() -> list:
        repo = clone_of("ancestor")
        os.rename(repo / "outputs", repo / "real_outputs")                 # the whole outputs tree is reached through a link
        assert link_dir(repo / "outputs", repo / "real_outputs")
        out = refused(repo, "outputs is a link", run_ftq(repo, "toys", dry=False, **real), "one registered root")
        unlink_dir(repo / "outputs")
        return out

    def child_cases() -> list:
        repo, bad = clone_of("child"), []
        for child in ("adapters", "report"):                               # a link below the registered FT-Q root
            plant(repo, "outputs/confrec/ftgrid_q/" + child, grid + "/adapters")
            r = run_ftq(repo, "toys", dry=False, **real)
            bad += refused(repo, "child " + child, r, "redirect")
            unlink_dir(repo / "outputs/confrec/ftgrid_q" / child)
        return bad

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        bad = [x for found in [f.result() for f in [ex.submit(fn) for fn in (dry_cases, registered_cases, ancestor_cases, child_cases)]]
               for x in found]
    assert not bad, "\n\n".join(f"== {name}\n{tail}" for name, tail in bad)


# ================================================================ 7. the DRY_RUN chains (the worlds are cached across sessions)
DRYB = "tmp_outputs/ftq_dryrun"                       # a DRY_RUN's temporary directory: outside outputs/confrec
GRID = f"{DRYB}/ftgrid"                               # run_ftgrid.sh's DRY_RUN root: the world the script reads
QHD = f"{DRYB}/ftmethod"                              # the nested slot's DRY_RUN root: its stage-1 files
QROOT_DRY = f"{DRYB}/ftgrid_q"                        # the FT-Q root of a DRY_RUN
WORLD_DIRS = (GRID, QHD)
SENTINEL = b'{"registered": "report", "never": "touched"}\n'
FTQ_LINES = ("scripts/sigir/run_ftq.sh", "src/confrec/ftq_panel.py", "tests/test_confrec_ftq.py")   # the record the script needs


def world_key() -> str:
    """sha1 over the code that builds a DRY world (everything but ftq_panel.py and the tests): a changed key rebuilds it."""
    h = hashlib.sha1(sys.version.encode())
    h.update("|".join(WORLD_DIRS).encode())
    files = [p for p in sorted((ROOT / "src" / "confrec").glob("*.py")) if p.name != "ftq_panel.py"]
    files += [ROOT / "scripts" / "sigir" / n for n in ("run_ftgrid.sh", "run_ftmethod.sh", "starperm_panel.py")]
    files += [ROOT / FR.ff.AMENDMENT]
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    block = re.search(r"# ---- DRY_RUN: run_ftgrid.*?# ---- the prompt", SCRIPT.read_text(encoding="utf-8"), re.S)
    h.update(block.group(0).encode() if block else b"no DRY block")      # the script's own world-building block, nothing else of it
    return h.hexdigest()[:20]


@pytest.fixture(scope="session")
def world_cache(tmp_path_factory) -> Path:
    base = tmp_path_factory.getbasetemp().parent / "ftq_world_cache"
    key = world_key()
    base.mkdir(parents=True, exist_ok=True)
    for old in base.iterdir():                          # one key at a time: a changed code base drops the old worlds
        if old.name != key:
            shutil.rmtree(old, ignore_errors=True)
    (base / key).mkdir(exist_ok=True)
    return base / key


def build_world(cache: Path, d: str) -> None:
    """The DRY world of dataset d (run_ftgrid.sh's stages 0-3 and the nested slot's stage 1), built by the script's own DRY block
    (STAGES=1) in a scratch repo and kept without the FT-Q root."""
    tmp = Path(tempfile.mkdtemp(prefix=f"world_{d}_", dir=str(cache)))
    try:
        repo = make_ftq_repo(tmp / "repo")
        r = run_ftq(repo, d, STAGES="1")
        assert r.returncode == 0, FR.tail(r)
        dest = cache / f"{d}.part{os.getpid()}"
        for rel in WORLD_DIRS:
            shutil.copytree(repo / rel, dest / rel)
        (dest / ".complete").write_text("ok", encoding="utf-8")
        if (cache / d).exists():
            shutil.rmtree(dest, ignore_errors=True)       # another session finished first
        else:
            os.replace(dest, cache / d)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def worlds(world_cache) -> Path:
    if FR.BASH is None or FAST:
        pytest.skip(FR.NO_BASH if FR.BASH is None else "FTQ_FAST=1")
    missing = [d for d in ("toys", "ml1m") if not (world_cache / d / ".complete").exists()]
    with cf.ThreadPoolExecutor(max_workers=2) as ex:       # a cold cache builds both worlds at once
        for f in [ex.submit(build_world, world_cache, d) for d in missing]:
            f.result()
    return world_cache


def seed_world(repo: Path, worlds: Path, d: str) -> None:
    for rel in WORLD_DIRS:
        shutil.copytree(worlds / d / rel, repo / rel)


def snapshot_world(repo: Path) -> dict:
    """{path: (size, mtime_ns, sha1)} of the grid world and the nested slot's files; the temporary pilot logs are the one thing a
    run may append to (the record of the FT-Q files)."""
    out = {}
    for rel in WORLD_DIRS:
        for p in sorted((repo / rel).rglob("*")):
            if p.is_file() and not p.name.startswith("PILOT_LOG"):
                st = p.stat()
                out[p.relative_to(repo).as_posix()] = (st.st_size, st.st_mtime_ns, sha1_file(p))
    return out


def make_chain(worlds: Path, tpf, d: str, rerun: bool) -> dict:
    repo = make_ftq_repo(tpf.mktemp(f"ftq_{d}") / "repo")
    seed_world(repo, worlds, d)
    reports = repo / GRID / "report"                      # a stand-in registered report: never touched
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{d}.json").write_bytes(SENTINEL)
    (reports / f"{d}_tables.csv").write_bytes(b"model,arm\n")
    c = {"d": d, "repo": repo, "root": repo / QROOT_DRY, "grid": repo / GRID, "before": snapshot_world(repo)}
    c["r"] = run_ftq(repo, d)
    assert c["r"].returncode == 0, FR.tail(c["r"])
    c["after"] = snapshot_world(repo)
    c["s1"] = FR.snapshot(c["root"])
    if rerun:
        c["r2"] = run_ftq(repo, d)
        assert c["r2"].returncode == 0, FR.tail(c["r2"])
        c["s2"] = FR.snapshot(c["root"])
    return c


@pytest.fixture(scope="module")
def chain_jobs(worlds, tmp_path_factory):
    """Both chains start together (the runs are process-start bound); each chain fixture waits for, and reports, its own chain."""
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        yield {d: ex.submit(make_chain, worlds, tmp_path_factory, d, d == "toys") for d in ("toys", "ml1m")}


@pytest.fixture(scope="module")
def chain_toys(chain_jobs):
    return chain_jobs["toys"].result()


@pytest.fixture(scope="module")
def chain_ml1m(chain_jobs):
    return chain_jobs["ml1m"].result()


def real_dirs(c: dict) -> tuple:
    """(the real adapters' directory, the path their train_config.json records for TRAIN) of the chain's dataset."""
    if c["d"] == "ml1m":
        return f"{GRID}/_dry/gateft/adapters", f"{GRID}/_dry/gateft/train.jsonl"
    return f"{GRID}/adapters/{c['d']}", f"{GRID}/panels/{c['d']}/train.jsonl"


def check_chain(c: dict) -> None:
    d, repo, root, grid, out = c["d"], c["repo"], c["root"], c["grid"], c["r"].stdout
    ra, train_ref = real_dirs(c)
    # the FT-Q root holds the teacher panel, the adapters, the scores and the report; nothing else
    assert {p.name for p in (root / "ftq" / d).iterdir()} == {"train_q.jsonl", "train_q.manifest.json"}
    assert {p.name for p in (root / "adapters" / d).iterdir()} == {"p0", "p1"}
    assert {p.name for p in (root / "scores" / d).iterdir()} == {"zeroshot", "s0", "s1", "s2", "p0", "p1"}
    assert {p.name for p in (root / "report").iterdir()} == {f"{d}.json", f"{d}_tables.csv", "NOTE_FTQ.txt"}
    assert {p.name for p in root.iterdir()} == {"ftq", "adapters", "scores", "report", "build"}
    # the registered grid world and the nested slot's files were not written (the stand-in registered report included)
    assert c["after"] == c["before"]
    assert (grid / "report" / f"{d}.json").read_bytes() == SENTINEL
    # the teacher panel: train.jsonl with the teacher labels, nothing else changed; the rate is beta
    man = read_json(root / "ftq" / d / "train_q.manifest.json")
    real = jsonl(grid / "panels" / d / "train.jsonl")
    teacher = jsonl(root / "ftq" / d / "train_q.jsonl")
    assert all(list(r) == list(t) and all(r[f] == t[f] for f in r if f != "candidate_labels") for r, t in zip(real, teacher))
    ex = ftprune.train_examples(real)
    q = [row["q_hat"] for row in tlo.read_qhat(repo / QHD / d / "train_qhat.csv.gz").values()]
    k = sum(int(y) for r in real for y in r["candidate_labels"])
    flat = [y for t in teacher for y in t["candidate_labels"]]
    assert flat == fq.teacher_labels(q, [ftprune.tie_key(x) for x in ex["key"]], k) and sum(flat) == k == man["k"]
    assert man["train"]["sha1"] == sha1_file(grid / "panels" / d / "train.jsonl") == read_json(
        repo / QHD / d / "qhat_manifest.json")["train"]["sha1"]
    assert man["train_q"]["sha1"] == sha1_file(root / "ftq" / d / "train_q.jsonl") and man["beta"] == k / man["n"]
    # stage 1 verified the panel on disk by recomputation (once per run) and printed its sha1 as a line for the pilot log
    assert out.count("byte for byte the teacher recomputed") == 1
    assert f"[stage 1] pilot-log line for the teacher panel of {d}: {QROOT_DRY}/ftq/{d}/train_q.jsonl = {man['train_q']['sha1']}" in out
    # the sidecar that says what the reports of this root are
    note = re.sub(r"\s+", " ", (root / "report" / "NOTE_FTQ.txt").read_text(encoding="utf-8"))
    for phrase in ("FT-Q TEACHER adapters", "not the within-item permutation adapters of FT-C",
                   "The names FT_C and FT-PERM (and the 'within-item permuted adapter' wording) are the permutation control's",
                   "every number under FT_C / PERM in these reports is FT-Q's",
                   "The ZS, FT and P1 blocks duplicate outputs/confrec/ftgrid/report/<D>.json and are cited only from there",
                   "FT_C_reading", "FT-Q control, root label teacher"):
        assert phrase in note, phrase
    assert "ML-1M's P1" not in note and "Nothing in this root is a permutation result" not in note      # the old, inaccurate point 2
    # the adapters: the real adapters' recorded arguments except --train, --out, --seed; provenance of the teacher panel
    real_cfg = read_json(repo / ra / "s0" / "train_config.json")
    assert real_cfg["train"] == train_ref
    assert ("[recipe] trainer flags recorded by the real s0-s2 (--train, --out, --seed replaced): "
            + " ".join(fq.recipe_flags(real_cfg))) in out
    assert FR.trained(c["r"]) == ["p0", "p1"]                              # the real adapters are never trained here
    assert out.index("[recipe] scorer flags recorded") < out.index("dry-run trainer: ")      # the pre-flight comes first
    for k_ in (0, 1):
        a = root / "adapters" / d / f"p{k_}"
        cfg = read_json(a / "train_config.json")
        assert cfg["train"] == f"{QROOT_DRY}/ftq/{d}/train_q.jsonl" and cfg["out"] == f"{QROOT_DRY}/adapters/{d}/p{k_}"
        assert cfg["seed"] == k_
        assert {x: v for x, v in cfg.items() if x not in fq.REPLACED} == {x: v for x, v in real_cfg.items()
                                                                         if x not in fq.REPLACED}
        prov = read_json(a / "ftq.json")
        assert prov["train_q_sha1"] == man["train_q"]["sha1"] and prov["seed"] == k_
    assert man["agreement_with_real_labels"]["share_teacher_equals_real_label"] < 1       # the control is not the real label
    # the like passes: the real s0 like pass's arguments, scored on the grid's eval.jsonl
    s0_cfg = read_json(grid / "scores" / d / "s0" / "like" / "report.json")["config"]
    s0_key = (grid / "scores" / d / "s0" / "like" / "run.key").read_text(encoding="utf-8").split()
    assert ("(--data, --output, --model, --lora replaced): " + " ".join(fq.score_flags(s0_cfg))) in out
    for k_ in (0, 1):
        lora = f"{QROOT_DRY}/adapters/{d}/p{k_}"
        rep = read_json(root / "scores" / d / f"p{k_}" / "like" / "report.json")
        cfg = rep["config"]
        assert set(cfg) == set(s0_cfg) and {x for x in s0_cfg if s0_cfg[x] != cfg[x]} == {"lora"} and cfg["lora"] == lora
        key = (root / "scores" / d / f"p{k_}" / "like" / "run.key").read_text(encoding="utf-8").split()
        assert key[:3] == [sha1_file(grid / "panels" / d / "eval.jsonl"), MODEL, VARIANT] == s0_key[:3]
        assert key[3] == fq.weights_sha1(root / "adapters" / d / f"p{k_}") and key[4:] == ["--lora", lora]
        assert rep["n_users"] == read_json(grid / "panels" / d / "ftgrid_split.json")["eval"]["users"]
    # the real scores are linked (symlinks; copies only where the platform has none), and the report reads them
    for m in fq.LINKED_MODELS:
        link, target = root / "scores" / d / m, grid / "scores" / d / m
        if os.path.islink(link):
            assert os.path.realpath(link) == os.path.realpath(target) and not os.path.isabs(os.readlink(link))
        else:
            assert "[copy]" in out and (link / "like" / "report.json").read_bytes() == (target / "like" / "report.json").read_bytes()
    rep = read_json(root / "report" / f"{d}.json")
    assert rep["meta"]["models_requested"] == ["zeroshot", "s0", "s1", "s2", "p0", "p1"] and rep["meta"]["domain"] == d
    assert (rep["meta"]["n_boot"], rep["meta"]["seed"]) == (20, 0)             # the DRY_RUN count (the real one is 2,000)
    assert (root / "report" / f"{d}_tables.csv").read_text(encoding="utf-8").startswith("domain,backbone,block")
    for m in ("zeroshot", "s0", "s1", "s2", "p0", "p1"):
        assert rep["runs"][m]["like"]["status"] == "OK", m
    assert not [e for e in rep["excluded_runs"] if e["model"] in ("p0", "p1")]
    assert rep["FT_C"]["descriptive"] is True and "mean_over_seeds_TEST" in rep["FT_C"]
    for s in ("seed0", "seed1"):
        assert rep["FT_C"][s].get("available") is not False and "TEST" in rep["FT_C"][s], rep["FT_C"][s]
    # the gates ran first and the FT-Q record was written on the temporary pilot log
    marks = [out.index(x) for x in ("freeze check OK (stage core)", "== stage 1", "== stage 2", "== stage 3", "== stage 4")]
    assert marks == sorted(marks) and "[dry] gate rehearsal" in out and "freeze check OK (stage amendment)" in out
    log = (grid / "_dry" / "PILOT_LOG.md").read_text(encoding="utf-8").lower()
    for rel in FTQ_LINES:
        assert f"{rel} = {sha1_file(repo / rel)}" in log


@needs_chain
def test_toys_chain_runs_the_whole_control_in_its_own_root(chain_toys):
    check_chain(chain_toys)


@needs_chain
def test_ml1m_chain_takes_gate_fts_adapters_as_the_real_ones(chain_ml1m):
    check_chain(chain_ml1m)
    root, repo = chain_ml1m["root"], chain_ml1m["repo"]
    gt = repo / GRID / "_dry" / "gateft"
    assert read_json(root / "ftq" / "ml1m" / "train_q.manifest.json")["split"]["variant"] == VARIANT
    # the ML-1M like passes of the real adapters are Gate-FT's (scored by their path): the teacher passes use their arguments
    assert read_json(repo / GRID / "scores" / "ml1m" / "s0" / "like" / "report.json")["config"]["lora"] == \
        f"{GRID}/_dry/gateft/adapters/s0"
    assert (gt / "adapters" / "s0" / "train_config.json").is_file()


@needs_chain
def test_toys_second_run_skips_everything_and_touches_nothing(chain_toys):
    c = chain_toys
    assert c["s1"] == c["s2"]
    out = c["r2"].stdout
    assert "dry-run trainer" not in out and "scores chunk" not in out and out.count(": scored") == 2
    assert out.count("[skip] adapter") == 2 and "[skip] run_ftgrid.sh's DRY_RUN world for toys" in out
    assert out.count("byte for byte the teacher recomputed") == 1 and "[stage 1] pilot-log line for the teacher panel of toys" in out
    for product in ("ftq/toys/train_q.manifest.json", "report/toys.json"):
        assert f"[skip] {QROOT_DRY}/{product}" in out, product
    assert "[dry] gate rehearsal" not in out                                   # the rehearsal runs once per world


# ---------------------------------------------------------------- the refusals, run concurrently on clones of the toys chain
def clone(c: dict, tmp: Path) -> Path:
    dest = tmp / "repo"
    shutil.copytree(c["repo"], dest)
    return dest


def forbid_training(r) -> str:
    return "" if "dry-run trainer" not in r.stdout else "a stage-2 training started"


def scenario_gate_ft(c, repo):
    gate = repo / GRID / "_dry" / "gateft" / "gate_ft.json"
    gate.write_text(json.dumps({"decision": "GATE_FT_FAIL"}), encoding="utf-8")
    before, errs = FR.snapshot(repo / QROOT_DRY), []
    r = run_ftq(repo, "toys")
    errs += [] if r.returncode == 4 and "Gate-FT decision GATE_FT_FAIL" in r.stderr and "== stage" not in r.stdout else [FR.tail(r)]
    gate.unlink()                                                              # no decision recorded at all
    r = run_ftq(repo, "toys", STAGES="3")
    errs += [] if r.returncode == 4 and "Gate-FT decision missing" in r.stderr else [FR.tail(r)]
    r = run_ftq(repo, "toys", STAGES="1")                                      # stage 1 is CPU and outcome-free: not gated
    errs += [] if r.returncode == 0 and "[skip]" in r.stdout else [FR.tail(r)]
    return errs + ([] if FR.snapshot(repo / QROOT_DRY) == before else ["a refused run wrote to the FT-Q root"])


def scenario_records(c, repo):
    log = repo / GRID / "_dry" / "PILOT_LOG.md"
    original, errs, before = log.read_text(encoding="utf-8"), [], FR.snapshot(repo / QROOT_DRY)
    log.write_text("# emptied\n", encoding="utf-8")                              # (a) no record at all
    r = run_ftq(repo, "toys", DRY_NO_RECORD="1")
    errs += [] if r.returncode == 4 and "stage amendment" in r.stderr else [FR.tail(r)]
    keep = [x for x in original.splitlines() if not x.startswith(FTQ_LINES)]
    assert len(keep) == len(original.splitlines()) - len(FTQ_LINES)
    log.write_text("\n".join(keep) + "\n", encoding="utf-8")                     # (b) only the FT-Q record is missing
    r = run_ftq(repo, "toys", DRY_NO_RECORD="1")
    errs += [] if r.returncode == 4 and "FT-Q record" in r.stderr and "stage core" not in r.stderr else [FR.tail(r)]
    log.write_text(original, encoding="utf-8")
    bound = repo / "src" / "confrec" / "metrics.py"                              # (c) a bound file changed since the record
    bound.write_text(bound.read_text(encoding="utf-8") + "\n# changed after the freeze check\n", encoding="utf-8")
    r = run_ftq(repo, "toys")
    errs += [] if r.returncode == 4 and "stage core" in r.stderr else [FR.tail(r)]
    shutil.copy2(ROOT / "src" / "confrec" / "metrics.py", bound)
    for rel in FTQ_LINES:                                                        # (d) an FT-Q file changed since its record
        path = repo / rel
        good = path.read_bytes()
        path.write_bytes(good + b"\n# changed after its record\n")
        r = run_ftq(repo, "toys", DRY_NO_RECORD="1")
        errs += [] if r.returncode == 4 and "FT-Q record" in r.stderr and rel in r.stderr and "== stage" not in r.stdout else [FR.tail(r)]
        path.write_bytes(good)
    return errs + ([] if FR.snapshot(repo / QROOT_DRY) == before else ["a refused run wrote to the FT-Q root"])


def scenario_missing_real_adapter(c, repo):
    shutil.rmtree(repo / GRID / "adapters" / "toys" / "s0")
    before = FR.snapshot(repo / QROOT_DRY)
    r = run_ftq(repo, "toys", STAGES="2")
    errs = [] if r.returncode == 1 and "missing the real adapter s0" in r.stderr and forbid_training(r) == "" else [FR.tail(r)]
    return errs + ([] if FR.snapshot(repo / QROOT_DRY) == before else ["a refused run wrote to the FT-Q root"])


def scenario_wrong_recorded_input(c, repo):
    root = repo / QROOT_DRY
    for k in (0, 1):
        shutil.rmtree(root / "adapters" / "toys" / f"p{k}")
    rep = repo / GRID / "scores" / "toys" / "s0" / "like" / "report.json"
    good = rep.read_text(encoding="utf-8")
    data = json.loads(good)
    data["config"]["chunk_users"] = 50                                             # the real s0 like pass is not the registered one
    rep.write_text(json.dumps(data), encoding="utf-8")
    r = run_ftq(repo, "toys", STAGES="2")
    errs = [] if r.returncode == 1 and "chunk_users" in r.stderr and forbid_training(r) == "" else [FR.tail(r)]
    rep.write_text(good, encoding="utf-8")
    cfg1 = repo / GRID / "adapters" / "toys" / "s1" / "train_config.json"         # s1 was trained with another recipe than s0
    cfg1.write_text(json.dumps({**read_json(cfg1), "lr": 5e-05}), encoding="utf-8")
    r = run_ftq(repo, "toys", STAGES="2")
    errs += [] if r.returncode == 1 and "not trained with one recipe" in r.stderr and forbid_training(r) == "" else [FR.tail(r)]
    return errs + ([] if not (root / "adapters" / "toys" / "p0").exists() else ["an adapter was created"])


def scenario_provenance(c, repo):
    root, errs = repo / QROOT_DRY, []
    prov = root / "adapters" / "toys" / "p0" / "ftq.json"
    good = prov.read_text(encoding="utf-8")
    prov.unlink()                                                                  # trained by hand
    r = run_ftq(repo, "toys", STAGES="2")
    errs += [] if r.returncode == 1 and "has no ftq.json" in r.stderr else [FR.tail(r)]
    prov.write_text(good, encoding="utf-8")
    teacher = root / "ftq" / "toys" / "train_q.jsonl"
    teacher.write_bytes(teacher.read_bytes() + b" ")                               # the panel on disk is not the manifest's
    r = run_ftq(repo, "toys", STAGES="2")
    errs += [] if r.returncode == 1 and "is not the teacher recomputed" in r.stderr and forbid_training(r) == "" else [FR.tail(r)]
    return errs


def scenario_stage_order(c, repo):
    root, errs = repo / QROOT_DRY, []
    shutil.rmtree(root / "ftq" / "toys")                                           # stage 1 not done
    before = FR.snapshot(root)
    for stages, msg in (("2", "(stage 1)"), ("3", "this script's stage 1")):
        r = run_ftq(repo, "toys", STAGES=stages)
        errs += [] if r.returncode == 1 and "train_q" in r.stderr and msg in r.stderr else [FR.tail(r)]
    errs += [] if FR.snapshot(root) == before else ["a refused run wrote to the FT-Q root"]
    r = run_ftq(repo, "toys", STAGES="1")                                          # ... and stage 1 rebuilds it, byte for byte
    same = (root / "ftq" / "toys" / "train_q.jsonl").read_bytes() == (c["root"] / "ftq" / "toys" / "train_q.jsonl").read_bytes()
    errs += [] if r.returncode == 0 and same else [FR.tail(r)]
    shutil.rmtree(root / "adapters" / "toys" / "p0")                               # stage 2 not done for p0
    shutil.rmtree(root / "scores" / "toys" / "p1")                                 # stage 3 not done for p1
    before = FR.snapshot(root)
    r = run_ftq(repo, "toys", STAGES="3")
    errs += [] if r.returncode == 1 and "adapter p0 of toys is missing or incomplete" in r.stderr else [FR.tail(r)]
    r = run_ftq(repo, "toys", STAGES="4")
    errs += [] if r.returncode == 1 and "missing the like pass" in r.stderr and "p1/like" in r.stderr else [FR.tail(r)]
    errs += [] if FR.snapshot(root) == before else ["a refused run wrote to the FT-Q root"]
    os.remove(repo / GRID / "scores" / "toys" / "s1" / "like" / "report.json")     # a real like pass is missing: nothing to link
    r = run_ftq(repo, "toys", STAGES="4", DRY_NO_WORLD="1")                        # (a DRY_RUN would first rebuild the synthetic world)
    return errs + ([] if r.returncode == 1 and "missing the real like pass" in r.stderr and "s1/like" in r.stderr else [FR.tail(r)])


def scenario_e1(c, repo):
    root, errs = repo / QROOT_DRY, []
    shutil.rmtree(root / "scores" / "toys" / "p0")
    r = run_ftq(repo, "toys", STAGES="3,4", DRY_E1_FAIL="p0/like")
    like = root / "scores" / "toys" / "p0" / "like"
    errs += [] if r.returncode == 0 else [FR.tail(r)]
    errs += [] if (like / "FAILED_INTEGRITY").is_file() and len(list(like.parent.glob("like.e1fail.*"))) == 1 else ["no FAILED_INTEGRITY"]
    errs += [] if r.stderr.count("[E1 failed]") == 1 and r.stderr.count("failed E1 twice") == 1 else ["E1 rerun-once messages"]
    errs += [] if r.stderr.count("NOTE: ") == 1 and "p0 is reported as missing, never replaced" in r.stderr else ["no closing note"]
    errs += [] if not (root / "scores" / "toys" / "p1" / "like" / "FAILED_INTEGRITY").exists() else ["p1 failed"]
    rep = read_json(root / "report" / "toys.json")
    errs += [] if [e["status"] for e in rep["excluded_runs"] if e["model"] == "p0"] == ["FAILED_INTEGRITY"] else ["p0 not excluded"]
    errs += [] if rep["runs"]["p1"]["like"]["status"] == "OK" and rep["FT_C"]["seed0"]["available"] is False else ["report"]
    errs += [] if (repo / GRID / "report" / "toys.json").read_bytes() == SENTINEL else ["the registered report changed"]
    return errs


def scenario_teacher_swap(c, repo):
    """The reviewer's plant: train_q.jsonl replaced by the REAL-label train.jsonl and its manifest re-tagged with the new sha1, the
    file's mtime and the step marker untouched. Stages 2 and 3 used to trust it ('provenance OK': p0 and p1 trained on real labels);
    now the teacher is recomputed from train.jsonl and the stage-1 q-hat file and the panel on disk must equal it byte for byte."""
    root, errs = repo / QROOT_DRY, []
    for k in (0, 1):
        shutil.rmtree(root / "adapters" / "toys" / f"p{k}")                        # so that stage 2 would train
    teacher, man = root / "ftq" / "toys" / "train_q.jsonl", root / "ftq" / "toys" / "train_q.manifest.json"
    real = (repo / GRID / "panels" / "toys" / "train.jsonl").read_bytes()
    st = teacher.stat()
    teacher.write_bytes(real)
    os.utime(teacher, ns=(st.st_atime_ns, st.st_mtime_ns))                         # no marker can notice a newer file
    m = read_json(man)
    m["train_q"]["sha1"] = sha1(real)
    man.write_bytes(fq.json_bytes(m))
    before = FR.snapshot(root)
    for stages in ("1", "2", "3"):
        r = run_ftq(repo, "toys", STAGES=stages)
        errs += [] if r.returncode == 1 and "is not the teacher recomputed" in r.stderr and forbid_training(r) == "" else [FR.tail(r)]
    errs += [] if FR.snapshot(root) == before else ["a refused run wrote to the FT-Q root"]
    return errs + ([] if not (root / "adapters" / "toys" / "p0").exists() else ["an adapter was trained on the swapped panel"])


def patch_fakes_for_a_transient_failure(repo: Path) -> None:
    """Test only: in this scratch repo's copy of run_ftgrid.sh's scorer stand-in DRY_E1_FAIL fails just the FIRST run of a run.key
    (a transient failure): once a DIR.e1fail.* sibling with the same run.key exists, the rerun passes."""
    fakes = repo / GRID / "_dry" / "ftgrid_fakes.py"
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


def leftover_e1fail(root: Path, run_key: str | None) -> Path:
    """Move p0's finished like pass aside as p0/like.e1fail.<an old time>, with the given run.key (None: the current one), so that
    p0 is scored again and a failed run of some earlier time is left next to it."""
    like = root / "scores" / "toys" / "p0" / "like"
    left = like.parent / "like.e1fail.20200101000000"
    shutil.copytree(like, left)
    if run_key is not None:
        (left / "run.key").write_text(run_key, encoding="utf-8")
    shutil.rmtree(like)
    return left


def scenario_e1_leftover_of_an_earlier_run(c, repo):
    """The reviewer's plant: a p0/like.e1fail.<time> left by an earlier run (another panel, adapter or argument vector: another
    run.key) used to turn a TRANSIENT first E1 failure into the final one (FAILED_INTEGRITY without the rerun). Now only an e1fail
    directory of the current run.key uses the rerun up."""
    root, errs = repo / QROOT_DRY, []
    left = leftover_e1fail(root, f"{'0' * 40} {MODEL} {VARIANT} {'1' * 40} --lora elsewhere/p0\n")
    patch_fakes_for_a_transient_failure(repo)
    r = run_ftq(repo, "toys", STAGES="3,4", DRY_E1_FAIL="p0/like")
    like = root / "scores" / "toys" / "p0" / "like"
    errs += [] if r.returncode == 0 else [FR.tail(r)]
    errs += [] if r.stderr.count("[E1 failed]") == 1 and "failed E1 twice" not in r.stderr else ["no rerun: " + FR.tail(r)]
    errs += [] if (like / "report.json").is_file() and not (like / "FAILED_INTEGRITY").exists() else ["p0's rerun did not stand"]
    errs += [] if left.is_dir() and len(list(like.parent.glob("like.e1fail.*"))) == 2 else ["the leftover and this run's e1fail"]
    rep = read_json(root / "report" / "toys.json")
    return errs + ([] if rep["runs"]["p0"]["like"]["status"] == "OK" and not [e for e in rep["excluded_runs"] if e["model"] == "p0"]
                   else ["p0 is excluded from the report"])


def scenario_e1_failure_of_the_same_run_key_uses_the_rerun_up(c, repo):
    """The rule itself, as run_ftgrid.sh has it: an e1fail directory of the CURRENT run.key (an earlier failed attempt of the very
    same panel, adapter and arguments) leaves no rerun: the next failure is final."""
    root, errs = repo / QROOT_DRY, []
    leftover_e1fail(root, None)
    r = run_ftq(repo, "toys", STAGES="3,4", DRY_E1_FAIL="p0/like")
    like = root / "scores" / "toys" / "p0" / "like"
    errs += [] if r.returncode == 0 else [FR.tail(r)]
    errs += [] if r.stderr.count("[E1 failed]") == 0 and r.stderr.count("failed E1 twice") == 1 else ["no immediate FAILED_INTEGRITY: " + FR.tail(r)]
    return errs + ([] if (like / "FAILED_INTEGRITY").is_file() and len(list(like.parent.glob("like.e1fail.*"))) == 1
                   else ["no FAILED_INTEGRITY"])


def scenario_temporary_files_of_the_bound_tools(c, repo):
    """Review round 3, m1: ftgrid_report and the scorer write <name>.tmp beside their products. A planted temporary name that leads to
    a registered file (a hard link, or a symlink where those can be made) used to be written through: the registered report, its
    tables or a real score file were overwritten. Now a stale temporary file is removed before the tool runs, and any link below the
    FT-Q root other than scores/D/{zeroshot,s0,s1,s2} (those resolved to the registered real score directory) is refused with
    exit 2 before anything is written."""
    root, errs = repo / QROOT_DRY, []
    reg_rep, reg_tab = repo / GRID / "report" / "toys.json", repo / GRID / "report" / "toys_tables.csv"
    s0_scores = repo / GRID / "scores" / "toys" / "s0" / "like" / "scores.csv.gz"
    before = {p: sha1_file(p) for p in (reg_rep, reg_tab, s0_scores)}
    for name in ("toys.json", "toys_tables.csv"):                                  # (a) the report is made again ...
        (root / "report" / name).unlink()
    os.link(reg_rep, root / "report" / "toys.json.tmp")                            # ... through temporary names that are the
    os.link(reg_tab, root / "report" / "toys_tables.csv.tmp")                      # registered report and tables (hard links)
    r = run_ftq(repo, "toys", STAGES="4")
    errs += [] if r.returncode == 0 and (root / "report" / "toys.json").is_file() and not list((root / "report").glob("*.tmp")) else [FR.tail(r)]
    like = root / "scores" / "toys" / "p0" / "like"                                # (b) p0 is scored again in place (no report, the
    (like / "report.json").unlink()                                                # run.key is current) through a temporary name that
    os.link(s0_scores, like / "scores.csv.gz.tmp")                                 # is the real s0's score file
    r = run_ftq(repo, "toys", STAGES="3")
    errs += [] if r.returncode == 0 and (like / "report.json").is_file() and not (like / "scores.csv.gz.tmp").exists() else [FR.tail(r)]
    errs += [] if {p: sha1_file(p) for p in before} == before else ["a registered file was written through a temporary name"]
    try:                                                                           # (c) the same as symlinks: refused by the sweep
        os.symlink(reg_rep, root / "report" / "toys.json.tmp")
        made = True
    except (OSError, NotImplementedError):
        made = False                                                               # (no symlink privilege: the hard links stand in)
    if made:
        (root / "report" / "toys.json").unlink()
        r = run_ftq(repo, "toys", STAGES="4")
        refusal = "is a link" in r.stderr or "redirect" in r.stderr              # the sweep, or the tree check that lists this name
        errs += [] if r.returncode == 2 and refusal and sha1_file(reg_rep) == before[reg_rep] else [FR.tail(r)]
        (root / "report" / "toys.json.tmp").unlink()
    wrong = root / "scores" / "toys" / "s1"                                        # (d) a link with an allowed name that leads
    if wrong.is_symlink() or getattr(os.path, "isjunction", lambda p: False)(wrong):                            # to the wrong score directory is refused too
        unlink_dir(wrong)
    else:
        shutil.rmtree(wrong)
    if link_dir(wrong, repo / GRID / "scores" / "toys" / "s2"):
        r = run_ftq(repo, "toys", STAGES="4")
        errs += [] if r.returncode == 2 and "not to the registered real score directory" in r.stderr else [FR.tail(r)]
    return errs


def scenario_failed_integrity_is_sticky_when_the_adapter_is_made_again(c, repo):
    """Review round 3, m4, stage 2: making an adapter again used to start a fresh E1 budget (the like pass of the new weights is a
    new run.key). While scores/D/p<k>/like or a like.stale.* of that seed holds FAILED_INTEGRITY, stage 2 refuses to train p<k>
    (exit 1, naming the directory) unless FTQ_ALLOW_RETRY=p<k> and a non-empty FTQ_RETRY_REASON are set; the override is logged."""
    root, errs = repo / QROOT_DRY, []
    like = root / "scores" / "toys" / "p0" / "like"
    (like / "FAILED_INTEGRITY").write_text("", encoding="utf-8")                   # p0's like pass failed E1 twice
    before = FR.snapshot(root)
    r = run_ftq(repo, "toys", STAGES="3")                                          # a finished failed state is resumed, not refused
    errs += [] if r.returncode == 0 and "[skip]" in r.stdout and FR.snapshot(root) == before else [FR.tail(r)]
    shutil.rmtree(root / "adapters" / "toys" / "p0")                               # p0 would be made again
    before = FR.snapshot(root)
    r = run_ftq(repo, "toys", STAGES="2")
    errs += [] if (r.returncode == 1 and f"{QROOT_DRY}/scores/toys/p0/like" in r.stderr and "FTQ_ALLOW_RETRY=p0" in r.stderr
                   and forbid_training(r) == "" and FR.snapshot(root) == before) else [FR.tail(r)]
    r = run_ftq(repo, "toys", STAGES="2", FTQ_ALLOW_RETRY="p1", FTQ_RETRY_REASON="another seed")      # the seed must be named
    errs += [] if r.returncode == 1 and forbid_training(r) == "" else [FR.tail(r)]
    r = run_ftq(repo, "toys", STAGES="2", FTQ_ALLOW_RETRY="p0")                    # and the reason must be given
    errs += [] if r.returncode == 2 and "FTQ_RETRY_REASON" in r.stderr and forbid_training(r) == "" else [FR.tail(r)]
    errs += [] if not (root / "report" / "NOTE_FTQ_overrides.txt").exists() and FR.snapshot(root) == before else ["a refusal wrote"]
    reason = "the trainer was killed at step 40; the weights were never used"
    r = run_ftq(repo, "toys", STAGES="2", FTQ_ALLOW_RETRY="p0", FTQ_RETRY_REASON=reason)
    errs += [] if r.returncode == 0 and FR.trained(r) == ["p0"] and "[retry] p0" in r.stderr else [FR.tail(r)]
    log = (root / "report" / "NOTE_FTQ_overrides.txt").read_text(encoding="utf-8").splitlines()
    errs += [] if len(log) == 1 and re.fullmatch(r"p0\t\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\t" + re.escape(reason), log[0]) else [f"log: {log}"]
    return errs


def scenario_failed_integrity_is_sticky_when_the_like_pass_changes_its_key(c, repo):
    """Review round 3, m4, stage 3: a like pass that would be scored under another run.key (the adapter was made again) while like/
    or a like.stale.* of that seed holds FAILED_INTEGRITY is refused (exit 1, naming the directory), unless the override is set;
    the other seed is not affected, and the marker that moved aside with the stale directory keeps the seed sticky."""
    root, errs = repo / QROOT_DRY, []
    like = root / "scores" / "toys" / "p0" / "like"
    (like / "FAILED_INTEGRITY").write_text("", encoding="utf-8")
    w0, w1 = (root / "adapters" / "toys" / f"p{k}" / "adapter_model.safetensors" for k in (0, 1))
    w0.write_bytes(w0.read_bytes() + b" made again")                               # the adapter was made again: another run.key
    before = FR.snapshot(root)
    r = run_ftq(repo, "toys", STAGES="3")
    errs += [] if (r.returncode == 1 and f"{QROOT_DRY}/scores/toys/p0/like" in r.stderr and "scores chunk" not in r.stdout
                   and FR.snapshot(root) == before) else [FR.tail(r)]
    stale = like.parent / "like.stale.20200101000000"                              # the marker of an earlier, moved-aside pass
    shutil.move(like, stale)
    before = FR.snapshot(root)
    r = run_ftq(repo, "toys", STAGES="3")
    errs += [] if r.returncode == 1 and f"{QROOT_DRY}/scores/toys/p0/like.stale.20200101000000" in r.stderr and FR.snapshot(root) == before \
        else [FR.tail(r)]
    reason = "the E1 failure was a full disk, see the log"
    r = run_ftq(repo, "toys", STAGES="3", FTQ_ALLOW_RETRY="p0", FTQ_RETRY_REASON=reason)
    errs += [] if r.returncode == 0 and (like / "report.json").is_file() and (stale / "FAILED_INTEGRITY").is_file() else [FR.tail(r)]
    log = (root / "report" / "NOTE_FTQ_overrides.txt").read_text(encoding="utf-8").splitlines()
    errs += [] if len(log) == 1 and log[0].startswith("p0\t") and log[0].endswith("\t" + reason) else [f"log: {log}"]
    w1.write_bytes(w1.read_bytes() + b" made again")                               # p1 never failed: scored again without override
    r = run_ftq(repo, "toys", STAGES="3")
    errs += [] if r.returncode == 0 and "[skip]" in r.stdout else [FR.tail(r)]
    errs += [] if len((root / "report" / "NOTE_FTQ_overrides.txt").read_text(encoding="utf-8").splitlines()) == 1 else ["p1 was logged"]
    w0.write_bytes(w0.read_bytes() + b" and again")                                # the moved-aside marker keeps p0 sticky
    r = run_ftq(repo, "toys", STAGES="3")
    errs += [] if r.returncode == 1 and "like.stale.20200101000000" in r.stderr else [FR.tail(r)]
    return errs


SCENARIOS = {"gate_ft": scenario_gate_ft, "records": scenario_records, "missing_real_adapter": scenario_missing_real_adapter,
             "wrong_recorded_input": scenario_wrong_recorded_input, "provenance": scenario_provenance,
             "teacher_swap": scenario_teacher_swap, "stage_order": scenario_stage_order, "e1": scenario_e1,
             "e1_leftover": scenario_e1_leftover_of_an_earlier_run,
             "e1_same_key": scenario_e1_failure_of_the_same_run_key_uses_the_rerun_up,
             "tmp_files": scenario_temporary_files_of_the_bound_tools,
             "sticky_stage2": scenario_failed_integrity_is_sticky_when_the_adapter_is_made_again,
             "sticky_stage3": scenario_failed_integrity_is_sticky_when_the_like_pass_changes_its_key}


@needs_chain
def test_refusals_and_the_e1_rule_on_clones_of_the_toys_chain(chain_toys, tmp_path):
    """Every scenario works on its own clone of the finished toys chain; they run concurrently (the script runs dominate)."""
    def work(name):
        sub = tmp_path / name
        sub.mkdir()
        return name, SCENARIOS[name](chain_toys, clone(chain_toys, sub))
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        results = dict(ex.map(work, SCENARIOS))
    failed = {name: errs for name, errs in results.items() if errs}
    assert not failed, "\n\n".join(f"== {name}\n" + "\n".join(map(str, errs)) for name, errs in failed.items())
