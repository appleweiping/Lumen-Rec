"""Tests of the train/test-shift diagnostic of the method slot (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_10.md section 2; the review finding
M3): src/confrec/ftmethod_shift_diag.py and scripts/sigir/run_ftmethod_shift_diag.sh. CPU only, deterministic, no network, no GPU, no
model, no vLLM.

  * the answer token is derived from the tokenizer (never hard-coded) and is a yes id of the scorer; the share s of the yes mass on it
    and the scores R (the scorer's reading plus b z) and C = R + log(s + (1 - s) exp(-b z)) are checked against hand-computed rows, and C
    against an explicit softmax in which only the answer token is shifted (the property that makes it the training-consistent score);
  * R is the stage-4 score: the real pyes_scorer run on the same panel with the same stand-in model writes the logits the diagnostic
    reads (up to the six decimals of scores.csv.gz), and the token-id lists the diagnostic sends are the scorer's for the same rows;
  * the rows are the first 1,000 CAL candidates in panel order (timestamp < T strictly, a cut inside the last user allowed);
  * the statistics (mean / median / 1st percentile of s, the pairs whose order differs between R and C, the UAUC of both), the reading
    rule at its boundary (p1 >= 0.99, the 1e-12 tolerance, every adapter, UNDETERMINED), the report from files end to end with the
    stand-in model, its strict JSON, and `verify` refusing every tampering (a scoring file, an adapter, the panel, the manifest, b, the
    mode, a missing adapter, an edited report);
  * the reuse of pyes_scorer by import (load_vllm's attributes, score_ids, the scorer's defaults; no vLLM call is written here) and no
    heavy import at module level;
  * the script: LF, `bash -n` clean, the flag audit, the DRY_RUN guards of run_ftmethod.sh (0 or 1, canonical roots in any spelling,
    links, the rehearsal allow-list), writing only below D/shift_diag/, and the DRY chain end to end on ML-1M (run_ftmethod.sh's cached
    DRY state): the gates (order and kill rule, cuts, Gate-FT, the freeze record, the diagnostic's own record, the date before every
    seed, no slot report yet, the adapters), a rerun that changes nothing, a low-share stand-in that reads NOT_EQUIVALENT, and the
    `verify` that stage 5 of run_ftmethod.sh waits for.

Run time: the DRY states are the cache of tests/test_confrec_ftmethod.py (built once per code version, shared); the script runs of the
guard and refusal tests run concurrently. FTMETHOD_FAST=1 skips every test that runs the script's DRY chain.
"""
from __future__ import annotations

import ast
import concurrent.futures as cf
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.confrec import forensics as fx
from src.confrec import ftmethod_report as fm
from src.confrec import ftmethod_shift_diag as sd
from src.confrec import pyes_scorer as ps
from src.confrec import train_lora_offset as tlo
from src.confrec.prompting import chat_ids, render_record, read_yes_no, yes_no_ids

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sigir" / "run_ftmethod_shift_diag.sh"
MODULE = ROOT / "src" / "confrec" / "ftmethod_shift_diag.py"


def _load(name: str):
    """A sibling test module as a library (its helpers; it is never edited here)."""
    spec = importlib.util.spec_from_file_location(f"{name}_for_shift_diag", ROOT / "tests" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FM = _load("test_confrec_ftmethod")        # the slot's helpers: the synthetic ML-1M world, the DRY states, the script runners
BASH, NO_BASH, FAST = FM.BASH, FM.NO_BASH, FM.FAST
needs_bash, needs_chain = FM.needs_bash, FM.needs_chain
DRY_ROOT, GRID_DRY, tail, run_script, run_diag, snapshot = FM.DRY_ROOT, FM.GRID_DRY, FM.tail, FM.run_script, FM.run_diag, FM.snapshot


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha1_file(p) -> str:
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()


YES_IDS, NO_IDS = frozenset({1, 3, 5, 7, 9}), frozenset({2, 4, 6, 8, 10})      # the stand-in tokenizer's ids; "Yes" is id 1


# ---------------------------------------------------------------- 1. the answer token, the share s, R and C
def trained_answer_id():
    """train_lora_yesno.answer_id, compiled from the module's source."""
    tree = ast.parse((ROOT / "src" / "confrec" / "train_lora_yesno.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "answer_id")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "answer_id", "exec"), ns)
    return ns["answer_id"]


def test_the_answer_token_is_derived_from_the_tokenizer_never_hard_coded():
    tok = sd.DryTok()
    yes, no = yes_no_ids(tok)
    assert (yes, no) == (YES_IDS, NO_IDS)
    assert sd.answer_id(tok, "Yes", yes) == 1 and sd.answer_id(tok, "No", no) == 2
    with pytest.raises(sd.DiagError, match="tokenizes to"):
        sd.answer_id(tok, "Yes Yes", yes)                       # two ids
    with pytest.raises(sd.DiagError, match="not one id from the scorer's set"):
        sd.answer_id(tok, "No", yes)                            # a single id, but not a yes id
    # a tokenizer whose "Yes" is another id (the real Qwen3 one is 9454): the id follows the tokenizer
    class Other(sd.DryTok):
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [5]} if text == "Yes" else super().__call__(text, add_special_tokens)
    assert sd.answer_id(Other(), "Yes", yes) == 5
    # train_lora_yesno's own function agrees (it is what YesNoSet trains the label token with); compiled from its source, as that
    # module imports torch at module level, which nothing here needs
    trained = trained_answer_id()
    assert trained(tok, "Yes", yes) == sd.answer_id(tok, "Yes", yes) and trained(Other(), "Yes", yes) == 5
    with pytest.raises(ValueError):
        trained(tok, "No", yes)
    # no integer constant of the module is the id 9454
    consts = [n.value for n in ast.walk(ast.parse(MODULE.read_text(encoding="utf-8"))) if isinstance(n, ast.Constant)]
    assert 9454 not in consts


def top_list(probs: dict) -> list:
    """[(token id, log p)] of a probability dict, in descending order (what the scorer keeps)."""
    return sorted(((i, math.log(p)) for i, p in probs.items()), key=lambda e: -e[1])


def test_the_share_and_the_scores_on_hand_computed_rows():
    # yes ids 1 (the answer token), 3 and 5; no ids 2 and 4
    top = top_list({1: 0.60, 3: 0.20, 5: 0.10, 2: 0.05, 4: 0.03, 100: 0.02})
    r = sd.row_reading(top, YES_IDS, NO_IDS, 1)
    assert r["s"] == pytest.approx(0.6 / 0.9) and r["censored"] == 0 and r["mass"] == pytest.approx(0.98)
    assert r["lp_yes"] - r["lp_no"] == pytest.approx(math.log(0.9 / 0.08))
    b_z = 0.7
    R = r["lp_yes"] - r["lp_no"] + b_z
    C = sd.consistent_shift([R], [r["s"]], [b_z])[0]
    assert C == pytest.approx(R + math.log(0.6 / 0.9 + (0.3 / 0.9) * math.exp(-b_z)))
    assert C == pytest.approx(math.log((0.6 * math.exp(b_z) + 0.3) / 0.08))            # only the answer token shifted by b z
    # the answer token outside the top-k while another yes id is in it: s = 0, so C is the unshifted logit L0 = R - b z
    top = top_list({3: 0.2, 5: 0.1, 2: 0.4, 4: 0.1, 100: 0.2})
    r = sd.row_reading(top, YES_IDS, NO_IDS, 1)
    assert r["s"] == 0.0
    assert sd.consistent_shift([1.5], [0.0], [b_z])[0] == pytest.approx(1.5 - b_z)
    # only the answer token among the yes ids: s = 1 exactly and C = R
    r = sd.row_reading(top_list({1: 0.5, 2: 0.3, 4: 0.1, 100: 0.1}), YES_IDS, NO_IDS, 1)
    assert r["s"] == 1.0 and sd.consistent_shift([2.0], [1.0], [-0.9])[0] == 2.0
    # no yes id in the top-k: s is undefined (censored 1: the scorer imputes its floor), no id at all: censored 2
    r = sd.row_reading(top_list({2: 0.5, 4: 0.3, 100: 0.2}), YES_IDS, NO_IDS, 1)
    assert r["s"] is None and r["censored"] == 1 and math.isfinite(r["lp_yes"])
    r = sd.row_reading(top_list({100: 0.5, 101: 0.5}), YES_IDS, NO_IDS, 1)
    assert r["s"] is None and r["censored"] == 2 and math.isnan(r["lp_yes"]) and r["mass"] == 0.0
    # the reading is prompting.read_yes_no's, token for token
    top = top_list({1: 0.4, 3: 0.1, 2: 0.2, 6: 0.05, 100: 0.25})
    obj = {t: SimpleNamespace(logprob=v) for t, v in top}
    assert read_yes_no(obj, YES_IDS, NO_IDS) == (sd.row_reading(top, YES_IDS, NO_IDS, 1)["lp_yes"],
                                                 sd.row_reading(top, YES_IDS, NO_IDS, 1)["lp_no"],
                                                 sd.row_reading(top, YES_IDS, NO_IDS, 1)["mass"], 0)


def test_C_is_the_logit_of_the_yes_over_the_no_mass_with_only_the_answer_token_shifted():
    """The defining property, on random distributions: shift the answer token's logit by b z in an explicit softmax over the top-k ids and
    read the summed yes mass over the summed no mass (what training optimises); C equals it to rounding."""
    rng = np.random.default_rng(0)
    worst, n = 0.0, 0
    for _ in range(400):
        logits = rng.normal(0, 3, 60)
        lp = logits - np.log(np.exp(logits).sum())
        top = [(int(i), float(lp[i])) for i in np.argsort(-lp)[:50]]
        present = {i for i, _ in top}
        if not (YES_IDS & present) or not (NO_IDS & present) or 1 not in present:
            continue
        r = sd.row_reading(top, YES_IDS, NO_IDS, 1)
        bz = float(rng.normal(0, 1.2))
        C = float(sd.consistent_shift([r["lp_yes"] - r["lp_no"] + bz], [r["s"]], [bz])[0])
        sh = dict(top)
        sh[1] += bz
        y = math.log(sum(math.exp(v) for t, v in sh.items() if t in YES_IDS))
        no = math.log(sum(math.exp(v) for t, v in sh.items() if t in NO_IDS))
        worst, n = max(worst, abs(C - (y - no))), n + 1
    assert n > 200 and worst < 1e-12
    # no shift, no difference; the sign of the correction follows the sign of b z when s < 1
    s = np.array([0.3, 0.7, 0.99, 1.0, 0.0])
    assert np.allclose(sd.consistent_shift(np.zeros(5), s, np.zeros(5)), 0.0, atol=1e-15)
    assert (sd.consistent_shift(np.zeros(5), s, np.full(5, 0.5))[:3] < 0).all()          # a positive shift on the answer token
    assert (sd.consistent_shift(np.zeros(5), s, np.full(5, -0.5))[:3] > 0).all()         # shrinks / grows the summed yes mass less
    assert np.isnan(sd.consistent_shift([np.nan, 1.0], [0.5, np.nan], [0.1, 0.1])).all()


def test_the_share_statistics_use_numpys_linear_percentile():
    s = np.array([1.0] * 97 + [0.99, 0.6, 0.2])
    st = sd.share_stats(s)
    assert st["n"] == 100 and st["mean"] == pytest.approx(s.mean()) and st["median"] == 1.0
    assert st["p1"] == pytest.approx(np.percentile(s, 1)) and 0.2 < st["p1"] < 0.99          # between the low tail's values
    assert st["min"] == 0.2 and st["share_below_0.99"] == pytest.approx(0.02) and st["p5"] == pytest.approx(np.percentile(s, 5))
    assert sd.share_stats([np.nan])["n"] == 0 and sd.share_stats([])["mean"] is None


def test_the_reading_rule_at_its_boundary_for_every_adapter():
    ok = {"o0": 0.999, "o1": 0.9995, "o2": 0.99}
    assert sd.reading(ok, "OK")["label"] == "EQUIVALENT" and sd.reading(ok, "OK")["equivalent_for_ranking"] is True
    assert sd.reading({**ok, "o1": 0.99 - 1e-13}, "OK")["label"] == "EQUIVALENT"            # inside the 1e-12 tolerance (item 3's habit)
    low = sd.reading({**ok, "o1": 0.98999}, "OK")
    assert low["label"] == "NOT_EQUIVALENT" and low["equivalent_for_ranking"] is False and low["p1_min"] == 0.98999
    assert "inconclusive for the method" in low["fail_reading"] and "failure for the kill rule" in low["fail_reading"]
    assert "stands" in low["pass_reading"] and "works against the method" in low["pass_reading"]
    assert sd.reading({**ok, "o2": 0.9899}, "OK")["label"] == "NOT_EQUIVALENT"            # one adapter below decides, not the pooled one
    assert sd.reading({"o0": 0.999, "o1": 0.999}, "OK")["label"] == "UNDETERMINED"        # an adapter missing
    assert sd.reading({**ok, "o0": None}, "OK")["label"] == "UNDETERMINED"
    assert sd.reading(ok, "INVALID")["label"] == "UNDETERMINED" and sd.reading(ok, "INCOMPLETE")["equivalent_for_ranking"] is None
    assert sd.P1_MIN == 0.99 and sd.N_ROWS == 1000 and sd.TOPK == 50 and sd.SEEDS == (0, 1, 2)


def brute_pairs(users, label, R, C, ok):
    n = disc = differs = both = rev = 0
    for u in set(users):
        idx = [i for i in range(len(users)) if users[i] == u and ok[i]]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                n += 1
                sr, sc = np.sign(R[i] - R[j]), np.sign(C[i] - C[j])
                d = label[i] != label[j]
                disc += d
                if sr != sc:
                    differs, both = differs + 1, both + int(d)
                if (R[i] - R[j]) * (C[i] - C[j]) < 0:
                    rev += 1
    return n, disc, differs, both, rev


def test_the_pairs_whose_order_differs_and_the_uauc_are_brute_force_correct():
    rng = np.random.default_rng(1)
    users = np.repeat(np.arange(30), 6).astype(str)
    label = rng.integers(0, 2, len(users))
    R = rng.normal(0, 1, len(users))
    C = R + rng.normal(0, 0.4, len(users))
    C[5], R[5] = R[4], R[4]                                                  # a tie in both: the same order (no change)
    C[11] = R[10]                                                            # a tie only in C: the sign differs from R's
    ok = rng.random(len(users)) > 0.1
    got = sd.pair_counts(users, label, R, C, ok)
    n, disc, differs, both, rev = brute_pairs(users.tolist(), label, R, C, ok)
    assert (got["n_pairs"], got["n_pairs_label_discordant"], got["n_order_differs"], got["n_order_differs_label_discordant"],
            got["n_strict_reversals"]) == (n, disc, differs, both, rev)
    assert got["n_order_differs"] > 0 and got["share_order_differs"] == pytest.approx(differs / n)
    groups = fx._groups(users)
    for score in (R, C):
        ref = fx.uauc(score, label, groups, ok)
        u, nu = sd.uauc_of(score, label, users, ok)
        assert nu == len(ref) and u == pytest.approx(float(np.mean(list(ref.values()))))
    same = sd.pair_counts(users, label, R, R, ok)
    assert same["n_order_differs"] == 0 and same["n_strict_reversals"] == 0 and same["n_pairs"] == n
    assert sd.uauc_of(R, np.zeros(len(users), int), users, ok) == (None, 0)             # no user with both classes


def test_adapter_stats_excludes_rows_without_a_yes_id_and_counts_them():
    rows = [top_list({1: 0.5, 3: 0.2, 2: 0.2, 100: 0.1}), top_list({2: 0.5, 4: 0.3, 100: 0.2}),            # yes side absent
            top_list({100: 0.6, 101: 0.4}), [], top_list({1: 0.7, 3: 0.05, 2: 0.1, 4: 0.05, 100: 0.1})]
    cens = [0, 1, 2, 3, 0]
    st, arr = sd.adapter_stats(rows, cens, YES_IDS, NO_IDS, 1, [0.5] * 5, 0.2, [1, 0, 1, 0, 1], ["u"] * 5)
    assert st["n_rows"] == 5 and st["n_used"] == 2 and st["n_yes_side_absent"] == 1 and st["censored"] == {"0": 2, "1": 1, "2": 1, "3": 1}
    assert st["censoring_mismatch_rows"] == 0 and arr["ok"].tolist() == [True, False, False, False, True]
    assert st["s"]["n"] == 2 and st["E1"]["ok"] is False and st["E1"]["overlength"] == 1
    assert st["pairs"]["n_pairs"] == 1                                                    # only the two usable rows pair up
    assert math.isnan(arr["R"][2]) and math.isnan(arr["C"][1]) and math.isfinite(arr["R"][1]) and math.isnan(arr["s"][1])
    assert st["mean_yes_no_mass"] == pytest.approx((0.9 + 0.8 + 0.0 + 0.0 + 0.9) / 5)    # the scorer's mean_yes_no_mass: 0 for unscored rows
    st2, _ = sd.adapter_stats(rows, [0, 1, 2, 3, 1], YES_IDS, NO_IDS, 1, [0.5] * 5, 0.2, [1, 0, 1, 0, 1], ["u"] * 5)
    assert st2["censoring_mismatch_rows"] == 1                                            # a stored censoring that is not the recomputation


# ---------------------------------------------------------------- 2. the rows: the first 1,000 CAL candidates in panel order
def synth_panel(tmp: Path, n_users: int = 400, per_user: int = 8, T: float = 500.0, interleave: bool = True) -> tuple:
    """A rated eval panel without prompts' text (the selection needs none): per user `per_user` candidates whose timestamps are before
    T for the odd candidates and after it for the even ones when `interleave` (so panel order is not time order); a candidate exactly at T
    is a TEST row."""
    panels = tmp / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    rows = []
    for u in range(n_users):
        ts = [(100 + j if (j % 2 == 1 or not interleave) and j < per_user - 1 else 1000 + j) for j in range(per_user)]
        if u % 7 == 0:
            ts[0] = T                                                            # ts == T: TEST, not CAL
        rows.append({"user_id": f"u{u:03d}", "source_event_id": f"u{u:03d}::1", "source": "ml1m", "history": [], "history_ratings": [],
                     "candidate_item_ids": [f"i{u}_{j}" for j in range(per_user)], "candidate_labels": [j % 2 for j in range(per_user)],
                     "candidate_timestamps": ts, "candidate_titles": [f"t{j}" for j in range(per_user)]})
    FM.write_jsonl(panels / "eval.jsonl", rows)
    split = panels / "ftgrid_split.json"
    split.write_text(json.dumps({"domain": "ml1m", "T": T, "variant": "V0", "files": {"eval.jsonl": sha1_file(panels / "eval.jsonl")}}),
                     encoding="utf-8")
    return panels, split, rows


def test_the_rows_are_the_first_1000_cal_candidates_in_panel_order(tmp_path):
    panels, split, rows = synth_panel(tmp_path)
    p = sd.Panel(panels, split, 1000)
    cal = [(r["user_id"], i) for r in rows for i, t in zip(r["candidate_item_ids"], r["candidate_timestamps"]) if t < 500.0]
    assert p.n_cal_rows == len(cal) == 1200 and len(p.sel) == 1000
    got = [(str(p.cx.users[k]), str(p.cx.item[k])) for k in p.sel]
    assert got == cal[:1000]                                                          # the file's row order, then candidate order
    assert (p.sel == np.sort(p.sel)).all() and p.cx.ts[p.sel].max() < 500.0
    assert not any(t == 500.0 for t in p.cx.ts[p.sel])                                # ts == T is a TEST row
    assert p.users_cut() == 1 and len({u for u, _ in got}) == 1000 // 3 + 1           # 3 CAL rows per user: the cut is inside the 334th
    assert p.selection_sha1 == sd.sha1_lines(f"{u}\t{i}" for u, i in got) and p.row_of[p.sel[-1]] == p.row_of[p.sel[-1]]
    # the mapping back to (row, candidate) is the panel's
    for k in p.sel[:50]:
        r = rows[int(p.row_of[k])]
        assert r["candidate_item_ids"][int(p.cand_of[k])] == p.cx.item[k] and r["user_id"] == p.cx.users[k]
    small = sd.Panel(panels, split, 10)
    assert len(small.sel) == 10 and small.sel.tolist() == p.sel[:10].tolist() and small.n_rows_requested == 10
    few = sd.Panel(panels, split, 10 ** 6)                                            # fewer than asked: all of them
    assert len(few.sel) == few.n_cal_rows == len(cal)
    assert sd.select_rows(p.cx, 3).tolist() == p.sel[:3].tolist()
    # a panel whose every candidate is a TEST row has nothing to score
    p2, s2, _ = synth_panel(tmp_path / "later", T=50.0)
    assert len(sd.Panel(p2, s2, 1000).sel) == 0


# ---------------------------------------------------------------- 3. a world: panel, q-hat, adapters, the stand-in model
def diag_world(tmp: Path, n_rows: int = 100) -> SimpleNamespace:
    """The synthetic ML-1M of test_confrec_ftmethod (full rated rows with histories and titles), its stage-1 q-hat files and manifest in
    a method directory, and the three prior-offset adapters o0-o2 (placeholder weights, train_config.json and offset.json as the
    report reads them), with the b of each seed."""
    tmp.mkdir(parents=True, exist_ok=True)
    w = FM.ml1m_world(tmp, n_users=70, per_user=7, n_train=10)
    mdir = tmp / "method"
    man = tlo.build_qhat(FM.qhat_args(w, mdir))
    msha = sha1_file(mdir / tlo.MANIFEST)
    b = (0.4, -0.3, 0.2)
    for k in range(3):
        a = mdir / "adapters" / f"o{k}"
        a.mkdir(parents=True)
        (a / "adapter_model.safetensors").write_bytes(f"weights of o{k}".encode())
        (a / "train_config.json").write_text(json.dumps({"seed": k, "variant": "V0", "hist_len_used": 10,
                                                         "prior_offset": {"b": b[k]}}), encoding="utf-8")
        (a / "offset.json").write_text(json.dumps(
            {"format": tlo.OFFSET_FORMAT, "seed": k, "domain": "ml1m", "b": b[k], "standardisation": man["standardisation"],
             "qhat_manifest_sha1": msha, "sft_adapter": str(tmp / f"s{k}"), "yes_token_id": 1, "b_lr": tlo.B_LR,
             "b_group": {"own_group": True, "lr": tlo.B_LR, "weight_decay": 0.0, "n_params_in_group": 1}, "dry_run": True}),
            encoding="utf-8")
    split = w.panels / "ftgrid_split.json"
    n_cal = sd.Panel(w.panels, split, 10 ** 6).n_cal_rows
    return SimpleNamespace(w=w, mdir=mdir, panels=w.panels, split=split, b=b, n_rows=min(n_rows, n_cal), n_cal=n_cal)


def cli(world, cmd: str, *extra, seed=None, dry=True, n_rows=None) -> int:
    argv = [cmd, "--domain", "ml1m", "--split", str(world.split), "--panels", str(world.panels), "--method_dir", str(world.mdir),
            "--n_rows", str(n_rows or world.n_rows)]
    if seed is not None:
        argv += ["--seed", str(seed), "--model", "dryrun/Qwen3-8B", "--variant", "V0"]
    if dry:
        argv.append("--dry_run")
    return sd.main(argv + list(extra))


def run_all(world, **kw) -> dict:
    for k in range(3):
        assert cli(world, "score", seed=k, **kw) == 0
    assert cli(world, "report") == 0
    return read_json(world.mdir / sd.DIR / sd.REPORT_FILE)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    wd = diag_world(tmp_path_factory.mktemp("diag_world"))
    wd.report = run_all(wd)
    return wd


def test_the_selected_rows_of_the_synthetic_world_are_cal_rows_and_the_cut_is_inside_a_user(world):
    p = sd.Panel(world.panels, world.split, world.n_rows)
    assert world.n_cal >= 60 and len(p.sel) == world.n_rows <= p.n_cal_rows and p.cx.ts[p.sel].max() < p.T
    rep = world.report
    assert rep["meta"]["n_rows_selected"] == world.n_rows and rep["meta"]["n_cal_rows"] == p.n_cal_rows
    assert rep["meta"]["selection_sha1"] == p.selection_sha1 and rep["meta"]["n_rows_requested"] == world.n_rows


def test_R_is_the_stage_4_score_and_the_prompts_are_the_scorers(world, tmp_path):
    """The real pyes_scorer runs on the whole eval panel with the same stand-in model (what stage 4 does for the real model): the logits
    it writes for the selected rows are the diagnostic's lp_yes - lp_no up to the six decimals of scores.csv.gz, and the token-id lists the
    scorer sends for those rows are the ones the diagnostic sends."""
    panel = sd.Panel(world.panels, world.split, world.n_rows)
    k = 1
    lora = (world.mdir / "adapters" / f"o{k}").as_posix()                      # the stand-in's tag holds the adapter path as given
    sent_by_scorer, sent_by_diag = [], []

    class Recorder:
        def __init__(self, inner, sink):
            self.inner, self.sink = inner, sink

        def generate(self, prompts, sp, use_tqdm=False, **kw):
            self.sink.extend(list(p["prompt_token_ids"]) for p in prompts)
            return self.inner.generate(prompts, sp, use_tqdm=use_tqdm, **kw)

    def scorer_model(args):
        m = sd.dry_model(SimpleNamespace(model=args.model, lora=args.lora, dry_share=sd.DRY_SHARE, dry_low_frac=sd.DRY_LOW_FRAC))
        m.llm = Recorder(m.llm, sent_by_scorer)
        return m
    out = tmp_path / "scorer"
    args = ps.parse_args(["--data", str(world.panels / "eval.jsonl"), "--output", str(out), "--model", "dryrun/Qwen3-8B", "--dtype", "float16",
                          "--topk_logprobs", "50", "--max_model_len", "4096", "--chunk_users", "100", "--variant", "V0", "--readout", "yesno",
                          "--questions", "like", "--lora", lora])
    ps.run(args, load_model=scorer_model)
    sc = fx.load_scores(out / "scores.csv.gz")
    # the diagnostic's own scoring of the same adapter, with a recording model, in a fresh method dir copy
    mdir2 = tmp_path / "method2"
    shutil.copytree(world.mdir, mdir2)
    shutil.rmtree(mdir2 / sd.DIR)

    def diag_model(a):
        m = sd.dry_model(a)
        m.llm = Recorder(m.llm, sent_by_diag)
        return m
    ns = SimpleNamespace(domain="ml1m", seed=k, split=str(world.split), panels=str(world.panels), method_dir=str(mdir2), model="dryrun/Qwen3-8B",
                         variant="V0", n_rows=world.n_rows, dry_run=True, dry_share=sd.DRY_SHARE, dry_low_frac=sd.DRY_LOW_FRAC,
                         lora_arg=lora, dtype="float16", max_model_len=4096, gpu_mem=0.88, vllm_seed=0)
    sd.score_adapter(ns, load_model=diag_model)
    rows = sd.read_top_file(mdir2 / sd.DIR / f"o{k}" / sd.TOP_FILE)
    assert len(rows) == world.n_rows == len(sent_by_diag)
    pairs = [(r, j) for r in panel.rows for j in range(len(r["candidate_item_ids"]))]
    assert len(sent_by_scorer) == len(pairs)
    ev = {(str(r["source_event_id"]), j): i for i, (r, j) in enumerate(pairs)}
    for row, ids in zip(rows, sent_by_diag):
        i = ev[(row["ev"], row["c"])]
        assert sent_by_scorer[i] == ids                                                # the same ids in the same prompt
        lp = {t: v for t, v in row["top"]}
        r = sd.row_reading(row["top"], YES_IDS, NO_IDS, 1)
        key = (row["ev"], row["c"])
        assert sc["L"]["like"][key] == pytest.approx(r["lp_yes"] - r["lp_no"], abs=6e-7)
        assert row["cens"] == 0 and len(lp) == 50


def test_the_stand_in_model_steers_the_share_and_the_reading(world, tmp_path):
    rep = world.report
    for name in ("o0", "o1", "o2"):
        e = rep["adapters"][name]
        assert e["s"]["p1"] > 0.999 and e["s"]["share_below_0.99"] == 0.0 and e["n_used"] == world.n_rows
        assert e["UAUC"]["R"] == pytest.approx(e["UAUC"]["C"], abs=1e-9) and e["pairs"]["n_order_differs"] == 0
    assert rep["reading"]["label"] == "EQUIVALENT" and rep["status"] == "OK" and rep["meta"]["registered"] is False
    low = diag_world(tmp_path / "low")
    for k in range(3):
        cli(low, "score", "--dry_low_frac", "0.2", seed=k)
    cli(low, "report")
    lr = read_json(low.mdir / sd.DIR / sd.REPORT_FILE)
    assert lr["reading"]["label"] == "NOT_EQUIVALENT" and all(e["s"]["p1"] < 0.9 for e in lr["adapters"].values())
    assert all(0.05 < e["s"]["share_below_0.99"] < 0.4 for e in lr["adapters"].values())
    assert any(e["pairs"]["n_order_differs"] > 0 for e in lr["adapters"].values())      # the low share reorders some within-user pairs
    assert lr["status"] == "OK" and lr["reading"]["p1_min"] < 0.9 and "inconclusive" in lr["reading"]["fail_reading"]


def test_the_report_is_strict_json_with_no_path_or_clock_and_the_blocks_the_addendum_names(world):
    text = (world.mdir / sd.DIR / sd.REPORT_FILE).read_text(encoding="utf-8")
    assert str(world.mdir) not in text and Path(world.mdir).as_posix() not in text and "NaN" not in text and "Infinity" not in text
    rep = json.loads(text)
    assert list(rep["adapters"]) == ["o0", "o1", "o2"] and rep["meta"]["answer_token"] == "Yes" and rep["meta"]["topk"] == 50
    assert rep["meta"]["answer_token_id"] == 1 and rep["meta"]["yes_ids"] == sorted(YES_IDS) and rep["meta"]["n_rows_registered"] == 1000
    for name, k in (("o0", 0), ("o1", 1), ("o2", 2)):
        e = rep["adapters"][name]
        assert e["b"] == world.b[k] and set(e["s"]) == {"n", "mean", "median", "p1", "p5", "min", "share_below_0.99"}
        assert set(e["UAUC"]) == {"R", "C", "C_minus_R", "n_users"} and "n_order_differs" in e["pairs"] and e["E1"]["ok"] is True
    assert rep["pooled"]["s"]["n"] == 3 * world.n_rows and rep["code_sha1"]["ftmethod_shift_diag.py"] == sha1_file(MODULE)
    assert set(rep["code_sha1"]) == set(sd.CODE_FILES) and rep["input_checks"] == {"problems": [], "excluded_adapters": []}
    csv_lines = (world.mdir / sd.DIR / sd.TABLE_FILE).read_text(encoding="utf-8").splitlines()
    assert csv_lines[0].split(",") == list(sd.REPORT_COLS) and [x.split(",")[1] for x in csv_lines[1:]] == ["o0", "o1", "o2", "pooled", "reading"]
    assert csv_lines[-1].endswith("EQUIVALENT")
    assert rep["spec"].startswith("idea-stage/PREREG_AMENDMENT_3_ADDENDUM_10.md section 2")


def test_a_rerun_of_score_and_report_changes_nothing(world, capsys):
    before = {p.name + str(p.parent.name): (p.stat().st_mtime_ns, p.read_bytes()) for p in (world.mdir / sd.DIR).rglob("*") if p.is_file()}
    for k in range(3):
        assert cli(world, "score", seed=k) == 0
    assert capsys.readouterr().out.count("[skip]") == 3
    assert cli(world, "report") == 0 and cli(world, "verify") == 0
    after = {p.name + str(p.parent.name): (p.stat().st_mtime_ns, p.read_bytes()) for p in (world.mdir / sd.DIR).rglob("*") if p.is_file()}
    assert before == after


def tampered(world, tmp_path, name: str):
    """A copy of the world's method directory (and panels) to damage."""
    dest = tmp_path / name
    shutil.copytree(world.mdir, dest / "method")
    shutil.copytree(world.panels, dest / "panels")
    wd = SimpleNamespace(**vars(world))
    wd.mdir, wd.panels, wd.split = dest / "method", dest / "panels", dest / "panels" / "ftgrid_split.json"
    return wd


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, (path, old)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_verify_refuses_every_tampering_and_the_wrong_mode(world, tmp_path, capsys):
    assert cli(world, "verify") == 0 and "verified" in capsys.readouterr().out
    od = lambda wd, k: wd.mdir / sd.DIR / f"o{k}"
    top = lambda wd, k: od(wd, k) / sd.TOP_FILE

    def damage_scoring(wd):
        top(wd, 1).write_bytes(top(wd, 1).read_bytes()[:-5] + b"xxxxx")

    def swap_scoring(wd):
        top(wd, 2).write_bytes(top(wd, 0).read_bytes())

    def retrain(wd):
        (wd.mdir / "adapters" / "o2" / "adapter_model.safetensors").write_bytes(b"retrained")

    def change_panel(wd):
        (wd.panels / "eval.jsonl").write_bytes((wd.panels / "eval.jsonl").read_bytes() + b"\n")

    def change_manifest(wd):
        edit(wd.mdir / tlo.MANIFEST, '"domain": "ml1m"', '"domain": "toys"')

    def drop_scoring(wd):
        shutil.rmtree(od(wd, 0))

    def change_b(wd):
        for f in ("offset.json", "train_config.json"):
            edit(wd.mdir / "adapters" / "o0" / f, '"b": 0.4', '"b": 0.41')

    def edit_report(wd):
        edit(wd.mdir / sd.DIR / sd.REPORT_FILE, '"EQUIVALENT"', '"NOT_EQUIVALENT"')

    def edit_table(wd):
        (wd.mdir / sd.DIR / sd.TABLE_FILE).write_text((wd.mdir / sd.DIR / sd.TABLE_FILE).read_text(encoding="utf-8") + "\n", encoding="utf-8")

    def no_report(wd):
        (wd.mdir / sd.DIR / sd.REPORT_FILE).unlink()

    # (name, damage, the status a fresh `report` step then records and a text of its problems; None: the report step heals it, as the
    # input is a legitimate change and the report is only stale)
    plan = [("scoring file damaged", damage_scoring, ("INVALID", "unreadable")),
            ("scoring file of another adapter", swap_scoring, ("INVALID", "differs from the sha1")),
            ("adapter retrained", retrain, ("INVALID", "scored another adapter")), ("panel changed", change_panel, ("INVALID", "eval_sha1")),
            ("manifest changed", change_manifest, ("INVALID", "domain")), ("a scoring directory missing", drop_scoring, ("INCOMPLETE", "o0")),
            ("b changed", change_b, None), ("report edited", edit_report, None), ("table edited", edit_table, None),
            ("no report", no_report, None)]
    bad = {}
    for name, fn, then in plan:
        wd = tampered(world, tmp_path, name.replace(" ", "_"))
        fn(wd)
        rc, err = cli(wd, "verify"), capsys.readouterr().err
        want = "is not recorded" if name == "no report" else "not the recomputation"
        if rc != 1 or want not in err:
            bad[name] = ("verify", rc, err[-200:])
            continue
        if then:                                                       # the report step records the damage; verify still refuses
            status, why = then
            assert cli(wd, "report") == 0
            capsys.readouterr()
            rep = read_json(wd.mdir / sd.DIR / sd.REPORT_FILE)
            if rep["status"] != status or why not in json.dumps(rep["input_checks"]) or cli(wd, "verify") != 1:
                bad[name] = ("report", rep["status"], json.dumps(rep["input_checks"])[:300])
            capsys.readouterr()
    assert not bad, "\n".join(f"{n}: {v}" for n, v in bad.items())
    assert cli(world, "verify", dry=False) == 1                  # a rehearsal's report is never a real run's
    capsys.readouterr()


def test_a_real_run_needs_real_adapters_and_the_registered_1000_rows(world, tmp_path, capsys):
    """Data scored without --dry_run (here through an injected stand-in, as no model exists) on adapters that are not DRY_RUN stand-ins are
    a real run's: `verify` then accepts them only on the registered 1,000 rows (the world has fewer CAL rows, so 1,000 requested = all of
    them); a request for another number is a rehearsal's whatever the mode."""
    wd = tampered(world, tmp_path, "real")
    shutil.rmtree(wd.mdir / sd.DIR)
    for k in range(3):
        edit(wd.mdir / "adapters" / f"o{k}" / "offset.json", '"dry_run": true', '"dry_run": false')

    def score(n_rows, dry=False):
        for k in range(3):
            ns = SimpleNamespace(domain="ml1m", seed=k, split=str(wd.split), panels=str(wd.panels), method_dir=str(wd.mdir),
                                 model="dryrun/Qwen3-8B", variant="V0", n_rows=n_rows, dry_run=dry, dry_share=sd.DRY_SHARE,
                                 dry_low_frac=sd.DRY_LOW_FRAC, lora_arg=None, dtype="float16", max_model_len=4096, gpu_mem=0.88, vllm_seed=0)
            sd.score_adapter(ns, load_model=sd.dry_model)
    score(1000)
    assert cli(wd, "report", dry=False, n_rows=1000) == 0
    rep = read_json(wd.mdir / sd.DIR / sd.REPORT_FILE)
    assert rep["status"] == "OK" and rep["meta"]["dry_run"] is False and rep["meta"]["registered"] is True
    assert rep["meta"]["n_rows_selected"] == world.n_cal < 1000 and rep["meta"]["n_rows_requested"] == 1000
    assert cli(wd, "verify", dry=False, n_rows=1000) == 0 and "verified" in capsys.readouterr().out
    assert cli(wd, "verify", dry=True, n_rows=1000) == 1                              # the mode must be the one --dry_run says
    capsys.readouterr()
    shutil.rmtree(wd.mdir / sd.DIR)
    score(world.n_rows)                                                               # another number of rows: not the registered run
    assert cli(wd, "report", dry=False, n_rows=world.n_rows) == 0
    assert read_json(wd.mdir / sd.DIR / sd.REPORT_FILE)["meta"]["registered"] is False
    assert cli(wd, "verify", dry=False, n_rows=world.n_rows) == 1 and "registered" in capsys.readouterr().err
    # stand-in adapters scored as a real run (or the reverse) are an input problem, not a reading
    wd2 = tampered(world, tmp_path, "mixed")
    for k in range(3):
        sd.score_adapter(SimpleNamespace(domain="ml1m", seed=k, split=str(wd2.split), panels=str(wd2.panels), method_dir=str(wd2.mdir),
                                         model="dryrun/Qwen3-8B", variant="V0", n_rows=1000, dry_run=False, dry_share=sd.DRY_SHARE,
                                         dry_low_frac=sd.DRY_LOW_FRAC, lora_arg=None, dtype="float16", max_model_len=4096, gpu_mem=0.88,
                                         vllm_seed=0), load_model=sd.dry_model)
    assert cli(wd2, "report", dry=False, n_rows=1000) == 0
    r2 = read_json(wd2.mdir / sd.DIR / sd.REPORT_FILE)
    assert r2["status"] == "INVALID" and any("DRY_RUN stand-in" in x for x in r2["input_checks"]["problems"])
    assert r2["reading"]["label"] == "UNDETERMINED"


# ---------------------------------------------------------------- 4. the scorer is reused by import; nothing heavy at module level
def test_pyes_scorer_is_reused_by_import_and_its_conventions_are_the_stage_4_ones():
    src = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import,)) for a in n.names}
    froms = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {"vllm", "torch", "transformers", "peft"} & imported and not any((m or "").split(".")[0] in {"vllm", "torch", "transformers", "peft"}
                                                                               for m in froms)
    assert "from src.confrec import pyes_scorer as ps" in src
    for needle in ("ps.load_vllm(", "ps.score_ids(", "ps.record_requests(", "ps.check_lora_variant("):
        assert needle in src, needle                                                   # loading, extraction, prompts and checks by import
    # the attributes load_vllm reads from its arguments are the ones vllm_args supplies
    load = next(n for n in ast.walk(ast.parse(Path(ps.__file__).read_text(encoding="utf-8"))) if isinstance(n, ast.FunctionDef)
                and n.name == "load_vllm")
    wanted = {n.attr for n in ast.walk(load) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "args"}
    ns = SimpleNamespace(model="m", lora="l", gpu_mem=0.88, max_model_len=4096, dtype="float16", vllm_seed=0)
    assert wanted <= set(vars(sd.vllm_args(ns)))
    # ... with the values stage 4 passes: float16, 4096, top-50, seed 0 and the scorer's default gpu memory
    d = ps.parse_args(["--data", "x", "--output", "y", "--model", "z"])
    a = sd.parse_args(["score", "--domain", "ml1m", "--split", "s", "--panels", "p", "--method_dir", "m", "--seed", "0", "--model", "q",
                       "--variant", "V1"])
    v = sd.vllm_args(SimpleNamespace(**vars(a), lora="adapter"))
    assert (v.dtype, v.max_model_len, v.topk_logprobs, v.seed, v.gpu_mem) == ("float16", 4096, 50, 0, d.gpu_mem) == (
        d.dtype, d.max_model_len, d.topk_logprobs, d.seed, d.gpu_mem)
    assert (a.dtype, a.max_model_len, a.gpu_mem, a.vllm_seed, a.n_rows) == (d.dtype, d.max_model_len, d.gpu_mem, d.seed, 1000)


def test_importing_the_module_and_recording_need_no_heavy_dependency(tmp_path):
    code = ("import sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('torch', 'vllm', 'transformers', 'peft'):\n"
            "            raise ImportError('blocked ' + name)\n"
            "sys.meta_path.insert(0, Block())\n"
            "import src.confrec.ftmethod_shift_diag as sd\n"
            "assert sd.main(['record', '--print', '--files', 'src/confrec/ftmethod_shift_diag.py']) == 0\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0 and r.stdout.startswith("src/confrec/ftmethod_shift_diag.py = "), r.stderr


def test_the_record_prints_checks_and_appends_the_sha1_of_the_diagnostics_own_files(tmp_path, capsys):
    files = ["scripts/sigir/run_ftmethod_shift_diag.sh", "src/confrec/ftmethod_shift_diag.py", "tests/test_confrec_ftmethod_shift_diag.py"]
    assert sd.main(["record", "--print", "--files", *files]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"{rel} = {sha1_file(ROOT / rel)}" for rel in files]
    log = tmp_path / "PILOT_LOG.md"
    log.write_text("# log\n" + lines[0].upper() + "\n", encoding="utf-8")            # a case-insensitive substring test, as ftgrid_freeze
    assert sd.main(["record", "--pilot_log", str(log), "--files", *files]) == 4 and files[1] in capsys.readouterr().err
    assert sd.main(["record", "--pilot_log", str(log), "--files", *files, "--append"]) == 0
    assert sd.main(["record", "--pilot_log", str(log), "--files", *files]) == 0
    assert sd.main(["record", "--pilot_log", str(tmp_path / "gone.md"), "--files", *files]) == 4
    assert sd.main(["record", "--pilot_log", str(log), "--files", "no/such/file.py"]) == 1
    assert sd.main(["record", "--files", *files]) == 2


def test_the_command_line_is_one_parser_with_four_commands_and_exit_codes(capsys):
    with pytest.raises(SystemExit) as e:
        sd.main(["nonsense"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        sd.main(["score", "--domain", "ml1m"])                                       # required arguments
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        sd.main(["report", "--domain", "beauty", "--split", "s", "--panels", "p", "--method_dir", "m"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        sd.main(["score", "--domain", "ml1m", "--split", "s", "--panels", "p", "--method_dir", "m", "--seed", "3", "--model", "q",
                 "--variant", "V1"])
    assert e.value.code == 2                                                         # the adapters are o0-o2
    assert sd.COMMANDS == ("score", "report", "verify", "record")


# ---------------------------------------------------------------- 5. the script: file, flags, guards
def test_script_is_lf_bash_and_bash_n_clean():
    raw = SCRIPT.read_bytes()
    assert b"\r\n" not in raw and raw.startswith(b"#!/usr/bin/env bash\n") and b"set -euo pipefail" in raw and b"\r\n" not in MODULE.read_bytes()
    if BASH is None:
        pytest.skip(NO_BASH)
    r = subprocess.run([BASH, "-n", SCRIPT.as_posix()], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_every_flag_the_script_passes_exists_in_the_real_argparse_and_the_contracts_audit_accepts_it():
    text = SCRIPT.read_text(encoding="utf-8")
    problems, count = FM.audit(text)
    assert not problems, "\n".join(problems)
    for key, n in {("src.confrec.ftmethod_shift_diag", "score"): 1, ("src.confrec.ftmethod_shift_diag", "report"): 1,
                   ("src.confrec.ftmethod_shift_diag", "verify"): 1, ("src.confrec.ftmethod_shift_diag", "record"): 2,
                   ("src.confrec.ftmethod_report", "slot"): 1, ("src.confrec.ftgrid_freeze", None): 1}.items():
        assert count.get(key, 0) >= n, (key, count)
    bad = text.replace("--lora_arg", "--loraarg").replace("--method_dir", "--methoddir").replace("ftmethod_shift_diag verify", "ftmethod_shift_diag verfy")
    problems, _ = FM.audit(bad)
    for what in ("--loraarg", "--methoddir", "no subcommand 'verfy'"):
        assert any(what in p for p in problems), what
    spec = importlib.util.spec_from_file_location("contracts_for_shift_diag", ROOT / "tests" / "test_confrec_contracts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert SCRIPT in sorted((ROOT / "scripts" / "sigir").glob("*.sh"))
    mod.test_every_script_flag_exists_in_the_real_argparse(SCRIPT)


def test_the_script_writes_only_below_the_dataset_shift_diag_directory():
    """No output flag, redirect, rm, mv, cp, touch or mkdir of the script aims anywhere but $SD (OUT_ROOT/D/shift_diag) or the temporary
    pilot log of a rehearsal; every path of its TREE lies below D/shift_diag (or is the dataset directory itself)."""
    text = SCRIPT.read_text(encoding="utf-8")
    code = re.sub(r"\\\n\s*", " ", "\n".join(x.split("#", 1)[0] for x in text.splitlines()))
    for pat in (r'--out_dir "\$(P|GRID|REG|M|OUT_ROOT)"', r'--output "\$(P|GRID|REG|M|OUT_ROOT)', r'> *"\$(P|GRID|REG|M|REP|QM|OUT_ROOT)',
                r'(rm|mv|cp|touch|mkdir)[^;|&\n]*"\$(P|GRID|REG|REGG|M|OUT_ROOT|REP|QM)[/"]', r"ln -s", r"git "):
        assert not re.search(pat, code), pat
    tree = re.search(r"TREE=\((.*?)\)\n", text, re.S).group(1)
    for token in re.findall(r'"([^"]+)"', tree):
        assert token in ("$D",) or token.startswith("$D/shift_diag"), token
    assert 'SD="$M/shift_diag"' in code and 'mkdir -p "$SD"' in code and 'mkdir -p "$SD/_dry"' in code
    # it only reads the dataset's slot files and the grid's panels
    assert code.count("--method_dir") >= 3 and "ftmethod_report slot" in code and "REP=" in code


def guard_cases() -> list:
    return [("beauty", {}, True, "usage"), ("ml1m", {"STAGES": "3"}, True, "unknown stage"),
            ("ml1m", {"MODEL": "dryrun/Llama-3.1-8B-Instruct"}, True, "single-backbone"),
            ("ml1m", {"OUT_ROOT": "outputs/confrec/ftmethod//"}, True, "never writes to a registered output root"),
            ("ml1m", {"OUT_ROOT": "./outputs/confrec/ftgrid_dryrun/"}, True, "never writes to a registered output root"),
            ("ml1m", {"OUT_ROOT": "outputs/confrec/ftmethod_llama"}, False, "one registered root"),
            ("ml1m", {"DRY_NO_RECORD": "yes"}, True, "DRY_NO_RECORD must be 0 or 1"),
            ("ml1m", {"DRY_NO_RECORD": "1"}, False, "belongs to a DRY_RUN=1 rehearsal"),
            ("ml1m", {"DRY_SHIFT_LOW_FRAC": "0.5"}, False, "stand-in of a DRY_RUN=1 rehearsal"),
            ("ml1m", {"DRY_TODAY": "20261001"}, False, "stand-in of a DRY_RUN=1 rehearsal"),
            ("ml1m", {"DRY_TODAY": "2026-12-01"}, True, "DRY_TODAY must be eight digits")] + [
        ("ml1m", {"DRY_RUN": v}, False, "DRY_RUN must be 0 or 1") for v in ("yes", "true", "2", "", "01")]


@needs_bash
def test_input_guards_refuse_before_anything_is_written(tmp_path):
    repo = FM.registered_repo(tmp_path / "repo")
    before = FM.registered_state(repo)
    cases = guard_cases()
    with cf.ThreadPoolExecutor(max_workers=FM.WORKERS) as ex:
        runs = list(ex.map(lambda c: (c, run_diag(repo, c[0], dry=c[2], **c[1])), cases))
    bad = [(c[1], tail(r)) for c, r in runs if r.returncode != 2 or c[3] not in r.stderr]
    assert not bad, "\n\n".join(f"== {env}\n{t}" for env, t in bad)
    assert FM.registered_state(repo) == before and not (repo / "tmp_outputs").exists()


@needs_bash
def test_a_dry_run_never_writes_under_outputs_confrec_in_any_spelling_and_a_real_one_takes_only_the_registered_root(tmp_path):
    repo = FM.registered_repo(tmp_path / "repo")
    before = FM.registered_state(repo)
    refused = FM.spellings(repo, "outputs/confrec/ftmethod") + ["outputs/confrec/ftgrid//", "outputs/confrec", ".", "/", "src", "../outside_repo",
                                                                  "tmp_outputs/../src"]
    accepted = [FM.DRY_ROOT, FM.DRY_ROOT + "//", "tmp_outputs/a/../b"]
    real_refused = FM.spellings(repo, "outputs/confrec/ftgrid")[:3] + ["outputs/confrec/ftmethod/sub", FM.DRY_ROOT, "."]
    real_accepted = FM.spellings(repo, "outputs/confrec/ftmethod")[:4] + ["outputs/confrec/ftgrid/../ftmethod"]
    jobs = ([("dry-refused", oc, dict(dry=True)) for oc in refused] + [("dry-accepted", oc, dict(dry=True, STAGES="9")) for oc in accepted]
            + [("real-refused", oc, dict(dry=False, MODEL="/m/Qwen3-8B", STAGES="1")) for oc in real_refused]
            + [("real-accepted", oc, dict(dry=False, MODEL="/m/Qwen3-8B", STAGES="1")) for oc in real_accepted])
    with cf.ThreadPoolExecutor(max_workers=FM.WORKERS) as ex:
        runs = list(ex.map(lambda j: (j, run_diag(repo, "ml1m", OUT_ROOT=j[1], **j[2])), jobs))
    bad = []
    for (kind, oc, _), r in runs:
        ok = {"dry-refused": r.returncode == 2 and ("never writes to a registered output root" in r.stderr or "filesystem root" in r.stderr
                                                      or "root is empty" in r.stderr),
              "dry-accepted": r.returncode == 2 and "unknown stage" in r.stderr,
              "real-refused": r.returncode == 2 and "one registered root" in r.stderr,
              "real-accepted": False}[kind]
        if kind == "real-accepted":      # past the root guards, the diagnostic stops at its first gate (no pilot log in the scratch repo)
            ok = r.returncode in (1, 4) and "one registered root" not in r.stderr and "never writes" not in r.stderr
        if not ok:
            bad.append((kind, oc, tail(r)))
    assert not bad, "\n\n".join(f"== {k} {oc!r}\n{t}" for k, oc, t in bad)
    assert FM.registered_state(repo) == before and not (repo / "tmp_outputs").exists()


@needs_bash
def test_links_between_the_registered_root_and_the_files_the_script_writes_are_refused(tmp_path):
    base = FM.registered_repo(tmp_path / "base")
    probe = tmp_path / "probe"
    (probe / "t").mkdir(parents=True)
    if not FM.link_dir(probe / "l", probe / "t"):
        pytest.skip("no directory links can be made here (no symlink privilege, no junctions)")
    FM.unlink_dir(probe / "l")
    slot, grid = "outputs/confrec/ftmethod", "outputs/confrec/ftgrid"
    state = FM.registered_state(base, FM.READ_ONLY_ROOTS)

    def clone_of(name: str) -> Path:
        dest = tmp_path / name / "repo"
        shutil.copytree(base, dest)
        return dest

    def plant(repo: Path, rel: str, target: str) -> None:
        link = repo / rel
        if link.exists():
            shutil.rmtree(link)
        assert FM.link_dir(link, repo / target), rel

    def check(name, repo, r, msg):
        ok = r.returncode == 2 and msg in r.stderr and FM.registered_state(repo, FM.READ_ONLY_ROOTS) == state
        return [] if ok else [(name, tail(r))]

    def dry():
        repo, bad = clone_of("dry"), []
        plant(repo, "tmp_outputs/alias", slot)
        bad += check("alias", repo, run_diag(repo, "ml1m", OUT_ROOT="tmp_outputs/alias/sub"), "never writes to a registered")
        FM.unlink_dir(repo / "tmp_outputs" / "alias")
        plant(repo, FM.GRID_DRY, grid)
        bad += check("world", repo, run_diag(repo, "ml1m"), "never writes to a registered")
        FM.unlink_dir(repo / FM.GRID_DRY)
        plant(repo, FM.DRY_ROOT + "/ml1m/shift_diag", grid + "/panels")
        bad += check("dry child", repo, run_diag(repo, "ml1m"), "redirect")
        return bad

    def real():
        repo, bad = clone_of("real"), []
        shutil.rmtree(repo / slot)
        plant(repo, slot, grid)
        bad += check("registered root", repo, run_diag(repo, "ml1m", dry=False, MODEL="/m/Qwen3-8B", STAGES="1"), "one registered root")
        FM.unlink_dir(repo / slot)
        for child, msg in (("ml1m", "redirect"), ("elsewhere", "is a link")):
            plant(repo, f"{slot}/{child}", grid + "/panels")
            bad += check("child " + child, repo, run_diag(repo, "ml1m", dry=False, MODEL="/m/Qwen3-8B", STAGES="1"), msg)
            FM.unlink_dir(repo / slot / child)
        return bad
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        bad = [x for found in [f.result() for f in [ex.submit(dry), ex.submit(real)]] for x in found]
    assert not bad, "\n\n".join(f"== {name}\n{t}" for name, t in bad)


# ---------------------------------------------------------------- 6. the DRY chain: the rehearsal on run_ftmethod.sh's cached ML-1M state
@pytest.fixture(scope="module", autouse=True)
def prebuild_states(request, tmp_path_factory):
    """The cold build of the shared DRY state overlaps the tests that run before the first one that needs it."""
    FM.prebuild_for(request, tmp_path_factory, {"states": ["ml1m"]})


@pytest.fixture(scope="module")
def states(tmp_path_factory):
    if BASH is None or FAST:
        pytest.skip(NO_BASH if BASH is None else "FTMETHOD_FAST=1")
    return FM.ensure_states(tmp_path_factory, ("ml1m",))


@pytest.fixture(scope="module")
def diag_chain(states, tmp_path_factory):
    repo = FM.make_repo(tmp_path_factory.mktemp("diag_chain") / "repo")
    FM.seed_state(repo, states, "ml1m")
    root = repo / DRY_ROOT
    before = {"root": snapshot(root), "grid": snapshot(repo / GRID_DRY)}
    log_before = (repo / GRID_DRY / "_dry" / "PILOT_LOG.md").read_text(encoding="utf-8")
    r = run_diag(repo, "ml1m")
    assert r.returncode == 0, tail(r)
    return {"repo": repo, "root": root, "r": r, "before": before, "log_before": log_before,
            "after": {"root": snapshot(root), "grid": snapshot(repo / GRID_DRY)}}


@needs_chain
def test_the_diagnostic_chain_writes_only_below_shift_diag_and_records_its_files(diag_chain):
    c = diag_chain
    root, repo = c["root"], c["repo"]
    new = sorted(set(c["after"]["root"]) - set(c["before"]["root"]))
    assert new and all(x.startswith("ml1m/shift_diag/") for x in new), new
    assert sorted(x for x in new) == sorted(
        [f"ml1m/shift_diag/o{k}/{n}" for k in range(3) for n in (sd.META_FILE, sd.TOP_FILE)]
        + [f"ml1m/shift_diag/{sd.REPORT_FILE}", f"ml1m/shift_diag/{sd.TABLE_FILE}"])
    assert {k: v for k, v in c["after"]["root"].items() if k in c["before"]["root"]} == c["before"]["root"]      # nothing else changed
    assert c["after"]["grid"] == c["before"]["grid"]                                                             # the world was not written
    log = (repo / GRID_DRY / "_dry" / "PILOT_LOG.md").read_text(encoding="utf-8")                                # the rehearsal's human step
    assert log.startswith(c["log_before"]) and log != c["log_before"]
    for rel in ("scripts/sigir/run_ftmethod_shift_diag.sh", "src/confrec/ftmethod_shift_diag.py", "tests/test_confrec_ftmethod_shift_diag.py"):
        assert f"{rel} = {sha1_file(repo / rel)}" in log
    out = c["r"].stdout
    for mark in ("freeze check OK (stage amendment)", "freeze check OK (stage core)", "freeze check OK (stage method)",
                 "[dry] record rehearsal", "shift diagnostic record OK", "== stage 1", "== stage 2", "verified: EQUIVALENT"):
        assert mark in out, mark
    marks = [out.index(m) for m in ("freeze check OK (stage method)", "[dry] record rehearsal", "== stage 1", "== stage 2")]
    assert marks == sorted(marks)
    assert out.count("[shift_diag score]") == 3 and "answer token 'Yes' = 1" in out
    rep = read_json(root / "ml1m" / sd.DIR / sd.REPORT_FILE)
    assert rep["status"] == "OK" and rep["meta"]["dry_run"] is True and rep["meta"]["registered"] is False
    assert rep["meta"]["n_rows_selected"] == rep["meta"]["n_cal_rows"] <= 1000 and rep["reading"]["label"] == "EQUIVALENT"
    for k in range(3):
        meta = read_json(root / "ml1m" / sd.DIR / f"o{k}" / sd.META_FILE)
        assert meta["adapter_weights_sha1"] == fm.adapter_weights_sha1(root / "ml1m" / "adapters" / f"o{k}") and meta["dry_run"] is True
        assert meta["answer_token_id"] == 1 and meta["topk"] == 50 and meta["backbone"] == "Qwen3-8B" and meta["variant"] == "V3"


def clone(c: dict, tmp: Path) -> Path:
    dest = tmp / "repo"
    shutil.copytree(c["repo"], dest)
    return dest


def refuses(r, code, *needles):
    """The run ended with `code`, said every needle on stderr and scored nothing."""
    return r.returncode == code and all(n in r.stderr for n in needles) and "[shift_diag score]" not in r.stdout


def scenario_gate_ft_fail(c, repo):
    (repo / GRID_DRY / "_dry" / "gateft" / "gate_ft.json").write_text(json.dumps({"decision": "GATE_FT_FAIL"}), encoding="utf-8")
    r = run_diag(repo, "ml1m")
    return [] if refuses(r, 4, "Gate-FT decision GATE_FT_FAIL") else [tail(r)]


def scenario_a_bound_file_changed_after_the_freeze_check(c, repo):
    bound = repo / "src" / "confrec" / "ftmethod_report.py"
    bound.write_bytes(bound.read_bytes() + b"\n# changed after the freeze check\n")
    r = run_diag(repo, "ml1m")
    return [] if refuses(r, 4, "freeze stage method", "is not in") else [tail(r)]


def own_record_changed(rel):
    """A file of the diagnostic's own record changed after it was recorded, and the rehearsal's human step is not done."""
    def scenario(c, repo):
        path = repo / rel
        path.write_bytes(path.read_bytes() + b"\n# changed after its record\n")
        r = run_diag(repo, "ml1m", DRY_NO_RECORD="1")
        return [] if refuses(r, 4, "its record", "is not in") else [tail(r)]
    return scenario


def scenario_the_kill_rule_refuses_before_the_world_is_looked_at(c, repo):
    root = repo / DRY_ROOT
    FM.write_dataset_report(root, "ml1m", "FAIL", p=0.4)
    FM.write_dataset_report(root, "toys", "FAIL", p=0.3)
    r = run_diag(repo, "games")
    return [] if refuses(r, 4, "killed after toys", "games refused") else [tail(r)]


def scenario_the_slot_report_closes_the_diagnostic(c, repo):
    """Stage 1 is refused once the dataset's slot report exists (the diagnostic is recorded before it, never after); stage 2 (CPU) is not."""
    root = repo / DRY_ROOT
    FM.write_dataset_report(root, "ml1m", "FAIL", p=0.4)                      # a registered-looking report: ml1m is decided
    r = run_diag(repo, "ml1m", STAGES="1")
    errs = [] if refuses(r, 4, "the slot report", "exists") else [tail(r)]
    r = run_diag(repo, "ml1m", STAGES="2")
    return errs + ([] if r.returncode == 0 and "verified" in r.stdout else [tail(r)])


def scenario_a_cut_dataset_is_refused(c, repo):
    log = repo / GRID_DRY / "_dry" / "PILOT_LOG.md"
    log.write_text(log.read_text(encoding="utf-8") + "\n- 2026-10-29 FTMETHOD_NOT_RUN ml1m: cut\n", encoding="utf-8")
    r = run_diag(repo, "ml1m")
    return [] if refuses(r, 4, "ml1m refused", "FTMETHOD_NOT_RUN ml1m") else [tail(r)]


def scenario_a_missing_adapter_is_refused(c, repo):
    shutil.rmtree(repo / DRY_ROOT / "ml1m" / "adapters" / "o2")
    r = run_diag(repo, "ml1m", STAGES="1")
    return [] if refuses(r, 1, "adapter o2 of ml1m is missing") else [tail(r)]


def scenario_the_clock_turning_during_stage_1_stops_the_next_adapter(c, repo):
    """The hard kill date is checked before every seed: the clock reads 2026-11-30 until o0 is scored and 2026-12-01 after it."""
    root = repo / DRY_ROOT
    shutil.rmtree(root / "ml1m" / sd.DIR)
    flip = root / "ml1m" / sd.DIR / "o0" / sd.META_FILE
    r = run_diag(repo, "ml1m", DRY_TODAY=None, BASH_ENV=FM.fake_date_env(repo.parent).as_posix(), FAKE_DATE_FLIP=flip.as_posix())
    ok = (r.returncode == 4 and "it is 20261201" in r.stderr and r.stdout.count("[shift_diag score]") == 1 and flip.is_file()
          and not (root / "ml1m" / sd.DIR / "o1").exists())
    return [] if ok else [tail(r)]


def scenario_a_low_share_reads_not_equivalent(c, repo):
    root = repo / DRY_ROOT
    shutil.rmtree(root / "ml1m" / sd.DIR)
    r = run_diag(repo, "ml1m", DRY_SHIFT_LOW_FRAC="0.2")
    rep = read_json(root / "ml1m" / sd.DIR / sd.REPORT_FILE) if r.returncode == 0 else {}
    ok = r.returncode == 0 and rep["reading"]["label"] == "NOT_EQUIVALENT" and "verified: NOT_EQUIVALENT" in r.stdout
    return [] if ok else [tail(r)]


def scenario_stage_2_alone_rebuilds_the_report_and_stage_1_skips_what_is_scored(c, repo):
    root, errs = repo / DRY_ROOT, []
    rep = root / "ml1m" / sd.DIR / sd.REPORT_FILE
    saved = rep.read_bytes()
    rep.unlink()
    (root / "ml1m" / sd.DIR / sd.TABLE_FILE).unlink()
    r = run_diag(repo, "ml1m", STAGES="2")                                       # the report step alone, from the scoring files
    errs += [] if r.returncode == 0 and rep.read_bytes() == saved else [tail(r)]
    r = run_diag(repo, "ml1m", STAGES="1")                                       # all three scored already: skipped, never rewritten
    return errs + ([] if r.returncode == 0 and r.stdout.count("[skip]") == 3 else [tail(r)])


def scenario_a_rerun_changes_nothing(c, repo):
    root = repo / DRY_ROOT
    before = snapshot(root)
    r = run_diag(repo, "ml1m")
    out = r.stdout
    ok = r.returncode == 0 and snapshot(root) == before and out.count("[skip]") == 3 and "[shift_diag score]" not in out
    return [] if ok else [tail(r)]


SCENARIOS = {
    "gate_ft_fail": scenario_gate_ft_fail, "bound_file": scenario_a_bound_file_changed_after_the_freeze_check,
    "record_module": own_record_changed("src/confrec/ftmethod_shift_diag.py"),
    "record_tests": own_record_changed("tests/test_confrec_ftmethod_shift_diag.py"),
    "kill_rule": scenario_the_kill_rule_refuses_before_the_world_is_looked_at,
    "slot_report_exists": scenario_the_slot_report_closes_the_diagnostic, "cut": scenario_a_cut_dataset_is_refused,
    "adapter_missing": scenario_a_missing_adapter_is_refused,
    "date_flip": scenario_the_clock_turning_during_stage_1_stops_the_next_adapter,
    "low_share": scenario_a_low_share_reads_not_equivalent, "stage_2_alone": scenario_stage_2_alone_rebuilds_the_report_and_stage_1_skips_what_is_scored,
    "rerun": scenario_a_rerun_changes_nothing}


@needs_chain
def test_refusals_the_date_and_the_reading_on_clones_of_the_diagnostic_chain(diag_chain, tmp_path):
    """Every scenario works on its own clone of the finished chain (one to two script runs); they run concurrently."""
    def work(name):
        sub = tmp_path / name
        sub.mkdir()
        return name, SCENARIOS[name](diag_chain, clone(diag_chain, sub))
    with cf.ThreadPoolExecutor(max_workers=FM.WORKERS) as ex:
        results = dict(ex.map(work, SCENARIOS))
    failed = {name: errs for name, errs in results.items() if errs}
    assert not failed, "\n\n".join(f"== {name}\n" + "\n".join(map(str, errs)) for name, errs in failed.items())
