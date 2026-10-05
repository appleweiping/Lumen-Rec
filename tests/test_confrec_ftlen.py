"""Tests of FT-L, the longer-training robustness arm (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_9.md; exploratory):
src/confrec/ftlen_panel.py and scripts/sigir/run_ftlen.sh. CPU only, deterministic, no network, no GPU, no model, no torch.

  * the training vector of s0 / s1 equals the real adapters' recorded vector except --out, --seed and --epochs (3), with --train kept,
    for the Amazon layout and for ML-1M, whose real adapters are Gate-FT's; a registered 1-epoch adapter is never an FT-L adapter;
  * the scoring vectors of the two arms (like on eval.jsonl, swap with --swap_k 8 on eval_sd_test.jsonl) equal the real s0's recorded
    passes except --data, --output, --lora; the recorded-input refusals; provenance; the link of the zero-shot scores;
  * the script is LF, `bash -n` clean, passes the flag audit, mirrors run_ftgrid.sh's helpers and E1 rule (one deliberate difference),
    and refuses (wrong dataset, model or root in any spelling, DRY_RUN anywhere in the repo or beside it, planted links, no Gate-FT
    PASS, a missing or stale freeze or FT-L record, a wrong recorded input);
  * run_ftlen.sh DRY_RUN=1 end to end on run_ftgrid.sh's own synthetic worlds of Toys (the Amazon layout) and ML-1M (Gate-FT's
    adapters): real code everywhere except the trainer and scorer stand-ins, the refusals and the E1 rule on clones of the chain, the
    sticky FAILED_INTEGRITY, and ftgrid_extra (the module as it is) building its blocks on the FT-L root.

Run time: the DRY worlds (run_ftgrid.sh's stages 0-4) are built once and cached across sessions (a sha1 of the code that builds them
is the key). The two chains and then the refusal scenarios start with the module, in the background, and run concurrently while the
unit and guard tests run (every script run is dominated by process starts); the tests that need a result wait for it. The DRY_RUN
report bootstraps 20 resamples. FTLEN_FAST=1 skips every test that runs the script's DRY chain.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from src.confrec import ftlen_panel as fl
from src.confrec import ftq_panel as fq
from src.confrec import pyes_scorer as ps

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sigir" / "run_ftlen.sh"
PANEL = ROOT / "src" / "confrec" / "ftlen_panel.py"
FAST = os.environ.get("FTLEN_FAST") == "1"


def _load(name: str):
    """A sibling test module as a library (its helpers; it is never edited here)."""
    spec = importlib.util.spec_from_file_location(f"{name}_for_ftlen", ROOT / "tests" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FR = _load("test_confrec_ftgrid_run")      # run_ftgrid.sh's chain helpers, its flag audit, the stand-ins' source
FQT = _load("test_confrec_ftq")            # FT-Q's test helpers (the real adapters' configs, links, shell parsing)
MODEL, VARIANT = FQT.MODEL, FQT.VARIANT
LAYOUTS = FQT.LAYOUTS
LROOT = "outputs/confrec/ftgrid_len3"
sha1, sha1_file, read_json, jsonl = FQT.sha1, FQT.sha1_file, FQT.read_json, FQT.jsonl
flag_pairs, shell_function, link_dir, unlink_dir = FQT.flag_pairs, FQT.shell_function, FQT.link_dir, FQT.unlink_dir
needs_bash = pytest.mark.skipif(FR.BASH is None, reason=FR.NO_BASH)
needs_chain = pytest.mark.skipif(FR.BASH is None or FAST, reason=FR.NO_BASH if FR.BASH is None else "FTLEN_FAST=1")


# ================================================================ 1. the recorded recipe (both layouts)
@pytest.fixture(scope="module")
def trainer_parser():
    """The real argparse parser of train_lora_yesno (torch replaced by stand-in modules for the import)."""
    with FR.stub_torch():
        parser = FR._parser("src.confrec.train_lora_yesno")
    return parser


def real_base(parser, layout: str, **kw) -> dict:
    """The recorded config of the real adapter s0 (the registered recipe), validated like the script does."""
    return FQT.recipe(FQT.good_cfgs(parser, layout), layout, **kw)


@pytest.mark.parametrize("layout, k", [("toys", 0), ("toys", 1), ("ml1m", 0), ("ml1m", 1)])
def test_the_ftlen_training_vector_is_the_recorded_vector_with_epochs_three_and_train_kept(trainer_parser, layout, k):
    """s_k is trained with the arguments of the real adapters' train_config.json: only --out, --seed and --epochs (3) change, and
    --train stays the real adapters' own (Gate-FT's file for ML-1M). Parsed by the real trainer parser the vectors agree."""
    base = real_base(trainer_parser, layout)
    out = f"{LROOT}/adapters/{layout}/s{k}"
    argv = fl.len_argv(base, out=out, seed=k)
    got = vars(trainer_parser.parse_args(argv))
    lay = LAYOUTS[layout]
    recorded = vars(trainer_parser.parse_args(                            # the vector of the real s0's train call, re-parsed
        ["--train", lay["train_ref"], "--model", FQT.MODEL, "--out", f"{lay['adapters']}/s0", "--variant", FQT.VARIANT, "--seed", "0",
         "--max_len", "1024", "--bsz", "8", "--grad_accum", "4"]))
    keep = [x for x in fq.TRAIN_KEYS if x not in ("out", "seed", "epochs")]
    assert set(got) == set(fq.TRAIN_KEYS)
    assert {x: got[x] for x in keep} == {x: recorded[x] for x in keep} == {x: base[x] for x in keep}     # --train included
    assert (got["epochs"], recorded["epochs"], got["out"], got["seed"]) == (3.0, 1.0, out, k)
    pairs = flag_pairs(argv)                                              # the token level: every recorded flag once, epochs 3
    assert pairs["--epochs"] == "3" and pairs["--train"] == lay["train_ref"] and pairs["--out"] == out
    assert set(pairs) == {f"--{x}" for x in fq.TRAIN_KEYS if base[x] is not None}


def test_len_flags_leave_out_only_out_and_seed_and_unset_flags(trainer_parser):
    cfg = real_base(trainer_parser, "toys")
    flags = fl.len_flags(cfg)
    assert flags[:2] == ["--train", LAYOUTS["toys"]["train_ref"]] and "--out" not in flags and "--seed" not in flags
    assert flag_pairs(flags)["--epochs"] == "3" and "--hist_len" not in flags and "--max_examples" not in flags
    cfg = {**cfg, "hist_len": 20, "max_examples": 100, "lr": 2.5e-05}
    pairs = flag_pairs(fl.len_flags(cfg))
    assert (pairs["--hist_len"], pairs["--max_examples"], pairs["--lr"]) == ("20", "100", "2.5e-05")
    got = vars(trainer_parser.parse_args(fl.len_argv(cfg, out="o", seed=1)))
    assert got["epochs"] == 3.0 and got["hist_len"] == 20 and got["lr"] == 2.5e-05
    with pytest.raises(fq.FtqError, match="no record of"):
        fl.len_flags({k: v for k, v in cfg.items() if k != "lora_r"})


def len3_config(parser, layout: str, k: int, **kw) -> dict:
    """The train_config.json an FT-L adapter s<k> records: the real adapters' vector with epochs 3 and its own out and seed."""
    base = real_base(parser, layout)
    argv = fl.len_argv(base, out=f"{LROOT}/adapters/{layout}/s{k}", seed=k)
    cfg = {**vars(parser.parse_args(argv)), **{x: base[x] for x in fq.TRAIN_FACTS}}
    return {**json.loads(json.dumps(cfg)), **kw}


@pytest.mark.parametrize("layout", ["toys", "ml1m"])
def test_a_trained_ftlen_adapter_must_record_the_real_adapters_arguments_with_three_epochs(trainer_parser, layout):
    ref = real_base(trainer_parser, layout)
    out = f"{LROOT}/adapters/{layout}/s1"
    good = len3_config(trainer_parser, layout, 1)
    fl.check_len_training(ref, good, out=out, seed=1)
    for change, msg in (({"epochs": 1.0}, "registered 1-epoch adapter"), ({"epochs": 2.0}, "epochs is 2.0"),
                        ({"lr": 2e-4}, "lr"), ({"max_len": 512}, "max_len"), ({"bsz": 4}, "bsz"), ({"n_examples": 15000}, "n_examples"),
                        ({"variant": "V0"}, "variant"), ({"train": "other/train.jsonl"}, "train is"), ({"seed": 0}, "seed is"),
                        ({"out": f"{LROOT}/adapters/{layout}/s0"}, "out is"), ({"lora_r": 8}, "lora_r")):
        with pytest.raises(fq.FtqError, match=msg):
            fl.check_len_training(ref, {**good, **change}, out=out, seed=1)


def write_real_adapters(tmp: Path, parser, layout: str, **change) -> tuple:
    """The real adapters' train_config.json files below tmp (paths of the recorded configs rewritten to tmp): (adapters dir, train
    file, the recorded train path)."""
    adapters, train = tmp / "real_adapters", tmp / "train.jsonl"
    tmp.mkdir(parents=True, exist_ok=True)
    train.write_bytes(b"rows\n")
    for k in (0, 1, 2):
        cfg = {**FQT.sft_config(parser, k, layout), "out": f"{adapters.as_posix()}/s{k}", "train": train.as_posix(), **change}
        (adapters / f"s{k}").mkdir(parents=True, exist_ok=True)
        (adapters / f"s{k}" / "train_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return adapters, train


@pytest.mark.parametrize("layout", ["toys", "ml1m"])
def test_the_recipe_command_prints_the_flags_and_refuses_what_is_not_the_registered_recipe(trainer_parser, tmp_path, layout, capsys):
    adapters, train = write_real_adapters(tmp_path, trainer_parser, layout)
    split = tmp_path / "split.json"
    split.write_text(json.dumps(FQT.SPLIT_RECIPE), encoding="utf-8")
    tail = ["--split", str(split)] + (["--split_recipe"] if layout == "toys" else [])
    common = ["recipe", "--adapters", adapters.as_posix(), "--model", FQT.MODEL, "--variant", FQT.VARIANT, "--train_ref", train.as_posix()]
    base = common + ["--train_file", train.as_posix()] + tail
    assert fl.main(base) == 0
    flags = capsys.readouterr().out.split()
    assert flags[:2] == ["--train", train.as_posix()] and flag_pairs(flags)["--epochs"] == "3" and "--seed" not in flags
    assert set(flag_pairs(flags)) == {f"--{k}" for k in fq.TRAIN_KEYS if k not in ("out", "seed", "hist_len", "max_examples")}
    other = tmp_path / "other.jsonl"
    other.write_bytes(b"other rows\n")
    assert fl.main(common + ["--train_file", str(other)] + tail) == 1                  # not the registered TRAIN bytes
    assert "differs in bytes" in capsys.readouterr().err
    assert fl.main(common + tail) == 0 and flag_pairs(capsys.readouterr().out.split())["--epochs"] == "3"    # --train_file is optional
    s1 = adapters / "s1" / "train_config.json"                                       # real adapters that do not share one recipe
    keep = s1.read_text(encoding="utf-8")
    s1.write_text(json.dumps({**json.loads(keep), "lr": 5e-05}), encoding="utf-8")
    assert fl.main(base) == 1 and "not trained with one recipe" in capsys.readouterr().err
    s1.write_text(json.dumps({**json.loads(keep), "seed": 7}), encoding="utf-8")
    assert fl.main(base) == 1 and "seed is 7" in capsys.readouterr().err
    s1.write_text(keep, encoding="utf-8")
    shutil.rmtree(adapters / "s2")
    assert fl.main(base) == 1 and "missing the real adapter s2" in capsys.readouterr().err
    assert fl.main(["recipe", "--adapters", str(adapters)]) == 2                                    # a usage refusal
    for i, change in enumerate(({"epochs": 3.0}, {"lr": 5e-05}, {"lora_r": 8}, {"mode": "mirror"})):
        adapters2, train2 = write_real_adapters(tmp_path / f"not_registered_{i}", trainer_parser, layout, **change)
        args = ["recipe", "--adapters", adapters2.as_posix(), "--model", FQT.MODEL, "--variant", FQT.VARIANT, "--train_ref",
                train2.as_posix()]
        assert fl.main(args) == 1 and "registered recipe" in capsys.readouterr().err   # real adapters that are not the 1-epoch ones


# ================================================================ 2. the recorded passes of the two arms (both layouts)
@pytest.fixture(scope="module")
def fakes(tmp_path_factory):
    """run_ftgrid.sh's DRY_RUN stand-ins, imported from the source the script writes (the real scorer, a fake model)."""
    d = tmp_path_factory.mktemp("ftlen_fakes")
    (d / "ftgrid_fakes.py").write_text(FR.fakes_source(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ftgrid_fakes_for_ftlen", d / "ftgrid_fakes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_argv(data, out, lora, arm: str = "like") -> list:
    """The pyes_scorer call of run_ftgrid.sh's score() for an arm of an adapter."""
    return ["--data", str(data), "--output", str(out), "--model", FQT.MODEL, "--dtype", "float16", "--topk_logprobs", "50",
            "--max_model_len", "4096", "--chunk_users", "100", "--variant", FQT.VARIANT, "--readout", "yesno",
            "--questions", "like", "--lora", lora] + (["--swap_k", "8"] if arm == "swap" else [])


@pytest.fixture(scope="module", params=["toys", "ml1m"])
def arm_world(request, tmp_path_factory, fakes):
    """The real s0's like and swap passes on tiny rated panels, recorded as run_ftgrid.sh records them (run.key and the real scorer's
    report.json), the adapters in the layout's directory (the grid's, or Gate-FT's for ML-1M)."""
    root = tmp_path_factory.mktemp(f"ftlen_arms_{request.param}")
    panels = {"like": FR.write_rows(root / "eval.jsonl", FR.rated_rows(n=6, hist=20)),
              "swap": FR.write_rows(root / "eval_sd_test.jsonl", FR.rated_rows(n=6, hist=20, seed=1))}
    adapters = (root / ("gateft/adapters" if request.param == "ml1m" else "adapters")).as_posix()
    scores, q_adapters = (root / "scores").as_posix(), (root / "q/adapters").as_posix()
    for d, name in ((adapters, "s0"), (q_adapters, "s0"), (q_adapters, "s1")):
        (Path(d) / name).mkdir(parents=True)
        (Path(d) / name / "adapter_model.safetensors").write_bytes(f"weights of {name} in {d}".encode())
    keys = {}
    for arm in fl.ARMS:
        fakes.scorer(score_argv(panels[arm], f"{scores}/s0/{arm}", f"{adapters}/s0", arm))
        keys[arm] = (f"{sha1_file(panels[arm])} {FQT.MODEL} {FQT.VARIANT} {fq.weights_sha1(Path(adapters) / 's0')} --lora {adapters}/s0"
                     + (" --swap_k 8" if arm == "swap" else ""))
        (root / "scores" / "s0" / arm / "run.key").write_text(keys[arm] + "\n", encoding="utf-8")
    return {"root": root, "panels": panels, "adapters": adapters, "scores": scores, "keys": keys, "q_adapters": q_adapters,
            "layout": request.param}


def scoring_args(w, arm, **kw) -> argparse.Namespace:
    d = dict(command="scoring", arm=arm, scores=w["scores"], adapters=w["adapters"], data=str(w["panels"][arm]), model=FQT.MODEL,
             variant=FQT.VARIANT)
    d.update(kw)
    return argparse.Namespace(**d)


def test_the_ftlen_scoring_vectors_equal_the_recorded_passes_except_data_output_lora(arm_world, fakes):
    """Both arms: the flags the script passes, plus the literal --data/--output/--model/--lora (and --swap_k 8 for swap), are the real
    s0's recorded call; and what the scorer records for s0 of the FT-L root equals what it recorded for the real s0 except `lora`."""
    w = arm_world
    like_flags, swap_flags = fl.cmd_scoring(scoring_args(w, "like")), fl.cmd_scoring(scoring_args(w, "swap"))
    assert like_flags == swap_flags                                       # the arm adds --swap_k and its panel, nothing else
    for arm, flags in (("like", like_flags), ("swap", swap_flags)):
        mine = score_argv(w["panels"][arm], f"{w['scores']}/s1/{arm}", f"{w['q_adapters']}/s1", arm)
        sibling = score_argv(w["panels"][arm], f"{w['scores']}/s0/{arm}", f"{w['adapters']}/s0", arm)
        assert flag_pairs(flags) == {f: v for f, v in flag_pairs(sibling).items()
                                     if f not in ("--data", "--output", "--model", "--lora", "--swap_k")}
        a = vars(ps.parse_args(["--data", str(w["panels"][arm]), "--output", f"{w['scores']}/s1/{arm}", "--model", FQT.MODEL] + flags
                               + ["--lora", f"{w['q_adapters']}/s1"] + (["--swap_k", "8"] if arm == "swap" else [])))
        b, c = vars(ps.parse_args(mine)), vars(ps.parse_args(sibling))
        assert a == b and {k: v for k, v in a.items() if k not in ("output", "lora")} == {
            k: v for k, v in c.items() if k not in ("output", "lora")}
        fakes.scorer(["--data", str(w["panels"][arm]), "--output", f"{w['scores']}/s1/{arm}", "--model", FQT.MODEL] + flags
                     + ["--lora", f"{w['q_adapters']}/s1"] + (["--swap_k", "8"] if arm == "swap" else []))
        ref = fq.report_config(read_json(f"{w['scores']}/s0/{arm}/report.json"), "s0")
        run = fq.report_config(read_json(f"{w['scores']}/s1/{arm}/report.json"), "s1")
        fl.check_same_arm_scoring(arm, ref, run, lora=f"{w['q_adapters']}/s1", where="s1")
        assert {k for k in ref if ref[k] != run[k]} == {"lora"} and (arm == "like") == (ref["swap_k"] == 0)
    assert fl.main(["verify_scores", "--arm", "swap", "--scores", w["scores"], "--ref_scores", w["scores"], "--adapters", w["q_adapters"],
                    "--seeds", "1"]) == 0


def record_args(w, arm, **kw) -> dict:
    d = dict(data_sha1=sha1_file(w["panels"][arm]), model=FQT.MODEL, variant=FQT.VARIANT, lora=f"{w['adapters']}/s0",
             lora_weights_sha1=fq.weights_sha1(Path(w["adapters"]) / "s0"))
    d.update(kw)
    return d


def test_the_recorded_passes_are_validated_against_the_registered_ones(arm_world):
    w = arm_world
    for arm in fl.ARMS:
        cfg = read_json(f"{w['scores']}/s0/{arm}/report.json")["config"]
        assert fl.check_arm_record(arm, w["keys"][arm], cfg, **record_args(w, arm)) == [
            "--dtype", "float16", "--topk_logprobs", "50", "--max_model_len", "4096", "--chunk_users", "100", "--variant",
            FQT.VARIANT, "--readout", "yesno", "--questions", "like"]
        key = w["keys"][arm].split()
        bad_keys = {"panel": " ".join(["0" * 40] + key[1:]), "model": " ".join([key[0], "/m/Llama"] + key[2:]),
                    "variant": " ".join(key[:2] + ["V0"] + key[3:]), "weights": " ".join(key[:3] + ["1" * 40] + key[4:]),
                    "extra args": w["keys"][arm] + " --seed 3", "other adapter": w["keys"][arm].replace("/s0", "/s1"),
                    "the other arm's arguments": (w["keys"][arm].replace(" --swap_k 8", "") if arm == "swap"
                                                  else w["keys"][arm] + " --swap_k 8")}
        for text in bad_keys.values():
            with pytest.raises(fq.FtqError, match="run.key"):
                fl.check_arm_record(arm, text, cfg, **record_args(w, arm))
    with pytest.raises(fq.FtqError, match="unknown arm"):
        fl.check_arm_record("nohist", w["keys"]["like"], {}, **record_args(w, "like"))


@pytest.mark.parametrize("arm", ["like", "swap"])
@pytest.mark.parametrize("change, msg", [
    ({"dtype": "bfloat16"}, "dtype"), ({"topk_logprobs": 20}, "topk_logprobs"), ({"max_model_len": 2048}, "max_model_len"),
    ({"chunk_users": 50}, "chunk_users"), ({"readout": "digits"}, "readout"), ({"questions": ["like", "dislike"]}, "questions"),
    ({"swap_k": 3}, "swap_k"), ({"swap_k": 0}, "swap_k"), ({"seed": 1}, "seed"), ({"n_users": 10}, "n_users"),
    ({"chunk_items": 10}, "chunk_items"), ({"hist_len": 0}, "hist_len"), ({"data_sha1": "0" * 40}, "data_sha1"),
    ({"model": "/m/Llama"}, "model"), ({"variant": "V0"}, "variant"), ({"lora": "elsewhere/s0"}, "lora")])
def test_a_pass_that_is_not_the_registered_one_is_refused(arm_world, arm, change, msg):
    w = arm_world
    if arm == "like" and change.get("swap_k") == 0:
        pytest.skip("a like pass records swap_k 0 already")
    cfg = {**read_json(f"{w['scores']}/s0/{arm}/report.json")["config"], **change}
    with pytest.raises(fq.FtqError, match=msg):
        fl.check_arm_record(arm, w["keys"][arm], cfg, **record_args(w, arm))


def test_the_scoring_command_refuses_a_missing_or_stale_pass(arm_world, tmp_path):
    w = arm_world
    for arm in fl.ARMS:
        assert fl.cmd_scoring(scoring_args(w, arm))
        with pytest.raises(fq.FtqError, match=f"missing the real {arm} pass of s0"):
            fl.cmd_scoring(scoring_args(w, arm, scores=str(tmp_path / "none")))
        other = FR.write_rows(tmp_path / f"{arm}2.jsonl", FR.rated_rows(n=7, hist=20, seed=5))     # the panel changed since the pass
        with pytest.raises(fq.FtqError, match="panel sha1"):
            fl.cmd_scoring(scoring_args(w, arm, data=str(other)))
    root = tmp_path / "copy"
    shutil.copytree(w["root"], root)
    a2, s2 = (root / Path(w["adapters"]).relative_to(w["root"])).as_posix(), (root / "scores").as_posix()
    for arm in fl.ARMS:
        (root / "scores" / "s0" / arm / "run.key").write_text(w["keys"][arm].replace(w["adapters"], a2) + "\n", encoding="utf-8")
        rep = read_json(root / "scores" / "s0" / arm / "report.json")
        rep["config"]["lora"] = f"{a2}/s0"
        (root / "scores" / "s0" / arm / "report.json").write_text(json.dumps(rep), encoding="utf-8")
        assert fl.cmd_scoring(scoring_args(w, arm, scores=s2, adapters=a2))                    # a consistent copy: accepted
    (Path(a2) / "s0" / "adapter_model.safetensors").write_bytes(b"retrained")                  # the adapter changed since the passes
    for arm in fl.ARMS:
        with pytest.raises(fq.FtqError, match="adapter weights sha1"):
            fl.cmd_scoring(scoring_args(w, arm, scores=s2, adapters=a2))
    assert fl.main(["scoring", "--arm", "swap", "--scores", s2, "--adapters", a2, "--data", str(w["panels"]["swap"]), "--model",
                    FQT.MODEL, "--variant", FQT.VARIANT]) == 1
    assert fl.main(["scoring", "--scores", s2]) == 2                                            # a usage refusal


def test_verify_scores_catches_every_other_argument_per_arm(arm_world, fakes):
    w = arm_world
    for arm in fl.ARMS:
        ref = fq.report_config(read_json(f"{w['scores']}/s0/{arm}/report.json"), "s0")
        for flag, value, key in (("--topk_logprobs", "40", "topk_logprobs"), ("--chunk_users", "50", "chunk_users"),
                                 ("--dtype", "bfloat16", "dtype"), ("--hist_len", "0", "hist_len"), ("--seed", "1", "seed")):
            out = f"{w['scores']}/bad_{arm}_{key}/{arm}"
            fakes.scorer(score_argv(w["panels"][arm], out, f"{w['q_adapters']}/s1", arm) + [flag, value])
            run = fq.report_config(read_json(f"{out}/report.json"), "bad")
            with pytest.raises(fq.FtqError, match=key):
                fl.check_same_arm_scoring(arm, ref, run, lora=f"{w['q_adapters']}/s1", where="bad")
        with pytest.raises(fq.FtqError, match="scored with"):
            fl.check_same_arm_scoring(arm, ref, {**ref, "lora": "x"}, lora=f"{w['q_adapters']}/s1", where="s1")


# ================================================================ 3. provenance, the zero-shot link, the record, info
def test_adapter_provenance_binds_an_adapter_to_three_epochs_and_the_train_panel(trainer_parser, tmp_path, capsys):
    real, q = tmp_path / "real", (tmp_path / "q").as_posix()
    train = tmp_path / "train.jsonl"
    train.write_bytes(b"registered train rows")
    s0 = FQT.sft_config(trainer_parser, 0, "toys")
    (real / "s0").mkdir(parents=True)
    (real / "s0" / "train_config.json").write_text(json.dumps(s0), encoding="utf-8")
    cfgs = {}
    for k in (0, 1):
        (Path(q) / f"s{k}").mkdir(parents=True)
        cfgs[k] = {**vars(trainer_parser.parse_args(fl.len_argv(s0, out=f"{q}/s{k}", seed=k))), **{x: s0[x] for x in fq.TRAIN_FACTS}}
        (Path(q) / f"s{k}" / "train_config.json").write_text(json.dumps(cfgs[k]), encoding="utf-8")
    args = ["verify_adapter", "--adapters", q, "--ref_adapters", str(real), "--seeds", "0,1", "--train_file", str(train)]
    assert fl.main(args) == 1 and "has no ftlen.json" in capsys.readouterr().err          # trained by hand: refused
    assert fl.main(args + ["--write"]) == 0
    prov = read_json(Path(q) / "s1" / "ftlen.json")
    assert (prov["adapter"], prov["seed"], prov["epochs"], prov["train_sha1"]) == ("s1", 1, 3, sha1(b"registered train rows"))
    assert prov["args"] == {x: cfgs[1][x] for x in fq.TRAIN_KEYS}
    assert fl.main(args) == 0                                                              # an existing adapter is accepted
    for change, msg in (({"epochs": 1}, "not s1 with 3"), ({"adapter": "s0"}, "not s1 with 3"), ({"train_sha1": "0" * 40}, "TRAIN file")):
        (Path(q) / "s1" / "ftlen.json").write_text(json.dumps({**prov, **change}), encoding="utf-8")
        assert fl.main(args) == 1 and msg in capsys.readouterr().err
    fl.write_provenance(Path(q) / "s1", seed=1, train_sha1=sha1(b"registered train rows"), args={})
    train.write_bytes(b"another train file")                                               # the registered panel changed
    assert fl.main(args) == 1 and "TRAIN file" in capsys.readouterr().err
    train.write_bytes(b"registered train rows")
    (Path(q) / "s0" / "train_config.json").write_text(json.dumps({**cfgs[0], "epochs": 1.0}), encoding="utf-8")
    assert fl.main(args) == 1 and "registered 1-epoch adapter" in capsys.readouterr().err   # a registered adapter in the root
    assert fl.main(args[:-4] + ["--seeds", "2", "--train_file", str(train)]) == 2


def test_the_zero_shot_scores_are_linked_never_copied_and_the_scores_directory_is_its_own(tmp_path):
    real = FQT.make_real_scores(tmp_path / "grid" / "scores" / "toys", ("zeroshot",))
    q = tmp_path / "q" / "scores" / "toys"
    (tmp_path / "probe").mkdir()
    if FQT.can_symlink(tmp_path / "probe"):
        notes = fl.cmd_link(argparse.Namespace(real_scores=str(real), q_scores=str(q), models="zeroshot", allow_copy=False))
        assert len(notes) == 1 and notes[0].startswith("[link]") and os.path.islink(q / "zeroshot")
        assert os.path.realpath(q / "zeroshot") == os.path.realpath(real / "zeroshot") and not os.path.isabs(os.readlink(q / "zeroshot"))
    else:
        with pytest.raises(fq.FtqError, match="never copied"):
            fl.cmd_link(argparse.Namespace(real_scores=str(real), q_scores=str(q), models="zeroshot", allow_copy=False))
        notes = fl.cmd_link(argparse.Namespace(real_scores=str(real), q_scores=str(q), models="zeroshot", allow_copy=True))
        assert len(notes) == 1 and notes[0].startswith("[copy]")
    for bad in (real, real / "sub", tmp_path / "grid" / "scores"):                       # never the registered scores directory
        with pytest.raises(fq.FtqError, match="root of its own") as raised:
            fl.cmd_link(argparse.Namespace(real_scores=str(real), q_scores=str(bad), models="zeroshot", allow_copy=False))
        assert "the FT-L root is a root of its own" in str(raised.value) and "FT-Q" not in str(raised.value)   # this is the FT-L root
    with pytest.raises(fq.FtqError, match="missing the real like pass"):
        fl.cmd_link(argparse.Namespace(real_scores=str(tmp_path / "nowhere"), q_scores=str(tmp_path / "q3"), models="zeroshot",
                                       allow_copy=False))
    assert fl.main(["link", "--real_scores", str(tmp_path / "nowhere"), "--q_scores", str(tmp_path / "q3")]) == 1
    assert fl.main(["link", "--q_scores", str(tmp_path / "q3")]) == 2 and not (tmp_path / "q3").exists()


def test_the_ftlen_record_gate_needs_the_sha1_of_the_three_files_in_the_pilot_log(tmp_path, capsys):
    (tmp_path / "a.sh").write_text("script a\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("module b\n", encoding="utf-8")
    log = tmp_path / "PILOT_LOG.md"
    log.write_text("# log\n", encoding="utf-8")
    base = ["record", "--pilot_log", str(log), "--files", "a.sh", "b.py", "--root", str(tmp_path)]
    assert fl.main(base) == 4 and "FT-L record" in capsys.readouterr().err                # not recorded: refused (exit 4)
    assert fl.main(base + ["--print"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"a.sh = {sha1_file(tmp_path / 'a.sh')}", f"b.py = {sha1_file(tmp_path / 'b.py')}"]
    log.write_text("# log\n" + lines[0].upper() + "\n" + lines[1] + "\n", encoding="utf-8")   # case-insensitive, as freeze
    assert fl.main(base) == 0 and "FT-L record OK" in capsys.readouterr().out
    (tmp_path / "b.py").write_text("module b, changed after the record\n", encoding="utf-8")
    assert fl.main(base) == 4 and "b.py" in capsys.readouterr().err                        # a changed file needs a new record
    assert fl.main(base + ["--append"]) == 0 and "[dry] recorded b.py" in capsys.readouterr().out
    log.write_bytes(b"# log without a final newline")
    assert fl.main(base + ["--append"]) == 0 and len(log.read_text(encoding="utf-8").splitlines()) == 3
    assert fl.main(["record", "--pilot_log", str(tmp_path / "none.md"), "--files", "a.sh", "--root", str(tmp_path)]) == 4
    assert fl.main(["record", "--files", "a.sh"]) == 2
    assert fq.record_lines(list(LEN_LINES)) == [f"scripts/sigir/run_ftlen.sh = {sha1_file(SCRIPT)}",
                                                f"src/confrec/ftlen_panel.py = {sha1_file(PANEL)}",
                                                f"tests/test_confrec_ftlen.py = {sha1_file(__file__)}"]
    assert "LEN_FILES=(scripts/sigir/run_ftlen.sh src/confrec/ftlen_panel.py tests/test_confrec_ftlen.py)" in SCRIPT.read_text(
        encoding="utf-8")


def test_info_reads_the_decisions_the_gates_use_and_the_command_line_is_one_flat_parser(tmp_path, capsys):
    sel, split, gate = tmp_path / "selection.json", tmp_path / "ftgrid_split.json", tmp_path / "gate_ft.json"
    sel.write_text(json.dumps({"decision": "FIX_FOUND", "gate_ft_prompt": "V1"}), encoding="utf-8")
    base = ["info", "--selection", str(sel), "--split", str(split), "--gate", str(gate)]
    assert fl.main(base) == 0 and capsys.readouterr().out.splitlines() == ["FIX_FOUND", "V1", "", "missing"]
    split.write_text(json.dumps({"variant": "V1"}), encoding="utf-8")
    gate.write_text(json.dumps({"decision": "GATE_FT_PASS"}), encoding="utf-8")
    assert fl.main(base) == 0 and capsys.readouterr().out.splitlines() == ["FIX_FOUND", "V1", "V1", "GATE_FT_PASS"]
    assert fl.main(["info", "--selection", str(tmp_path / "gone.json"), "--split", str(split), "--gate", str(gate)]) == 1
    assert set(fl.COMMANDS) == {"recipe", "scoring", "verify_adapter", "verify_scores", "link", "record", "info"}
    for cmd in fl.COMMANDS:
        assert fl.main([cmd]) == 2 and "needs" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        fl.main(["nonsense"])
    p = subprocess.run([sys.executable, "-m", "src.confrec.ftlen_panel", "record", "--files", "x"], cwd=ROOT, capture_output=True,
                       text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert p.returncode == 2 and "needs --pilot_log" in p.stderr
    code = ("import sys; import src.confrec.ftlen_panel; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'transformers', 'vllm', 'peft', 'numpy')]; assert not bad, bad")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stderr                                                    # the module needs no torch, not even numpy


# ================================================================ 4. the script file, its flags and its registered text
def test_script_is_lf_bash_and_bash_n_clean_and_the_module_is_lf():
    raw = SCRIPT.read_bytes()
    assert b"\r\n" not in raw and raw.startswith(b"#!/usr/bin/env bash\n") and b"set -euo pipefail" in raw
    assert b"\r\n" not in PANEL.read_bytes() and b"\r\n" not in Path(__file__).read_bytes()
    if FR.BASH is None:
        pytest.skip(FR.NO_BASH)
    r = subprocess.run([FR.BASH, "-n", SCRIPT.as_posix()], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_every_flag_the_script_passes_exists_in_the_real_argparse(trainer_parser):
    """The stricter audit of test_confrec_ftgrid_run (heredocs ignored, arrays expanded, literal values checked against choices and
    types), with the RECIPE and SARGS flags filled in from ftlen_panel's own output, so the flags taken from the recorded configs are
    audited too (the trainer's with --epochs 3)."""
    text = SCRIPT.read_text(encoding="utf-8")
    cfg = {"dtype": "float16", "topk_logprobs": 50, "max_model_len": 4096, "chunk_users": 100, "variant": FQT.VARIANT, "readout": "yesno",
           "questions": ["like"]}
    arrays = (f"\nRECIPE=({' '.join(fl.len_flags(real_base(trainer_parser, 'toys')))})\nSARGS=({' '.join(fq.score_flags(cfg))})\n")
    problems, count = FR.audit(text + arrays)
    assert not problems, "\n".join(problems)
    for target, n in {"src.confrec.pyes_scorer": 1, "src.confrec.train_lora_yesno": 1, "src.confrec.ftgrid_freeze": 2,
                      "src.confrec.ftgrid_report": 1, "src.confrec.ftlen_panel": 9}.items():
        assert count.get(target, 0) >= n, (target, count)
    bad = text.replace("--seeds", "--seedz").replace("--train_file", "--trainfile").replace("--scores_root", "--scoresroot")
    problems, _ = FR.audit(bad + arrays)
    assert any("--seedz" in p for p in problems) and any("--scoresroot" in p for p in problems)


def test_the_script_never_names_an_output_below_the_registered_grid_root():
    """Nothing is written below the grid root: no output flag, redirect, rm or mv aims at $GRID, $P, $RS, $RA or $REG, and the FT-L
    root is the only root a run writes."""
    text = re.sub(r"# ---- DRY_RUN: run_ftgrid.*?# ---- the prompt", "# ---- the prompt", SCRIPT.read_text(encoding="utf-8"), flags=re.S)
    code = re.sub(r"\\\n\s*", " ", "\n".join(x.split("#", 1)[0] for x in text.splitlines()))
    for pat in (r'--out_dir "\$(P|RS|RA|GRID|REG)', r'--output "\$(P|RS|RA|GRID|REG)', r'--out "\$(P|RS|RA|GRID|REG)',
                r'> *"\$(P|RS|RA|GRID|REG|REGREP)', r'(?<![\w.-])(rm|mv|cp|touch|mkdir)\s[^;|&\n]*"\$(P|RS|RA|GRID|REG|REGREP)[/"]', r'ln -s', r'git '):
        assert not re.search(pat, code), pat
    for needle in ('REGL=outputs/confrec/ftgrid_len3 ', 'QA="$OUT_ROOT/adapters/$D"', 'QS="$OUT_ROOT/scores/$D"',
                   'REP="$OUT_ROOT/report/$D.json"', '--scores_root "$OUT_ROOT/scores"', "--models zeroshot,s0,s1", "ftlen_panel link",
                   'SWAP_K=8', '--swap_k "$SWAP_K"', '"$P/eval_sd_test.jsonl" "$QS/s$seed/swap"'):
        assert needle in code, needle
    # the epochs and every other recipe flag come from the recorded configs (RECIPE), never from a literal in the training call; no
    # other arm exists in the code
    train_call = [ln for ln in code.splitlines() if "-m src.confrec.train_lora_yesno" in ln]
    assert len(train_call) == 1 and '"${RECIPE[@]}"' in train_call[0] and "--epochs" not in train_call[0], train_call
    score_calls = [ln for ln in code.splitlines() if "-m src.confrec.pyes_scorer" in ln]
    assert len(score_calls) == 1 and '"${SARGS[@]}"' in score_calls[0] and "--epochs" not in score_calls[0], score_calls
    for word in ("nohist", "starperm", "pseudo", "knockout", "--hist_len", "--epochs 3 --"):
        assert word not in code, word


def test_the_helpers_mirrored_from_run_ftgrid_are_its_text_and_the_e1_rule_is_the_same_but_for_one_line():
    mine = SCRIPT.read_text(encoding="utf-8")
    theirs = (ROOT / "scripts" / "sigir" / "run_ftgrid.sh").read_text(encoding="utf-8")
    for name in ("fresh", "step", "adapter_done", "e1_ok"):
        assert shell_function(mine, name) == shell_function(theirs, name), name
    needles = ('key="$(', 'wsha=$(', "run.key", "$dir.stale", "$dir.e1fail", "FAILED_INTEGRITY", 'e1_ok "$dir"',
               'score "$data" "$dir" "$@"')
    keep = lambda text: [ln for ln in shell_function(text, "score").splitlines()
                         if any(n in ln for n in needles) and "compgen" not in ln and "e1_failed_before" not in ln]
    assert keep(mine) == keep(theirs) and len(keep(mine)) >= 8
    assert 'if compgen -G "$dir.e1fail.*" > /dev/null; then' in shell_function(theirs, "score")        # the deliberate deviation
    assert 'if e1_failed_before "$dir" "$key"; then' in shell_function(mine, "score") and "compgen" not in shell_function(mine, "score")
    helper = shell_function(mine, "e1_failed_before")
    assert 'for d in "$1".e1fail.*; do' in helper and '[ "$(cat "$d/run.key")" = "$2" ]' in helper
    for line in ('if [ "$SEL_DECISION" = FIX_FOUND ] && [ ! -f "$G/confirm/gate.json" ]; then', 'SEL="$G/dev/selection.json"',
                 'VARIANT="${VARIANT:-$SEL_PROMPT}"'):
        assert line in theirs and line in mine, line
    # the registered scoring calls of run_ftgrid.sh: like on eval.jsonl, swap with --swap_k 8 on eval_sd_test.jsonl
    assert 'score "$P/eval_sd_test.jsonl" "$S/$m/swap" "${LORA[@]}" --swap_k 8' in theirs
    assert 'score "$P/eval_sd_test.jsonl" "$QS/s$seed/swap" --lora "$QA/s$seed" --swap_k "$SWAP_K"' in mine and "SWAP_K=8" in mine


def test_the_registered_text_of_addendum_9_is_what_the_script_and_the_module_implement():
    a9 = re.sub(r"\s+", " ", (ROOT / "idea-stage" / "PREREG_AMENDMENT_3_ADDENDUM_9.md").read_text(encoding="utf-8"))
    for needle in ("`--epochs 3`", "outputs/confrec/ftgrid_len3/", "Seeds 0 and 1", "named s0 and s1", "ML-1M and Toys", "the same `like` question",
                   "swap-prior arm on S_d's TEST rows", "are read from the real adapters' recorded configuration", "linked, never rescored",
                   "No other arm (no-history, star permutation, knockout)", "src/confrec/ftlen_plumbing.py", "scripts/sigir/run_ftlen.sh",
                   "recorded in PILOT_LOG before the first adapter is trained", "no bound file changes", "ML-1M about 5.6, Toys about 7.7",
                   "two seeds, exploratory", "G_wu, dUAUC(L - q-hat) and the e-share"):
        assert needle in a9, needle
    assert fl.EPOCHS == 3 and fl.SEEDS == (0, 1) and fl.SWAP_K == 8 and fl.ARMS == ("like", "swap") and fl.DOMAINS == ("ml1m", "toys")
    assert fl.LINKED_MODELS == ("zeroshot",) and fl.parse_args(["link"]).models == "zeroshot"          # only the zero-shot scores are linked
    assert fl.ARM_DATA == {"like": "eval.jsonl", "swap": "eval_sd_test.jsonl"}
    code = SCRIPT.read_text(encoding="utf-8")
    assert code.count("for seed in 0 1;") >= 3 and "ml1m|toys) ;;" in code and "Qwen3-8B" in code
    head = re.sub(r"\s+", " ", re.sub(r"(?m)^\s*#", "", code.split("set -euo pipefail")[0]))
    # the header: the record, what is used as it is, the sticky rule, the sweep, the allow-list, ftgrid_extra's command and label
    assert "the sha1 of this script, of src/confrec/ftlen_panel.py and of tests/test_confrec_ftlen.py are in docs/sigir/PILOT_LOG.md" in head
    assert "Imported or used as they are, and recorded separately (not part of that record): src/confrec/ftq_panel.py" in head
    assert "FAILED_INTEGRITY is sticky per seed" in head and "FTLEN_ALLOW_RETRY" in head and "NOTE_FTLEN_overrides.txt" in head
    assert "find OUT_ROOT -type l" in head and "report/D.json.tmp" in head and "like/*.tmp" in head and "are not enumerated" in head
    assert "The DRY_RUN guard is an allow-list" in head and "A relative OUT_ROOT resolves from the repo root" in head
    assert "every comparison is made on lower-case canonical forms" in head
    assert "ftgrid_extra is NOT run by this script" in head and "--root_label llama" in head and "--models zeroshot,s0,s1" in head
    assert "ML-1M 2 x 3 x 46 min" in head and "Toys 2 x 3 x 48 min" in head


def test_the_ftlen_files_are_new_files_outside_every_freeze_block():
    """No bound file changes: the files of this task are not in any FREEZE_FILES block of Amendment 3 or its addenda, and the bound
    files it imports or mirrors are."""
    ff = FR.ff
    texts = [ROOT / ff.AMENDMENT] + sorted((ROOT / "idea-stage").glob("PREREG_AMENDMENT_3_ADDENDUM_*.md"))
    listed = {f for p in texts for files in ff.parse_blocks(p.read_text(encoding="utf-8")).values() for f in files}
    assert listed and not listed & {"scripts/sigir/run_ftlen.sh", "src/confrec/ftlen_panel.py", "tests/test_confrec_ftlen.py"}
    assert {"src/confrec/ftgrid_data.py", "scripts/sigir/run_ftgrid.sh", "src/confrec/pyes_scorer.py",
            "src/confrec/train_lora_yesno.py"} <= listed


# ================================================================ 5. the input guards and the output roots
LEN_LINES = ("scripts/sigir/run_ftlen.sh", "src/confrec/ftlen_panel.py", "tests/test_confrec_ftlen.py")   # the record the script needs
DRYB = "tmp_outputs/ftlen_dryrun"                     # a DRY_RUN's temporary directory: outside outputs/confrec
GRID = f"{DRYB}/ftgrid"                               # run_ftgrid.sh's DRY_RUN root: the world the script reads
LROOT_DRY = f"{DRYB}/ftgrid_len3"                     # the FT-L root of a DRY_RUN
WORLD_DIRS = (GRID,)
SENTINEL = b'{"registered": "report", "never": "touched"}\n'
LLAMA_PATH = FQT.LLAMA_PATH


@pytest.fixture(scope="module")
def guard_repo(tmp_path_factory):
    if FR.BASH is None:
        pytest.skip(FR.NO_BASH)
    return make_ftlen_repo(tmp_path_factory.mktemp("ftlen_guards") / "repo")


def make_ftlen_repo(dest: Path) -> Path:
    """The repo parts the chain runs (run_ftgrid.sh's copy with the real ftgrid_report) plus this script and its test file."""
    FR.make_repo(dest, real_f2=True)
    shutil.copy2(SCRIPT, dest / "scripts" / "sigir" / "run_ftlen.sh")
    (dest / "tests").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), dest / "tests" / "test_confrec_ftlen.py")           # the third file of the FT-L record
    return dest


def run_ftlen(repo: Path, *args, dry: bool = True, **env) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in (
        "MODEL", "OUT_ROOT", "VARIANT", "STAGES", "DRY_RUN", "DRY_GATE", "DRY_E1_FAIL", "DRY_NO_RECORD", "DRY_NO_WORLD",
        "FTLEN_ALLOW_RETRY", "FTLEN_RETRY_REASON", "PYTHONPATH")}
    e.update(PYTHON=sys.executable.replace("\\", "/"), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if dry:
        e["DRY_RUN"] = "1"
    e.update(env)
    script = (repo / "scripts" / "sigir" / "run_ftlen.sh").as_posix()
    return subprocess.run([FR.BASH, script, *args], env=e, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=3600)


@needs_bash
@pytest.mark.parametrize("args, dry, env, rc, msg", [
    (["beauty"], True, {}, 2, "usage"), (["games"], True, {}, 2, "usage"), ([], True, {}, 2, "usage"),
    (["toys"], True, {"STAGES": "0,7"}, 2, "unknown stage"), (["toys"], True, {"STAGES": "4"}, 2, "unknown stage"),
    (["toys"], True, {"OUT_ROOT": "outputs/confrec/ftgrid_len3"}, 2, "never writes to a registered output root"),
    (["toys"], True, {"OUT_ROOT": "./outputs/confrec/ftgrid/"}, 2, "never writes to a registered output root"),
    (["toys"], True, {"MODEL": LLAMA_PATH}, 2, "is not Qwen3-8B"), (["toys"], False, {"MODEL": LLAMA_PATH}, 2, "is not Qwen3-8B"),
    (["toys"], False, {"MODEL": "/models/Qwen3-8B", "OUT_ROOT": "outputs/confrec/ftgrid_q"}, 2, "one registered root"),
    (["toys"], False, {"MODEL": "/models/Qwen3-8B", "OUT_ROOT": "outputs/confrec/ftgrid"}, 2, "one registered root"),
    (["ml1m"], False, {"MODEL": "/models/Qwen3-8B"}, 1, "missing outputs/confrec/gatefix/dev/selection.json"),
    (["toys"], False, {"MODEL": "/models/Qwen3-8B", "DRY_RUN": "0"}, 1, "missing outputs/confrec/gatefix/dev/selection.json"),
    (["toys"], False, {"DRY_RUN": "yes"}, 2, "DRY_RUN must be 0 or 1"), (["toys"], False, {"DRY_RUN": ""}, 2, "DRY_RUN must be 0 or 1"),
    (["toys"], False, {"DRY_RUN": " 1"}, 2, "DRY_RUN must be 0 or 1"), (["toys"], False, {"DRY_RUN": "true"}, 2, "DRY_RUN must be 0 or 1"),
    (["toys"], True, {"DRY_NO_RECORD": "yes"}, 2, "DRY_NO_RECORD must be 0 or 1"),
    (["toys"], True, {"DRY_NO_WORLD": "true"}, 2, "DRY_NO_WORLD must be 0 or 1"),
    # the override of the sticky FAILED_INTEGRITY names a seed and needs a reason, before anything starts
    (["toys"], True, {"FTLEN_ALLOW_RETRY": "s0"}, 2, "needs a non-empty FTLEN_RETRY_REASON"),
    (["toys"], True, {"FTLEN_ALLOW_RETRY": "s0", "FTLEN_RETRY_REASON": ""}, 2, "needs a non-empty FTLEN_RETRY_REASON"),
    (["toys"], True, {"FTLEN_ALLOW_RETRY": "s2", "FTLEN_RETRY_REASON": "why"}, 2, "FTLEN_ALLOW_RETRY must be s0, s1 or s0,s1"),
    (["toys"], True, {"FTLEN_ALLOW_RETRY": "p0", "FTLEN_RETRY_REASON": "why"}, 2, "FTLEN_ALLOW_RETRY must be s0, s1 or s0,s1"),
    # ... and the accepted forms pass that validation (the run then stops at the unknown stage, before anything is built)
    (["toys"], True, {"FTLEN_ALLOW_RETRY": "s0,s1", "FTLEN_RETRY_REASON": "why", "STAGES": "9"}, 2, "unknown stage"),
    (["toys"], True, {"FTLEN_ALLOW_RETRY": "s1,s0", "FTLEN_RETRY_REASON": "why", "STAGES": "9"}, 2, "unknown stage"),
    (["toys"], True, {"FTLEN_ALLOW_RETRY": "s1", "FTLEN_RETRY_REASON": "why", "STAGES": "9"}, 2, "unknown stage")])
def test_input_guards_refuse_before_anything_is_written(guard_repo, args, dry, env, rc, msg):
    r = run_ftlen(guard_repo, *args, dry=dry, **env)
    assert r.returncode == rc and msg in r.stderr, FR.tail(r)
    assert not (guard_repo / "outputs").exists() and not (guard_repo / "tmp_outputs").exists()


REGISTERED_FILES = {                                  # what a registered tree holds: a refused run must leave all of it as it is
    "outputs/confrec/ftgrid/report/toys.json": b"registered report\n",
    "outputs/confrec/ftgrid/adapters/toys/s0/train_config.json": b"{}\n",
    "outputs/confrec/ftgrid/scores/toys/s0/like/report.json": b"{}\n",
    "outputs/confrec/ftgrid/scores/toys/zeroshot/like/report.json": b"{}\n",
    "outputs/confrec/ftgrid/panels/toys/train.jsonl": b"rows\n",
    "outputs/confrec/ftgrid_q/report/toys.json": b"registered FT-Q report\n",
    "outputs/confrec/ftgrid_len3/report/toys.json": b"the FT-L report\n",
    "outputs/confrec/gateft/gate_ft.json": b"{}\n",
    "outputs/confrec/gatefix/dev/note.txt": b"gate fix\n"}


def registered_repo(dest: Path) -> Path:
    """A scratch repo whose registered roots (the grid, the FT-Q root, the FT-L root, Gate-FT, the gate fix) hold files."""
    repo = make_ftlen_repo(dest)
    for rel, data in REGISTERED_FILES.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(data)
    return repo


def registered_state(repo: Path) -> dict:
    """{path: (size, mtime_ns, sha1)} of everything below the registered roots other than the FT-L root, names included."""
    out = {}
    for rel in ("ftgrid", "ftgrid_q", "gateft", "gatefix"):
        base = repo / "outputs" / "confrec" / rel
        for p in sorted(base.rglob("*")) if base.exists() else []:
            out[p.relative_to(repo).as_posix()] = (p.stat().st_size, p.stat().st_mtime_ns, sha1_file(p)) if p.is_file() else "dir"
    return out


def run_many(repo: Path, cases: list, *, dry: bool, **env) -> list:
    """[(OUT_ROOT spelling, result)], the script runs concurrently (every run is dominated by process starts)."""
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        return list(ex.map(lambda oc: (oc, run_ftlen(repo, "toys", dry=dry, OUT_ROOT=oc, **env)), cases))


def spellings(repo: Path, rel: str) -> list:
    """The spellings of a path below the repo that a string comparison does not equate with `rel` (// . .. absolute, native)."""
    absolute = (repo / rel).as_posix()
    head, _, tail = rel.rpartition("/")
    out = [rel + "//", rel.replace("/", "//", 1), head + "/./" + tail, "tmp_outputs/../" + rel, absolute]
    if os.name == "nt":                                # D:\... spellings and a case-insensitive filesystem
        out += [absolute.replace("/", "\\"), absolute.upper(), rel.upper()]
    return out


@needs_bash
def test_a_dry_run_never_writes_anywhere_in_the_repo_or_beside_it_and_a_real_run_takes_the_ftlen_root_only(tmp_path):
    """The allow-list and the canonical forms (every spelling): a DRY_RUN root is under tmp_outputs or outside the repo's parent
    directory (never outputs/confrec/ftgrid, ftgrid_q, ftgrid_len3, outputs/summary, data/raw, src, ../outside_repo, ...); a real run
    takes outputs/confrec/ftgrid_len3 in any spelling and nothing else. Nothing is written by a refusal."""
    repo = registered_repo(tmp_path / "repo")
    before = registered_state(repo)
    away = Path(tempfile.mkdtemp(prefix="ftlen_dry_away_"))                  # outside the repo and outside its parent directory
    try:
        dry_refused = (spellings(repo, "outputs/confrec/ftgrid") + spellings(repo, "outputs/confrec/ftgrid_len3") + [
            "outputs/confrec/ftgrid_q//", "outputs/confrec/gateft", "outputs/confrec/ftgrid/sub", "outputs/confrec", "outputs", ".", "/",
            "outputs/summary", "outputs/baselines/x", "data/raw", "src", "scripts", "docs", "tests", "idea-stage", "Paper",
            "../outside_repo", "..", "tmp_outputs/../src"])
        dry_accepted = [DRYB + "/ftgrid_len3", DRYB + "/ftgrid_len3//", "./" + DRYB + "/x", "tmp_outputs//ftlen_dryrun/./y",
                        (away / "ftlen root").as_posix(), (away / "ftlen root").as_posix() + "/"]
        real = {"MODEL": "/models/Qwen3-8B"}
        real_refused = spellings(repo, "outputs/confrec/ftgrid") + [
            "outputs/confrec/ftgrid", "outputs/confrec/ftgrid_q", "outputs/confrec/ftgrid_len3/sub", "outputs/confrec/ftgrid_len3/..",
            "outputs/confrec/ftgrid/../ftgrid", DRYB + "/ftgrid_len3", "tmp_outputs/x", "."]
        real_accepted = spellings(repo, "outputs/confrec/ftgrid_len3") + [
            "outputs/confrec/ftgrid_len3", "./outputs/confrec/ftgrid_len3", "outputs/confrec/./ftgrid_len3/",
            "outputs/confrec/ftgrid/../ftgrid_len3"]
        bad = [(f"DRY {oc}", FR.tail(r)) for oc, r in run_many(repo, dry_refused, dry=True)
               if r.returncode != 2 or not ("never writes to a registered output root" in r.stderr or "filesystem root" in r.stderr)]
        bad += [(f"DRY accepted {oc}", FR.tail(r)) for oc, r in run_many(repo, dry_accepted, dry=True, STAGES="9")
                if r.returncode != 2 or "unknown stage" not in r.stderr]
        bad += [(f"real {oc}", FR.tail(r)) for oc, r in run_many(repo, real_refused, dry=False, **real)
                if r.returncode != 2 or "one registered root" not in r.stderr]
        bad += [(f"real accepted {oc}", FR.tail(r)) for oc, r in run_many(repo, real_accepted, dry=False, **real)
                if r.returncode != 1 or "missing outputs/confrec/gatefix/dev/selection.json" not in r.stderr]
        assert not bad, "\n\n".join(f"== {name}\n{tail}" for name, tail in bad)
        assert registered_state(repo) == before and not (repo / "tmp_outputs").exists() and not (away / "ftlen root").exists()
        assert not (repo / "adapters").exists() and (repo / "outputs/confrec/ftgrid/report/toys.json").read_bytes() == b"registered report\n"
    finally:
        shutil.rmtree(away, ignore_errors=True)


@needs_bash
def test_a_link_between_the_registered_roots_and_the_files_a_run_writes_is_refused(tmp_path):
    """Planted links (symlinks, or junctions on a Windows box): an alias of the registered grid as a DRY root, the world's own path
    leading into the grid, a link below the DRY root, a link inside the world, the registered FT-L root itself a link to the grid, the
    whole outputs tree reached through a link, and a link below the FT-L root that would carry the adapters or the report into the
    grid. Every one is refused with exit 2 before anything is written, and the registered tree is left as it was."""
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
            shutil.rmtree(link)
        assert link_dir(link, repo / target), rel

    def refused(repo: Path, name: str, r, msg: str) -> list:
        ok = r.returncode == 2 and msg in r.stderr and registered_state(repo) == state
        return [] if ok else [(name, FR.tail(r) + "\nregistered tree unchanged: " + str(registered_state(repo) == state))]

    def dry_cases() -> list:
        repo, bad = clone_of("dry"), []
        plant(repo, "tmp_outputs/alias", grid)
        bad += refused(repo, "alias", run_ftlen(repo, "toys", OUT_ROOT="tmp_outputs/alias/sub"), "never writes to a registered")
        unlink_dir(repo / "tmp_outputs" / "alias")
        plant(repo, DRYB + "/ftgrid", grid)
        bad += refused(repo, "world", run_ftlen(repo, "toys"), "never writes to a registered")
        unlink_dir(repo / DRYB / "ftgrid")
        plant(repo, DRYB + "/ftgrid_len3/scores", grid + "/scores")
        r = run_ftlen(repo, "toys")
        bad += refused(repo, "dry child", r, "resolves to") + ([] if not (repo / DRYB / "ftgrid").exists() else [("dry child", "built")])
        unlink_dir(repo / DRYB / "ftgrid_len3" / "scores")
        plant(repo, DRYB + "/ftgrid/panels", grid + "/panels")
        bad += refused(repo, "world child", run_ftlen(repo, "toys"), "never writes to a registered")
        unlink_dir(repo / DRYB / "ftgrid" / "panels")
        return bad

    def real_cases() -> list:
        repo, bad = clone_of("real"), []
        shutil.rmtree(repo / "outputs/confrec/ftgrid_len3")
        plant(repo, "outputs/confrec/ftgrid_len3", grid)                      # the registered FT-L root is a link to the grid
        bad += refused(repo, "registered root", run_ftlen(repo, "toys", dry=False, **real), "one registered root")
        unlink_dir(repo / "outputs/confrec/ftgrid_len3")
        (repo / "outputs/confrec/ftgrid_len3").mkdir()
        for child in ("adapters", "report"):                                  # a link below the registered FT-L root
            plant(repo, "outputs/confrec/ftgrid_len3/" + child, grid + "/adapters")
            bad += refused(repo, "child " + child, run_ftlen(repo, "toys", dry=False, **real), "redirect")
            unlink_dir(repo / "outputs/confrec/ftgrid_len3" / child)
        return bad

    def ancestor_cases() -> list:
        repo = clone_of("ancestor")
        os.rename(repo / "outputs", repo / "real_outputs")                    # the whole outputs tree is reached through a link
        assert link_dir(repo / "outputs", repo / "real_outputs")
        out = refused(repo, "outputs is a link", run_ftlen(repo, "toys", dry=False, **real), "one registered root")
        unlink_dir(repo / "outputs")
        return out

    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        bad = [x for found in [f.result() for f in [ex.submit(fn) for fn in (dry_cases, real_cases, ancestor_cases)]] for x in found]
    assert not bad, "\n\n".join(f"== {name}\n{tail}" for name, tail in bad)


# ================================================================ 6. the DRY_RUN chains (the worlds are cached across sessions)
def world_key() -> str:
    """sha1 over the code that builds a DRY world (everything but ftlen_panel.py and the tests): a changed key rebuilds it."""
    h = hashlib.sha1(sys.version.encode())
    h.update("|".join(WORLD_DIRS).encode())
    files = [p for p in sorted((ROOT / "src" / "confrec").glob("*.py")) if p.name != "ftlen_panel.py"]
    files += [ROOT / "scripts" / "sigir" / n for n in ("run_ftgrid.sh", "starperm_panel.py")]
    files += [ROOT / FR.ff.AMENDMENT]
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    block = re.search(r"# ---- DRY_RUN: run_ftgrid.*?# ---- the prompt", SCRIPT.read_text(encoding="utf-8"), re.S)
    h.update(block.group(0).encode() if block else b"no DRY block")      # the script's own world-building block, nothing else of it
    return h.hexdigest()[:20]


@pytest.fixture(scope="session")
def world_cache(tmp_path_factory) -> Path:
    base = tmp_path_factory.getbasetemp().parent / "ftlen_world_cache"
    key = world_key()
    base.mkdir(parents=True, exist_ok=True)
    for old in base.iterdir():                          # one key at a time: a changed code base drops the old worlds
        if old.name != key:
            shutil.rmtree(old, ignore_errors=True)
    (base / key).mkdir(exist_ok=True)
    return base / key


def build_world(cache: Path, d: str) -> None:
    """The DRY world of dataset d (run_ftgrid.sh's stages 0-4), built by the script's own DRY block: STAGES=3 builds it and then
    stops at the stage-3 refusal (there are no passes of s0 and s1 yet); kept without the FT-L root."""
    tmp = Path(tempfile.mkdtemp(prefix=f"world_{d}_", dir=str(cache)))
    try:
        repo = make_ftlen_repo(tmp / "repo")
        r = run_ftlen(repo, d, STAGES="3")
        assert r.returncode == 1 and "(stage 2)" in r.stderr and (repo / GRID / "scores" / d / "s0" / "swap" / "report.json").is_file(), FR.tail(r)
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


def seed_world(repo: Path, worlds: Path, d: str) -> None:
    for rel in WORLD_DIRS:
        shutil.copytree(worlds / d / rel, repo / rel)


def snapshot_world(repo: Path) -> dict:
    """{path: (size, mtime_ns, sha1)} of the grid world; the temporary pilot logs are the one thing a run may append to (the record
    of the FT-L files)."""
    out = {}
    for rel in WORLD_DIRS:
        for p in sorted((repo / rel).rglob("*")):
            if p.is_file() and not p.name.startswith("PILOT_LOG"):
                st = p.stat()
                out[p.relative_to(repo).as_posix()] = (st.st_size, st.st_mtime_ns, sha1_file(p))
    return out


def make_chain(worlds: Path, tpf, d: str, rerun: bool) -> dict:
    repo = make_ftlen_repo(tpf.mktemp(f"ftlen_{d}") / "repo")
    seed_world(repo, worlds, d)
    reports = repo / GRID / "report"                      # a stand-in registered (one-epoch) report: never touched
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{d}.json").write_bytes(SENTINEL)
    (reports / f"{d}_tables.csv").write_bytes(b"model,arm\n")
    c = {"d": d, "repo": repo, "root": repo / LROOT_DRY, "grid": repo / GRID, "before": snapshot_world(repo)}
    c["r"] = run_ftlen(repo, d)
    assert c["r"].returncode == 0, FR.tail(c["r"])
    c["after"] = snapshot_world(repo)
    c["s1"] = FR.snapshot(c["root"])
    if rerun:
        c["r2"] = run_ftlen(repo, d)
        assert c["r2"].returncode == 0, FR.tail(c["r2"])
        c["s2"] = FR.snapshot(c["root"])
    return c


@pytest.fixture(scope="module", autouse=True)
def pipeline(world_cache, tmp_path_factory):
    """The two DRY chains and then the refusal scenarios start with the module and run in the background while the unit and guard
    tests run (a cold cache builds both worlds first, concurrently); the tests that need a result wait for it. None under
    FTLEN_FAST=1 or without bash."""
    if FR.BASH is None or FAST:
        yield None
        return
    ex = cf.ThreadPoolExecutor(max_workers=5)
    worlds = {d: ex.submit(ensure_world, world_cache, d) for d in ("toys", "ml1m")}
    chains = {d: ex.submit(run_chain, worlds[d], world_cache, tmp_path_factory, d) for d in ("toys", "ml1m")}
    scenarios = ex.submit(run_scenarios, chains["toys"], tmp_path_factory)
    yield {"toys": chains["toys"], "ml1m": chains["ml1m"], "scenarios": scenarios}
    ex.shutdown(wait=True, cancel_futures=True)


def ensure_world(cache: Path, d: str) -> None:
    if not (cache / d / ".complete").exists():
        build_world(cache, d)


def run_chain(world_future, cache: Path, tpf, d: str) -> dict:
    world_future.result()
    return make_chain(cache, tpf, d, d == "toys")


@pytest.fixture(scope="module")
def chain_toys(pipeline):
    if pipeline is None:
        pytest.skip("FTLEN_FAST=1 or no bash")
    return pipeline["toys"].result()


@pytest.fixture(scope="module")
def chain_ml1m(pipeline):
    if pipeline is None:
        pytest.skip("FTLEN_FAST=1 or no bash")
    return pipeline["ml1m"].result()


def real_dirs(c: dict) -> tuple:
    """(the real adapters' directory, the path their train_config.json records for TRAIN) of the chain's dataset."""
    if c["d"] == "ml1m":
        return f"{GRID}/_dry/gateft/adapters", f"{GRID}/_dry/gateft/train.jsonl"
    return f"{GRID}/adapters/{c['d']}", f"{GRID}/panels/{c['d']}/train.jsonl"


def check_chain(c: dict) -> None:
    d, repo, root, grid, out = c["d"], c["repo"], c["root"], c["grid"], c["r"].stdout
    ra, train_ref = real_dirs(c)
    # the FT-L root holds the adapters, the scores and the report; nothing else
    assert {p.name for p in (root / "adapters" / d).iterdir()} == {"s0", "s1"}
    assert {p.name for p in (root / "scores" / d).iterdir()} == {"zeroshot", "s0", "s1"}
    for m in ("s0", "s1"):
        assert {p.name for p in (root / "scores" / d / m).iterdir()} == {"like", "swap"}            # no other arm is scored
    assert {p.name for p in (root / "report").iterdir()} == {f"{d}.json", f"{d}_tables.csv", "NOTE_FTLEN.txt"}
    assert {p.name for p in root.iterdir()} == {"adapters", "scores", "report", "build"}
    # the registered grid world was not written (the stand-in registered report included)
    assert c["after"] == c["before"] and (grid / "report" / f"{d}.json").read_bytes() == SENTINEL
    # the adapters: the real adapters' recorded arguments except --out, --seed and --epochs (3); --train kept; provenance
    real_cfg = read_json(repo / ra / "s0" / "train_config.json")
    assert real_cfg["train"] == train_ref and real_cfg["epochs"] == 1.0
    assert ("[recipe] trainer flags recorded by the real s0-s2 (--out and --seed replaced, --epochs 3, --train kept): "
            + " ".join(fl.len_flags(real_cfg))) in out
    assert FR.trained(c["r"]) == ["s0", "s1"]                                           # the real adapters are never trained here
    assert out.index("[recipe] scorer flags recorded") < out.index("dry-run trainer: ")       # the pre-flight comes first
    train_sha1 = sha1_file(grid / "panels" / d / "train.jsonl")
    for k in (0, 1):
        a = root / "adapters" / d / f"s{k}"
        cfg = read_json(a / "train_config.json")
        assert cfg["epochs"] == 3.0 and cfg["train"] == train_ref and cfg["out"] == f"{LROOT_DRY}/adapters/{d}/s{k}" and cfg["seed"] == k
        assert {x: v for x, v in cfg.items() if x not in ("out", "seed", "epochs")} == {
            x: v for x, v in real_cfg.items() if x not in ("out", "seed", "epochs")}      # every other recorded argument and fact
        prov = read_json(a / "ftlen.json")
        assert (prov["adapter"], prov["epochs"], prov["seed"], prov["train_sha1"]) == (f"s{k}", 3, k, train_sha1)
    # the passes: the real s0's like and swap arguments, scored on the grid's eval.jsonl and eval_sd_test.jsonl
    for arm in fl.ARMS:
        s0_cfg = read_json(grid / "scores" / d / "s0" / arm / "report.json")["config"]
        s0_key = (grid / "scores" / d / "s0" / arm / "run.key").read_text(encoding="utf-8").split()
        panel = grid / "panels" / d / fl.ARM_DATA[arm]
        assert ("(--data, --output, --model, --lora replaced; swap adds --swap_k 8): " + " ".join(fq.score_flags(s0_cfg))) in out
        for k in (0, 1):
            lora = f"{LROOT_DRY}/adapters/{d}/s{k}"
            rep = read_json(root / "scores" / d / f"s{k}" / arm / "report.json")
            cfg = rep["config"]
            assert set(cfg) == set(s0_cfg) and {x for x in s0_cfg if s0_cfg[x] != cfg[x]} == {"lora"} and cfg["lora"] == lora
            assert cfg["swap_k"] == (8 if arm == "swap" else 0) and cfg["data_sha1"] == sha1_file(panel)
            key = (root / "scores" / d / f"s{k}" / arm / "run.key").read_text(encoding="utf-8").split()
            assert key[:3] == [sha1_file(panel), MODEL, VARIANT] == s0_key[:3]
            assert key[3] == fq.weights_sha1(root / "adapters" / d / f"s{k}") and key[4:] == ["--lora", lora] + (
                ["--swap_k", "8"] if arm == "swap" else [])
    # the registered zero-shot scores are linked (a symlink; a copy only where the platform has none), and the report reads them
    link, target = root / "scores" / d / "zeroshot", grid / "scores" / d / "zeroshot"
    if os.path.islink(link):
        assert os.path.realpath(link) == os.path.realpath(target) and not os.path.isabs(os.readlink(link))
    else:
        assert "[copy]" in out and (link / "like" / "report.json").read_bytes() == (target / "like" / "report.json").read_bytes()
    rep = read_json(root / "report" / f"{d}.json")
    assert rep["meta"]["models_requested"] == ["zeroshot", "s0", "s1"] and rep["meta"]["domain"] == d
    assert (rep["meta"]["n_boot"], rep["meta"]["seed"]) == (20, 0)                    # the DRY_RUN count (the real one is 2,000)
    assert (root / "report" / f"{d}_tables.csv").read_text(encoding="utf-8").startswith("domain,backbone,block")
    for m in ("zeroshot", "s0", "s1"):
        assert rep["runs"][m]["like"]["status"] == "OK" and rep["runs"][m]["swap"]["status"] == "OK", m
    assert not rep["excluded_runs"]
    assert rep["P1"]["decision"]["verdict"] == "INCOMPLETE"                            # two seeds: the three-seed decision is not made
    # the sidecar that says what the reports of this root are
    note = re.sub(r"\s+", " ", (root / "report" / "NOTE_FTLEN.txt").read_text(encoding="utf-8"))
    for phrase in ("EXPLORATORY: no hypothesis, no Holm family", "3-EPOCH adapters", "--epochs 3 and nothing else changed",
                   "They are not the registered adapters", "zeroshot here is a link to the registered zero-shot scores",
                   "outputs/confrec/ftgrid/report/<D>.json, is the ONE-EPOCH result, and the only registered one",
                   "INCOMPLETE or descriptive here, and no decision or Holm result of this file is registered", "two seeds, exploratory",
                   "ftgrid_extra is not run by this script"):
        assert phrase in note, phrase
    # the gates ran first and the FT-L record was written on the temporary pilot log
    marks = [out.index(x) for x in ("freeze check OK (stage core)", "== stage 1", "== stage 2", "== stage 3")]
    assert marks == sorted(marks) and "[dry] gate rehearsal" in out and "freeze check OK (stage amendment)" in out
    log = (grid / "_dry" / "PILOT_LOG.md").read_text(encoding="utf-8").lower()
    for rel in LEN_LINES:
        assert f"{rel} = {sha1_file(repo / rel)}" in log


@needs_chain
def test_toys_chain_trains_and_scores_three_epoch_adapters_in_its_own_root(chain_toys):
    check_chain(chain_toys)


@needs_chain
def test_ml1m_chain_takes_gate_fts_adapters_as_the_real_ones(chain_ml1m):
    check_chain(chain_ml1m)
    assert read_json(chain_ml1m["root"] / "report" / "ml1m.json")["P1"]["role"].startswith("P1 confirmatory")
    gt = chain_ml1m["repo"] / GRID / "_dry" / "gateft"
    assert read_json(chain_ml1m["repo"] / GRID / "scores" / "ml1m" / "s0" / "swap" / "report.json")["config"]["lora"] == \
        f"{GRID}/_dry/gateft/adapters/s0" and (gt / "adapters" / "s0" / "train_config.json").is_file()


@needs_chain
def test_toys_second_run_skips_everything_and_touches_nothing(chain_toys):
    c = chain_toys
    assert c["s1"] == c["s2"]
    out = c["r2"].stdout
    assert "dry-run trainer" not in out and "scores chunk" not in out and out.count(": scored") == 4
    assert out.count("[skip] adapter") == 2 and "[skip] run_ftgrid.sh's DRY_RUN world for toys" in out
    assert f"[skip] {LROOT_DRY}/report/toys.json" in out and "[dry] gate rehearsal" not in out   # the rehearsal runs once per world


# ---------------------------------------------------------------- the refusals, run concurrently on clones of the toys chain
def clone(c: dict, tmp: Path) -> Path:
    dest = tmp / "repo"
    shutil.copytree(c["repo"], dest)
    return dest


def forbid_training(r) -> str:
    return "" if "dry-run trainer" not in r.stdout else "a stage-1 training started"


def scenario_gate_ft(c, repo):
    gate = repo / GRID / "_dry" / "gateft" / "gate_ft.json"
    gate.write_text(json.dumps({"decision": "GATE_FT_FAIL"}), encoding="utf-8")
    before, errs = FR.snapshot(repo / LROOT_DRY), []
    r = run_ftlen(repo, "toys")
    errs += [] if r.returncode == 4 and "Gate-FT decision GATE_FT_FAIL" in r.stderr and "== stage" not in r.stdout else [FR.tail(r)]
    gate.unlink()                                                              # no decision recorded at all
    r = run_ftlen(repo, "toys", STAGES="1")                                    # every stage is gated: training included
    errs += [] if r.returncode == 4 and "Gate-FT decision missing" in r.stderr and forbid_training(r) == "" else [FR.tail(r)]
    return errs + ([] if FR.snapshot(repo / LROOT_DRY) == before else ["a refused run wrote to the FT-L root"])


def scenario_records(c, repo):
    log = repo / GRID / "_dry" / "PILOT_LOG.md"
    original, errs, before = log.read_text(encoding="utf-8"), [], FR.snapshot(repo / LROOT_DRY)
    log.write_text("# emptied\n", encoding="utf-8")                              # (a) no record at all
    r = run_ftlen(repo, "toys", DRY_NO_RECORD="1")
    errs += [] if r.returncode == 4 and "stage amendment" in r.stderr else [FR.tail(r)]
    keep = [x for x in original.splitlines() if not x.startswith(LEN_LINES)]
    assert len(original.splitlines()) - len(keep) >= len(LEN_LINES)                  # (the world may hold older lines too)
    log.write_text("\n".join(keep) + "\n", encoding="utf-8")                     # (b) only the FT-L record is missing
    r = run_ftlen(repo, "toys", DRY_NO_RECORD="1")
    errs += [] if r.returncode == 4 and "FT-L record" in r.stderr and "stage core" not in r.stderr else [FR.tail(r)]
    log.write_text(original, encoding="utf-8")
    bound = repo / "src" / "confrec" / "metrics.py"                              # (c) a bound file changed since the record
    bound.write_text(bound.read_text(encoding="utf-8") + "\n# changed after the freeze check\n", encoding="utf-8")
    r = run_ftlen(repo, "toys")
    errs += [] if r.returncode == 4 and "stage core" in r.stderr else [FR.tail(r)]
    shutil.copy2(ROOT / "src" / "confrec" / "metrics.py", bound)
    paths = [repo / rel for rel in LEN_LINES]                                    # (d) the three FT-L files changed since their record:
    good = [p.read_bytes() for p in paths]                                       # every one of them is named by the refusal
    for p, g in zip(paths, good):
        p.write_bytes(g + b"\n# changed after its record\n")
    r = run_ftlen(repo, "toys", DRY_NO_RECORD="1")
    errs += [] if (r.returncode == 4 and "FT-L record" in r.stderr and all(rel in r.stderr for rel in LEN_LINES)
                   and "== stage" not in r.stdout) else [FR.tail(r)]
    for p, g in zip(paths, good):
        p.write_bytes(g)
    return errs + ([] if FR.snapshot(repo / LROOT_DRY) == before else ["a refused run wrote to the FT-L root"])


def scenario_missing_or_unregistered_real_adapters(c, repo):
    """No hour is spent on a missing real adapter, a "real" adapter that is not the registered 1-epoch one, or a real like or swap
    pass that is not the registered one (the other recorded-input checks, such as real adapters that do not share one recipe, are
    unit-tested on the module)."""
    root, errs = repo / LROOT_DRY, []
    for k in (0, 1):
        shutil.rmtree(root / "adapters" / "toys" / f"s{k}")                          # so that stage 1 would train
    cfg0 = repo / GRID / "adapters" / "toys" / "s0" / "train_config.json"
    good0 = cfg0.read_text(encoding="utf-8")
    cfg0.write_text(json.dumps({**json.loads(good0), "epochs": 3.0}), encoding="utf-8")        # the "real" adapters are 3-epoch ones
    for name in ("s1", "s2"):
        p = repo / GRID / "adapters" / "toys" / name / "train_config.json"
        p.write_text(json.dumps({**json.loads(p.read_text(encoding="utf-8")), "epochs": 3.0}), encoding="utf-8")
    r = run_ftlen(repo, "toys", STAGES="1")
    errs += [] if r.returncode == 1 and "registered recipe" in r.stderr and "epochs" in r.stderr and forbid_training(r) == "" else [FR.tail(r)]
    for name in ("s1", "s2"):
        p = repo / GRID / "adapters" / "toys" / name / "train_config.json"
        p.write_text(json.dumps({**json.loads(p.read_text(encoding="utf-8")), "epochs": 1.0}), encoding="utf-8")
    cfg0.write_text(good0, encoding="utf-8")
    for arm, key, value in (("like", "chunk_users", 50), ("swap", "swap_k", 4)):
        rep = repo / GRID / "scores" / "toys" / "s0" / arm / "report.json"          # the real pass is not the registered one
        good = rep.read_text(encoding="utf-8")
        data = json.loads(good)
        data["config"][key] = value
        rep.write_text(json.dumps(data), encoding="utf-8")
        r = run_ftlen(repo, "toys", STAGES="1")
        errs += [] if r.returncode == 1 and key in r.stderr and arm in r.stderr and forbid_training(r) == "" else [FR.tail(r)]
        rep.write_text(good, encoding="utf-8")
    shutil.rmtree(repo / GRID / "adapters" / "toys" / "s0")                          # a missing real adapter is named
    r = run_ftlen(repo, "toys", STAGES="1")
    errs += [] if r.returncode == 1 and "missing the real adapter s0" in r.stderr and forbid_training(r) == "" else [FR.tail(r)]
    return errs + ([] if not (root / "adapters" / "toys" / "s0").exists() else ["an adapter was created"])


def scenario_provenance(c, repo):
    """An adapter trained by hand (no ftlen.json), and a registered 1-epoch adapter placed in the root, are never taken for FT-L
    adapters: nothing is trained and nothing is scored."""
    root, errs = repo / LROOT_DRY, []
    prov = root / "adapters" / "toys" / "s0" / "ftlen.json"
    good = prov.read_text(encoding="utf-8")
    prov.unlink()                                                                  # trained by hand
    r = run_ftlen(repo, "toys", STAGES="1")
    errs += [] if r.returncode == 1 and "has no ftlen.json" in r.stderr and forbid_training(r) == "" else [FR.tail(r)]
    prov.write_text(good, encoding="utf-8")
    cfg = root / "adapters" / "toys" / "s1" / "train_config.json"                  # a registered 1-epoch adapter placed as s1
    cfg.write_text(json.dumps({**json.loads(cfg.read_text(encoding="utf-8")), "epochs": 1.0}), encoding="utf-8")
    before = FR.snapshot(root)
    r = run_ftlen(repo, "toys", STAGES="2")                                        # ... and it is not scored
    errs += [] if (r.returncode == 1 and "registered 1-epoch adapter" in r.stderr and "scores chunk" not in r.stdout
                   and FR.snapshot(root) == before) else [FR.tail(r)]
    return errs


def scenario_stage_order(c, repo):
    root, errs = repo / LROOT_DRY, []
    shutil.rmtree(root / "adapters" / "toys")                                      # stage 1 not done
    before = FR.snapshot(root)
    r = run_ftlen(repo, "toys", STAGES="2")
    errs += [] if r.returncode == 1 and "adapter s0 of toys is missing or incomplete" in r.stderr and "scores chunk" not in r.stdout else [FR.tail(r)]
    shutil.rmtree(root / "scores" / "toys" / "s1")                                 # stage 2 not done for s1
    before = FR.snapshot(root)
    r = run_ftlen(repo, "toys", STAGES="3")
    errs += [] if r.returncode == 1 and "missing the like pass" in r.stderr and "s1/like" in r.stderr else [FR.tail(r)]
    errs += [] if FR.snapshot(root) == before else ["a refused run wrote to the FT-L root"]
    os.remove(repo / GRID / "scores" / "toys" / "zeroshot" / "swap" / "report.json")   # the registered zero-shot swap pass is missing
    r = run_ftlen(repo, "toys", STAGES="3", DRY_NO_WORLD="1")                      # (a DRY_RUN would first rebuild the synthetic world)
    return errs + ([] if r.returncode == 1 and "missing the registered zero-shot swap pass" in r.stderr else [FR.tail(r)])


def scenario_e1(c, repo):
    """E1 per arm: a swap pass that fails E1 twice leaves FAILED_INTEGRITY for that arm only, is reported as missing and never
    replaced, and the report is still made (the registered report of the dataset is not touched)."""
    root, errs = repo / LROOT_DRY, []
    scores = root / "scores" / "toys"
    shutil.rmtree(scores / "s0" / "swap")
    r = run_ftlen(repo, "toys", STAGES="2,3", DRY_E1_FAIL="s0/swap")
    swap = scores / "s0" / "swap"
    errs += [] if r.returncode == 0 else [FR.tail(r)]
    errs += [] if (swap / "FAILED_INTEGRITY").is_file() and len(list(swap.parent.glob("swap.e1fail.*"))) == 1 else ["no FAILED_INTEGRITY"]
    errs += [] if r.stderr.count("[E1 failed]") == 1 and r.stderr.count("failed E1 twice") == 1 else ["E1 rerun-once messages"]
    errs += [] if r.stderr.count("NOTE: ") == 1 and "that pass of s0 is reported as missing, never replaced" in r.stderr else ["no closing note"]
    errs += [] if not (scores / "s0" / "like" / "FAILED_INTEGRITY").exists() and not (scores / "s1" / "swap" / "FAILED_INTEGRITY").exists() else ["other passes failed"]
    rep = read_json(root / "report" / "toys.json")
    errs += [] if [(e["model"], e["arm"], e["status"]) for e in rep["excluded_runs"]] == [("s0", "swap", "FAILED_INTEGRITY")] else ["swap not excluded"]
    errs += [] if rep["runs"]["s1"]["swap"]["status"] == "OK" and rep["runs"]["s0"]["like"]["status"] == "OK" else ["report"]
    errs += [] if (repo / GRID / "report" / "toys.json").read_bytes() == SENTINEL else ["the registered report changed"]
    return errs


def scenario_e1_leftover_and_same_key(c, repo):
    """The same-key refinement of the rerun-once rule, both sides in one run. s1/swap has a leftover swap.e1fail.<time> of ANOTHER
    run.key (an earlier panel, adapter or arguments): run_ftgrid.sh would count it and turn this TRANSIENT first failure into the
    final one; here the failure is rerun once and the rerun stands. s0/swap has an e1fail of the CURRENT run.key (an earlier failed
    attempt of the same inputs): the rerun is used up, so the failure of this run is final at once (no rerun)."""
    root, errs = repo / LROOT_DRY, []
    scores = root / "scores" / "toys"
    swap0, swap1 = scores / "s0" / "swap", scores / "s1" / "swap"
    old0, old1 = scores / "s0" / "swap.e1fail.20200101000000", scores / "s1" / "swap.e1fail.20200101000000"
    shutil.copytree(swap1, old1)
    (old1 / "run.key").write_text(f"{'0' * 40} {MODEL} {VARIANT} {'1' * 40} --lora elsewhere/s1 --swap_k 8\n", encoding="utf-8")
    shutil.copytree(swap0, old0)                                                   # run.key: the CURRENT key of s0's swap pass
    assert (old0 / "run.key").read_text(encoding="utf-8") == (swap0 / "run.key").read_text(encoding="utf-8")
    shutil.rmtree(swap0)
    shutil.rmtree(swap1)
    fakes = repo / GRID / "_dry" / "ftgrid_fakes.py"                               # test only: the stand-in fails the FIRST run of a
    text = fakes.read_text(encoding="utf-8")                                       # run.key, and always fails s0's passes
    helper = ("def _failed_before(output):\n"
              "    out = Path(str(output))\n"
              "    key = (out / 'run.key').read_text(encoding='utf-8') if (out / 'run.key').is_file() else None\n"
              "    return any((d / 'run.key').is_file() and (d / 'run.key').read_text(encoding='utf-8') == key\n"
              "               for d in out.parent.glob(out.name + '.e1fail.*'))\n\n\n")
    text = text.replace("def fake_model(args):", helper + "def fake_model(args):")
    text, n = re.subn(r"(mass = 0\.5 if fail and fail in str\(args\.output\)\.replace\(.*?\))( else 0\.99999)",
                      lambda m: m.group(1) + ' and (not _failed_before(args.output) or "/s0/" in str(args.output).replace(chr(92), "/"))'
                      + m.group(2), text)
    assert n == 1
    fakes.write_text(text, encoding="utf-8")
    r = run_ftlen(repo, "toys", STAGES="2", DRY_E1_FAIL="swap")        # (no leading slash: MSYS would turn it into a path)
    errs += [] if r.returncode == 0 and r.stderr.count("[E1 failed]") == 1 and r.stderr.count("failed E1 twice") == 1 else [FR.tail(r)]
    errs += [] if (swap1 / "report.json").is_file() and not (swap1 / "FAILED_INTEGRITY").exists() and old1.is_dir() \
        and len(list(swap1.parent.glob("swap.e1fail.*"))) == 2 else ["the rerun of s1 did not stand (a leftover counted)"]
    errs += [] if (swap0 / "FAILED_INTEGRITY").is_file() and old0.is_dir() and len(list(swap0.parent.glob("swap.e1fail.*"))) == 1 \
        else ["s0's failure was not final at once (the rerun of the same key was not used up)"]
    return errs


def scenario_temporary_files_of_the_bound_tools(c, repo):
    """ftgrid_report and the scorer write <name>.tmp beside their products. A planted temporary name that leads to a registered file
    (a hard link, or a symlink where those can be made) used to be written through; now a stale temporary file is removed before the
    tool runs and any link below the FT-L root other than scores/D/zeroshot is refused (exit 2) before anything is written."""
    root, errs = repo / LROOT_DRY, []
    reg_rep, reg_tab = repo / GRID / "report" / "toys.json", repo / GRID / "report" / "toys_tables.csv"
    s0_scores = repo / GRID / "scores" / "toys" / "s0" / "like" / "scores.csv.gz"
    before = {p: sha1_file(p) for p in (reg_rep, reg_tab, s0_scores)}
    for name in ("toys.json", "toys_tables.csv"):                                  # (a) the report is made again ...
        (root / "report" / name).unlink()
    os.link(reg_rep, root / "report" / "toys.json.tmp")                            # ... through temporary names that are the
    os.link(reg_tab, root / "report" / "toys_tables.csv.tmp")                      # registered report and tables (hard links)
    r = run_ftlen(repo, "toys", STAGES="3")
    errs += [] if r.returncode == 0 and (root / "report" / "toys.json").is_file() and not list((root / "report").glob("*.tmp")) else [FR.tail(r)]
    like = root / "scores" / "toys" / "s0" / "like"                                # (b) s0 is scored again in place (no report, the
    (like / "report.json").unlink()                                                # run.key is current) through a temporary name that
    os.link(s0_scores, like / "scores.csv.gz.tmp")                                 # is the real s0's score file
    r = run_ftlen(repo, "toys", STAGES="2")
    errs += [] if r.returncode == 0 and (like / "report.json").is_file() and not (like / "scores.csv.gz.tmp").exists() else [FR.tail(r)]
    errs += [] if {p: sha1_file(p) for p in before} == before else ["a registered file was written through a temporary name"]
    try:                                                                           # (c) the same as symlinks: refused
        os.symlink(reg_rep, root / "report" / "toys.json.tmp")
        made = True
    except (OSError, NotImplementedError):
        made = False                                                               # (no symlink privilege: the hard links stand in)
    if made:
        (root / "report" / "toys.json").unlink()
        r = run_ftlen(repo, "toys", STAGES="3")
        refusal = "is a link" in r.stderr or "redirect" in r.stderr
        errs += [] if r.returncode == 2 and refusal and sha1_file(reg_rep) == before[reg_rep] else [FR.tail(r)]
        (root / "report" / "toys.json.tmp").unlink()
    wrong = root / "scores" / "toys" / "zeroshot"                                  # (d) a link with the allowed name that leads to
    if wrong.is_symlink() or getattr(os.path, "isjunction", lambda p: False)(wrong):   # the wrong score directory is refused too
        unlink_dir(wrong)
    else:
        shutil.rmtree(wrong)
    if link_dir(wrong, repo / GRID / "scores" / "toys" / "s2"):
        r = run_ftlen(repo, "toys", STAGES="3")
        errs += [] if r.returncode == 2 and "not to the registered zero-shot score directory" in r.stderr else [FR.tail(r)]
    return errs


def scenario_sticky_stage_1(c, repo):
    """FAILED_INTEGRITY is sticky per seed, across both arms, in stage 1: while scores/D/s<k>/like or /swap holds it, making s<k> again
    is refused (exit 1, naming the directory) unless FTLEN_ALLOW_RETRY=s<k> and a non-empty FTLEN_RETRY_REASON are set; the seed must
    be the one named, the reason is logged (seed, UTC date, reason), nothing is written by a refusal."""
    root, errs = repo / LROOT_DRY, []
    (root / "scores" / "toys" / "s0" / "swap" / "FAILED_INTEGRITY").write_text("", encoding="utf-8")   # s0's swap pass failed E1 twice
    shutil.rmtree(root / "adapters" / "toys" / "s0")                                      # s0 would be made again
    before = FR.snapshot(root)
    r = run_ftlen(repo, "toys", STAGES="1")
    errs += [] if (r.returncode == 1 and f"{LROOT_DRY}/scores/toys/s0/swap" in r.stderr and "FTLEN_ALLOW_RETRY=s0" in r.stderr
                   and forbid_training(r) == "" and FR.snapshot(root) == before) else [FR.tail(r)]
    r = run_ftlen(repo, "toys", STAGES="1", FTLEN_ALLOW_RETRY="s1", FTLEN_RETRY_REASON="another seed")        # the seed must be named
    errs += [] if r.returncode == 1 and forbid_training(r) == "" and FR.snapshot(root) == before else [FR.tail(r)]
    r = run_ftlen(repo, "toys", STAGES="1", FTLEN_ALLOW_RETRY="s0")                        # and the reason must be given
    errs += [] if r.returncode == 2 and "FTLEN_RETRY_REASON" in r.stderr and forbid_training(r) == "" else [FR.tail(r)]
    errs += [] if not (root / "report" / "NOTE_FTLEN_overrides.txt").exists() and FR.snapshot(root) == before else ["a refusal wrote"]
    reason = "the trainer was killed at step 40; the weights were never used"
    r = run_ftlen(repo, "toys", STAGES="1", FTLEN_ALLOW_RETRY="s0", FTLEN_RETRY_REASON=reason)
    errs += [] if r.returncode == 0 and FR.trained(r) == ["s0"] and "[retry] s0" in r.stderr else [FR.tail(r)]
    log = (root / "report" / "NOTE_FTLEN_overrides.txt").read_text(encoding="utf-8").splitlines()
    return errs + ([] if len(log) == 1 and re.fullmatch(r"s0\t\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\t" + re.escape(reason), log[0])
                   else [f"log: {log}"])


def scenario_sticky_stage_2(c, repo):
    """The same rule in stage 2: a finished failed state is resumed without refusal; a pass that would be scored again under another
    run.key (the adapter was made again) while FAILED_INTEGRITY sits in like/, swap/ or a .stale.* of that seed is refused (exit 1,
    nothing scored) unless the override is set; the other seed is not affected (it is scored again in the override run)."""
    root, errs = repo / LROOT_DRY, []
    scores = root / "scores" / "toys"
    (scores / "s0" / "swap" / "FAILED_INTEGRITY").write_text("", encoding="utf-8")        # s0's swap pass failed E1 twice
    before = FR.snapshot(root)
    r = run_ftlen(repo, "toys", STAGES="2")                                               # a finished failed state is resumed
    errs += [] if r.returncode == 0 and "[skip]" in r.stdout and FR.snapshot(root) == before else [FR.tail(r)]
    w0, w1 = (root / "adapters" / "toys" / f"s{k}" / "adapter_model.safetensors" for k in (0, 1))
    w0.write_bytes(w0.read_bytes() + b" made again")                                      # the weights changed: another run.key
    r = run_ftlen(repo, "toys", STAGES="2")
    errs += [] if r.returncode == 1 and f"{LROOT_DRY}/scores/toys/s0/swap" in r.stderr and "scores chunk" not in r.stdout else [FR.tail(r)]
    stale = scores / "s0" / "swap.stale.20200101000000"                                   # the marker moved aside by an earlier re-score
    shutil.move(scores / "s0" / "swap", stale)
    r = run_ftlen(repo, "toys", STAGES="2")
    errs += [] if r.returncode == 1 and "swap.stale.20200101000000" in r.stderr and "scores chunk" not in r.stdout else [FR.tail(r)]
    w1.write_bytes(w1.read_bytes() + b" made again")                                      # s1 never failed: its passes are scored again
    r = run_ftlen(repo, "toys", STAGES="2", FTLEN_ALLOW_RETRY="s0", FTLEN_RETRY_REASON="the E1 failure was a full disk")
    errs += [] if (r.returncode == 0 and (scores / "s0" / "swap" / "report.json").is_file() and (stale / "FAILED_INTEGRITY").is_file()
                   and len(list((scores / "s1").glob("*.stale.*"))) == 2 and r.stderr.count("[retry] s0") == 1
                   and "[retry] s1" not in r.stderr) else [FR.tail(r)]
    w0.write_bytes(w0.read_bytes() + b" and again")                                       # the moved-aside marker keeps s0 sticky
    r = run_ftlen(repo, "toys", STAGES="2")
    return errs + ([] if r.returncode == 1 and "swap.stale.20200101000000" in r.stderr else [FR.tail(r)])


SCENARIOS = {"gate_ft": scenario_gate_ft, "records": scenario_records, "recorded_inputs": scenario_missing_or_unregistered_real_adapters,
             "provenance": scenario_provenance, "stage_order": scenario_stage_order, "e1": scenario_e1,
             "e1_leftover": scenario_e1_leftover_and_same_key, "tmp_files": scenario_temporary_files_of_the_bound_tools,
             "sticky_stage_1": scenario_sticky_stage_1, "sticky_stage_2": scenario_sticky_stage_2}


def run_scenarios(chain_future, tpf) -> dict:
    """Every scenario works on its own clone of the finished toys chain; they run concurrently (the script runs dominate), and one
    failing scenario never hides the others."""
    c = chain_future.result()
    base = tpf.mktemp("ftlen_scenarios")

    def work(name):
        sub = base / name
        sub.mkdir()
        try:
            return name, SCENARIOS[name](c, clone(c, sub))
        except Exception as e:
            return name, [f"{type(e).__name__}: {e}"]
    with cf.ThreadPoolExecutor(max_workers=len(SCENARIOS)) as ex:
        return dict(ex.map(work, SCENARIOS))


@needs_chain
def test_refusals_and_the_e1_rule_on_clones_of_the_toys_chain(pipeline):
    results = pipeline["scenarios"].result()
    assert set(results) == set(SCENARIOS)
    failed = {name: errs for name, errs in results.items() if errs}
    assert not failed, "\n\n".join(f"== {name}\n" + "\n".join(map(str, errs)) for name, errs in failed.items())


# ================================================================ 7. ftgrid_extra, the module as it is, on the FT-L root
def extra_build(c: dict, out: Path, label: str):
    """`ftgrid_extra build` as the header of run_ftlen.sh gives it (the module as it is, by hand; --n_boot 20 for the rehearsal)."""
    d = c["d"]
    argv = [sys.executable, "-m", "src.confrec.ftgrid_extra", "build", "--domain", d, "--split",
            str(c["grid"] / "panels" / d / "ftgrid_split.json"), "--panels", str(c["grid"] / "panels" / d), "--scores_root",
            str(c["root"] / "scores"), "--models", "zeroshot,s0,s1", "--raw", str(c["grid"] / "_dry" / "raw"), "--out", str(out),
            "--root_label", label, "--n_boot", "20"]
    r = subprocess.run(argv, cwd=c["repo"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONPATH": str(c["repo"]), "PYTHONUTF8": "1"})
    return r, (read_json(out) if r.returncode == 0 else None)


def finite_est(block) -> bool:
    return isinstance(block, dict) and isinstance(block.get("est"), float) and block["est"] == block["est"] and abs(block["est"]) < 1e9


@needs_chain
def test_ftgrid_extra_builds_its_blocks_on_the_ftlen_root_and_keeps_the_file_out_of_every_family(chain_toys, chain_ml1m, tmp_path):
    """The runner run_ftextra.sh refuses this root; the module takes it by hand (--root_label llama, --models zeroshot,s0,s1): the
    blocks E-W (G_wu), E-J (dUAUC(L - q-hat)) and E-G (the e-share) are computed from the like and swap arms of s0 and s1 and the
    registered zero-shot link (the star-permutation arms of s0 and s1 are absent and not needed), the FT regime is two seeds of
    three (complete false, every statement that needs three seeds is incomplete), the file is not family-eligible, the module records
    the backbone/root mismatch as an input problem and `summarize` refuses the file. With the label `main` the same file would be a
    family-eligible main-root panel: that is why the label is llama."""
    jobs = {("toys", "llama"): (chain_toys, tmp_path / "toys_llama" / "toys.json"),
            ("ml1m", "llama"): (chain_ml1m, tmp_path / "ml1m_llama" / "ml1m.json"),
            ("toys", "main"): (chain_toys, tmp_path / "toys_main" / "toys.json")}
    with cf.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futures = {k: ex.submit(extra_build, c, out, k[1]) for k, (c, out) in jobs.items()}
        built = {k: f.result() for k, f in futures.items()}
    for k, (r, _) in built.items():
        assert r.returncode == 0, (k, FR.tail(r))
    for d in ("toys", "ml1m"):
        res = built[(d, "llama")][1]
        assert res["status"]["root_label"] == "llama" and res["status"]["family_eligible_panel"] is False
        assert res["status"]["FT_C_reading"].startswith("not registered") and res["meta"]["root_label"] == "llama"
        assert res["meta"]["models_requested"] == ["zeroshot", "s0", "s1"] and (res["meta"]["n_boot"], res["meta"]["seed"]) == (20, 0)
        assert res["meta"]["regimes"]["FT"] == {"models": ["s0", "s1"], "missing_or_excluded": [], "n_registered": 3}
        assert res["meta"]["regimes"]["ZS"]["models"] == ["zeroshot"] and not res["excluded_runs"]
        runs = {m: {a: v.get("status") for a, v in arms.items()} for m, arms in res["runs"].items()}
        assert runs == {"zeroshot": {"like": "OK", "swap": "OK", "starperm0": "OK", "starperm1": "OK"},
                        "s0": {"like": "OK", "swap": "OK", "starperm0": "ABSENT", "starperm1": "ABSENT"},
                        "s1": {"like": "OK", "swap": "OK", "starperm0": "ABSENT", "starperm1": "ABSENT"}}
        problems = res["input_checks"]["problems"]                                       # the root/backbone mismatch, nothing else
        assert len(problems) == 1 and "does not belong to the llama root" in problems[0], problems
        # E-W: G_wu of the fine-tuned regime, two seeds of three
        ft = res["E_W"]["FT"]
        assert ft["complete"] is False and ft["models"] == ["s0", "s1"] and res["E_W"]["ZS"]["complete"] is True
        assert finite_est(ft["G_wu"]["mean_over_seeds"]) and set(ft["G_wu"]["per_model"]) == {"s0", "s1"}
        assert ft["G_wu"]["seeds"]["n_seeds"] == 2 and ft["G_wu"]["seeds"]["n_seeds_registered"] == 3 \
            and ft["G_wu"]["seeds"]["complete"] is False
        assert res["E_W"]["P1_wu"]["available"] is False and res["E_W"]["P1_wu"]["decision"]["verdict"] == "P1_WU_INCOMPLETE"
        # E-J: dUAUC(L - q-hat) of the FT regime, and q-hat_T computed from the raw files
        ej = res["E_J"]["FT"]
        assert ej["complete"] is False and finite_est(ej["dUAUC_L_minus_q_hat"]["mean_over_seeds"])
        assert set(ej["dUAUC_L_minus_q_hat"]["per_seed"]) == {"s0", "s1"} and ej["dUAUC_L_minus_q_hat"]["seeds"]["n_seeds"] == 2
        assert res["E_J"]["q_hat_T"]["available"] is True and res["E_J"]["ZS"]["complete"] is True
        # E-G: the e-share of the FT regime
        eg = res["E_G"]["FT"]
        assert eg["complete"] is False and finite_est(eg["e_share"]["mean_over_seeds"]) and set(eg["e_share"]["per_model"]) == {"s0", "s1"}
        assert eg["e_share"]["seeds"]["n_seeds"] == 2 and finite_est(res["E_G"]["ZS"]["e_share"]["mean_over_seeds"])
        # the registered E-B of the FT regime is two seeds: incomplete; P1 needs three seeds
        eb = res["registered_pooled"]["E_B"]
        assert eb["complete"] is False and res["registered_pooled"]["P1"]["decision"]["verdict"] == "INCOMPLETE"
    # with the label main (an exploratory file labelled as the registered root) the file would be a family-eligible Toys panel
    main = built[("toys", "main")][1]
    assert main["status"]["family_eligible_panel"] is True and main["input_checks"]["problems"] == []
    # summarize refuses the llama-labelled file (an exploratory file is never summarized), and writes nothing
    refused = subprocess.run([sys.executable, "-m", "src.confrec.ftgrid_extra", "summarize", "--files",
                              str(jobs[("toys", "llama")][1]), "--out", str(tmp_path / "summary.json")], cwd=chain_toys["repo"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace",
                             env={**os.environ, "PYTHONPATH": str(chain_toys["repo"]), "PYTHONUTF8": "1"})
    assert refused.returncode == 2 and "input_checks.problems is not empty" in refused.stderr and not (tmp_path / "summary.json").exists()


def test_the_real_ftlen_root_is_no_registered_root_of_ftgrid_extra_so_every_label_is_accepted_there():
    """The label check of ftgrid_extra compares canonical paths component by component: outputs/confrec/ftgrid_len3 is not below
    outputs/confrec/ftgrid (a string prefix would say so), it is none of the three registered roots, and any label is accepted
    there. The one the header names, llama, is the one that keeps the file out of every family."""
    from src.confrec import ftgrid_extra as fe
    scores = ROOT / LROOT / "scores"
    assert fe.root_label_of(scores) is None
    assert [fe.root_label_of(ROOT / rel / "scores") for rel in ("outputs/confrec/ftgrid", "outputs/confrec/ftgrid_q",
                                                                  "outputs/confrec/ftgrid_llama")] == ["main", "teacher", "llama"]
    for label in fe.ROOT_LABELS:
        fe.check_root_label(argparse.Namespace(scores_root=str(scores), root_label=label))          # none is refused
    with pytest.raises(SystemExit):                                                                   # a registered root still is
        fe.check_root_label(argparse.Namespace(scores_root=str(ROOT / "outputs/confrec/ftgrid/scores"), root_label="llama"))
    runner = (ROOT / "scripts" / "sigir" / "run_ftextra.sh").read_text(encoding="utf-8")
    assert "ftgrid_len3" not in runner                                                                # the runner does not know this root
    head = SCRIPT.read_text(encoding="utf-8").split("set -euo pipefail")[0]
    assert "--root_label llama" in head and "--scores_root outputs/confrec/ftgrid_len3/scores" in head


@needs_bash
def test_the_extra_runner_refuses_the_ftlen_root_so_the_module_is_run_by_hand(guard_repo):
    """run_ftextra.sh takes the three registered roots only (a real run) and never a path at or under outputs/confrec/ftgrid* (a
    rehearsal): both refuse outputs/confrec/ftgrid_len3 with exit 2 before anything is read or written."""
    shutil.copy2(ROOT / "scripts" / "sigir" / "run_ftextra.sh", guard_repo / "scripts" / "sigir" / "run_ftextra.sh")
    for dry in (False, True):
        env = {k: v for k, v in os.environ.items() if k not in ("OUT_ROOT", "DRY_RUN", "PILOT_LOG", "ROOT_LABEL", "PYTHONPATH")}
        env.update(PYTHON=sys.executable.replace("\\", "/"), OUT_ROOT=LROOT)
        if dry:
            env.update(DRY_RUN="1", PILOT_LOG="x.md")
        r = subprocess.run([FR.BASH, (guard_repo / "scripts" / "sigir" / "run_ftextra.sh").as_posix(), "toys"], env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        assert r.returncode == 2 and ("is not a registered root" in r.stderr or "never touches a path at or under" in r.stderr), FR.tail(r)
    assert not (guard_repo / "outputs").exists() and not (guard_repo / "tmp_outputs").exists()
