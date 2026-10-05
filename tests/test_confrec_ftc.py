"""Tests of FT-C on Toys (Amendment 3 addendum 6 item 8; docs/sigir/FTEXTRA_IMPL_SPEC.md, X2): src/confrec/ftc_panel.py and
scripts/sigir/run_ftc.sh. CPU only, deterministic, no network, no GPU, no model, no torch.

  * the training and scoring argument vectors of p0 / p1 equal the recorded ones of the Toys adapters s0-s2 and of the s0 like
    pass except --train, --out, --seed (and --lora / --output): parsed by the real trainer and scorer argparse;
  * the permuted Toys panel is ftgrid_data's own permutation (byte for byte what ftgrid_data writes for ML-1M), preserves
    every item's label sum and rating multiset and differs from train.jsonl only in the moved fields; the verification
    catches every kind of corruption; the manifest is complete and a rerun is byte-identical;
  * the refusals of a recorded input that is not the registered one (a recipe, a like pass);
  * the script is LF, `bash -n` clean, passes both flag audits, and refuses (not toys, no Gate-FT PASS, a missing s0
    adapter, a missing or stale freeze record, a missing FT-C record, a wrong model or root);
  * run_ftc.sh DRY_RUN=1 end to end on run_ftgrid.sh's own synthetic Toys world (real code everywhere except the trainer and
    the scorer stand-ins): training p0, p1 with s0's recorded recipe, the like passes with s0's recorded arguments, the E1
    rerun-once rule, ftgrid_report with the six models into report/toys_ftc.json while the registered toys.json is never
    touched, and a rerun that changes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from src.confrec import ftc_panel as fp
from src.confrec import ftgrid_data as fd
from src.confrec import pyes_scorer as ps

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sigir" / "run_ftc.sh"
PANEL = ROOT / "src" / "confrec" / "ftc_panel.py"


def _load(name: str):
    """A sibling test module as a library (its helpers; it is never edited here)."""
    spec = importlib.util.spec_from_file_location(f"{name}_for_ftc", ROOT / "tests" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FR = _load("test_confrec_ftgrid_run")      # run_ftgrid.sh's chain helpers, its flag audit, the stand-ins' source
FD = _load("test_confrec_ftgrid_data")     # synthetic rated panels and ftgrid_data's runner
MODEL, VARIANT = "dryrun/Qwen3-8B", "V3"
ADIR = "outputs/confrec/ftgrid/adapters/toys"
SDIR = "outputs/confrec/ftgrid/scores/toys"
TRAIN_REF = "outputs/confrec/ftgrid/panels/toys/train.jsonl"
PERM = "outputs/confrec/ftgrid/ftc/toys/train_perm.jsonl"
SPLIT_RECIPE = {"train": {"overlength": {"micro_bsz": 8, "grad_accum": 4, "max_len_used": 1024}, "examples_written": 15010}}


def sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def sha1_file(p) -> str:
    return sha1(Path(p).read_bytes())


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 1. the training argument vector
@pytest.fixture(scope="module")
def trainer_parser():
    """The real argparse parser of train_lora_yesno (torch replaced by stand-in modules for the import)."""
    with FR.stub_torch():
        parser = FR._parser("src.confrec.train_lora_yesno")
    return parser


def sft_config(parser, k: int, *, bsz: int = 8, accum: int = 4, max_len: int = 1024, model: str = MODEL,
               variant: str = VARIANT) -> dict:
    """The train_config.json train_lora_yesno writes for run_ftgrid.sh's train_adapter call of seed k (its flags exactly:
    --train --model --out --variant --seed --max_len --bsz --grad_accum; vars(args) plus the facts of the run)."""
    argv = ["--train", TRAIN_REF, "--model", model, "--out", f"{ADIR}/s{k}", "--variant", variant, "--seed", str(k),
            "--max_len", str(max_len), "--bsz", str(bsz), "--grad_accum", str(accum)]
    cfg = {**vars(parser.parse_args(argv)), "loss": "last_token", "hist_len_used": 20, "panel_kind": "rated",
           "max_history_len_in_panel": 20, "n_examples": 15010, "n_skipped_overlength": 0}
    return json.loads(json.dumps(cfg))


def flag_pairs(argv: list) -> dict:
    assert len(argv) % 2 == 0 and all(a.startswith("--") for a in argv[::2]), argv   # every flag takes one value
    assert len(set(argv[::2])) == len(argv) // 2, argv                               # no flag twice
    return dict(zip(argv[::2], argv[1::2]))


def test_train_keys_are_exactly_the_trainers_recorded_arguments(trainer_parser):
    dests = {a.dest for a in trainer_parser._actions if a.option_strings and a.dest != "help"}
    assert dests == set(fp.TRAIN_KEYS)
    cfg = sft_config(trainer_parser, 0)
    assert set(fp.TRAIN_KEYS) <= set(cfg) and set(fp.TRAIN_FACTS) <= set(cfg)


@pytest.mark.parametrize("k, bsz, accum, max_len", [(0, 8, 4, 1024), (1, 4, 8, 1280)])
def test_the_ftc_training_vector_equals_the_recorded_vector_except_the_three_flags(trainer_parser, k, bsz, accum, max_len):
    """p_k is trained with the arguments s_k's train_config.json records; only --train, --out and --seed are replaced.
    Parsed by the real trainer parser, the two vectors agree on every other argument, and the three replaced ones are
    the given ones. (Seed 1 here is the recipe of a domain whose length rule chose 4 x 8 at 1280.)"""
    cfgs = {i: sft_config(trainer_parser, i, bsz=bsz, accum=accum, max_len=max_len) for i in (0, 1, 2)}
    split = {"train": {"overlength": {"micro_bsz": bsz, "grad_accum": accum, "max_len_used": max_len},
                       "examples_written": 15010}}
    s0 = fp.check_recipe(cfgs, adapters=ADIR, model=MODEL, variant=VARIANT, train_ref=TRAIN_REF, split=split)
    assert s0 == cfgs[0]
    out = f"{ADIR}/p{k}"
    argv = fp.train_argv(s0, train=PERM, out=out, seed=k)
    got = vars(trainer_parser.parse_args(argv))
    recorded = vars(trainer_parser.parse_args(                      # the vector of s0's train_adapter call, re-parsed
        ["--train", TRAIN_REF, "--model", MODEL, "--out", f"{ADIR}/s0", "--variant", VARIANT, "--seed", "0",
         "--max_len", str(max_len), "--bsz", str(bsz), "--grad_accum", str(accum)]))
    keep = [x for x in fp.TRAIN_KEYS if x not in fp.REPLACED]
    assert set(got) == set(fp.TRAIN_KEYS)
    assert {x: got[x] for x in keep} == {x: recorded[x] for x in keep} == {x: s0[x] for x in keep}
    assert {x: got[x] for x in fp.REPLACED} == {"train": PERM, "out": out, "seed": k}
    # the token level: the vector holds every recorded flag once (an unset optional flag is the default again)
    pairs = flag_pairs(argv)
    assert set(pairs) == {f"--{x}" for x in fp.TRAIN_KEYS if s0[x] is not None}
    assert {f: v for f, v in pairs.items() if f[2:] not in fp.REPLACED} == {
        f"--{x}": fp.flag_value(s0[x]) for x in keep if s0[x] is not None}
    assert fp.recipe_flags(s0) == [t for f, v in pairs.items() if f[2:] not in fp.REPLACED for t in (f, v)]
    assert (pairs["--bsz"], pairs["--grad_accum"], pairs["--max_len"]) == (str(bsz), str(accum), str(max_len))


def test_unset_and_set_optional_flags_round_trip(trainer_parser):
    cfg = sft_config(trainer_parser, 0)
    cfg.update(hist_len=20, max_examples=100, epochs=0.5, lr=2.5e-05)
    got = vars(trainer_parser.parse_args(fp.train_argv(cfg, train="a", out="b", seed=3)))
    assert got == {**{x: cfg[x] for x in fp.TRAIN_KEYS}, "train": "a", "out": "b", "seed": 3}
    cfg.update(hist_len=None, max_examples=None)
    pairs = flag_pairs(fp.train_argv(cfg, train="a", out="b", seed=3))
    assert "--hist_len" not in pairs and "--max_examples" not in pairs
    assert vars(trainer_parser.parse_args(fp.train_argv(cfg, train="a", out="b", seed=3)))["hist_len"] is None
    with pytest.raises(fp.FtcError, match="no record of"):
        fp.train_argv({k: v for k, v in cfg.items() if k != "lora_r"}, train="a", out="b", seed=0)


def good_cfgs(parser) -> dict:
    return {k: sft_config(parser, k) for k in (0, 1, 2)}


def recipe(cfgs, split=SPLIT_RECIPE, **kw):
    args = dict(adapters=ADIR, model=MODEL, variant=VARIANT, train_ref=TRAIN_REF, split=split)
    args.update(kw)
    return fp.check_recipe(cfgs, **args)


def mutate(cfgs: dict, k: int, **kw) -> dict:
    out = {i: dict(c) for i, c in cfgs.items()}
    out[k].update(kw)
    return out


def test_a_valid_recipe_is_accepted_and_its_flags_are_the_run_ftgrid_ones(trainer_parser):
    cfgs = good_cfgs(trainer_parser)
    s0 = recipe(cfgs)
    assert fp.recipe_flags(s0) == ["--model", MODEL, "--mode", "standard", "--variant", VARIANT, "--max_len", "1024",
                                   "--epochs", "1.0", "--lr", "0.0001", "--bsz", "8", "--grad_accum", "4", "--lora_r", "16"]
    assert recipe(cfgs, split=None) == s0


@pytest.mark.parametrize("k, change, msg", [
    (0, {"seed": 7}, "seed is 7"), (1, {"out": "elsewhere/s1"}, "out is"),
    (0, {"train": "other/train.jsonl"}, "train is"), (2, {"model": "/m/Llama"}, "model is"),
    (1, {"variant": "V0"}, "variant is"), (1, {"bsz": 4, "grad_accum": 8}, "not trained with one recipe"),
    (2, {"max_len": 1280}, "not trained with one recipe"), (2, {"n_examples": 15009}, "not trained with one recipe"),
])
def test_adapters_that_do_not_share_one_recipe_are_refused(trainer_parser, k, change, msg):
    with pytest.raises(fp.FtcError, match=msg):
        recipe(mutate(good_cfgs(trainer_parser), k, **change))


@pytest.mark.parametrize("change", [{"epochs": 2.0}, {"lr": 5e-05}, {"lora_r": 8}, {"mode": "mirror"}, {"hist_len": 20},
                                    {"max_examples": 5}])
def test_a_recipe_that_agrees_with_itself_but_is_not_the_registered_one_is_refused(trainer_parser, change):
    cfgs = {i: {**c, **change} for i, c in good_cfgs(trainer_parser).items()}
    with pytest.raises(fp.FtcError, match="registered recipe"):
        recipe(cfgs)


def test_the_recipe_is_checked_against_the_batch_the_split_and_the_examples(trainer_parser):
    cfgs = {k: sft_config(trainer_parser, k, bsz=8, accum=8) for k in (0, 1, 2)}
    with pytest.raises(fp.FtcError, match="effective batch"):
        recipe(cfgs, split=None)
    cfgs = good_cfgs(trainer_parser)
    bad = {"train": {"overlength": {"micro_bsz": 4, "grad_accum": 8, "max_len_used": 1280}, "examples_written": 15010}}
    with pytest.raises(fp.FtcError, match=r"differs from ftgrid_split.json's"):
        recipe(cfgs, split=bad)
    with pytest.raises(fp.FtcError, match="not the registered TRAIN file"):
        recipe(cfgs, split={"train": {**SPLIT_RECIPE["train"], "examples_written": 20000}})
    with pytest.raises(fp.FtcError, match="needs the configs of the adapters s0-s2"):
        recipe({0: cfgs[0], 1: cfgs[1]})


def test_a_missing_sft_adapter_is_named(tmp_path):
    for k in (0, 1):
        (tmp_path / f"s{k}").mkdir()
        (tmp_path / f"s{k}" / "train_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(fp.FtcError, match=r"missing the Toys adapter s2"):
        fp.sft_configs(str(tmp_path))
    assert fp.main(["recipe", "--adapters", str(tmp_path), "--model", MODEL, "--variant", VARIANT,
                    "--train_ref", TRAIN_REF]) == 1                          # an error (exit 1), not a usage refusal
    assert fp.main(["recipe", "--adapters", str(tmp_path)]) == 2             # a usage refusal


def test_a_trained_ftc_adapter_must_record_the_siblings_arguments(trainer_parser):
    s0 = sft_config(trainer_parser, 0)
    p0 = {**vars(trainer_parser.parse_args(fp.train_argv(s0, train=PERM, out=f"{ADIR}/p0", seed=0))),
          **{k: s0[k] for k in fp.TRAIN_FACTS}}
    fp.check_same_training(s0, p0, train=PERM, out=f"{ADIR}/p0", seed=0)
    for change, msg in (({"lr": 2e-4}, "lr"), ({"max_len": 512}, "max_len"), ({"bsz": 4}, "bsz"),
                        ({"n_examples": 15000}, "n_examples"), ({"variant": "V0"}, "variant"),
                        ({"train": TRAIN_REF}, "train is"), ({"seed": 1}, "seed is"), ({"out": f"{ADIR}/p1"}, "out is")):
        with pytest.raises(fp.FtcError, match=msg):
            fp.check_same_training(s0, {**p0, **change}, train=PERM, out=f"{ADIR}/p0", seed=0)


# ---------------------------------------------------------------- 2. the scoring argument vector
@pytest.fixture(scope="module")
def fakes(tmp_path_factory):
    """run_ftgrid.sh's DRY_RUN stand-ins, imported from the source the script writes (the real scorer, a fake model)."""
    d = tmp_path_factory.mktemp("fakes")
    (d / "ftgrid_fakes.py").write_text(FR.fakes_source(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ftgrid_fakes_for_ftc", d / "ftgrid_fakes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_argv(data, out, lora) -> list:
    """The pyes_scorer call of run_ftgrid.sh's score() for a like pass of an adapter."""
    return ["--data", str(data), "--output", str(out), "--model", MODEL, "--dtype", "float16", "--topk_logprobs", "50",
            "--max_model_len", "4096", "--chunk_users", "100", "--variant", VARIANT, "--readout", "yesno",
            "--questions", "like", "--lora", lora]


@pytest.fixture(scope="module")
def like_world(tmp_path_factory, fakes):
    """A scored like pass of an adapter s0 on a tiny rated panel, recorded as run_ftgrid.sh records it (run.key, the real
    scorer's report.json), and an adapter p0 beside it."""
    root = tmp_path_factory.mktemp("like")
    panel = FR.write_rows(root / "eval.jsonl", FR.rated_rows(n=6, hist=20))
    adapters, scores = (root / "adapters").as_posix(), (root / "scores").as_posix()
    for name in ("s0", "p0"):
        (root / "adapters" / name).mkdir(parents=True)
        (root / "adapters" / name / "adapter_model.safetensors").write_bytes(f"weights of {name}".encode())
    fakes.scorer(score_argv(panel, f"{scores}/s0/like", f"{adapters}/s0"))
    key = f"{sha1_file(panel)} {MODEL} {VARIANT} {fp.weights_sha1(root / 'adapters' / 's0')} --lora {adapters}/s0"
    (root / "scores" / "s0" / "like" / "run.key").write_text(key + "\n", encoding="utf-8")
    return {"root": root, "panel": panel, "adapters": adapters, "scores": scores, "key": key}


def scoring_args(w, **kw) -> argparse.Namespace:
    d = dict(command="scoring", scores=w["scores"], adapters=w["adapters"], data=str(w["panel"]), model=MODEL,
             variant=VARIANT)
    d.update(kw)
    return argparse.Namespace(**d)


def test_the_ftc_scoring_vector_equals_the_recorded_like_pass_except_lora_and_output(like_world, fakes):
    w = like_world
    flags = fp.cmd_scoring(scoring_args(w))
    mine = score_argv(w["panel"], f"{w['scores']}/p0/like", f"{w['adapters']}/p0")
    sibling = score_argv(w["panel"], f"{w['scores']}/s0/like", f"{w['adapters']}/s0")
    # flags + the literal --data/--output/--model/--lora of the script's score() = the like pass's vector
    assert flag_pairs(flags) == {f: v for f, v in flag_pairs(sibling).items()
                                 if f not in ("--data", "--output", "--model", "--lora")}
    a = vars(ps.parse_args(["--data", str(w["panel"]), "--output", f"{w['scores']}/p0/like", "--model", MODEL] + flags
                           + ["--lora", f"{w['adapters']}/p0"]))
    b = vars(ps.parse_args(mine))
    c = vars(ps.parse_args(sibling))
    assert a == b and {k: v for k, v in a.items() if k not in ("output", "lora")} == {
        k: v for k, v in c.items() if k not in ("output", "lora")}
    # and what the scorer then records for p0 equals what it recorded for s0 except the adapter
    fakes.scorer(["--data", str(w["panel"]), "--output", f"{w['scores']}/p0/like", "--model", MODEL] + flags
                 + ["--lora", f"{w['adapters']}/p0"])
    ref = fp.report_config(read_json(f"{w['scores']}/s0/like/report.json"), "s0")
    run = fp.report_config(read_json(f"{w['scores']}/p0/like/report.json"), "p0")
    fp.check_same_scoring(ref, run, lora=f"{w['adapters']}/p0", where="p0")
    assert ref["lora"] != run["lora"] and {k for k in ref if ref[k] != run[k]} == {"lora"}
    assert fp.main(["verify_scores", "--scores", w["scores"], "--adapters", w["adapters"], "--p_seed", "0"]) == 0


def test_a_p_run_scored_with_other_arguments_is_caught(like_world, fakes):
    w = like_world
    ref = fp.report_config(read_json(f"{w['scores']}/s0/like/report.json"), "s0")
    for flag, value, key in (("--topk_logprobs", "40", "topk_logprobs"), ("--chunk_users", "50", "chunk_users"),
                             ("--dtype", "bfloat16", "dtype"), ("--swap_k", "2", "swap_k"), ("--hist_len", "0", "hist_len"),
                             ("--seed", "1", "seed")):
        out = f"{w['scores']}/bad_{key}/like"
        fakes.scorer(score_argv(w["panel"], out, f"{w['adapters']}/p0") + [flag, value])
        run = fp.report_config(read_json(f"{out}/report.json"), "bad")
        with pytest.raises(fp.FtcError, match=key):
            fp.check_same_scoring(ref, run, lora=f"{w['adapters']}/p0", where="bad")
    with pytest.raises(fp.FtcError, match="scored with"):
        fp.check_same_scoring(ref, {**ref, "lora": "x"}, lora=f"{w['adapters']}/p0", where="p0")


def record_args(w, **kw) -> dict:
    d = dict(data_sha1=sha1_file(w["panel"]), model=MODEL, variant=VARIANT, lora=f"{w['adapters']}/s0",
             lora_weights_sha1=fp.weights_sha1(Path(w["root"]) / "adapters" / "s0"))
    d.update(kw)
    return d


def test_the_recorded_like_pass_is_validated_against_the_registered_one(like_world):
    w = like_world
    cfg = read_json(f"{w['scores']}/s0/like/report.json")["config"]
    assert fp.check_scoring_record(w["key"], cfg, **record_args(w)) == [
        "--dtype", "float16", "--topk_logprobs", "50", "--max_model_len", "4096", "--chunk_users", "100", "--variant",
        VARIANT, "--readout", "yesno", "--questions", "like"]
    key = w["key"].split()
    bad_keys = {"panel": " ".join(["0" * 40] + key[1:]), "model": " ".join([key[0], "/m/Llama"] + key[2:]),
                "variant": " ".join(key[:2] + ["V0"] + key[3:]), "weights": " ".join(key[:3] + ["1" * 40] + key[4:]),
                "extra args": w["key"] + " --swap_k 8", "other adapter": w["key"].replace("/s0", "/s1")}
    for name, text in bad_keys.items():
        with pytest.raises(fp.FtcError, match="run.key"):
            fp.check_scoring_record(text, cfg, **record_args(w))
        assert name


@pytest.mark.parametrize("change, msg", [
    ({"dtype": "bfloat16"}, "dtype"), ({"topk_logprobs": 20}, "topk_logprobs"), ({"max_model_len": 2048}, "max_model_len"),
    ({"chunk_users": 50}, "chunk_users"), ({"readout": "digits"}, "readout"), ({"questions": ["like", "dislike"]}, "questions"),
    ({"swap_k": 8}, "swap_k"), ({"seed": 1}, "seed"), ({"n_users": 10}, "n_users"), ({"chunk_items": 10}, "chunk_items"),
    ({"hist_len": 0}, "hist_len"), ({"data_sha1": "0" * 40}, "data_sha1"), ({"model": "/m/Llama"}, "model"),
    ({"variant": "V0"}, "variant"), ({"lora": "elsewhere/s0"}, "lora"),
])
def test_a_like_pass_that_is_not_the_registered_one_is_refused(like_world, change, msg):
    w = like_world
    cfg = {**read_json(f"{w['scores']}/s0/like/report.json")["config"], **change}
    with pytest.raises(fp.FtcError, match=msg):
        fp.check_scoring_record(w["key"], cfg, **record_args(w))


def test_the_scoring_command_refuses_a_missing_or_stale_like_pass(like_world, tmp_path):
    w = like_world
    ns = scoring_args(w)
    assert fp.cmd_scoring(ns)
    with pytest.raises(fp.FtcError, match="missing the like pass of the Toys adapter s0"):
        fp.cmd_scoring(scoring_args(w, scores=str(tmp_path / "none")))
    other = FR.write_rows(tmp_path / "eval2.jsonl", FR.rated_rows(n=7, hist=20))        # eval.jsonl changed since the pass
    with pytest.raises(fp.FtcError, match="panel sha1"):
        fp.cmd_scoring(scoring_args(w, data=str(other)))
    root = tmp_path / "copy"
    shutil.copytree(w["root"], root)
    a2, s2 = (root / "adapters").as_posix(), (root / "scores").as_posix()
    (root / "scores" / "s0" / "like" / "run.key").write_text(w["key"].replace(w["adapters"], a2) + "\n", encoding="utf-8")
    rep = read_json(root / "scores" / "s0" / "like" / "report.json")
    rep["config"]["lora"] = f"{a2}/s0"
    (root / "scores" / "s0" / "like" / "report.json").write_text(json.dumps(rep), encoding="utf-8")
    assert fp.cmd_scoring(scoring_args(w, scores=s2, adapters=a2))                       # consistent copy: accepted
    (root / "adapters" / "s0" / "adapter_model.safetensors").write_bytes(b"retrained")   # the adapter changed since
    with pytest.raises(fp.FtcError, match="adapter weights sha1"):
        fp.cmd_scoring(scoring_args(w, scores=s2, adapters=a2))
    assert fp.main(["scoring", "--scores", s2, "--adapters", a2, "--data", str(w["panel"]), "--model", MODEL,
                    "--variant", VARIANT]) == 1


def test_the_hist_len_of_the_record_is_the_variants_registered_window(like_world):
    w = like_world
    cfg = read_json(f"{w['scores']}/s0/like/report.json")["config"]
    assert cfg["hist_len"] == 20 and cfg["panel_kind"] == "rated" and cfg["variant"] == "V3"
    with pytest.raises(fp.FtcError, match="panel_kind|not in"):
        fp.check_scoring_record(w["key"], {**cfg, "panel_kind": None}, **record_args(w))


# ---------------------------------------------------------------- 3. the permuted panel
def panel_rows(n_users: int = 40, seed: int = 3) -> list:
    """Rated rows of 8 candidates over 12 items, so that every item has many examples; row 0's earliest candidate is the
    item 'solo' (one TRAIN example)."""
    rows = FD.random_panel(n_users, n_cands=8, n_items=12, seed=seed)
    j0 = min(range(8), key=lambda j: rows[0]["candidate_timestamps"][j])
    rows[0]["candidate_item_ids"][j0] = "solo"
    return rows


def jsonl(path) -> list:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def test_the_permuted_panel_is_what_ftgrid_data_writes_for_ml1m_byte_for_byte(tmp_path):
    """FT-C's panel is built `exactly as section 9 builds the ML-1M one`: ftc_panel on an ML-1M train.jsonl reproduces the
    train_perm.jsonl ftgrid_data itself wrote next to it (same function, seed 0, same writer)."""
    rep, out = FD.run(tmp_path, panel_rows(), "--n_train", 30, domain="ml1m")
    man = fp.build(out / "train.jsonl", tmp_path / "ftc", split=out / "ftgrid_split.json", domain="ml1m")
    assert (tmp_path / "ftc" / "train_perm.jsonl").read_bytes() == (out / "train_perm.jsonl").read_bytes()
    assert man["train_perm"]["sha1"] == rep["files"]["train_perm.jsonl"] and man["train"]["sha1"] == rep["files"]["train.jsonl"]
    assert man["split"]["ftgrid_split_sha1"] == sha1_file(out / "ftgrid_split.json")


@pytest.fixture(scope="module")
def toys_panel(tmp_path_factory):
    """A Toys-shaped split (ftgrid_data does not write train_perm.jsonl for it) and FT-C's permuted panel."""
    tmp = tmp_path_factory.mktemp("toyspanel")
    rep, out = FD.run(tmp, panel_rows(), "--n_train", 30, domain="toys")
    assert not (out / "train_perm.jsonl").exists()
    man = fp.build(out / "train.jsonl", tmp / "ftc", split=out / "ftgrid_split.json", domain="toys")
    return {"tmp": tmp, "out": out, "ftc": tmp / "ftc", "rep": rep, "man": man}


def test_the_permuted_panel_preserves_label_sums_and_differs_only_in_the_moved_fields(toys_panel):
    t = toys_panel
    train, perm = jsonl(t["out"] / "train.jsonl"), jsonl(t["ftc"] / "train_perm.jsonl")
    assert len(train) == len(perm) > 20
    sums, pairs, n_ex, changed = defaultdict(lambda: [0, 0]), defaultdict(lambda: [[], []]), Counter(), 0
    for r, p in zip(train, perm):
        assert list(r) == list(p)                                                   # the same keys in the same order
        assert all(r[k] == p[k] for k in r if k not in ("candidate_labels", "candidate_ratings"))
        assert len(p["candidate_labels"]) == len(p["candidate_ratings"]) == len(r["candidate_labels"])
        for i, y, z, a, b in zip(r["candidate_item_ids"], r["candidate_labels"], p["candidate_labels"],
                                 r["candidate_ratings"], p["candidate_ratings"]):
            sums[i][0] += y
            sums[i][1] += z
            pairs[i][0].append((y, a))
            pairs[i][1].append((z, b))
            n_ex[i] += 1
            changed += y != z
            assert z == int(b >= 4)                                                 # the star rating moved with its label
    assert all(a == b for a, b in sums.values())                                    # every item's label sum preserved
    assert all(sorted(a) == sorted(b) for a, b in pairs.values())                   # ... and its (label, rating) pairs
    assert changed > 0 and n_ex["solo"] == 1 and sums["solo"][0] == sums["solo"][1]
    man = t["man"]
    assert man["n_labels_changed"] == changed
    assert man["train"]["examples"] == man["train_perm"]["examples"] == sum(n_ex.values())
    assert man["items"] == {"items": len(n_ex), "items_with_one_example": sum(c == 1 for c in n_ex.values()),
                            "items_with_2plus_examples": sum(c >= 2 for c in n_ex.values()),
                            "items_with_both_classes": sum(0 < s[0] < n_ex[i] for i, s in sums.items() if n_ex[i] >= 2)}


def test_the_manifest_records_the_hashes_the_seed_and_the_code_and_a_rerun_is_byte_identical(toys_panel, tmp_path):
    t = toys_panel
    man = read_json(t["ftc"] / "train_perm.manifest.json")
    assert man == t["man"] and man["seed"] == 0 and man["domain"] == "toys"
    assert man["function"] == "src.confrec.ftgrid_data.permute_within_item"
    assert man["train"]["sha1"] == sha1_file(t["out"] / "train.jsonl") == t["rep"]["files"]["train.jsonl"]
    assert man["train_perm"]["sha1"] == sha1_file(t["ftc"] / "train_perm.jsonl")
    assert man["code_sha1"] == {"ftc_panel.py": sha1_file(PANEL), "ftgrid_data.py": sha1_file(fd.__file__),
                                "build_rated_panels.py": sha1_file(fd.brp.__file__)}
    assert man["moved_fields"] == ["candidate_labels", "candidate_ratings"] and all(
        v is True for v in man["checks"].values())
    assert man["split"]["T"] == t["rep"]["T"] and man["split"]["train_sha1_recorded"] == man["train"]["sha1"]
    text = (t["ftc"] / "train_perm.manifest.json").read_text(encoding="utf-8")
    assert "\\" not in text and ":/" not in text and "NaN" not in text            # no path, no clock, strict JSON
    assert b"\r" not in (t["ftc"] / "train_perm.manifest.json").read_bytes()      # LF on every platform (its sha1 is recorded)
    again = tmp_path / "again"
    fp.build(t["out"] / "train.jsonl", again, split=t["out"] / "ftgrid_split.json")
    for name in ("train_perm.jsonl", "train_perm.manifest.json"):
        assert (again / name).read_bytes() == (t["ftc"] / name).read_bytes()
    before = {n: (t["ftc"] / n).stat().st_mtime_ns for n in ("train_perm.jsonl", "train_perm.manifest.json")}
    fp.build(t["out"] / "train.jsonl", t["ftc"], split=t["out"] / "ftgrid_split.json")      # onto the existing files
    assert {n: (t["ftc"] / n).stat().st_mtime_ns for n in before} == before            # untouched: the skip rules hold
    assert not list(t["ftc"].glob("*.tmp"))


def rewrite(path: Path, rows: list, ascii_only: bool = False) -> None:
    dump = (lambda r: json.dumps(r)) if ascii_only else (lambda r: json.dumps(r, ensure_ascii=False))
    path.write_bytes(b"".join((dump(r) + "\n").encode("utf-8") for r in rows))


def test_a_train_file_that_is_not_the_registered_one_is_refused(toys_panel, tmp_path):
    t = toys_panel
    train = jsonl(t["out"] / "train.jsonl")
    split = read_json(t["out"] / "ftgrid_split.json")
    # (a) another sha1 than the split records for train.jsonl
    changed = [dict(r) for r in train]
    changed[0] = {**changed[0], "history_titles": ["x"] * len(changed[0]["history_titles"])}
    rewrite(tmp_path / "train.jsonl", changed)
    with pytest.raises(fp.FtcError, match="records for train.jsonl"):
        fp.build(tmp_path / "train.jsonl", tmp_path / "o1", split=t["out"] / "ftgrid_split.json")
    # (b) a split of another dataset
    (tmp_path / "split_ml1m.json").write_text(json.dumps({**split, "domain": "ml1m"}), encoding="utf-8")
    with pytest.raises(fp.FtcError, match="split of 'ml1m'"):
        fp.build(t["out"] / "train.jsonl", tmp_path / "o2", split=tmp_path / "split_ml1m.json", domain="toys")
    # (c) lines that are not what ftgrid_data writes (ASCII-escaped non-ASCII text)
    odd = [dict(r) for r in train]
    odd[0]["candidate_titles"] = ["Café " + x for x in odd[0]["candidate_titles"]]
    rewrite(tmp_path / "odd.jsonl", odd, ascii_only=True)
    with pytest.raises(fp.FtcError, match="not in the format ftgrid_data writes"):
        fp.build(tmp_path / "odd.jsonl", tmp_path / "o3")
    # (d) a blank line, a missing file, no rows, a row that is not a rated-panel row
    (tmp_path / "blank.jsonl").write_bytes((t["out"] / "train.jsonl").read_bytes() + b"\n\n")
    with pytest.raises(fp.FtcError, match="blank lines"):
        fp.build(tmp_path / "blank.jsonl", tmp_path / "o4")
    (tmp_path / "empty.jsonl").write_bytes(b"")
    with pytest.raises(fp.FtcError, match="no row"):
        fp.build(tmp_path / "empty.jsonl", tmp_path / "o5")
    bad = [dict(r) for r in train]
    del bad[0]["candidate_labels"]
    rewrite(tmp_path / "bad.jsonl", bad)
    with pytest.raises(fp.FtcError, match="rated-panel row"):
        fp.build(tmp_path / "bad.jsonl", tmp_path / "o6")
    assert fp.main(["build", "--train", str(tmp_path / "missing.jsonl"), "--out_dir", str(tmp_path / "o7")]) == 2
    assert fp.main(["build", "--out_dir", str(tmp_path / "o7")]) == 2
    assert not any((tmp_path / f"o{k}").exists() and any((tmp_path / f"o{k}").iterdir()) for k in range(1, 8))


def test_the_verification_catches_every_kind_of_corruption(toys_panel):
    rows = jsonl(toys_panel["out"] / "train.jsonl")
    perm, n = fd.permute_within_item(rows, 0)
    perm = json.loads(json.dumps(perm))
    assert fp.verify_permutation(rows, perm)["labels_changed"] == n > 0

    def corrupt(fn):
        bad = json.loads(json.dumps(perm))
        fn(bad)
        return bad

    def flip_label_only(b):                                       # a label moves without its rating
        b[1]["candidate_labels"][0] = 1 - b[1]["candidate_labels"][0]

    def new_rating_and_label(b):                                  # a pair that no example of the item had
        b[1]["candidate_labels"][0], b[1]["candidate_ratings"][0] = 1, 4.5

    def other_field(b):
        b[2]["candidate_titles"][0] = "tampered"

    def other_key(b):
        b[3]["extra"] = 1

    def shorter(b):
        b[4]["candidate_labels"].pop()

    def unequal_rows(b):
        b.pop()

    for fn, msg in ((flip_label_only, "not preserved"), (new_rating_and_label, "not preserved"),
                    (other_field, r"changed \(only"), (other_key, "other keys"), (shorter, "changed|length"),
                    (unequal_rows, "rows")):
        with pytest.raises(fp.FtcError, match=msg):
            fp.verify_permutation(rows, corrupt(fn))
    no_ratings = [{k: v for k, v in r.items() if k != "candidate_ratings"} for r in rows]
    perm_nr = [{k: v for k, v in r.items() if k != "candidate_ratings"} for r in perm]
    assert fp.verify_permutation(no_ratings, perm_nr)["moved_fields"] == ["candidate_labels"]
    with pytest.raises(fp.FtcError, match="present in some rows only"):
        fp.verify_permutation(rows[:3] + no_ratings[3:6], perm[:3] + perm_nr[3:6])


def test_the_build_command_prints_the_manifest_and_the_module_needs_no_torch(toys_panel, tmp_path, capsys):
    t = toys_panel
    assert fp.main(["build", "--train", str(t["out"] / "train.jsonl"), "--out_dir", str(tmp_path / "ftc"), "--split",
                    str(t["out"] / "ftgrid_split.json")]) == 0
    cap = capsys.readouterr()
    assert json.loads(cap.out) == t["man"]
    assert f"{t['man']['n_labels_changed']} of {t['man']['train']['examples']} TRAIN labels moved" in cap.err
    code = ("import sys; import src.confrec.ftc_panel; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'transformers', 'vllm', 'peft')]; "
            "assert not bad, bad")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------- 4. provenance, the record, the commands
def test_adapter_provenance_binds_an_adapter_to_the_permuted_panel(trainer_parser, tmp_path, capsys):
    adapters = tmp_path / "adapters"
    A = adapters.as_posix()
    perm = tmp_path / "train_perm.jsonl"
    perm.write_bytes(b"permuted rows")
    manifest = tmp_path / "train_perm.manifest.json"
    manifest.write_text(json.dumps({"train_perm": {"sha1": sha1(b"permuted rows")}}), encoding="utf-8")
    s0 = sft_config(trainer_parser, 0)
    (adapters / "s0").mkdir(parents=True)
    (adapters / "s0" / "train_config.json").write_text(json.dumps(s0), encoding="utf-8")
    (adapters / "p0").mkdir()
    p0 = {**vars(trainer_parser.parse_args(fp.train_argv(s0, train=str(perm), out=f"{A}/p0", seed=0))),
          **{k: s0[k] for k in fp.TRAIN_FACTS}}
    (adapters / "p0" / "train_config.json").write_text(json.dumps(p0), encoding="utf-8")
    args = ["verify_adapter", "--adapters", A, "--p_seed", "0", "--train", str(perm), "--perm_manifest", str(manifest)]
    assert fp.main(args) == 1 and "has no ftc.json" in capsys.readouterr().err        # trained by hand: refused
    assert fp.main(args + ["--write"]) == 0
    prov = read_json(adapters / "p0" / "ftc.json")
    assert prov["train_perm_sha1"] == sha1(b"permuted rows") and prov["adapter"] == "p0" and prov["args"] == {
        k: p0[k] for k in fp.TRAIN_KEYS}
    assert fp.main(args) == 0                                                          # an existing adapter is accepted
    prov["train_perm_sha1"] = "0" * 40                                                 # trained on another permutation
    (adapters / "p0" / "ftc.json").write_text(json.dumps(prov), encoding="utf-8")
    assert fp.main(args) == 1 and "was trained on a permuted panel" in capsys.readouterr().err
    fp.write_provenance(adapters / "p0", p_seed=0, perm_sha1=sha1(b"permuted rows"), manifest_sha1="x", args={})
    perm.write_bytes(b"another permutation")                                           # the panel on disk changed
    assert fp.main(args) == 1 and "is not the file" in capsys.readouterr().err
    perm.write_bytes(b"permuted rows")
    (adapters / "p0" / "train_config.json").write_text(json.dumps({**p0, "lr": 3e-4}), encoding="utf-8")
    assert fp.main(args) == 1 and "lr" in capsys.readouterr().err                      # arguments other than s0's


def test_the_ftc_record_gate_needs_the_sha1_of_the_files_in_the_pilot_log(tmp_path, capsys):
    (tmp_path / "a.sh").write_text("script a\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("module b\n", encoding="utf-8")
    log = tmp_path / "PILOT_LOG.md"
    log.write_text("# log\n", encoding="utf-8")
    base = ["record", "--pilot_log", str(log), "--files", "a.sh", "b.py", "--root", str(tmp_path)]
    assert fp.main(base) == 4 and "a.sh" in capsys.readouterr().err                   # not recorded: refused (exit 4)
    assert fp.main(base + ["--print"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"a.sh = {sha1_file(tmp_path / 'a.sh')}", f"b.py = {sha1_file(tmp_path / 'b.py')}"]
    log.write_text("# log\n" + lines[0].upper() + "\n" + lines[1] + "\n", encoding="utf-8")   # case-insensitive, as freeze
    assert fp.main(base) == 0 and "FT-C record OK" in capsys.readouterr().out
    (tmp_path / "b.py").write_text("module b, changed after the record\n", encoding="utf-8")
    assert fp.main(base) == 4 and "b.py" in capsys.readouterr().err                   # a changed file needs a new record
    assert fp.main(["record", "--pilot_log", str(tmp_path / "none.md"), "--files", "a.sh", "--root", str(tmp_path)]) == 4
    assert fp.main(["record", "--files", "a.sh"]) == 2                                 # usage
    # the default root is this checkout: the record of the real files is printable
    assert fp.record_lines(["scripts/sigir/run_ftc.sh", "src/confrec/ftc_panel.py"]) == [
        f"scripts/sigir/run_ftc.sh = {sha1_file(SCRIPT)}", f"src/confrec/ftc_panel.py = {sha1_file(PANEL)}"]


def test_the_command_line_is_one_flat_parser_with_every_command_and_exit_codes(capsys):
    assert set(fp.COMMANDS) == {"build", "recipe", "scoring", "verify_adapter", "verify_scores", "record"}
    for cmd in ("recipe", "scoring", "verify_adapter", "verify_scores", "record", "build"):
        assert fp.main([cmd]) == 2 and "needs" in capsys.readouterr().err               # a missing flag is a usage refusal
    with pytest.raises(SystemExit):
        fp.main(["nonsense"])
    p = subprocess.run([sys.executable, "-m", "src.confrec.ftc_panel", "record", "--files", "x"], cwd=ROOT,
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert p.returncode == 2 and "needs --pilot_log" in p.stderr


# ---------------------------------------------------------------- 5. the script file, its flags and its input guards
LLAMA_PATH = "/models/Llama-3.1-8B-Instruct"
needs_bash = pytest.mark.skipif(FR.BASH is None, reason=FR.NO_BASH)


def test_script_is_lf_bash_and_bash_n_clean():
    raw = SCRIPT.read_bytes()
    assert b"\r\n" not in raw and raw.startswith(b"#!/usr/bin/env bash\n") and b"set -euo pipefail" in raw
    assert b"\r\n" not in PANEL.read_bytes()
    if FR.BASH is None:
        pytest.skip(FR.NO_BASH)
    r = subprocess.run([FR.BASH, "-n", SCRIPT.as_posix()], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_every_flag_the_script_passes_exists_in_the_real_argparse(trainer_parser, like_world):
    """The stricter audit of test_confrec_ftgrid_run (heredocs ignored, arrays expanded, literal values checked against
    choices and types), with RECIPE and SCORING flags filled in from ftc_panel's own output: the flags the script takes
    from the recorded configs are audited too."""
    text = SCRIPT.read_text(encoding="utf-8")
    cfg = read_json(f"{like_world['scores']}/s0/like/report.json")["config"]
    arrays = (f"\nRECIPE=({' '.join(fp.recipe_flags(sft_config(trainer_parser, 0)))})"
              f"\nSARGS=({' '.join(fp.score_flags(cfg))})\n")
    problems, count = FR.audit(text + arrays)
    assert not problems, "\n".join(problems)
    for target, n in {"src.confrec.pyes_scorer": 2, "src.confrec.train_lora_yesno": 1, "src.confrec.ftgrid_freeze": 2,
                      "src.confrec.ftgrid_report": 1, "src.confrec.ftc_panel": 7}.items():
        assert count.get(target, 0) >= n, (target, count)
    bad = text.replace("--p_seed", "--pseed").replace("--perm_manifest", "--perm_manifes").replace(
        "--scores_root", "--scoresroot")
    problems, _ = FR.audit(bad + arrays)
    assert any("--pseed" in p for p in problems) and any("--scoresroot" in p for p in problems)


def test_the_integration_audit_of_test_confrec_contracts_accepts_the_script(trainer_parser):
    mod = _load("test_confrec_contracts")
    assert SCRIPT in sorted((ROOT / "scripts" / "sigir").glob("*.sh"))        # found by its glob: no registration needed
    with FR.stub_torch():
        mod.test_every_script_flag_exists_in_the_real_argparse(SCRIPT)


def test_the_script_never_names_a_bound_output_it_must_not_write():
    """No stage writes panels/toys, report/toys.json, the s0-s2 adapters or any zeroshot / s0-s2 scoring dir."""
    code = re.sub(r"\\\n\s*", " ", "\n".join(x.split("#", 1)[0] for x in SCRIPT.read_text(encoding="utf-8").splitlines()))
    for pat in (r'--out_dir "\$P"', r'--output "\$S/(zeroshot|s[012])', r'--out "\$ADIR/s', r'--out "\$REGREP"',
                r'> *"\$REGREP"', r'rm -rf "\$P"', r'rm -rf "\$ADIR/s'):
        assert not re.search(pat, code), pat
    assert "REP=\"$OUT_ROOT/report/toys_ftc.json\"" in code and 'REGREP="$OUT_ROOT/report/toys.json"' in code
    assert not re.search(r"git ", code)


@pytest.fixture(scope="module")
def guard_repo(tmp_path_factory):
    if FR.BASH is None:
        pytest.skip(FR.NO_BASH)
    return make_ftc_repo(tmp_path_factory.mktemp("guards") / "repo")


def make_ftc_repo(dest: Path) -> Path:
    """The repo parts the chain runs (run_ftgrid.sh's copy with the real ftgrid_report) plus this script."""
    FR.make_repo(dest, real_f2=True)
    shutil.copy2(SCRIPT, dest / "scripts" / "sigir" / "run_ftc.sh")
    return dest


def run_ftc(repo: Path, *args, dry: bool = True, **env) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in (
        "MODEL", "OUT_ROOT", "VARIANT", "STAGES", "DRY_RUN", "DRY_GATE", "DRY_E1_FAIL", "DRY_NO_RECORD", "PYTHONPATH")}
    e.update(PYTHON=sys.executable.replace("\\", "/"), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if dry:
        e["DRY_RUN"] = "1"
    e.update(env)
    script = (repo / "scripts" / "sigir" / "run_ftc.sh").as_posix()
    return subprocess.run([FR.BASH, script, *args], env=e, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=3600)


@needs_bash
@pytest.mark.parametrize("args, dry, env, rc, msg", [
    (["ml1m"], True, {}, 2, "Toys only"), (["games"], True, {}, 2, "Toys only"), (["sports"], False, {}, 2, "Toys only"),
    (["beauty"], True, {}, 2, "usage"), ([], True, {}, 2, "usage"),
    (["toys"], True, {"STAGES": "0,7"}, 2, "unknown stage"), (["toys"], True, {"STAGES": "5"}, 2, "unknown stage"),
    (["toys"], True, {"OUT_ROOT": "outputs/confrec/ftgrid"}, 2, "never writes to a registered output root"),
    (["toys"], True, {"OUT_ROOT": "./outputs/confrec/ftgrid_llama/"}, 2, "never writes to a registered output root"),
    (["toys"], True, {"MODEL": LLAMA_PATH}, 2, "is not Qwen3-8B"),
    (["toys"], False, {"MODEL": LLAMA_PATH}, 2, "is not Qwen3-8B"),
    (["toys"], False, {"MODEL": "/models/Qwen3-8B", "OUT_ROOT": "outputs/confrec/ftgrid_dryrun"}, 2, "one registered root"),
    (["toys"], False, {"MODEL": "/models/Qwen3-8B", "OUT_ROOT": "outputs/confrec/ftgrid_llama"}, 2, "one registered root"),
    (["toys"], False, {"MODEL": "/models/Qwen3-8B"}, 1, "missing outputs/confrec/gatefix/dev/selection.json"),
])
def test_input_guards_refuse_before_anything_is_written(guard_repo, args, dry, env, rc, msg):
    r = run_ftc(guard_repo, *args, dry=dry, **env)
    assert r.returncode == rc and msg in r.stderr, FR.tail(r)
    assert not (guard_repo / "outputs").exists()


# ---------------------------------------------------------------- 6. the DRY_RUN chain
DRY_ROOT = "outputs/confrec/ftgrid_dryrun"          # run_ftgrid.sh's DRY_RUN root: the world of run_ftc.sh
SENTINEL = b'{"registered": "report/toys.json", "never": "rewritten"}\n'
FTC_LINES = ("scripts/sigir/run_ftc.sh", "src/confrec/ftc_panel.py")


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """run_ftc.sh DRY_RUN=1 toys end to end (run_ftgrid.sh's own synthetic Toys world is built first), then the same
    command a second time. A stand-in registered report sits in report/ before the first run."""
    if FR.BASH is None:
        pytest.skip(FR.NO_BASH)
    repo = make_ftc_repo(tmp_path_factory.mktemp("chain") / "repo")
    reports = repo / DRY_ROOT / "report"
    reports.mkdir(parents=True)
    (reports / "toys.json").write_bytes(SENTINEL)
    (reports / "toys_tables.csv").write_bytes(b"model,arm\n")
    c = {"repo": repo, "root": repo / DRY_ROOT, "r": run_ftc(repo, "toys")}
    assert c["r"].returncode == 0, FR.tail(c["r"])
    c["s1"] = FR.snapshot(c["root"])
    c["r2"] = run_ftc(repo, "toys")
    assert c["r2"].returncode == 0, FR.tail(c["r2"])
    c["s2"] = FR.snapshot(c["root"])
    return c


def clone(chain, tmp_path) -> Path:
    dest = tmp_path / "repo"
    shutil.copytree(chain["repo"], dest)
    return dest


def test_chain_layout_leaves_panels_and_the_registered_adapters_alone(chain):
    root = chain["root"]
    assert {p.name for p in (root / "panels" / "toys").iterdir()} == FR.expected_panels("toys")   # no train_perm there
    assert {p.name for p in (root / "ftc" / "toys").iterdir()} == {"train_perm.jsonl", "train_perm.manifest.json"}
    assert {p.name for p in (root / "adapters" / "toys").iterdir()} == {"s0", "s1", "s2", "p0", "p1"}
    got = {m.name: {a.name for a in m.iterdir() if "." not in a.name} for m in (root / "scores" / "toys").iterdir()}
    assert got == {m: {"like"} for m in ("zeroshot", "s0", "s1", "s2", "p0", "p1")}
    assert {p.name for p in (root / "report").iterdir()} == {"toys.json", "toys_tables.csv", "toys_ftc.json",
                                                              "toys_ftc_tables.csv"}
    assert (root / "report" / "toys.json").read_bytes() == SENTINEL                  # the registered report: untouched
    assert (root / "report" / "toys_tables.csv").read_bytes() == b"model,arm\n"
    man = read_json(root / "ftc" / "toys" / "train_perm.manifest.json")
    assert man["train"]["sha1"] == sha1_file(root / "panels" / "toys" / "train.jsonl")
    assert man["train_perm"]["sha1"] == sha1_file(root / "ftc" / "toys" / "train_perm.jsonl") and man["seed"] == 0
    assert man["n_labels_changed"] > 0 and all(v is True for v in man["checks"].values())


def test_chain_trains_p0_and_p1_with_the_recorded_recipe_of_s0(chain):
    root, r = chain["root"], chain["r"]
    assert FR.trained(r) == ["p0", "p1"]                                              # s0-s2 are never trained here
    s0 = read_json(root / "adapters" / "toys" / "s0" / "train_config.json")
    assert ("[recipe] trainer flags recorded by s0-s2 (--train, --out, --seed replaced): "
            + " ".join(fp.recipe_flags(s0))) in r.stdout                               # the log names the vector it used
    assert r.stdout.index("[recipe] scorer flags recorded") < r.stdout.index("dry-run trainer: ")   # the pre-flight comes first
    ov = read_json(root / "panels" / "toys" / "ftgrid_split.json")["train"]["overlength"]
    assert (s0["bsz"], s0["grad_accum"], s0["max_len"]) == (ov["micro_bsz"], ov["grad_accum"], ov["max_len_used"])
    manifest = read_json(root / "ftc" / "toys" / "train_perm.manifest.json")
    for k in (0, 1):
        a = root / "adapters" / "toys" / f"p{k}"
        cfg = read_json(a / "train_config.json")
        assert cfg["train"] == f"{DRY_ROOT}/ftc/toys/train_perm.jsonl" and cfg["seed"] == k
        assert cfg["out"] == f"{DRY_ROOT}/adapters/toys/p{k}"
        assert {x: v for x, v in cfg.items() if x not in fp.REPLACED} == {x: v for x, v in s0.items()
                                                                         if x not in fp.REPLACED}
        prov = read_json(a / "ftc.json")
        assert prov["train_perm_sha1"] == manifest["train_perm"]["sha1"] and prov["seed"] == k
        assert sha1_file(a / "adapter_model.safetensors") != sha1_file(root / "adapters" / "toys" / "s0" /
                                                                     "adapter_model.safetensors")


def test_chain_scores_the_p_like_passes_with_the_arguments_of_the_s0_like_pass(chain):
    repo, root = chain["repo"], chain["root"]
    s0_cfg = read_json(root / "scores" / "toys" / "s0" / "like" / "report.json")["config"]
    s0_key = (root / "scores" / "toys" / "s0" / "like" / "run.key").read_text(encoding="utf-8").split()
    assert ("(--data, --output, --model, --lora replaced): " + " ".join(fp.score_flags(s0_cfg))) in chain["r"].stdout
    for k in (0, 1):
        lora = f"{DRY_ROOT}/adapters/toys/p{k}"
        cfg = FR.check_scoring_dir(repo, "toys", f"p{k}", "like", lora)["config"]
        assert set(cfg) == set(s0_cfg) and {x for x in s0_cfg if s0_cfg[x] != cfg[x]} == {"lora"}
        key = (root / "scores" / "toys" / f"p{k}" / "like" / "run.key").read_text(encoding="utf-8").split()
        assert key[:3] == s0_key[:3] and key[4] == s0_key[4] == "--lora" and key[3] != s0_key[3] and key[5] != s0_key[5]
        assert not (root / "scores" / "toys" / f"p{k}" / "like" / "adopted_from").exists()


def test_chain_gates_run_first_and_the_record_is_rehearsed_then_recorded(chain):
    out, repo = chain["r"].stdout, chain["repo"]
    assert "[dry] gate rehearsal" in out and "freeze check OK (stage amendment)" in out
    assert "freeze check OK (stage core)" in out and "FT-C record OK" in out
    marks = [out.index(x) for x in ("freeze check OK (stage core)", "FT-C record OK", "== stage 1", "== stage 2",
                                    "== stage 3", "== stage 4")]
    assert marks == sorted(marks)
    log = (chain["root"] / "_dry" / "PILOT_LOG.md").read_text(encoding="utf-8").lower()
    for rel in FTC_LINES:
        assert f"{rel} = {sha1_file(repo / rel)}" in log


def test_chain_report_has_the_six_models_and_the_ftc_contrast_and_leaves_toys_json_alone(chain):
    root = chain["root"]
    rep = read_json(root / "report" / "toys_ftc.json")
    assert rep["meta"]["models_requested"] == ["zeroshot", "s0", "s1", "s2", "p0", "p1"] and rep["meta"]["domain"] == "toys"
    assert (rep["meta"]["n_boot"], rep["meta"]["seed"]) == (200, 0)                    # the DRY_RUN constants of run_ftgrid.sh
    assert (root / "report" / "toys_ftc_tables.csv").read_text(encoding="utf-8").startswith("domain,backbone,block")
    for m in ("p0", "p1"):
        assert rep["runs"][m]["like"]["status"] == "OK"
    assert not [e for e in rep["excluded_runs"] if e["model"] in ("p0", "p1")]
    assert rep["FT_C"]["descriptive"] is True
    for s in ("seed0", "seed1"):
        assert rep["FT_C"][s].get("available") is not False and "TEST" in rep["FT_C"][s], rep["FT_C"][s]
    assert "mean_over_seeds_TEST" in rep["FT_C"]


def test_chain_second_run_skips_everything_and_touches_nothing(chain):
    assert chain["s1"] == chain["s2"]
    out = chain["r2"].stdout
    assert "dry-run trainer" not in out and "scores chunk" not in out and out.count(": scored") == 2
    assert out.count("[skip] adapter") == 2 and "[skip] run_ftgrid.sh's DRY_RUN world for toys exists" in out
    for product in ("ftc/toys/train_perm.manifest.json", "report/toys_ftc.json"):
        assert f"[skip] {DRY_ROOT}/{product}" in out, product


def test_without_a_gate_ft_pass_no_gpu_stage_runs(chain, tmp_path):
    repo = clone(chain, tmp_path)
    (repo / DRY_ROOT / "_dry" / "gateft" / "gate_ft.json").write_text(json.dumps({"decision": "GATE_FT_FAIL"}),
                                                                       encoding="utf-8")
    before = FR.snapshot(repo / DRY_ROOT)
    for stages in ("all", "2", "3", "4"):
        r = run_ftc(repo, "toys", STAGES=stages)
        assert r.returncode == 4 and "Gate-FT decision GATE_FT_FAIL" in r.stderr, FR.tail(r)
        assert "== stage" not in r.stdout
    assert FR.snapshot(repo / DRY_ROOT) == before
    (repo / DRY_ROOT / "_dry" / "gateft" / "gate_ft.json").unlink()                  # no decision recorded at all
    r = run_ftc(repo, "toys", STAGES="3")
    assert r.returncode == 4 and "Gate-FT decision missing" in r.stderr, FR.tail(r)
    r = run_ftc(repo, "toys", STAGES="1")                                             # stage 1 is CPU and outcome-free
    assert r.returncode == 0 and "[skip]" in r.stdout and FR.snapshot(repo / DRY_ROOT) == before


def test_a_missing_s0_adapter_is_an_error_naming_it(chain, tmp_path):
    repo = clone(chain, tmp_path)
    shutil.rmtree(repo / DRY_ROOT / "adapters" / "toys" / "s0")
    before = FR.snapshot(repo / DRY_ROOT)
    r = run_ftc(repo, "toys", STAGES="2")
    assert r.returncode == 1 and "missing the Toys adapter s0" in r.stderr, FR.tail(r)
    r = run_ftc(repo, "toys", STAGES="3")
    assert r.returncode == 1 and "s0" in r.stderr, FR.tail(r)
    assert FR.snapshot(repo / DRY_ROOT) == before


def test_a_missing_or_stale_freeze_or_ftc_record_is_refused(chain, tmp_path):
    repo = clone(chain, tmp_path)
    log = repo / DRY_ROOT / "_dry" / "PILOT_LOG.md"
    original = log.read_text(encoding="utf-8")
    before = FR.snapshot(repo / DRY_ROOT)
    log.write_text("# emptied\n", encoding="utf-8")                                    # (a) no record at all
    r = run_ftc(repo, "toys", DRY_NO_RECORD="1")
    assert r.returncode == 4 and "[dry] gate rehearsal" in r.stdout and "stage amendment" in r.stderr, FR.tail(r)
    keep = [x for x in original.splitlines() if not x.startswith(FTC_LINES)]
    assert len(keep) == len(original.splitlines()) - 2
    log.write_text("\n".join(keep) + "\n", encoding="utf-8")                           # (b) only the FT-C record is missing
    r = run_ftc(repo, "toys", DRY_NO_RECORD="1")
    assert r.returncode == 4 and "FT-C record" in r.stderr and "stage core" not in r.stderr, FR.tail(r)
    log.write_text(original, encoding="utf-8")
    bound = repo / "src" / "confrec" / "metrics.py"                                     # (c) a bound file changed since
    bound.write_text(bound.read_text(encoding="utf-8") + "\n# changed after the freeze check\n", encoding="utf-8")
    r = run_ftc(repo, "toys")
    assert r.returncode == 4 and "stage core" in r.stderr, FR.tail(r)
    shutil.copy2(ROOT / "src" / "confrec" / "metrics.py", bound)
    script = repo / "scripts" / "sigir" / "run_ftc.sh"                                  # (d) an FT-C file changed since
    script.write_bytes(script.read_bytes() + b"\n# changed after its record\n")
    r = run_ftc(repo, "toys", DRY_NO_RECORD="1")
    assert r.returncode == 4 and "FT-C record" in r.stderr, FR.tail(r)
    assert "== stage" not in r.stdout
    assert {k: v for k, v in FR.snapshot(repo / DRY_ROOT).items()} == before


def test_an_adapter_trained_on_another_panel_or_by_hand_is_never_reused(chain, tmp_path):
    repo = clone(chain, tmp_path)
    root = repo / DRY_ROOT
    prov = root / "adapters" / "toys" / "p0" / "ftc.json"
    good = prov.read_text(encoding="utf-8")
    prov.unlink()                                                                       # trained by hand
    r = run_ftc(repo, "toys", STAGES="2")
    assert r.returncode == 1 and "has no ftc.json" in r.stderr, FR.tail(r)
    prov.write_text(good.replace(read_json_text(good)["train_perm_sha1"], "0" * 40), encoding="utf-8")
    r = run_ftc(repo, "toys", STAGES="2")                                               # trained on another permutation
    assert r.returncode == 1 and "was trained on a permuted panel" in r.stderr, FR.tail(r)
    prov.write_text(good, encoding="utf-8")
    perm = root / "ftc" / "toys" / "train_perm.jsonl"
    perm.write_bytes(perm.read_bytes() + b" ")                                          # the panel on disk is not the manifest's
    r = run_ftc(repo, "toys", STAGES="2")
    assert r.returncode == 1 and "is not the file" in r.stderr, FR.tail(r)


def read_json_text(text: str) -> dict:
    return json.loads(text)


def test_e1_failure_of_a_p_run_is_rerun_once_then_recorded_as_failed_integrity(chain, tmp_path):
    repo = clone(chain, tmp_path)
    root = repo / DRY_ROOT
    shutil.rmtree(root / "scores" / "toys" / "p0")
    r = run_ftc(repo, "toys", STAGES="3,4", DRY_E1_FAIL="p0/like")
    assert r.returncode == 0, FR.tail(r)
    like = root / "scores" / "toys" / "p0" / "like"
    assert (like / "FAILED_INTEGRITY").is_file() and len(list(like.parent.glob("like.e1fail.*"))) == 1
    assert r.stderr.count("[E1 failed]") == 1 and r.stderr.count("failed E1 twice") == 1
    assert r.stderr.count("NOTE: ") == 1 and "p0 is reported as missing, never replaced" in r.stderr
    assert not (root / "scores" / "toys" / "p1" / "like" / "FAILED_INTEGRITY").exists()
    rep = read_json(root / "report" / "toys_ftc.json")
    assert [e["status"] for e in rep["excluded_runs"] if e["model"] == "p0"] == ["FAILED_INTEGRITY"]
    assert rep["runs"]["p1"]["like"]["status"] == "OK" and rep["FT_C"]["seed0"]["available"] is False
    assert rep["FT_C"]["seed1"].get("available") is not False
    assert (root / "report" / "toys.json").read_bytes() == SENTINEL


def test_a_changed_run_key_moves_the_p_dir_aside_and_rescores_only_it(chain, tmp_path):
    repo = clone(chain, tmp_path)
    d = repo / DRY_ROOT / "scores" / "toys" / "p1" / "like"
    d.joinpath("run.key").write_text("an older key\n", encoding="utf-8")
    r = run_ftc(repo, "toys", STAGES="3")
    assert r.returncode == 0, FR.tail(r)
    assert f"[moved aside] {DRY_ROOT}/scores/toys/p1/like" in r.stdout and len(list(d.parent.glob("like.stale.*"))) == 1
    assert r.stdout.count(": scored") == 1
    FR.check_scoring_dir(repo, "toys", "p1", "like", f"{DRY_ROOT}/adapters/toys/p1")


# ---------------------------------------------------------------- 7. the registered text, the freeze blocks, the pre-flight
def test_the_registered_paths_of_addendum_6_are_the_scripts_paths():
    a6 = (ROOT / "idea-stage" / "PREREG_AMENDMENT_3_ADDENDUM_6.md").read_text(encoding="utf-8")
    for needle in ("outputs/confrec/ftgrid/ftc/toys/", "report/toys_ftc.json", "scripts/sigir/run_ftc.sh",
                   "src/confrec/ftc_panel.py", "ftgrid_report", "permuted adapters p0, p1"):
        assert needle in a6, needle
    code = SCRIPT.read_text(encoding="utf-8")
    for needle in ("REG=outputs/confrec/ftgrid ", 'FTC="$OUT_ROOT/ftc/toys"', 'REP="$OUT_ROOT/report/toys_ftc.json"',
                   "--models zeroshot,s0,s1,s2,p0,p1", '--out "$REP"', 'PERM="$FTC/train_perm.jsonl"',
                   'PMAN="$FTC/train_perm.manifest.json"', "--seed 0"):
        assert needle in code, needle
    assert fp.PERM_SEEDS == (0, 1) and fp.SEED == 0 and code.count("for seed in 0 1;") == 2   # p0, p1 (Amendment 3 section 9)


def test_the_ftc_files_are_new_files_outside_every_freeze_block():
    """No bound file changes: the files of this task are not in any FREEZE_FILES block of Amendment 3 or its addenda."""
    texts = [ROOT / FR.ff.AMENDMENT] + sorted((ROOT / "idea-stage").glob("PREREG_AMENDMENT_3_ADDENDUM_*.md"))
    listed = {f for p in texts for files in FR.ff.parse_blocks(p.read_text(encoding="utf-8")).values() for f in files}
    assert listed and not listed & {"scripts/sigir/run_ftc.sh", "src/confrec/ftc_panel.py", "tests/test_confrec_ftc.py"}
    assert {"src/confrec/ftgrid_data.py", "scripts/sigir/run_ftgrid.sh"} <= listed          # the bound files it imports / mirrors


def test_a_wrong_recorded_input_stops_the_job_before_the_first_training(chain, tmp_path):
    repo = clone(chain, tmp_path)
    root = repo / DRY_ROOT
    for k in (0, 1):
        shutil.rmtree(root / "adapters" / "toys" / f"p{k}")
    rep = root / "scores" / "toys" / "s0" / "like" / "report.json"
    data = read_json(rep)
    data["config"]["chunk_users"] = 50                                  # the s0 like pass is not the registered one
    rep.write_text(json.dumps(data), encoding="utf-8")
    r = run_ftc(repo, "toys", STAGES="2")
    assert r.returncode == 1 and "chunk_users" in r.stderr, FR.tail(r)
    assert "dry-run trainer" not in r.stdout and not (root / "adapters" / "toys" / "p0").exists()
    rep.write_text(json.dumps(read_json(chain["root"] / "scores" / "toys" / "s0" / "like" / "report.json")), encoding="utf-8")
    cfg1 = root / "adapters" / "toys" / "s1" / "train_config.json"       # s1 was trained with another recipe than s0
    cfg1.write_text(json.dumps({**read_json(cfg1), "lr": 5e-05}), encoding="utf-8")
    r = run_ftc(repo, "toys", STAGES="2")
    assert r.returncode == 1 and "not trained with one recipe" in r.stderr, FR.tail(r)
    assert "dry-run trainer" not in r.stdout and not (root / "adapters" / "toys" / "p0").exists()


# ---------------------------------------------------------------- 8. the script mirrors run_ftgrid.sh
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
    for name in ("jget", "fresh", "step", "adapter_done", "e1_ok"):
        assert shell_function(mine, name) == shell_function(theirs, name), name
    needles = ('key="$(', 'wsha=$(', "run.key", "$dir.stale", "$dir.e1fail", "FAILED_INTEGRITY", 'e1_ok "$dir"',
               'score "$data" "$dir" "$@"', "compgen -G")
    keep = lambda text: [ln for ln in shell_function(text, "score").splitlines() if any(n in ln for n in needles)]
    assert keep(mine) == keep(theirs) and len(keep(mine)) >= 10                # key, skip, stale, E1 rerun-once, FAILED
    # the freeze checks are the same commands, the gate-fix rule and the Gate-FT rule the same tests
    for line in ('  if [ "$SEL_DECISION" = FIX_FOUND ] && [ ! -f "$G/confirm/gate.json" ]; then',
                 'SEL="$G/dev/selection.json"', 'SEL_PROMPT=$(jget "$SEL" gate_ft_prompt)',
                 'VARIANT="${VARIANT:-$SEL_PROMPT}"'):
        assert line in theirs.replace("    echo", "  echo") or line.strip() in theirs, line
        assert line.strip() in mine, line
    for exit_rule in ("exit 4", "exit 2", "exit 1"):
        assert exit_rule in mine and exit_rule in theirs


def test_a_later_stage_without_its_earlier_stage_is_refused_and_writes_nothing(chain, tmp_path):
    repo = clone(chain, tmp_path)
    root = repo / DRY_ROOT
    shutil.rmtree(root / "ftc" / "toys")                                              # stage 1 not done
    before = FR.snapshot(root)
    for stages, msg in (("2", "(stage 1)"), ("3", "this script's stage 1")):
        r = run_ftc(repo, "toys", STAGES=stages)
        assert r.returncode == 1 and "train_perm" in r.stderr and msg in r.stderr, FR.tail(r)
    assert FR.snapshot(root) == before
    r = run_ftc(repo, "toys", STAGES="1")                                             # ... and stage 1 rebuilds it
    assert r.returncode == 0 and (root / "ftc" / "toys" / "train_perm.manifest.json").is_file(), FR.tail(r)
    assert (root / "ftc" / "toys" / "train_perm.jsonl").read_bytes() == (
        chain["root"] / "ftc" / "toys" / "train_perm.jsonl").read_bytes()
    shutil.rmtree(root / "adapters" / "toys" / "p0")                                  # stage 2 not done for p0
    shutil.rmtree(root / "scores" / "toys" / "p1")                                    # stage 3 not done for p1
    before = FR.snapshot(root)
    r = run_ftc(repo, "toys", STAGES="3")
    assert r.returncode == 1 and "adapter p0 of toys is missing or incomplete" in r.stderr, FR.tail(r)
    r = run_ftc(repo, "toys", STAGES="4")
    assert r.returncode == 1 and "missing the like pass" in r.stderr and "p1/like" in r.stderr, FR.tail(r)
    assert FR.snapshot(root) == before
