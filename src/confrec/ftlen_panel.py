"""FT-L, the longer-training robustness arm: the recorded recipe, the recorded scoring arguments of two arms, the control-root
plumbing and the equality checks behind scripts/sigir/run_ftlen.sh (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_9.md; exploratory: no
hypothesis, no Holm family). CPU only, no torch. It adds nothing to ftq_panel.py and changes nothing in it: what FT-L shares with FT-Q
(the recipe validation, the run.key and report-config readers, the link plumbing, the pilot-log record) is imported from it, read-only.

    python -m src.confrec.ftlen_panel recipe --adapters outputs/confrec/ftgrid/adapters/toys --model M --variant V \
        --train_ref outputs/confrec/ftgrid/panels/toys/train.jsonl --train_file outputs/confrec/ftgrid/panels/toys/train.jsonl \
        --split outputs/confrec/ftgrid/panels/toys/ftgrid_split.json --split_recipe
    python -m src.confrec.ftlen_panel scoring --arm like|swap --scores outputs/confrec/ftgrid/scores/toys \
        --adapters outputs/confrec/ftgrid/adapters/toys --data outputs/confrec/ftgrid/panels/toys/eval.jsonl|eval_sd_test.jsonl \
        --model M --variant V
    python -m src.confrec.ftlen_panel verify_adapter --adapters outputs/confrec/ftgrid_len3/adapters/toys \
        --ref_adapters outputs/confrec/ftgrid/adapters/toys --seeds 0,1 --train_file outputs/confrec/ftgrid/panels/toys/train.jsonl \
        [--write]
    python -m src.confrec.ftlen_panel verify_scores --arm like|swap --scores outputs/confrec/ftgrid_len3/scores/toys \
        --ref_scores outputs/confrec/ftgrid/scores/toys --adapters outputs/confrec/ftgrid_len3/adapters/toys --seeds 0
    python -m src.confrec.ftlen_panel link --real_scores outputs/confrec/ftgrid/scores/toys \
        --q_scores outputs/confrec/ftgrid_len3/scores/toys --models zeroshot [--allow_copy]
    python -m src.confrec.ftlen_panel record --pilot_log docs/sigir/PILOT_LOG.md --files scripts/sigir/run_ftlen.sh \
        src/confrec/ftlen_panel.py tests/test_confrec_ftlen.py [--print]
    python -m src.confrec.ftlen_panel info --selection .../selection.json --split .../ftgrid_split.json --gate .../gate_ft.json

recipe          the trainer flags of FT-L: every argument the real adapters s0-s2 recorded in their train_config.json (ML-1M: Gate-FT's,
                outputs/confrec/gateft/adapters), with --train kept as recorded (the real-label registered TRAIN panel, never a
                teacher or a permutation), `--epochs 3` in place of the recorded 1.0, and --out and --seed left to the caller (one
                token per line). s0, s1 and s2 must agree on every argument and on the registered constants of Amendment 3 section 2
                (ftq_panel.check_recipe: standard mode, 1 epoch, lr 1e-4, r 16, an effective batch of 32, and with --split_recipe the
                split's TRAIN length rule); their recorded TRAIN file must hold the bytes of --train_file.
scoring         the scorer flags of the real s0's pass of the arm (run.key and the config of its report.json: nothing else of that
                report is read), except --data, --output, --model and --lora: `like` is scored on eval.jsonl (CAL and TEST rows) and
                `swap` (--swap_k 8, which stays an extra argument of the call) on eval_sd_test.jsonl (S_d's TEST rows), exactly as the
                registered scoring stages score the registered adapters; the record must be the registered pass of that arm.
verify_adapter  after training: the recorded arguments of each s<k> equal the real s0's except --out, --seed and --epochs (3.0), and
                --train is the real adapters' own; --write records provenance (ftlen.json: s<k>, 3 epochs, the sha1 of the TRAIN panel);
                without --write an existing adapter must carry a matching record, so a registered 1-epoch adapter placed in this root,
                or one trained by hand, is never taken for an FT-L adapter.
verify_scores   after scoring: the recorded config of each s<k> pass of the arm equals that of the real s0's pass of the same arm
                except `lora`.
link            scores/<d>/zeroshot of the FT-L root becomes a relative symlink to the registered zero-shot scores, never a copy;
                --allow_copy (DRY_RUN only) falls back to a copy where the platform has no symlinks (a Windows machine).
record          0 iff the pilot log holds the sha1 of every listed file (addendum 9 section 3: recorded before the first adapter is
                trained); --print writes the lines to paste into the log.
info            decision and gate_ft_prompt of selection.json, the variant of the split and the decision of gate_ft.json.
Exit codes: 0 done; 1 error or an inconsistent record; 2 refused input; 4 refused by a registered gate (record).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.confrec import ftq_panel as fq
from src.confrec.ftq_panel import FtqError, file_sha1, json_bytes, read_json
from src.confrec.prompting import resolve_hist_len

SPEC = "idea-stage/PREREG_AMENDMENT_3_ADDENDUM_9.md (FT-L, exploratory)"
DOMAINS = ("ml1m", "toys")                  # addendum 9 section 1
EPOCHS = 3                                  # the one change against the registered recipe (--epochs 3)
SEEDS = (0, 1)                              # two adapters per dataset, named s0 and s1
SWAP_K = 8                                  # the swap-prior arm of the registered decomposition stage (--swap_k 8)
ARMS = ("like", "swap")
ARM_DATA = {"like": "eval.jsonl", "swap": "eval_sd_test.jsonl"}
PROVENANCE_NAME = "ftlen.json"
LINKED_MODELS = ("zeroshot",)
COMMANDS = ("recipe", "scoring", "verify_adapter", "verify_scores", "link", "record", "info")
FtlenError = FtqError


# ---------------------------------------------------------------- the recorded training recipe
def len_flags(base: dict) -> list:
    """The trainer flags of FT-L: every flag of the real adapters' recorded config (a whole train_config.json), --train kept as
    recorded, `--epochs 3` for the recorded epochs, --out and --seed left out (the caller's); an unset optional flag is left out."""
    a = fq.train_args(base, "the real adapter s0's train_config.json")
    out = ["--train", str(a["train"])]
    for k in fq.TRAIN_KEYS:
        if k in ("train", "out", "seed") or a[k] is None:
            continue
        out += [f"--{k}", str(EPOCHS) if k == "epochs" else fq.flag_value(a[k])]
    return out


def len_argv(base: dict, *, out: str, seed: int) -> list:
    """The trainer's whole argument vector for the adapter at `out`, as run_ftlen.sh passes it."""
    return ["--out", str(out), "--seed", str(int(seed))] + len_flags(base)


def check_len_training(ref: dict, run: dict, *, out: str, seed: int, where: str = "train_config.json") -> None:
    """A trained FT-L adapter: its recorded arguments equal the real s0's except --out, --seed (the given ones) and --epochs, which
    is 3; --train is the real adapters' own; the prompt-determined facts agree."""
    a, b = fq.train_args(ref, "the real adapter s0's train_config.json"), fq.train_args(run, where)
    for key, want in (("out", str(out)), ("seed", int(seed)), ("train", a["train"])):
        if b[key] != want:
            raise FtqError(f"{where}: {key} is {b[key]!r}, expected {want!r}")
    if b["epochs"] != float(EPOCHS):
        raise FtqError(f"{where}: epochs is {b['epochs']!r}, FT-L trains for {EPOCHS} (a registered 1-epoch adapter is not an "
                       "FT-L adapter)")
    diff = [k for k in fq.TRAIN_KEYS if k not in ("out", "seed", "train", "epochs") and a[k] != b[k]]
    diff += [k for k in fq.TRAIN_FACTS if ref.get(k) != run.get(k)]
    if diff:
        raise FtqError(f"{where} differs from the recorded recipe of the real s0 in {diff} (only --out, --seed and --epochs may)")


# ---------------------------------------------------------------- provenance of an adapter
def write_provenance(adapter_dir, *, seed: int, train_sha1: str, args: dict) -> None:
    rec = {"purpose": "FT-L adapter (3 epochs, exploratory) trained by scripts/sigir/run_ftlen.sh", "adapter": f"s{seed}",
           "seed": seed, "epochs": EPOCHS, "train_sha1": train_sha1, "args": args,
           "code_sha1": {"ftlen_panel.py": file_sha1(__file__)}}
    (Path(adapter_dir) / PROVENANCE_NAME).write_bytes(json_bytes(rec))


def check_provenance(adapter_dir, *, seed: int, train_sha1: str) -> None:
    path = Path(adapter_dir) / PROVENANCE_NAME
    if not path.is_file():
        raise FtqError(f"{adapter_dir} has no {PROVENANCE_NAME}: it was not trained by run_ftlen.sh (a registered adapter, or one "
                       "trained by hand); move it aside and rerun stage 1")
    rec = read_json(path)
    if rec.get("adapter") != f"s{seed}" or rec.get("epochs") != EPOCHS:
        raise FtqError(f"{adapter_dir} records adapter {rec.get('adapter')!r} with {rec.get('epochs')!r} epochs, not s{seed} with "
                       f"{EPOCHS}: move it aside and rerun stage 1")
    if rec.get("train_sha1") != train_sha1:
        raise FtqError(f"{adapter_dir} was trained on a TRAIN file with sha1 {rec.get('train_sha1')}, the registered train.jsonl "
                       f"has {train_sha1}: move the adapter aside and rerun stage 1")


# ---------------------------------------------------------------- the recorded scoring of the two arms
def check_arm_record(arm: str, key_text: str, cfg: dict, *, data_sha1: str, model: str, variant: str, lora: str,
                     lora_weights_sha1: str) -> list:
    """Validate the real s0's pass of the arm (its run.key and report config): returns the scorer flags FT-L passes. The registered
    passes are run_ftgrid.sh's score() calls: `--lora A` on eval.jsonl for like, `--lora A --swap_k 8` on eval_sd_test.jsonl for swap."""
    if arm not in ARMS:
        raise FtqError(f"unknown arm {arm!r} (FT-L scores {', '.join(ARMS)})", 2)
    what_pass = f"the {arm} pass of s0"
    want_args = ["--lora", lora] + (["--swap_k", str(SWAP_K)] if arm == "swap" else [])
    key = fq.parse_run_key(key_text)
    for what, got, want in (("panel sha1", key["panel_sha1"], data_sha1), ("model", key["model"], model),
                            ("variant", key["variant"], variant),
                            ("adapter weights sha1", key["weights_sha1"], lora_weights_sha1), ("arguments", key["args"], want_args)):
        if got != want:
            raise FtqError(f"run.key of {what_pass}: {what} is {got!r}, not {want!r} (the panel or the adapter changed since the "
                           "pass, or it is not the registered pass)")
    for what, want in (("data_sha1", data_sha1), ("model", model), ("variant", variant), ("lora", lora)):
        if cfg.get(what) != want:
            raise FtqError(f"report.json config of {what_pass}: {what} is {cfg.get(what)!r}, not {want!r}")
    for what, want in {**fq.SCORE_REGISTERED, "swap_k": SWAP_K if arm == "swap" else 0}.items():
        if cfg.get(what) != want:
            raise FtqError(f"report.json config of {what_pass}: {what} is {cfg.get(what)!r}; the registered {arm} pass has "
                           f"{want!r} (run_ftgrid.sh score())")
    try:
        window = resolve_hist_len(variant, str(cfg.get("panel_kind")), None)
    except ValueError as e:
        raise FtqError(f"report.json config of {what_pass}: {e}") from None
    if cfg.get("hist_len") != window:
        raise FtqError(f"report.json config of {what_pass}: hist_len {cfg.get('hist_len')!r}, the registered window of "
                       f"{variant} is {window}")
    return fq.score_flags(cfg)


def check_same_arm_scoring(arm: str, ref_cfg: dict, run_cfg: dict, *, lora: str, where: str) -> None:
    """A scored s<k>: its recorded config equals that of the real s0's pass of the same arm except `lora` (the given adapter)."""
    if run_cfg.get("lora") != lora:
        raise FtqError(f"{where}: scored with {run_cfg.get('lora')!r}, not {lora!r}")
    diff = sorted(k for k in set(ref_cfg) | set(run_cfg) if k != "lora" and ref_cfg.get(k) != run_cfg.get(k))
    if diff:
        raise FtqError(f"{where}: the recorded scoring arguments differ from the {arm} pass of the real s0 in {diff}")


# ---------------------------------------------------------------- command line
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("command", choices=list(COMMANDS))
    ap.add_argument("--arm", default=None, choices=list(ARMS), help="scoring, verify_scores: like or swap")
    ap.add_argument("--split", default=None, help="ftgrid_split.json of the dataset (info: also read)")
    ap.add_argument("--split_recipe", action="store_true", help="recipe: also compare with the split's TRAIN length rule")
    ap.add_argument("--adapters", default=None, help="recipe / scoring: the real adapters' directory (s0-s2); verify_*: the FT-L "
                                                     "root's adapters/<d>/ (s<k>)")
    ap.add_argument("--ref_adapters", default=None, help="verify_adapter: the real adapters' directory (s0's config)")
    ap.add_argument("--scores", default=None, help="scoring: the real scores/<d>/ ; verify_scores: the FT-L root's scores/<d>/")
    ap.add_argument("--ref_scores", default=None, help="verify_scores: the real scores/<d>/ (the pass of s0)")
    ap.add_argument("--real_scores", default=None, help="link: the real scores/<d>/")
    ap.add_argument("--q_scores", default=None, help="link: the FT-L root's scores/<d>/")
    ap.add_argument("--models", default=",".join(LINKED_MODELS), help="link: the models to link")
    ap.add_argument("--allow_copy", action="store_true", help="link: DRY_RUN only, copy where symlinks are unavailable")
    ap.add_argument("--model", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--train_ref", default=None, help="recipe: the TRAIN file path the real adapters recorded")
    ap.add_argument("--train_file", default=None, help="recipe, verify_adapter: the registered panels/<d>/train.jsonl")
    ap.add_argument("--data", default=None, help="scoring: panels/<d>/eval.jsonl (like) or eval_sd_test.jsonl (swap)")
    ap.add_argument("--seeds", default=None, help="verify_*: comma list of the FT-L adapters (0,1)")
    ap.add_argument("--write", action="store_true", help="verify_adapter: write the adapters' provenance records")
    ap.add_argument("--selection", default=None, help="info: selection.json")
    ap.add_argument("--gate", default=None, help="info: gate_ft.json")
    ap.add_argument("--pilot_log", default=None, help="record: docs/sigir/PILOT_LOG.md")
    ap.add_argument("--files", nargs="+", default=None, help="record: repo-relative files whose sha1 the log must hold")
    ap.add_argument("--root", default=None, help="record: the repo root (default: this checkout)")
    ap.add_argument("--print", dest="do_print", action="store_true", help="record: print the lines for the pilot log")
    ap.add_argument("--append", action="store_true", help="record: DRY_RUN only, append the missing lines to the (temporary) pilot "
                                                          "log, the human step of the rehearsal")
    return ap.parse_args(argv)


def need(a: argparse.Namespace, *names: str) -> None:
    missing = [f"--{n}" for n in names if getattr(a, n) is None]
    if missing:
        raise FtqError(f"{a.command} needs {' '.join(missing)}", 2)


def seeds_of(a: argparse.Namespace) -> list:
    try:
        seeds = [int(x) for x in str(a.seeds).split(",") if x != ""]
    except ValueError:
        raise FtqError(f"--seeds {a.seeds!r} is not a comma list of integers", 2) from None
    bad = [s for s in seeds if s not in SEEDS]
    if not seeds or bad:
        raise FtqError(f"--seeds {a.seeds!r}: the FT-L adapters are s0 and s1", 2)
    return seeds


def cmd_recipe(a) -> list:
    need(a, "adapters", "model", "variant", "train_ref")
    adapters = a.adapters.rstrip("/")
    split = read_json(a.split) if a.split else None
    base = fq.check_recipe(fq.sft_configs(adapters), adapters=adapters, model=a.model, variant=a.variant, train_ref=a.train_ref,
                           split=split, split_recipe=a.split_recipe)
    if a.train_file:
        fq.check_train_file(a.train_ref, a.train_file)
    return len_flags(base)


def cmd_scoring(a) -> list:
    need(a, "arm", "scores", "adapters", "data", "model", "variant")
    scores, adapters = a.scores.rstrip("/"), a.adapters.rstrip("/")
    ref = Path(scores) / "s0" / a.arm
    for name in ("run.key", "report.json"):
        if not (ref / name).is_file():
            raise FtqError(f"missing the real {a.arm} pass of s0 ({ref / name}): run_ftgrid.sh scores it, FT-L reads its "
                           "arguments")
    cfg = fq.report_config(read_json(ref / "report.json"), str(ref / "report.json"))
    return check_arm_record(a.arm, (ref / "run.key").read_text(encoding="utf-8"), cfg, data_sha1=file_sha1(a.data), model=a.model,
                            variant=a.variant, lora=f"{adapters}/s0", lora_weights_sha1=fq.weights_sha1(Path(adapters) / "s0"))


def cmd_verify_adapter(a) -> None:
    need(a, "adapters", "ref_adapters", "seeds", "train_file")
    adapters = a.adapters.rstrip("/")
    ref = read_json(Path(a.ref_adapters.rstrip("/")) / "s0" / "train_config.json")
    train_sha1 = file_sha1(a.train_file)
    for k in seeds_of(a):
        out = f"{adapters}/s{k}"
        cfg = read_json(Path(out) / "train_config.json")
        check_len_training(ref, cfg, out=out, seed=k, where=f"{out}/train_config.json")
        if a.write:
            write_provenance(out, seed=k, train_sha1=train_sha1, args=fq.train_args(cfg))
        check_provenance(out, seed=k, train_sha1=train_sha1)


def cmd_verify_scores(a) -> None:
    need(a, "arm", "scores", "ref_scores", "adapters", "seeds")
    scores, adapters = a.scores.rstrip("/"), a.adapters.rstrip("/")
    ref = fq.report_config(read_json(Path(a.ref_scores.rstrip("/")) / "s0" / a.arm / "report.json"),
                           f"the {a.arm} pass of the real s0")
    for k in seeds_of(a):
        run = Path(scores) / f"s{k}" / a.arm / "report.json"
        check_same_arm_scoring(a.arm, ref, fq.report_config(read_json(run), str(run)), lora=f"{adapters}/s{k}", where=str(run))


def cmd_link(a) -> list:
    need(a, "real_scores", "q_scores")
    try:
        return fq.link_models(a.real_scores, a.q_scores, [m for m in a.models.split(",") if m], allow_copy=a.allow_copy)
    except FtqError as e:                          # the shared helper speaks of "the FT-Q root": here it is the FT-L root
        raise FtqError(str(e).replace("FT-Q", "FT-L"), e.code) from None


def cmd_record(a) -> int:
    need(a, "pilot_log", "files")
    if a.do_print:
        print("\n".join(fq.record_lines(a.files, a.root)))
        return 0
    missing = fq.record_missing(a.pilot_log, a.files, a.root)
    if missing and a.append:
        log = Path(a.pilot_log)
        lead = b"" if not log.stat().st_size or log.read_bytes().endswith(b"\n") else b"\n"
        with open(log, "ab") as f:
            f.write(lead + ("\n".join(fq.record_lines(missing, a.root)) + "\n").encode("utf-8"))
        print(f"[dry] recorded {', '.join(missing)} in {log} (the human step, on the temporary pilot log)")
        missing = fq.record_missing(a.pilot_log, a.files, a.root)
    if missing:
        raise FtqError(f"FT-L record (Amendment 3 addendum 9 section 3): the sha1 of these files is not in {a.pilot_log}: "
                       f"{missing}; run `python -m src.confrec.ftlen_panel record --pilot_log {a.pilot_log} --files "
                       f"{' '.join(a.files)} --print`, record the lines in the pilot log and push it", 4)
    print(f"FT-L record OK: the sha1 of {', '.join(a.files)} is in {a.pilot_log}")
    return 0


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        if a.command == "recipe":
            print("\n".join(cmd_recipe(a)))
        elif a.command == "scoring":
            print("\n".join(cmd_scoring(a)))
        elif a.command == "verify_adapter":
            cmd_verify_adapter(a)
            print(f"adapters s{a.seeds}: the recorded arguments equal the real s0's except --out, --seed and --epochs ({EPOCHS}); "
                  "provenance OK")
        elif a.command == "verify_scores":
            cmd_verify_scores(a)
            print(f"{a.arm} pass of s{a.seeds}: the recorded scoring arguments equal the real s0's {a.arm} pass except lora")
        elif a.command == "link":
            print("\n".join(cmd_link(a)))
        elif a.command == "info":
            print("\n".join(fq.cmd_info(a)))
        else:
            return cmd_record(a)
    except FtqError as e:
        print(f"ftlen_panel {a.command}: {e}", file=sys.stderr)
        return e.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
