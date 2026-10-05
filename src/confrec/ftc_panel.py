"""FT-C on Toys: the within-item permuted TRAIN panel, the recorded recipe and the equality checks behind
scripts/sigir/run_ftc.sh (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_6.md item 8; the control of Amendment 3 section 9).
CPU only, no torch; every bound module is imported read-only and none is changed.

    python -m src.confrec.ftc_panel build --train outputs/confrec/ftgrid/panels/toys/train.jsonl \
        --out_dir outputs/confrec/ftgrid/ftc/toys --split outputs/confrec/ftgrid/panels/toys/ftgrid_split.json [--seed 0]
    python -m src.confrec.ftc_panel recipe --adapters outputs/confrec/ftgrid/adapters/toys --model M --variant V \
        --train_ref outputs/confrec/ftgrid/panels/toys/train.jsonl [--split .../ftgrid_split.json]
    python -m src.confrec.ftc_panel scoring --scores outputs/confrec/ftgrid/scores/toys --adapters .../adapters/toys \
        --data .../panels/toys/eval.jsonl --model M --variant V
    python -m src.confrec.ftc_panel verify_adapter --adapters .../adapters/toys --p_seed 0 --train .../ftc/toys/train_perm.jsonl \
        --perm_manifest .../ftc/toys/train_perm.manifest.json [--write]
    python -m src.confrec.ftc_panel verify_scores --scores .../scores/toys --adapters .../adapters/toys --p_seed 0
    python -m src.confrec.ftc_panel record --pilot_log docs/sigir/PILOT_LOG.md --files scripts/sigir/run_ftc.sh \
        src/confrec/ftc_panel.py [--print]

build           train_perm.jsonl = ftgrid_data.permute_within_item(the rows of train.jsonl, seed 0): the function and the
                writer (ftgrid_data.row_bytes) that build the ML-1M file of section 9, imported and never copied. The
                input must be the registered train.jsonl (its sha1 is the one ftgrid_split.json records, and every line is
                what ftgrid_data writes). Asserted twice, on the rows and again on the bytes about to be written (parsed
                back): every item's label sum and rating multiset (indeed its multiset of (label, rating) pairs) is
                preserved, a single-example item is unchanged, and nothing but candidate_labels / candidate_ratings differs
                from train.jsonl. Writes train_perm.jsonl and train_perm.manifest.json (sha1 of both files, number of
                labels changed, seed, sha1 of the code; strict JSON, no clock, host or path, so a rerun is byte-identical).
recipe          the trainer flags of the recorded Toys adapters s0-s2: every key that train_lora_yesno's train_config.json
                records as an argument, except --train, --out and --seed (printed one token per line; an unset optional
                flag is left out, which is the trainer's default again). s0, s1 and s2 must agree on every argument, on
                the registered constants of Amendment 3 section 2 and (with --split) on the split's recipe.
scoring         the scorer flags of the recorded like pass of s0 (run.key and the config of its report.json: nothing
                else of that report is read), except --data, --output, --model and --lora. The record must be the
                registered like pass of the Toys adapter s0 on this eval.jsonl.
verify_adapter  after training: the recorded arguments of p<k> equal those of s0 except --train, --out and --seed (and the
                prompt-determined facts n_examples and n_skipped_overlength agree); --write records provenance
                (ftc.json: the sha1 of the permuted panel it was trained on); without --write an existing adapter must
                carry a matching record, so an adapter trained on another panel is never reused.
verify_scores   after scoring: the recorded config of p<k>'s like pass equals that of s0's like pass except `lora`.
record          0 iff the pilot log holds the sha1 of every listed file (Addendum 6 section 11: recorded before the Toys
                permuted adapters are trained); --print writes the lines to paste into the log.
Exit codes: 0 done; 1 error or an inconsistent record; 2 refused input; 4 refused by a registered gate (record).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from src.confrec import build_rated_panels as brp
from src.confrec import ftgrid_data as fd
from src.confrec.prompting import resolve_hist_len
from src.confrec.stats import strict_json

SPEC = "idea-stage/PREREG_AMENDMENT_3_ADDENDUM_6.md item 8; idea-stage/PREREG_AMENDMENT_3.md section 9"
SEED = 0                                    # the permutation seed of section 9 (the ML-1M file's)
PERM_NAME, MANIFEST_NAME, PROVENANCE_NAME = "train_perm.jsonl", "train_perm.manifest.json", "ftc.json"
SFT_SEEDS, PERM_SEEDS = (0, 1, 2), (0, 1)   # the adapters s0-s2 whose recipe FT-C uses; the FT-C adapters p0, p1
REPLACED = ("train", "out", "seed")         # the only arguments FT-C changes
# everything train_lora_yesno's main() records as vars(args) in train_config.json (its argparse destinations)
TRAIN_KEYS = ("train", "model", "out", "mode", "variant", "seed", "hist_len", "max_len", "epochs", "lr", "bsz",
              "grad_accum", "lora_r", "max_examples")
# facts of train_config.json that the prompts (not the labels) determine: equal for a permuted panel of the same rows
TRAIN_FACTS = ("loss", "hist_len_used", "panel_kind", "max_history_len_in_panel", "n_examples", "n_skipped_overlength")
# Amendment 3 section 2 (LoRA r 16, lr 1e-4, one epoch, standard mode) and the trainer's defaults that run_ftgrid.sh never
# overrides (the variant's registered history window, no example cap): what the recorded s0-s2 must be
TRAIN_REGISTERED = {"mode": "standard", "hist_len": None, "epochs": 1.0, "lr": 1e-4, "lora_r": 16, "max_examples": None}
EFFECTIVE_BATCH = 32                        # micro-batch x accumulation (Amendment 3 section 2)
# the like pass of run_ftgrid.sh's score(): fp16, top-50 logprobs, max_model_len 4096, 100-user chunks, yes/no readout,
# the question `like`, no swap prior, no user cap, the scorer's default seed and chunk_items
SCORE_REGISTERED = {"questions": ["like"], "swap_k": 0, "seed": 0, "dtype": "float16", "topk_logprobs": 50,
                    "max_model_len": 4096, "chunk_users": 100, "chunk_items": 1000, "readout": "yesno", "n_users": None}
COMMANDS = ("build", "recipe", "scoring", "verify_adapter", "verify_scores", "record")


class FtcError(Exception):
    """A refused or inconsistent input: the message goes to stderr and `code` is the exit status."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def file_sha1(path) -> str:
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError as e:
        raise FtcError(f"cannot read {path}: {e}") from None
    return h.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise FtcError(f"cannot read {path}: {e}") from None


def weights_sha1(adapter_dir) -> str:
    """The sha1 run_ftgrid.sh's score() puts in run.key: `cat DIR/adapter_model.* | sha1sum`."""
    parts = sorted(Path(adapter_dir).glob("adapter_model.*"))
    if not parts:
        raise FtcError(f"{adapter_dir}: no adapter_model.* weights")
    h = hashlib.sha1()
    for p in parts:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- the permuted panel
def read_rows(path) -> tuple[list, list]:
    """([line bytes], [row]) of a panel file; blank lines are not rows (as ftgrid_data.read_panel)."""
    lines, rows = [], []
    try:
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    lines.append(line)
                    rows.append(json.loads(line))
    except (OSError, ValueError) as e:
        raise FtcError(f"cannot read {path}: {e}") from None
    return lines, rows


def moved_fields(rows: list) -> tuple:
    """The per-candidate fields the permutation moves (ftgrid_data.MOVED; the ratings only if every row has them)."""
    has = ["candidate_ratings" in r for r in rows]
    if any(has) and not all(has):
        raise FtcError("candidate_ratings is present in some rows only")
    return fd.MOVED if all(has) else fd.MOVED[:1]


def verify_permutation(train_rows: list, perm_rows: list) -> dict:
    """The registered assertions of section 9, recomputed from the two row lists with no help from the permuting function:
    same rows in the same order with the same keys; nothing but the moved fields differs; every item keeps its label sum
    and its multiset of (label, rating) pairs; an item with one example is unchanged. Raises FtcError on a violation;
    returns the counts the manifest records."""
    if len(train_rows) != len(perm_rows):
        raise FtcError(f"the permuted panel has {len(perm_rows)} rows, train.jsonl {len(train_rows)}")
    moved = moved_fields(train_rows)
    where = defaultdict(list)                      # item -> [(row, candidate)]
    for k, (r, p) in enumerate(zip(train_rows, perm_rows)):
        if list(r) != list(p):
            raise FtcError(f"row {k}: the permuted row has other keys than train.jsonl's")
        changed = [f for f in r if r[f] != p[f]]
        if any(f not in moved for f in changed):
            raise FtcError(f"row {k}: {[f for f in changed if f not in moved]} changed (only {list(moved)} may move)")
        if any(len(p[f]) != len(r[f]) for f in moved):
            raise FtcError(f"row {k}: a candidate list changed its length")
        for j, it in enumerate(r["candidate_item_ids"]):
            where[str(it)].append((k, j))

    def pair(rows, k, j):
        return (int(rows[k]["candidate_labels"][j]),) + tuple(rows[k][f][j] for f in moved[1:])

    n_changed = n_single = n_mixed = n_multi = 0
    for item, pos in where.items():
        before = [pair(train_rows, k, j) for k, j in pos]
        after = [pair(perm_rows, k, j) for k, j in pos]
        if sum(x[0] for x in before) != sum(x[0] for x in after):
            raise FtcError(f"item {item}: the label sum is not preserved")
        if sorted(before) != sorted(after):
            raise FtcError(f"item {item}: the multiset of (label, rating) pairs is not preserved")
        if len(pos) == 1:
            n_single += 1
            if before != after:
                raise FtcError(f"item {item}: a single example changed")
        else:
            n_multi += 1
            n_mixed += len({x[0] for x in before}) > 1
        n_changed += sum(b[0] != a[0] for b, a in zip(before, after))
    return {"rows": len(train_rows), "examples": sum(len(v) for v in where.values()), "items": len(where),
            "items_with_one_example": n_single, "items_with_2plus_examples": n_multi,
            "items_with_both_classes": n_mixed, "labels_changed": n_changed, "moved_fields": list(moved)}


def build(train, out_dir, *, seed: int = SEED, split=None, domain: str = "toys") -> dict:
    """train_perm.jsonl and train_perm.manifest.json under out_dir from the registered train.jsonl; the manifest."""
    train, out = Path(train), Path(out_dir)
    if not train.is_file():
        raise FtcError(f"{train}: no such file (run_ftgrid.sh stage 0 writes panels/<d>/train.jsonl)", 2)
    lines, rows = read_rows(train)
    if not rows:
        raise FtcError(f"{train} holds no row")
    train_sha1 = file_sha1(train)
    if sha1_bytes(b"".join(lines)) != train_sha1:
        raise FtcError(f"{train} has blank lines: it is not a file ftgrid_data wrote")
    try:
        fd.check_rows(rows, "train.jsonl")
    except SystemExit as e:
        raise FtcError(str(e.code)) from None
    for k, (line, row) in enumerate(zip(lines, rows)):
        if fd.row_bytes(row) != line:
            raise FtcError(f"{train}: row {k} is not in the format ftgrid_data writes (json.dumps(row, ensure_ascii="
                           "False) + newline): FT-C needs the registered train.jsonl byte for byte")
    split_info = None
    if split is not None:
        sp = read_json(split)
        recorded = (sp.get("files") or {}).get("train.jsonl")
        if recorded != train_sha1:
            raise FtcError(f"{train}: sha1 {train_sha1} is not the one {split} records for train.jsonl ({recorded}): "
                           "FT-C permutes the registered TRAIN file only")
        if sp.get("domain") not in (None, domain):
            raise FtcError(f"{split} is the split of {sp.get('domain')!r}, not {domain!r}")
        split_info = {"ftgrid_split_sha1": file_sha1(split), "domain": sp.get("domain"), "T": sp.get("T"),
                      "variant": sp.get("variant"), "train_sha1_recorded": recorded}
    perm, n_changed = fd.permute_within_item(rows, seed)           # the function of ML-1M's section 9, imported
    perm_lines = [fd.row_bytes(r) for r in perm]                    # ... and its writer
    facts = verify_permutation(rows, [json.loads(b) for b in perm_lines])    # recomputed on what will be on disk
    if facts["labels_changed"] != n_changed:
        raise FtcError(f"permute_within_item reports {n_changed} changed labels, the files differ in "
                       f"{facts['labels_changed']}")
    if not n_changed:
        print("WARNING: the within-item permutation changed no label (no item has examples of both classes)",
              file=sys.stderr)
    perm_sha1 = sha1_bytes(b"".join(perm_lines))
    manifest = strict_json({
        "purpose": "FT-C permuted TRAIN panel (Amendment 3 addendum 6 item 8; Amendment 3 section 9)",
        "domain": domain, "seed": seed, "function": "src.confrec.ftgrid_data.permute_within_item",
        "moved_fields": facts["moved_fields"],
        "train": {"file": train.name, "sha1": train_sha1, "rows": facts["rows"], "examples": facts["examples"]},
        "train_perm": {"file": PERM_NAME, "sha1": perm_sha1, "rows": len(perm_lines), "examples": facts["examples"]},
        "n_labels_changed": n_changed,
        "items": {k: facts[k] for k in ("items", "items_with_one_example", "items_with_2plus_examples",
                                        "items_with_both_classes")},
        "checks": {"label_sum_and_label_rating_pairs_preserved_per_item": True, "single_example_items_unchanged": True,
                   "only_moved_fields_differ": True, "train_file_is_what_ftgrid_data_writes": True,
                   "train_sha1_is_the_splits": None if split_info is None else True},
        "split": split_info,
        "code_sha1": {"ftc_panel.py": file_sha1(__file__), "ftgrid_data.py": file_sha1(fd.__file__),
                      "build_rated_panels.py": file_sha1(brp.__file__)}})
    out.mkdir(parents=True, exist_ok=True)
    staged = {PERM_NAME: out / (PERM_NAME + ".tmp"), MANIFEST_NAME: out / (MANIFEST_NAME + ".tmp")}
    try:
        staged[PERM_NAME].write_bytes(b"".join(perm_lines))
        staged[MANIFEST_NAME].write_bytes((json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        for name in (PERM_NAME, MANIFEST_NAME):                      # the manifest last
            brp.commit(staged.pop(name), out / name)
    finally:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)
    if file_sha1(out / PERM_NAME) != perm_sha1:
        raise FtcError(f"{out / PERM_NAME} is not the file that was verified")
    return manifest


# ---------------------------------------------------------------- the recorded training recipe
def train_args(cfg: dict, where: str = "train_config.json") -> dict:
    """The arguments train_lora_yesno recorded in a train_config.json (every argparse destination)."""
    missing = [k for k in TRAIN_KEYS if k not in cfg]
    if missing:
        raise FtcError(f"{where}: no record of {missing}")
    return {k: cfg[k] for k in TRAIN_KEYS}


def flag_value(v) -> str:
    """A recorded argument as a command-line token (floats by repr, so float(token) == the recorded value)."""
    return repr(v) if isinstance(v, float) else str(v)


def recipe_flags(cfg: dict) -> list:
    """The recorded flags except --train, --out and --seed; an unset optional flag (None) is the default again."""
    a = train_args(cfg)
    return [t for k in TRAIN_KEYS if k not in REPLACED and a[k] is not None for t in (f"--{k}", flag_value(a[k]))]


def train_argv(cfg: dict, *, train: str, out: str, seed: int) -> list:
    """The trainer's whole argument vector of a recorded config with --train, --out and --seed replaced."""
    return ["--train", str(train), "--out", str(out), "--seed", str(int(seed))] + recipe_flags(cfg)


def check_recipe(cfgs: dict, *, adapters: str, model: str, variant: str, train_ref: str, split: dict | None = None) -> dict:
    """Validate the train_config.json of the SFT adapters s0-s2 ({seed: config}); returns s0's whole recorded config."""
    if sorted(cfgs) != list(SFT_SEEDS):
        raise FtcError(f"the recipe needs the configs of the adapters s0-s2, got seeds {sorted(cfgs)}")
    base, facts = None, None
    for k in SFT_SEEDS:
        where = f"{adapters}/s{k}/train_config.json"
        a = train_args(cfgs[k], where)
        for key, want in (("seed", k), ("out", f"{adapters}/s{k}"), ("train", train_ref), ("model", model),
                          ("variant", variant)):
            if a[key] != want:
                raise FtcError(f"{where}: {key} is {a[key]!r}, not {want!r}")
        shared = {x: v for x, v in a.items() if x not in ("out", "seed")}
        f = {x: cfgs[k].get(x) for x in ("n_examples", "n_skipped_overlength")}
        if base is None:
            base, facts = shared, f
        elif shared != base or f != facts:
            diff = sorted(x for x in set(shared) | set(f) if {**shared, **f}.get(x) != {**base, **facts}.get(x))
            raise FtcError(f"the Toys adapters s0-s2 were not trained with one recipe: s{k} differs from s0 in {diff}")
    for key, want in TRAIN_REGISTERED.items():
        if base[key] != want:
            raise FtcError(f"adapters/s0 recorded {key} = {base[key]!r}, the registered recipe (Amendment 3 section 2) has "
                           f"{want!r}")
    if base["bsz"] * base["grad_accum"] != EFFECTIVE_BATCH:
        raise FtcError(f"adapters/s0 recorded bsz {base['bsz']} x grad_accum {base['grad_accum']}: not an effective batch "
                       f"of {EFFECTIVE_BATCH} (Amendment 3 section 2)")
    if split is not None:
        ov = (split.get("train") or {}).get("overlength") or {}
        want = {"bsz": ov.get("micro_bsz"), "grad_accum": ov.get("grad_accum"), "max_len": ov.get("max_len_used")}
        bad = {key: (base[key], v) for key, v in want.items() if base[key] != v}
        if bad:
            raise FtcError(f"the recipe recorded by adapters/s0 differs from ftgrid_split.json's (recorded, split): {bad}")
        written = (split.get("train") or {}).get("examples_written")
        if None not in (facts["n_examples"], facts["n_skipped_overlength"], written) \
                and facts["n_examples"] + facts["n_skipped_overlength"] != written:
            raise FtcError(f"adapters/s0 trained on {facts['n_examples']} examples (+ {facts['n_skipped_overlength']} "
                           f"skipped), ftgrid_split.json writes {written}: not the registered TRAIN file")
    return dict(cfgs[SFT_SEEDS[0]])


def check_same_training(ref: dict, run: dict, *, train: str, out: str, seed: int, where: str = "train_config.json") -> None:
    """A trained p<k>: its recorded arguments equal s0's except --train, --out and --seed (which are the given ones), and
    the prompt-determined facts agree."""
    a, b = train_args(ref, "adapters/s0/train_config.json"), train_args(run, where)
    for key, want in (("train", str(train)), ("out", str(out)), ("seed", int(seed))):
        if b[key] != want:
            raise FtcError(f"{where}: {key} is {b[key]!r}, expected {want!r}")
    diff = [k for k in TRAIN_KEYS if k not in REPLACED and a[k] != b[k]]
    diff += [k for k in TRAIN_FACTS if ref.get(k) != run.get(k)]
    if diff:
        raise FtcError(f"{where} differs from the recorded recipe of s0 in {diff} (only {list(REPLACED)} may differ)")


# ---------------------------------------------------------------- the recorded like pass
def parse_run_key(text: str) -> dict:
    """run_ftgrid.sh's run.key: `panel_sha1 MODEL VARIANT adapter_weights_sha1 ARGS...` (score())."""
    tok = text.split()
    if len(tok) < 4:
        raise FtcError(f"run.key {text!r} is not `panel_sha1 model variant weights_sha1 args...`")
    return {"panel_sha1": tok[0], "model": tok[1], "variant": tok[2], "weights_sha1": tok[3], "args": tok[4:]}


def report_config(report: dict, where: str) -> dict:
    cfg = report.get("config") if isinstance(report, dict) else None
    if not isinstance(cfg, dict):
        raise FtcError(f"{where}: no config in report.json")
    return cfg


def score_flags(cfg: dict) -> list:
    """The scorer flags that a recorded like pass determines, except --data, --output, --model and --lora."""
    return ["--dtype", str(cfg["dtype"]), "--topk_logprobs", str(cfg["topk_logprobs"]), "--max_model_len",
            str(cfg["max_model_len"]), "--chunk_users", str(cfg["chunk_users"]), "--variant", str(cfg["variant"]),
            "--readout", str(cfg["readout"]), "--questions", ",".join(cfg["questions"])]


def check_scoring_record(key_text: str, cfg: dict, *, data_sha1: str, model: str, variant: str, lora: str,
                         lora_weights_sha1: str) -> list:
    """Validate the like pass of s0 (its run.key and report config); returns the scorer flags FT-C passes."""
    key = parse_run_key(key_text)
    for what, got, want in (("panel sha1", key["panel_sha1"], data_sha1), ("model", key["model"], model),
                            ("variant", key["variant"], variant), ("adapter weights sha1", key["weights_sha1"],
                                                                    lora_weights_sha1),
                            ("arguments", key["args"], ["--lora", lora])):
        if got != want:
            raise FtcError(f"run.key of the like pass of s0: {what} is {got!r}, not {want!r} (eval.jsonl or the adapter "
                           "changed since the pass, or it is not the registered pass)")
    for what, want in (("data_sha1", data_sha1), ("model", model), ("variant", variant), ("lora", lora)):
        if cfg.get(what) != want:
            raise FtcError(f"report.json config of the like pass of s0: {what} is {cfg.get(what)!r}, not {want!r}")
    for what, want in SCORE_REGISTERED.items():
        if cfg.get(what) != want:
            raise FtcError(f"report.json config of the like pass of s0: {what} is {cfg.get(what)!r}; the registered like "
                           f"pass has {want!r} (run_ftgrid.sh score())")
    try:
        window = resolve_hist_len(variant, str(cfg.get("panel_kind")), None)
    except ValueError as e:
        raise FtcError(f"report.json config of the like pass of s0: {e}") from None
    if cfg.get("hist_len") != window:
        raise FtcError(f"report.json config of the like pass of s0: hist_len {cfg.get('hist_len')!r}, the registered "
                       f"window of {variant} is {window}")
    return score_flags(cfg)


def check_same_scoring(ref_cfg: dict, run_cfg: dict, *, lora: str, where: str) -> None:
    """A scored p<k>: its recorded config equals that of s0's like pass except `lora` (which is the given adapter)."""
    if run_cfg.get("lora") != lora:
        raise FtcError(f"{where}: scored with {run_cfg.get('lora')!r}, not {lora!r}")
    diff = sorted(k for k in set(ref_cfg) | set(run_cfg) if k != "lora" and ref_cfg.get(k) != run_cfg.get(k))
    if diff:
        raise FtcError(f"{where}: the recorded scoring arguments differ from the like pass of s0 in {diff}")


# ---------------------------------------------------------------- provenance of an adapter
def write_provenance(adapter_dir, *, p_seed: int, perm_sha1: str, manifest_sha1: str, args: dict) -> None:
    rec = {"purpose": "FT-C adapter trained by scripts/sigir/run_ftc.sh", "adapter": f"p{p_seed}", "seed": p_seed,
           "train_perm_sha1": perm_sha1, "manifest_sha1": manifest_sha1, "args": args,
           "code_sha1": {"ftc_panel.py": file_sha1(__file__)}}
    path = Path(adapter_dir) / PROVENANCE_NAME
    path.write_bytes((json.dumps(strict_json(rec), indent=2, allow_nan=False) + "\n").encode("utf-8"))


def check_provenance(adapter_dir, *, p_seed: int, perm_sha1: str) -> None:
    path = Path(adapter_dir) / PROVENANCE_NAME
    if not path.is_file():
        raise FtcError(f"{adapter_dir} has no {PROVENANCE_NAME}: it was not trained by run_ftc.sh on the registered "
                       "permuted panel; move it aside and rerun stage 2")
    rec = read_json(path)
    if rec.get("adapter") != f"p{p_seed}" or rec.get("train_perm_sha1") != perm_sha1:
        raise FtcError(f"{adapter_dir} was trained on a permuted panel with sha1 {rec.get('train_perm_sha1')}, the current "
                       f"train_perm.jsonl has {perm_sha1}: move the adapter aside and rerun stage 2")


# ---------------------------------------------------------------- the pilot-log record
def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def record_lines(files, root=None) -> list:
    root = Path(root) if root else default_root()
    return [f"{rel} = {file_sha1(root / rel)}" for rel in files]


def record_missing(pilot_log, files, root=None) -> list:
    """The files whose sha1 is not in the pilot log (a case-insensitive substring test, as ftgrid_freeze)."""
    if not Path(pilot_log).exists():
        raise FtcError(f"pilot log {pilot_log} does not exist", 4)
    root = Path(root) if root else default_root()
    log = Path(pilot_log).read_text(encoding="utf-8").lower()
    return [rel for rel in files if file_sha1(root / rel).lower() not in log]


# ---------------------------------------------------------------- command line
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("command", choices=list(COMMANDS))
    ap.add_argument("--domain", default="toys", help="build: the dataset recorded in the manifest (toys)")
    ap.add_argument("--train", default=None, help="build: panels/<d>/train.jsonl; verify_adapter: the permuted panel path")
    ap.add_argument("--out_dir", default=None, help="build: ftc/<d>/ (train_perm.jsonl and its manifest go here)")
    ap.add_argument("--split", default=None, help="ftgrid_split.json of the dataset")
    ap.add_argument("--seed", type=int, default=SEED, help="build: the permutation seed (section 9: 0)")
    ap.add_argument("--adapters", default=None, help="adapters/<d>/ (s0-s2 are read, p<k> are verified)")
    ap.add_argument("--scores", default=None, help="scores/<d>/ (the like passes of s0 and p<k>)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--train_ref", default=None, help="recipe: the TRAIN file path the adapters s0-s2 recorded")
    ap.add_argument("--data", default=None, help="scoring: panels/<d>/eval.jsonl")
    ap.add_argument("--p_seed", type=int, default=None, help="verify_*: the FT-C adapter p<k>")
    ap.add_argument("--perm_manifest", default=None, help="verify_adapter: train_perm.manifest.json")
    ap.add_argument("--write", action="store_true", help="verify_adapter: write the adapter's provenance record")
    ap.add_argument("--pilot_log", default=None, help="record: docs/sigir/PILOT_LOG.md")
    ap.add_argument("--files", nargs="+", default=None, help="record: repo-relative files whose sha1 the log must hold")
    ap.add_argument("--root", default=None, help="record: the repo root (default: this checkout)")
    ap.add_argument("--print", dest="do_print", action="store_true", help="record: print the lines for the pilot log")
    return ap.parse_args(argv)


def need(a: argparse.Namespace, *names: str) -> None:
    missing = [f"--{n}" for n in names if getattr(a, n) is None]
    if missing:
        raise FtcError(f"{a.command} needs {' '.join(missing)}", 2)


def sft_configs(adapters: str) -> dict:
    """{seed: train_config.json of the SFT adapter s<seed>}; a missing adapter is an error naming it."""
    out = {}
    for k in SFT_SEEDS:
        p = Path(adapters) / f"s{k}" / "train_config.json"
        if not p.is_file():
            raise FtcError(f"missing the Toys adapter s{k} ({p}): run_ftgrid.sh stage 1 trains it, FT-C reads its recipe")
        out[k] = read_json(p)
    return out


def cmd_recipe(a) -> list:
    need(a, "adapters", "model", "variant", "train_ref")
    adapters = a.adapters.rstrip("/")
    split = read_json(a.split) if a.split else None
    base = check_recipe(sft_configs(adapters), adapters=adapters, model=a.model, variant=a.variant,
                        train_ref=a.train_ref, split=split)
    return recipe_flags(base)


def cmd_scoring(a) -> list:
    need(a, "scores", "adapters", "data", "model", "variant")
    scores, adapters = a.scores.rstrip("/"), a.adapters.rstrip("/")
    like = Path(scores) / "s0" / "like"
    for name in ("run.key", "report.json"):
        if not (like / name).is_file():
            raise FtcError(f"missing the like pass of the Toys adapter s0 ({like / name}): run_ftgrid.sh stage 3 scores it, "
                           "FT-C reads its arguments")
    cfg = report_config(read_json(like / "report.json"), str(like / "report.json"))
    return check_scoring_record((like / "run.key").read_text(encoding="utf-8"), cfg, data_sha1=file_sha1(a.data),
                                model=a.model, variant=a.variant, lora=f"{adapters}/s0",
                                lora_weights_sha1=weights_sha1(Path(adapters) / "s0"))


def cmd_verify_adapter(a) -> None:
    need(a, "adapters", "p_seed", "train", "perm_manifest")
    adapters = a.adapters.rstrip("/")
    out = f"{adapters}/p{a.p_seed}"
    cfg = read_json(Path(out) / "train_config.json")
    check_same_training(read_json(Path(adapters) / "s0" / "train_config.json"), cfg, train=a.train, out=out,
                        seed=a.p_seed, where=f"{out}/train_config.json")
    man = read_json(a.perm_manifest)
    perm_sha1 = man["train_perm"]["sha1"]
    if file_sha1(a.train) != perm_sha1:
        raise FtcError(f"{a.train} is not the file {a.perm_manifest} records (sha1 {perm_sha1})")
    if a.write:
        write_provenance(out, p_seed=a.p_seed, perm_sha1=perm_sha1, manifest_sha1=file_sha1(a.perm_manifest),
                         args=train_args(cfg))
    check_provenance(out, p_seed=a.p_seed, perm_sha1=perm_sha1)


def cmd_verify_scores(a) -> None:
    need(a, "scores", "adapters", "p_seed")
    scores, adapters = a.scores.rstrip("/"), a.adapters.rstrip("/")
    ref = report_config(read_json(Path(scores) / "s0" / "like" / "report.json"), "the like pass of s0")
    run = Path(scores) / f"p{a.p_seed}" / "like" / "report.json"
    check_same_scoring(ref, report_config(read_json(run), str(run)), lora=f"{adapters}/p{a.p_seed}", where=str(run))


def cmd_record(a) -> int:
    need(a, "pilot_log", "files")
    if a.do_print:
        print("\n".join(record_lines(a.files, a.root)))
        return 0
    missing = record_missing(a.pilot_log, a.files, a.root)
    if missing:
        raise FtcError(f"FT-C record (Amendment 3 addendum 6 section 11): the sha1 of these files is not in {a.pilot_log}: "
                       f"{missing}; run `python -m src.confrec.ftc_panel record --pilot_log {a.pilot_log} --files "
                       f"{' '.join(a.files)} --print`, record the lines in the pilot log and push it", 4)
    print(f"FT-C record OK: the sha1 of {', '.join(a.files)} is in {a.pilot_log}")
    return 0


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        if a.command == "build":
            need(a, "train", "out_dir")
            man = build(a.train, a.out_dir, seed=a.seed, split=a.split, domain=a.domain)
            print(json.dumps(man, allow_nan=False))
            print(f"ftc_panel build: {man['n_labels_changed']} of {man['train']['examples']} TRAIN labels moved "
                  f"({man['n_labels_changed'] / man['train']['examples']:.1%}); {man['items']['items_with_both_classes']} of "
                  f"{man['items']['items']} items have both classes among their examples", file=sys.stderr)
        elif a.command == "recipe":
            print("\n".join(cmd_recipe(a)))
        elif a.command == "scoring":
            print("\n".join(cmd_scoring(a)))
        elif a.command == "verify_adapter":
            cmd_verify_adapter(a)
            print(f"adapter p{a.p_seed}: the recorded arguments equal s0's except {', '.join(REPLACED)}; provenance OK")
        elif a.command == "verify_scores":
            cmd_verify_scores(a)
            print(f"like pass of p{a.p_seed}: the recorded scoring arguments equal the like pass of s0's except lora")
        else:
            return cmd_record(a)
    except FtcError as e:
        print(f"ftc_panel {a.command}: {e}", file=sys.stderr)
        return e.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
