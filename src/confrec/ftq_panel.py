"""FT-Q, the item-only teacher control: the teacher panel, the recorded recipe, the control-root plumbing and the equality
checks behind scripts/sigir/run_ftq.sh (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_8.md section 2; read with addendum 6 item 8).
CPU only, no torch; every bound module is imported read-only and none is changed.

    python -m src.confrec.ftq_panel build --domain toys --train outputs/confrec/ftgrid/panels/toys/train.jsonl \
        --qhat outputs/confrec/ftmethod/toys/train_qhat.csv.gz --qmanifest outputs/confrec/ftmethod/toys/qhat_manifest.json \
        --split outputs/confrec/ftgrid/panels/toys/ftgrid_split.json --out_dir outputs/confrec/ftgrid_q/ftq/toys
    python -m src.confrec.ftq_panel recipe --adapters outputs/confrec/ftgrid/adapters/toys --model M --variant V \
        --train_ref outputs/confrec/ftgrid/panels/toys/train.jsonl --train_file outputs/confrec/ftgrid/panels/toys/train.jsonl \
        --split outputs/confrec/ftgrid/panels/toys/ftgrid_split.json --split_recipe
    python -m src.confrec.ftq_panel scoring --scores outputs/confrec/ftgrid/scores/toys --adapters outputs/confrec/ftgrid/adapters/toys \
        --data outputs/confrec/ftgrid/panels/toys/eval.jsonl --model M --variant V
    python -m src.confrec.ftq_panel verify_adapter --adapters outputs/confrec/ftgrid_q/adapters/toys \
        --ref_adapters outputs/confrec/ftgrid/adapters/toys --p_seeds 0,1 --train .../ftq/toys/train_q.jsonl \
        --teacher_manifest .../ftq/toys/train_q.manifest.json [--write]
    python -m src.confrec.ftq_panel verify_scores --scores outputs/confrec/ftgrid_q/scores/toys \
        --ref_scores outputs/confrec/ftgrid/scores/toys --adapters outputs/confrec/ftgrid_q/adapters/toys --p_seeds 0
    python -m src.confrec.ftq_panel link --real_scores outputs/confrec/ftgrid/scores/toys \
        --q_scores outputs/confrec/ftgrid_q/scores/toys --models zeroshot,s0,s1,s2 [--allow_copy]
    python -m src.confrec.ftq_panel record --pilot_log docs/sigir/PILOT_LOG.md --files scripts/sigir/run_ftq.sh \
        src/confrec/ftq_panel.py [--print]
    python -m src.confrec.ftq_panel info --selection .../selection.json --split .../ftgrid_split.json --gate .../gate_ft.json

build           train_q.jsonl = the registered train.jsonl with `candidate_labels` replaced by the teacher labels and nothing else
                changed (histories, prompts, ratings, order). Teacher (addendum 8 section 2): q-hat_e is the prior-only item mean of
                TRAIN example e, read from the nested slot's stage-1 file train_qhat.csv.gz (forensics.prior_means, k = 5); beta is
                the mean of the real TRAIN labels; the round(beta n) examples with the largest q-hat get label 1 and the others 0;
                ties (and a non-finite q-hat, which sorts after every finite one) go by the fixed seed-0 random key per example of
                ftprune.tie_key("<user_id>::<item_id>"), smallest key first, the order of ftprune.removal_mask. The label of an
                example is a function of the q-hat and the tie key of the examples only (teacher_labels has no access to a label;
                only k = round(beta n) depends on the labels, as registered), so the teacher rate equals beta exactly. Inputs are
                checked: the registered train.jsonl (its sha1 is ftgrid_split.json's and the nested slot's manifest's), the stage-1
                files against the manifest (train_lora_offset.load_train_z: format, sha1s, one q-hat row per example with the same
                timestamp and label, z recomputed), and the q-hat rows aligned one to one, in train.jsonl order, with the TRAIN
                examples (source_event_id, candidate index, user, item, timestamp, label). The teacher is computed twice (a sort
                and a numpy lexsort) and the written file is re-read and compared with train.jsonl field by field. Writes
                train_q.jsonl and train_q.manifest.json (sha1 of train.jsonl, train_qhat.csv.gz, its manifest and train_q.jsonl,
                beta, n, k, the tie seed, the code sha1; strict JSON, no clock, host or path: a rerun is byte-identical).
recipe          the trainer flags of the real adapters s0-s2 (ML-1M: Gate-FT's, outputs/confrec/gateft/adapters): every key of their
                train_config.json that records an argument, except --train, --out and --seed (one token per line; an unset
                optional flag is left out, which is the trainer's default again). s0, s1 and s2 must agree on every argument, on the
                registered constants of Amendment 3 section 2 and, with --split_recipe (the Amazon datasets), on the split's recipe;
                their recorded TRAIN file must hold the bytes of --train_file (ML-1M: Gate-FT's train.jsonl is the grid's).
scoring         the scorer flags of the real like pass of s0 (run.key and the config of its report.json: nothing else of that report
                is read), except --data, --output, --model and --lora; the record must be the registered like pass of s0.
verify_adapter  after training: the recorded arguments of each p<k> equal those of the real s0 except --train, --out and --seed (and
                the prompt-determined facts agree); --write records provenance (ftq.json: the sha1 of the teacher panel it was
                trained on); without --write an existing adapter must carry a matching record, so an adapter trained on another
                panel is never reused.
verify_scores   after scoring: the recorded config of each p<k> like pass equals that of the real s0 like pass except `lora`.
link            scores/<d>/{zeroshot,s0,s1,s2} of the FT-Q root become symlinks (relative) to the real score directories, never copies;
                --allow_copy (DRY_RUN only) falls back to a copy where the platform has no symlinks (a Windows machine).
record          0 iff the pilot log holds the sha1 of every listed file (addendum 8 section 4: recorded before the first teacher
                adapter is trained); --print writes the lines to paste into the log.
info            decision and gate_ft_prompt of selection.json, the variant of the split and the decision of gate_ft.json, one per line.
Exit codes: 0 done; 1 error or an inconsistent record; 2 refused input; 4 refused by a registered gate (record).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

from src.confrec.prompting import resolve_hist_len

SPEC = "idea-stage/PREREG_AMENDMENT_3_ADDENDUM_8.md section 2 (FT-Q); idea-stage/PREREG_AMENDMENT_3_ADDENDUM_6.md item 8"
DOMAINS = ("ml1m", "toys", "games", "sports")
TIE_SEED = 0                                # addendum 8 section 2: ties by a fixed seed-0 random key (ftprune.TIE_SEED)
TEACHER_NAME, MANIFEST_NAME, PROVENANCE_NAME = "train_q.jsonl", "train_q.manifest.json", "ftq.json"
MANIFEST_FORMAT = "ftq_teacher_v1"
SFT_SEEDS, CONTROL_SEEDS = (0, 1, 2), (0, 1)   # the real adapters whose recipe FT-Q uses; the teacher adapters p0, p1
LINKED_MODELS = ("zeroshot", "s0", "s1", "s2")
REPLACED = ("train", "out", "seed")         # the only arguments FT-Q changes
# everything train_lora_yesno's main() records as vars(args) in train_config.json (its argparse destinations)
TRAIN_KEYS = ("train", "model", "out", "mode", "variant", "seed", "hist_len", "max_len", "epochs", "lr", "bsz",
              "grad_accum", "lora_r", "max_examples")
# facts of train_config.json that the prompts (not the labels) determine: equal for a panel with other labels
TRAIN_FACTS = ("loss", "hist_len_used", "panel_kind", "max_history_len_in_panel", "n_examples", "n_skipped_overlength")
# Amendment 3 section 2 (LoRA r 16, lr 1e-4, one epoch, standard mode) and the trainer's defaults that run_ftgrid.sh and
# run_gateft.sh never override (the variant's registered history window, no example cap): what the recorded s0-s2 must be
TRAIN_REGISTERED = {"mode": "standard", "hist_len": None, "epochs": 1.0, "lr": 1e-4, "lora_r": 16, "max_examples": None}
EFFECTIVE_BATCH = 32                        # micro-batch x accumulation (Amendment 3 section 2)
# the like pass of run_ftgrid.sh's score(): fp16, top-50 logprobs, max_model_len 4096, 100-user chunks, yes/no readout,
# the question `like`, no swap prior, no user cap, the scorer's default seed and chunk_items
SCORE_REGISTERED = {"questions": ["like"], "swap_k": 0, "seed": 0, "dtype": "float16", "topk_logprobs": 50,
                    "max_model_len": 4096, "chunk_users": 100, "chunk_items": 1000, "readout": "yesno", "n_users": None}
COMMANDS = ("build", "recipe", "scoring", "verify_adapter", "verify_scores", "link", "record", "info")


class FtqError(Exception):
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
        raise FtqError(f"cannot read {path}: {e}") from None
    return h.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise FtqError(f"cannot read {path}: {e}") from None


def json_bytes(obj) -> bytes:
    """Strict JSON (no NaN), two-space indent, LF, UTF-8: the same bytes on every platform."""
    return (json.dumps(obj, indent=2, allow_nan=False) + "\n").encode("utf-8")


def weights_sha1(adapter_dir) -> str:
    """The sha1 run_ftgrid.sh's score() puts in run.key: `cat DIR/adapter_model.* | sha1sum`."""
    parts = sorted(Path(adapter_dir).glob("adapter_model.*"))
    if not parts:
        raise FtqError(f"{adapter_dir}: no adapter_model.* weights")
    h = hashlib.sha1()
    for p in parts:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- the teacher
def teacher_k(beta: float, n: int) -> int:
    """round(beta n): the number of examples that get label 1."""
    return int(round(beta * n))


def teacher_order(q_hat, tie) -> list:
    """Example indices in teacher order: a finite q-hat first, the largest first; a non-finite q-hat after every finite one; ties
    (and the non-finite) by the tie key ascending. The order of ftprune.removal_mask, for one class."""
    return sorted(range(len(q_hat)), key=lambda j: (0, -q_hat[j], tie[j]) if math.isfinite(q_hat[j])
                  else (1, 0.0, tie[j]))


def teacher_labels(q_hat, tie, k: int) -> list:
    """The teacher: label 1 for the first k examples in teacher order, 0 for the others. It sees the q-hat and the tie key of
    every example and nothing else (no label, no user, no time): a pure function of (q_hat, tie, k)."""
    n = len(q_hat)
    if len(tie) != n:
        raise FtqError(f"{len(tie)} tie keys for {n} q-hat values")
    if not 0 <= k <= n:
        raise FtqError(f"the teacher cannot label {k} of {n} examples 1")
    labels = [0] * n
    for j in teacher_order(q_hat, tie)[:k]:
        labels[j] = 1
    return labels


def teacher_labels_lexsort(q_hat, tie, k: int) -> list:
    """The same teacher through numpy.lexsort (a second implementation, compared with teacher_labels when the panel is built)."""
    import numpy as np
    q = np.asarray(q_hat, float)
    finite = np.isfinite(q)
    order = np.lexsort((np.asarray(tie, dtype=str), np.where(finite, -q, 0.0), (~finite).astype(int)))
    labels = np.zeros(len(q), int)
    labels[order[:k]] = 1
    return labels.tolist()


def read_rows(path) -> tuple:
    """([line bytes], [row]) of a panel file; blank lines are not rows (as ftgrid_data.read_panel)."""
    lines, rows = [], []
    try:
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    lines.append(line)
                    rows.append(json.loads(line))
    except (OSError, ValueError) as e:
        raise FtqError(f"cannot read {path}: {e}") from None
    return lines, rows


def verify_teacher_rows(train_rows: list, teacher_rows: list, labels: list) -> dict:
    """The registered "nothing else changed", recomputed from the two row lists: the same rows in the same order with the same keys,
    every field but candidate_labels equal (candidate_ratings, histories, ids, timestamps, order), and candidate_labels the
    teacher's. Raises FtqError on a violation; returns the counts the manifest records."""
    if len(train_rows) != len(teacher_rows):
        raise FtqError(f"the teacher panel has {len(teacher_rows)} rows, train.jsonl {len(train_rows)}")
    pos = 0
    for k, (r, t) in enumerate(zip(train_rows, teacher_rows)):
        if list(r) != list(t):
            raise FtqError(f"row {k}: the teacher row has other keys than train.jsonl's")
        changed = [f for f in r if r[f] != t[f]]
        if any(f != "candidate_labels" for f in changed):
            raise FtqError(f"row {k}: {[f for f in changed if f != 'candidate_labels']} changed (only candidate_labels may)")
        m = len(r["candidate_labels"])
        if len(t["candidate_labels"]) != m:
            raise FtqError(f"row {k}: candidate_labels changed its length")
        if [int(x) for x in t["candidate_labels"]] != labels[pos:pos + m]:
            raise FtqError(f"row {k}: candidate_labels are not the teacher's")
        pos += m
    if pos != len(labels):
        raise FtqError(f"{len(labels)} teacher labels for {pos} candidates")
    return {"rows": len(train_rows), "examples": pos}


def _tlo():
    from src.confrec import train_lora_offset as tlo
    return tlo


def aligned_qhat(train_path, qhat_path, qmanifest_path, ex: dict) -> list:
    """The q-hat of every TRAIN example, in train.jsonl order, after the stage-1 files have been checked against their manifest
    (train_lora_offset.load_train_z) and aligned one to one with the examples (ex = ftprune.train_examples)."""
    tlo = _tlo()
    try:
        tlo.load_train_z(train_path, qhat_path, qmanifest_path)
        q = tlo.read_qhat(qhat_path)
    except SystemExit as e:
        raise FtqError(f"the stage-1 q-hat files are not those of this train.jsonl: {e.code}") from None
    if len(q) != ex["n"]:
        raise FtqError(f"{qhat_path}: {len(q)} q-hat rows for {ex['n']} TRAIN examples")
    out = []
    for j, (key, row) in enumerate(q.items()):
        same = (key == (ex["user"][j], ex["item"][j]) and row["source_event_id"] == ex["ev"][j]
                and row["cand_idx"] == int(ex["cand"][j]) and row["ts"] == float(ex["ts"][j])
                and row["label"] == int(ex["y"][j]))
        if not same:
            raise FtqError(f"{qhat_path}: row {j} {key} is not TRAIN example {j} {(ex['user'][j], ex['item'][j])} of "
                           f"{train_path} (source event, candidate index, timestamp or label differ): the rows must align "
                           "one to one in train.jsonl order")
        out.append(row["q_hat"])
    return out


def build(train, qhat, qmanifest, out_dir, *, domain: str, split=None) -> dict:
    """train_q.jsonl and train_q.manifest.json under out_dir; the manifest."""
    import random
    from src.confrec import build_rated_panels as brp
    from src.confrec import ftgrid_data as fd
    from src.confrec import ftprune
    from src.confrec.stats import strict_json
    train, qhat, qmanifest, out = Path(train), Path(qhat), Path(qmanifest), Path(out_dir)
    for p, hint in ((train, "run_ftgrid.sh stage 0 writes panels/<d>/train.jsonl"),
                    (qhat, "run_ftmethod.sh <d> with STAGES=1 writes the nested slot's stage-1 files"),
                    (qmanifest, "run_ftmethod.sh <d> with STAGES=1 writes the nested slot's stage-1 files")):
        if not p.is_file():
            raise FtqError(f"{p}: no such file ({hint})", 1)
    lines, rows = read_rows(train)
    if not rows:
        raise FtqError(f"{train} holds no row")
    train_sha1 = file_sha1(train)
    if sha1_bytes(b"".join(lines)) != train_sha1:
        raise FtqError(f"{train} has blank lines: it is not a file ftgrid_data wrote")
    try:
        fd.check_rows(rows, "train.jsonl")
    except SystemExit as e:
        raise FtqError(str(e.code)) from None
    for k, (line, row) in enumerate(zip(lines, rows)):
        if fd.row_bytes(row) != line:
            raise FtqError(f"{train}: row {k} is not in the format ftgrid_data writes (json.dumps(row, ensure_ascii="
                           "False) + newline): FT-Q needs the registered train.jsonl byte for byte")
    split_info = None
    if split is not None:
        sp = read_json(split)
        recorded = (sp.get("files") or {}).get("train.jsonl")
        if recorded != train_sha1:
            raise FtqError(f"{train}: sha1 {train_sha1} is not the one {split} records for train.jsonl ({recorded}): FT-Q "
                           "labels the registered TRAIN file only")
        if sp.get("domain") not in (None, domain):
            raise FtqError(f"{split} is the split of {sp.get('domain')!r}, not {domain!r}")
        split_info = {"ftgrid_split_sha1": file_sha1(split), "domain": sp.get("domain"), "T": sp.get("T"),
                      "variant": sp.get("variant"), "train_sha1_recorded": recorded}
    qman = read_json(qmanifest)
    if qman.get("domain") not in (None, domain):
        raise FtqError(f"{qmanifest} is the manifest of {qman.get('domain')!r}, not {domain!r}")
    if split_info is not None and qman.get("split_sha1") not in (None, split_info["ftgrid_split_sha1"]):
        raise FtqError(f"{qmanifest} was written for another ftgrid_split.json ({qman.get('split_sha1')}) than {split}")
    try:
        ex = ftprune.train_examples(rows)
    except SystemExit as e:
        raise FtqError(str(e.code)) from None
    q_hat = aligned_qhat(train, qhat, qmanifest, ex)
    n = ex["n"]
    y = [int(v) for v in ex["y"]]
    positives = sum(y)
    beta = positives / n
    k = teacher_k(beta, n)
    if k != positives:
        raise FtqError(f"round(beta n) = {k} is not the {positives} real positives of {n} examples")
    tie = [ftprune.tie_key(key) for key in ex["key"]]
    labels = teacher_labels(q_hat, tie, k)
    if labels != teacher_labels_lexsort(q_hat, tie, k):
        raise FtqError("the teacher differs between its two implementations (sort and lexsort)")
    shuffled = random.Random(0).sample(range(n), n)       # a function of (q-hat, tie key) pairs, not of the examples' positions
    again = teacher_labels([q_hat[j] for j in shuffled], [tie[j] for j in shuffled], k)
    back = [0] * n
    for i, j in enumerate(shuffled):
        back[j] = again[i]
    if back != labels:
        raise FtqError("the teacher depends on the order of the examples, not on their q-hat and tie key only")
    if sum(labels) != positives or sum(labels) / n != beta:
        raise FtqError(f"the teacher rate {sum(labels)}/{n} is not beta = {positives}/{n}")
    new_rows, pos = [], 0
    for r in rows:
        m = len(r["candidate_labels"])
        new_rows.append(dict(r, candidate_labels=labels[pos:pos + m]))
        pos += m
    teacher_lines = [fd.row_bytes(r) for r in new_rows]
    facts = verify_teacher_rows(rows, [json.loads(b) for b in teacher_lines], labels)   # recomputed on what is written
    teacher_sha1 = sha1_bytes(b"".join(teacher_lines))
    both = sum(1 for a, b in zip(labels, y) if a == 1 and b == 1)
    n_nonfinite = int(sum(1 for v in q_hat if not math.isfinite(v)))
    manifest = strict_json({
        "format": MANIFEST_FORMAT, "spec": SPEC, "domain": domain,
        "teacher": {"definition": "label 1 for the round(beta n) TRAIN examples with the largest q-hat, 0 for the others; ties "
                                  "and a non-finite q-hat by ftprune.tie_key('<user_id>::<item_id>') ascending; a function of "
                                  "q-hat and the tie key only (addendum 8 section 2)",
                    "q_hat": "train_qhat.csv.gz column q_hat of the nested slot's stage 1 (forensics.prior_means "
                             "mean_prior_shrunk, k = 5, other users' first ratings strictly before the example's time)",
                    "tie_seed": TIE_SEED, "tie_key": f"sha1('tie:{TIE_SEED}:<user_id>::<item_id>')",
                    "order": "ftprune.removal_mask's order for one class: finite q-hat descending, then the tie key ascending"},
        "beta": beta, "n": n, "k": k, "n_real_positives": positives, "n_teacher_positives": int(sum(labels)),
        "n_qhat_nonfinite": n_nonfinite,
        "agreement_with_real_labels": {
            "share_teacher_equals_real_label": sum(1 for a, b in zip(labels, y) if a == b) / n,
            "teacher_positive_among_real_positive": both / positives if positives else None,
            "real_positive_among_teacher_positive": both / k if k else None},
        "train": {"file": train.name, "sha1": train_sha1, "rows": facts["rows"], "examples": n},
        "qhat": {"file": qhat.name, "sha1": file_sha1(qhat), "manifest_file": qmanifest.name,
                 "manifest_sha1": file_sha1(qmanifest), "manifest_train_sha1": (qman.get("train") or {}).get("sha1")},
        "train_q": {"file": TEACHER_NAME, "sha1": teacher_sha1, "rows": len(teacher_lines), "examples": n},
        "checks": {"train_file_is_what_ftgrid_data_writes": True, "train_sha1_is_the_splits":
                   None if split_info is None else True, "qhat_manifest_train_sha1_is_the_panels": True,
                   "qhat_rows_align_one_to_one_with_the_train_examples": True,
                   "teacher_rate_equals_beta_exactly": True, "teacher_is_a_function_of_qhat_and_the_tie_key_only": True,
                   "teacher_is_shuffle_invariant": True,
                   "only_candidate_labels_differ_from_train": True, "two_implementations_agree": True},
        "split": split_info,
        "code_sha1": {"ftq_panel.py": file_sha1(__file__), "ftprune.py": file_sha1(ftprune.__file__),
                      "train_lora_offset.py": file_sha1(_tlo().__file__), "ftgrid_data.py": file_sha1(fd.__file__)}})
    out.mkdir(parents=True, exist_ok=True)
    staged = {TEACHER_NAME: out / (TEACHER_NAME + ".tmp"), MANIFEST_NAME: out / (MANIFEST_NAME + ".tmp")}
    try:
        staged[TEACHER_NAME].write_bytes(b"".join(teacher_lines))
        staged[MANIFEST_NAME].write_bytes(json_bytes(manifest))
        for name in (TEACHER_NAME, MANIFEST_NAME):                      # the manifest last
            brp.commit(staged.pop(name), out / name)
    finally:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)
    if file_sha1(out / TEACHER_NAME) != teacher_sha1:
        raise FtqError(f"{out / TEACHER_NAME} is not the file that was verified")
    return manifest


# ---------------------------------------------------------------- the recorded training recipe
def train_args(cfg: dict, where: str = "train_config.json") -> dict:
    """The arguments train_lora_yesno recorded in a train_config.json (every argparse destination)."""
    missing = [k for k in TRAIN_KEYS if k not in cfg]
    if missing:
        raise FtqError(f"{where}: no record of {missing}")
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


def check_recipe(cfgs: dict, *, adapters: str, model: str, variant: str, train_ref: str, split: dict | None = None,
                 split_recipe: bool = True) -> dict:
    """Validate the train_config.json of the real adapters s0-s2 ({seed: config}); returns s0's whole recorded config.
    split_recipe compares micro-batch, accumulation and max_len with the split's TRAIN length rule (the Amazon datasets; ML-1M's
    adapters are Gate-FT's, trained before that split existed)."""
    if sorted(cfgs) != list(SFT_SEEDS):
        raise FtqError(f"the recipe needs the configs of the adapters s0-s2, got seeds {sorted(cfgs)}")
    base, facts = None, None
    for k in SFT_SEEDS:
        where = f"{adapters}/s{k}/train_config.json"
        a = train_args(cfgs[k], where)
        for key, want in (("seed", k), ("out", f"{adapters}/s{k}"), ("train", train_ref), ("model", model),
                          ("variant", variant)):
            if a[key] != want:
                raise FtqError(f"{where}: {key} is {a[key]!r}, not {want!r}")
        shared = {x: v for x, v in a.items() if x not in ("out", "seed")}
        f = {x: cfgs[k].get(x) for x in ("n_examples", "n_skipped_overlength")}
        if base is None:
            base, facts = shared, f
        elif shared != base or f != facts:
            diff = sorted(x for x in set(shared) | set(f) if {**shared, **f}.get(x) != {**base, **facts}.get(x))
            raise FtqError(f"the adapters s0-s2 were not trained with one recipe: s{k} differs from s0 in {diff}")
    for key, want in TRAIN_REGISTERED.items():
        if base[key] != want:
            raise FtqError(f"adapters/s0 recorded {key} = {base[key]!r}, the registered recipe (Amendment 3 section 2) has "
                           f"{want!r}")
    if base["bsz"] * base["grad_accum"] != EFFECTIVE_BATCH:
        raise FtqError(f"adapters/s0 recorded bsz {base['bsz']} x grad_accum {base['grad_accum']}: not an effective batch "
                       f"of {EFFECTIVE_BATCH} (Amendment 3 section 2)")
    if split is not None:
        tr = split.get("train") or {}
        if split_recipe:
            ov = tr.get("overlength") or {}
            want = {"bsz": ov.get("micro_bsz"), "grad_accum": ov.get("grad_accum"), "max_len": ov.get("max_len_used")}
            bad = {key: (base[key], v) for key, v in want.items() if base[key] != v}
            if bad:
                raise FtqError(f"the recipe recorded by adapters/s0 differs from ftgrid_split.json's (recorded, split): {bad}")
        written = tr.get("examples_written")
        if None not in (facts["n_examples"], facts["n_skipped_overlength"], written) \
                and facts["n_examples"] + facts["n_skipped_overlength"] != written:
            raise FtqError(f"adapters/s0 trained on {facts['n_examples']} examples (+ {facts['n_skipped_overlength']} "
                           f"skipped), ftgrid_split.json writes {written}: not the registered TRAIN file")
    return dict(cfgs[SFT_SEEDS[0]])


def check_train_file(train_ref: str, train_file: str) -> None:
    """The TRAIN file the real adapters recorded holds the bytes of the registered panels train.jsonl (ML-1M: Gate-FT's file
    outputs/confrec/gateft/train.jsonl is the grid's, Amendment 3 section 1)."""
    if str(train_ref) == str(train_file):
        return
    for p in (train_ref, train_file):
        if not Path(p).is_file():
            raise FtqError(f"{p}: no such file (the TRAIN file of the real adapters and the registered train.jsonl are compared)")
    if file_sha1(train_ref) != file_sha1(train_file):
        raise FtqError(f"the real adapters trained on {train_ref}, which differs in bytes from the registered {train_file}")


def check_same_training(ref: dict, run: dict, *, train: str, out: str, seed: int, where: str = "train_config.json") -> None:
    """A trained p<k>: its recorded arguments equal the real s0's except --train, --out and --seed (which are the given ones), and
    the prompt-determined facts agree."""
    a, b = train_args(ref, "the real adapter s0's train_config.json"), train_args(run, where)
    for key, want in (("train", str(train)), ("out", str(out)), ("seed", int(seed))):
        if b[key] != want:
            raise FtqError(f"{where}: {key} is {b[key]!r}, expected {want!r}")
    diff = [k for k in TRAIN_KEYS if k not in REPLACED and a[k] != b[k]]
    diff += [k for k in TRAIN_FACTS if ref.get(k) != run.get(k)]
    if diff:
        raise FtqError(f"{where} differs from the recorded recipe of s0 in {diff} (only {list(REPLACED)} may differ)")


# ---------------------------------------------------------------- the recorded like pass
def parse_run_key(text: str) -> dict:
    """run_ftgrid.sh's run.key: `panel_sha1 MODEL VARIANT adapter_weights_sha1 ARGS...` (score())."""
    tok = text.split()
    if len(tok) < 4:
        raise FtqError(f"run.key {text!r} is not `panel_sha1 model variant weights_sha1 args...`")
    return {"panel_sha1": tok[0], "model": tok[1], "variant": tok[2], "weights_sha1": tok[3], "args": tok[4:]}


def report_config(report: dict, where: str) -> dict:
    cfg = report.get("config") if isinstance(report, dict) else None
    if not isinstance(cfg, dict):
        raise FtqError(f"{where}: no config in report.json")
    return cfg


def score_flags(cfg: dict) -> list:
    """The scorer flags that a recorded like pass determines, except --data, --output, --model and --lora."""
    return ["--dtype", str(cfg["dtype"]), "--topk_logprobs", str(cfg["topk_logprobs"]), "--max_model_len",
            str(cfg["max_model_len"]), "--chunk_users", str(cfg["chunk_users"]), "--variant", str(cfg["variant"]),
            "--readout", str(cfg["readout"]), "--questions", ",".join(cfg["questions"])]


def check_scoring_record(key_text: str, cfg: dict, *, data_sha1: str, model: str, variant: str, lora: str,
                         lora_weights_sha1: str) -> list:
    """Validate the real like pass of s0 (its run.key and report config); returns the scorer flags FT-Q passes."""
    key = parse_run_key(key_text)
    for what, got, want in (("panel sha1", key["panel_sha1"], data_sha1), ("model", key["model"], model),
                            ("variant", key["variant"], variant), ("adapter weights sha1", key["weights_sha1"],
                                                                    lora_weights_sha1),
                            ("arguments", key["args"], ["--lora", lora])):
        if got != want:
            raise FtqError(f"run.key of the like pass of s0: {what} is {got!r}, not {want!r} (eval.jsonl or the adapter "
                           "changed since the pass, or it is not the registered pass)")
    for what, want in (("data_sha1", data_sha1), ("model", model), ("variant", variant), ("lora", lora)):
        if cfg.get(what) != want:
            raise FtqError(f"report.json config of the like pass of s0: {what} is {cfg.get(what)!r}, not {want!r}")
    for what, want in SCORE_REGISTERED.items():
        if cfg.get(what) != want:
            raise FtqError(f"report.json config of the like pass of s0: {what} is {cfg.get(what)!r}; the registered like "
                           f"pass has {want!r} (run_ftgrid.sh score())")
    try:
        window = resolve_hist_len(variant, str(cfg.get("panel_kind")), None)
    except ValueError as e:
        raise FtqError(f"report.json config of the like pass of s0: {e}") from None
    if cfg.get("hist_len") != window:
        raise FtqError(f"report.json config of the like pass of s0: hist_len {cfg.get('hist_len')!r}, the registered "
                       f"window of {variant} is {window}")
    return score_flags(cfg)


def check_same_scoring(ref_cfg: dict, run_cfg: dict, *, lora: str, where: str) -> None:
    """A scored p<k>: its recorded config equals that of the real s0's like pass except `lora` (which is the given adapter)."""
    if run_cfg.get("lora") != lora:
        raise FtqError(f"{where}: scored with {run_cfg.get('lora')!r}, not {lora!r}")
    diff = sorted(k for k in set(ref_cfg) | set(run_cfg) if k != "lora" and ref_cfg.get(k) != run_cfg.get(k))
    if diff:
        raise FtqError(f"{where}: the recorded scoring arguments differ from the like pass of s0 in {diff}")


# ---------------------------------------------------------------- provenance of an adapter
def write_provenance(adapter_dir, *, p_seed: int, train_q_sha1: str, manifest_sha1: str, args: dict) -> None:
    rec = {"purpose": "FT-Q teacher adapter trained by scripts/sigir/run_ftq.sh", "adapter": f"p{p_seed}", "seed": p_seed,
           "train_q_sha1": train_q_sha1, "manifest_sha1": manifest_sha1, "args": args,
           "code_sha1": {"ftq_panel.py": file_sha1(__file__)}}
    (Path(adapter_dir) / PROVENANCE_NAME).write_bytes(json_bytes(rec))


def check_provenance(adapter_dir, *, p_seed: int, train_q_sha1: str) -> None:
    path = Path(adapter_dir) / PROVENANCE_NAME
    if not path.is_file():
        raise FtqError(f"{adapter_dir} has no {PROVENANCE_NAME}: it was not trained by run_ftq.sh on the registered teacher "
                       "panel; move it aside and rerun stage 2")
    rec = read_json(path)
    if rec.get("adapter") != f"p{p_seed}" or rec.get("train_q_sha1") != train_q_sha1:
        raise FtqError(f"{adapter_dir} was trained on a teacher panel with sha1 {rec.get('train_q_sha1')}, the current "
                       f"train_q.jsonl has {train_q_sha1}: move the adapter aside and rerun stage 2")


# ---------------------------------------------------------------- the control root's links to the real scores
def link_models(real_scores, q_scores, models, allow_copy: bool = False) -> list:
    """Make q_scores/<model> a relative symlink to real_scores/<model> for every model; returns one note per model. Never a
    copy unless allow_copy (DRY_RUN on a platform without symlinks); never an existing link that points elsewhere."""
    real, q = Path(real_scores), Path(q_scores)
    notes = []
    for m in models:
        target = real / m
        if not (target / "like" / "report.json").is_file():
            raise FtqError(f"missing the real like pass {target / 'like'} (run_ftgrid.sh {real.name} scores it): FT-Q links the "
                           "real scores, it never rescores them")
    q.mkdir(parents=True, exist_ok=True)
    for m in models:
        target, link = real / m, q / m
        if os.path.islink(link):
            if os.path.realpath(link) != os.path.realpath(target):
                raise FtqError(f"{link} is a link to {os.path.realpath(link)}, not to {os.path.realpath(target)}")
            notes.append(f"[link] {link} -> {os.path.relpath(target, start=q)}")
            continue
        if link.exists():
            if allow_copy and all((target / "like" / f).is_file() and (link / "like" / f).is_file()
                                  and file_sha1(target / "like" / f) == file_sha1(link / "like" / f)
                                  for f in ("report.json", "run.key")):
                notes.append(f"[copy] {link}: a copy of {target} (DRY_RUN without symlinks)")
                continue
            raise FtqError(f"{link} exists and is not a link to {target}: the real scores are linked, never copied")
        try:
            os.symlink(os.path.relpath(target, start=q), link, target_is_directory=True)
            notes.append(f"[link] {link} -> {os.path.relpath(target, start=q)}")
        except (OSError, NotImplementedError) as e:
            if not allow_copy:
                raise FtqError(f"cannot link {link} to {target}: {e}; the real scores are linked, never copied") from None
            shutil.copytree(target, link)
            notes.append(f"[copy] {link}: no symlinks here ({type(e).__name__}); a copy of {target}, DRY_RUN only")
    return notes


# ---------------------------------------------------------------- the pilot-log record
def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def record_lines(files, root=None) -> list:
    root = Path(root) if root else default_root()
    return [f"{rel} = {file_sha1(root / rel)}" for rel in files]


def record_missing(pilot_log, files, root=None) -> list:
    """The files whose sha1 is not in the pilot log (a case-insensitive substring test, as ftgrid_freeze)."""
    if not Path(pilot_log).exists():
        raise FtqError(f"pilot log {pilot_log} does not exist", 4)
    root = Path(root) if root else default_root()
    log = Path(pilot_log).read_text(encoding="utf-8").lower()
    return [rel for rel in files if file_sha1(root / rel).lower() not in log]


# ---------------------------------------------------------------- command line
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("command", choices=list(COMMANDS))
    ap.add_argument("--domain", default=None, help="build: ml1m, toys, games or sports")
    ap.add_argument("--train", default=None, help="build: panels/<d>/train.jsonl; verify_adapter: the teacher panel train_q.jsonl")
    ap.add_argument("--qhat", default=None, help="build: the nested slot's train_qhat.csv.gz")
    ap.add_argument("--qmanifest", default=None, help="build: the nested slot's qhat_manifest.json")
    ap.add_argument("--out_dir", default=None, help="build: ftq/<d>/ (train_q.jsonl and its manifest go here)")
    ap.add_argument("--split", default=None, help="ftgrid_split.json of the dataset (info: also read)")
    ap.add_argument("--split_recipe", action="store_true", help="recipe: also compare with the split's TRAIN length rule")
    ap.add_argument("--adapters", default=None, help="recipe / scoring: the real adapters' directory (s0-s2); verify_*: the "
                                                     "FT-Q root's adapters/<d>/ (p<k>)")
    ap.add_argument("--ref_adapters", default=None, help="verify_adapter: the real adapters' directory (s0's config)")
    ap.add_argument("--scores", default=None, help="scoring: the real scores/<d>/ ; verify_scores: the FT-Q root's scores/<d>/")
    ap.add_argument("--ref_scores", default=None, help="verify_scores: the real scores/<d>/ (the like pass of s0)")
    ap.add_argument("--real_scores", default=None, help="link: the real scores/<d>/")
    ap.add_argument("--q_scores", default=None, help="link: the FT-Q root's scores/<d>/")
    ap.add_argument("--models", default=",".join(LINKED_MODELS), help="link: the models to link")
    ap.add_argument("--allow_copy", action="store_true", help="link: DRY_RUN only, copy where symlinks are unavailable")
    ap.add_argument("--model", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--train_ref", default=None, help="recipe: the TRAIN file path the real adapters recorded")
    ap.add_argument("--train_file", default=None, help="recipe: the registered panels/<d>/train.jsonl (bytes compared)")
    ap.add_argument("--data", default=None, help="scoring: panels/<d>/eval.jsonl")
    ap.add_argument("--p_seeds", default=None, help="verify_*: comma list of the teacher adapters (0,1)")
    ap.add_argument("--teacher_manifest", default=None, help="verify_adapter: train_q.manifest.json")
    ap.add_argument("--write", action="store_true", help="verify_adapter: write the adapters' provenance records")
    ap.add_argument("--selection", default=None, help="info: selection.json")
    ap.add_argument("--gate", default=None, help="info: gate_ft.json")
    ap.add_argument("--pilot_log", default=None, help="record: docs/sigir/PILOT_LOG.md")
    ap.add_argument("--files", nargs="+", default=None, help="record: repo-relative files whose sha1 the log must hold")
    ap.add_argument("--root", default=None, help="record: the repo root (default: this checkout)")
    ap.add_argument("--print", dest="do_print", action="store_true", help="record: print the lines for the pilot log")
    return ap.parse_args(argv)


def need(a: argparse.Namespace, *names: str) -> None:
    missing = [f"--{n}" for n in names if getattr(a, n) is None]
    if missing:
        raise FtqError(f"{a.command} needs {' '.join(missing)}", 2)


def seeds_of(a: argparse.Namespace) -> list:
    try:
        seeds = [int(x) for x in str(a.p_seeds).split(",") if x != ""]
    except ValueError:
        raise FtqError(f"--p_seeds {a.p_seeds!r} is not a comma list of integers", 2) from None
    bad = [s for s in seeds if s not in CONTROL_SEEDS]
    if not seeds or bad:
        raise FtqError(f"--p_seeds {a.p_seeds!r}: the teacher adapters are p0 and p1", 2)
    return seeds


def sft_configs(adapters: str) -> dict:
    """{seed: train_config.json of the real adapter s<seed>}; a missing adapter is an error naming it."""
    out = {}
    for k in SFT_SEEDS:
        p = Path(adapters) / f"s{k}" / "train_config.json"
        if not p.is_file():
            raise FtqError(f"missing the real adapter s{k} ({p}): run_ftgrid.sh stage 1 trains it (ML-1M: run_gateft.sh), "
                           "FT-Q reads its recipe")
        out[k] = read_json(p)
    return out


def cmd_recipe(a) -> list:
    need(a, "adapters", "model", "variant", "train_ref")
    adapters = a.adapters.rstrip("/")
    split = read_json(a.split) if a.split else None
    base = check_recipe(sft_configs(adapters), adapters=adapters, model=a.model, variant=a.variant, train_ref=a.train_ref,
                        split=split, split_recipe=a.split_recipe)
    if a.train_file:
        check_train_file(a.train_ref, a.train_file)
    return recipe_flags(base)


def cmd_scoring(a) -> list:
    need(a, "scores", "adapters", "data", "model", "variant")
    scores, adapters = a.scores.rstrip("/"), a.adapters.rstrip("/")
    like = Path(scores) / "s0" / "like"
    for name in ("run.key", "report.json"):
        if not (like / name).is_file():
            raise FtqError(f"missing the real like pass of s0 ({like / name}): run_ftgrid.sh stage 3 scores it, FT-Q reads "
                           "its arguments")
    cfg = report_config(read_json(like / "report.json"), str(like / "report.json"))
    return check_scoring_record((like / "run.key").read_text(encoding="utf-8"), cfg, data_sha1=file_sha1(a.data),
                                model=a.model, variant=a.variant, lora=f"{adapters}/s0",
                                lora_weights_sha1=weights_sha1(Path(adapters) / "s0"))


def cmd_verify_adapter(a) -> None:
    need(a, "adapters", "ref_adapters", "p_seeds", "train", "teacher_manifest")
    adapters = a.adapters.rstrip("/")
    ref = read_json(Path(a.ref_adapters.rstrip("/")) / "s0" / "train_config.json")
    man = read_json(a.teacher_manifest)
    sha = man["train_q"]["sha1"]
    if file_sha1(a.train) != sha:
        raise FtqError(f"{a.train} is not the file {a.teacher_manifest} records (sha1 {sha})")
    for k in seeds_of(a):
        out = f"{adapters}/p{k}"
        cfg = read_json(Path(out) / "train_config.json")
        check_same_training(ref, cfg, train=a.train, out=out, seed=k, where=f"{out}/train_config.json")
        if a.write:
            write_provenance(out, p_seed=k, train_q_sha1=sha, manifest_sha1=file_sha1(a.teacher_manifest),
                             args=train_args(cfg))
        check_provenance(out, p_seed=k, train_q_sha1=sha)


def cmd_verify_scores(a) -> None:
    need(a, "scores", "ref_scores", "adapters", "p_seeds")
    scores, adapters = a.scores.rstrip("/"), a.adapters.rstrip("/")
    ref = report_config(read_json(Path(a.ref_scores.rstrip("/")) / "s0" / "like" / "report.json"), "the like pass of s0")
    for k in seeds_of(a):
        run = Path(scores) / f"p{k}" / "like" / "report.json"
        check_same_scoring(ref, report_config(read_json(run), str(run)), lora=f"{adapters}/p{k}", where=str(run))


def cmd_link(a) -> list:
    need(a, "real_scores", "q_scores")
    return link_models(a.real_scores, a.q_scores, [m for m in a.models.split(",") if m], allow_copy=a.allow_copy)


def cmd_record(a) -> int:
    need(a, "pilot_log", "files")
    if a.do_print:
        print("\n".join(record_lines(a.files, a.root)))
        return 0
    missing = record_missing(a.pilot_log, a.files, a.root)
    if missing:
        raise FtqError(f"FT-Q record (Amendment 3 addendum 8 section 4): the sha1 of these files is not in {a.pilot_log}: "
                       f"{missing}; run `python -m src.confrec.ftq_panel record --pilot_log {a.pilot_log} --files "
                       f"{' '.join(a.files)} --print`, record the lines in the pilot log and push it", 4)
    print(f"FT-Q record OK: the sha1 of {', '.join(a.files)} is in {a.pilot_log}")
    return 0


def cmd_info(a) -> list:
    need(a, "selection", "split", "gate")
    sel = read_json(a.selection)
    split = read_json(a.split) if Path(a.split).is_file() else {}
    gate = read_json(a.gate) if Path(a.gate).is_file() else {}
    return [str(sel.get("decision") or ""), str(sel.get("gate_ft_prompt") or ""), str(split.get("variant") or ""),
            str(gate.get("decision") or "missing")]


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        if a.command == "build":
            need(a, "domain", "train", "qhat", "qmanifest", "out_dir")
            if a.domain not in DOMAINS:
                raise FtqError(f"--domain {a.domain!r}: FT-Q runs on {', '.join(DOMAINS)}", 2)
            man = build(a.train, a.qhat, a.qmanifest, a.out_dir, domain=a.domain, split=a.split)
            print(json.dumps(man, allow_nan=False))
            print(f"ftq_panel build: {a.domain}: the teacher labels {man['k']} of {man['n']} TRAIN examples 1 (beta = "
                  f"{man['beta']:.4f}) and agrees with the real label on "
                  f"{man['agreement_with_real_labels']['share_teacher_equals_real_label']:.1%}", file=sys.stderr)
        elif a.command == "recipe":
            print("\n".join(cmd_recipe(a)))
        elif a.command == "scoring":
            print("\n".join(cmd_scoring(a)))
        elif a.command == "verify_adapter":
            cmd_verify_adapter(a)
            print(f"adapters p{a.p_seeds}: the recorded arguments equal the real s0's except {', '.join(REPLACED)}; "
                  "provenance OK")
        elif a.command == "verify_scores":
            cmd_verify_scores(a)
            print(f"like pass of p{a.p_seeds}: the recorded scoring arguments equal the real like pass of s0's except lora")
        elif a.command == "link":
            print("\n".join(cmd_link(a)))
        elif a.command == "info":
            print("\n".join(cmd_info(a)))
        else:
            return cmd_record(a)
    except FtqError as e:
        print(f"ftq_panel {a.command}: {e}", file=sys.stderr)
        return e.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
