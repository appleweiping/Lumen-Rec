"""Prior-offset LoRA of the nested method slot (idea-stage/PREREG_AMENDMENT_3.md section 7, FT-M; addendum 2 item 2).

Model (section 7). During training the Yes-token logit of the position that predicts the answer is shifted before the
full-vocabulary softmax, l'_Yes = l_Yes + b * z(q-hat): z = the TRAIN-standardised prior-only item mean of the example
(mean and SD over the TRAIN examples of the domain, recorded), b a learnable scalar initialised at 0, no gradient into
q-hat. The loss is Gate-FT's last-token cross-entropy of the answer token under l' (full-vocabulary softmax, so the
Yes + No mass stays constrained and E1 keeps its meaning). At test time the ranking score is the scorer's
logit(Yes) - logit(No) + b * z(q-hat) (src/confrec/ftmethod_report.py).

    # stage 1 of run_ftmethod.sh (CPU): q-hat of every TRAIN example and EVAL pair and the TRAIN standardisation
    python -m src.confrec.train_lora_offset qhat --domain ml1m --panels outputs/confrec/ftgrid/panels/ml1m \
        --raw data/raw --out_dir outputs/confrec/ftmethod/ml1m
    # stage 3 (GPU): one prior-offset adapter, the section-2 recipe of its SFT comparator (same seed), standard mode
    python -m src.confrec.train_lora_offset train --train outputs/confrec/ftgrid/panels/ml1m/train.jsonl \
        --model <Qwen3-8B> --out outputs/confrec/ftmethod/ml1m/adapters/o0 --variant V1 --seed 0 \
        --qhat outputs/confrec/ftmethod/ml1m/train_qhat.csv.gz --manifest outputs/confrec/ftmethod/ml1m/qhat_manifest.json \
        --sft_adapter outputs/confrec/gateft/adapters/s0 [--max_len 1024 --bsz 8 --grad_accum 4] [--b_lr LR]

qhat. q-hat(u, i, t) = forensics.prior_means' mean_prior_shrunk (A3 section 3, k = 5): (s + k g) / (n + k), s and n = the
sum and count of other users' first ratings of i strictly before the candidate's timestamp t, g = the mean of all other
users' first ratings strictly before t (raw events through forensics.load_raw_events, one forensics.scan_events pass over
the TRAIN and EVAL pairs together: a pair's value depends only on its own user, item and t, so this equals one scan per
role, and the EVAL values are those of ftgrid_report's E-A reference). TRAIN examples precede T, so no rating at or after
T enters a TRAIN-side feature. The standardisation constants are the mean and the population SD (ddof 0) of the finite
TRAIN q-hat over every example of train.jsonl; z = (q-hat - mean) / sd, and z = 0 (no offset) where q-hat is not finite
(none is expected: the shrinkage target covers every pair with an earlier rating anywhere). Outputs (--out_dir, written
only when their bytes change, the manifest last; no wall-clock, host or absolute path):
    train_qhat.csv.gz / eval_qhat.csv.gz   one row per example / pair in panel order: source_event_id, cand_idx,
                                           user_id, item_id, ts, label, q_hat, q_hat_unshrunk, n_prior, global_prior, z
    qhat_manifest.json                     the standardisation constants, the sha1 of both csv files, of train.jsonl,
                                           eval.jsonl and ftgrid_split.json, and the code sha1 (FREEZE `method`,
                                           `ftgrid_freeze --print --stage method --split <manifest>`, addendum 2 item 2)

train. The frozen pieces are reused by import: train_lora_yesno's row shuffle (random.Random(seed)), check_training_panel,
YesNoSet, collate and training_arguments (so the TrainingArguments are the section-2 recipe's), and lora_trainer's
last_token_trainer / last_token_loss. Composition:
  * PriorOffsetSet = YesNoSet (standard mode) built over the single-candidate restriction of every train row (a pointwise
    prompt does not depend on the row's other candidates, so the examples, their order and the overlength skips are
    exactly YesNoSet(rows)'s, tested); a generator feeding YesNoSet records which example each item is, and every kept
    item carries its z (joined by (user_id, item_id) from train_qhat.csv.gz, whose sha1 and constants are checked
    against the manifest, and train.jsonl against the manifest's sha1).
  * collate_offset = train_lora_yesno.collate plus a float32 tensor "z".
  * prior_offset_trainer(Trainer) subclasses lora_trainer.last_token_trainer(Trainer); its compute_loss hands the
    parent's compute_loss (that is last_token_loss, with its checks and its stock sum / num_items_in_batch rule) a view
    of the model whose kept logits equal the model's, in float32, except the Yes column of the position that predicts
    the answer: + b * z (`shift_yes`). With b = 0 the loss, the gradients and so the run are SFT's (tested).
  * b = torch.nn.Parameter(0.0) registered on the PEFT model as `prior_offset_b`, so the Trainer's optimizer updates it
    (the optimizer is built before train(), which reuses it, and must hold b); its name carries no "lora_", so
    save_pretrained writes the LoRA weights only and pyes_scorer --lora loads the adapter exactly as an SFT adapter
    (asserted on the written file).
  * --sft_adapter (the SFT comparator of the same seed: ML-1M Gate-FT's) must share the recipe: train file bytes, seed,
    variant, mode, max_len, epochs, lr, micro-batch, accumulation, LoRA rank, window, backbone and, once the examples are
    built, their count and overlength skips (SystemExit before the model is loaded otherwise).
  * --b_lr (default: none) puts b in an optimizer group of its own with that learning rate; without it b shares the
    LoRA weights' group (the recipe's lr; see the report: with AdamW the total movement of b is bounded by the summed
    learning rates, about 0.5 * lr * steps).
Outputs next to the adapter: train_report.json (before the model is loaded, as train_lora_yesno), train_config.json
(train_lora_yesno's keys, loss "last_token_prior_offset", and prior_offset {b, ...}) and offset.json (b, the
standardisation constants, the yes token id, the manifest and data sha1s, the code sha1; read by ftmethod_report.py).
torch, transformers and peft are imported inside the functions that need them, never at module import.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import random
from pathlib import Path
from typing import NamedTuple

import numpy as np

from src.confrec import forensics as fx
from src.confrec.build_rated_panels import commit
from src.confrec.prompting import GATE_VARIANTS
from src.confrec.split_panel import filter_candidates
from src.confrec.stats import strict_json

SPEC = "idea-stage/PREREG_AMENDMENT_3.md section 7 (FT-M) with sections 2 and 3; PREREG_AMENDMENT_3_ADDENDUM_2.md item 2"
SHRINK_K = 5.0                       # A3 section 3: q-hat shrunk with k = 5 (forensics.prior_means)
SD_DDOF = 0                          # population SD of the TRAIN q-hat
MANIFEST = "qhat_manifest.json"
TRAIN_QHAT, EVAL_QHAT = "train_qhat.csv.gz", "eval_qhat.csv.gz"
OFFSET_JSON = "offset.json"
MANIFEST_FORMAT, OFFSET_FORMAT = "ftmethod_qhat_v1", "ftmethod_offset_v1"
OFFSET_PARAM = "prior_offset_b"      # b on the PEFT model; no "lora_" in the name, so never part of the saved adapter
QHAT_COLS = ("source_event_id", "cand_idx", "user_id", "item_id", "ts", "label", "q_hat", "q_hat_unshrunk",
             "n_prior", "global_prior", "z")
# recipe keys the prior-offset run shares with its SFT comparator (train_config.json of train_lora_yesno)
RECIPE_KEYS = ("mode", "variant", "seed", "max_len", "epochs", "lr", "bsz", "grad_accum", "lora_r", "max_examples")
CODE_FILES = ("train_lora_offset.py", "lora_trainer.py", "train_lora_yesno.py", "prompting.py", "forensics.py")


class Example(NamedTuple):
    ev: str
    cand: int
    user: str
    item: str
    ts: float
    label: int


# ---------------------------------------------------------------- small helpers
def read_rows(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def code_sha1() -> dict:
    here = Path(__file__).resolve().parent
    return {name: fx.file_sha1(here / name) for name in CODE_FILES if (here / name).is_file()}


def _cell(x) -> str:
    """A number as the csv writes it: integral values as integers, others as the shortest round-trip repr, NaN empty."""
    x = float(x)
    if not math.isfinite(x):
        return ""
    return str(int(x)) if x.is_integer() and abs(x) < 2 ** 53 else repr(x)


def _float(s: str) -> float:
    return float(s) if s not in ("", None) else float("nan")


def gz_bytes(text: str) -> bytes:
    """Deterministic gzip (no file name, mtime 0)."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as f:
        f.write(text.encode("utf-8"))
    return buf.getvalue()


def write_if_changed(path: Path, data: bytes) -> bool:
    """Atomic write that leaves `path` (and its mtime) untouched when it already holds these bytes."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    return commit(tmp, path)


def json_bytes(obj) -> bytes:
    return (json.dumps(strict_json(obj), indent=2, allow_nan=False) + "\n").encode("utf-8")


# ---------------------------------------------------------------- q-hat and the TRAIN standardisation
def panel_examples(rows: list, name: str) -> list:
    """Every candidate of a rated panel in row / candidate order; (user_id, item_id) must be unique (one row per user,
    distinct items in a row) and every candidate timestamp finite."""
    out, seen = [], set()
    for r in rows:
        u = str(r["user_id"])
        ev = str(r.get("source_event_id", u))
        ids, ts, lab = r["candidate_item_ids"], r["candidate_timestamps"], r["candidate_labels"]
        if not len(ids) == len(ts) == len(lab):
            raise SystemExit(f"{name}: user {u!r}: candidate lists misaligned")
        for j, (i, t, y) in enumerate(zip(ids, ts, lab)):
            key = (u, str(i))
            if key in seen:
                raise SystemExit(f"{name}: (user_id, item_id) {key} occurs twice")
            seen.add(key)
            t = float(t)
            if not math.isfinite(t):
                raise SystemExit(f"{name}: candidate {key} has no finite timestamp")
            out.append(Example(ev, j, u, str(i), t, int(y)))
    return out


def qhat_arrays(examples: list, events: dict, shrink_k: float = SHRINK_K) -> tuple[dict, dict]:
    """forensics.prior_means over the examples (one scan_events pass): mean_prior_shrunk is q-hat."""
    users, items = [e.user for e in examples], [e.item for e in examples]
    times = np.array([e.ts for e in examples], float)
    scan = fx.scan_events(events, set(items), set(zip(users, items)))
    pm = fx.prior_means(scan, users, items, times, shrink_k)
    info = {k: int(scan[k]) for k in ("n_raw_events", "n_first_events", "n_candidate_pairs_not_in_raw")}
    return pm, info


def standardisation(q) -> dict:
    """Mean and population SD of the finite TRAIN q-hat (A3 section 7: TRAIN-standardised)."""
    q = np.asarray(q, float)
    fin = q[np.isfinite(q)]
    if len(fin) < 2:
        raise SystemExit("fewer than 2 finite TRAIN q-hat values: z is undefined")
    sd = float(fin.std(ddof=SD_DDOF))
    if not sd > 0:
        raise SystemExit("the TRAIN q-hat has SD 0: z is undefined")
    return {"mean": float(fin.mean()), "sd": sd, "ddof": SD_DDOF, "n_examples": int(len(q)), "n_finite": int(len(fin)),
            "n_nonfinite_z_imputed_0": int(len(q) - len(fin))}


def zscore(q, consts: dict) -> np.ndarray:
    """z = (q-hat - mean) / sd with the recorded TRAIN constants; 0 where q-hat is not finite (no offset)."""
    q = np.asarray(q, float)
    with np.errstate(invalid="ignore"):
        z = (q - float(consts["mean"])) / float(consts["sd"])
    return np.where(np.isfinite(z), z, 0.0)


def qhat_csv_bytes(examples: list, pm: dict, z) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(QHAT_COLS)
    for k, e in enumerate(examples):
        w.writerow([e.ev, e.cand, e.user, e.item, _cell(e.ts), e.label, _cell(pm["mean_prior_shrunk"][k]),
                    _cell(pm["mean_prior"][k]), _cell(pm["n_prior"][k]), _cell(pm["global_prior"][k]), _cell(z[k])])
    return gz_bytes(buf.getvalue())


def read_qhat(path) -> dict:
    """{(user_id, item_id): row} of a q-hat csv (train_qhat.csv.gz or eval_qhat.csv.gz), in file order."""
    out = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        if tuple(rd.fieldnames or ()) != QHAT_COLS:
            raise SystemExit(f"{path}: columns {rd.fieldnames} are not {list(QHAT_COLS)}")
        for r in rd:
            key = (r["user_id"], r["item_id"])
            if key in out:
                raise SystemExit(f"{path}: {key} occurs twice")
            out[key] = {"source_event_id": r["source_event_id"], "cand_idx": int(r["cand_idx"]), "ts": _float(r["ts"]),
                        "label": int(r["label"]), "q_hat": _float(r["q_hat"]), "z": _float(r["z"])}
    return out


def build_qhat(a) -> dict:
    """The qhat subcommand: q-hat of every TRAIN example and EVAL pair, the TRAIN constants and the manifest."""
    panels, out = Path(a.panels), Path(a.out_dir)
    split_p = panels / "ftgrid_split.json"
    if not split_p.is_file():
        raise SystemExit(f"{split_p}: no ftgrid_split.json (run_ftgrid.sh stage 0)")
    split = json.loads(split_p.read_text(encoding="utf-8"))
    if split.get("domain") != a.domain:
        raise SystemExit(f"{split_p} is the split of {split.get('domain')!r}, not of {a.domain!r}")
    T = float(split["T"])
    sha = {}
    for name in ("train.jsonl", "eval.jsonl"):
        p = panels / name
        if not p.is_file():
            raise SystemExit(f"{p}: missing (run_ftgrid.sh stage 0)")
        sha[name] = fx.file_sha1(p)
        rec = (split.get("files") or {}).get(name)
        if rec != sha[name]:
            raise SystemExit(f"{p}: sha1 {sha[name]} is not the one ftgrid_split.json records ({rec})")
    train_rows, eval_rows = read_rows(panels / "train.jsonl"), read_rows(panels / "eval.jsonl")
    tr, ev = panel_examples(train_rows, "train.jsonl"), panel_examples(eval_rows, "eval.jsonl")
    late = sum(e.ts >= T for e in tr)
    if late:
        raise SystemExit(f"train.jsonl holds {late} examples at or after T = {T}: a TRAIN-side feature would see them")
    shared = {e.user for e in tr} & {e.user for e in ev}
    if shared:
        raise SystemExit(f"TRAIN and EVAL share {len(shared)} users (A3 section 1)")
    source = str(train_rows[0].get("source") or a.domain)
    print(f"[ftmethod qhat] {a.domain}: {len(tr)} TRAIN examples, {len(ev)} EVAL pairs; raw events of {source!r}",
          flush=True)
    pm, info = qhat_arrays(tr + ev, fx.load_raw_events(a.raw, source))
    n = len(tr)
    pm_tr = {k: np.asarray(v[:n], float) for k, v in pm.items()}
    pm_ev = {k: np.asarray(v[n:], float) for k, v in pm.items()}
    consts = standardisation(pm_tr["mean_prior_shrunk"])
    z_tr, z_ev = zscore(pm_tr["mean_prior_shrunk"], consts), zscore(pm_ev["mean_prior_shrunk"], consts)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, ex, p_, z_ in ((TRAIN_QHAT, tr, pm_tr, z_tr), (EVAL_QHAT, ev, pm_ev, z_ev)):
        data = qhat_csv_bytes(ex, p_, z_)
        files[name] = hashlib.sha1(data).hexdigest()
        write_if_changed(out / name, data)
    manifest = {
        "format": MANIFEST_FORMAT, "spec": SPEC, "domain": a.domain, "source": source, "T": T,
        "q_hat": {"estimator": "forensics.prior_means mean_prior_shrunk (one forensics.scan_events pass over the TRAIN "
                               "examples and EVAL pairs)", "shrink_k": SHRINK_K,
                  "definition": "q-hat(u, i, t) = (s + k g) / (n + k): s, n = sum and count of other users' first "
                                "ratings of i strictly before t; g = mean of all other users' first ratings strictly "
                                "before t (A3 section 3; addendum 1 item 2)"},
        "standardisation": {**consts, "over": "every example of train.jsonl (after the cap), finite q-hat; population "
                                              "SD; z = (q-hat - mean) / sd, z = 0 where q-hat is not finite"},
        "train": {"file": "train.jsonl", "sha1": sha["train.jsonl"], "n_examples": len(tr),
                  "n_users": len({e.user for e in tr}), "n_qhat_nonfinite": int((~np.isfinite(pm_tr["mean_prior_shrunk"]))
                                                                               .sum())},
        "eval": {"file": "eval.jsonl", "sha1": sha["eval.jsonl"], "n_pairs": len(ev), "n_users": len({e.user for e in ev}),
                 "n_qhat_nonfinite": int((~np.isfinite(pm_ev["mean_prior_shrunk"])).sum())},
        "split_sha1": fx.file_sha1(split_p), "raw": info, "files": files, "code_sha1": code_sha1()}
    write_if_changed(out / MANIFEST, json_bytes(manifest))
    st = manifest["standardisation"]
    print(f"[ftmethod qhat] {a.domain}: TRAIN q-hat mean {st['mean']:.6f}, SD {st['sd']:.6f} over {st['n_finite']} "
          f"examples ({st['n_nonfinite_z_imputed_0']} imputed z = 0); wrote {out / MANIFEST}", flush=True)
    return manifest


def load_train_z(train_path, qhat_path, manifest_path) -> dict:
    """z of every train.jsonl example from the stage-1 files, after checking them against the manifest (train.jsonl
    bytes, train_qhat.csv.gz bytes, one q-hat row per example with the same timestamp and label, z recomputed from the
    recorded constants equal to the stored z)."""
    mp = Path(manifest_path)
    man = json.loads(mp.read_text(encoding="utf-8"))
    if man.get("format") != MANIFEST_FORMAT:
        raise SystemExit(f"{mp}: not a {MANIFEST_FORMAT} manifest")
    tsha, qsha = fx.file_sha1(train_path), fx.file_sha1(qhat_path)
    if tsha != man["train"]["sha1"]:
        raise SystemExit(f"{train_path}: sha1 {tsha} is not the train.jsonl the manifest standardised "
                         f"({man['train']['sha1']})")
    if qsha != man["files"][TRAIN_QHAT]:
        raise SystemExit(f"{qhat_path}: sha1 {qsha} is not the manifest's {TRAIN_QHAT} ({man['files'][TRAIN_QHAT]})")
    q = read_qhat(qhat_path)
    ex = panel_examples(read_rows(train_path), str(train_path))
    if len(ex) != len(q):
        raise SystemExit(f"{qhat_path}: {len(q)} rows for {len(ex)} train.jsonl examples")
    bad = [(e.user, e.item) for e in ex if (e.user, e.item) not in q or q[(e.user, e.item)]["ts"] != e.ts
           or q[(e.user, e.item)]["label"] != e.label]
    if bad:
        raise SystemExit(f"{qhat_path}: {len(bad)} examples without their q-hat row (or with another timestamp / label), "
                         f"e.g. {bad[:3]}")
    consts = man["standardisation"]
    qh = np.array([q[(e.user, e.item)]["q_hat"] for e in ex], float)
    z = zscore(qh, consts)
    stored = np.array([q[(e.user, e.item)]["z"] for e in ex], float)
    if not np.array_equal(z, stored):
        raise SystemExit(f"{qhat_path}: stored z differs from (q-hat - mean) / sd with the manifest's constants")
    keys = [(e.user, e.item) for e in ex]
    return {"z_of": dict(zip(keys, z.tolist())), "imputed": {k for k, v in zip(keys, qh) if not math.isfinite(v)},
            "standardisation": consts, "domain": man["domain"], "manifest_sha1": fx.file_sha1(mp), "train_sha1": tsha,
            "qhat_sha1": qsha}


# ---------------------------------------------------------------- the SFT comparator (section 7: same recipe)
def recipe_problems(cfg: dict, a, hist_len_used: int) -> list:
    """How the run `a` differs from its SFT comparator's train_config.json (empty = same recipe and TRAIN file)."""
    probs = []
    if cfg.get("loss") != "last_token":
        probs.append(f"loss {cfg.get('loss')!r} is not the section-2 last-token loss")
    for k in RECIPE_KEYS:
        if cfg.get(k) != getattr(a, k):
            probs.append(f"{k}: SFT {cfg.get(k)!r} != {getattr(a, k)!r}")
    if cfg.get("hist_len_used") != hist_len_used:
        probs.append(f"history window: SFT {cfg.get('hist_len_used')!r} != {hist_len_used!r}")
    if Path(str(cfg.get("model", ""))).name != Path(str(a.model)).name:
        probs.append(f"backbone: SFT {cfg.get('model')!r} != {a.model!r}")
    sft_train = Path(str(cfg.get("train", "")))
    if not sft_train.is_file():
        probs.append(f"the SFT's train file {sft_train} does not exist: the TRAIN set cannot be compared")
    elif fx.file_sha1(sft_train) != fx.file_sha1(a.train):
        probs.append(f"TRAIN file: SFT {sft_train} and {a.train} differ in bytes")
    return probs


def sft_comparator(a, hist_len_used: int) -> dict:
    """The SFT adapter of the same seed (--sft_adapter) and its recipe; SystemExit when the recipes differ."""
    cfg_p = Path(a.sft_adapter) / "train_config.json"
    if not cfg_p.is_file():
        raise SystemExit(f"--sft_adapter {a.sft_adapter}: no train_config.json (the section-2 SFT adapter of seed "
                         f"{a.seed}; ML-1M: Gate-FT's)")
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    probs = recipe_problems(cfg, a, hist_len_used)
    if probs:
        raise SystemExit("A3 section 7: the prior-offset run shares its SFT comparator's TRAIN set and recipe; "
                         f"{cfg_p} differs: " + "; ".join(probs))
    return {"dir": str(a.sft_adapter), "config": cfg, "config_sha1": fx.file_sha1(cfg_p)}


def example_count_problems(cfg: dict, n_examples: int, n_skipped: int) -> list:
    probs = []
    if cfg.get("n_examples") != n_examples:
        probs.append(f"examples: SFT {cfg.get('n_examples')!r} != {n_examples}")
    if cfg.get("n_skipped_overlength") != n_skipped:
        probs.append(f"overlength skips: SFT {cfg.get('n_skipped_overlength')!r} != {n_skipped}")
    return probs


# ---------------------------------------------------------------- the dataset (torch only through train_lora_yesno)
def single_candidate_rows(rows: list) -> tuple[list, list]:
    """(one row per candidate, restricted to that candidate by split_panel.filter_candidates, and its (user_id,
    item_id)) in row / candidate order: the order in which YesNoSet renders the examples of `rows`."""
    singles, keys = [], []
    for r in rows:
        n = len(r["candidate_labels"])
        for j in range(n):
            singles.append(filter_candidates(r, [k == j for k in range(n)]))
            keys.append((str(r["user_id"]), str(r["candidate_item_ids"][j])))
    return singles, keys


def prior_offset_set(rows: list, z_of: dict, tok, hist_len, max_len: int, variant: str = "V0"):
    """train_lora_yesno.YesNoSet (standard mode) over `rows` whose every kept example carries its z.

    YesNoSet is fed the single-candidate rows through a generator that notes, row by row, whether the example was kept
    (the overlength rule skips), so .keys / .z follow .items exactly. The items equal YesNoSet(rows, ...).items."""
    from src.confrec.train_lora_yesno import YesNoSet

    class PriorOffsetSet(YesNoSet):
        def __init__(self):
            singles, keys = single_candidate_rows(rows)
            kept = []

            def feed():
                for r in singles:
                    n0 = len(self.items)
                    yield r
                    kept.append(len(self.items) - n0)
            super().__init__(feed(), tok, "standard", hist_len, max_len, variant)
            if len(kept) != len(singles) or any(k not in (0, 1) for k in kept) or sum(kept) != len(self.items) \
                    or len(kept) - sum(kept) != self.n_skipped:
                raise AssertionError("PriorOffsetSet lost track of YesNoSet's examples")
            self.keys = [k for k, n in zip(keys, kept) if n]
            missing = [k for k in self.keys if k not in z_of]
            if missing:
                raise SystemExit(f"{len(missing)} training examples have no z (q-hat row), e.g. {missing[:3]}")
            self.z = [float(z_of[k]) for k in self.keys]
            self.yes_id = self.answer["Yes"]

        def __getitem__(self, i):
            item = super().__getitem__(i)
            item["z"] = self.z[i]
            return item

        def truncate(self, n: int) -> None:
            """train_lora_yesno's --max_examples (the first n examples), applied to the z list too."""
            self.items, self.keys, self.z = self.items[:n], self.keys[:n], self.z[:n]

    return PriorOffsetSet()


def collate_offset(batch, pad_id):
    """train_lora_yesno.collate (left padding, answer last) plus z as a float32 tensor."""
    import torch

    from src.confrec.train_lora_yesno import collate
    out = collate(batch, pad_id)
    out["z"] = torch.tensor([float(b["z"]) for b in batch], dtype=torch.float32)
    return out


# ---------------------------------------------------------------- the loss (torch)
def shift_yes(logits, yes_id: int, b, z):
    """(B, V) logits -> the same logits with column yes_id increased by b * z: l'_Yes = l_Yes + b z. z is detached (no
    gradient into q-hat); b keeps its graph (a Parameter or a float)."""
    import torch
    if logits.ndim != 2:
        raise ValueError(f"logits must be (batch, vocab), got {tuple(logits.shape)}")
    z = torch.as_tensor(z).detach().to(device=logits.device, dtype=logits.dtype).reshape(-1)
    if z.shape[0] != logits.shape[0]:
        raise ValueError(f"{z.shape[0]} z values for a batch of {logits.shape[0]}")
    col = torch.zeros(logits.shape[-1], dtype=logits.dtype, device=logits.device)
    col[yes_id] = 1.0
    b = torch.as_tensor(b, dtype=logits.dtype, device=logits.device)
    return logits + (b * z)[:, None] * col


class PriorOffsetModel:
    """The model as lora_trainer.last_token_loss sees it in the prior-offset run: the same forward, kept logits in
    float32, with the Yes logit of the position that predicts the answer (index -2 of the kept positions) shifted."""

    def __init__(self, model, yes_id: int, b, z):
        self.model, self.yes_id, self.b, self.z = model, yes_id, b, z

    def __call__(self, input_ids, attention_mask, logits_to_keep):
        import torch
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep)
        lg = out.logits.float()
        last = shift_yes(lg[:, -2, :], self.yes_id, self.b, self.z)
        out.logits = torch.cat([lg[:, :-2, :], last[:, None, :], lg[:, -1:, :]], dim=1)
        return out


def own_param_group(optimizer, b, lr: float) -> None:
    """--b_lr: b leaves the Trainer's parameter groups for a group of its own (lr, no weight decay); the learning-rate
    schedule, created afterwards, scales it like every other group. Groups left empty are dropped."""
    found = 0
    for g in optimizer.param_groups:
        keep = [p for p in g["params"] if p is not b]
        found += len(g["params"]) - len(keep)
        g["params"] = keep
    if found != 1:
        raise RuntimeError(f"b appears {found} times in the optimizer's parameter groups (expected once)")
    optimizer.param_groups[:] = [g for g in optimizer.param_groups if g["params"]]
    optimizer.add_param_group({"params": [b], "lr": float(lr), "weight_decay": 0.0})


def prior_offset_trainer(trainer_base, yes_id: int, b, b_lr: float | None = None):
    """`trainer_base` is transformers.Trainer: the subclass of lora_trainer.last_token_trainer(trainer_base) whose loss
    is last_token_loss on the model view with the shifted Yes logit (z from the batch)."""
    from src.confrec.lora_trainer import last_token_trainer

    class PriorOffsetTrainer(last_token_trainer(trainer_base)):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            view = PriorOffsetModel(model, yes_id, b, inputs["z"])
            return super().compute_loss(view, inputs, return_outputs, num_items_in_batch)

        def create_optimizer(self, *args, **kwargs):
            opt = super().create_optimizer(*args, **kwargs)
            if b_lr is not None:
                own_param_group(self.optimizer, b, b_lr)
            return opt

    return PriorOffsetTrainer


def attach_prior_offset(model):
    """b = 0 as a float32 Parameter registered on the (PEFT) model, on the device of its output head."""
    import torch
    head = model.get_output_embeddings()
    b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device=head.weight.device))
    model.register_parameter(OFFSET_PARAM, b)
    if not any(p is b and p.requires_grad for _, p in model.named_parameters()):
        raise RuntimeError(f"{OFFSET_PARAM} is not a trainable parameter of the model")
    return b


def b_in_optimizer(optimizer, b) -> bool:
    return any(p is b for g in optimizer.param_groups for p in g["params"])


def saved_adapter_keys(out: Path) -> list:
    """Tensor names of the written adapter (safetensors only; [] when absent)."""
    p = Path(out) / "adapter_model.safetensors"
    if not p.is_file():
        return []
    from safetensors import safe_open
    with safe_open(str(p), framework="pt") as f:
        return list(f.keys())


# ---------------------------------------------------------------- records
def write_train_report(out: Path, a, panel: dict, ds, n_built: int, zinfo: dict, dry_run: bool = False) -> None:
    """train_lora_yesno's train_report.json keys, plus the z of the kept examples."""
    z = np.asarray(ds.z, float)
    rep = dict(n_examples=len(ds), n_examples_built=n_built, n_skipped_overlength=ds.n_skipped, max_len=a.max_len,
               max_prompt_len=ds.max_prompt_len, mode=a.mode, variant=a.variant, hist_len=panel["hist_len_used"],
               panel_kind=panel["panel_kind"], max_history_len_in_panel=panel["max_history_len_in_panel"],
               answer_ids=ds.answer, train=a.train, model=a.model, qhat=a.qhat, manifest=a.manifest,
               qhat_manifest_sha1=zinfo["manifest_sha1"], n_z_imputed=sum(k in zinfo["imputed"] for k in ds.keys),
               z_mean_kept=float(z.mean()) if len(z) else None, z_sd_kept=float(z.std()) if len(z) else None,
               **({"dry_run": True} if dry_run else {}))
    (Path(out) / "train_report.json").write_text(json.dumps(strict_json(rep), indent=2), encoding="utf-8")


def write_offset_and_config(out: Path, a, panel: dict, ds, zinfo: dict, sft: dict, b: float,
                            dry_run: bool = False) -> dict:
    """offset.json (read by ftmethod_report.py) and train_config.json (train_lora_yesno's keys; b recorded, A3
    section 7)."""
    off = {"format": OFFSET_FORMAT, "spec": SPEC, "domain": zinfo["domain"], "seed": a.seed, "b": float(b), "b_init": 0.0,
           "b_lr": a.b_lr, "b_lr_rule": ("--b_lr: an optimizer group of its own" if a.b_lr is not None else
                                         "none: b is in the Trainer's group of the LoRA weights (lr = --lr)"),
           "yes_token_id": int(ds.yes_id), "answer_ids": ds.answer, "standardisation": zinfo["standardisation"],
           "qhat_manifest_sha1": zinfo["manifest_sha1"], "train_qhat_sha1": zinfo["qhat_sha1"],
           "train_sha1": zinfo["train_sha1"], "n_examples": len(ds),
           "n_z_imputed": sum(k in zinfo["imputed"] for k in ds.keys), "sft_adapter": sft["dir"],
           "sft_train_config_sha1": sft["config_sha1"],
           "test_time_score": "the scorer's logit(Yes) - logit(No) + b * z(q-hat), z with these standardisation constants",
           "code_sha1": code_sha1(), **({"dry_run": True} if dry_run else {})}
    out = Path(out)
    (out / OFFSET_JSON).write_text(json.dumps(strict_json(off), indent=2, allow_nan=False), encoding="utf-8")
    cfg = {**vars(a), "loss": "last_token_prior_offset", "hist_len_used": panel["hist_len_used"],
           "panel_kind": panel["panel_kind"], "max_history_len_in_panel": panel["max_history_len_in_panel"],
           "n_examples": len(ds), "n_skipped_overlength": ds.n_skipped,
           "prior_offset": {"b": float(b), "b_init": 0.0, "b_lr": a.b_lr, "yes_token_id": int(ds.yes_id),
                            "standardisation": zinfo["standardisation"], "qhat_manifest_sha1": zinfo["manifest_sha1"],
                            "offset_json": OFFSET_JSON},
           **({"dry_run": True} if dry_run else {})}
    (out / "train_config.json").write_text(json.dumps(strict_json(cfg), indent=2), encoding="utf-8")
    return off


# ---------------------------------------------------------------- training (GPU)
def train(a) -> dict:
    rows = read_rows(a.train)
    random.Random(a.seed).shuffle(rows)                      # train_lora_yesno.main's example order
    from src.confrec import train_lora_yesno as tl
    panel = tl.check_training_panel(rows, a.variant, a.hist_len, a.train)   # before any model / tokenizer load
    hist_len_used = panel["hist_len_used"]
    zinfo = load_train_z(a.train, a.qhat, a.manifest)
    sft = sft_comparator(a, hist_len_used)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ds = prior_offset_set(rows, zinfo["z_of"], tok, a.hist_len, a.max_len, a.variant)
    n_built = len(ds)
    if a.max_examples:
        ds.truncate(a.max_examples)
    print(f"examples: {n_built} built, {ds.n_skipped} skipped (prompt + answer > max_len={a.max_len}), {len(ds)} used; "
          f"longest prompt {ds.max_prompt_len} tokens; variant {a.variant}, hist_len {hist_len_used}; prior offset on "
          f"token {ds.yes_id}, TRAIN q-hat mean {zinfo['standardisation']['mean']:.4f} SD "
          f"{zinfo['standardisation']['sd']:.4f}", flush=True)
    probs = example_count_problems(sft["config"], len(ds), ds.n_skipped)
    if probs:
        raise SystemExit(f"A3 section 7: the prior-offset run trains on its SFT comparator's examples; {a.sft_adapter}: "
                         + "; ".join(probs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_train_report(out, a, panel, ds, n_built, zinfo)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             task_type="CAUSAL_LM"))
    b = attach_prior_offset(model)
    args = TrainingArguments(**tl.training_arguments(a))
    trainer = prior_offset_trainer(Trainer, ds.yes_id, b, a.b_lr)(
        model=model, args=args, train_dataset=ds, data_collator=lambda batch: collate_offset(batch, tok.pad_token_id))
    trainer.create_optimizer()        # the Trainer's own optimizer, which train() reuses: b must be in it before any step
    if not b_in_optimizer(trainer.optimizer, b):
        raise RuntimeError(f"{OFFSET_PARAM} is not in the Trainer's optimizer: b would never be learned")
    trainer.train()
    if trainer.optimizer is not None and not b_in_optimizer(trainer.optimizer, b):
        raise RuntimeError(f"{OFFSET_PARAM} left the Trainer's optimizer during training")
    b_final = float(b.detach().float().cpu())
    print(f"prior offset b = {b_final:.6g}", flush=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    leaked = [k for k in saved_adapter_keys(out) if OFFSET_PARAM in k]
    if leaked:
        raise RuntimeError(f"the saved adapter carries {leaked}: pyes_scorer --lora would not load it as an SFT adapter")
    return write_offset_and_config(out, a, panel, ds, zinfo, sft, b_final)


# ---------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qhat", help="stage 1 (CPU): q-hat of TRAIN and EVAL and the TRAIN standardisation manifest")
    q.add_argument("--domain", required=True, help="ml1m, toys, games or sports")
    q.add_argument("--panels", required=True, help="the ftgrid panels/<d>/ (train.jsonl, eval.jsonl, ftgrid_split.json)")
    q.add_argument("--raw", required=True, help="raw data root (data/raw; forensics.load_raw_events)")
    q.add_argument("--out_dir", required=True, help="outputs/confrec/ftmethod/<d>/")
    t = sub.add_parser("train", help="stage 3 (GPU): one prior-offset LoRA adapter")
    t.add_argument("--train", required=True, help="the ftgrid train.jsonl (the SFT comparator's TRAIN file)")
    t.add_argument("--model", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--mode", choices=["standard"], default="standard", help="A3 section 7: standard mode only")
    t.add_argument("--variant", choices=list(GATE_VARIANTS), default="V0",
                   help="the selected prompt (selection.json gate_ft_prompt); pyes_scorer --lora must use the same")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--hist_len", type=int, default=None, help="default: the variant's registered history window")
    t.add_argument("--max_len", type=int, default=1024)
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--bsz", type=int, default=8)
    t.add_argument("--grad_accum", type=int, default=4)
    t.add_argument("--lora_r", type=int, default=16)
    t.add_argument("--max_examples", type=int, default=None)
    t.add_argument("--qhat", required=True, help="train_qhat.csv.gz of stage 1")
    t.add_argument("--manifest", required=True, help="qhat_manifest.json of stage 1 (the recorded constants)")
    t.add_argument("--sft_adapter", required=True,
                   help="the SFT comparator of the same seed (section 2 adapter; ML-1M: Gate-FT's): same recipe")
    t.add_argument("--b_lr", type=float, default=None,
                   help="learning rate of b in a group of its own (default: none, b shares the LoRA weights' group)")
    a = ap.parse_args(argv)
    if a.cmd == "train" and a.b_lr is not None and not a.b_lr > 0:
        ap.error("--b_lr must be > 0")
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.cmd == "qhat":
        return build_qhat(a)
    return train(a)


if __name__ == "__main__":
    main()
