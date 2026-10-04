"""S6, pruning by uncertainty (FT-P): idea-stage/PREREG_AMENDMENT_3.md section 6, with its sections 0-3, 10, 11 and
idea-stage/PREREG_AMENDMENT_3_ADDENDUM_1.md. ML-1M only, on the Gate-FT backbone (Qwen3-8B) and the selected prompt
variant (selection.json gate_ft_prompt). Runner: scripts/sigir/run_ftprune.sh (stages A-E).

    # stage B (CPU): the signals, the registered subsets, the pruned TRAIN files and the manifest FREEZE `prune` records
    python -m src.confrec.ftprune signals --train outputs/confrec/gateft/train.jsonl \
        --zs_dir outputs/confrec/ftprune/zs_train --raw data/raw --variant V1 \
        --split_report outputs/confrec/gateft/gateft_split.json --out_dir outputs/confrec/ftprune
    # every file the manifest lists still has its recorded sha1 (the runner calls it before any training)
    python -m src.confrec.ftprune verify --out_dir outputs/confrec/ftprune
    # stage E: the registered analysis (alias `prn` of the paper)
    python -m src.confrec.ftprune analyze --confirm_panel outputs/confrec/gatefix/panels/ml1m_confirm_h20.jsonl \
        --split_report outputs/confrec/gateft/gateft_split.json --scores_root outputs/confrec/ftprune/scores \
        --gateft_scores outputs/confrec/gateft/scores --adapters_root outputs/confrec/ftprune/adapters \
        --gateft_adapters outputs/confrec/gateft/adapters --variant V1 \
        --manifest outputs/confrec/ftprune/prune_manifest.json --out outputs/confrec/ftprune/pruning_ml1m.json \
        [--arms P0,P1,P2,P3] [--n_boot 2000] [--seed 0]

TRAIN = Gate-FT's TRAIN set, outputs/confrec/gateft/train.jsonl (the DEV users' candidates before T, the G9 set; with
--split_report its sha1, its example count and every timestamp < T are checked against gateft_split.json). One example =
one candidate of a train.jsonl row, key "<user_id>::<item_id>" (unique: one row per user, distinct items in a row).

signals (A3 section 6, computed on the TRAIN examples before any S6 training):
  L_ZS   the like logit of the zero-shot pass on train.jsonl (pyes_scorer under the selected variant, no adapter: stage A),
         joined on (source_event_id, cand_idx) and checked against user, item and label; the pass must meet E1
         (A3 section 2) and score every example;
  beta   the TRAIN label mean; tau = the (1 - beta)-quantile (numpy linear, the quantile convention of A3 section 1) of
         the finite L_ZS over the TRAIN examples; u = -|L_ZS - tau| (largest u = nearest the boundary = most uncertain);
  q_hat  forensics.prior_means(...)["mean_prior_shrunk"] (A3 section 3; k = 5): the mean of OTHER users' first ratings of
         the item strictly before the candidate's timestamp, shrunk to the leave-user-out global mean before it, from the
         raw ML-1M events (forensics.load_raw_events / scan_events, as ftgrid_report computes q-hat). Every TRAIN timestamp
         is < T, so no rating at or after T enters. Each example must be its user's own first raw rating of the item at
         the candidate's timestamp (the raw data is the panel's source);
  C      (2y - 1) (q_hat - mean q_hat), the mean over the TRAIN examples with a finite q_hat.
Arms: in every pruned arm exactly n_remove(n_c) = floor(n_c / 4 + 1/2) examples of each label class c are removed (25%;
round half up when n_c is not a multiple of 4). P0 removes nothing; P1 seed s removes per class the examples with the
smallest sha1("P1:<s>:<key>") (a uniformly random draw that depends on the seed); P2 the largest u; P3 the largest C;
ties of P2 / P3 (equal values) go by the seed-0 random key sha1("tie:0:<key>") (smallest first). An example without a
finite signal sorts after every example with one. P2 and P3 do not depend on the seed.
Outputs under --out_dir (written only when their bytes change, so a rerun on the same inputs touches nothing):
  signals/ml1m_signals.csv.gz   one row per TRAIN example in train.jsonl order: key, user_id, item_id, source_event_id,
                                cand_idx, timestamp, label, L_ZS, q_hat, u, C, tie_key (gzip without name or mtime)
  subsets/<arm>_s<seed>.txt     the KEPT keys of P0-P3, seeds 0-4, in train.jsonl order, '\\n'-joined (no final newline)
  train/<arm>_s<seed>.jsonl     the pruned TRAIN set of P1-P3 s0-s4 (train.jsonl's rows in order, every candidate_* list
                                filtered together by split_panel.filter_candidates, history untouched, rows left without a
                                candidate dropped) and of P0 s3, s4 (a byte copy of train.jsonl); P0 s0-s2 are the Gate-FT
                                adapters, never retrained
  prune_manifest.json           deterministic (no wall-clock, host or path): the sha1 of the signals file, of every subset
                                index file and of every pruned train file, the counts kept / removed per class, beta, tau,
                                the inputs' sha1 and the code sha1 of this file. FREEZE `prune` records its sha1.

analyze (A3 section 6; TEST rows and the per-user AUC exactly as Gate-FT's G9 statistic: gateft_eval.run_stats on the
CONFIRM panel with T of gateft_split.json): per arm and seed the UAUC on TEST rows; per contrast A - B (P2 - P1, the one
confirmatory test, alpha = 0.05, no Holm; P3 - P1, P1 - P0, P2 - P0 descriptive) the seed-averaged UAUC contrast (seeds
paired by number), its 95% user-bootstrap CI and p (2,000 resamples, seed 0, A3 section 3's p with +1/(B + 1) smoothing:
ftgrid_report.mean_draws / boot_p), the 5 paired seed differences and their SD (ddof 1). The P2 - P1 label (codes):
BEATS_RANDOM iff CI lower bound > 0, mean > 2 x SD and all 5 differences > 0; WORSE_THAN_RANDOM the mirrored condition;
ABOUT_EQUAL iff the 95% CI lies within [-0.01, 0.01]; else INCONCLUSIVE; INCOMPLETE when a P1 or P2 seed is missing or
excluded (never replaced, A3 section 2); DESCRIPTIVE_MIN_N below 150 users (A3 section 1: no CI-based claim). Outputs
--out (strict JSON) and the same stem with .csv. Exit 0 when the confirmatory label is not INCOMPLETE, else 2.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np

from src.confrec import forensics as fx
from src.confrec import ftgrid_report as fr
from src.confrec import gateft_eval as ge
from src.confrec.split_panel import filter_candidates
from src.confrec.stats import percentile_ci, strict_json

NAN = float("nan")
SPEC = "idea-stage/PREREG_AMENDMENT_3.md section 6 (with sections 0-3, 10, 11; PREREG_AMENDMENT_3_ADDENDUM_1.md)"
DOMAIN = "ml1m"
ARMS = ("P0", "P1", "P2", "P3")
SEEDS = (0, 1, 2, 3, 4)
GATEFT_SEEDS = (0, 1, 2)                 # P0 s0-s2 = the Gate-FT adapters (never retrained, never rescored)
TRAINED = {"P0": (3, 4), "P1": SEEDS, "P2": SEEDS, "P3": SEEDS}   # adapters run_ftprune.sh trains
PRUNE_FRAC = 0.25                        # A3 section 6: exactly 25% of each label class
SHRINK_K = 5.0                           # A3 section 3: q-hat shrinkage (forensics.prior_means)
TIE_SEED = 0                             # A3 section 6: ties by a seed-0 random key
N_BOOT, SEED = 2000, 0                   # A3 section 3
EQUIV = 0.01                             # A3 section 6: "about equal" iff the 95% CI lies within +-0.01
MIN_N = 150                              # A3 section 1: fewer users = descriptive
SEP = "::"
CONFIRMATORY = ("P2", "P1")
CONTRASTS = (("P2", "P1", "confirmatory"), ("P3", "P1", "descriptive"), ("P1", "P0", "descriptive"),
             ("P2", "P0", "descriptive"), ("P3", "P0", "descriptive"))
SIGNAL_COLS = ("key", "user_id", "item_id", "source_event_id", "cand_idx", "timestamp", "label", "L_ZS", "q_hat", "u",
               "C", "tie_key")
SIGNALS_FILE = "signals/ml1m_signals.csv.gz"
MANIFEST = "prune_manifest.json"
MANIFEST_FORMAT = "ftprune_manifest_v1"
RESULT_FORMAT = "ftprune_result_v1"
LABELS = ("BEATS_RANDOM", "WORSE_THAN_RANDOM", "ABOUT_EQUAL", "INCONCLUSIVE", "INCOMPLETE", "DESCRIPTIVE_MIN_N")
WORDING = {
    "BEATS_RANDOM": "pruning the zero-shot model's uncertain examples helped (this signal, ML-1M, Qwen3-8B; never "
                    "'uncertainty improves ranking', Amendment 2 P; single-backbone: body only without a Llama "
                    "replication)",
    "WORSE_THAN_RANDOM": "pruning the zero-shot model's uncertain examples was worse than pruning at random (this signal, "
                         "ML-1M, Qwen3-8B)",
    "ABOUT_EQUAL": "pruning the zero-shot model's uncertain examples was about equal to pruning at random (95% CI within "
                   "+-0.01 UAUC; this signal, ML-1M, Qwen3-8B)",
    "INCONCLUSIVE": "inconclusive: pruning the zero-shot model's uncertain examples neither beat, nor lost to, nor equalled "
                    "random pruning by the registered rule",
    "INCOMPLETE": "incomplete: a registered seed of P1 or P2 is missing or excluded (never replaced); no claim",
    "DESCRIPTIVE_MIN_N": "descriptive only: fewer than 150 users (A3 section 1); no CI-based claim",
}
SIGNAL_OPERATIONALIZATIONS = (
    "example = one candidate of a train.jsonl row; key = user_id + '::' + item_id (unique; neither id contains '::')",
    "L_ZS = the like logit of the zero-shot pass on train.jsonl (pyes_scorer, the selected variant, no adapter), joined on "
    "(source_event_id, cand_idx) and checked against user, item and label; the pass must meet E1 (censored = 2 share <= "
    "0.005, no overlength prompt, mean Yes+No mass >= 0.95; ftgrid_report.integrity) and score every example; a "
    "non-finite logit (censored 2 / 3) leaves L_ZS and u missing",
    "beta = the label mean over every TRAIN example; tau = numpy.quantile(finite L_ZS, 1 - beta) (linear interpolation, "
    "the convention of A3 section 1's T); u = -|L_ZS - tau|",
    "q_hat = forensics.prior_means(scan_events(raw ML-1M events, TRAIN items, TRAIN (user, item) pairs), users, items, "
    "candidate timestamps, k = 5)['mean_prior_shrunk']: other users' first ratings of the item strictly before the "
    "timestamp, shrunk to the leave-user-out global mean before it; C = (2y - 1)(q_hat - mean q_hat), the mean over the "
    "examples with a finite q_hat (a per-class constant shift: it does not change which examples P3 removes)",
    "n_remove(class) = floor(n_class / 4 + 1/2) (round half up; exactly 25% when n_class is a multiple of 4)",
    "P1 seed s removes per class the n_remove examples with the smallest sha1('P1:<s>:<key>'); P2 / P3 remove per class "
    "the n_remove examples with the largest u / C, ties (equal values) by the smallest sha1('tie:0:<key>') (the seed-0 "
    "random key); an example without a finite signal sorts after every example with one; P2 and P3 do not depend on the "
    "seed (their five subset files are identical)",
    "subset files list the KEPT keys in train.jsonl order, one per line (no final newline); pruned train files keep "
    "train.jsonl's rows in order, filter every candidate_* list together (split_panel.filter_candidates; the history is "
    "untouched) and drop the rows left without a candidate; json.dumps(row, ensure_ascii=False) per line, as gateft_data "
    "writes train.jsonl; P0 s3 and s4 train on a byte copy of train.jsonl",
)
ANALYSIS_OPERATIONALIZATIONS = (
    "TEST rows = the CONFIRM panel's candidates with ts >= T (T of gateft_split.json); per run the per-user AUC (ties "
    "1/2) over the run's finite TEST rows of every user with both classes there, i.e. gateft_eval.run_stats' auc_post, "
    "the G9 statistic; the per-seed UAUC of an arm is its mean over those users",
    "contrast A - B: seeds paired by number (A s with B s), using the seeds whose two runs are OK; users = those with an "
    "AUC in every run of the contrast; d_s = mean over those users of AUC(A s) - AUC(B s); the seed-averaged UAUC "
    "contrast = mean over users of [mean over seeds of AUC(A)] - [mean over seeds of AUC(B)] (= the mean of the d_s)",
    "CI = 95% percentile interval of 2,000 user resamples (seed 0; stats.paired_bootstrap's draws over the sorted user "
    "ids, ftgrid_report.mean_draws); p = min(1, 2 min((#{d* <= 0} + 1)/(B + 1), (#{d* >= 0} + 1)/(B + 1))) "
    "(ftgrid_report.boot_p, A3 section 3)",
    "sd_seed = SD (ddof 1) of the paired seed differences d_s",
    "P2 - P1 (the only confirmatory test, alpha 0.05, no Holm): BEATS_RANDOM iff lo > 0, mean > 2 sd_seed and all 5 "
    "d_s > 0; WORSE_THAN_RANDOM iff hi < 0, mean < -2 sd_seed and all 5 d_s < 0; ABOUT_EQUAL iff -0.01 <= lo and hi <= "
    "0.01; else INCONCLUSIVE; checked in that order (BEATS and ABOUT_EQUAL can hold together, a CI inside (0, 0.01]: "
    "both flags are reported); INCOMPLETE when a P1 or P2 seed is missing or excluded or the runs differ in variant, "
    "window or backbone; DESCRIPTIVE_MIN_N below 150 users (A3 section 1). p < 0.05 is reported, the label follows the "
    "section-6 rule as written",
    "a run is excluded and listed, never replaced: ABSENT (no directory), INCOMPLETE (no report.json or scores.csv.gz), "
    "FAILED_INTEGRITY (the runner's marker after a second E1 failure, or E1 failing per gateft_eval.run_stats: "
    "censored = 2 share > 0.005, an overlength prompt, mean Yes+No mass < 0.95), PANEL_MISMATCH (data_sha1 is not the "
    "CONFIRM panel's, or a score row outside it), VARIANT_MISMATCH, ADAPTER_MISMATCH (report.json lora is not the "
    "expected adapter directory, or that adapter's train_config.json names another seed, variant or train file), "
    "UNREADABLE",
    "P0 s0-s2 = the Gate-FT adapters and their Gate-FT like passes on the CONFIRM panel (never retrained or rescored); "
    "P0 s3, s4 and P1-P3 s0-s4 are run_ftprune.sh's adapters (train/<arm>_s<seed>.jsonl, the Gate-FT recipe)",
    "P3 - P1, P1 - P0 and P2 - P0 are descriptive (the same machinery, no label); an arm cut at the 2026-10-22 "
    "checkpoint (RUN_P3=0) is NOT_RUN and its contrasts are not run",
)


# ---------------------------------------------------------------- small helpers
def sha1_file(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def hkey(*parts) -> str:
    """sha1 hex of the parts joined by ':' (a deterministic pseudo-random key; hex strings of equal length sort as
    their numbers)."""
    return hashlib.sha1(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def tie_key(key: str) -> str:
    return hkey("tie", TIE_SEED, key)


def p1_key(seed: int, key: str) -> str:
    return hkey("P1", seed, key)


def n_remove(n: int) -> int:
    """Examples removed from a label class of n examples: 25%, round half up."""
    return int(math.floor(n * PRUNE_FRAC + 0.5))


def _fmt(x) -> str:
    x = float(x)
    return repr(x) if math.isfinite(x) else "nan"


def gz_bytes(text: str) -> bytes:
    """gzip of the text without file name and with mtime 0, so equal text gives equal bytes."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as f:
        f.write(text.encode("utf-8"))
    return buf.getvalue()


def put(path, data: bytes) -> bool:
    """Write data atomically unless the file already holds exactly these bytes (then the file and its mtime stay).
    True if the file was written."""
    path = Path(path)
    if path.is_file() and path.stat().st_size == len(data) and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


def json_bytes(obj) -> bytes:
    return (json.dumps(strict_json(obj), indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_rows(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_tag(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


def subset_rel(arm: str, seed: int) -> str:
    return f"subsets/{run_tag(arm, seed)}.txt"


def train_rel(arm: str, seed: int) -> str:
    return f"train/{run_tag(arm, seed)}.jsonl"


def has_train_file(arm: str, seed: int) -> bool:
    return seed in TRAINED[arm]


# ---------------------------------------------------------------- TRAIN examples and signals
def train_examples(rows: list) -> dict:
    """One entry per (row, candidate) in train.jsonl order: key, user, item, ev (source_event_id), cand (index in the
    row), ts, y; `starts` holds each row's first example index (and n at the end)."""
    ex = {k: [] for k in ("key", "user", "item", "ev", "cand", "ts", "y")}
    starts = []
    for r in rows:
        u, ev = str(r["user_id"]), str(r.get("source_event_id", r["user_id"]))
        items = [str(i) for i in r["candidate_item_ids"]]
        n = len(items)
        for name in ("candidate_labels", "candidate_timestamps"):
            if len(r.get(name) or []) != n:
                raise SystemExit(f"{ev}: {name} has {len(r.get(name) or [])} entries for {n} candidates")
        starts.append(len(ex["key"]))
        for c, (i, t, y) in enumerate(zip(items, r["candidate_timestamps"], r["candidate_labels"])):
            if SEP in u or SEP in i:
                raise SystemExit(f"{ev}: user_id {u!r} or item_id {i!r} contains '{SEP}' (the example key separator)")
            if int(y) not in (0, 1):
                raise SystemExit(f"{ev}: candidate label {y!r} is not 0/1")
            ex["key"].append(u + SEP + i)
            ex["user"].append(u)
            ex["item"].append(i)
            ex["ev"].append(ev)
            ex["cand"].append(c)
            ex["ts"].append(float(t))
            ex["y"].append(int(y))
    if len(set(ex["key"])) != len(ex["key"]):
        raise SystemExit("train.jsonl holds a (user_id, item_id) pair twice: the example key is not unique")
    starts.append(len(ex["key"]))
    out = {k: v for k, v in ex.items()}
    out["cand"] = np.array(ex["cand"], int)
    out["ts"] = np.array(ex["ts"], float)
    out["y"] = np.array(ex["y"], int)
    out["starts"] = starts
    out["n"] = len(ex["key"])
    return out


def zs_logits(zs_dir, ex: dict, variant: str, train_sha1: str) -> tuple:
    """L_ZS aligned to the TRAIN examples and the zero-shot run's record; refuses a pass that is not the zero-shot like
    pass of this train.jsonl under `variant`, that fails E1 or that does not score every example."""
    d = Path(zs_dir)
    for f in ("report.json", "scores.csv.gz"):
        if not (d / f).is_file():
            raise SystemExit(f"{d / f} does not exist: the zero-shot TRAIN pass (stage A) is not finished")
    if (d / "FAILED_INTEGRITY").exists():
        raise SystemExit(f"{d} is marked FAILED_INTEGRITY (E1 failed twice, A3 section 2): S6 has no U signal")
    rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
    sc = fx.load_scores(d / "scores.csv.gz")
    integ = fr.integrity(rep, sc["censoring"])
    problems = []
    if rep.get("variant") != variant:
        problems.append(f"variant {rep.get('variant')!r} is not the selected {variant!r}")
    if rep.get("lora"):
        problems.append("the pass scored an adapter (the U signal is zero-shot)")
    if rep.get("data_sha1") != train_sha1:
        problems.append(f"data_sha1 {rep.get('data_sha1')} is not train.jsonl's {train_sha1}")
    if "like" not in (rep.get("questions") or []):
        problems.append(f"questions {rep.get('questions')} lack 'like'")
    if rep.get("readout", "yesno") != "yesno":
        problems.append(f"readout {rep.get('readout')!r} is not yesno")
    if not integ["E1"]:
        problems.append(f"E1 fails ({fr._e1_text(integ)})")
    if problems:
        raise SystemExit(f"{d}: not usable as the zero-shot TRAIN pass of S6: " + "; ".join(problems))
    like = sc["L"].get("like", {})
    L = np.full(ex["n"], NAN)
    missing = mismatched = 0
    seen = set()
    for j in range(ex["n"]):
        k = (ex["ev"][j], int(ex["cand"][j]))
        m = sc["meta"].get(k)
        if m is None:
            missing += 1
            continue
        seen.add(k)
        if m[0] != ex["user"][j] or m[1] != ex["item"][j] or int(m[2]) != int(ex["y"][j]):
            mismatched += 1
            continue
        L[j] = like.get(k, NAN)
    extra = len(set(sc["meta"]) - seen)
    if missing or mismatched or extra:
        raise SystemExit(f"{d}: the scores do not match train.jsonl ({missing} examples without a score row, "
                         f"{mismatched} with another user / item / label, {extra} score rows outside it)")
    rec = {"variant": rep.get("variant"), "readout": rep.get("readout", "yesno"), "questions": rep.get("questions"),
           "hist_len": rep.get("hist_len"), "lora": None, "data_sha1": rep.get("data_sha1"),
           "backbone": rep.get("backbone") or (Path(str(rep["model"])).name if rep.get("model") else None),
           "scorer": rep.get("scorer"), "prompts_sha1": (rep.get("config") or {}).get("prompts_sha1"),
           "scores_sha1": sha1_file(d / "scores.csv.gz"), "integrity": integ,
           "n_examples_scored": int(ex["n"]), "n_nonfinite_like": int((~np.isfinite(L)).sum())}
    return L, rec


def raw_ratings_path(raw) -> Path:
    raw = Path(raw)
    return raw / "ratings.dat" if (raw / "ratings.dat").exists() else raw / "ml-1m" / "ratings.dat"


def q_hat_of(ex: dict, raw) -> tuple:
    """q-hat per TRAIN example (forensics.prior_means' shrunk prior-only leave-user-out item mean, k = 5) and its record.
    Refuses raw data that is not the panel's source (a TRAIN pair missing from it, or its own first rating at another
    timestamp)."""
    rp = raw_ratings_path(raw)
    if not rp.is_file():
        raise SystemExit(f"{rp} does not exist (the raw ML-1M ratings: q-hat needs them)")
    events = fx.load_raw_events(raw, DOMAIN)
    users, items, times = ex["user"], ex["item"], ex["ts"]
    scan = fx.scan_events(events, set(items), set(zip(users, items)))
    pm = fx.prior_means(scan, users, items, times, SHRINK_K)
    own_bad = sum(1 for u, i, t in zip(users, items, times.tolist()) if scan["own"].get((u, i), (NAN,))[0] != t)
    if scan["n_candidate_pairs_not_in_raw"] or own_bad:
        raise SystemExit(f"{rp}: {scan['n_candidate_pairs_not_in_raw']} TRAIN pairs are not in the raw data and "
                         f"{own_bad} are not their user's first rating at the candidate's timestamp: not the panel's "
                         "source")
    q = np.asarray(pm["mean_prior_shrunk"], float)
    n_prior = np.asarray(pm["n_prior"], float)
    rec = {"estimator": "forensics.prior_means mean_prior_shrunk", "shrink_k": SHRINK_K,
           "ratings_sha1": sha1_file(rp), "ratings_file": rp.name, "n_raw_events": scan["n_raw_events"],
           "n_first_events": scan["n_first_events"], "n_candidate_pairs_not_in_raw": scan["n_candidate_pairs_not_in_raw"],
           "n_nonfinite_q_hat": int((~np.isfinite(q)).sum()),
           "share_without_prior_rating": float(np.mean(n_prior == 0)) if len(n_prior) else NAN,
           "n_prior_median": float(np.median(n_prior)) if len(n_prior) else NAN}
    return q, rec


def u_signal(L, y) -> tuple:
    """(u, beta, tau): beta = label mean, tau = the (1 - beta)-quantile (numpy linear) of the finite logits,
    u = -|L - tau| (NaN where L is not finite)."""
    L, y = np.asarray(L, float), np.asarray(y, int)
    fin = np.isfinite(L)
    if not fin.any():
        raise SystemExit("no finite zero-shot logit among the TRAIN examples")
    beta = float(y.mean())
    tau = float(np.quantile(L[fin], 1.0 - beta))
    u = np.full(len(L), NAN)
    u[fin] = -np.abs(L[fin] - tau)
    return u, beta, tau


def c_signal(q, y) -> tuple:
    """(C, mean q_hat): C = (2y - 1)(q_hat - mean q_hat), the mean over the finite q_hat (NaN where q_hat is not)."""
    q, y = np.asarray(q, float), np.asarray(y, int)
    fin = np.isfinite(q)
    if not fin.any():
        raise SystemExit("no finite q-hat among the TRAIN examples")
    mq = float(q[fin].mean())
    C = np.full(len(q), NAN)
    C[fin] = (2 * y[fin] - 1) * (q[fin] - mq)
    return C, mq


def removal_mask(score, y, order_key: list) -> np.ndarray:
    """Removed examples: per label class the n_remove(n_c) first in the order (finite score descending; an example
    without a finite score after every one with; then order_key ascending)."""
    score, y = np.asarray(score, float), np.asarray(y, int)
    out = np.zeros(len(y), bool)
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        k = n_remove(len(idx))
        order = sorted(idx.tolist(), key=lambda j: (0, -score[j], order_key[j]) if math.isfinite(score[j])
                       else (1, 0.0, order_key[j]))
        out[order[:k]] = True
    return out


def subsets(sig: dict) -> dict:
    """{(arm, seed): removed mask} of the four arms and five seeds (A3 section 6)."""
    n, y, keys = len(sig["y"]), sig["y"], sig["key"]
    tie = sig["tie_key"]
    zero = np.zeros(n)
    p2 = removal_mask(sig["u"], y, tie)
    p3 = removal_mask(sig["C"], y, tie)
    out = {}
    for s in SEEDS:
        out[("P0", s)] = np.zeros(n, bool)
        out[("P1", s)] = removal_mask(zero, y, [p1_key(s, k) for k in keys])
        out[("P2", s)] = p2.copy()
        out[("P3", s)] = p3.copy()
    return out


def signals_text(ex: dict, L, q, u, C, tie: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(SIGNAL_COLS)
    for j in range(ex["n"]):
        w.writerow([ex["key"][j], ex["user"][j], ex["item"][j], ex["ev"][j], int(ex["cand"][j]), int(ex["ts"][j]),
                    int(ex["y"][j]), _fmt(L[j]), _fmt(q[j]), _fmt(u[j]), _fmt(C[j]), tie[j]])
    return buf.getvalue()


def read_signals(path) -> dict:
    """signals csv.gz -> {column: list} (numbers as float, key / ids / tie_key as str)."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {c: [r[c] for r in rows] for c in SIGNAL_COLS}
    for c in ("cand_idx", "timestamp", "label"):
        out[c] = [int(x) for x in out[c]]
    for c in ("L_ZS", "q_hat", "u", "C"):
        out[c] = [float(x) for x in out[c]]
    return out


def pruned_text(rows: list, ex: dict, keep) -> tuple:
    """(jsonl text, rows written, examples, positives) of train.jsonl restricted to the kept examples."""
    lines, n_rows, n_ex, n_pos = [], 0, 0, 0
    for j, r in enumerate(rows):
        a, b = ex["starts"][j], ex["starts"][j + 1]
        k = [bool(x) for x in keep[a:b]]
        if not any(k):
            continue
        sub = filter_candidates(r, k)
        lines.append(json.dumps(sub, ensure_ascii=False) + "\n")
        n_rows += 1
        n_ex += sum(k)
        n_pos += int(sum(int(v) for v in sub["candidate_labels"]))
    return "".join(lines), n_rows, n_ex, n_pos


def _class_counts(y, mask) -> dict:
    y, mask = np.asarray(y, int), np.asarray(mask, bool)
    return {str(c): int(((y == c) & mask).sum()) for c in (1, 0)}


def _describe(removed, y, L, u, q, C) -> dict:
    """Means of the signals over the removed and the kept examples, per class (descriptive)."""
    out = {}
    for c in (1, 0):
        blk = {}
        for part, m in (("removed", removed & (y == c)), ("kept", ~removed & (y == c))):
            blk[part] = {v: (float(np.nanmean(x[m])) if m.any() and np.isfinite(x[m]).any() else NAN)
                         for v, x in (("L_ZS", L), ("u", u), ("q_hat", q), ("C", C))}
        out[str(c)] = blk
    return out


def build_signals(train, zs_dir, raw, variant: str, out_dir, split_report=None) -> dict:
    """Stage B: the signals, subsets, pruned train files and the manifest (returned). Files are written only when their
    bytes change."""
    train, out = Path(train), Path(out_dir)
    train_bytes = train.read_bytes()
    train_sha1 = sha1_bytes(train_bytes)
    rows = read_rows(train)
    if not rows:
        raise SystemExit(f"{train}: no rows")
    ex = train_examples(rows)
    split_rec = None
    if split_report:
        sp = json.loads(Path(split_report).read_text(encoding="utf-8"))
        st = sp.get("train") or {}
        bad = []
        if st.get("sha1") is not None and st["sha1"] != train_sha1:
            bad.append(f"train.jsonl sha1 {train_sha1} is not the split report's {st['sha1']}")
        if st.get("candidates") is not None and int(st["candidates"]) != ex["n"]:
            bad.append(f"{ex['n']} examples, the split report records {st['candidates']}")
        T = float(sp["T"])
        if not np.all(ex["ts"] < T):
            bad.append(f"{int((ex['ts'] >= T).sum())} TRAIN examples at or after T = {T}")
        if bad:
            raise SystemExit(f"{train} is not Gate-FT's TRAIN set: " + "; ".join(bad))
        split_rec = {"sha1": sha1_file(split_report), "T": T, "train_rows": st.get("rows"),
                     "train_candidates": st.get("candidates"), "train_positive_rate": st.get("positive_rate")}
    L, zs_rec = zs_logits(zs_dir, ex, variant, train_sha1)
    q, q_rec = q_hat_of(ex, raw)
    y = ex["y"]
    u, beta, tau = u_signal(L, y)
    C, mq = c_signal(q, y)
    tie = [tie_key(k) for k in ex["key"]]
    sig = {"key": ex["key"], "y": y, "u": u, "C": C, "tie_key": tie}
    removed = subsets(sig)

    files = {}
    text = signals_text(ex, L, q, u, C, tie)
    data = gz_bytes(text)
    put(out / SIGNALS_FILE, data)
    files[SIGNALS_FILE] = sha1_bytes(data)
    sub_rec, train_rec = {}, {}
    for (arm, s), rm in removed.items():
        keep = ~rm
        data = "\n".join(k for k, kp in zip(ex["key"], keep) if kp).encode("utf-8")
        rel = subset_rel(arm, s)
        put(out / rel, data)
        files[rel] = sha1_bytes(data)
        sub_rec[run_tag(arm, s)] = {"file": rel, "sha1": files[rel], "n_keep": _class_counts(y, keep),
                                    "n_removed": _class_counts(y, rm)}
        if not has_train_file(arm, s):
            continue
        rel = train_rel(arm, s)
        if arm == "P0":
            data, n_rows, n_ex, n_pos = train_bytes, len(rows), ex["n"], int(y.sum())
        else:
            txt, n_rows, n_ex, n_pos = pruned_text(rows, ex, keep)
            data = txt.encode("utf-8")
        put(out / rel, data)
        files[rel] = sha1_bytes(data)
        train_rec[run_tag(arm, s)] = {"file": rel, "sha1": files[rel], "rows": n_rows, "examples": n_ex,
                                      "positives": n_pos, "negatives": n_ex - n_pos}
    reser, _, _, _ = pruned_text(rows, ex, np.ones(ex["n"], bool))
    classes = {str(c): {"n": int((y == c).sum()), "n_remove": n_remove(int((y == c).sum())),
                        "n_keep": int((y == c).sum()) - n_remove(int((y == c).sum()))} for c in (1, 0)}
    p2, p3, p1 = removed[("P2", 0)], removed[("P3", 0)], removed[("P1", 0)]

    def overlap(a, b):
        return {str(c): (float((a & b & (y == c)).sum() / max(1, (a & (y == c)).sum()))) for c in (1, 0)}
    man = {
        "format": MANIFEST_FORMAT, "alias": "prn", "spec": SPEC, "domain": DOMAIN, "variant": variant,
        "code_sha1": {"ftprune.py": sha1_file(__file__)},
        "inputs": {"train_jsonl": {"sha1": train_sha1, "rows": len(rows), "examples": ex["n"],
                                   "positives": int(y.sum()), "negatives": int(ex["n"] - y.sum())},
                   "gateft_split": split_rec, "zero_shot_like_pass": zs_rec, "raw": q_rec},
        "registered": {"prune_fraction": PRUNE_FRAC, "seeds": list(SEEDS), "gateft_seeds_of_P0": list(GATEFT_SEEDS),
                       "tie_seed": TIE_SEED, "shrink_k": SHRINK_K,
                       "arms": {"P0": "removes nothing (s0-s2 = the Gate-FT adapters; s3, s4 trained)",
                                "P1": "a uniformly random 25% per class (the draw depends on the seed)",
                                "P2": "the 25% per class with the largest u (the zero-shot model's most uncertain)",
                                "P3": "the 25% per class with the largest C (the most prior-congruent)"}},
        "signals": {"file": SIGNALS_FILE, "sha1": files[SIGNALS_FILE], "columns": list(SIGNAL_COLS),
                    "label_mean_beta": beta, "tau_quantile": 1.0 - beta, "tau": tau,
                    "quantile_method": "numpy.quantile, linear", "mean_q_hat": mq,
                    "n_nonfinite": {"L_ZS": int((~np.isfinite(L)).sum()), "q_hat": int((~np.isfinite(q)).sum())}},
        "classes": classes,
        "subsets": sub_rec,
        "train_files": train_rec,
        "checks": {"P0_train_is_a_byte_copy_of_train_jsonl": all(
                       files[train_rel("P0", s)] == train_sha1 for s in TRAINED["P0"]),
                   "pruned_writer_reproduces_train_jsonl_bytes": reser.encode("utf-8") == train_bytes,
                   "P2_P3_identical_across_seeds": all(
                       np.array_equal(removed[(a, s)], removed[(a, 0)]) for a in ("P2", "P3") for s in SEEDS),
                   "P1_differs_across_seeds": len({removed[("P1", s)].tobytes() for s in SEEDS}) == len(SEEDS)},
        "describe": {"note": "descriptive only (TRAIN-side signals, no outcome)",
                     "removed_signal_means": {"P1_s0": _describe(p1, y, L, u, q, C),
                                              "P2": _describe(p2, y, L, u, q, C),
                                              "P3": _describe(p3, y, L, u, q, C)},
                     "share_of_P2_removed_also_removed_by": {"P3": overlap(p2, p3), "P1_s0": overlap(p2, p1)}},
        "files": dict(sorted(files.items())),
        "operationalizations": list(SIGNAL_OPERATIONALIZATIONS),
    }
    put(out / MANIFEST, json_bytes(man))
    return man


# ---------------------------------------------------------------- verify
def manifest_files(man: dict) -> dict:
    return dict(man.get("files") or {})


def verify(out_dir) -> list:
    """Problems (empty = every file the manifest lists exists with its recorded sha1)."""
    out = Path(out_dir)
    mp = out / MANIFEST
    if not mp.is_file():
        return [f"{mp} does not exist (stage B)"]
    man = json.loads(mp.read_text(encoding="utf-8"))
    probs = []
    files = manifest_files(man)
    need = {SIGNALS_FILE} | {subset_rel(a, s) for a in ARMS for s in SEEDS} | \
        {train_rel(a, s) for a in ARMS for s in SEEDS if has_train_file(a, s)}
    if set(files) != need:
        probs.append(f"the manifest lists {sorted(set(files) ^ need)} unexpectedly (or lacks them)")
    for rel, sha in sorted(files.items()):
        p = out / rel
        if not p.is_file():
            probs.append(f"{rel}: missing")
        elif sha1_file(p) != sha:
            probs.append(f"{rel}: sha1 {sha1_file(p)} is not the recorded {sha}")
    return probs


# ---------------------------------------------------------------- analysis
def _norm(p) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def load_run(arm: str, seed: int, run_dir, adapter, ts_map: dict, T: float, panel_sha1: str, variant: str,
             train_name: str) -> dict:
    """One scored adapter -> {status, reason, ...} (status OK or the exclusion code; see ANALYSIS_OPERATIONALIZATIONS)."""
    d = Path(run_dir)
    out = {"run": f"{arm}/s{seed}", "source": "gateft" if arm == "P0" and seed in GATEFT_SEEDS else "ftprune"}
    if not d.is_dir():
        return {**out, "status": "ABSENT", "reason": f"no scoring directory for {arm} s{seed}"}
    if (d / "FAILED_INTEGRITY").exists():
        return {**out, "status": "FAILED_INTEGRITY", "reason": "marked FAILED_INTEGRITY by the runner (E1 failed twice; "
                                                              "A3 section 2: missing, never replaced)"}
    miss = [f for f in ("report.json", "scores.csv.gz") if not (d / f).is_file()]
    if miss:
        return {**out, "status": "INCOMPLETE", "reason": f"missing {miss}"}
    try:
        st = ge.run_stats(d, ts_map, T)
    except SystemExit as e:
        return {**out, "status": "PANEL_MISMATCH", "reason": str(e)}
    except (OSError, ValueError, KeyError, EOFError, csv.Error) as e:
        return {**out, "status": "UNREADABLE", "reason": f"{type(e).__name__}: {e}"}
    integ = st["integrity"]
    out.update(integrity=integ, variant=st["variant"], hist_len=st["hist_len"],
               backbone=Path(str(st["model"])).name if st.get("model") else None)
    lora = st.get("lora")
    cfg_check = None
    if not integ["ok"]:
        status, reason = "FAILED_INTEGRITY", (f"E1 fails: censored=2 share {integ['censored2_share']}, overlength "
                                              f"{integ['overlength']}, mean Yes+No mass {integ['mean_yes_no_mass']}")
    elif st["data_sha1"] != panel_sha1:
        status, reason = "PANEL_MISMATCH", f"data_sha1 {st['data_sha1']} is not the CONFIRM panel's {panel_sha1}"
    elif st["variant"] != variant:
        status, reason = "VARIANT_MISMATCH", f"variant {st['variant']} is not {variant}"
    elif not lora or _norm(lora) != _norm(adapter):
        status, reason = "ADAPTER_MISMATCH", f"scored adapter {lora!r} is not the expected {Path(adapter).as_posix()!r}"
    else:
        status, reason = "OK", None
        tc = Path(adapter) / "train_config.json"
        if tc.is_file():
            cfg = json.loads(tc.read_text(encoding="utf-8"))
            bad = []
            if int(cfg.get("seed", -1)) != seed:
                bad.append(f"seed {cfg.get('seed')}")
            if cfg.get("variant", "V0") != variant:
                bad.append(f"variant {cfg.get('variant')}")
            if Path(str(cfg.get("train", ""))).name != train_name:
                bad.append(f"train {Path(str(cfg.get('train', ''))).name}")
            cfg_check = not bad
            if bad:
                status, reason = "ADAPTER_MISMATCH", (f"{tc.as_posix()} names {', '.join(bad)} (expected seed {seed}, "
                                                      f"{variant}, {train_name})")
    out.update(status=status, reason=reason, train_config_checked=cfg_check)
    if status == "OK":
        auc = st["auc_post"]
        out.update(UAUC=float(np.mean(list(auc.values()))) if auc else NAN, n_users=len(auc), auc_post=auc)
    return out


def contrast(auc_a: dict, auc_b: dict, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """auc_a / auc_b: {seed: {user: AUC}} of the usable runs of arms A and B. The seed-averaged UAUC contrast A - B on
    the users present in every paired run, its user-bootstrap CI and p, the paired seed differences and their SD."""
    pairs = [s for s in SEEDS if s in auc_a and s in auc_b]
    users = sorted(set.intersection(*[set(auc_a[s]) & set(auc_b[s]) for s in pairs])) if pairs else []
    out = {"seeds_paired": pairs, "seeds_missing": [s for s in SEEDS if s not in pairs],
           "complete": len(pairs) == len(SEEDS), "n_users": len(users), "n_boot": int(n_boot), "boot_seed": int(seed)}
    if not pairs or not users:
        out.update(est=NAN, lo=NAN, hi=NAN, p=NAN, per_seed={}, sd_seed=NAN)
        return out
    A = np.array([[auc_a[s][u] for s in pairs] for u in users], float).reshape(len(users), len(pairs))
    B = np.array([[auc_b[s][u] for s in pairs] for u in users], float).reshape(len(users), len(pairs))
    V = A - B
    draws = fr.mean_draws(V, n_boot, seed).mean(1)
    lo, hi = percentile_ci(draws)
    per = V.mean(0)
    out.update(est=float(V.mean()), lo=lo, hi=hi, p=fr.boot_p(draws), n_boot_finite=int(np.isfinite(draws).sum()),
               per_seed={f"s{s}": float(x) for s, x in zip(pairs, per)},
               sd_seed=float(np.std(per, ddof=1)) if len(per) > 1 else NAN,
               UAUC_a_seed_averaged=float(A.mean()), UAUC_b_seed_averaged=float(B.mean()),
               UAUC_a_per_seed={f"s{s}": float(x) for s, x in zip(pairs, A.mean(0))},
               UAUC_b_per_seed={f"s{s}": float(x) for s, x in zip(pairs, B.mean(0))})
    return out


def conditions(est: float, lo: float, hi: float, per_seed) -> dict:
    """The section-6 clauses of the P2 - P1 rule on a contrast's numbers (False when a number is missing)."""
    d = np.asarray(list(per_seed), float)
    fin = len(d) > 1 and np.isfinite(d).all() and all(math.isfinite(x) for x in (est, lo, hi))
    sd = float(np.std(d, ddof=1)) if fin else NAN
    c = {"sd_seed": sd, "n_seeds": int(len(d)),
         "ci_lo_gt_0": bool(fin and lo > 0), "mean_gt_2sd": bool(fin and est > 2 * sd),
         "all_positive": bool(fin and (d > 0).all()),
         "ci_hi_lt_0": bool(fin and hi < 0), "mean_lt_minus_2sd": bool(fin and est < -2 * sd),
         "all_negative": bool(fin and (d < 0).all()),
         "ci_within_equivalence": bool(fin and lo >= -EQUIV and hi <= EQUIV)}
    c["beats"] = c["ci_lo_gt_0"] and c["mean_gt_2sd"] and c["all_positive"]
    c["worse"] = c["ci_hi_lt_0"] and c["mean_lt_minus_2sd"] and c["all_negative"]
    c["about_equal"] = c["ci_within_equivalence"]
    c["beats_and_about_equal_overlap"] = c["beats"] and c["about_equal"]
    return c


def claim_label(complete: bool, n_users: int, cond: dict) -> str:
    """The registered P2 - P1 label (A3 section 6; INCOMPLETE and the section-1 minimum n first)."""
    if not complete:
        return "INCOMPLETE"
    if n_users < MIN_N:
        return "DESCRIPTIVE_MIN_N"
    if cond["beats"]:
        return "BEATS_RANDOM"
    if cond["worse"]:
        return "WORSE_THAN_RANDOM"
    if cond["about_equal"]:
        return "ABOUT_EQUAL"
    return "INCONCLUSIVE"


def expected_layout(scores_root, gateft_scores, adapters_root, gateft_adapters) -> dict:
    """{(arm, seed): (scoring dir, adapter dir, train file name)} of the registered runs."""
    out = {}
    for arm in ARMS:
        for s in SEEDS:
            if arm == "P0" and s in GATEFT_SEEDS:
                out[(arm, s)] = (Path(gateft_scores) / f"s{s}", Path(gateft_adapters) / f"s{s}", "train.jsonl")
            else:
                out[(arm, s)] = (Path(scores_root) / arm / f"s{s}", Path(adapters_root) / arm / f"s{s}",
                                 Path(train_rel(arm, s)).name)
    return out


def analyze(confirm_panel, split_report, scores_root, gateft_scores, adapters_root, gateft_adapters, variant: str,
            arms=ARMS, manifest=None, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    T = float(json.loads(Path(split_report).read_text(encoding="utf-8"))["T"])
    ts_map = ge.panel_timestamps(confirm_panel)
    panel_sha1 = sha1_file(confirm_panel)
    arms = tuple(a for a in ARMS if a in set(arms))
    if not {"P0", "P1", "P2"} <= set(arms):
        raise SystemExit(f"--arms {arms}: P0, P1 and P2 are always run (A3 section 6; only P3 can be cut)")
    man_rec = None
    if manifest:
        man = json.loads(Path(manifest).read_text(encoding="utf-8"))
        man_rec = {"sha1": sha1_file(manifest), "variant": man.get("variant"),
                   "tau": (man.get("signals") or {}).get("tau"),
                   "label_mean_beta": (man.get("signals") or {}).get("label_mean_beta")}
        if man.get("variant") not in (None, variant):
            raise SystemExit(f"{manifest}: variant {man.get('variant')} is not {variant}")
    lay = expected_layout(scores_root, gateft_scores, adapters_root, gateft_adapters)
    runs = {}
    for (arm, s), (d, a, tname) in lay.items():
        if arm in arms:
            runs[(arm, s)] = load_run(arm, s, d, a, ts_map, T, panel_sha1, variant, tname)
    ok = {k: r for k, r in runs.items() if r["status"] == "OK"}
    problems = []
    settings = sorted({(str(r["variant"]), str(r["hist_len"]), str(r["backbone"])) for r in ok.values()})
    if len(settings) > 1:
        problems.append(f"the OK runs differ in (variant, hist_len, backbone): {settings}")
    arm_out = {}
    for arm in ARMS:
        if arm not in arms:
            arm_out[arm] = {"status": "NOT_RUN", "reason": f"{arm} cut (RUN_P3=0, A3 section 10 checkpoint 2026-10-22): "
                                                           "reported as not run"}
            continue
        seeds = {}
        for s in SEEDS:
            r = runs[(arm, s)]
            seeds[f"s{s}"] = {k: v for k, v in r.items() if k != "auc_post"}
        per = [seeds[f"s{s}"].get("UAUC", NAN) if runs[(arm, s)]["status"] == "OK" else NAN for s in SEEDS]
        okseeds = [s for s in SEEDS if runs[(arm, s)]["status"] == "OK"]
        users = sorted(set.intersection(*[set(runs[(arm, s)]["auc_post"]) for s in okseeds])) if okseeds else []
        avg = (float(np.mean([np.mean([runs[(arm, s)]["auc_post"][u] for s in okseeds]) for u in users]))
               if users else NAN)
        arm_out[arm] = {"status": "RUN", "complete": len(okseeds) == len(SEEDS), "seeds_ok": okseeds,
                        "UAUC_per_seed": per, "UAUC_seed_averaged": avg, "n_users_common": len(users),
                        "sd_seed": float(np.std([x for x in per if math.isfinite(x)], ddof=1))
                        if len(okseeds) > 1 else NAN, "seeds": seeds}
    con_out = {}
    for a, b, role in CONTRASTS:
        name = f"{a}-{b}"
        if a not in arms or b not in arms:
            con_out[name] = {"role": role, "status": "NOT_RUN", "reason": f"{a if a not in arms else b} not run (cut)"}
            continue
        auc_a = {s: runs[(a, s)]["auc_post"] for s in SEEDS if runs[(a, s)]["status"] == "OK"}
        auc_b = {s: runs[(b, s)]["auc_post"] for s in SEEDS if runs[(b, s)]["status"] == "OK"}
        c = contrast(auc_a, auc_b, n_boot, seed)
        excluded = {f"{x}/s{s}": runs[(x, s)]["status"] for x in (a, b) for s in SEEDS
                    if runs[(x, s)]["status"] != "OK"}
        c["complete"] = bool(c["complete"] and not problems)
        c.update(role=role, status="OK" if c["complete"] else "INCOMPLETE", excluded_runs=excluded,
                 descriptive_min_n=c["n_users"] < MIN_N)
        if c["descriptive_min_n"]:
            c["p"] = None          # A3 section 1: no p-value below 150 users (the interval stays, as a description)
        c["p_lt_0.05"] = None if c["p"] is None or not math.isfinite(c["p"]) else bool(c["p"] < 0.05)
        if role == "confirmatory":
            cond = conditions(c["est"], c["lo"], c["hi"], c["per_seed"].values())
            c["conditions"] = cond
            c["label"] = claim_label(c["complete"], c["n_users"], cond)
        else:
            c["label"] = None
        con_out[name] = c
    conf = con_out["P2-P1"]
    res = {"format": RESULT_FORMAT, "alias": "prn", "spec": SPEC, "domain": DOMAIN, "variant": variant, "T": T,
           "n_boot": int(n_boot), "seed": int(seed), "registered_resampling": int(n_boot) == N_BOOT and int(seed) == SEED,
           "inputs": {"confirm_panel_sha1": panel_sha1, "gateft_split_sha1": sha1_file(split_report),
                      "manifest": man_rec, "arms_run": list(arms)},
           "claim": {"contrast": "P2-P1", "role": "confirmatory (the only S6 test; alpha 0.05, no Holm)",
                     "label": conf["label"], "wording": WORDING[conf["label"]], "est": conf["est"], "lo": conf["lo"],
                     "hi": conf["hi"], "p": conf["p"], "n_users": conf["n_users"], "per_seed": conf["per_seed"],
                     "sd_seed": conf["sd_seed"]},
           "arms": arm_out, "contrasts": con_out, "problems": problems,
           "rule": "A3 section 6: P2 beats random iff the CI lower bound > 0, the mean exceeds 2 x SD of the 5 paired "
                   "seed differences and all 5 differences are positive; worse than random under the mirrored "
                   "condition; about equal iff the 95% CI of the contrast lies within +-0.01; otherwise inconclusive. "
                   "P3 - P1, P1 - P0 and P2 - P0 are descriptive.",
           "operationalizations": list(ANALYSIS_OPERATIONALIZATIONS)}
    return res


CSV_COLS = ("kind", "name", "role", "status", "est", "lo", "hi", "p", "n_users", "sd_seed", "label")


def result_csv(res: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLS)

    def num(x):
        return "" if x is None or (isinstance(x, float) and not math.isfinite(x)) else repr(float(x))
    for arm, a in res["arms"].items():
        if a["status"] == "NOT_RUN":
            w.writerow(["arm", arm, "", "NOT_RUN", "", "", "", "", "", "", ""])
            continue
        for s in SEEDS:
            r = a["seeds"][f"s{s}"]
            w.writerow(["run", f"{arm}/s{s}", r.get("source", ""), r["status"], num(r.get("UAUC")), "", "", "",
                        r.get("n_users", ""), "", ""])
        w.writerow(["arm", arm, "seed_averaged", "complete" if a["complete"] else "INCOMPLETE",
                    num(a["UAUC_seed_averaged"]), "", "", "", a["n_users_common"], num(a["sd_seed"]), ""])
    for name, c in res["contrasts"].items():
        if c.get("status") == "NOT_RUN":
            w.writerow(["contrast", name, c["role"], "NOT_RUN", "", "", "", "", "", "", ""])
            continue
        w.writerow(["contrast", name, c["role"], c["status"], num(c["est"]), num(c["lo"]), num(c["hi"]), num(c["p"]),
                    c["n_users"], num(c["sd_seed"]), c["label"] or ""])
    return buf.getvalue()


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S6 pruning by uncertainty (A3 section 6): signals, verify, analyze")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("signals", help="stage B: signals, registered subsets, pruned TRAIN files, prune_manifest.json")
    s.add_argument("--train", required=True, help="outputs/confrec/gateft/train.jsonl (Gate-FT's TRAIN set)")
    s.add_argument("--zs_dir", required=True, help="the zero-shot like pass of --train (pyes_scorer output dir)")
    s.add_argument("--raw", required=True, help="data/raw (ML-1M ratings at RAW/ml-1m/ratings.dat or RAW/ratings.dat)")
    s.add_argument("--variant", required=True, help="the selected prompt variant (selection.json gate_ft_prompt)")
    s.add_argument("--out_dir", required=True)
    s.add_argument("--split_report", default=None, help="outputs/confrec/gateft/gateft_split.json (checks the TRAIN set)")
    v = sub.add_parser("verify", help="every file of the manifest has its recorded sha1")
    v.add_argument("--out_dir", required=True)
    z = sub.add_parser("analyze", help="stage E: per-seed UAUCs, the contrasts and the P2 - P1 label")
    z.add_argument("--confirm_panel", required=True)
    z.add_argument("--split_report", required=True, help="gateft_split.json (T)")
    z.add_argument("--scores_root", required=True, help="<arm>/s<seed>/ scoring dirs of run_ftprune.sh")
    z.add_argument("--gateft_scores", required=True, help="s0-s2 scoring dirs of Gate-FT (P0 s0-s2)")
    z.add_argument("--adapters_root", required=True, help="<arm>/s<seed>/ adapters of run_ftprune.sh")
    z.add_argument("--gateft_adapters", required=True, help="s0-s2 Gate-FT adapters")
    z.add_argument("--variant", required=True)
    z.add_argument("--out", required=True)
    z.add_argument("--manifest", default=None)
    z.add_argument("--arms", default=",".join(ARMS), help="arms run (P3 may be cut: P0,P1,P2)")
    z.add_argument("--n_boot", type=int, default=N_BOOT)
    z.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    if a.cmd == "signals":
        man = build_signals(a.train, a.zs_dir, a.raw, a.variant, a.out_dir, a.split_report)
        cl = man["classes"]
        print(f"S6 signals: {man['inputs']['train_jsonl']['examples']} TRAIN examples, beta = "
              f"{man['signals']['label_mean_beta']:.4f}, tau = {man['signals']['tau']:.4f}; removed per class: "
              f"positives {cl['1']['n_remove']}/{cl['1']['n']}, negatives {cl['0']['n_remove']}/{cl['0']['n']}; "
              f"manifest {Path(a.out_dir) / MANIFEST}")
        return 0
    if a.cmd == "verify":
        probs = verify(a.out_dir)
        for p in probs:
            print("VERIFY:", p, flush=True)
        if probs:
            return 1
        print(f"verify OK: every file of {Path(a.out_dir) / MANIFEST} has its recorded sha1")
        return 0
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        raise SystemExit(f"--arms: unknown {bad} (choose from {list(ARMS)})")
    res = analyze(a.confirm_panel, a.split_report, a.scores_root, a.gateft_scores, a.adapters_root, a.gateft_adapters,
                  a.variant, arms, a.manifest, a.n_boot, a.seed)
    out = Path(a.out)
    put(out, json_bytes(res))
    put(out.with_suffix(".csv"), result_csv(res).encode("utf-8"))
    c = res["claim"]
    print(f"S6 P2 - P1 (seed-averaged UAUC contrast) = {c['est']} [{c['lo']}, {c['hi']}], p = {c['p']}, "
          f"{c['n_users']} users; per seed {c['per_seed']}")
    for name, x in res["contrasts"].items():
        if name != "P2-P1":
            print(f"  {name} ({x['role']}): {x.get('status')} est {x.get('est')} [{x.get('lo')}, {x.get('hi')}]")
    for p in res["problems"]:
        print("PROBLEM:", p)
    print("LABEL:", c["label"])
    return 2 if c["label"] == "INCOMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
