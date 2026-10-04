"""Diagnosis battery of idea-stage/PREREG_AMENDMENT_2.md section D, plus the amendment-2 freeze helpers used by
scripts/sigir/run_gatefix.sh and scripts/sigir/run_diag_battery.sh.

Every battery reading is INTERPRETIVE ONLY: it runs on burned Pilot-1 users and can never change G0-G9 (prompt bank,
thresholds, splits, contingency). The registered interpretive thresholds are applied to point estimates; CIs are
reported (users resampled for pair statistics, items for item-level correlations, amendment 1 C1).

Subcommands
  make     derive a battery panel from a burned rated panel (rows keep user_id, labels and candidate order):
    t0       one row per unique candidate item, empty history, domain_kind kept; scored with
             `--variant T0_probe --questions like` (the probe text lives in the variant). Each row carries the item
             reference for the T0 Spearman: probe_ref_prior_mean = mean over the item's panel pairs of the
             prior-only leave-user-out item mean (below), and the all-time item mean as a sensitivity.
    t1       positive control: candidate_extras "(This user later rated this item r/5.)" with the true rating r.
    t2       candidate_extras "Average rating by other users before this date: m/5 (n ratings)" (the registered
             template verbatim, also for n = 1; "... no ratings yet" when n = 0), m = prior-only leave-user-out item
             mean: every OTHER user's first rating of the item (the panel builder's per-(user, item) dedup) with a
             timestamp strictly before the candidate's. m is printed with 2 decimals; the exact m and n are kept in
             candidate_prior_mean / candidate_prior_n.
    starperm K copies (default 2) of each row whose rendered history window (last --hist_len entries) has its
             "(rated r/5)" suffixes permuted among the same items: each copy is a uniformly random derangement of
             the window positions (no suffix stays on its item; the identity when the window has < 2 events), seeded
             per (seed, copy, source_event_id), redrawn while its displayed ratings equal an earlier copy's whenever
             another sequence exists (always for >= 3 events with >= 2 distinct ratings), so the K copies are K
             different prompts. Titles stay with their items; history_ratings follow the suffixes. source_event_id
             gets "::perm<k>"; perm_sigma[j] = window index whose suffix moved to window position j; perm_n_changed =
             displayed ratings that changed (a swap of two equal ratings changes none).
  analyze  section-D readings from ROOT (layout below) -> JSON: T0 Spearman(P(Yes) of the probe, prior-only item
           mean) with the like-logit rho (probe logit and all-time mean as sensitivities); T1 UAUC; T2 transmission
           (UAUC_T2 - 0.5) / (UAUC_itemmean_prior - 0.5) on pairs with n >= 1 (all pairs, n = 0 imputed, as a
           sensitivity; undefined bootstrap replicates counted; a caveat when the denominator CI covers 0.5); T3
           digit-readout UAUC (E[r]) vs yes/no; star permutation dUAUC = UAUC(L) - mean_k UAUC(L_perm_k) and the
           pooled within-user SD (divisor N - n_users, as SD(a) in amendment 1 P1.5b) of tau_P = L - mean_k
           L_perm_k; T4 Llama UAUC. Missing inputs are reported, not fatal; a duplicated score row is an error.
           Provenance (per arm): the report.json settings (variant, readout, backbone, questions [like], lora null,
           float16, top-50, max_model_len 4096, registered history window, prompts == score rows), every battery arm
           scored exactly its battery panel (data_sha1), that panel derives from the panel and rows the base arm
           scored, and T3/T4 scored the base panel; a mismatch sets provenance_ok = false (and is printed), it does
           not stop the analysis. OPERATIONALIZATIONS (the choices section D leaves open) are copied into the JSON.
  t1check  the registered T1 stop rule, run by run_diag_battery.sh right after the T1 arm and before every other
           arm: exit 0 when T1 UAUC >= 0.90, 5 when < 0.90 ("readout broken; stop and debug"), 2 when the T1 arm is
           missing or did not score the T1 panel with the registered settings (JSON record ROOT/ml1m/t1check.json).
  freeze   FREEZE.txt for the amendment-2 freeze rule: sha1 of the amendment file, sha1 of the rendered prompt bank
           (every prompting.VARIANTS key x first --n_rows rows of each dev panel x every candidate x --question; each
           prompt hashed as "<panel basename>\\x1f<variant>\\x1f<source_event_id>\\x1f<cand_idx>\\x1f<prompt>\\x1e" with
           prompt = system + "\\x1d" + user when a system message exists, else user), sha1 of the battery's own
           prompts and rules (battery_bank: T0/T0 digits/T1/T2/T3/star-permutation renders of the same rows plus
           battery_spec(), so the T1/T2 extras, digit questions, permutation scheme and OPERATIONALIZATIONS are
           frozen before any battery GPU job), the user-id list sha1s of the panel manifest (its `freeze` block),
           prompting.PROMPT_STRINGS_SHA1 and the dev panel file sha1s. An existing FREEZE.txt that differs from the
           recomputed text is an error. --check additionally requires FREEZE.txt to exist unchanged and every
           REQUIRED sha1 (amendment, prompt bank, battery bank, user-id lists) to appear in --pilot_log.
  vstar    print V* from gatefix_select's selection.json (refuses V0 or an explicit fix_found = false).

    python -m src.confrec.diag_battery make --kind t2 --panel outputs/confrec/panels/ml1m_rated.jsonl \
        --raw data/raw/ml-1m --source ml1m --out outputs/confrec/diag/panels/ml1m_t2.jsonl
    python -m src.confrec.diag_battery make --kind t2 --panel outputs/confrec/panels/toys_rated.jsonl \
        --raw data/raw --source amazon --domain toys --n_users 500 --out outputs/confrec/diag/panels/toys_t2.jsonl
    python -m src.confrec.diag_battery t1check --root outputs/confrec/diag
    python -m src.confrec.diag_battery analyze --root outputs/confrec/diag --out outputs/confrec/diag/diag.json
    python -m src.confrec.diag_battery freeze --amendment idea-stage/PREREG_AMENDMENT_2.md \
        --dev_panels outputs/confrec/gatefix/panels/ml1m_dev_h20.jsonl,outputs/confrec/gatefix/panels/toys_dev_h20.jsonl \
        --manifest outputs/confrec/gatefix/panels/manifest.json --out outputs/confrec/gatefix/FREEZE.txt \
        [--check --pilot_log docs/sigir/PILOT_LOG.md]
    python -m src.confrec.diag_battery vstar --selection outputs/confrec/gatefix/dev/selection.json

ROOT layout for analyze (written by scripts/sigir/run_diag_battery.sh; scores are pyes_scorer outputs):
    ROOT/panels/{ml1m_t0, ml1m_t1, ml1m_t2, ml1m_starperm, toys_t2}.jsonl   (+ .meta.json sidecars)
    ROOT/ml1m/{base, t0, t0_digits, t1, t2, t3, starperm, t4_llama}/scores.csv.gz
        base = V0 like yes/no (Qwen3-8B) on the burned panel, the paired reference of T1/T2/T3/star permutation;
        t0 = T0_probe yes/no (the registered T0 reading reads P(Yes) = exp(lp_yes)); t0_digits = T0_probe --readout
        digits (optional, E[r] Spearman reported as a secondary quantity); t3 = V0 like with --readout digits
        (exp_rating column);
        t4_llama = V0 like with Llama-3.1-8B-Instruct
    ROOT/toys/{base, t2}/scores.csv.gz      (the first 500 burned Toys users)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from itertools import accumulate
from pathlib import Path

import numpy as np

from src.confrec.categories import CATEGORY
from src.confrec.metrics import auroc
from src.confrec.stats import cluster_bootstrap, paired_bootstrap, percentile_ci, spearman, strict_json

NAN = float("nan")
KINDS = ("t0", "t1", "t2", "starperm")
GATE_VARIANTS = ("V0", "V1", "V2", "V3", "V4", "V5", "V7")
T1_EXTRA = "(This user later rated this item {r}/5.)"
T2_PREFIX = "Average rating by other users before this date: "
INTERPRETIVE = ("PREREG_AMENDMENT_2 section D: interpretive only, on burned users; cannot change G0-G9 "
                "(prompt bank, thresholds, splits, contingency).")
# registered interpretive thresholds (amendment 2, section D table), applied to point estimates
THRESHOLDS = {
    "T0": {"knowledge_exists_rho_ge": 0.5, "knowledge_limited_rho_lt": 0.3, "companion_like_logit_rho_about": 0.15},
    "T1": {"readout_broken_uauc_lt": 0.90},
    "T2": {"knowledge_limited_transmission_ge": 0.8, "readout_limited_transmission_lt": 0.5},
    "T3": {"format_bottleneck_duauc_ge": 0.03},
    "star_permutation": {"abs_duauc_lt": 0.005, "within_user_sd_tau_lt": 0.10},
    "T4": {"exploratory_uauc_ge": 0.62},
}
STARPERM_K, STARPERM_SEED, V0_HIST = 2, 0, 10   # registered K = 2; V0's rendered rated history window
MAX_REDRAWS = 1000                               # perm_copies: draws per copy when looking for a different prompt
T1_BROKEN_EXIT = 5                               # t1check exit code for "readout broken; stop and debug"
# Choices section D leaves open, fixed here before any battery GPU job (hashed into FREEZE.txt's battery bank and
# copied into diag.json). None of them can change G0-G9.
OPERATIONALIZATIONS = (
    "all readings: registered thresholds applied to point estimates; CIs are 95% percentile bootstraps over users "
    "(pair statistics) or items (item-level correlations), reported only",
    "UAUC: mean over users with both label classes among the pairs of the per-user AUC (ties averaged); rows "
    "censored 2/3 or with a non-finite value are dropped; a duplicated (source_event_id, cand_idx) is an error",
    "T0 quantity: Spearman (ties averaged) over the probed items of P(Yes) = exp(lp_yes) of the T0_probe yes/no "
    "readout (the raw next-token probability, judge.md 'Read P(Yes)'); the T0 logit log P(Yes) - log P(No) and the "
    "T0_probe digit E[r] are reported as sensitivity / secondary quantities and never enter the reading",
    "T0 reference: item-level prior-only item mean = mean over the item's panel pairs of the pair-level prior-only "
    "leave-user-out item mean (pairs without a prior rating skipped; items with none excluded); the all-time item "
    "mean of the first ratings is a sensitivity",
    "T0 companion 'like-logit rho of about 0.15' (no tolerance registered): reported as the Spearman of the V0 like "
    "logit with the pair-level prior-only item mean over the burned ML-1M pairs (user-cluster CI) and at item level "
    "(per-item mean like logit); no cutoff is applied, the knowledge_exists reading names the observed value",
    "prior-only leave-user-out item mean (T0, T2): every OTHER user's first rating of the item (the panel builder's "
    "per-(user, item) dedup) with a timestamp strictly before the candidate's timestamp",
    "T1: scored first; diag_battery t1check stops run_diag_battery.sh (exit 5) before any other battery arm when T1 "
    "UAUC < 0.90 ('stop and debug'); T1_OVERRIDE=1 continues only after the debugging is recorded in PILOT_LOG",
    "T2 extra: the registered template verbatim, m printed with 2 decimals, '(n ratings)' also for n = 1, "
    "'... no ratings yet' when n = 0",
    "T2 transmission: primary population = pairs whose candidate has >= 1 prior rating, both UAUC_T2 and "
    "UAUC_itemmean_prior on those pairs and over the users common to both; ratio of the two means; all pairs with the "
    "n = 0 means imputed, and the 2-decimal displayed mean, are sensitivities; bootstrap replicates whose item-mean "
    "UAUC is <= 0.5 are undefined and counted, and the reading carries a caveat when the 95% CI of "
    "UAUC_itemmean_prior covers 0.5",
    "T3: UAUC of E[r] (V0 like, digit readout) minus the yes/no UAUC on the same pairs, paired user bootstrap",
    "star permutation: each of the K = 2 copies is a uniformly random derangement of the positions of the rendered "
    "last-10 window (the identity when < 2 events), seeded per (seed 0, copy, source_event_id); a copy whose "
    "displayed ratings equal an earlier copy's is redrawn whenever a different sequence exists (always for >= 3 "
    "events with >= 2 distinct ratings); titles stay with their items; dUAUC = UAUC(L) - mean over users of the "
    "per-user mean over k of AUC(L_perm_k); tau_P = L - mean_k L_perm_k on the pairs finite in base and every copy; "
    "within-user SD pooled with divisor N - n_users",
    "T4: Llama-3.1-8B-Instruct V0 like UAUC on the burned ML-1M panel; >= 0.62 is exploratory only (no backbone "
    "rescue, G8)",
    "provenance: every arm's report.json must show its registered variant / readout / backbone, questions [like], "
    "lora null, float16, top-50 logprobs, max_model_len 4096, the registered history window and as many prompts as "
    "score rows; battery arms must have scored their battery panel, derived from the panel the base arm scored",
)
_RATED = re.compile(r"^(.*) \(rated (\d+)/5\)$", re.S)   # rated-panel history line "<title> (rated r/5)"
_PERM = re.compile(r"^(.*)::perm(\d+)$", re.S)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
VSTAR_KEYS = ("v_star", "vstar", "V_star", "Vstar", "V*", "selected_variant", "selected", "winner")


# ---------------------------------------------------------------- io
def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def _open(path):
    path = str(path)
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.endswith(".gz") else \
        open(path, encoding="utf-8", newline="")


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    """Atomic (<path>.tmp then os.replace), one strict-JSON row per line, as build_rated_panels writes panels."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(strict_json(r), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(strict_json(obj), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def file_sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- prior-only leave-user-out item mean
def first_ratings(raw, source: str, domain: str | None = None, items=None) -> dict:
    """{item: [(ts, rating, user)] sorted}: each user's FIRST rating of the item, i.e. the minimum (ts, rating) over
    that user's events of the item -- the same event build_rated_panels keeps (sorted(set(events)) then first per
    item). Only items in `items` are kept when given. ML-1M reads raw/ratings.dat; Amazon reads
    raw/amazon_<domain>/<Category>.jsonl.gz (rows with a rating; the panel's candidate items all have a title)."""
    keep = None if items is None else {str(i) for i in items}
    first: dict = {}

    def add(u, i, ts, r):
        if keep is not None and i not in keep:
            return
        k, v = (u, i), (ts, r)
        if k not in first or v < first[k]:
            first[k] = v

    raw = Path(raw)
    if source == "ml1m":
        with open(raw / "ratings.dat", encoding="latin-1") as f:
            for line in f:
                uid, mid, r, ts = line.rstrip("\n").split("::")
                add(uid, mid, int(ts), float(r))
    elif source == "amazon":
        if domain not in CATEGORY:
            raise SystemExit(f"--domain {domain!r} not in {sorted(CATEGORY)}")
        with gzip.open(raw / f"amazon_{domain}" / f"{CATEGORY[domain]}.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("rating") is None:
                    continue
                add(str(r["user_id"]), str(r["parent_asin"]), int(r["timestamp"]), float(r["rating"]))
    else:
        raise SystemExit(f"unknown source {source!r}")
    out: dict = defaultdict(list)
    for (u, i), (ts, r) in first.items():
        out[i].append((ts, r, u))
    for v in out.values():
        v.sort()
    return dict(out)


class PriorIndex:
    """Prior-only leave-user-out item mean: query(user, item, ts) -> (mean, n) over the other users' first ratings of
    `item` with timestamp strictly before `ts` ((nan, 0) when there is none)."""

    def __init__(self, by_item: dict):
        self.ts = {i: [e[0] for e in v] for i, v in by_item.items()}
        self.cum = {i: [0.0] + list(accumulate(e[1] for e in v)) for i, v in by_item.items()}
        self.own = {(e[2], i): (e[0], e[1]) for i, v in by_item.items() for e in v}

    def query(self, user, item, ts) -> tuple[float, int]:
        user, item = str(user), str(item)
        t = self.ts.get(item, [])
        k = bisect_left(t, ts)
        s, n = (self.cum[item][k] if t else 0.0), k
        own = self.own.get((user, item))
        if own is not None and own[0] < ts:   # the user's own earlier rating never counts (leave-user-out)
            s, n = s - own[1], n - 1
        return (s / n if n > 0 else NAN), n


def t2_extra(mean: float, n: int) -> str:
    """The registered template verbatim ("(n ratings)" for every n >= 1); m with 2 decimals."""
    if n <= 0:
        return T2_PREFIX + "no ratings yet"
    return T2_PREFIX + f"{mean:.2f}/5 ({n} ratings)"


# ---------------------------------------------------------------- make
def _star(x) -> int:
    v = float(x)
    if not (v.is_integer() and 1 <= v <= 5):
        raise ValueError(f"rating {x!r} is not an integer star 1-5")
    return int(v)


def _no_extras(rec: dict) -> None:
    if any(rec.get("candidate_extras") or []):
        raise ValueError(f"{rec.get('source_event_id')}: panel already carries candidate_extras; derive battery "
                         "panels from the plain burned panel")


def make_t1(rows: list[dict]) -> list[dict]:
    out = []
    for rec in rows:
        _no_extras(rec)
        rat = rec["candidate_ratings"]
        if len(rat) != len(rec["candidate_item_ids"]):
            raise ValueError(f"{rec.get('source_event_id')}: candidate_ratings misaligned")
        r = dict(rec)
        r["candidate_extras"] = [T1_EXTRA.format(r=_star(x)) for x in rat]
        r["battery"] = "t1"
        out.append(r)
    return out


def make_t2(rows: list[dict], prior: PriorIndex) -> tuple[list[dict], dict]:
    out, n_pairs, n0 = [], 0, 0
    for rec in rows:
        _no_extras(rec)
        u, ts = str(rec["user_id"]), rec["candidate_timestamps"]
        if len(ts) != len(rec["candidate_item_ids"]):
            raise ValueError(f"{rec.get('source_event_id')}: candidate_timestamps misaligned")
        means, ns = [], []
        for iid, t in zip(rec["candidate_item_ids"], ts):
            m, n = prior.query(u, iid, int(t))
            means.append(m)
            ns.append(n)
        r = dict(rec)
        r["candidate_extras"] = [t2_extra(m, n) for m, n in zip(means, ns)]
        r["candidate_prior_mean"] = [None if not math.isfinite(m) else m for m in means]
        r["candidate_prior_n"] = ns
        r["battery"] = "t2"
        out.append(r)
        n_pairs += len(ns)
        n0 += sum(n == 0 for n in ns)
    return out, {"n_pairs": n_pairs, "n_pairs_no_prior_rating": n0}


def make_t0(rows: list[dict], prior: PriorIndex, by_item: dict, source: str) -> list[dict]:
    """One probe row per unique candidate item (first-appearance order)."""
    items: dict = {}
    for rec in rows:
        u = str(rec["user_id"])
        texts = rec.get("candidate_texts") or [""] * len(rec["candidate_titles"])
        tss = rec.get("candidate_timestamps")
        for k, (iid, title, text) in enumerate(zip(rec["candidate_item_ids"], rec["candidate_titles"], texts)):
            if iid not in items:
                items[iid] = {"title": title, "text": text, "pm": [],
                              "domain_kind": rec.get("domain_kind") or ("movie" if source == "ml1m" else "product"),
                              "source": rec.get("source", source)}
            if tss is not None:
                m, n = prior.query(u, iid, int(tss[k]))
                if n > 0:
                    items[iid]["pm"].append(m)
    out = []
    for iid, it in items.items():
        allr = [e[1] for e in by_item.get(str(iid), [])]
        out.append({
            "user_id": f"t0::{iid}", "source_event_id": f"t0::{iid}",
            "history": [], "history_item_ids": [], "history_titles": [], "history_ratings": [], "history_meta": [],
            "candidate_item_ids": [iid], "candidate_titles": [it["title"]], "candidate_texts": [it["text"]],
            "candidate_labels": [0],   # placeholder: a no-user probe has no label (pyes_scorer needs the field)
            "domain_kind": it["domain_kind"], "source": it["source"], "battery": "t0",
            "probe_ref_prior_mean": float(np.mean(it["pm"])) if it["pm"] else None,
            "probe_ref_n_pairs": len(it["pm"]),
            "probe_ref_alltime_mean": float(np.mean(allr)) if allr else None,
            "probe_ref_alltime_n": len(allr),
        })
    return out


def random_derangement(n: int, rng: random.Random) -> list[int]:
    """A uniformly random derangement of range(n): uniform shuffles (rng.shuffle) rejected until no position is fixed
    (expected e draws), i.e. uniform over the derangements. The identity when n < 2 (no derangement exists)."""
    if n < 2:
        return list(range(n))
    while True:
        s = list(range(n))
        rng.shuffle(s)
        if all(s[j] != j for j in range(n)):
            return s


def perm_copies(values: list, k: int, seed: int, sid: str) -> tuple[list[list[int]], int]:
    """K suffix permutations of one rendered window (sigma with new[j] = values[sigma[j]]).

    Copy kk is a uniformly random derangement of the window positions drawn from random.Random(f"{seed}:{kk}:{sid}")
    (so copy 0 does not depend on K), redrawn from the same stream while its displayed ratings equal an earlier
    copy's, so the K copies are K different prompts whenever the window allows it: for K = 2 that is every window with
    >= 3 events and >= 2 distinct ratings (test_derangements_reach_two_displayed_sequences). Windows with < 3 events or
    one distinct rating cannot differ and are not redrawn; beyond K = 2 at most MAX_REDRAWS draws are tried.
    Returns (sigmas, number of copies whose displayed ratings equal an earlier copy's)."""
    n = len(values)
    can_differ = n >= 3 and len(set(values)) >= 2
    sigmas, shown, dup = [], [], 0
    for kk in range(k):
        rng = random.Random(f"{seed}:{kk}:{sid}")
        sigma = random_derangement(n, rng)
        tries = 1
        while can_differ and tuple(values[j] for j in sigma) in shown and tries < MAX_REDRAWS:
            sigma = random_derangement(n, rng)
            tries += 1
        disp = tuple(values[j] for j in sigma)
        dup += disp in shown
        shown.append(disp)
        sigmas.append(sigma)
    return sigmas, dup


def make_starperm(rows: list[dict], k: int = STARPERM_K, seed: int = STARPERM_SEED,
                  hist_len: int = V0_HIST) -> tuple[list[dict], dict]:
    if k < 1:
        raise SystemExit("--k must be >= 1")
    out, changed, n_rows_nochange, n_dup = [], [], 0, 0
    for rec in rows:
        sid = str(rec.get("source_event_id", rec["user_id"]))
        hist = list(rec["history"])
        start = max(0, len(hist) - hist_len) if hist_len > 0 else 0
        parts = []
        for j in range(start, len(hist)):
            mt = _RATED.match(str(hist[j]))
            if not mt:
                raise ValueError(f"{sid}: history line {hist[j]!r} has no '(rated r/5)' suffix")
            parts.append((mt.group(1), int(mt.group(2))))
        hr = rec.get("history_ratings")
        if hr is not None:
            if len(hr) != len(hist):
                raise ValueError(f"{sid}: history_ratings misaligned")
            bad = [j for j, (_, v) in enumerate(parts) if _star(hr[start + j]) != v]
            if bad:
                raise ValueError(f"{sid}: history suffix and history_ratings disagree at {bad}")
        vals = [v for _, v in parts]
        sigmas, dup = perm_copies(vals, k, seed, sid)
        n_dup += dup
        for kk, sigma in enumerate(sigmas):
            r = dict(rec)
            nh = list(hist)
            for j, (title, _) in enumerate(parts):
                nh[start + j] = f"{title} (rated {vals[sigma[j]]}/5)"
            r["history"] = nh
            if hr is not None:
                nr = list(hr)
                for j in range(len(parts)):
                    nr[start + j] = hr[start + sigma[j]]
                r["history_ratings"] = nr
            n_ch = sum(vals[sigma[j]] != vals[j] for j in range(len(vals)))
            r.update(source_event_id=f"{sid}::perm{kk}", perm_of=sid, perm_k=kk, perm_sigma=sigma,
                     perm_window=len(vals), perm_n_changed=n_ch, battery="starperm")
            out.append(r)
            changed.append(n_ch)
            n_rows_nochange += n_ch == 0
    return out, {"k": k, "seed": seed, "hist_len": hist_len, "scheme": "uniform random derangement per copy, copies "
                 "redrawn to differ in displayed ratings where possible",
                 "mean_n_changed": float(np.mean(changed)) if changed else NAN,
                 "copies_without_change": n_rows_nochange, "copies_equal_to_an_earlier_copy": n_dup}


def make(kind: str, panel, out, raw=None, source: str = "ml1m", domain=None, n_users=None, k: int = STARPERM_K,
         seed: int = STARPERM_SEED, hist_len: int = V0_HIST) -> dict:
    rows = read_jsonl(panel)
    if n_users:
        rows = rows[:n_users]
    if not rows:
        raise SystemExit(f"no rows in {panel}")
    info: dict = {}
    if kind in ("t0", "t2"):
        if raw is None:
            raise SystemExit(f"--raw is required for --kind {kind}")
        need = {str(i) for r in rows for i in r["candidate_item_ids"]}
        by_item = first_ratings(raw, source, domain, items=need)
        prior = PriorIndex(by_item)
        if kind == "t0":
            new = make_t0(rows, prior, by_item, source)
            info = {"n_items": len(new), "n_items_without_prior": sum(r["probe_ref_prior_mean"] is None for r in new)}
        else:
            new, info = make_t2(rows, prior)
    elif kind == "t1":
        new = make_t1(rows)
    elif kind == "starperm":
        new, info = make_starperm(rows, k=k, seed=seed, hist_len=hist_len)
    else:
        raise SystemExit(f"unknown --kind {kind!r}; choose from {KINDS}")
    out = Path(out)
    meta = {"kind": kind, "panel": str(panel), "panel_sha1": file_sha1(panel), "n_rows_in": len(rows),
            "n_rows_out": len(new), "n_users": n_users, "source": source, "domain": domain,
            "interpretive_only": True, **info}
    write_json(out.with_suffix(".meta.json"), meta)   # sidecar first: the panel appears last, atomically
    write_jsonl(out, new)
    return meta


# ---------------------------------------------------------------- analyze: loading and per-user AUC
LOGIT, EXP, LPY = 3, 4, 5   # value columns of load_arm rows: (user, item, label, logit, exp_rating, lp_yes)


def load_arm(path, question: str = "like") -> dict:
    """pyes_scorer scores.csv.gz -> {"rows": {(source_event_id, cand_idx): (user, item, label, logit, exp_rating,
    lp_yes)}, "counts", "mean_yes_no_mass"}. Rows of other questions are ignored. A value is NaN when non-finite or
    when the row is censored 2/3 (dropped from every statistic and counted); exp_rating / lp_yes are NaN when the
    column is absent. A duplicated (source_event_id, cand_idx) is an error (a stale or mixed output directory;
    gatefix_select refuses the same), never silently overwritten."""
    rows, counts, mass = {}, Counter(), []
    with _open(path) as f:
        for r in csv.DictReader(f):
            if r.get("question", question) != question:
                continue
            code = (r.get("censored") or "na").strip()
            counts["n"] += 1
            counts[f"censored_{code}"] += 1
            key = (r["source_event_id"], int(r["cand_idx"]))
            if key in rows:
                raise ValueError(f"{path}: duplicated score row (source_event_id, cand_idx) = {key} for question "
                                 f"{question!r} (stale or mixed output directory)")
            dead = code in ("2", "3")
            lg, er, ly = _num(r.get("logit")), _num(r.get("exp_rating")), _num(r.get("lp_yes"))
            if dead or not math.isfinite(lg):
                lg = NAN
                counts["dropped_logit"] += 1
            if dead or not math.isfinite(er):
                er = NAN
            if dead or not math.isfinite(ly):
                ly = NAN
            m = _num(r.get("yes_no_mass"))
            if math.isfinite(m):
                mass.append(m)
            rows[key] = (r["user_id"], r["item_id"], int(float(r["label"])), lg, er, ly)
    return {"rows": rows, "counts": dict(counts), "mean_yes_no_mass": float(np.mean(mass)) if mass else NAN,
            "path": str(path)}


def _arrays(arm: dict, col: int = LOGIT, keys=None):
    """(keys, users, labels, values) over the arm's rows (or `keys`) whose value is finite."""
    keys = [k for k in (keys if keys is not None else arm["rows"]) if math.isfinite(arm["rows"][k][col])]
    v = arm["rows"]
    return (keys, np.array([v[k][0] for k in keys], dtype=object), np.array([v[k][2] for k in keys], int),
            np.array([v[k][col] for k in keys], float))


def _aligned(a: dict, b: dict, col_a: int = LOGIT, col_b: int = LOGIT):
    """(users, labels, va, vb) over keys present and finite in both arms; the labels must agree."""
    keys = [k for k in a["rows"] if k in b["rows"] and math.isfinite(a["rows"][k][col_a])
            and math.isfinite(b["rows"][k][col_b])]
    bad = [k for k in keys if a["rows"][k][2] != b["rows"][k][2] or a["rows"][k][0] != b["rows"][k][0]]
    if bad:
        raise ValueError(f"{a['path']} vs {b['path']}: user/label disagree on {len(bad)} pairs, e.g. {bad[:3]}")
    return (np.array([a["rows"][k][0] for k in keys], dtype=object), np.array([a["rows"][k][2] for k in keys], int),
            np.array([a["rows"][k][col_a] for k in keys], float), np.array([b["rows"][k][col_b] for k in keys], float))


def _by_user(users) -> dict:
    g: dict = defaultdict(list)
    for j, u in enumerate(users):
        g[u].append(j)
    return {u: np.asarray(v, int) for u, v in g.items()}


def per_user_auc(users, labels, scores) -> dict:
    """{user: AUC} for users with both classes among the given rows (ties averaged)."""
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    out = {}
    for u, idx in _by_user(users).items():
        y = labels[idx]
        if 0 < y.sum() < len(y):
            out[u] = auroc(scores[idx], y)
    return out


def uauc_ci(pu: dict, n_boot: int, seed: int) -> dict:
    """Mean per-user AUC with a user-bootstrap percentile CI."""
    v = np.array([x for x in pu.values() if math.isfinite(x)], float)
    if len(v) == 0:
        return {"UAUC": NAN, "lo": NAN, "hi": NAN, "n_users": 0}
    rng = np.random.default_rng(seed)
    lo, hi = percentile_ci([float(v[rng.integers(0, len(v), len(v))].mean()) for _ in range(n_boot)])
    return {"UAUC": float(v.mean()), "lo": lo, "hi": hi, "n_users": int(len(v))}


def transmission(t2: dict, item: dict, n_boot: int, seed: int) -> dict:
    """(mean t2 - 0.5) / (mean item - 0.5) over common users, user-bootstrap CI (both means on the same resample).
    Undefined (NaN) when the item-mean UAUC is <= 0.5; undefined replicates are left out of the percentile CI and
    counted (n_boot_undefined / share_boot_undefined), so a CI resting on few defined replicates is visible."""
    keys = sorted(u for u in set(t2) & set(item) if math.isfinite(t2[u]) and math.isfinite(item[u]))
    a, b = np.array([t2[u] for u in keys], float), np.array([item[u] for u in keys], float)

    def ratio(x, y):
        return float((x.mean() - 0.5) / (y.mean() - 0.5)) if len(y) and y.mean() > 0.5 else NAN

    if not keys:
        return {"est": NAN, "lo": NAN, "hi": NAN, "n_users": 0, "n_boot": n_boot, "n_boot_undefined": n_boot,
                "share_boot_undefined": 1.0 if n_boot else NAN}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        boots.append(ratio(a[idx], b[idx]))
    lo, hi = percentile_ci(boots)
    und = int(sum(not math.isfinite(x) for x in boots))
    return {"est": ratio(a, b), "lo": lo, "hi": hi, "n_users": len(keys),
            "UAUC_T2": float(a.mean()), "UAUC_itemmean_prior": float(b.mean()), "n_boot": n_boot,
            "n_boot_undefined": und, "share_boot_undefined": und / n_boot if n_boot else NAN}


def pooled_within_sd(values, users) -> float:
    """Pooled within-user SD with divisor N - n_users (amendment 1 P1.5b, SD_pair_df)."""
    values = np.asarray(values, float)
    groups = _by_user(users)
    ss = sum(float(((values[idx] - values[idx].mean()) ** 2).sum()) for idx in groups.values())
    df = len(values) - len(groups)
    return math.sqrt(ss / df) if df > 0 else NAN


def _corr_ci(x, y, clusters, n_boot: int, seed: int) -> dict:
    """Spearman (ties averaged) with a cluster-bootstrap CI (users for pairs, items for item-level rows)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, cl = x[ok], y[ok], np.asarray(clusters, dtype=object)[ok]
    if ok.sum() < 3:
        return {"est": NAN, "lo": NAN, "hi": NAN, "n": int(ok.sum())}
    r = cluster_bootstrap(lambda idx: spearman(x[idx], y[idx]), cl.astype(str), n_boot=n_boot, seed=seed)
    return {"est": r["est"], "lo": r["lo"], "hi": r["hi"], "n": int(ok.sum()), "n_clusters": r["n_clusters"]}


def _report(arm_dir: Path) -> dict | None:
    p = arm_dir / "report.json"
    if not p.exists():
        return None
    rep = json.loads(p.read_text(encoding="utf-8"))
    keep = ("data_path", "data_sha1", "model", "backbone", "variant", "readout", "questions", "hist_len",
            "max_model_len", "n_main_prompts", "censored_main", "n_overlength", "mean_yes_no_mass", "system_sha1")
    out = {k: rep[k] for k in keep if k in rep}
    out["prompts_sha1"] = (rep.get("config") or {}).get("prompts_sha1")
    return out


def _arm_info(arm: dict, arm_dir: Path) -> dict:
    return {"scores": arm["path"], "counts": arm["counts"], "mean_yes_no_mass": arm["mean_yes_no_mass"],
            "report": _report(arm_dir)}


# ---------------------------------------------------------------- analyze: blocks
def _reading(code: str, text: str, caveat: str | None = None) -> dict:
    out = {"code": code, "text": text, "interpretive_only": True}
    if caveat:
        out["caveat"] = caveat
    return out


def block_base(base: dict, n_boot: int, seed: int) -> dict:
    _, u, y, s = _arrays(base)
    return {**uauc_ci(per_user_auc(u, y, s), n_boot, seed), "n_pairs": int(len(s))}


def block_t1(t1: dict, base: dict | None, n_boot: int, seed: int) -> dict:
    _, u, y, s = _arrays(t1)
    res = {**uauc_ci(per_user_auc(u, y, s), n_boot, seed), "n_pairs": int(len(s))}
    if base is not None:
        u, y, a, b = _aligned(t1, base)
        res["dUAUC_T1_minus_base"] = paired_bootstrap(per_user_auc(u, y, a), per_user_auc(u, y, b), n_boot, seed)
    thr = THRESHOLDS["T1"]["readout_broken_uauc_lt"]
    res["reading"] = (_reading("undetermined", "T1 UAUC not computable") if not math.isfinite(res["UAUC"]) else
                      _reading("readout_broken_stop_and_debug", f"T1 UAUC < {thr}: readout broken; stop and debug")
                      if res["UAUC"] < thr else
                      _reading("readout_responds", f"T1 UAUC >= {thr}: the readout follows a stated rating"))
    return res


def _prior_map(panel_rows: list[dict]) -> dict:
    out = {}
    for rec in panel_rows:
        sid = str(rec["source_event_id"])
        for k, (m, n) in enumerate(zip(rec["candidate_prior_mean"], rec["candidate_prior_n"])):
            out[(sid, k)] = (NAN if m is None else float(m), int(n))
    return out


def block_t2(t2: dict, panel_rows: list[dict], base: dict | None, n_boot: int, seed: int) -> dict:
    prior = _prior_map(panel_rows)
    keys = [k for k in t2["rows"] if k in prior]
    if len(keys) != len(t2["rows"]):
        raise ValueError(f"{t2['path']}: {len(t2['rows']) - len(keys)} scored pairs are not in the T2 panel")
    keys, u, y, s = _arrays(t2, keys=keys)
    m = np.array([prior[k][0] for k in keys], float)
    n = np.array([prior[k][1] for k in keys], int)
    has = n > 0
    res = {"n_pairs": int(len(keys)), "n_pairs_no_prior_rating": int((~has).sum()),
           "share_no_prior_rating": float((~has).mean()) if len(has) else NAN}
    # primary: pairs with >= 1 prior rating, both scores on the same pairs
    pu_t2, pu_item = per_user_auc(u[has], y[has], s[has]), per_user_auc(u[has], y[has], m[has])
    res["primary"] = {"pairs": "candidate has >= 1 prior rating by another user",
                      "UAUC_T2": uauc_ci(pu_t2, n_boot, seed), "UAUC_itemmean_prior": uauc_ci(pu_item, n_boot, seed),
                      "transmission": transmission(pu_t2, pu_item, n_boot, seed),
                      "UAUC_itemmean_prior_displayed_2dp": uauc_ci(
                          per_user_auc(u[has], y[has], np.round(m[has], 2)), n_boot, seed)["UAUC"]}
    # sensitivity: all pairs, no-prior items imputed with the mean of the defined prior means
    imp = float(np.mean(m[has])) if has.any() else NAN
    m_all = np.where(has, m, imp)
    pu_t2a, pu_ia = per_user_auc(u, y, s), per_user_auc(u, y, m_all)
    res["all_pairs_imputed"] = {"imputed_value": imp, "UAUC_T2": uauc_ci(pu_t2a, n_boot, seed)["UAUC"],
                                "UAUC_itemmean_prior": uauc_ci(pu_ia, n_boot, seed)["UAUC"],
                                "transmission": transmission(pu_t2a, pu_ia, n_boot, seed)}
    if base is not None:
        ub, yb, a, b = _aligned(t2, base)
        res["dUAUC_T2_minus_base"] = paired_bootstrap(per_user_auc(ub, yb, a), per_user_auc(ub, yb, b), n_boot, seed)
    trm = res["primary"]["transmission"]
    tr, den = trm["est"], res["primary"]["UAUC_itemmean_prior"]
    covers = bool(den["lo"] <= 0.5) if math.isfinite(den["lo"]) else None
    res["primary"]["denominator_ci_covers_0.5"] = covers
    caveat = (f"the 95% CI of UAUC_itemmean_prior [{den['lo']:.4f}, {den['hi']:.4f}] covers 0.5, so the "
              f"transmission ratio is unstable (CI [{trm['lo']:.3f}, {trm['hi']:.3f}], {trm['n_boot_undefined']} of "
              f"{trm['n_boot']} bootstrap replicates undefined); the registered reading uses the point estimate"
              if covers else None)
    hi, lo = THRESHOLDS["T2"]["knowledge_limited_transmission_ge"], THRESHOLDS["T2"]["readout_limited_transmission_lt"]
    res["reading"] = (_reading("undetermined", "transmission undefined (item-mean UAUC <= 0.5 or no pairs)", caveat)
                      if not math.isfinite(tr) else
                      _reading("knowledge_limited", f"transmission >= {hi}: knowledge-limited", caveat) if tr >= hi
                      else _reading("readout_limited", f"transmission < {lo}: readout-limited", caveat) if tr < lo
                      else _reading("between_thresholds", f"{lo} <= transmission < {hi}: no registered reading",
                                    caveat))
    return res


def block_t3(t3: dict, base: dict, n_boot: int, seed: int) -> dict:
    u, y, er, yn = _aligned(t3, base, EXP, LOGIT)
    pu_er, pu_yn = per_user_auc(u, y, er), per_user_auc(u, y, yn)
    res = {"n_pairs": int(len(er)), "UAUC_digits_exp_rating": uauc_ci(pu_er, n_boot, seed),
           "UAUC_yesno_same_pairs": uauc_ci(pu_yn, n_boot, seed),
           "dUAUC_digits_minus_yesno": paired_bootstrap(pu_er, pu_yn, n_boot, seed)}
    has_er = any(math.isfinite(v[EXP]) for v in t3["rows"].values())
    if has_er:   # the logit column is log P(4|5) - log P(1|2) only under the digit readout
        u2, y2, lg, yn2 = _aligned(t3, base, LOGIT, LOGIT)
        res["UAUC_digits_logit_45_vs_12"] = uauc_ci(per_user_auc(u2, y2, lg), n_boot, seed)["UAUC"]
    d, thr = res["dUAUC_digits_minus_yesno"]["est"], THRESHOLDS["T3"]["format_bottleneck_duauc_ge"]
    res["reading"] = (_reading("undetermined", "the T3 arm has no finite exp_rating (was it scored with --readout "
                               "digits? see provenance)") if not has_er else
                      _reading("undetermined", "no common finite pairs") if not math.isfinite(d) else
                      _reading("format_bottleneck", f"digit UAUC >= yes/no + {thr}: format bottleneck; reported as a "
                               "secondary channel, never a gate rescue") if d >= thr else
                      _reading("no_format_bottleneck", f"digit UAUC < yes/no + {thr}"))
    return res


def block_starperm(base: dict, perm: dict, panel_rows: list[dict] | None, n_boot: int, seed: int) -> dict:
    by: dict = defaultdict(dict)
    for (sid, ci), v in perm["rows"].items():
        mt = _PERM.match(sid)
        if not mt:
            raise ValueError(f"{perm['path']}: source_event_id {sid!r} has no ::perm<k> suffix")
        by[(mt.group(1), ci)][int(mt.group(2))] = v
    ks = sorted({kk for d in by.values() for kk in d})
    keys = [k for k in base["rows"] if math.isfinite(base["rows"][k][LOGIT]) and k in by
            and all(kk in by[k] and math.isfinite(by[k][kk][LOGIT]) for kk in ks)]
    bad = [k for k in keys if any(by[k][kk][2] != base["rows"][k][2] for kk in ks)]
    if bad:
        raise ValueError(f"star permutation labels disagree with base on {len(bad)} pairs, e.g. {bad[:3]}")
    u = np.array([base["rows"][k][0] for k in keys], dtype=object)
    y = np.array([base["rows"][k][2] for k in keys], int)
    L = np.array([base["rows"][k][LOGIT] for k in keys], float)
    Lp = np.array([[by[k][kk][LOGIT] for kk in ks] for k in keys], float).reshape(len(keys), len(ks))
    res: dict = {"K": len(ks), "n_pairs": int(len(keys))}
    if not keys:
        res["reading"] = _reading("undetermined", "no common finite pairs")
        return res
    tau = L - Lp.mean(1)
    pu_base = per_user_auc(u, y, L)
    pu_perm_k = [per_user_auc(u, y, Lp[:, j]) for j in range(len(ks))]
    pu_perm = {w: float(np.mean([d[w] for d in pu_perm_k])) for w in pu_base if all(w in d for d in pu_perm_k)}
    res.update(UAUC_base=uauc_ci(pu_base, n_boot, seed)["UAUC"],
               UAUC_perm_mean_over_k=uauc_ci(pu_perm, n_boot, seed)["UAUC"],
               UAUC_perm_k=[uauc_ci(d, n_boot, seed)["UAUC"] for d in pu_perm_k],
               dUAUC_base_minus_perm=paired_bootstrap(pu_base, pu_perm, n_boot, seed),
               dUAUC_base_minus_meanperm_logit=paired_bootstrap(pu_base, per_user_auc(u, y, Lp.mean(1)), n_boot,
                                                                seed),
               tau_mean=float(tau.mean()), tau_sd_total=float(tau.std(ddof=1)) if len(tau) > 1 else NAN,
               tau_sd_within_user=pooled_within_sd(tau, u), mean_abs_tau=float(np.abs(tau).mean()))
    if panel_rows:
        nch: dict = defaultdict(int)
        for rec in panel_rows:
            nch[str(rec["user_id"])] += int(rec.get("perm_n_changed", 0))
        unchanged = {w for w in set(u) if nch.get(str(w), 0) == 0}
        keep = np.array([w not in unchanged for w in u], bool)
        res["users_without_any_rating_change"] = len(unchanged)
        if keep.any():
            pb = per_user_auc(u[keep], y[keep], L[keep])
            pp = {w: v for w, v in pu_perm.items() if w in pb}
            res["changed_users_only"] = {"dUAUC_base_minus_perm": paired_bootstrap(pb, pp, n_boot, seed),
                                         "tau_sd_within_user": pooled_within_sd(tau[keep], u[keep])}
    d, sd = res["dUAUC_base_minus_perm"]["est"], res["tau_sd_within_user"]
    t = THRESHOLDS["star_permutation"]
    res["reading"] = (_reading("undetermined", "dUAUC or SD(tau_P) not computable")
                      if not (math.isfinite(d) and math.isfinite(sd)) else
                      _reading("model_ignores_user_ratings", f"|dUAUC| < {t['abs_duauc_lt']} and within-user "
                               f"SD(tau_P) < {t['within_user_sd_tau_lt']}: the model ignores the user's ratings "
                               "(a hypothesis to re-test on fresh users)")
                      if abs(d) < t["abs_duauc_lt"] and sd < t["within_user_sd_tau_lt"] else
                      _reading("condition_not_met", "the registered 'ignores ratings' condition does not hold"))
    return res


def block_t4(llama: dict, base: dict | None, n_boot: int, seed: int) -> dict:
    _, u, y, s = _arrays(llama)
    res = {**uauc_ci(per_user_auc(u, y, s), n_boot, seed), "n_pairs": int(len(s))}
    if base is not None:
        ua, ya, a, b = _aligned(llama, base)
        res["dUAUC_llama_minus_qwen"] = paired_bootstrap(per_user_auc(ua, ya, a), per_user_auc(ua, ya, b), n_boot,
                                                         seed)
    thr = THRESHOLDS["T4"]["exploratory_uauc_ge"]
    res["reading"] = (_reading("undetermined", "T4 UAUC not computable") if not math.isfinite(res["UAUC"]) else
                      _reading("at_or_above_0.62_exploratory_only", f"UAUC >= {thr}: exploratory only (no backbone "
                               "rescue, G8)") if res["UAUC"] >= thr else
                      _reading("below_0.62", f"UAUC < {thr} (exploratory)"))
    return res


def block_t0(t0: dict, t0_rows: list[dict], base: dict | None, t2_rows: list[dict] | None, n_boot: int,
             seed: int, t0_digits: dict | None = None) -> dict:
    """T0 reading on P(Yes) = exp(lp_yes) of the yes/no probe (amendment 2 section D "P(Yes)"; judge.md "Read
    P(Yes)"). The probe logit (log P(Yes) - log P(No), which ranks items differently whenever the Yes+No mass varies)
    and the all-time item mean are sensitivities; t0_digits (same probe, --readout digits) adds the E[r] Spearman as a
    secondary quantity. None of them enters the reading."""
    ref = {str(r["candidate_item_ids"][0]): r for r in t0_rows}
    items, lg, ly = [], [], []
    for (sid, ci), v in t0["rows"].items():
        items.append(str(v[1]))
        lg.append(v[LOGIT])
        ly.append(v[LPY])
    missing = [i for i in items if i not in ref]
    if missing:
        raise ValueError(f"{t0['path']}: {len(missing)} probed items not in the T0 panel, e.g. {missing[:3]}")
    if len(set(items)) != len(items):
        raise ValueError(f"{t0['path']}: an item is probed more than once")
    lg, pyes = np.array(lg, float), np.exp(np.array(ly, float))
    prior = np.array([_num(ref[i]["probe_ref_prior_mean"]) for i in items], float)
    allt = np.array([_num(ref[i]["probe_ref_alltime_mean"]) for i in items], float)
    res: dict = {"n_items": len(items), "n_items_finite_pyes": int(np.isfinite(pyes).sum()),
                 "n_items_finite_logit": int(np.isfinite(lg).sum()),
                 "mean_yes_no_mass": t0["mean_yes_no_mass"],
                 "spearman_T0_pyes_vs_prior_item_mean": _corr_ci(pyes, prior, items, n_boot, seed),
                 "spearman_T0_pyes_vs_alltime_item_mean": _corr_ci(pyes, allt, items, n_boot, seed),
                 "spearman_T0_logit_vs_prior_item_mean": _corr_ci(lg, prior, items, n_boot, seed)}
    if t0_digits is not None:
        dv = {str(v[1]): v[EXP] for v in t0_digits["rows"].values()}
        it = [i for i in dv if i in ref]
        res["spearman_T0_digits_exp_rating_vs_prior_item_mean"] = _corr_ci(
            [dv[i] for i in it], [_num(ref[i]["probe_ref_prior_mean"]) for i in it], it, n_boot, seed)
    if base is not None and t2_rows:
        pm = _prior_map(t2_rows)
        keys = [k for k in base["rows"] if k in pm and math.isfinite(base["rows"][k][LOGIT])]
        L = np.array([base["rows"][k][LOGIT] for k in keys], float)
        P = np.array([pm[k][0] for k in keys], float)
        users = [base["rows"][k][0] for k in keys]
        res["like_logit_rho_pairs"] = _corr_ci(L, P, users, n_boot, seed)
        acc: dict = defaultdict(list)
        for k, v in zip(keys, L):
            acc[str(base["rows"][k][1])].append(v)
        it = [i for i in acc if i in ref]
        res["like_logit_rho_items"] = _corr_ci([np.mean(acc[i]) for i in it],
                                               [_num(ref[i]["probe_ref_prior_mean"]) for i in it], it, n_boot, seed)
    rho, t = res["spearman_T0_pyes_vs_prior_item_mean"]["est"], THRESHOLDS["T0"]
    like = (res.get("like_logit_rho_pairs") or {}).get("est", NAN)
    res["reading"] = (_reading("undetermined", "T0 Spearman of P(Yes) not computable (lp_yes absent or no finite "
                               "pairs)") if not math.isfinite(rho) else
                      _reading("knowledge_exists", f"T0 rho >= {t['knowledge_exists_rho_ge']}: item-quality knowledge "
                               f"exists; the registered reading 'exists but is not used' also needs a like-logit rho "
                               f"of about {t['companion_like_logit_rho_about']} (observed {like:.3f}; no numeric "
                               "tolerance registered)") if rho >= t["knowledge_exists_rho_ge"] else
                      _reading("knowledge_limited", f"T0 rho < {t['knowledge_limited_rho_lt']}: knowledge-limited")
                      if rho < t["knowledge_limited_rho_lt"] else
                      _reading("between_thresholds", "0.3 <= T0 rho < 0.5: no registered reading"))
    res["like_logit_rho_for_reading"] = like
    return res


DERIVED = {("ml1m", "t0"): "ml1m_t0", ("ml1m", "t0_digits"): "ml1m_t0", ("ml1m", "t1"): "ml1m_t1",
           ("ml1m", "t2"): "ml1m_t2", ("ml1m", "starperm"): "ml1m_starperm", ("toys", "t2"): "toys_t2"}
SAME_PANEL_AS_BASE = (("ml1m", "t3"), ("ml1m", "t4_llama"))
ARMS = (("ml1m", "base"), ("ml1m", "t0"), ("ml1m", "t0_digits"), ("ml1m", "t1"), ("ml1m", "t2"), ("ml1m", "t3"),
        ("ml1m", "starperm"), ("ml1m", "t4_llama"), ("toys", "base"), ("toys", "t2"))
# arm -> (variant, readout, backbone) as run_diag_battery.sh scores it; every arm also has ARM_COMMON and the
# variant's registered history window ARM_HIST
QWEN, LLAMA = "Qwen3-8B", "Llama-3.1-8B-Instruct"
ARM_SPEC = {"base": ("V0", "yesno", QWEN), "t0": ("T0_probe", "yesno", QWEN), "t0_digits": ("T0_probe", "digits", QWEN),
            "t1": ("V0", "yesno", QWEN), "t2": ("V0", "yesno", QWEN), "t3": ("V0", "digits", QWEN),
            "starperm": ("V0", "yesno", QWEN), "t4_llama": ("V0", "yesno", LLAMA)}
ARM_COMMON = {"questions": ["like"], "lora": None, "dtype": "float16", "topk_logprobs": 50, "max_model_len": 4096,
              "panel_kind": "rated"}
ARM_HIST = {"V0": V0_HIST, "T0_probe": 0}


def _alnum(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def arm_settings_errors(name: str, rep: dict) -> list[str]:
    """Scorer settings of one arm's report.json against ARM_SPEC / ARM_COMMON / ARM_HIST (backbone: the report's
    backbone, else the model directory name, must contain the registered name, case and punctuation ignored)."""
    variant, readout, backbone = ARM_SPEC[name]
    cfg = rep.get("config") or {}

    def got(k, default=None):
        return rep[k] if k in rep else cfg.get(k, default)

    err = []
    if got("variant") != variant:
        err.append(f"variant {got('variant')!r} != {variant!r}")
    if got("readout", "yesno") != readout:
        err.append(f"readout {got('readout', 'yesno')!r} != {readout!r}")
    for k, v in ARM_COMMON.items():
        if got(k) != v:
            err.append(f"{k} {got(k)!r} != {v!r}")
    if got("hist_len") != ARM_HIST[variant]:
        err.append(f"hist_len {got('hist_len')!r} != {ARM_HIST[variant]} (registered window of {variant})")
    bb = got("backbone") or Path(str(got("model") or "")).name
    if _alnum(backbone) not in _alnum(bb):
        err.append(f"backbone {bb!r} is not {backbone}")
    return err


def provenance(root) -> dict:
    """Per arm (ARMS): the scorer settings of its report.json (arm_settings_errors), and which panel bytes it scored
    (data_sha1): a battery arm must have scored exactly its battery panel, that panel must derive from the panel the
    base arm scored (.meta.json panel_sha1) with the same rows (meta n_rows_in == base report n_users), and T3/T4 must
    have scored the base panel. ok = False on any mismatch; None when a report or panel needed for the check is
    absent (nothing to check yet); analyze adds the prompts-vs-score-rows check."""
    root = Path(root)
    reps = {}
    for dom, name in ARMS:
        p = root / dom / name / "report.json"
        reps[(dom, name)] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def sha(dom, name):
        return (reps.get((dom, name)) or {}).get("data_sha1")

    out = {}
    for dom, name in ARMS:
        rep = reps[(dom, name)]
        rec: dict = {"report_present": rep is not None,
                     "settings_errors": arm_settings_errors(name, rep) if rep is not None else [],
                     "arm_data_sha1": sha(dom, name)}
        checks = []
        if (dom, name) in DERIVED:
            pn = DERIVED[(dom, name)]
            panel, meta = root / "panels" / f"{pn}.jsonl", root / "panels" / f"{pn}.meta.json"
            m = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
            rec.update(panel_sha1=file_sha1(panel) if panel.exists() else None, derived_from_sha1=m.get("panel_sha1"),
                       base_data_sha1=sha(dom, "base"), meta={k: v for k, v in m.items() if k != "panel_sha1"})
            checks = [(rec["arm_data_sha1"], rec["panel_sha1"]), (rec["derived_from_sha1"], rec["base_data_sha1"]),
                      (m.get("n_rows_in"), (reps.get((dom, "base")) or {}).get("n_users"))]
        elif (dom, name) in SAME_PANEL_AS_BASE:
            rec["base_data_sha1"] = sha(dom, "base")
            checks = [(rec["arm_data_sha1"], rec["base_data_sha1"]),
                      (rep.get("n_users") if rep else None, (reps.get((dom, "base")) or {}).get("n_users"))]
        rec["sha1_ok"] = None if any(a is None or b is None for a, b in checks) else all(a == b for a, b in checks)
        rec["ok"] = (False if rec["settings_errors"] or rec["sha1_ok"] is False else
                     None if rep is None or rec["sha1_ok"] is None else True)
        out[f"{dom}/{name}"] = rec
    return out


def _check_prompt_rows(prov: dict, key: str, info: dict) -> None:
    """report.json n_main_prompts must equal the arm's score rows (a stale or mixed output directory otherwise)."""
    rep, n = info.get("report"), info["counts"].get("n", 0)
    if rep is not None and _num(rep.get("n_main_prompts")) != n:
        rec = prov.setdefault(key, {"settings_errors": [], "ok": None})
        rec["settings_errors"].append(f"report.json n_main_prompts {rep.get('n_main_prompts')!r} != {n} score rows "
                                      "(stale or mixed output directory)")
        rec["ok"] = False


def analyze(root, n_boot: int = 2000, seed: int = 0) -> dict:
    root = Path(root)
    P = root / "panels"
    prov = provenance(root)
    res: dict = {"interpretive_only": True, "note": INTERPRETIVE, "thresholds": THRESHOLDS,
                 "operationalizations": list(OPERATIONALIZATIONS), "n_boot": n_boot, "seed": seed, "root": str(root),
                 "provenance_ok": None, "provenance": prov, "ml1m": {}, "toys": {}, "arms": {}}
    arms: dict = {}

    def arm(dom, name):
        if (dom, name) not in arms:
            d = root / dom / name
            arms[(dom, name)] = load_arm(d / "scores.csv.gz") if (d / "scores.csv.gz").exists() else None
            if arms[(dom, name)] is not None:
                res["arms"][f"{dom}/{name}"] = _arm_info(arms[(dom, name)], d)
        return arms[(dom, name)]

    def panel(name):
        p = P / f"{name}.jsonl"
        return read_jsonl(p) if p.exists() else None

    def need(dom, block, *paths):
        miss = [str(p) for p in paths if not Path(p).exists()]
        if miss:
            res[dom][block] = {"status": "missing", "missing": miss, "interpretive_only": True}
        return not miss

    def sc(dom, name):
        return root / dom / name / "scores.csv.gz"

    for dom in ("ml1m", "toys"):
        if need(dom, "base", sc(dom, "base")):
            res[dom]["base"] = {**block_base(arm(dom, "base"), n_boot, seed), "arm": "V0 like yes/no (Qwen3-8B)"}
        if need(dom, "T2", sc(dom, "t2"), P / f"{dom}_t2.jsonl"):
            res[dom]["T2"] = block_t2(arm(dom, "t2"), panel(f"{dom}_t2"), arm(dom, "base"), n_boot, seed)
    if need("ml1m", "T0", sc("ml1m", "t0"), P / "ml1m_t0.jsonl"):
        res["ml1m"]["T0"] = block_t0(arm("ml1m", "t0"), panel("ml1m_t0"), arm("ml1m", "base"), panel("ml1m_t2"),
                                     n_boot, seed, t0_digits=arm("ml1m", "t0_digits"))
    if need("ml1m", "T1", sc("ml1m", "t1")):
        res["ml1m"]["T1"] = block_t1(arm("ml1m", "t1"), arm("ml1m", "base"), n_boot, seed)
    if need("ml1m", "T3", sc("ml1m", "t3"), sc("ml1m", "base")):
        res["ml1m"]["T3"] = block_t3(arm("ml1m", "t3"), arm("ml1m", "base"), n_boot, seed)
    if need("ml1m", "star_permutation", sc("ml1m", "starperm"), sc("ml1m", "base")):
        res["ml1m"]["star_permutation"] = block_starperm(arm("ml1m", "base"), arm("ml1m", "starperm"),
                                                         panel("ml1m_starperm"), n_boot, seed)
    if need("ml1m", "T4", sc("ml1m", "t4_llama")):
        res["ml1m"]["T4"] = block_t4(arm("ml1m", "t4_llama"), arm("ml1m", "base"), n_boot, seed)
    for key, info in res["arms"].items():
        _check_prompt_rows(prov, key, info)
    res["provenance_ok"] = not any(v["ok"] is False for v in prov.values())
    res["readings"] = {f"{dom}/{b}": v["reading"]["code"] for dom in ("ml1m", "toys")
                       for b, v in res[dom].items() if isinstance(v, dict) and "reading" in v}
    return res


def t1check(root, n_boot: int = 2000, seed: int = 0) -> tuple[dict, int]:
    """The registered T1 stop rule, run right after the T1 arm and before any other battery arm: (result, exit code)
    with 0 = readout responds (UAUC >= 0.90), T1_BROKEN_EXIT = UAUC < 0.90 ("readout broken; stop and debug"), 2 = the
    T1 arm is missing, not computable, or did not score the T1 battery panel with the registered settings."""
    root = Path(root)
    d = root / "ml1m" / "t1"
    res: dict = {"interpretive_only": True, "note": INTERPRETIVE, "threshold": THRESHOLDS["T1"]}
    if not (d / "scores.csv.gz").exists():
        res["status"] = f"missing {d / 'scores.csv.gz'}"
        return res, 2
    pv = provenance(root)["ml1m/t1"]
    arm = load_arm(d / "scores.csv.gz")
    info = _arm_info(arm, d)
    prov = {"ml1m/t1": pv}
    _check_prompt_rows(prov, "ml1m/t1", info)
    # the base arm is not scored yet: only the T1-panel bytes and the settings can be checked here
    panel_ok = pv.get("arm_data_sha1") is not None and pv.get("arm_data_sha1") == pv.get("panel_sha1")
    res.update(arm=info, provenance=pv, T1=block_t1(arm, None, n_boot, seed))
    if pv["settings_errors"] or not panel_ok:
        res["status"] = ("the T1 arm did not score the T1 battery panel with the registered settings: "
                         f"{pv['settings_errors'] or 'data_sha1 != T1 panel sha1'}")
        return res, 2
    code = res["T1"]["reading"]["code"]
    res["status"] = code
    return res, (0 if code == "readout_responds" else T1_BROKEN_EXIT if code == "readout_broken_stop_and_debug"
                 else 2)


# ---------------------------------------------------------------- freeze (amendment 2 freeze rule)
def prompt_bank(panels: list[tuple[str, list[dict]]], variants, render, question: str = "like",
                n_rows: int = 50) -> dict:
    """sha1 of the rendered prompt bank (see module docstring) and one sha1 per variant over the same prompts."""
    h, per, n = hashlib.sha1(), {v: hashlib.sha1() for v in variants}, 0
    for name, rows in panels:
        for v in variants:
            for rec in rows[:n_rows]:
                sid = str(rec.get("source_event_id", rec["user_id"]))
                for i in range(len(rec["candidate_item_ids"])):
                    system, user = render(rec, i, question, variant=v)
                    p = f"{system}\x1d{user}" if system else user
                    b = f"{name}\x1f{v}\x1f{sid}\x1f{i}\x1f{p}".encode("utf-8") + b"\x1e"
                    h.update(b)
                    per[v].update(b)
                    n += 1
    return {"sha1": h.hexdigest(), "per_variant": {v: x.hexdigest() for v, x in per.items()}, "n_prompts": n}


# fixed (prior mean, n) values the freeze renders the T2 template with (the real means need the raw ratings)
T2_EXAMPLES = ((NAN, 0), (1.0, 1), (3.456, 12), (4.995, 250))
BATTERY_ARMS = (("T0", "T0_probe", "yesno"), ("T0_digits", "T0_probe", "digits"), ("T1", "V0", "yesno"),
                ("T2", "V0", "yesno"), ("T3", "V0", "digits"), ("starperm", "V0", "yesno"))


def battery_spec() -> dict:
    """Every fixed string and rule of the battery that its rendered prompts alone do not pin down."""
    return {"operationalizations": list(OPERATIONALIZATIONS), "thresholds": THRESHOLDS, "t1_extra": T1_EXTRA,
            "t2_examples": [t2_extra(m, n) for m, n in T2_EXAMPLES],
            "starperm": {"k": STARPERM_K, "seed": STARPERM_SEED, "hist_len": V0_HIST, "max_redraws": MAX_REDRAWS},
            "arms": [list(a) for a in BATTERY_ARMS]}


def battery_bank(panels: list[tuple[str, list[dict]]], render, n_rows: int = 50) -> dict:
    """sha1 of the diagnosis battery's own prompts (section D), so they are frozen with the G1 bank before any battery
    GPU job: on the first n_rows rows of each panel, every candidate of the rows each arm scores, rendered with the
    question like as run_diag_battery.sh scores them (BATTERY_ARMS: T0 / T0_digits = T0_probe yes/no / digits on the
    make_t0 rows, T1 = V0 on the make_t1 rows, T2 = V0 with the T2 template filled from T2_EXAMPLES cycled over the
    candidates, T3 = V0 with the digit readout, starperm = V0 on the make_starperm rows), each hashed as
    "<panel>\\x1f<arm>\\x1f<source_event_id>\\x1f<cand_idx>\\x1f<prompt>\\x1e"; then the canonical JSON of
    battery_spec() (operationalizations, thresholds, extra templates, permutation scheme)."""
    h, per, n = hashlib.sha1(), {a: hashlib.sha1() for a, _, _ in BATTERY_ARMS}, 0
    for name, rows in panels:
        rows = rows[:n_rows]
        src = "ml1m" if rows and rows[0].get("source") == "ml1m" else "amazon"
        t2 = []
        for rec in rows:
            r = dict(rec)
            r["candidate_extras"] = [t2_extra(*T2_EXAMPLES[i % len(T2_EXAMPLES)])
                                     for i in range(len(rec["candidate_item_ids"]))]
            t2.append(r)
        t0 = make_t0(rows, PriorIndex({}), {}, src)
        arm_rows = {"T0": t0, "T0_digits": t0, "T1": make_t1(rows), "T2": t2, "T3": rows,
                    "starperm": make_starperm(rows, STARPERM_K, STARPERM_SEED, V0_HIST)[0]}
        for arm, variant, readout in BATTERY_ARMS:
            for rec in arm_rows[arm]:
                sid = str(rec.get("source_event_id", rec["user_id"]))
                for i in range(len(rec["candidate_item_ids"])):
                    system, user = render(rec, i, "like", variant=variant, readout=readout)
                    p = f"{system}\x1d{user}" if system else user
                    b = f"{name}\x1f{arm}\x1f{sid}\x1f{i}\x1f{p}".encode("utf-8") + b"\x1e"
                    h.update(b)
                    per[arm].update(b)
                    n += 1
    spec = json.dumps(strict_json(battery_spec()), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    h.update(spec.encode("utf-8"))
    return {"sha1": h.hexdigest(), "per_arm": {a: x.hexdigest() for a, x in per.items()}, "n_prompts": n,
            "spec_sha1": hashlib.sha1(spec.encode("utf-8")).hexdigest()}


def user_list_sha1s(manifest) -> dict:
    """{json path: sha1} of the user-id lists: the manifest's `freeze` block (build_confirm_panels.py writes the DEV
    and CONFIRM user-id list sha1s there) when it holds any; otherwise every 40-hex string whose path mentions
    'user' and 'sha1'/'hash' (case-insensitive). Null entries (a split that was not built) are skipped."""
    fz = manifest.get("freeze") if isinstance(manifest, dict) else None
    if isinstance(fz, dict):
        got = {f"freeze.{k}": v.lower() for k, v in fz.items() if isinstance(v, str) and _HEX40.match(v.lower())}
        if got:
            return got
    out: dict = {}

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(x, list):
            for j, v in enumerate(x):
                walk(v, f"{path}[{j}]")
        elif isinstance(x, str) and _HEX40.match(x.lower()) and re.search(r"user", path, re.I) \
                and re.search(r"sha1|hash", path, re.I):
            out[path] = x.lower()

    walk(manifest, "")
    return out


def freeze_text(amendment, dev_panels: list, manifest, variants=None, n_rows: int = 50, question: str = "like",
                render=None) -> tuple[str, list[str]]:
    """(FREEZE.txt text, sha1s that must be recorded in PILOT_LOG). Deterministic: no timestamps."""
    from src.confrec import prompting   # lazy: the bank is whatever prompting renders at freeze time
    render = render or prompting.render
    variants = list(variants or prompting.VARIANTS)
    panels = [(Path(p).name, read_jsonl(p)[:n_rows]) for p in dev_panels]
    bank = prompt_bank(panels, variants, render, question, n_rows)
    bat = battery_bank(panels, render, n_rows)
    strings_sha1 = getattr(prompting, "PROMPT_STRINGS_SHA1", None)
    users = user_list_sha1s(json.loads(Path(manifest).read_text(encoding="utf-8")))
    if not users:
        raise SystemExit(f"{manifest}: no user-id list sha1 found (keys mentioning 'user' and 'sha1')")
    am = file_sha1(amendment)
    lines = ["# Amendment-2 freeze (idea-stage/PREREG_AMENDMENT_2.md): record every REQUIRED sha1 in "
             "docs/sigir/PILOT_LOG.md before any GPU job, diagnostics included.",
             f"amendment_path = {amendment}",
             f"amendment_sha1 = {am}  REQUIRED",
             f"prompt_bank_sha1 = {bank['sha1']}  REQUIRED",
             f"prompt_bank_definition = variants={','.join(variants)}; question={question}; first {n_rows} rows of "
             f"each dev panel; every candidate; sha1 over '<panel basename>\\x1f<variant>\\x1f<source_event_id>"
             f"\\x1f<cand_idx>\\x1f<system\\x1duser | user>\\x1e' in panel, variant, row, candidate order",
             f"prompt_bank_panels = {','.join(str(p) for p in dev_panels)}",
             f"prompt_bank_n_prompts = {bank['n_prompts']}"]
    lines += [f"prompt_bank_sha1[{v}] = {s}" for v, s in bank["per_variant"].items()]
    if strings_sha1:
        lines.append(f"prompt_strings_sha1 = {strings_sha1}  (prompting.PROMPT_STRINGS_SHA1: every fixed string and "
                     "rule of the bank, incl. the T0 and digit wording)")
    lines += [f"battery_bank_sha1 = {bat['sha1']}  REQUIRED",
              f"battery_bank_definition = the diagnosis battery's prompts (diag_battery.battery_bank): arms "
              f"{','.join(a for a, _, _ in BATTERY_ARMS)}; question=like; first {n_rows} rows of each dev panel; every "
              "candidate; sha1 over '<panel basename>\\x1f<arm>\\x1f<source_event_id>\\x1f<cand_idx>\\x1f"
              "<system\\x1duser | user>\\x1e' in panel, arm, row, candidate order, then the canonical JSON of "
              "diag_battery.battery_spec() (operationalizations, thresholds, T1/T2 templates, permutation scheme)",
              f"battery_bank_n_prompts = {bat['n_prompts']}",
              f"battery_spec_sha1 = {bat['spec_sha1']}"]
    lines += [f"battery_bank_sha1[{a}] = {s}" for a, s in bat["per_arm"].items()]
    lines += [f"user_ids_sha1[{k}] = {v}  REQUIRED" for k, v in sorted(users.items())]
    lines += [f"panel_file_sha1[{p}] = {file_sha1(p)}" for p in dev_panels]
    return "\n".join(lines) + "\n", [am, bank["sha1"], bat["sha1"], *sorted(users.values())]


def freeze(amendment, dev_panels, manifest, out, check: bool = False, pilot_log=None, variants=None,
           n_rows: int = 50, question: str = "like", render=None) -> str:
    text, required = freeze_text(amendment, dev_panels, manifest, variants, n_rows, question, render)
    out = Path(out)
    if out.exists():
        old = out.read_text(encoding="utf-8")
        if old != text:
            diff = sorted(set(text.splitlines()) ^ set(old.splitlines()))
            raise SystemExit(f"{out} differs from the recomputed freeze (something hashed changed after the freeze; "
                             "investigate, never just delete the file):\n" + "\n".join(diff))
    elif check:
        raise SystemExit(f"{out} does not exist: run the freeze (run_gatefix.sh stage 0) and record it first")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, out)
    if check:
        if pilot_log is None or not Path(pilot_log).exists():
            raise SystemExit(f"--check needs an existing --pilot_log (got {pilot_log})")
        log = Path(pilot_log).read_text(encoding="utf-8").lower()
        missing = [s for s in required if s not in log]
        if missing:
            raise SystemExit(f"freeze rule: {pilot_log} lacks the required sha1(s) {missing}; record them first")
    return text


# ---------------------------------------------------------------- vstar
def selected_variant(sel: dict) -> str:
    """V* from gatefix_select's selection.json (top level or one level down); refuses V0, fix_found = false and a
    top-level decision / outcome other than FIX_FOUND."""
    for k in ("decision", "outcome"):
        if isinstance(sel.get(k), str) and sel[k] != "FIX_FOUND":
            raise SystemExit(f"selection {k} = {sel[k]!r}, not FIX_FOUND: Stage 2 must not run")
    found = {}
    for path, d in [("", sel)] + [(f"{k}.", v) for k, v in sel.items() if isinstance(v, dict)]:
        for k in VSTAR_KEYS:
            if isinstance(d.get(k), str):
                found[path + k] = d[k]
        for k in ("fix_found", "FIX_FOUND", "fix_is_found"):
            if k in d and not d[k]:
                raise SystemExit(f"selection says {path}{k} = {d[k]!r}: no fix found, Stage 2 must not run")
    vals = set(found.values())
    if len(vals) != 1:
        raise SystemExit(f"cannot read V* from the selection (candidates {found or 'none'}; keys tried {VSTAR_KEYS})")
    v = vals.pop()
    if v not in GATE_VARIANTS:
        raise SystemExit(f"V* {v!r} is not a registered gate variant {GATE_VARIANTS}")
    if v == "V0":
        raise SystemExit("V* is V0: no fix (G5), Stage 2 must not run")
    return v


# ---------------------------------------------------------------- cli
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Amendment-2 diagnosis battery (interpretive only) and freeze helpers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--kind", choices=KINDS, required=True)
    m.add_argument("--panel", required=True)
    m.add_argument("--raw", default=None, help="ML-1M dir (ratings.dat) or the Amazon root holding amazon_<domain>/")
    m.add_argument("--source", choices=["ml1m", "amazon"], required=True)
    m.add_argument("--domain", default=None)
    m.add_argument("--out", required=True)
    m.add_argument("--n_users", type=int, default=None, help="first N panel rows only")
    m.add_argument("--k", type=int, default=STARPERM_K, help="starperm copies (registered K = 2)")
    m.add_argument("--seed", type=int, default=STARPERM_SEED)
    m.add_argument("--hist_len", type=int, default=V0_HIST, help="starperm: rendered history window (V0: 10)")
    a = sub.add_parser("analyze")
    a.add_argument("--root", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--n_boot", type=int, default=2000)
    a.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("t1check", help="T1 stop rule: exit 0 if T1 UAUC >= 0.90, 5 if < 0.90, 2 on input problems")
    t.add_argument("--root", required=True)
    t.add_argument("--out", default=None, help="JSON record (default ROOT/ml1m/t1check.json)")
    t.add_argument("--n_boot", type=int, default=2000)
    t.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("freeze")
    f.add_argument("--amendment", required=True)
    f.add_argument("--dev_panels", required=True, help="comma list")
    f.add_argument("--manifest", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--variants", default=None, help="comma list (default: every prompting.VARIANTS key)")
    f.add_argument("--n_rows", type=int, default=50)
    f.add_argument("--question", default="like")
    f.add_argument("--check", action="store_true")
    f.add_argument("--pilot_log", default=None)
    v = sub.add_parser("vstar")
    v.add_argument("--selection", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "make":
        meta = make(args.kind, args.panel, args.out, args.raw, args.source, args.domain, args.n_users, args.k,
                    args.seed, args.hist_len)
        print(json.dumps(strict_json(meta)))
    elif args.cmd == "analyze":
        res = analyze(args.root, args.n_boot, args.seed)
        write_json(args.out, res)
        print(json.dumps(strict_json(res["readings"]), indent=1))
        if not res["provenance_ok"]:
            bad = [k for k, v in res["provenance"].items() if v["ok"] is False]
            print(f"WARNING: provenance mismatch for {bad}: an arm scored other panel bytes than its battery/base "
                  "panel (see 'provenance' in the output)")
        print(INTERPRETIVE)
    elif args.cmd == "t1check":
        res, code = t1check(args.root, args.n_boot, args.seed)
        write_json(args.out or Path(args.root) / "ml1m" / "t1check.json", res)
        t1 = res.get("T1") or {}
        print(json.dumps(strict_json({"status": res["status"], "UAUC": t1.get("UAUC"), "lo": t1.get("lo"),
                                      "hi": t1.get("hi"), "n_users": t1.get("n_users"), "exit": code})))
        if code == T1_BROKEN_EXIT:
            print(f"T1 UAUC < {THRESHOLDS['T1']['readout_broken_uauc_lt']}: readout broken; stop and debug "
                  "(amendment 2 section D)")
        raise SystemExit(code)
    elif args.cmd == "freeze":
        text = freeze(args.amendment, [p for p in args.dev_panels.split(",") if p], args.manifest, args.out,
                      args.check, args.pilot_log, [x for x in args.variants.split(",") if x] if args.variants
                      else None, args.n_rows, args.question)
        print(text, end="")
        if args.check:
            print(f"freeze check OK: {args.out} unchanged and every required sha1 is in {args.pilot_log}")
    else:
        print(selected_variant(json.loads(Path(args.selection).read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
