"""Pilot 3 (§3.3 pseudonym knockout) analysis. Registered: idea-stage/triage_verdict.md §3.3; binding
operationalization: idea-stage/PREREG_AMENDMENT_1.md P3 and C1 (every CI resamples users, or items for item-level
correlations).

    python -m src.confrec.pilot_pseudonym --real <dir_real> --pseudo <dir_pseudo> --placebo <dir_placebo> \
        --panel outputs/confrec/panels/toys_rated.jsonl --out <dir>/pilot_pseudonym.json [--n_boot 2000] \
        [--bi_boot 1000] [--seed 0] [--gate outputs/confrec/pilot1_mirror/decision.json]

Each DIR holds pyes_scorer output: scores.csv.gz (report.json = completion marker) and optionally swap_prior.csv.gz
(a scores.csv.gz path is accepted too). L = log P(Yes) - log P(No) of the *like* question. Rows with a non-finite
logit (censored 2/3) are dropped and counted per arm, question and code; censored=1 rows (a bound) are kept. The
analysis set is the (source_event_id, cand_idx) rows finite in all three arms whose item matches the REAL panel.
  * treated = candidate_brand_in_title: from the panel rows if present, else from <panel stem>_pseudo.jsonl, else
    recomputed with pseudonymize's full stop-list, regex and treated rule (cross-checked when both exist);
    popularity = candidate_popularity (all-time category events);
  * Δ_arm = L_real - L_arm on treated rows; head/tail = stats.rank_bins(pop, 5) bins 4/0 among treated;
    head-minus-tail, mean Δ and slope = OLS of Δ on log1p(pop), each with a user-cluster bootstrap CI; per-decile
    means (rank_bins(pop, 10)) with bin sizes;
  * UAUC per arm over ALL candidates (primary), treated, tail (rank_bins(pop, 5) == 0 over all candidates) and the
    treated tail; paired user-bootstrap ΔUAUC real - pseudo and real - placebo on identical rows;
  * Bias Index (metrics.bias_index_ci, residual-adjusted; --bi_boot resamples) per arm: conf = sigmoid(L),
    correct = label, group = head/mid/tail by rank_bins(pop, 5), user clusters; over all candidates and treated;
  * cross-check (needs dislike rows and swap_prior.csv.gz in real and pseudo): Spearman corr(a, log-pop) over pairs
    (a = (L_like + L_dislike)/2 centred within user; user-cluster CI) and corr(pi, log-pop) over items (pi = mean
    donor like-logit; item bootstrap), per arm and real - pseudo; repeated on candidate_popularity_prior
    (popularity_prior_robustness) as in pilot_mirror.
Decision (point estimates; CIs reported; evaluated in this order, see decide()): NULL, POSITIVE, NEGATIVE, else
AMBIGUOUS; UNDETERMINED (listing the missing inputs) when a rule reached in that order has a non-finite input,
e.g. an empty tail bin. Scope: only `store` strings are substituted; the Δ population is store-in-title candidates
(share reported).
Token-channel gate (triage_verdict.md §3, common to pilots 1 and 3): --gate reads pilot1_gate's decision.json;
gate.pass false -> verdict GATE_FAIL_UNINTERPRETABLE (the rule's verdict kept as verdict_if_gate_passed); a missing
or unreadable gate leaves decision.token_channel_gate null (verdict provisional).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.confrec.metrics import auroc, bias_index_ci
from src.confrec.stats import (cluster_bootstrap, ols_slope, paired_bootstrap, rank_bins, sigmoid, spearman,
                               strict_json)

NAN = float("nan")
ARMS = ("real", "pseudo", "placebo")
TH = {"null_dUAUC": 0.02, "positive_hmt": 0.20, "placebo_band": 0.05, "tail_dUAUC": 0.01, "negative_gap": 0.05}
SCOPE = ("Only the `store` field is substituted (brand tokens in titles beyond it are not implemented); the Δ "
         "analyses cover store-in-title candidates, whose share of all candidates is treated.share.")


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def _open(path):
    path = str(path)
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.endswith(".gz") else \
        open(path, encoding="utf-8", newline="")


def _nan_ci() -> dict:
    return {"est": NAN, "lo": NAN, "hi": NAN, "n_clusters": 0}


def load_arm(path) -> dict:
    """pyes_scorer DIR (or its scores.csv.gz) -> {L: {question: {key: logit}}, meta: {key: (user, item, label)},
    censoring, pi: {question: {item: mean donor logit}} | None, complete}. key = (source_event_id, cand_idx)."""
    p = Path(path)
    d, sp = (p, p / "scores.csv.gz") if p.is_dir() else (p.parent, p)
    L, meta, cens, seen = defaultdict(dict), {}, defaultdict(Counter), set()
    with _open(sp) as f:
        for r in csv.DictReader(f):
            q, k = r["question"], (str(r["source_event_id"]), int(r["cand_idx"]))
            code = (r.get("censored") or "na").strip()
            cens[q]["n"] += 1
            cens[q][code] += 1
            if (q, k) in seen:  # the last row wins
                cens[q]["duplicate"] += 1
            seen.add((q, k))
            meta[k] = (str(r["user_id"]), str(r["item_id"]), int(float(r["label"])))
            lg = _num(r["logit"])
            if np.isfinite(lg) and code not in ("2", "3"):
                L[q][k] = lg
            else:
                L[q].pop(k, None)
                cens[q]["dropped_nonfinite"] += 1
    pi, swap_cens = None, None
    if (d / "swap_prior.csv.gz").exists():
        acc, swap_cens = defaultdict(lambda: defaultdict(list)), defaultdict(Counter)
        with _open(d / "swap_prior.csv.gz") as f:
            for r in csv.DictReader(f):
                q, code, lg = r.get("question") or "like", (r.get("censored") or "na").strip(), _num(r["logit"])
                swap_cens[q]["n"] += 1
                swap_cens[q][code] += 1
                if np.isfinite(lg) and code not in ("2", "3"):
                    acc[q][str(r["item_id"])].append(lg)
                else:
                    swap_cens[q]["dropped_nonfinite"] += 1
        pi = {q: {i: float(np.mean(v)) for i, v in dd.items()} for q, dd in acc.items()}
        swap_cens = {q: dict(c) for q, c in swap_cens.items()}
    return {"L": dict(L), "meta": meta, "censoring": {q: dict(c) for q, c in sorted(cens.items())}, "pi": pi,
            "swap_censoring": swap_cens, "complete": (d / "report.json").exists(), "dir": str(d)}


def _groups(x) -> dict:
    out = defaultdict(list)
    for j, v in enumerate(x):
        out[v].append(j)
    return {k: np.array(v, int) for k, v in out.items()}


def uauc(score, label, users: dict, mask) -> dict:
    """Per-user AUC over rows in `mask`, for users with both classes there."""
    out = {}
    for u, idx in users.items():
        idx = idx[mask[idx]]
        y = label[idx]
        if 0 < y.sum() < len(y):
            out[u] = auroc(score[idx], y)
    return out


def head_minus_tail(d, head, tail, clusters, n_boot: int = 2000, seed: int = 0) -> dict:
    """mean(d[head]) - mean(d[tail]) with a cluster (user) bootstrap CI; head/tail membership is fixed."""
    d, head, tail = np.asarray(d, float), np.asarray(head, bool), np.asarray(tail, bool)
    if not head.any() or not tail.any():
        return {**_nan_ci(), "n_head": int(head.sum()), "n_tail": int(tail.sum())}

    def stat(idx):
        h, t = idx[head[idx]], idx[tail[idx]]
        return float(d[h].mean() - d[t].mean()) if len(h) and len(t) else NAN
    out = cluster_bootstrap(stat, clusters, n_boot, seed)
    out.update(n_head=int(head.sum()), n_tail=int(tail.sum()), mean_head=float(d[head].mean()),
               mean_tail=float(d[tail].mean()))
    return out


def _boot(stat, clusters, n_boot, seed) -> dict:
    return cluster_bootstrap(stat, clusters, n_boot, seed) if len(clusters) else _nan_ci()


def _fin(x) -> bool:
    return x is not None and bool(np.isfinite(x))


def _tri(ok, *inputs):
    """Condition value: None (undetermined) when an input is non-finite, else bool."""
    return bool(ok) if all(_fin(x) for x in inputs) else None


def _and3(vals) -> bool | None:
    """Kleene AND: False if any condition is False, None if none is False but one is undetermined, else True."""
    vals = list(vals)
    return False if any(v is False for v in vals) else None if any(v is None for v in vals) else True


def decide(dU_all: float, dU_tail: float, hmt_ps: dict, hmt_pl: dict, slope_ps: dict) -> dict:
    """Pre-registered §3.3 rule on point estimates (amendment 1 P3), evaluated in order NULL, POSITIVE, NEGATIVE,
    else AMBIGUOUS. dU_* = ΔUAUC(real - pseudo) over all candidates / tail candidates. A condition with a
    non-finite input (e.g. an empty tail bin) is undetermined; the first rule that is not False decides: its name
    if True, UNDETERMINED if it cannot be settled (the missing inputs are listed). Missing data is never AMBIGUOUS."""
    dU_all, dU_tail = _num(dU_all), _num(dU_tail)
    e_ps, lo_ps, e_pl = _num(hmt_ps["est"]), _num(hmt_ps["lo"]), _num(hmt_pl["est"])
    lo_s, hi_s = _num(slope_ps["lo"]), _num(slope_ps["hi"])
    c = {
        "null_dUAUC_all_gt": _tri(dU_all > TH["null_dUAUC"], dU_all),
        "null_dUAUC_tail_gt": _tri(dU_tail > TH["null_dUAUC"], dU_tail),
        "pos_hmt_pseudo_ge": _tri(e_ps >= TH["positive_hmt"], e_ps),
        "pos_hmt_pseudo_lo_gt0": _tri(lo_ps > 0, lo_ps),
        "pos_hmt_placebo_within": _tri(abs(e_pl) <= TH["placebo_band"], e_pl),
        "pos_dUAUC_tail_within": _tri(abs(dU_tail) <= TH["tail_dUAUC"], dU_tail),
        "neg_slope_ci_contains0": _tri(lo_s <= 0 <= hi_s, lo_s, hi_s),
        "neg_hmt_gap_within": _tri(abs(e_ps - e_pl) <= TH["negative_gap"], e_ps, e_pl),
    }
    rules = {"NULL": _and3(v for k, v in c.items() if k.startswith("null_")),
             "POSITIVE": _and3(v for k, v in c.items() if k.startswith("pos_")),
             "NEGATIVE": _and3(v for k, v in c.items() if k.startswith("neg_"))}
    verdict = "AMBIGUOUS"
    for name, v in rules.items():
        if v is None:
            verdict = "UNDETERMINED"
            break
        if v:
            verdict = name
            break
    inputs = {"dUAUC_real_minus_pseudo_all": dU_all, "dUAUC_real_minus_pseudo_tail": dU_tail,
              "hmt_pseudo.est": e_ps, "hmt_pseudo.lo": lo_ps, "hmt_placebo.est": e_pl,
              "slope_pseudo.lo": lo_s, "slope_pseudo.hi": hi_s}
    return {"verdict": verdict, "order": ["NULL", "POSITIVE", "NEGATIVE", "AMBIGUOUS"], "conditions": c,
            "rules": rules, "missing_inputs": [k for k, v in inputs.items() if not _fin(v)],
            "values": {"dUAUC_real_minus_pseudo_all": dU_all, "dUAUC_real_minus_pseudo_tail": dU_tail,
                       "hmt_pseudo": {k: hmt_ps[k] for k in ("est", "lo", "hi")},
                       "hmt_placebo": {k: hmt_pl[k] for k in ("est", "lo", "hi")},
                       "hmt_pseudo_minus_placebo_est": e_ps - e_pl if _fin(e_ps) and _fin(e_pl) else NAN,
                       "slope_pseudo": {k: slope_ps[k] for k in ("est", "lo", "hi")},
                       "n_head_treated": hmt_ps.get("n_head"), "n_tail_treated": hmt_ps.get("n_tail")},
            "thresholds": TH,
            "rule": "NULL iff dUAUC(real-pseudo) > 0.02 both overall and tail; POSITIVE iff hmt_pseudo >= 0.20 and "
                    "its CI lo > 0 and |hmt_placebo| <= 0.05 and |dUAUC_tail(real-pseudo)| <= 0.01; NEGATIVE iff "
                    "the pseudo slope CI contains 0 and |hmt_pseudo - hmt_placebo| <= 0.05; else AMBIGUOUS. "
                    "UNDETERMINED (not a registered outcome): a rule reached in this order has a non-finite input "
                    "and no False condition"}


def treated_from_panel(panel_path, rows: list) -> tuple[dict, str, dict]:
    """{source_event_id: [bool per candidate]}, source, cross-check counts."""
    from src.confrec.pseudonymize import panel_brands, treated_flags
    if rows and all("candidate_brand_in_title" in r for r in rows):
        return {str(r["source_event_id"]): list(map(bool, r["candidate_brand_in_title"])) for r in rows}, "panel", {}
    idx = panel_brands(rows)  # same stop-list (static + common_word), regex and treated rule as pseudonymize
    keys = set(idx["keys"])
    rec = {str(r["source_event_id"]): treated_flags(r, keys, idx["rx"]) for r in rows}
    sib = Path(panel_path).with_name(Path(panel_path).stem + "_pseudo.jsonl")
    if not sib.exists():
        return rec, "recomputed", {}
    got = {}
    for line in open(sib, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            got[str(r["source_event_id"])] = list(map(bool, r["candidate_brand_in_title"]))
    mism = sum(a != b for ev, f in got.items() for a, b in zip(f, rec.get(ev, [])))
    return got, "pseudo_panel", {"pseudo_panel": str(sib), "n_flag_mismatch_vs_recomputed": int(mism),
                                 "n_events_missing_in_pseudo_panel": int(len(set(rec) - set(got)))}


def analyze(arms: dict, panel_rows: list, flags: dict, n_boot: int = 2000, seed: int = 0,
            bi_boot: int | None = None) -> dict:
    """bi_boot: bootstrap size of the Bias Index CIs (default n_boot; metrics.bias_index is the slow step)."""
    boot = dict(n_boot=n_boot, seed=seed)
    bi_boot = n_boot if bi_boot is None else bi_boot
    like = {a: arms[a]["L"].get("like", {}) for a in ARMS}
    panel = {str(r["source_event_id"]): r for r in panel_rows}
    common = sorted(set(like["real"]) & set(like["pseudo"]) & set(like["placebo"]))
    join, keys = Counter(), []
    for k in common:
        ev, c = k
        r, m = panel.get(ev), arms["real"]["meta"][k]
        if r is None:
            join["rows_without_panel_event"] += 1
            continue
        ids = r["candidate_item_ids"]
        if c >= len(ids) or str(ids[c]) != m[1]:
            join["rows_item_mismatch"] += 1
            continue
        if any(arms[a]["meta"][k] != m for a in ("pseudo", "placebo")):
            join["rows_meta_mismatch_across_arms"] += 1
            continue
        if "candidate_labels" in r and int(r["candidate_labels"][c]) != m[2]:
            join["rows_label_mismatch_vs_panel"] += 1
        keys.append(k)
    n = len(keys)
    user = np.array([arms["real"]["meta"][k][0] for k in keys], dtype=object)
    item = np.array([arms["real"]["meta"][k][1] for k in keys], dtype=object)
    label = np.array([arms["real"]["meta"][k][2] for k in keys], int)
    pop = np.array([_num(panel[ev]["candidate_popularity"][c]) for ev, c in keys], float)
    has_prior = any("candidate_popularity_prior" in r for r in panel_rows)
    pop_prior = np.array([_num(panel[ev]["candidate_popularity_prior"][c])
                          if "candidate_popularity_prior" in panel[ev] else NAN for ev, c in keys], float)
    treated =np.array([bool((flags.get(ev) or [])[c]) if c < len(flags.get(ev) or []) else False
                        for ev, c in keys], bool)
    L = {a: np.array([like[a][k] for k in keys], float) for a in ARMS}
    users = _groups(user)
    res = {"n_boot": n_boot, "bi_boot": bi_boot, "seed": seed, "scope_note": SCOPE,
           "censoring": {a: arms[a]["censoring"] for a in ARMS},
           "complete": {a: arms[a]["complete"] for a in ARMS},
           "rows": {"n_like_finite": {a: len(like[a]) for a in ARMS}, "n_common_finite": len(common),
                    "n_dropped_not_finite_in_all_arms": len(set().union(*(arms[a]["meta"] for a in ARMS)))
                    - len(common),
                    "panel_join": dict(join), "n_analysed": n, "n_users": len(users),
                    "n_nonfinite_popularity": int((~np.isfinite(pop)).sum())},
           "treated": {"n_treated": int(treated.sum()), "n_candidates": n,
                       "share": float(treated.mean()) if n else NAN,
                       "n_users_with_treated": int(len(set(user[treated])))}}

    # --- Δ logit on treated candidates ---
    T = np.flatnonzero(treated & np.isfinite(pop))
    u_t, pop_t, lp_t = user[T], pop[T], np.log1p(pop[T])
    b5_t, b10_t = rank_bins(pop_t, 5), rank_bins(pop_t, 10)
    head, tail = b5_t == 4, b5_t == 0
    d = {a: (L["real"] - L[a])[T] for a in ("pseudo", "placebo")}
    d["pseudo_minus_placebo"] = d["pseudo"] - d["placebo"]  # = L_placebo - L_pseudo
    delta = {"definition": "delta_arm = L_real - L_arm on treated rows; head/tail = rank_bins(pop, 5) bins 4/0 "
                           "among treated; CIs resample users", "n_treated_rows": int(len(T))}
    for name, x in d.items():
        delta[name] = {
            "mean": _boot(lambda idx, x=x: float(x[idx].mean()) if len(idx) else NAN, u_t, **boot),
            "head_minus_tail": head_minus_tail(x, head, tail, u_t, **boot),
            "slope_logpop": _boot(lambda idx, x=x: ols_slope(lp_t[idx], x[idx]), u_t, **boot)}
    delta["by_decile"] = [{"bin": b, "n": int(m.sum()), "n_users": int(len(set(u_t[m]))),
                           "pop_min": float(pop_t[m].min()) if m.any() else NAN,
                           "pop_max": float(pop_t[m].max()) if m.any() else NAN,
                           "delta_pseudo": float(d["pseudo"][m].mean()) if m.any() else NAN,
                           "delta_placebo": float(d["placebo"][m].mean()) if m.any() else NAN}
                          for b in range(10) for m in [b10_t == b]]
    res["delta"] = delta

    # --- UAUC: all candidates (primary), treated, tail, treated tail ---
    fin = np.isfinite(pop)
    b5 = np.full(n, -1)
    b5[fin] = rank_bins(pop[fin], 5)
    tail_t = np.zeros(n, bool)
    tail_t[T[tail]] = True
    masks = {"all": np.ones(n, bool), "treated": treated, "tail": b5 == 0, "tail_treated": tail_t}
    pu = {a: {m: uauc(L[a], label, users, mk) for m, mk in masks.items()} for a in ARMS}
    res["uauc"] = {a: {m: {"UAUC": float(np.mean(list(v.values()))) if v else NAN, "n_users": len(v),
                           "n_rows": int(masks[m].sum())} for m, v in pu[a].items()} for a in ARMS}
    res["uauc"]["tail_definition"] = "tail = rank_bins(candidate_popularity, 5) == 0 over all analysed candidates"
    res["dUAUC"] = {f"real_minus_{a}": {m: paired_bootstrap(pu["real"][m], pu[a][m], **boot) for m in masks}
                    for a in ("pseudo", "placebo")}

    # --- confidence-matched Bias Index per arm ---
    grp = np.where(b5 == 4, "head", np.where(b5 == 0, "tail", "mid"))
    grp_t = np.where(b5_t == 4, "head", np.where(b5_t == 0, "tail", "mid"))
    bi_kw = dict(n_bins=10, adjust=True, n_boot=bi_boot, seed=seed)
    res["bias_index"] = {"definition": "metrics.bias_index_ci(sigmoid(L_like), label, head/mid/tail by "
                                       "rank_bins(pop, 5), user clusters), residual-adjusted",
                         **{a: bias_index_ci(sigmoid(L[a][fin]), label[fin], grp[fin], user[fin], **bi_kw)
                            if fin.any() else {} for a in ARMS}}
    res["bias_index_treated"] = {a: bias_index_ci(sigmoid(L[a][T]), label[T], grp_t, u_t, **bi_kw)
                                 if len(T) else {} for a in ARMS}

    res["cross_check"] = cross_check(arms, keys, user, item, pop, treated, boot, pop_prior if has_prior else None)
    res["decision"] = decide(res["dUAUC"]["real_minus_pseudo"]["all"]["est"],
                             res["dUAUC"]["real_minus_pseudo"]["tail"]["est"],
                             delta["pseudo"]["head_minus_tail"], delta["placebo"]["head_minus_tail"],
                             delta["pseudo"]["slope_logpop"])
    res["decision"]["values"].update(n_analysed=n, n_rows_tail_all_candidates=int(masks["tail"].sum()),
                                     n_users_dUAUC_tail=res["dUAUC"]["real_minus_pseudo"]["tail"]["n"])
    return res


def cross_check(arms, keys, user, item, pop, treated, boot, pop_prior=None) -> dict:
    """§3.3 cross-check with pilot 1: corr(a, log-pop) over pairs and corr(pi, log-pop) over items, real vs pseudo,
    on log1p(candidate_popularity); repeated on log1p(candidate_popularity_prior) when the panel has it
    (popularity_prior_robustness, as in pilot_mirror)."""
    out = _corr_blocks(arms, keys, user, item, np.log1p(pop), treated, boot, "candidate_popularity")
    if pop_prior is not None and np.isfinite(pop_prior).any():
        out["popularity_prior_robustness"] = {
            "popularity": "log1p(candidate_popularity_prior): events on the item strictly before the candidate",
            **_corr_blocks(arms, keys, user, item, np.log1p(pop_prior), treated, boot, "candidate_popularity_prior")}
    else:
        out["popularity_prior_robustness"] = {"available": False, "reason": "panel has no candidate_popularity_prior"}
    return out


def _corr_blocks(arms, keys, user, item, lp_row, treated, boot, pop_name: str) -> dict:
    out = {}
    R, P = arms["real"], arms["pseudo"]
    if all("dislike" in A["L"] for A in (R, P)):
        ok = np.array([all(k in A["L"][q] for A in (R, P) for q in ("like", "dislike")) for k in keys], bool)
        idx = np.flatnonzero(ok & np.isfinite(lp_row))
        a = {}
        for name, A in (("real", R), ("pseudo", P)):
            v = np.array([(A["L"]["like"][keys[j]] + A["L"]["dislike"][keys[j]]) / 2 for j in idx], float)
            for g in _groups(user[idx]).values():
                v[g] = v[g] - v[g].mean() if len(g) >= 2 else NAN  # one row: centring leaves a constant 0
            a[name] = v
        lp, us, tr = lp_row[idx], user[idx], treated[idx]
        blk = {"definition": "a = (L_like + L_dislike)/2 centred within user over rows finite in both arms (users "
                             f"with >= 2 such rows); Spearman vs log1p({pop_name}); user-cluster CI",
               "n_pairs": int(np.isfinite(a["real"]).sum()) if len(idx) else 0}
        for sub, m in (("all", np.ones(len(idx), bool)), ("treated", tr)):
            s = np.flatnonzero(m)
            blk[sub] = {name: _boot(lambda j, v=v: spearman(v[s][j], lp[s][j]), us[s], **boot)
                        for name, v in a.items()}
            blk[sub]["real_minus_pseudo"] = _boot(
                lambda j: spearman(a["real"][s][j], lp[s][j]) - spearman(a["pseudo"][s][j], lp[s][j]), us[s], **boot)
        out["corr_a_logpop"] = blk
    else:
        out["corr_a_logpop"] = {"available": False, "reason": "dislike not scored in both real and pseudo"}
    if all(A["pi"] and "like" in A["pi"] for A in (R, P)):
        acc = defaultdict(list)
        for it, v in zip(item, lp_row):
            if np.isfinite(v):
                acc[it].append(v)
        its = sorted(i for i in acc if i in R["pi"]["like"] and i in P["pi"]["like"])
        lp = np.array([np.mean(acc[i]) for i in its], float)
        pi = {name: np.array([A["pi"]["like"][i] for i in its], float) for name, A in (("real", R), ("pseudo", P))}
        cl = np.arange(len(its))
        out["corr_pi_logpop"] = {
            "definition": "pi(i) = mean finite donor like-logit (swap_prior.csv.gz); Spearman vs the item mean of "
                          f"log1p({pop_name}) over items with pi in both arms; item bootstrap", "n_items": len(its),
            **{name: _boot(lambda j, v=v: spearman(v[j], lp[j]), cl, **boot) for name, v in pi.items()},
            "real_minus_pseudo": _boot(lambda j: spearman(pi["real"][j], lp[j]) - spearman(pi["pseudo"][j], lp[j]),
                                       cl, **boot),
            "swap_censoring": {"real": R["swap_censoring"], "pseudo": P["swap_censoring"]}}
    else:
        out["corr_pi_logpop"] = {"available": False, "reason": "swap_prior.csv.gz with like rows missing in real or "
                                                               "pseudo"}
    return out


def token_gate(path) -> dict:
    """Pilot-1 token-channel gate from pilot1_gate's decision.json: {checked, pass (bool, or None if unknown), ...}."""
    if not path:
        return {"checked": False, "pass": None, "note": "no --gate given"}
    p = Path(path)
    try:
        g = json.loads(p.read_text(encoding="utf-8")).get("gate") or {}
    except (OSError, ValueError) as exc:
        return {"checked": False, "pass": None, "path": str(p), "note": f"gate unreadable ({type(exc).__name__}): "
                                                                        "run pilot 1 (pilot1_gate.py) first"}
    ok = g.get("pass")
    return {"checked": isinstance(ok, bool), "pass": ok if isinstance(ok, bool) else None, "path": str(p),
            **{k: g.get(k) for k in ("ml1m_raw_UAUC", "sports_raw_NDCG@10", "input_checks_ok")}}


def apply_gate(decision: dict, gate: dict) -> dict:
    """triage_verdict.md §3: on a failed token-channel gate nothing in pilot 3 is interpretable."""
    decision["token_channel_gate"] = gate["pass"]
    if gate["pass"] is False:
        decision["verdict_if_gate_passed"] = decision["verdict"]
        decision["verdict"] = "GATE_FAIL_UNINTERPRETABLE"
    return decision


def main() -> None:
    ap = argparse.ArgumentParser()
    for x in ARMS:
        ap.add_argument(f"--{x}", required=True, help="pyes_scorer output dir (or its scores.csv.gz)")
    ap.add_argument("--panel", required=True, help="the REAL rated panel (jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--bi_boot", type=int, default=1000,
                    help="bootstrap size of the Bias Index CIs (metrics.bias_index_ci default; the slow step)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", default=None,
                    help="pilot1_gate decision.json (token-channel gate common to pilots 1 and 3)")
    a = ap.parse_args()
    arms = {x: load_arm(getattr(a, x)) for x in ARMS}
    for x in ARMS:
        if not arms[x]["complete"]:
            print(f"WARNING {arms[x]['dir']}: no report.json (scorer run may be incomplete)")
    rows = [json.loads(line) for line in open(a.panel, encoding="utf-8") if line.strip()]
    flags, src, chk = treated_from_panel(a.panel, rows)
    res = analyze(arms, rows, flags, a.n_boot, a.seed, bi_boot=a.bi_boot)
    res["treated"].update(source=src, **chk)
    res["token_channel_gate"] = gate = token_gate(a.gate)
    apply_gate(res["decision"], gate)
    if gate["pass"] is not True:
        print(f"WARNING token-channel gate {'FAILED' if gate['pass'] is False else 'not checked'} "
              f"({gate.get('path') or 'no --gate'}): pilot-3 verdict {res['decision']['verdict']}")
    res["inputs"] = {"panel": a.panel, "gate": a.gate, **{x: arms[x]["dir"] for x in ARMS}}
    text = json.dumps(strict_json(res), indent=2, allow_nan=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
