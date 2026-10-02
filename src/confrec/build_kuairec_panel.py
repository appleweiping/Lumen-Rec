"""Pilot 2 (B1/A7) panel: fully observed KuaiRec pairs (MAR labels) plus the exposure covariates of a logged view.

Binding spec: idea-stage/PREREG_AMENDMENT_1.md section P2. KuaiRec (Gao et al., CIKM'22) removes every small-matrix
interaction from the big matrix, so the registered label "observed in big with watch_ratio >= thr" is 0 for every
small-matrix pair. The panel therefore stores the MAR outcome and the covariates n_u, n_v; pilot_kuairec simulates
the logged view O(u,v) ~ Bernoulli(min(1, c * n_u * n_v^alpha)) on the SAME pairs.
  candidates   per user, n_cands random small-matrix videos with a usable title; candidate_labels = 1[wr >= thr] (MAR)
  n_v          candidate_popularity = # distinct big-matrix users (all users, unfiltered) with a row for v
  n_u          user_exposure = # big-matrix rows of u
  history      u's last hist_len distinct big-matrix positives (wr >= thr) over the whole log, candidate videos and
               caption-less videos excluded (standard KuaiRec protocol), as caption strings
Text is NaN-safe; 'UNKNOWN' / 'nan' count as missing; title = caption or manual_cover_text; videos without a title
are excluded from candidates and history. Sidecar <out stem>.meta.json: shared_pairs (small and big overlap,
expected ~0), target_density D = # distinct big (u,v) pairs on small-matrix videos / (# big users x # small-matrix
videos) (distinct pairs, because mean e(u,v) is a per-pair exposure probability; the raw-row count is
target_density_rows, equal when big has no repeated pairs), target_density_excl_small_users (sensitivity: big users
outside the small matrix only, whose small-video cells KuaiRec did not remove), caption coverage, popularity summary.

    python -m src.confrec.build_kuairec_panel --root data/raw/KuaiRec/data \
        --caption data/raw/KuaiRec/data/kuairec_caption_category.csv --out outputs/confrec/panels/kuairec.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.confrec.stats import rank_bins, strict_json

MISSING = {"", "nan", "unknown"}
CAT_COLS = ("first_level_category_name", "second_level_category_name", "third_level_category_name")


def s(x) -> str:
    """NaN-safe text cell; the KuaiRec placeholders 'UNKNOWN' / 'nan' (any case) count as missing."""
    t = "" if pd.isna(x) else str(x).strip()
    return "" if t.casefold() in MISSING else t


def _tags(x) -> str:
    t = s(x)
    if t.startswith("[") and t.endswith("]"):          # list-like "[a,b]" / "['a', 'b']"; "[]" -> ""
        t = ", ".join(p for p in (s(q.strip().strip("'\"")) for q in t[1:-1].split(",")) if p)
    return t


def caption_table(cap: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """video_id -> (title, text). title = caption or manual_cover_text ('' if neither is usable); text carries the
    category path and topic tags when present. The first row of a duplicated video_id wins."""
    vid = pd.to_numeric(cap["video_id"], errors="coerce")
    out: dict[int, tuple[str, str]] = {}
    for v, r in zip(vid, cap.to_dict("records")):
        if pd.isna(v) or int(v) in out:
            continue
        title = s(r.get("caption")) or s(r.get("manual_cover_text"))
        cats = [c for c in (s(r.get(k)) for k in CAT_COLS) if c]
        tags = _tags(r.get("topic_tag"))
        parts = ([f"Category: {' / '.join(cats)}."] if cats else []) + ([f"Tags: {tags}."] if tags else [])
        out[int(v)] = (title, " ".join(parts))
    return out


def _last_distinct(vids, exclude: set, k: int) -> list[int]:
    out, seen = [], set()
    for v in reversed(vids):
        if v in exclude or v in seen:
            continue
        seen.add(v)
        out.append(v)
        if len(out) == k:
            break
    return out[::-1]


def _summary(x) -> dict:
    x = np.asarray(x, float)
    q = np.quantile(x, [0.2, 0.5, 0.8]) if len(x) else [np.nan] * 3
    return {"n": int(len(x)), "min": float(x.min()) if len(x) else None, "q20": q[0], "median": q[1], "q80": q[2],
            "max": float(x.max()) if len(x) else None, "n_zero": int((x == 0).sum()),
            "n_distinct": int(len(np.unique(x)))}


def build(small: pd.DataFrame, big: pd.DataFrame, captions: dict[int, tuple[str, str]], n_users: int = 300,
          n_cands: int = 200, thr: float = 2.0, hist_len: int = 10, seed: int = 0, min_hist: int = 3):
    """small: user_id, video_id, watch_ratio. big (UNFILTERED, all users): user_id, video_id, watch_ratio, timestamp.
    Returns (rows, meta)."""
    small = small.dropna(subset=["user_id", "video_id", "watch_ratio"])
    n_small_dup = int(small.duplicated(["user_id", "video_id"]).sum())
    small = small.drop_duplicates(["user_id", "video_id"], keep="first")
    su, sv = small.user_id.to_numpy(np.int64), small.video_id.to_numpy(np.int64)
    sw = small.watch_ratio.to_numpy(float)
    big = big.dropna(subset=["user_id", "video_id"])
    bu, bv = big.user_id.to_numpy(np.int64), big.video_id.to_numpy(np.int64)
    bw = big.watch_ratio.to_numpy(float)
    bt = np.nan_to_num(pd.to_numeric(big.timestamp, errors="coerce").to_numpy(float), nan=-np.inf)  # undated = oldest
    assert min(su.min(), sv.min(), bu.min(), bv.min()) >= 0, "negative KuaiRec ids"

    M = int(max(sv.max(), bv.max())) + 1
    big_keys = np.unique(bu * M + bv)                       # distinct (u, v) pairs of the big matrix
    shared = np.isin(su * M + sv, big_keys)
    n_v = np.bincount(big_keys % M, minlength=M)            # distinct big users per video, all users
    big_users, rows_per_user = np.unique(bu, return_counts=True)
    n_u = dict(zip(big_users.tolist(), rows_per_user.tolist()))
    small_videos = np.unique(sv)
    denom = len(big_users) * len(small_videos)
    on_small = np.isin(big_keys % M, small_videos)
    density = float(on_small.sum() / denom)
    density_rows = float(np.isin(bv, small_videos).sum() / denom)
    # sensitivity: big users outside the small matrix only (KuaiRec removed the small users' small-video cells)
    other = ~np.isin(big_users, su)
    n_other = int(other.sum())
    density_excl = (float((on_small & ~np.isin(big_keys // M, su)).sum() / (n_other * len(small_videos)))
                    if n_other else float("nan"))

    titled = np.array(sorted(v for v, (t, _) in captions.items() if t), np.int64)
    title_ok = np.isin(sv, titled)
    pool = ~shared & title_ok
    o = np.lexsort((sv, su))
    pu, pv, pw = su[o][pool[o]], sv[o][pool[o]], sw[o][pool[o]]

    hmask = (bw >= thr) & np.isin(bv, titled)               # positives with a usable title
    hu, hv, ht = bu[hmask], bv[hmask], bt[hmask]
    o = np.lexsort((np.arange(len(hu)), ht, hu))            # by user, then time, then file order
    hu, hv = hu[o], hv[o]

    rng = np.random.default_rng(seed)
    order = rng.permutation(np.unique(su))
    rows, skipped_pool, skipped_hist = [], 0, 0
    for u in order.tolist():
        lo, hi = np.searchsorted(pu, u, "left"), np.searchsorted(pu, u, "right")
        if hi - lo < n_cands:
            skipped_pool += 1
            continue
        pick = lo + rng.choice(hi - lo, n_cands, replace=False)
        cv, cw = pv[pick].tolist(), pw[pick].tolist()
        a, b = np.searchsorted(hu, u, "left"), np.searchsorted(hu, u, "right")
        hist = _last_distinct(hv[a:b].tolist(), set(cv), hist_len)
        if len(hist) < min_hist:
            skipped_hist += 1
            continue
        rows.append({
            "user_id": str(u), "source_event_id": f"{u}::kuairec",
            "history": [captions[v][0] for v in hist],
            "history_item_ids": [str(v) for v in hist],
            "candidate_item_ids": [str(v) for v in cv],
            "candidate_titles": [captions[v][0] for v in cv],
            "candidate_texts": [captions[v][1] for v in cv],
            "candidate_labels": [int(w >= thr) for w in cw],             # MAR
            "candidate_watch_ratio": [float(w) for w in cw],
            "candidate_popularity": [int(n_v[v]) for v in cv],           # n_v
            "user_exposure": int(n_u.get(u, 0)),                         # n_u
            "source": "kuairec",
        })
        if len(rows) >= n_users:
            break

    labels = np.array([x for r in rows for x in r["candidate_labels"]], float)
    pop = np.array([x for r in rows for x in r["candidate_popularity"]], float)
    if not len(rows):
        raise ValueError("no eligible KuaiRec user (check --n_cands / --min_hist / captions)")
    mar = float(labels.mean())
    if not 0 < mar < 1:
        raise ValueError(f"degenerate MAR labels at thr={thr}: positive rate {mar}")
    grp = rank_bins(pop, 5)
    all_v = np.union1d(small_videos, np.unique(bv))
    meta = {
        "n_users": len(rows), "n_users_requested": n_users, "n_cands": n_cands, "n_pairs": int(len(labels)),
        "thr": thr, "hist_len": hist_len, "min_hist": min_hist, "seed": seed, "mar_pos_rate": mar,
        "n_users_skipped_pool": skipped_pool, "n_users_skipped_history": skipped_hist,
        "n_small_rows": int(len(su)), "n_small_duplicates_dropped": n_small_dup, "n_small_users": int(len(order)),
        "n_small_videos": int(len(small_videos)), "n_big_rows": int(len(bu)), "n_big_pairs": int(len(big_keys)),
        "n_big_users": int(len(big_users)), "n_small_users_in_big": int(len(big_users) - n_other),
        "shared_pairs": int(shared.sum()),
        "target_density": density, "target_density_rows": density_rows,
        "target_density_excl_small_users": density_excl,
        "n_caption_missing": int((~np.isin(small_videos, titled)).sum()),
        "n_caption_missing_all_videos": int((~np.isin(all_v, titled)).sum()),
        "n_small_pairs_no_title": int((~title_ok).sum()),
        "popularity": {**_summary(pop), "share_head_bin": float((grp == 4).mean()),
                       "share_tail_bin": float((grp == 0).mean())},
        "user_exposure": _summary([r["user_exposure"] for r in rows]),
    }
    if len(rows) < n_users:
        print(f"[warn] only {len(rows)} of {n_users} users eligible", flush=True)
    sh = meta["popularity"]
    if not (0.15 <= sh["share_head_bin"] <= 0.25 and 0.15 <= sh["share_tail_bin"] <= 0.25):
        print(f"[warn] n_v head/tail quintile shares {sh['share_head_bin']:.3f}/{sh['share_tail_bin']:.3f} outside "
              "0.15-0.25 (ties): the pilot flags endpoint 3 as not valid", flush=True)
    return rows, meta


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir with small_matrix.csv and big_matrix.csv")
    ap.add_argument("--caption", default=None, help="kuairec_caption_category.csv (default: <root>/...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_users", type=int, default=300)
    ap.add_argument("--n_cands", type=int, default=200)
    ap.add_argument("--thr", type=float, default=2.0)
    ap.add_argument("--hist_len", type=int, default=10)
    ap.add_argument("--min_hist", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    root = Path(a.root)
    dt = {"user_id": "int64", "video_id": "int64", "watch_ratio": "float64"}
    small = pd.read_csv(root / "small_matrix.csv", usecols=list(dt), dtype=dt)
    big = pd.read_csv(root / "big_matrix.csv", usecols=[*dt, "timestamp"], dtype={**dt, "timestamp": "float64"})
    bad: list = []
    cap = pd.read_csv(a.caption or root / "kuairec_caption_category.csv", engine="python", dtype=str,
                      on_bad_lines=lambda line: bad.append(line))   # returning None skips the line
    rows, meta = build(small, big, caption_table(cap), a.n_users, a.n_cands, a.thr, a.hist_len, a.seed, a.min_hist)
    meta.update({"n_caption_rows": int(len(cap)), "n_caption_bad_lines": len(bad)})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out.with_suffix(".meta.json").write_text(json.dumps(strict_json(meta), indent=2, allow_nan=False),
                                             encoding="utf-8")
    print(json.dumps(strict_json(meta)))


if __name__ == "__main__":
    main()
