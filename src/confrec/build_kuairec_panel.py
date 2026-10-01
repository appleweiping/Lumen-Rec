"""Pilot 2 (B1/A7): KuaiRec panel with MAR (fully-observed) vs MNAR (logged) labels for the SAME pairs.

KuaiRec (Gao et al., CIKM'22): `small_matrix.csv` is (almost) fully observed for 1,411 users × 3,327 videos;
`big_matrix.csv` is the ordinary logged (MNAR) feedback. For U users × C candidate videos from the small matrix:
  label_mar  = 1[watch_ratio_small >= thr]
  label_mnar = 1[(u,i) observed in big matrix with watch_ratio >= thr]   (unobserved -> 0, the usual convention)
History = the user's big-matrix videos with watch_ratio >= thr (most recent `hist_len`), excluding candidates.
Pairs present in BOTH matrices are reported and dropped from the candidate pool (pre-registered).

    python -m src.confrec.build_kuairec_panel --root data/raw/KuaiRec/data --out outputs/confrec/panels/kuairec.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir with small_matrix.csv, big_matrix.csv, kuairec_caption_category.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_users", type=int, default=300)
    ap.add_argument("--n_cands", type=int, default=200)
    ap.add_argument("--thr", type=float, default=2.0)
    ap.add_argument("--hist_len", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.root)
    small = pd.read_csv(root / "small_matrix.csv", usecols=["user_id", "video_id", "watch_ratio"])
    big = pd.read_csv(root / "big_matrix.csv", usecols=["user_id", "video_id", "watch_ratio", "timestamp"])
    cap = pd.read_csv(root / "kuairec_caption_category.csv", engine="python", on_bad_lines="skip")
    cap["video_id"] = pd.to_numeric(cap["video_id"], errors="coerce")
    cap = cap.dropna(subset=["video_id"]).astype({"video_id": int}).set_index("video_id")

    def text(v):
        if v not in cap.index:
            return None, ""
        r = cap.loc[v]
        title = str(r.get("caption") or r.get("manual_cover_text") or "").strip()
        cat = " / ".join(str(r.get(c)) for c in ("first_level_category_name", "second_level_category_name")
                         if pd.notna(r.get(c)))
        tags = str(r.get("topic_tag") or "")
        return (title or None), f"Category: {cat}. Tags: {tags}"

    small_users = set(small.user_id)
    big = big[big.user_id.isin(small_users)]
    shared = small.merge(big[["user_id", "video_id"]].drop_duplicates(), on=["user_id", "video_id"])
    shared_keys = set(zip(shared.user_id, shared.video_id))
    big_pos = big[big.watch_ratio >= a.thr]
    big_pos_keys = set(zip(big_pos.user_id, big_pos.video_id))
    pop = big.groupby("video_id").size().to_dict()

    rng = random.Random(a.seed)
    users = sorted(small_users)
    rng.shuffle(users)
    out_rows = []
    for u in users:
        su = small[small.user_id == u]
        cands = [(int(v), float(w)) for v, w in zip(su.video_id, su.watch_ratio)
                 if (u, int(v)) not in shared_keys and text(int(v))[0]]
        if len(cands) < a.n_cands:
            continue
        cands = rng.sample(cands, a.n_cands)
        cand_ids = {v for v, _ in cands}
        h = big_pos[(big_pos.user_id == u) & (~big_pos.video_id.isin(cand_ids))].sort_values("timestamp")
        hist = [text(int(v))[0] for v in h.video_id if text(int(v))[0]][-a.hist_len:]
        if len(hist) < 3:
            continue
        out_rows.append({
            "user_id": str(u), "source_event_id": f"{u}::kuairec",
            "history": hist,
            "candidate_item_ids": [str(v) for v, _ in cands],
            "candidate_titles": [text(v)[0] for v, _ in cands],
            "candidate_texts": [text(v)[1] for v, _ in cands],
            "candidate_labels": [int(w >= a.thr) for _, w in cands],          # MAR
            "candidate_labels_mnar": [int((u, v) in big_pos_keys) for v, _ in cands],
            "candidate_watch_ratio": [w for _, w in cands],
            "candidate_popularity": [int(pop.get(v, 0)) for v, _ in cands],
            "source": "kuairec",
        })
        if len(out_rows) >= a.n_users:
            break
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = sum(len(r["candidate_labels"]) for r in out_rows)
    meta = {"users": len(out_rows), "candidates": n, "thr": a.thr,
            "mar_pos_rate": sum(sum(r["candidate_labels"]) for r in out_rows) / max(1, n),
            "mnar_pos_rate": sum(sum(r["candidate_labels_mnar"]) for r in out_rows) / max(1, n),
            "shared_small_big_pairs_dropped": len(shared_keys)}
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
