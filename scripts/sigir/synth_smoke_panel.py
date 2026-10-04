"""Synthetic rated panel for infrastructure smoke tests (idea-stage/PREREG_AMENDMENT_2.md carve-out O).

No real user, item or rating data: titles are invented from a fixed syllable list, labels are random. The rows follow the
real rated-panel schema (`outputs/confrec/panels/toys_rated.jsonl`) with Toys-like prompt lengths (long titles and a
category text), so memory and throughput measured on it bound the real Gate-FT run.

    python scripts/sigir/synth_smoke_panel.py --out /tmp/panel.jsonl [--n_users 96] [--n_cands 8] [--hist 10] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYLLABLES = ("ba", "ko", "ri", "ta", "mu", "len", "sor", "vi", "dax", "pel", "nu", "quo", "zil", "hep", "tar", "mon")


def name(rnd: random.Random, words: int) -> str:
    return " ".join(rnd.choice(SYLLABLES).capitalize() + rnd.choice(SYLLABLES) for _ in range(words))


def synth_rows(n_users: int, n_cands: int, hist: int, seed: int = 0, title_words: int = 12, text_words: int = 60) -> list:
    rnd = random.Random(seed)
    rows = []
    for u in range(n_users):
        h_titles = [f"Synthetic {name(rnd, title_words)}" for _ in range(hist)]
        h_ratings = [float(rnd.randint(1, 5)) for _ in range(hist)]
        c_titles = [f"Synthetic {name(rnd, title_words)}" for _ in range(n_cands)]
        labels = [1 if rnd.random() < 0.65 else 0 for _ in range(n_cands)]
        ts0 = 1_600_000_000_000 + u * 1_000_000
        rows.append({
            "user_id": f"S{u}", "source_event_id": f"S{u}::{ts0}", "source": "toys",
            "history": [f"{t} (rated {int(r)}/5)" for t, r in zip(h_titles, h_ratings)],
            "history_item_ids": [f"SH{u}_{k}" for k in range(hist)], "history_titles": h_titles,
            "history_ratings": h_ratings, "history_brands": [name(rnd, 1) for _ in range(hist)],
            "candidate_item_ids": [f"SC{u}_{k}" for k in range(n_cands)], "candidate_titles": c_titles,
            "candidate_texts": [f"Categories: {name(rnd, text_words)}. " for _ in range(n_cands)],
            "candidate_brands": [name(rnd, 1) for _ in range(n_cands)],
            "candidate_ratings": [5.0 if y else 1.0 for y in labels], "candidate_labels": labels,
            "candidate_popularity": [rnd.randint(1, 500) for _ in range(n_cands)],
            "candidate_popularity_prior": [rnd.randint(0, 400) for _ in range(n_cands)],
            "candidate_timestamps": sorted(ts0 + 1000 * (k + 1) for k in range(n_cands)),
        })
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_users", type=int, default=96)
    ap.add_argument("--n_cands", type=int, default=8)
    ap.add_argument("--hist", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title_words", type=int, default=12)
    ap.add_argument("--text_words", type=int, default=60)
    a = ap.parse_args(argv)
    rows = synth_rows(a.n_users, a.n_cands, a.hist, a.seed, a.title_words, a.text_words)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {out}: {len(rows)} synthetic users x {a.n_cands} candidates (no real data)")


if __name__ == "__main__":
    main()
