"""Synthesize tiny raw inputs for the CPU dry run (about 1/30 of the server scale).

    python synth.py            (writes ./data/raw/..., ./panels/sports_next_1k.jsonl, ./ref_ranks/..., ./lumen_users.csv)

ML-1M (movies.dat / ratings.dat, latin-1), Amazon-2023 slim toys + games (jsonl.gz, with stores incl. title brands,
generic 'Kids'/'Games', a data-driven common word 'Magic', a re-reviewed item, exact duplicates, a lone surrogate,
missing titles/ratings), KuaiRec small/big/caption CSVs under a directory whose name contains a space, a Lumen-style
next-item panel (101 candidates, positive_item_index, candidate_popularity_groups) and matching C-CRP ref ranks.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
rng = random.Random(7)
nrng = np.random.default_rng(7)


def zipf_pick(n, s=1.1):
    w = 1 / np.arange(1, n + 1) ** s
    return w / w.sum()


# ------------------------------------------------------------------ ML-1M
def ml1m(n_users=160, n_movies=400):
    d = RAW / "ml-1m"
    d.mkdir(parents=True, exist_ok=True)
    genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Children's", "Animation"]
    with open(d / "movies.dat", "w", encoding="latin-1", newline="") as f:
        for m in range(1, n_movies + 1):
            t = f"Movie {m} {'Amélie ' if m % 37 == 0 else ''}Story ({1950 + m % 50})"
            f.write(f"{m}::{t}::{'|'.join(rng.sample(genres, 2))}\n")
    q = nrng.normal(0, 1, n_movies)
    pw = zipf_pick(n_movies, 0.9)
    with open(d / "ratings.dat", "w", encoding="latin-1", newline="") as f:
        for u in range(1, n_users + 1):
            k = rng.randint(40, 110)
            ms = nrng.choice(n_movies, k, replace=False, p=pw) + 1
            t = 956703932 + u * 1000
            bias = rng.gauss(0, 0.6)
            for m in ms:
                t += rng.randint(1, 500)
                r = int(np.clip(round(3 + bias + 0.9 * q[m - 1] + rng.gauss(0, 1.2)), 1, 5))
                f.write(f"{u}::{m}::{r}::{t}\n")


# ------------------------------------------------------------------ Amazon 2023 (slim)
TOY_STORES = ["LEGO", "Hasbro", "Mattel", "Melissa & Doug", "Fisher-Price", "Hape", "Ravensburger®", "Pokémon",
              "Kids", "Generic", "Ty", "Magic", "Playmobil", "VTech", "Crayola", "Nerf", "Funko", "Bandai",
              "Schleich", "Spin Master", "LeapFrog", "Little Tikes", "Step2", "Intex", "KidKraft", "Brio",
              "Magna-Tiles", "Learning Resources", "Janod", "Djeco"] + [f"Toyco{k:03d}" for k in range(40)]
GAME_STORES = ["Nintendo", "Sony", "Microsoft", "PowerA", "Turtle Beach", "Mad Catz", "Razer", "Logitech G",
               "HORI", "Generic", "Games", "8BitDo", "Sega", "Capcom", "Ubisoft", "Bandai Namco", "Square Enix",
               "Konami", "Atlus", "Thrustmaster"] + [f"Gameco{k:03d}" for k in range(30)]
TOY_NOUNS = ["Building Set", "Action Figure", "Plush Bear", "Puzzle 1000 Pieces", "Doll House", "Race Car",
             "Board Game", "Craft Kit", "Train Set", "Water Gun", "Science Kit", "Card Deck"]
GAME_NOUNS = ["Wireless Controller", "Gaming Headset", "Charging Dock", "Racing Wheel", "Arcade Stick",
              "Adventure Game", "RPG Deluxe Edition", "Memory Card", "Carrying Case", "Fighting Game"]


def amazon(domain, cat, stores, nouns, top_cat, n_items=700, n_users=220, seed=0):
    r = random.Random(seed)
    d = RAW / f"amazon_{domain}"
    d.mkdir(parents=True, exist_ok=True)
    items = []
    with gzip.open(d / f"meta_{cat}.jsonl.gz", "wt", encoding="utf-8") as f:
        for k in range(n_items):
            asin = f"B{domain[:1].upper()}{k:07d}"
            st = stores[min(int(r.paretovariate(0.9)) - 1, len(stores) - 1)] if r.random() < 0.93 else None
            noun = r.choice(nouns)
            if st and r.random() < 0.6:
                title = f"{st} {noun} {k}" if r.random() < 0.7 else f"{noun} {k} by {st}"
            else:
                title = f"{noun} {k}"
            desc = [r.choice(["Great gift for kids aged 3+.", "Durable and fun.", "Compatible with LEGO bricks.",
                              "Includes magic tricks for the whole family.", "A wonder of design.",
                              "Batteries not included."])]
            if k == 5:  # lone surrogate escape as written by slim_amazon2023 (ASCII escape)
                line = json.dumps({"parent_asin": asin, "title": title + " \ud83d", "categories": [top_cat, "Misc"],
                                   "description": desc, "store": st})
                f.write(line + "\n")
                items.append(asin)
                continue
            if k == 6:
                title = None  # skipped by the builder
            f.write(json.dumps({"parent_asin": asin, "title": title, "categories": [top_cat, noun.split()[0]],
                                "description": desc, "store": st}, ensure_ascii=False) + "\n")
            if title:
                items.append(asin)
    q = {a: r.gauss(0, 1) for a in items}
    pw = zipf_pick(len(items), 0.8)
    with gzip.open(d / f"{cat}.jsonl.gz", "wt", encoding="utf-8") as f:
        for u in range(n_users):
            uid = f"A{domain[:2].upper()}{u:05d}XYZ"
            k = r.randint(28, 60)
            picks = nrng.choice(len(items), k, replace=False, p=pw)
            t = 1500000000000 + u * 10_000_000
            bias = r.gauss(0, 0.5)
            evs = []
            for j in picks:
                t += r.randint(1000, 10_000_000)
                a = items[j]
                rating = float(int(np.clip(round(3.2 + bias + 1.0 * q[a] + r.gauss(0, 1.4)), 1, 5)))
                evs.append({"user_id": uid, "parent_asin": a, "rating": rating, "timestamp": t})
            if u % 5 == 0:  # re-reviewed item: same parent_asin later with another rating (dedup keeps the first)
                e = dict(evs[3])
                e["timestamp"] = t + 5000
                e["rating"] = 6.0 - e["rating"]
                evs.append(e)
            if u % 7 == 0:
                evs.append(dict(evs[1]))  # exact duplicate row
            if u % 11 == 0:
                evs.append({"user_id": uid, "parent_asin": "BNOTINMETA", "rating": 5.0, "timestamp": t + 9})
                evs.append({"user_id": uid, "parent_asin": items[0], "rating": None, "timestamp": t + 10})
            for e in evs:
                f.write(json.dumps(e) + "\n")


# ------------------------------------------------------------------ KuaiRec
def kuairec(n_small_users=15, n_small_videos=300, n_big_users=200, n_videos=900):
    d = RAW / "KuaiRec" / "KuaiRec 2.0" / "data"
    d.mkdir(parents=True, exist_ok=True)
    small_users = list(range(14, 14 + n_small_users))
    small_videos = list(range(n_small_videos))
    cols = ["user_id", "video_id", "play_duration", "video_duration", "time", "date", "timestamp", "watch_ratio"]
    with open(d / "small_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for u in small_users:
            for v in small_videos:
                wr = float(np.exp(nrng.normal(-0.2 + 0.25 * math.sin(v), 0.8)))
                w.writerow([u, v, int(wr * 10000), 10000, "2020-07-05 05:27:48.378", 20200705,
                            1593898068.378 + v, round(wr, 6)])
    pv = zipf_pick(n_videos, 0.7)
    vperm = nrng.permutation(n_videos)
    with open(d / "big_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for u in range(n_big_users):
            k = rng.randint(60, 140)
            vs = vperm[nrng.choice(n_videos, k, replace=False, p=pv)]
            t = 1593000000.0 + u
            for v in vs:
                if u in small_users and v < n_small_videos:
                    continue  # KuaiRec removes the small-matrix cells from big
                t += rng.uniform(10, 1000)
                wr = float(np.exp(nrng.normal(0.0, 0.9)))
                ts = "" if rng.random() < 0.01 else f"{t:.3f}"
                w.writerow([u, int(v), int(wr * 9000), 9000, "", 20200701, ts, round(wr, 6)])
    cat1 = ["美食", "搞笑", "游戏", "UNKNOWN", "生活"]
    with open(d / "kuairec_caption_category.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "manual_cover_text", "caption", "topic_tag", "first_level_category_id",
                    "first_level_category_name", "second_level_category_id", "second_level_category_name",
                    "third_level_category_id", "third_level_category_name"])
        for v in range(n_videos):
            if v % 41 == 0:
                continue  # absent from the caption file
            cap = f"视频{v} 今天的{rng.choice(['美食', '日常', '游戏', '旅行'])}分享, 很好看" if v % 13 else ""
            cover = "UNKNOWN" if v % 17 == 0 else f"封面{v}"
            if v % 29 == 0:
                cap, cover = "nan", ""
            tag = rng.choice(["[]", "[美食,搞笑]", "['日常']", "", "UNKNOWN"])
            c1 = rng.choice(cat1)
            w.writerow([v, cover, cap, tag, 1, c1, -124 if c1 == "UNKNOWN" else 2, "UNKNOWN", -124, "UNKNOWN"])
        f.write('9999,bad,"line",with,too,many,fields,x,y,z,extra,extra2\n')  # skipped via on_bad_lines


# ------------------------------------------------------------------ Lumen next-item panel + C-CRP ref ranks
def next_item(n_events=34, n_cands=101):
    pd_ = HERE / "panels"
    pd_.mkdir(parents=True, exist_ok=True)
    rows, ref = [], []
    for e in range(n_events):
        uid = f"AE{e:04d}SPORTSUSER{e * 7 % 13:02d}"
        ts = 1576559236892 + e * 99991
        ids = [f"B0S{e:03d}{c:04d}" for c in range(n_cands)]
        pos = rng.randrange(n_cands)
        groups = [rng.choice(["head", "mid", "mid", "tail", "tail"]) for _ in ids]
        hist_n = rng.randint(3, 20)
        rows.append({
            "source_event_id": f"{uid}::{ts}", "user_id": uid,
            "history": [f"Sports gear {e}-{h} (Outdoor)" for h in range(hist_n)],
            "history_item_ids": [f"B0H{e:03d}{h:03d}" for h in range(hist_n)],
            "candidate_item_ids": ids,
            "candidate_titles": [f"Sports item {i}" for i in ids],
            "candidate_texts": [f"Category: Sports. Popularity group {g}." for g in groups],
            "candidate_popularity_groups": groups,
            "candidate_labels": [int(c == pos) for c in range(n_cands)],
            "positive_item_id": ids[pos], "positive_item_title": f"Sports item {ids[pos]}",
            "positive_item_text": "", "positive_item_index": pos, "timestamp": str(ts), "split_name": "test",
            "num_candidates": n_cands, "source_pointwise_size": n_cands})
        ref.append((f"{uid}::{ts}", uid, rng.randint(1, 40), n_cands))
    for k in range(6):  # ref events outside the 1k subset (the real ref file has all 10k events)
        ref.append((f"AEXTRA{k}::{k}", f"AEXTRA{k}", rng.randint(1, 101), n_cands))
    with open(pd_ / "sports_next_1k.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rd = HERE / "ref_ranks" / "sports"
    rd.mkdir(parents=True, exist_ok=True)
    with gzip.open(rd / "ccrp_v3.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_event_id", "user_id", "positive_rank", "num_candidates"])
        for row in sorted(ref):
            w.writerow(row)


def main():
    import argparse
    global HERE, RAW
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE))
    ap.add_argument("--amazon_users", type=int, default=220)
    ap.add_argument("--ml_users", type=int, default=160)
    ap.add_argument("--kuairec", action="store_true")
    a = ap.parse_args()
    HERE = Path(a.root)
    RAW = HERE / "data" / "raw"
    ml1m(n_users=a.ml_users)
    amazon("toys", "Toys_and_Games", TOY_STORES, TOY_NOUNS, "Toys & Games", n_users=a.amazon_users, seed=1)
    amazon("games", "Video_Games", GAME_STORES, GAME_NOUNS, "Video Games", n_users=a.amazon_users, seed=2)
    if a.kuairec:
        kuairec()
    next_item()
    print("synthesized under", RAW)


if __name__ == "__main__":
    main()
