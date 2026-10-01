"""Pilot 3 (A2): pseudonym knockout of brand names in personalised prompts, with a real-brand-swap placebo.

Given a rated panel (from build_rated_panels, with history_titles/history_brands/candidate_brands), write two
transformed panels in which every occurrence of each brand string — in history titles, candidate titles and
candidate texts — is replaced consistently within the example:
  * pseudo : brand -> a deterministic invented name (hash-seeded syllables; same brand -> same pseudonym)
  * placebo: brand -> another REAL brand from the same brand-popularity tercile (fixed derangement)
Items whose brand string does not occur in their own title are left untouched and flagged
`candidate_brand_in_title = False` so the analysis can restrict to treated candidates.

    python -m src.confrec.pseudonymize --panel toys_rated.jsonl --out_dir outputs/confrec/panels
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

SYL = ["vel", "mor", "tan", "rik", "sol", "dra", "ken", "lu", "qua", "zen", "bro", "tis", "nal", "or",
       "vex", "ami", "dor", "fen", "gla", "hy", "ivo", "jas", "kor", "lem", "nyx", "pra", "rul", "sab"]


def pseudonym(brand: str) -> str:
    h = int(hashlib.sha1(brand.lower().encode()).hexdigest(), 16)
    n = 2 + h % 2
    parts = [SYL[(h >> (5 * k)) % len(SYL)] for k in range(n)]
    return "".join(parts).capitalize()


def sub(text: str, mapping: dict[str, str]) -> str:
    for b in sorted(mapping, key=len, reverse=True):  # longest first avoids partial overlaps
        if b:
            text = re.sub(re.escape(b), mapping[b], text, flags=re.IGNORECASE)
    return text


def rebuild(rec: dict, mapping: dict[str, str]) -> dict:
    r = dict(rec)
    hist_titles = [sub(t, mapping) for t in rec["history_titles"]]
    r["history"] = [f"{t} (rated {int(x)}/5)" for t, x in zip(hist_titles, rec["history_ratings"])]
    r["candidate_titles"] = [sub(t, mapping) for t in rec["candidate_titles"]]
    r["candidate_texts"] = [sub(t, mapping) for t in rec["candidate_texts"]]
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.panel, encoding="utf-8")]
    # brand popularity = summed candidate popularity of items carrying the brand (panel-local proxy)
    bpop = Counter()
    for r in rows:
        for b, p in zip(r["candidate_brands"], r["candidate_popularity"]):
            if b:
                bpop[b] += p
    brands = sorted(bpop, key=lambda b: bpop[b])
    k = len(brands)
    rng = random.Random(a.seed)
    placebo = {}
    for t in range(3):  # derangement within each popularity tercile
        tier = brands[t * k // 3:(t + 1) * k // 3]
        perm = tier[:]
        for _ in range(100):
            rng.shuffle(perm)
            if all(x != y for x, y in zip(tier, perm)) or len(tier) < 2:
                break
        placebo.update(dict(zip(tier, perm)))
    stem = Path(a.panel).stem
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_treated = n_cand = 0
    with open(out / f"{stem}_pseudo.jsonl", "w", encoding="utf-8") as fp, \
            open(out / f"{stem}_placebo.jsonl", "w", encoding="utf-8") as fq:
        for r in rows:
            used = {b for b in r["candidate_brands"] + r["history_brands"] if b}
            flags = [bool(b) and b.lower() in t.lower() for b, t in zip(r["candidate_brands"], r["candidate_titles"])]
            n_treated += sum(flags)
            n_cand += len(flags)
            ps = rebuild(r, {b: pseudonym(b) for b in used})
            pl = rebuild(r, {b: placebo.get(b, b) for b in used})
            for x in (ps, pl):
                x["candidate_brand_in_title"] = flags
            fp.write(json.dumps(ps, ensure_ascii=False) + "\n")
            fq.write(json.dumps(pl, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "brands": k, "treated_candidates": n_treated, "candidates": n_cand}))


if __name__ == "__main__":
    main()
