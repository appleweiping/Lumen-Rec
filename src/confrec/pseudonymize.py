"""Pilot 3 (§3.3; binding operationalization idea-stage/PREREG_AMENDMENT_1.md P3): pseudonym knockout of store
names in rated-panel prompts, with a popularity-matched real-brand placebo.

    python -m src.confrec.pseudonymize --panel outputs/confrec/panels/toys_rated.jsonl [--seed 0] [--out_dir DIR]

writes <stem>_pseudo.jsonl, <stem>_placebo.jsonl and <stem>_pseudonym_report.json next to the panel (or in
--out_dir). Brands are the panel's `store` strings (candidate_brands + history_brands), grouped by casefolded key.
  * stop-list (never substituted; counted per reason in the report): empty; placeholder (STOP_GENERIC); generic_word
    (GENERIC_WORDS: category and retail words such as Toys, Kids, Games, whose Title-Case category labels open
    every candidate text); short (< 4 characters); no_letters (digits/punctuation only); and the data-driven
    common_word rule: the store occurs as an all-lowercase whole word in >= COMMON_MIN distinct texts of OTHER
    stores' items and in at least as many such texts as it matches in its own items' texts (a store name used as
    an ordinary word, e.g. 'for kids'; brand mentions are capitalised). Re-evaluated until no key is added;
  * ONE compiled regex for all substitutable spellings: whole word ((?<!\\w)...(?!\\w)), case-insensitive, longest
    brand first at each position. It is a trie-factored alternation of re.escape(brand) whose sibling characters
    are distinct re.IGNORECASE classes, so it matches exactly what the longest-first sorted alternation matches.
    A callback maps each match through its casefolded key, so inserted text is never re-scanned (no swap chains,
    no re-substitution inside a pseudonym);
  * pseudo: one global map per panel, key -> hash-seeded syllable name, injective over the panel's keys (re-hashed
    with a salt counter on collision; the syllable-count range widens with the salt, so the name space never runs
    out), never casefold-equal to a real store string (panel or sidecar) and never containing one as a whole word;
  * placebo: derangement (no fixed points) within popularity terciles over the SAME key set as pseudo (history-only
    brands included). Popularity = <panel stem>.brand_pop.json (category-wide events per store, from
    build_rated_panels) when present, else panel-wide occurrences over candidates + histories (source recorded).
    Terciles are stats.rank_bins(pop, 3); a tier with < 2 brands merges into a neighbour.
Substituted fields: history_titles, candidate_titles, candidate_texts; `history` is re-rendered exactly
"<title> (rated r/5)" (int r). Added per row: candidate_brand_in_title (= the treated set: the candidate's own
store is substitutable, occurs as a whole word, case-insensitively, in its ORIGINAL title, and every such
occurrence lies inside a replaced span, i.e. no partly overlapping store match leaves it in place) and
n_subst (substitutions in the row; identical in both arms because both use the same regex and key set).
Integrity checks (key sets equal, no fixed points, injective pseudonyms, equal n_subst, no treated own store left in
the pseudo arm) are computed, written to the report, and raise RuntimeError when violated.
Scope (amendment 1 P3): brand tokens in titles beyond the `store` field are not substituted, so the analysis
population is store-in-title candidates; their share is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.confrec.stats import rank_bins, strict_json

SYL = ["vel", "mor", "tan", "rik", "sol", "dra", "ken", "lu", "qua", "zen", "bro", "tis", "nal", "or",
       "vex", "ami", "dor", "fen", "gla", "hy", "ivo", "jas", "kor", "lem", "nyx", "pra", "rul", "sab"]
STOP_GENERIC = frozenset({"generic", "unknown", "n/a", "none", "amazon", "amazon basics", "amazonbasics",
                          "unbranded", "various", "other", "brand", "no brand", "does not apply"})
GENERIC_WORDS = frozenset({
    "toys", "toy store", "toy shop", "toys & games", "toys and games", "games & toys", "kids toys", "baby toys",
    "games", "game", "game store", "video games", "video game", "gaming", "gamer", "gamers", "console",
    "consoles", "controller", "controllers", "accessories", "accessory", "electronics", "kids", "kid", "baby",
    "babies", "children", "child", "toddler", "toddlers", "family", "puzzle", "puzzles", "party", "parties",
    "craft", "crafts", "arts", "arts & crafts", "gift", "gifts", "play", "learning", "educational", "education",
    "school", "hobby", "hobbies", "novelty", "costume", "costumes", "collectibles", "dolls", "plush", "books",
    "music", "sports", "outdoor", "outdoors", "home", "kitchen", "store", "shop", "official", "official store",
    "seller", "retail", "direct", "deals", "imports", "products", "supplies", "goods"})
MIN_LEN = 4
COMMON_MIN = 3  # common_word: >= this many distinct other-store texts use the store as a lowercase word
SALT_WIDEN = 16  # every SALT_WIDEN re-draws allow one more syllable count
MAX_SALT = 10_000
FIELDS = ("history_titles", "candidate_titles", "candidate_texts")
OWN = {"history_titles": "history_brands", "candidate_titles": "candidate_brands",
       "candidate_texts": "candidate_brands"}
NEVER = re.compile(r"(?!x)x")


def key(brand) -> str:
    return str(brand or "").strip().casefold()


def stop_reason(brand) -> str | None:
    """Why a store string is never substituted by the static stop-list (None = substitutable)."""
    s = str(brand or "").strip()
    if not s:
        return "empty"
    norm = " ".join(s.casefold().split())
    if norm in STOP_GENERIC:
        return "placeholder"
    if norm in GENERIC_WORDS:
        return "generic_word"
    if len(s) < MIN_LEN:
        return "short"
    if not any(ch.isalpha() for ch in s):
        return "no_letters"
    return None


def pseudonym(brand: str, salt: int = 0) -> str:
    """Deterministic syllable name for a brand (case-insensitive); `salt` > 0 draws an alternative. Draws with
    salt < SALT_WIDEN have 2-3 syllables; each further SALT_WIDEN salts allow one more syllable."""
    h = int(hashlib.sha1((key(brand) + (f"#{salt}" if salt else "")).encode()).hexdigest(), 16)
    w = 2 + salt // SALT_WIDEN
    n, h = 2 + h % w, h // w
    parts = []
    for _ in range(n):
        h, k = divmod(h, len(SYL))
        parts.append(SYL[k])
    return "".join(parts).capitalize()


def _char_classes(chars) -> dict:
    """char -> representative of its re.IGNORECASE equivalence class (smallest code point among `chars`)."""
    rep, reps = {}, []
    for c in sorted(chars):
        if c.lower() == c and c.upper() == c:  # uncased: only matches itself
            rep[c] = c
            continue
        r = next((r for r, p in reps if p.fullmatch(c)), None)
        if r is None:
            reps.append((c, re.compile(re.escape(c), re.IGNORECASE)))
            r = c
        rep[c] = r
    return rep


def _trie_pattern(root: dict) -> str:
    """Pattern for a char trie (iterative, so long brands cannot exhaust the recursion limit). An ending node
    wraps its continuation in a greedy optional group: a longer brand is tried before ending here."""
    out, stack = {}, [(root, False)]
    while stack:
        node, done = stack.pop()
        if not done:
            stack.append((node, True))
            stack.extend((sub, False) for ch, sub in node.items() if ch)
            continue
        kids = [re.escape(ch) + out.pop(id(sub)) for ch, sub in sorted(node.items()) if ch]
        body = "" if not kids else kids[0] if len(kids) == 1 else "(?:" + "|".join(kids) + ")"
        out[id(node)] = f"(?:{body})?" if kids and "" in node else body
    return out[id(root)]


def brand_regex(brands) -> re.Pattern:
    """One whole-word, case-insensitive pattern matching any of `brands`, longest brand first at a position.
    Brands are rewritten char by char to their IGNORECASE class representative (same matched strings), so trie
    siblings never match the same character and the trie equals the longest-first sorted alternation."""
    bs = {s for s in (str(x).strip() for x in brands) if s}
    rep = _char_classes({ch for b in bs for ch in b})
    trie: dict = {}
    for b in {"".join(rep[ch] for ch in b) for b in bs}:
        node = trie
        for ch in b:
            node = node.setdefault(ch, {})
        node[""] = {}
    return re.compile(r"(?<!\w)(?:" + _trie_pattern(trie) + r")(?!\w)", re.IGNORECASE) if trie else NEVER


def resolver(spellings: dict):
    """matched text -> key, for the rare match whose casefold differs from its brand's (re's simple case folding
    lets 'İ' match 'i', but 'İ'.casefold() != 'i'): the first spelling that fully matches it under IGNORECASE."""
    cache: dict = {}

    def f(text: str):
        if text not in cache:
            cache[text] = next((k for k, ss in spellings.items() for s in ss
                                if re.fullmatch(re.escape(s), text, re.IGNORECASE)), None)
        return cache[text]
    return f


def _key_of(text: str, keys, resolve=None):
    k = text.casefold()
    return k if k in keys or resolve is None else resolve(text)


def subn(text: str, rx: re.Pattern, mapping: dict, hits: list | None = None, resolve=None) -> tuple[str, int]:
    """Single-pass substitution; `mapping` is keyed by casefolded brand. Returns (text, n substitutions)."""
    def rep(m):
        k = _key_of(m.group(0), mapping, resolve)
        if k not in mapping:
            raise KeyError(f"regex match {m.group(0)!r} has no mapping key")
        if hits is not None:
            hits.append(k)
        return mapping[k]
    return rx.subn(rep, str(text))


def sub(text: str, mapping: dict, rx: re.Pattern | None = None) -> str:
    """Convenience: substitute {brand: replacement} in one pass (whole word, case-insensitive)."""
    return subn(text, rx or brand_regex(mapping), {key(b): v for b, v in mapping.items()},
                resolve=resolver({key(b): [str(b).strip()] for b in mapping}))[0]


_WORD: dict = {}


def _word(brand: str) -> re.Pattern:
    p = _WORD.get(brand)
    if p is None:
        p = _WORD[brand] = re.compile(r"(?<!\w)" + re.escape(brand) + r"(?!\w)", re.IGNORECASE)
    return p


def has_word(brand: str, text: str) -> bool:
    """Whole-word, case-insensitive occurrence of `brand` in `text`."""
    return _word(brand).search(str(text)) is not None


def brand_index(rows: list) -> dict:
    """Panel-wide brands after the static stop-list: substitutable keys with spellings / canonical spelling / slot
    occurrences, the stop-list tally, and keys seen in candidates vs histories."""
    occ, spell, stop = Counter(), defaultdict(Counter), defaultdict(Counter)
    in_cand, in_hist = set(), set()
    for r in rows:
        for field, seen in (("candidate_brands", in_cand), ("history_brands", in_hist)):
            for b in r.get(field) or []:
                s = str(b or "").strip()
                why = stop_reason(s)
                if why:
                    stop[why][s] += 1
                    continue
                k = key(s)
                occ[k] += 1
                spell[k][s] += 1
                seen.add(k)
    canon = {k: min(c, key=lambda s: (-c[s], s)) for k, c in spell.items()}
    return {"keys": sorted(occ), "occ": occ, "spellings": spell, "canon": canon, "stop": stop,
            "in_cand": in_cand, "in_hist": in_hist}


def own_texts(rows: list) -> list:
    """Distinct (own store key, text) pairs over history titles, candidate titles and candidate texts."""
    out = set()
    for r in rows:
        for f in FIELDS:
            ob = r.get(OWN[f]) or []
            for j, t in enumerate(r.get(f) or []):
                out.add((key(ob[j]) if j < len(ob) else "", str(t)))
    return sorted(out)


def common_word_keys(texts: list, spellings: dict) -> dict:
    """Data-driven common_word stop rule over distinct (own key, text) pairs: key -> {own, foreign_lower}, where
    own = texts of its own items it matches in and foreign_lower = other stores' texts in which it matches as an
    all-lowercase word. Flagged iff foreign_lower >= max(COMMON_MIN, own); repeated on the remaining keys (a
    removed longer store can unmask a shorter one) until nothing is added."""
    live, flagged, cache = set(spellings), {}, {}
    todo = range(len(texts))
    while live:
        order = sorted(live)  # resolver order must not depend on the hash seed
        rx, resolve = brand_regex([s for k in order for s in spellings[k]]), resolver({k: spellings[k] for k in order})
        for i in todo:
            hits = [(_key_of(m.group(0), live, resolve), m.group(0).islower()) for m in rx.finditer(texts[i][1])]
            if hits:
                cache[i] = hits
            else:
                cache.pop(i, None)
        own, low = Counter(), Counter()
        for i, hits in cache.items():
            o = texts[i][0]
            own.update({k for k, _ in hits if k == o})
            low.update({k for k, lower in hits if lower and k != o})
        new = {k for k in live if low[k] >= max(COMMON_MIN, own[k])}
        if not new:
            break
        flagged.update({k: {"own": int(own[k]), "foreign_lower": int(low[k])} for k in new})
        live -= new
        todo = [i for i, hits in cache.items() if any(k in new for k, _ in hits)]
    return flagged


def panel_brands(rows: list) -> dict:
    """brand_index after the full stop-list (static + common_word), with the compiled regex and match resolver."""
    idx = brand_index(rows)
    common = common_word_keys(own_texts(rows), idx["spellings"])
    idx["common_word"] = {idx["canon"][k]: v for k, v in sorted(common.items())}
    for k in common:
        for s, c in idx["spellings"].pop(k).items():
            idx["stop"]["common_word"][s] += c
        idx["occ"].pop(k)
        idx["canon"].pop(k)
        idx["in_cand"].discard(k)
        idx["in_hist"].discard(k)
    idx["keys"] = sorted(idx["spellings"])
    idx["rx"] = brand_regex([s for k in idx["keys"] for s in idx["spellings"][k]])
    idx["resolve"] = resolver(idx["spellings"])
    return idx


def treated_flags(rec: dict, keys, rx: re.Pattern | None = None) -> list[bool]:
    """Per candidate: its own store is substitutable (key in `keys`), occurs as a whole word in its ORIGINAL title,
    and every such occurrence lies inside a span the substitution regex `rx` replaces (its own match or a longer
    store containing it); an occurrence only partly overlapped by another store's match would survive."""
    keys = set(keys)
    rx = rx or brand_regex(keys)
    titles = rec["candidate_titles"]
    out = []
    for b, t in zip(rec.get("candidate_brands") or [""] * len(titles), titles):
        s, t = str(b or "").strip(), str(t)
        occ = [m.span() for m in _word(s).finditer(t)] if stop_reason(s) is None and key(s) in keys else []
        spans = [m.span() for m in rx.finditer(t)] if occ else []
        out.append(bool(occ) and all(any(x <= i and j <= y for x, y in spans) for i, j in occ))
    return out


def pseudonym_map(keys, forbidden: set, rx: re.Pattern) -> tuple[dict, int]:
    """Injective key -> pseudonym; a name equal to a used name or a real store (casefold), or containing a
    substitutable brand as a whole word, is re-drawn with the next salt."""
    out, used, n_coll = {}, set(), 0
    for k in sorted(keys):
        for salt in range(MAX_SALT):
            name = pseudonym(k, salt)
            f = name.casefold()
            if f not in used and f not in forbidden and not rx.search(name):
                break
            n_coll += 1
        else:
            raise RuntimeError(f"no free pseudonym for {k!r} after {MAX_SALT} draws")
        out[k] = name
        used.add(f)
    return out, n_coll


def _derange(xs: list, rng: random.Random) -> dict:
    for _ in range(1000):  # uniform over derangements (acceptance ~1/e)
        p = xs[:]
        rng.shuffle(p)
        if all(a != b for a, b in zip(xs, p)):
            return dict(zip(xs, p))
    return dict(zip(xs, xs[1:] + xs[:1]))


def placebo_map(keys, pop: dict, seed: int = 0) -> tuple[dict, list]:
    """Derangement of `keys` within popularity terciles (rank_bins on pop); tiers of < 2 merge into a neighbour."""
    keys = sorted(keys)
    if len(keys) < 2:
        raise SystemExit(f"placebo needs >= 2 substitutable brands, got {len(keys)} (no `store` strings in the panel? "
                         "check the panel meta's n_items_with_store: the Amazon meta file must be slimmed with `store`)")
    bins = rank_bins([pop[k] for k in keys], 3)
    tiers = [t for t in ([k for k, b in zip(keys, bins) if b == j] for j in range(3)) if t]
    while len(tiers) > 1 and min(map(len, tiers)) < 2:
        i = next(i for i, t in enumerate(tiers) if len(t) < 2)
        if i == 0:
            j = 1
        elif i == len(tiers) - 1:
            j = i - 1
        else:
            j = i - 1 if len(tiers[i - 1]) <= len(tiers[i + 1]) else i + 1
        lo, hi = min(i, j), max(i, j)
        tiers[lo:hi + 1] = [tiers[lo] + tiers[hi]]
    rng = random.Random(seed)
    out = {}
    for t in tiers:
        out.update(_derange(sorted(t), rng))
    if set(out) != set(keys) or any(out[k] == k for k in out):
        raise RuntimeError("placebo is not a derangement of the key set")
    return out, [len(t) for t in tiers]


def rebuild(rec: dict, rx: re.Pattern, mapping: dict, hits: dict | None = None,
            resolve=None) -> tuple[dict, Counter]:
    """Substituted copy of a panel row and the per-field substitution counts. `hits[field]` collects
    (own brand key, matched key) pairs when given."""
    r, n = dict(rec), Counter()
    for f in FIELDS:
        if f not in rec:
            continue
        own, vals = rec.get(OWN[f]) or [], []
        for j, t in enumerate(rec[f]):
            h = [] if hits is not None else None
            t2, c = subn(t, rx, mapping, h, resolve)
            vals.append(t2)
            n[f] += c
            if h:
                hits[f].extend((key(own[j]) if j < len(own) else "", k) for k in h)
        r[f] = vals
    if len(r["history_titles"]) != len(rec["history_ratings"]):
        raise ValueError(f"history_titles/history_ratings length mismatch in {rec.get('source_event_id')}")
    r["history"] = [f"{t} (rated {int(x)}/5)" for t, x in zip(r["history_titles"], rec["history_ratings"])]
    return r, n


def load_brand_pop(panel) -> tuple[dict | None, str]:
    p = Path(panel).with_suffix(".brand_pop.json")
    return (json.loads(p.read_text(encoding="utf-8")), str(p)) if p.exists() else (None, "")


def build_maps(rows: list, idx: dict, brand_pop: dict | None = None, seed: int = 0) -> dict:
    """Pseudonym and placebo maps over idx['keys'] (both keyed by casefolded brand) with their provenance."""
    keys, canon = idx["keys"], idx["canon"]
    forbidden = {key(s) for r in rows for f in ("candidate_brands", "history_brands") for s in r.get(f) or []
                 if key(s)} | {key(s) for s in (brand_pop or {})}
    ps_map, n_coll = pseudonym_map(keys, forbidden, idx["rx"])
    if brand_pop is not None:
        agg = Counter()
        for s, v in brand_pop.items():
            agg[key(s)] += float(v)
        missing = [k for k in keys if k not in agg]
        pop, pop_source = {k: agg.get(k, 0.0) for k in keys}, "sidecar"
    else:
        missing, pop, pop_source = [], {k: float(idx["occ"][k]) for k in keys}, "panel_occurrence"
    pl_keys, tiers = placebo_map(keys, pop, seed)
    return {"pseudo": ps_map, "placebo": {k: canon[pl_keys[k]] for k in keys}, "placebo_keys": pl_keys,
            "n_collisions": n_coll, "forbidden": forbidden, "pop_source": pop_source, "missing": missing,
            "tiers": tiers}


def _require(checks: dict) -> None:
    """Boolean checks must be True, count checks 0 (raised explicitly: survives python -O)."""
    bad = {k: v for k, v in checks.items() if ((v is not True) if isinstance(v, bool) else (v != 0))}
    if bad:
        raise RuntimeError(f"pseudonymize integrity checks failed: {bad}")


def pseudonymize(rows: list, brand_pop: dict | None = None, seed: int = 0) -> tuple[list, list, dict]:
    """(pseudo rows, placebo rows, report) for a rated panel; brand_pop = {store: category events} or None."""
    idx = panel_brands(rows)
    keys, canon, rx, resolve = idx["keys"], idx["canon"], idx["rx"], idx["resolve"]
    mp = build_maps(rows, idx, brand_pop, seed)
    ps_map, pl_map = mp["pseudo"], mp["placebo"]
    checks = {"placebo_keys_equal_pseudo_keys": set(pl_map) == set(ps_map) == set(keys),
              "placebo_fixed_points": sum(mp["placebo_keys"][k] == k for k in keys),
              "pseudonyms_injective": len({v.casefold() for v in ps_map.values()}) == len(keys),
              "pseudonyms_equal_to_a_real_store": sum(v.casefold() in mp["forbidden"] for v in ps_map.values())}
    _require(checks)

    out_ps, out_pl, hits, key_set = [], [], defaultdict(list), set(keys)
    n_tr = n_c = n_stop_in_title = n_diff = n_left_ps = n_in_pl = 0
    tot = {"pseudo": Counter(), "placebo": Counter()}
    for r in rows:
        flags = treated_flags(r, key_set, rx)
        n_tr += sum(flags)
        n_c += len(r["candidate_titles"])
        n_stop_in_title += sum(bool(stop_reason(b) and str(b or "").strip() and has_word(str(b).strip(), t))
                               for b, t in zip(r.get("candidate_brands") or [], r["candidate_titles"]))
        ps, n_ps = rebuild(r, rx, ps_map, hits, resolve)
        pl, n_pl = rebuild(r, rx, pl_map, None, resolve)
        n_diff += n_ps != n_pl
        for j, f in enumerate(flags):
            if f:
                b = str(r["candidate_brands"][j]).strip()
                n_left_ps += has_word(b, ps["candidate_titles"][j])
                n_in_pl += has_word(b, pl["candidate_titles"][j])  # another brand of the title mapped onto it
        tot["pseudo"].update(n_ps)
        tot["placebo"].update(n_pl)
        for x, n in ((ps, n_ps), (pl, n_pl)):
            x["candidate_brand_in_title"] = flags
            x["n_subst"] = int(sum(n.values()))
        out_ps.append(ps)
        out_pl.append(pl)
    checks.update(rows_with_unequal_n_subst=n_diff, treated_own_store_left_in_pseudo_title=n_left_ps)
    _require(checks)

    per_brand, foreign = Counter(), Counter()
    for f, pairs in hits.items():
        for own, k in pairs:
            per_brand[k] += 1
            foreign[k] += own != k
    n_rows = max(1, len(rows))
    stopped = {s for c in idx["stop"].values() for s in c if s}
    report = {
        "n_rows": len(rows), "n_candidates": n_c, "seed": seed,
        "n_brand_strings": len({s for k in keys for s in idx["spellings"][k]} | stopped),
        "n_brands": len(keys), "n_brands_in_candidates": len(idx["in_cand"]),
        "n_brands_history_only": len(idx["in_hist"] - idx["in_cand"]),
        "n_stoplisted": len(stopped),
        "stoplist": {why: {"strings": len([s for s in c if s]), "slots": int(sum(c.values()))}
                     for why, c in sorted(idx["stop"].items())},
        "stoplist_rule": {"min_len": MIN_LEN, "placeholder": sorted(STOP_GENERIC),
                          "generic_word": sorted(GENERIC_WORDS), "no_letters": True,
                          "common_word": f"all-lowercase whole-word match in >= max({COMMON_MIN}, own) distinct "
                                         "other-store texts (own = own-item texts matched), iterated"},
        "common_word_stores": [{"brand": b, **v} for b, v in sorted(idx["common_word"].items(),
                                                                       key=lambda x: -x[1]["foreign_lower"])][:100],
        "n_collisions_resolved": mp["n_collisions"],
        "popularity_source": mp["pop_source"], "n_keys_missing_from_sidecar": len(mp["missing"]),
        "placebo_tier_sizes": mp["tiers"],
        "n_treated": n_tr, "treated_share": n_tr / max(1, n_c),
        "n_candidates_stoplisted_store_in_title": n_stop_in_title,
        "n_treated_own_store_in_placebo_title": n_in_pl,
        "mean_n_subst": {a: sum(t.values()) / n_rows for a, t in tot.items()},
        "n_subst_by_field": {a: {f: int(t[f]) for f in FIELDS} for a, t in tot.items()},
        "checks": checks,
        "top_brands_by_matches": [{"brand": canon[k], "n_matches": int(n), "n_in_other_items_text": int(foreign[k])}
                                  for k, n in per_brand.most_common(25)],
        "scope_note": "only the `store` field is substituted (brand tokens in titles beyond it are not); the "
                      "analysis population is store-in-title candidates (treated_share)",
    }
    return out_ps, out_pl, report


def _write_jsonl(path: Path, rows: list) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out_dir", default=None, help="default: the panel's directory")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    panel = Path(a.panel)
    rows = [json.loads(line) for line in open(panel, encoding="utf-8") if line.strip()]
    brand_pop, pop_path = load_brand_pop(panel)
    ps, pl, rep = pseudonymize(rows, brand_pop, a.seed)
    rep.update(panel=str(panel), brand_pop_path=pop_path or None)
    out = Path(a.out_dir) if a.out_dir else panel.parent
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / f"{panel.stem}_pseudo.jsonl", ps)
    _write_jsonl(out / f"{panel.stem}_placebo.jsonl", pl)
    text = json.dumps(strict_json(rep), indent=2, ensure_ascii=False, allow_nan=False)
    (out / f"{panel.stem}_pseudonym_report.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
