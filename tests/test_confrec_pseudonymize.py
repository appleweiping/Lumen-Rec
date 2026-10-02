import json
import random
import re
import string
import subprocess
import sys
from collections import Counter

import pytest

from src.confrec.pseudonymize import (NEVER, SYL, _require, brand_regex, build_maps, common_word_keys, has_word,
                                      own_texts, panel_brands, placebo_map, pseudonym, pseudonym_map, pseudonymize,
                                      stop_reason, sub, treated_flags)
from src.confrec.stats import rank_bins


def _rec(uid, hist_titles, hist_brands, cand_titles, cand_brands, texts=None, ratings=None):
    n = len(cand_titles)
    return {"user_id": uid, "source_event_id": f"{uid}::1",
            "history": [f"{t} (rated {r}/5)" for t, r in zip(hist_titles, ratings or [5] * len(hist_titles))],
            "history_titles": hist_titles, "history_ratings": [float(r) for r in (ratings or [5] * len(hist_titles))],
            "history_brands": hist_brands, "history_item_ids": [f"h{j}" for j in range(len(hist_titles))],
            "candidate_titles": cand_titles, "candidate_texts": texts or [""] * n, "candidate_brands": cand_brands,
            "candidate_popularity": list(range(1, n + 1)), "candidate_labels": [j % 2 for j in range(n)],
            "candidate_item_ids": [f"{uid}c{j}" for j in range(n)]}


def test_pseudonym_is_deterministic_and_consistent():
    assert pseudonym("LEGO") == pseudonym("lego")
    assert pseudonym("LEGO") != pseudonym("Hasbro")
    m = {"LEGO": pseudonym("LEGO")}
    assert sub("LEGO City Fire Truck by lego", m).count(pseudonym("LEGO")) == 2


def test_swap_pair_is_single_pass():
    # sequential re.sub turned this into 'Mattel Barbie and Mattel Monopoly'
    m = {"Mattel": "Hasbro", "Hasbro": "Mattel"}
    assert sub("Mattel Barbie and Hasbro Monopoly", m) == "Hasbro Barbie and Mattel Monopoly"
    assert sub("Hasbro Monopoly; Mattel UNO", m) == "Mattel Monopoly; Hasbro UNO"
    # 3-cycle: every brand moves exactly once
    assert sub("Alpha Beta Gamma", {"Alpha": "Beta", "Beta": "Gamma", "Gamma": "Alpha"}) == "Beta Gamma Alpha"
    # a shorter brand never matches inside inserted text
    assert sub("Lego set", {"Lego": "Orvel", "Or": "Zz"}) == "Orvel set"


def test_whole_word_only():
    assert sub("Melissa & Doug Shape Sorter", {"Hape": "Velmor"}) == "Melissa & Doug Shape Sorter"
    assert sub("Hape Wooden Blocks", {"Hape": "Velmor"}) == "Velmor Wooden Blocks"
    assert sub("Party Activity Set by Ty", {"Ty": "Zan"}) == "Party Activity Set by Zan"
    assert sub("Funny game for kids", {"Fun": "Qua"}) == "Funny game for kids"
    assert not has_word("Hape", "Wooden Shape Sorter") and has_word("Hape", "HAPE pound-a-peg")
    assert not has_word("Ty", "Party Activity Set") and not has_word("Fun", "Funny game")
    # longest brand wins at a position; a longer brand that fails the word boundary falls back to the shorter
    m = {"Hape": "A1", "Hape Toys": "B2"}
    assert sub("Hape Toys set; hape toyshop", m) == "B2 set; A1 toyshop"


def test_unicode_simple_case_fold_match_is_resolved():
    # re's IGNORECASE lets 'İ' match 'i' although 'İ'.casefold() != 'i'; the match must still map to its brand
    assert sub("istanbul oyuncak set", {"İstanbul Oyuncak": "Velmor"}) == "Velmor set"
    rows = [_rec("u", ["istanbul oyuncak car"], ["İstanbul Oyuncak"], ["Hape Blocks", "LEGO City"], ["Hape", "LEGO"])]
    ps, pl, _ = pseudonymize(rows)
    assert "istanbul" not in ps[0]["history"][0].lower() and ps[0]["n_subst"] == pl[0]["n_subst"] == 3


def _alternation(brands):
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(b) for b in sorted(brands, key=lambda b: (-len(b), b)))
                      + r")(?!\w)", re.IGNORECASE)


def test_trie_regex_equals_sorted_alternation():
    # incl. characters re.IGNORECASE treats as equal although str.lower() differs: s/ſ, k/K(Kelvin), µ/μ, ß/ẞ,
    # i/İ/ı, σ/ς
    for letters in ("abcdeHAPE &.-'", "sſSkKKµμßẞiİıσςΣ a-"):
        rng = random.Random(1)
        brands = sorted({"".join(rng.choice(letters) for _ in range(rng.randint(1, 6))).strip()
                         for _ in range(300)} - {""})
        alt, trie = _alternation(brands), brand_regex(brands)
        for _ in range(400):
            text = "".join(rng.choice(letters + "  ") for _ in range(80))
            assert [m.span() for m in alt.finditer(text)] == [m.span() for m in trie.finditer(text)], text
    # case-equivalent trie siblings: the trie used to stop at 'sa' although 'ſa b' is the longer match
    assert [m.span() for m in brand_regex(["sa", "ſa b"]).finditer("sa b")] == [(0, 4)]
    assert brand_regex([]).search("anything") is None


def test_long_store_strings_compile():
    # the recursive trie raised RecursionError beyond ~1000 characters
    long = "Acme " * 600
    assert brand_regex([long.strip(), "Acme"]).fullmatch(long.strip())
    chain = ["a" * n for n in range(1, 400)]                     # 399 nested optional groups
    assert brand_regex(chain).fullmatch("a" * 399)
    rows = [_rec("u", [], [], [f"{long.strip()} Truck", "LEGO Car"], [long.strip(), "LEGO"])]
    ps, pl, rep = pseudonymize(rows)
    assert ps[0]["candidate_brand_in_title"] == [True, True] and rep["n_brands"] == 2


def test_stop_list():
    for b in ("", "  ", "Ty", "Fun", "Toy", "3M", "1234", "#1 -", "Generic", "UNKNOWN", "n/a", "Amazon Basics",
              "AmazonBasics", "No Brand", "does not apply", "Various"):
        assert stop_reason(b) is not None, b
    for b in ("Kids", "TOYS", "Games", "Video  Games", "Toys & Games", "Puzzle", "Party", "Accessories"):
        assert stop_reason(b) == "generic_word", b
    for b in ("LEGO", "Hape", "Melissa & Doug", "Amazon Essentials", "Learning Resources", "Kids Preferred"):
        assert stop_reason(b) is None, b


def test_common_word_store_leaves_ordinary_words():
    # reviewer probe: with stores Kids/Toys, 'for kids' and the 'Categories: Toys & Games' prefix were rewritten in
    # every row of both arms, and a 'Kids' title was flagged treated
    text = "Categories: Toys & Games. Great gift for kids and toddlers."
    rows = [_rec("u1", ["LEGO City"], ["LEGO"], ["Hape Blocks", "Wooden puzzle for kids", "Kids Car for Kids"],
                 ["Hape", "Hape", "Kids"], texts=[text, "x", "y"]),
            _rec("u2", ["Mattel Barbie"], ["Mattel"], ["Toys Truck", "LEGO Duplo"], ["Toys", "LEGO"], texts=["a", "b"])]
    ps, pl, rep = pseudonymize(rows)
    for arm in (ps, pl):
        assert arm[0]["candidate_texts"][0] == text
        assert arm[0]["candidate_titles"][1:] == ["Wooden puzzle for kids", "Kids Car for Kids"]
        assert arm[1]["candidate_titles"][0] == "Toys Truck"
        assert arm[0]["candidate_brand_in_title"] == [True, False, False]
    assert ps[0]["candidate_titles"][0] != "Hape Blocks" and rep["stoplist"]["generic_word"]["strings"] == 2


def test_data_driven_common_word_rule():
    filler = ["a wonder of a set", "no wonder kids love it", "wonder and joy", "the magic box opens",
              "a magic box trick", "inside the magic box", "Compatible with LEGO", "fits LEGO bricks",
              "works with LEGO", "not lego branded"]
    rows = [_rec(f"u{j}", [], [], [f"Item {j}", "Wonder Puzzle", "LEGO Car", "Magic Box Kit", "Magic Wand"],
                 ["Acme Toys Co", "Wonder", "LEGO", "Magic Box", "Magic"], texts=[t, "", "", "", ""])
            for j, t in enumerate(filler)]
    idx = panel_brands(rows)
    # Wonder: lowercase in 3 other-store texts vs 1 own text; Magic Box likewise; Magic only becomes visible once
    # the longer Magic Box is removed (second iteration); LEGO is mentioned capitalised (1 lowercase text only)
    assert set(idx["common_word"]) == {"Wonder", "Magic Box", "Magic"}
    assert idx["keys"] == ["acme toys co", "lego"]
    assert idx["common_word"]["Wonder"] == {"own": 1, "foreign_lower": 3}
    raw = common_word_keys(own_texts(rows), {"lego": {"LEGO": 1}})
    assert raw == {}
    ps, pl, rep = pseudonymize(rows)
    assert ps[0]["candidate_texts"][0] == "a wonder of a set" and ps[0]["candidate_titles"][1] == "Wonder Puzzle"
    assert "LEGO" not in ps[6]["candidate_texts"][0] and rep["stoplist"]["common_word"]["strings"] == 3


def test_overlapping_stores_treated_only_when_removed():
    # 'Big Toy' wins the leftmost-longest match, so 'Toy Box Co' survives in its own title: not treated
    rows = [_rec("u", ["Big Toy Wagon"], ["Big Toy"],
                 ["Big Toy Box Co Storage Bench", "Big Toy Truck", "Toy Box Co Bin"], ["Toy Box Co", "Big Toy", "Toy Box Co"])]
    ps, pl, rep = pseudonymize(rows)
    assert ps[0]["candidate_brand_in_title"] == [False, True, True]
    assert ps[0]["candidate_titles"][0].endswith(" Box Co Storage Bench")
    assert rep["checks"]["treated_own_store_left_in_pseudo_title"] == 0
    # an own store fully inside a longer store's match is removed with it: treated
    rows = [_rec("u", [], [], ["Hape Toys Xylophone", "Hape Toys Drum", "Hape Blocks"], ["Hape", "Hape Toys", "Hape"])]
    ps, pl, rep = pseudonymize(rows)
    assert ps[0]["candidate_brand_in_title"] == [True, True, True]
    assert not any(has_word("Hape", t) for t in ps[0]["candidate_titles"])


def test_pseudonyms_injective_on_3000_brands():
    rng = random.Random(0)
    brands = {"".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(4, 10))) for _ in range(3000)}
    keys = sorted({b.casefold() for b in brands})
    rx = brand_regex(brands)
    m, n_coll = pseudonym_map(keys, set(keys), rx)
    names = [v.casefold() for v in m.values()]
    assert len(set(names)) == len(keys)                          # injective
    assert not set(names) & set(keys)                            # never a real brand
    assert all(rx.search(v) is None for v in m.values())         # never contains a real brand as a word
    assert n_coll > 0                                            # collisions happened and were resolved


def test_pseudonym_space_never_exhausted():
    # 2-3 syllables give 28^2 + 28^3 = 22,736 names; the salt loop used to spin forever beyond that
    assert len(SYL) ** 2 + len(SYL) ** 3 == 22736
    keys = [f"store{i:05d}" for i in range(23500)]
    m, n_coll = pseudonym_map(keys, set(keys), NEVER)
    names = {v.casefold() for v in m.values()}
    assert len(names) == len(keys) and not names & set(keys) and n_coll > 0
    assert max(len(v) for v in m.values()) > 3 * 3            # some names needed a 4th syllable
    assert pseudonym("LEGO", 3) == pseudonym("lego", 3)        # salts below the widening keep 2-3 syllables


def test_integrity_checks_raise_without_assert():
    _require({"keys_equal": True, "fixed_points": 0})
    for bad in ({"keys_equal": False}, {"fixed_points": 1}, {"fixed_points": 2}):   # 1 == True must not pass
        with pytest.raises(RuntimeError):
            _require(bad)


def test_pseudonym_never_equals_a_real_brand():
    taken = pseudonym("Acme Corp")
    m, n = pseudonym_map(["acme corp"], {taken.casefold()}, brand_regex([taken]))
    assert m["acme corp"] != taken and n >= 1


def test_placebo_is_derangement_within_tiers():
    keys = [f"brand{i:03d}" for i in range(50)]
    pop = {k: i for i, k in enumerate(keys)}
    m, tiers = placebo_map(keys, pop, seed=3)
    assert set(m) == set(keys) and all(m[k] != k for k in keys)
    tier_of = dict(zip(keys, rank_bins([pop[k] for k in keys], 3)))
    assert sorted(tiers) == sorted(Counter(tier_of.values()).values()) and min(tiers) >= 16
    assert all(tier_of[k] == tier_of[m[k]] for k in keys)        # popularity tercile matched
    m2, tiers2 = placebo_map(["a", "b", "c", "d"], {"a": 1, "b": 2, "c": 3, "d": 4})
    assert all(m2[k] != k for k in m2) and min(tiers2) >= 2      # tiny tiers merged, still no fixed points


def test_history_only_brand_substituted_in_both_arms():
    rows = [_rec("u1", ["Mattel Barbie Doll", "LEGO Star Wars"], ["Mattel", "LEGO"],
                 ["LEGO Technic Car", "Hape Shape Sorter", "Generic Ball"], ["LEGO", "Hape", "Generic"],
                 texts=["by LEGO", "Hape toys are fun", ""]),
            _rec("u2", ["Hasbro Monopoly"], ["Hasbro"], ["Hape Xylophone", "LEGO Duplo"], ["Hape", "LEGO"])]
    ps, pl, rep = pseudonymize(rows)
    idx = panel_brands(rows)
    keys = idx["keys"]
    assert set(keys) == {"mattel", "lego", "hape", "hasbro"}     # Mattel/Hasbro appear only in histories
    assert rep["n_brands_history_only"] == 2
    mp = build_maps(rows, idx)                                    # the maps pseudonymize used (same seed)
    assert set(mp["placebo"]) == set(mp["pseudo"]) == set(keys)
    assert all(mp["placebo_keys"][k] != k for k in keys)
    assert len({v.casefold() for v in mp["pseudo"].values()}) == len(keys)
    assert pl[0]["history"][0] == f"{mp['placebo']['mattel']} Barbie Doll (rated 5/5)"
    assert ps[1]["history"][0] == f"{mp['pseudo']['hasbro']} Monopoly (rated 5/5)"
    for arm in (ps, pl):
        assert "Mattel" not in arm[0]["history"][0] and "Hasbro" not in arm[1]["history"][0]
        assert arm[0]["history"][0].endswith("(rated 5/5)")
        assert arm[0]["candidate_titles"][2] == "Generic Ball"   # stop-listed store untouched
        assert [a["n_subst"] for a in arm] == [r["n_subst"] for r in ps]
    # placebo inserts another REAL brand
    real = {"Mattel", "LEGO", "Hape", "Hasbro"}
    assert pl[0]["history"][0].split(" ")[0] in real - {"Mattel"}
    assert ps[0]["candidate_brand_in_title"] == [True, True, False]
    assert rep["checks"] == {"placebo_keys_equal_pseudo_keys": True, "placebo_fixed_points": 0,
                             "pseudonyms_injective": True, "pseudonyms_equal_to_a_real_store": 0,
                             "rows_with_unequal_n_subst": 0, "treated_own_store_left_in_pseudo_title": 0}
    assert rep["popularity_source"] == "panel_occurrence"
    assert rep["stoplist"]["placeholder"]["strings"] == 1
    # n_subst counts every substituted span: u1 = 2 history + 2 titles + 2 texts
    assert ps[0]["n_subst"] == 6 and pl[0]["n_subst"] == 6


def test_placebo_swap_pair_in_one_row():
    # two substitutable brands -> the placebo derangement is the Mattel<->Hasbro 2-cycle
    rows = [_rec("u", ["Hasbro Monopoly"], ["Hasbro"], ["Hasbro Monopoly; Mattel UNO", "Mattel Barbie"],
                 ["Hasbro", "Mattel"])]
    ps, pl, rep = pseudonymize(rows)
    assert pl[0]["candidate_titles"] == ["Mattel Monopoly; Hasbro UNO", "Hasbro Barbie"]
    assert pl[0]["history"] == ["Mattel Monopoly (rated 5/5)"]
    assert "Hasbro" not in ps[0]["candidate_titles"][0] and "Mattel" not in ps[0]["candidate_titles"][0]
    assert ps[0]["n_subst"] == pl[0]["n_subst"] == 4


def test_outputs_independent_of_hash_seed(tmp_path):
    rows = [_rec(f"u{k}", ["Alpha Toys car", "Beta Games set"], ["Alpha Toys", "Beta Games"],
                 ["Gamma Kids doll", "Delta Co ball", "Alpha Toys truck", "Wonder Kite"],
                 ["Gamma Kids", "Delta Co", "Alpha Toys", "Wonder"], texts=[f"a wonder {k}", "", "", ""])
            for k in range(5)]
    p = tmp_path / "p.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    outs = []
    for hs in ("1", "2"):
        d = tmp_path / hs
        subprocess.run([sys.executable, "-m", "src.confrec.pseudonymize", "--panel", str(p), "--out_dir", str(d)],
                       check=True, capture_output=True, env={**__import__("os").environ, "PYTHONHASHSEED": hs})
        outs.append([(d / f"p_{a}.jsonl").read_text(encoding="utf-8") for a in ("pseudo", "placebo")]
                    + [(d / "p_pseudonym_report.json").read_text(encoding="utf-8")])
    assert outs[0] == outs[1]
    assert json.loads(outs[0][2])["common_word_stores"][0]["brand"] == "Wonder"


def test_treated_flag_whole_word_on_original_title():
    rec = _rec("u", [], [], ["Wooden Shape Sorter", "Hape Pound-a-Peg", "Party Set", "Ty Beanie"],
               ["Hape", "Hape", "Ty", "Ty"])
    assert treated_flags(rec, {"hape"}) == [False, True, False, False]   # 'Ty' is stop-listed (short)
    idx = panel_brands([rec])
    assert treated_flags(rec, set(idx["keys"]), idx["rx"]) == [False, True, False, False]


def test_cli_writes_panels_and_report(tmp_path):
    rows = [_rec("u", ["LEGO Star Wars", "Barbie Doll"], ["LEGO", "Mattel"],
                 ["LEGO Technic Car", "Generic Ball"], ["LEGO", ""], texts=["by LEGO", ""], ratings=[5, 2])]
    p = tmp_path / "toy_rated.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "toy_rated.brand_pop.json").write_text(json.dumps({"LEGO": 900, "Mattel": 500, "Other": 3}))
    subprocess.run([sys.executable, "-m", "src.confrec.pseudonymize", "--panel", str(p)], check=True,
                   capture_output=True)
    ps = json.loads((tmp_path / "toy_rated_pseudo.jsonl").read_text(encoding="utf-8"))
    pl = json.loads((tmp_path / "toy_rated_placebo.jsonl").read_text(encoding="utf-8"))
    rep = json.loads((tmp_path / "toy_rated_pseudonym_report.json").read_text(encoding="utf-8"))
    assert ps["candidate_brand_in_title"] == [True, False] == pl["candidate_brand_in_title"]
    assert "LEGO" not in ps["candidate_titles"][0] and ps["candidate_titles"][0].endswith(" Technic Car")
    assert ps["history"][1] == "Barbie Doll (rated 2/5)"         # 'Mattel' is not in the title text
    assert pl["candidate_titles"][0] == "Mattel Technic Car" and pl["history"][0] == "Mattel Star Wars (rated 5/5)"
    assert ps["candidate_titles"][1] == "Generic Ball" and ps["n_subst"] == pl["n_subst"] == 3
    assert rep["popularity_source"] == "sidecar" and rep["n_keys_missing_from_sidecar"] == 0
    assert rep["n_brands"] == 2 and rep["treated_share"] == 0.5
