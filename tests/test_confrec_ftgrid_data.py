"""src/confrec/ftgrid_data.py: the Amendment-3 data roles (section 1) and the FT-C permutation file (section 9).
Synthetic rated panels only; CPU, deterministic, no torch / transformers (the tokenizer is a fake)."""
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from src.confrec import ftgrid_data as fd
from src.confrec import gateft_data as gd
from src.confrec import prompting

ROOT = Path(__file__).resolve().parents[1]
LONG_TITLE = " ".join(["w"] * 99)          # 197 characters: under the 200-character title cap, 99 fake tokens


def mk_row(uid, cands, n_hist=12, source="ml1m", hist_title=None):
    """A build_rated_panels row (hist_len-20 fields). cands: [(item_id, ts, label, popularity)] in candidate order; the
    source_event_id ends in the first (smallest) candidate timestamp, as the builder writes it; a like is rated 4-5."""
    items, ts = [str(c[0]) for c in cands], [int(c[1]) for c in cands]
    labs = [int(c[2]) for c in cands]
    titles = [hist_title or f"Old {uid} {k}" for k in range(n_hist)]
    stars = [float(1 + k % 5) for k in range(n_hist)]
    return {"user_id": uid, "source_event_id": f"{uid}::{min(ts)}",
            "history": [f"{t} (rated {int(s)}/5)" for t, s in zip(titles, stars)],
            "history_item_ids": [f"h{k}" for k in range(n_hist)], "history_titles": titles, "history_ratings": stars,
            "history_brands": [""] * n_hist, "history_meta": [f"Genres: G{k}" for k in range(n_hist)],
            "candidate_item_ids": items, "candidate_titles": [f"Movie {i}" for i in items],
            "candidate_texts": [f"Genres: X{i}" for i in items], "candidate_brands": [""] * len(items),
            "candidate_ratings": [(4.0 + k % 2) if y else (1.0 + k % 2) for k, y in enumerate(labs)],
            "candidate_labels": labs, "candidate_popularity": [int(c[3]) for c in cands],
            "candidate_popularity_prior": [0] * len(items), "candidate_timestamps": ts,
            "source": source, "domain_kind": "movie" if source == "ml1m" else "product"}


def random_panel(n_users, n_cands=10, n_items=60, seed=0, prefix="u", **kw):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        items = rng.choice(n_items, n_cands, replace=False)
        ts = rng.integers(1_000, 11_000, n_cands)
        labs = (rng.random(n_cands) < 0.5).astype(int)
        labs[:2] = (1, 0)
        pops = rng.integers(1, 500, n_cands)
        rows.append(mk_row(f"{prefix}{u}", [(f"i{i}", t, y, p) for i, t, y, p in zip(items, ts, labs, pops)], **kw))
    return rows


def write_panel(path, rows):
    path.write_bytes(b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in rows))
    return path


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def run(tmp_path, rows, *args, domain="toys", name="o", tokenizer=None):
    panel = write_panel(tmp_path / f"{name}.panel.jsonl", rows)
    out = tmp_path / name
    rep = fd.main(["--domain", domain, "--panel_all", str(panel), "--out_dir", str(out), *map(str, args)],
                  tokenizer=tokenizer)
    return rep, out


class WordTok:
    """Fake tokenizer: one id per whitespace-separated word of the templated chat; remembers every text it tokenised."""
    chat_template = "fake"                  # no enable_thinking: chat_ids skips the think-block check

    def __init__(self):
        self.texts = []

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, **kw):
        return "".join(f"<{m['role']}> {m['content']} </{m['role']}> " for m in msg) + "<assistant>"

    def __call__(self, text, add_special_tokens=False):
        self.texts.append(text)
        return {"input_ids": list(range(len(text.split())))}


# ---------------------------------------------------------------- roles, T, TRAIN


def test_train_and_eval_are_row_blocks_and_must_not_share_users(tmp_path):
    rows = random_panel(12)
    rep, out = run(tmp_path, rows, "--n_train", 5, "--n_eval_max", 4)
    assert rep["n_rows"] == 12 and rep["train"]["users"] == 5 and rep["eval"]["users"] == 4
    assert (out / "train_users.txt").read_text(encoding="utf-8") == "\n".join(r["user_id"] for r in rows[:5])
    assert (out / "eval_users.txt").read_text(encoding="utf-8") == "\n".join(r["user_id"] for r in rows[5:9])
    lines = (tmp_path / "o.panel.jsonl").read_bytes().splitlines(keepends=True)
    assert (out / "eval.jsonl").read_bytes() == b"".join(lines[5:9])            # EVAL rows byte for byte
    ts = np.concatenate([r["candidate_timestamps"] for r in rows[:9]])           # rows after EVAL do not enter T
    assert rep["T"] == float(np.quantile(ts.astype(float), 0.8))
    clash = [dict(r) for r in rows]
    clash[7]["user_id"] = clash[2]["user_id"]                                    # a TRAIN user again in EVAL
    with pytest.raises(SystemExit, match="share"):
        run(tmp_path, clash, "--n_train", 5, name="c1")
    twice = [dict(r) for r in rows]
    twice[8]["user_id"] = twice[6]["user_id"]
    with pytest.raises(SystemExit, match="twice"):
        run(tmp_path, twice, "--n_train", 5, name="c2")
    with pytest.raises(SystemExit, match="no EVAL row"):
        run(tmp_path, rows, "--n_train", 12, name="c3")
    assert not any((tmp_path / c).exists() for c in ("c1", "c2", "c3"))           # a refusal writes nothing


def test_T_and_the_train_file_equal_gate_ft_on_the_same_inputs(tmp_path):
    dev, conf = random_panel(30, prefix="d", seed=1), random_panel(20, prefix="c", seed=2)
    dp, cp = write_panel(tmp_path / "dev.jsonl", dev), write_panel(tmp_path / "conf.jsonl", conf)
    g = gd.main(["--dev_panel", str(dp), "--confirm_panel", str(cp), "--out_dir", str(tmp_path / "g")])
    gsplit = tmp_path / "g" / "gateft_split.json"
    dev_sha = sha1("\n".join(sorted(r["user_id"] for r in dev)).encode("utf-8"))   # build_confirm_panels' DEV list
    args = ("--n_train", 30, "--gateft_split", gsplit, "--ml1m_dev_users_sha1", dev_sha)
    rep, out = run(tmp_path, dev + conf, *args, domain="ml1m")
    assert rep["T"] == g["T"] == json.loads(gsplit.read_text(encoding="utf-8"))["T"]
    assert rep["gateft_T_match"] is True and rep["eval"]["users"] == 20
    assert (out / "train.jsonl").read_bytes() == (tmp_path / "g" / "train.jsonl").read_bytes()   # the G9 set
    assert rep["train"]["examples_pre_T"] == rep["train"]["examples_written"] == g["train"]["candidates"]

    def bad_split(name, **change):
        s = json.loads(gsplit.read_text(encoding="utf-8"))
        for k, v in change.items():
            if k == "train_sha1":
                s["train"]["sha1"] = v
            else:
                s[k] = v
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(s), encoding="utf-8")
        return p

    with pytest.raises(SystemExit, match="Gate-FT's T"):
        run(tmp_path, dev + conf, "--n_train", 30, "--gateft_split", bad_split("t", T=g["T"] + 1), domain="ml1m",
            name="b1")
    assert list((tmp_path / "b1").iterdir()) == []                               # staged files removed
    with pytest.raises(SystemExit, match="G9"):
        run(tmp_path, dev + conf, "--n_train", 30, "--gateft_split", bad_split("s", train_sha1="0" * 40),
            domain="ml1m", name="b2")
    with pytest.raises(SystemExit, match="DEV list"):
        run(tmp_path, dev + conf, "--n_train", 30, "--ml1m_dev_users_sha1", "0" * 40, domain="ml1m", name="b3")
    with pytest.raises(SystemExit, match="DEV list"):                            # one TRAIN user off
        run(tmp_path, dev + conf, "--n_train", 29, "--ml1m_dev_users_sha1", dev_sha, domain="ml1m", name="b4")
    with pytest.raises(SystemExit):                                              # Gate-FT's split is ML-1M's
        run(tmp_path, dev + conf, "--n_train", 30, "--gateft_split", gsplit, domain="toys", name="b5")


def test_train_keeps_only_pre_T_candidates_with_aligned_lists(tmp_path):
    rows = random_panel(40)
    rows[3] = mk_row("late", [(f"i{k}", 20_000 + k, k % 2, 10) for k in range(10)])   # every candidate after T
    rep, out = run(tmp_path, rows, "--n_train", 25)
    T = rep["T"]
    assert T == float(np.quantile(np.concatenate([r["candidate_timestamps"] for r in rows]).astype(float), 0.8))
    by_user = {r["user_id"]: r for r in rows[:25]}
    keep = {u: [j for j, t in enumerate(r["candidate_timestamps"]) if t < T] for u, r in by_user.items()}
    train = jsonl(out / "train.jsonl")
    assert [r["user_id"] for r in train] == [u for u in by_user if keep[u]] and "late" not in {r["user_id"] for r in train}
    for r in train:
        src, kj = by_user[r["user_id"]], keep[r["user_id"]]
        assert list(r) == list(src)
        for k, v in src.items():
            assert r[k] == ([v[j] for j in kj] if k.startswith("candidate_") else v)   # lists filtered together
        assert all(t < T for t in r["candidate_timestamps"])
    n_pre = sum(map(len, keep.values()))
    assert rep["train"]["examples_pre_T"] == rep["train"]["examples_written"] == n_pre
    assert rep["train"]["candidates_total"] == sum(len(r["candidate_labels"]) for r in rows[:25])
    assert rep["train"]["cap_applied"] is False and rep["train"]["cap"] == 24000


def test_history_must_precede_every_candidate_and_train_must_precede_T(tmp_path):
    rows = random_panel(10)
    bad = [dict(r) for r in rows]
    bad[1]["source_event_id"] = f"{bad[1]['user_id']}::{max(bad[1]['candidate_timestamps'])}"
    with pytest.raises(SystemExit, match="history"):
        run(tmp_path, bad, "--n_train", 5, name="h1")
    bad[1]["source_event_id"] = "no-timestamp"
    with pytest.raises(SystemExit, match="source_event_id"):
        run(tmp_path, bad, "--n_train", 5, name="h2")
    late = [mk_row(f"t{u}", [(f"i{k}", 5_000 + k, k % 2, 3) for k in range(10)]) for u in range(2)]
    early = [mk_row(f"e{u}", [(f"i{k}", 10 + k, k % 2, 3) for k in range(10)]) for u in range(10)]
    with pytest.raises(SystemExit, match="precedes T"):                          # no TRAIN example at all
        run(tmp_path, late + early, "--n_train", 2, name="h3")


def test_cap_keeps_a_seeded_uniform_subset(tmp_path):
    rows = random_panel(40)
    full, o_full = run(tmp_path, rows, "--n_train", 30, name="full")
    n_pre = full["train"]["examples_pre_T"]
    assert n_pre > 150 and full["train"]["cap_applied"] is False
    rep, out = run(tmp_path, rows, "--n_train", 30, "--train_cap", 50, name="cap")
    t = rep["train"]
    assert t["cap_applied"] is True and t["examples_written"] == 50 and t["cap"] == 50 and t["examples_pre_T"] == n_pre

    def examples(path):
        return [(r["user_id"], i, ts, y) for r in jsonl(path) for i, ts, y in
                zip(r["candidate_item_ids"], r["candidate_timestamps"], r["candidate_labels"])]

    pre, kept = examples(o_full / "train.jsonl"), examples(out / "train.jsonl")
    assert len(set(kept)) == 50 and set(kept) <= set(pre)
    assert kept == [e for e in pre if e in set(kept)]                            # row and candidate order kept
    assert all(r["candidate_labels"] for r in jsonl(out / "train.jsonl"))         # no row left empty
    assert len({e[0] for e in kept}) > 15                                        # spread over users, not a prefix
    _, again = run(tmp_path, rows, "--n_train", 30, "--train_cap", 50, name="cap2")
    assert (again / "train.jsonl").read_bytes() == (out / "train.jsonl").read_bytes()
    _, other = run(tmp_path, rows, "--n_train", 30, "--train_cap", 50, "--seed", 1, name="cap3")
    assert (other / "train.jsonl").read_bytes() != (out / "train.jsonl").read_bytes()
    exact, o_exact = run(tmp_path, rows, "--n_train", 30, "--train_cap", n_pre, name="cap4")
    assert exact["train"]["cap_applied"] is False
    assert (o_exact / "train.jsonl").read_bytes() == (o_full / "train.jsonl").read_bytes()


# ---------------------------------------------------------------- power facts and S_d


def toy_panel():
    """2 TRAIN + 4 EVAL users x 4 candidates; the timestamps are 1..24, so T = 12.5 at quantile 0.5. Candidate order is
    deliberately not chronological."""
    return [mk_row("A", [("i1", 1, 1, 5), ("i2", 2, 0, 6), ("i3", 13, 1, 7), ("i4", 14, 0, 8)], source="toys"),
            mk_row("B", [("i5", 3, 0, 9), ("i6", 4, 1, 10), ("i1", 5, 1, 11), ("i7", 15, 0, 12)], source="toys"),
            mk_row("C", [("i3", 17, 0, 40), ("i1", 6, 1, 41), ("i8", 18, 1, 42), ("i2", 16, 1, 2)], source="toys"),
            mk_row("D", [("i5", 19, 1, 43), ("i11", 7, 1, 44), ("i12", 8, 1, 45), ("i9", 20, 1, 46)], source="toys"),
            mk_row("E", [("i13", 9, 1, 47), ("i14", 10, 0, 48), ("i15", 11, 1, 49), ("i1", 21, 0, 50)], source="toys"),
            mk_row("F", [("i10", 22, 0, 1), ("i6", 23, 1, 3), ("i4", 24, 0, 51), ("i16", 12, 1, 52)], source="toys")]


def test_power_facts_and_S_d_on_a_hand_computed_toy(tmp_path):
    rep, out = run(tmp_path, toy_panel(), "--n_train", 2, "--quantile", 0.5)
    assert rep["T"] == 12.5
    assert {k: rep["train"][k] for k in ("users", "candidates_total", "examples_pre_T", "examples_written")} == \
        {"users": 2, "candidates_total": 8, "examples_pre_T": 5, "examples_written": 5}
    assert [(r["user_id"], r["candidate_item_ids"]) for r in jsonl(out / "train.jsonl")] == \
        [("A", ["i1", "i2"]), ("B", ["i5", "i6", "i1"])]
    e = rep["eval"]
    # CAL: C 6 | D 7, 8 | E 9, 10, 11 | F 12;  TEST: C 17, 18, 16 | D 19, 20 | E 21 | F 22, 23, 24
    assert (e["users"], e["candidates"], e["cal_candidates"], e["test_candidates"]) == (4, 16, 7, 9)
    # both classes among TEST: C (0, 1, 1) and F (0, 1, 0); D is all likes, E has one TEST candidate
    assert e["users_both_classes_test"] == 2
    # tail = rank_bins(popularity of all 16 EVAL candidates, 5) == 0 = the 3 least popular: F22 (0), C16 (1), F23 (1);
    # only F has both classes there (binning the 9 TEST candidates alone would give F22, C16 and no such user)
    assert e["users_both_classes_test_tail"] == 1
    assert e["users_both_classes_all_rows"] == 3                                 # all but D
    # items with a TRAIN example: i1, i2, i5, i6 (i3, i4 are TRAIN users' post-T candidates, not examples)
    assert e["share_test_pairs_item_seen_in_train"] == pytest.approx(4 / 9)
    assert rep["sd"]["users"] == 2 and (out / "sd_users.txt").read_text(encoding="utf-8") == "C\nF"
    sd = jsonl(out / "eval_sd_test.jsonl")
    assert [(r["user_id"], r["candidate_item_ids"], r["candidate_timestamps"], r["candidate_labels"],
             r["candidate_popularity"]) for r in sd] == [("C", ["i3", "i8", "i2"], [17, 18, 16], [0, 1, 1], [40, 42, 2]),
                                                          ("F", ["i10", "i6", "i4"], [22, 23, 24], [0, 1, 0], [1, 3, 51])]
    assert sd[0]["history"] == toy_panel()[2]["history"]
    one, o1 = run(tmp_path, toy_panel(), "--n_train", 2, "--quantile", 0.5, "--s_max", 1, name="s1")
    assert one["sd"]["users"] == 1 and (o1 / "sd_users.txt").read_text(encoding="utf-8") == "C"
    assert [r["user_id"] for r in jsonl(o1 / "eval_sd_test.jsonl")] == ["C"]
    # the seen share counts the examples written: after a cap of 2 only their items are seen
    cap, oc = run(tmp_path, toy_panel(), "--n_train", 2, "--quantile", 0.5, "--train_cap", 2, name="cap")
    seen = {i for r in jsonl(oc / "train.jsonl") for i in r["candidate_item_ids"]}
    test_items = ["i3", "i8", "i2", "i5", "i9", "i1", "i10", "i6", "i4"]
    assert cap["eval"]["share_test_pairs_item_seen_in_train"] == pytest.approx(
        sum(i in seen for i in test_items) / 9)


# ---------------------------------------------------------------- FT-C permutation (section 9)


def test_ftc_permutation_preserves_every_item_label_sum(tmp_path):
    rows = random_panel(40, n_cands=8, n_items=12, seed=3)
    j0 = int(np.argmin(rows[0]["candidate_timestamps"]))
    rows[0]["candidate_item_ids"][j0] = "solo"                                   # an item with one TRAIN example
    rep, out = run(tmp_path, rows, "--n_train", 30, domain="ml1m")
    train, perm = jsonl(out / "train.jsonl"), jsonl(out / "train_perm.jsonl")
    assert len(train) == len(perm) and rep["train"]["examples_written"] > 150
    sums, ratings, n_ex, changed = defaultdict(lambda: [0, 0]), defaultdict(lambda: [[], []]), Counter(), 0
    for r, p in zip(train, perm):
        assert list(r) == list(p)
        assert all(r[k] == p[k] for k in r if k not in ("candidate_labels", "candidate_ratings"))
        assert len(p["candidate_labels"]) == len(p["candidate_ratings"]) == len(r["candidate_labels"])
        for i, y, z, a, b in zip(r["candidate_item_ids"], r["candidate_labels"], p["candidate_labels"],
                                 r["candidate_ratings"], p["candidate_ratings"]):
            sums[i][0] += y
            sums[i][1] += z
            ratings[i][0].append(a)
            ratings[i][1].append(b)
            n_ex[i] += 1
            changed += y != z
            assert z == int(b >= 4)                                              # the star rating moved with its label
    assert all(a == b for a, b in sums.values())                                 # every item's label sum preserved
    assert all(sorted(a) == sorted(b) for a, b in ratings.values())
    assert changed > 0
    assert n_ex["solo"] == 1 and sums["solo"][0] == sums["solo"][1]
    assert rep["files"]["train_perm.jsonl"] == sha1((out / "train_perm.jsonl").read_bytes())
    _, again = run(tmp_path, rows, "--n_train", 30, domain="ml1m", name="again")
    assert (again / "train_perm.jsonl").read_bytes() == (out / "train_perm.jsonl").read_bytes()
    toys, o_toys = run(tmp_path, rows, "--n_train", 30, domain="toys", name="toys")
    assert not (o_toys / "train_perm.jsonl").exists() and "train_perm.jsonl" not in toys["files"]
    run(tmp_path, rows, "--n_train", 30, domain="toys", name="again")            # a stale permutation file is removed
    assert not (again / "train_perm.jsonl").exists()


def test_permutation_function_keeps_singletons_and_label_sums():
    rows = [mk_row(f"u{k}", [("a", 1, k % 2, 1), ("b", 2, 1, 1), (f"only{k}", 3, k % 2, 1)]) for k in range(6)]
    perm, n_changed = fd.permute_within_item(rows, 0)
    for k, (r, p) in enumerate(zip(rows, perm)):
        assert p["candidate_labels"][2] == r["candidate_labels"][2]              # single-example items unchanged
        assert p["candidate_labels"][1] == 1                                     # an all-like item stays all-like
    assert sum(p["candidate_labels"][0] for p in perm) == 3 and rows[0]["candidate_labels"] == [0, 1, 0]
    assert n_changed == sum(r["candidate_labels"][0] != p["candidate_labels"][0] for r, p in zip(rows, perm))
    assert fd.permute_within_item(rows, 0)[0] == perm                            # seeded


# ---------------------------------------------------------------- TRAIN length rule


@pytest.mark.parametrize("lengths, share, max_len, shape", [
    ([100] * 98 + [1025] * 2, 0.02, 1024, (8, 4)),        # exactly 2% above: G9's 1,024 with skip-and-count
    ([100] * 97 + [1025] * 3, 0.03, 1280, (4, 8)),        # above 2%: p99.5 = 1025 rounded up to 1280
    ([100] * 90 + [1280] * 10, 0.10, 1280, (4, 8)),       # p99.5 already a multiple of 256
    ([100] * 90 + [1281] * 10, 0.10, 1536, (4, 8)),
    ([1024] * 50, 0.0, 1024, (8, 4)),                     # 1,024 itself is not above 1,024
])
def test_length_rule_branches(lengths, share, max_len, shape):
    r = fd.length_rule(lengths)
    assert r["share_above_1024"] == pytest.approx(share) and r["max_len_used"] == max_len
    assert r["p995"] == pytest.approx(float(np.percentile(lengths, 99.5))) and r["max_len_registered"] == 1024
    assert (r["micro_bsz"], r["grad_accum"]) == shape and r["micro_bsz"] * r["grad_accum"] == 32


def test_length_rule_measures_every_train_like_prompt(tmp_path):
    rows = random_panel(30)
    tok = WordTok()
    rep, _ = run(tmp_path, rows, "--n_train", 20, tokenizer=tok)
    ov = rep["train"]["overlength"]
    assert len(tok.texts) == rep["train"]["examples_written"]                    # one `like` prompt per example
    assert all(prompting.QUESTIONS["like"] in t and "dislike" not in t for t in tok.texts)
    lengths = [len(t.split()) + 1 for t in tok.texts]                            # + the answer token
    assert ov["share_above_1024"] == 0.0 and ov["p995"] == pytest.approx(float(np.percentile(lengths, 99.5)))
    assert (ov["max_len_used"], ov["micro_bsz"], ov["grad_accum"]) == (1024, 8, 4)

    for r in rows[:2]:                                                           # two TRAIN users with long histories
        r["history_titles"] = [LONG_TITLE] * len(r["history"])
        r["history"] = [f"{LONG_TITLE} (rated {int(s)}/5)" for s in r["history_ratings"]]
    tok = WordTok()
    rep, _ = run(tmp_path, rows, "--n_train", 20, tokenizer=tok, name="long")
    ov = rep["train"]["overlength"]
    lengths = [len(t.split()) + 1 for t in tok.texts]
    share = sum(x > 1024 for x in lengths) / len(lengths)
    p995 = float(np.percentile(lengths, 99.5))
    assert share > 0.02 and ov["share_above_1024"] == pytest.approx(share) and ov["p995"] == pytest.approx(p995)
    assert ov["max_len_used"] == 256 * math.ceil(p995 / 256) > 1024 and (ov["micro_bsz"], ov["grad_accum"]) == (4, 8)

    rep, _ = run(tmp_path, rows, "--n_train", 20, name="notok")                  # no tokenizer: recorded as null
    assert rep["train"]["overlength"] == {"max_len_registered": 1024, "share_above_1024": None, "p995": None,
                                          "max_len_used": 1024, "micro_bsz": 8, "grad_accum": 4}


def test_length_rule_renders_the_selected_variant(tmp_path, monkeypatch):
    rows = random_panel(20)                                                      # 12 history events each
    tok = WordTok()
    rep, _ = run(tmp_path, rows, "--n_train", 12, "--variant", "V7", tokenizer=tok)
    assert rep["variant"] == "V7" and len(tok.texts) == rep["train"]["examples_written"]
    assert all(prompting.THRESHOLD_QUESTIONS["like"] in t and prompting.SYSTEM_MESSAGES["rated_movie"] in t
               for t in tok.texts)
    with pytest.raises(SystemExit, match="hist_len 20"):                         # V7's 20-event window must exist
        run(tmp_path, random_panel(20, n_hist=10), "--n_train", 12, "--variant", "V7", name="short")
    calls = []                                                                   # --tokenizer is loaded on demand
    monkeypatch.setattr(fd, "load_tokenizer", lambda path: calls.append(path) or WordTok())
    rep, _ = run(tmp_path, rows, "--n_train", 12, "--tokenizer", "models/Some-8B", name="cli_tok")
    assert calls == ["models/Some-8B"] and rep["train"]["overlength"]["share_above_1024"] == 0.0


# ---------------------------------------------------------------- the split json and determinism

SCHEMA = {
    None: ["domain", "panel_all_sha1", "n_rows", "variant", "quantile", "T", "train", "eval", "sd", "files", "code_sha1",
           "args", "gateft_T_match"],
    "train": ["users", "user_ids_sha1", "candidates_total", "examples_pre_T", "cap", "cap_applied", "examples_written",
              "overlength"],
    "eval": ["users", "user_ids_sha1", "candidates", "cal_candidates", "test_candidates", "users_both_classes_test",
             "users_both_classes_test_tail", "users_both_classes_all_rows", "share_test_pairs_item_seen_in_train"],
    "sd": ["users", "user_ids_sha1", "user_ids_path"],
}


def test_split_json_follows_the_spec_schema(tmp_path):
    rows = random_panel(30)
    rep, out = run(tmp_path, rows, "--n_train", 20)
    text = (out / "ftgrid_split.json").read_text(encoding="utf-8")
    js = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    assert js == rep and list(js) == SCHEMA[None]
    assert all(list(js[k]) == SCHEMA[k] for k in ("train", "eval", "sd"))
    assert list(js["train"]["overlength"]) == ["max_len_registered", "share_above_1024", "p995", "max_len_used",
                                               "micro_bsz", "grad_accum"]
    assert (js["domain"], js["variant"], js["quantile"], js["gateft_T_match"]) == ("toys", "V0", 0.8, None)
    assert js["args"] == {"seed": 0, "n_train": 20, "n_eval_max": 3000, "s_max": 1000, "train_cap": 24000,
                          "tokenizer_used": False, "tokenizer": None, "dev_users_sha1_checked": False,
                          "gateft_split_checked": False}
    assert js["panel_all_sha1"] == sha1((tmp_path / "o.panel.jsonl").read_bytes()) and js["n_rows"] == 30
    assert js["code_sha1"] == {"ftgrid_data.py": sha1((ROOT / "src/confrec/ftgrid_data.py").read_bytes()),
                               "gateft_data.py": sha1((ROOT / "src/confrec/gateft_data.py").read_bytes()),
                               "build_rated_panels.py": sha1((ROOT / "src/confrec/build_rated_panels.py").read_bytes())}
    assert sorted(js["files"]) == sorted(p.name for p in out.iterdir() if p.name != "ftgrid_split.json") == \
        ["eval.jsonl", "eval_sd_test.jsonl", "eval_users.txt", "sd_users.txt", "train.jsonl", "train_users.txt"]
    assert all(sha1((out / n).read_bytes()) == s for n, s in js["files"].items())
    for part, name in (("train", "train_users.txt"), ("eval", "eval_users.txt"), ("sd", "sd_users.txt")):
        b = (out / name).read_bytes()
        assert not b.endswith(b"\n") and sha1(b) == js[part]["user_ids_sha1"] and len(b.split(b"\n")) == js[part]["users"]
    assert js["sd"]["user_ids_path"] == "sd_users.txt"
    sd = jsonl(out / "eval_sd_test.jsonl")
    assert [r["user_id"] for r in sd] == (out / "sd_users.txt").read_text(encoding="utf-8").split("\n")
    assert all(t >= js["T"] for r in sd for t in r["candidate_timestamps"])      # TEST rows only
    assert js["sd"]["users"] == min(1000, js["eval"]["users_both_classes_test"])


def test_outputs_are_byte_identical_on_a_rerun_and_carry_no_path(tmp_path):
    rows = random_panel(30)
    rep1, o1 = run(tmp_path, rows, "--n_train", 20, domain="ml1m", name="first")
    rep2, o2 = run(tmp_path, rows, "--n_train", 20, domain="ml1m", name="second")
    names = sorted(p.name for p in o1.iterdir())
    assert names == sorted(p.name for p in o2.iterdir()) == sorted([*rep1["files"], "ftgrid_split.json"])
    assert all((o1 / n).read_bytes() == (o2 / n).read_bytes() for n in names) and rep1 == rep2
    text = (o1 / "ftgrid_split.json").read_text(encoding="utf-8")
    for s in (str(tmp_path), tmp_path.as_posix(), "first", "panel.jsonl", "\\\\"):
        assert s not in text
    before = {n: (o1 / n).stat().st_mtime_ns for n in names}
    rep3, _ = run(tmp_path, rows, "--n_train", 20, domain="ml1m", name="first")  # rerun into the same directory
    assert rep3 == rep1 and {n: (o1 / n).stat().st_mtime_ns for n in names} == before   # unchanged files untouched
    assert not list(o1.glob("*.tmp"))


def test_cli_matches_main(tmp_path):
    rows = random_panel(15)
    rep, _ = run(tmp_path, rows, "--n_train", 8)
    out = tmp_path / "cli"
    res = subprocess.run([sys.executable, "-m", "src.confrec.ftgrid_data", "--domain", "toys", "--panel_all",
                          str(tmp_path / "o.panel.jsonl"), "--out_dir", str(out), "--n_train", "8"],
                         cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    assert json.loads((out / "ftgrid_split.json").read_text(encoding="utf-8")) == rep


def test_module_import_needs_no_torch_or_transformers():
    code = ("import sys; import src.confrec.ftgrid_data; "
            "print([m for m in ('torch', 'transformers') if m in sys.modules])")
    res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "[]"


def test_bad_arguments_and_rows_are_refused(tmp_path):
    rows = random_panel(12)
    for k, args in enumerate((["--variant", "V6"], ["--quantile", "1.0"], ["--n_train", "0"], ["--s_max", "0"],
                              ["--train_cap", "0"], ["--domain", "../x"])):
        with pytest.raises(SystemExit):
            run(tmp_path, rows, "--n_train", 5, *args, name=f"a{k}")
    mis = [dict(r) for r in rows]
    mis[6] = dict(mis[6], candidate_titles=mis[6]["candidate_titles"][:-1])
    with pytest.raises(SystemExit, match="misaligned"):
        run(tmp_path, mis, "--n_train", 5, name="m1")
    lab = [dict(r) for r in rows]
    lab[2] = dict(lab[2], candidate_labels=[2] + lab[2]["candidate_labels"][1:])
    with pytest.raises(SystemExit, match="0/1"):
        run(tmp_path, lab, "--n_train", 5, name="m2")
    nxt = [dict(r) for r in rows]
    nxt[0] = {k: v for k, v in nxt[0].items() if k != "history_ratings"}        # a next-item row
    with pytest.raises(SystemExit, match="rated"):
        run(tmp_path, nxt, "--n_train", 5, name="m3")
    nopop = [dict(r) for r in rows]
    nopop[7] = {k: v for k, v in nopop[7].items() if k != "candidate_popularity"}
    with pytest.raises(SystemExit, match="candidate_popularity"):
        run(tmp_path, nopop, "--n_train", 5, name="m4")
    with pytest.raises(SystemExit):                                              # missing inputs fail before any work
        fd.main(["--domain", "toys", "--panel_all", str(tmp_path / "none.jsonl"), "--out_dir", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        run(tmp_path, rows, "--n_train", 5, "--gateft_split", tmp_path / "none.json", domain="ml1m", name="m5")
    assert not (tmp_path / "x").exists() and not (tmp_path / "m5").exists()
