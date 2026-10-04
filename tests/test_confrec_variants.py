"""Amendment-2 prompt bank (prompting.VARIANTS): V0 byte identity with the Pilot-1 prompts (real pilot panels, incl.
rows that carry 20 history events), one hand-written golden prompt per variant and panel kind, the registered strings
against the variants JSON, candidate_extras, history-window rules and the digit readout math.

The Pilot-1 panels and reports are searched in order: $LUMEN_PILOT_PANELS (panels, with <panel>.report.json next to
them or ../pilot1_mirror/<panel>/report.json), the repo's outputs/confrec/panels with
outputs/confrec/pilot1_mirror/<panel>/report.json (the GPU-server layout of run_pilot1_mirror.sh), then the local
read-only scratch copy; the identity tests skip only when no location holds both the panel and its report."""
import hashlib
import json
import math
import os
from collections import namedtuple
from pathlib import Path

import pytest

from src.confrec import prompting as pr
from src.confrec import pyes_scorer as ps
from src.confrec.prompting import (GATE_VARIANTS, VARIANTS, build_prompt, digit_ids, read_digits, read_digits_full,
                                   render, render_record, resolve_hist_len)

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("D:/_Organized/Temp-Review/_RootDirs/temp/claude/D--/22c9b1b2-12a5-4956-a490-79bcb9beb0d8/scratchpad")
SPEC_JSON = [ROOT / "idea-stage" / "deliberation_2026-10-02" / "gatefix_prompt_variants_v1.json",
             SCRATCH / "deliberation" / "gatefix_prompt_variants_v1.json"]
PILOT_QS = ["like", "dislike", "like_para"]
LP = namedtuple("LP", "logprob")


def _pilot_dirs():
    env = os.environ.get("LUMEN_PILOT_PANELS")
    return ([Path(env)] if env else []) + [ROOT / "outputs" / "confrec" / "panels", SCRATCH / "pilot_panels"]


def _pilot_paths(name):
    """(panel, report) of the first location holding both; skip when none does."""
    stem = name[:-len(".jsonl")]
    for d in _pilot_dirs():
        reps = [d / f"{stem}.report.json", d.parent / "pilot1_mirror" / stem / "report.json"]
        rep = next((r for r in reps if r.is_file()), None)
        if (d / name).is_file() and rep is not None:
            return d / name, rep
    pytest.skip(f"no Pilot-1 panel {name} with its report.json in {[str(d) for d in _pilot_dirs()]} "
                "(set LUMEN_PILOT_PANELS)")


def _pilot(name):
    p, rep_path = _pilot_paths(name)
    rep = json.loads(rep_path.read_text("utf-8"))
    assert ps.file_sha1(p) == rep["config"]["data_sha1"], f"{p} is not the panel {rep_path} scored"
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()], rep


def test_pilot_locations_are_searched_in_order(tmp_path, monkeypatch):
    """Server layout (outputs/confrec/pilot1_mirror/<panel>/report.json) and $LUMEN_PILOT_PANELS are found; a panel
    without its report is passed over, not read unguarded."""
    srv = tmp_path / "outputs" / "confrec"
    (srv / "panels").mkdir(parents=True)
    (srv / "panels" / "x_rated.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("LUMEN_PILOT_PANELS", str(srv / "panels"))
    monkeypatch.setitem(globals(), "SCRATCH", tmp_path / "nowhere")
    with pytest.raises(pytest.skip.Exception, match="LUMEN_PILOT_PANELS"):
        _pilot_paths("x_rated.jsonl")                                    # panel present, report missing
    (srv / "pilot1_mirror" / "x_rated").mkdir(parents=True)
    (srv / "pilot1_mirror" / "x_rated" / "report.json").write_text("{}", encoding="utf-8")
    assert _pilot_paths("x_rated.jsonl") == (srv / "panels" / "x_rated.jsonl",
                                             srv / "pilot1_mirror" / "x_rated" / "report.json")
    (srv / "panels" / "x_rated.report.json").write_text("{}", encoding="utf-8")   # a sibling report wins
    assert _pilot_paths("x_rated.jsonl")[1] == srv / "panels" / "x_rated.report.json"


def _hash(records, questions, *a, **k):
    return ps.prompts_sha1(p for r in records for *_, p in ps.record_requests(r, questions, *a, **k))


# ---------------------------------------------------------------- (a) V0 = the Pilot-1 prompts, byte for byte
def test_v0_reproduces_pilot_ml1m_prompts_sha1():
    recs, rep = _pilot("ml1m_rated.jsonl")
    assert rep["config"]["prompts_sha1"] == "12e83c4fcca40db398ac58acbba2a7eb9c770524"
    assert rep["config"]["questions"] == PILOT_QS and rep["config"]["hist_len"] == 10
    assert _hash(recs, PILOT_QS, 10) == "12e83c4fcca40db398ac58acbba2a7eb9c770524"          # contract form
    assert _hash(recs, PILOT_QS) == "12e83c4fcca40db398ac58acbba2a7eb9c770524"              # registered window
    assert _hash(recs, PILOT_QS, None, "V0", "rated") == "12e83c4fcca40db398ac58acbba2a7eb9c770524"


def test_v0_reproduces_pilot_toys_and_sports_prompts_sha1():
    recs, rep = _pilot("toys_rated.jsonl")
    assert _hash(recs, PILOT_QS) == rep["config"]["prompts_sha1"] == "cd47bdb8038b7bc8d5e62064a92988e4a4545fe8"
    recs, rep = _pilot("sports_next_1k.jsonl")     # next-item V0 registers the last 5 events (the pilot's --hist_len)
    qs = ["next", "like", "dislike", "like_para"]
    assert rep["config"]["questions"] == qs and rep["config"]["hist_len"] == 5
    assert {pr.panel_kind_of(r) for r in recs} == {"next_item"}
    assert _hash(recs, qs) == rep["config"]["prompts_sha1"] == "81a2106b0cc22f5f47c15d838be4a710ec7c1088"


def test_scorer_config_on_pilot_ml1m_matches_the_pilot_report(tmp_path, capsys):
    _, rep = _pilot("ml1m_rated.jsonl")
    panel, _ = _pilot_paths("ml1m_rated.jsonl")
    args = ps.parse_args(["--data", str(panel), "--output", str(tmp_path / "_never_written"),
                          "--model", rep["config"]["model"], "--questions", "like,dislike,like_para",
                          "--swap_k", "8", "--dry_run"])
    out = ps.run(args, load_model=None)
    assert not (tmp_path / "_never_written").exists()
    for k in ("data_sha1", "n_records", "questions", "hist_len", "swap_k", "seed", "dtype", "topk_logprobs",
              "prompts_sha1", "swap_prompts_sha1", "scorer"):
        assert out["config"][k] == rep["config"][k], k
    assert out["config"]["max_model_len"] == 4096 and out["variant"] == "V0" and out["system_sha1"] is None
    assert "unregistered" not in out["config"] and out["prompt_strings_sha1"] == pr.PROMPT_STRINGS_SHA1


# ---------------------------------------------------------------- (b) rows that carry 20 history events
def _with_h20(recs):
    """The amendment-2 rebuild keeps evs[:first candidate][-20:]: a Pilot-1 row with exactly 10 events may gain up
    to 10 older ones; a row with fewer had no older events. Prepend 10 dummy older events to every truncated row
    (all aligned history fields) and add history_meta / domain_kind as build_rated_panels now writes them."""
    out = []
    for r in recs:
        r = dict(r)
        if len(r["history"]) == 10:
            old = [(f"Dummy Older Movie {k} (19{50 + k})", float(k % 5 + 1)) for k in range(10)]
            r["history"] = [f"{t} (rated {int(x)}/5)" for t, x in old] + r["history"]
            r["history_item_ids"] = [f"dummy{k}" for k in range(10)] + r["history_item_ids"]
            r["history_titles"] = [t for t, _ in old] + r["history_titles"]
            r["history_ratings"] = [x for _, x in old] + r["history_ratings"]
            r["history_brands"] = [""] * 10 + r["history_brands"]
        r["history_meta"] = [f"Genres: G{k % 7}" for k in range(len(r["history"]))]
        r["domain_kind"] = "movie"
        out.append(r)
    return out


def test_v0_on_rows_with_20_history_events_still_reproduces_pilot_hash():
    recs, _ = _pilot("ml1m_rated.jsonl")
    h20 = _with_h20(recs)
    assert sum(len(r["history"]) == 20 for r in h20) == sum(len(r["history"]) == 10 for r in recs) > 1000
    assert _hash(h20, PILOT_QS) == "12e83c4fcca40db398ac58acbba2a7eb9c770524"
    assert _hash(h20, PILOT_QS, 10) == "12e83c4fcca40db398ac58acbba2a7eb9c770524"
    # V3 does use the 20 events (and differs), V0 with hist_len 20 would too
    assert _hash(h20, ["like"], None, "V3") != _hash(h20, ["like"])
    r = next(x for x in h20 if len(x["history"]) == 20)
    assert render(r, 0, "like", "V3")[1].count(" (rated ") == 20
    assert render(r, 0, "like", "V0")[1].count(" (rated ") == 10 and "Dummy Older" not in render(r, 0, "like")[1]
    # V0's window is exactly the last 10 entries whatever the row holds
    short = next(x for x in recs if len(x["history"]) == 5)
    longer = dict(short, history=[f"Extra {k} (rated 3/5)" for k in range(10)] + short["history"])
    assert render(longer, 0, "like")[1] == build_prompt(longer["history"][-10:], longer["candidate_titles"][0],
                                                        longer["candidate_texts"][0], "like", 10)


@pytest.mark.parametrize("name", ["ml1m_rated.jsonl", "toys_rated.jsonl"])
def test_every_variant_renders_every_real_pilot_row(name):
    recs, _ = _pilot(name)
    h20 = _with_h20(recs[:500])
    if name.startswith("toys"):
        for r in h20:
            r["domain_kind"] = "product"
            r["history_meta"] = [f"Categories: Toys & Games {k}" for k in range(len(r["history"]))]
    seen = set()
    for v in GATE_VARIANTS + ("T0_probe",):
        h = _hash(h20, ["like"], None, v)
        assert h not in seen, v
        seen.add(h)


# ---------------------------------------------------------------- (c) golden prompts, written out by hand
RATED_MOVIE = ("You are a movie recommender. You predict how a specific user will rate movies from that user's past "
               "ratings. Ratings range from 1 to 5 stars; 4-5 stars means the user liked the movie and 1-2 stars "
               "means the user disliked it.")
RATED_PRODUCT = ("You are a product recommender. You predict how a specific user will rate products from that user's "
                 "past ratings. Ratings range from 1 to 5 stars; 4-5 stars means the user liked the product and 1-2 "
                 "stars means the user disliked it.")
NEXT_PRODUCT = ("You are a product recommender. You predict a specific user's preferences from that user's purchase "
                "history.")
RATINGS = [3, 5, 1, 4, 2, 5, 3, 4, 1, 5, 2, 4]   # history events 1..12


def golden_row(**kw):
    titles = [f"M{k} ({1980 + k})" for k in range(1, 13)]
    row = {"user_id": "u1", "source_event_id": "u1::100", "source": "ml1m", "domain_kind": "movie",
           "history": [f"{t} (rated {r}/5)" for t, r in zip(titles, RATINGS)],
           "history_item_ids": [f"m{k}" for k in range(1, 13)], "history_titles": titles,
           "history_ratings": [float(r) for r in RATINGS],
           "history_meta": [("" if k == 7 else f"Genres: G{k}") for k in range(1, 13)],
           "candidate_item_ids": ["c1", "c2"], "candidate_titles": ["Toy Story (1995)", "Fargo (1996)"],
           "candidate_texts": ["Genres: Animation, Comedy", ""], "candidate_labels": [1, 0],
           "candidate_ratings": [5.0, 2.0]}
    row.update(kw)
    return row


LAST10 = """- M3 (1983) (rated 1/5)
- M4 (1984) (rated 4/5)
- M5 (1985) (rated 2/5)
- M6 (1986) (rated 5/5)
- M7 (1987) (rated 3/5)
- M8 (1988) (rated 4/5)
- M9 (1989) (rated 1/5)
- M10 (1990) (rated 5/5)
- M11 (1991) (rated 2/5)
- M12 (1992) (rated 4/5)"""
ALL12 = """- M1 (1981) (rated 3/5)
- M2 (1982) (rated 5/5)
""" + LAST10
# M7's meta is empty: its line drops the " [...]" brackets (prompting.OPERATIONALIZATIONS[0], the stated
# operationalization of L_rated_meta "- {title} [{meta}] (rated {r}/5)" for items without genres / categories)
META10 = """- M3 (1983) [Genres: G3] (rated 1/5)
- M4 (1984) [Genres: G4] (rated 4/5)
- M5 (1985) [Genres: G5] (rated 2/5)
- M6 (1986) [Genres: G6] (rated 5/5)
- M7 (1987) (rated 3/5)
- M8 (1988) [Genres: G8] (rated 4/5)
- M9 (1989) [Genres: G9] (rated 1/5)
- M10 (1990) [Genres: G10] (rated 5/5)
- M11 (1991) [Genres: G11] (rated 2/5)
- M12 (1992) [Genres: G12] (rated 4/5)"""
META12 = """- M1 (1981) [Genres: G1] (rated 3/5)
- M2 (1982) [Genres: G2] (rated 5/5)
""" + META10
TOY = "Candidate item:\nTitle: Toy Story (1995)\nDescription: Genres: Animation, Comedy\n\n"
PERSONA = "You are an expert recommendation system.\n\n"
CHRONO = "User history (oldest to newest):\n"
LIKE = "Would this user like the candidate item? Answer with only Yes or No."
TH_LIKE = "Will this user rate the candidate item 4 stars or higher (on a 1-5 scale)? Answer with only Yes or No."
TH_DISLIKE = "Will this user rate the candidate item 2 stars or lower (on a 1-5 scale)? Answer with only Yes or No."

GOLDEN_RATED = {  # (variant, candidate, question) -> (system, user)
    ("V0", 0, "like"): (None, PERSONA + CHRONO + LAST10 + "\n\n" + TOY + LIKE),
    ("V1", 0, "like"): (None, PERSONA + CHRONO + LAST10 + "\n\n" + TOY + TH_LIKE),
    ("V2", 0, "like"): (None, """You are an expert recommendation system.

Items this user liked (rated 4-5 stars), oldest to newest:
- M4 (1984)
- M6 (1986)
- M8 (1988)
- M10 (1990)
- M12 (1992)

Items this user disliked (rated 1-3 stars), oldest to newest:
- M3 (1983)
- M5 (1985)
- M7 (1987)
- M9 (1989)
- M11 (1991)

Candidate item:
Title: Toy Story (1995)
Description: Genres: Animation, Comedy

Would this user like the candidate item? Answer with only Yes or No."""),
    ("V3", 1, "like"): (None, PERSONA + CHRONO + ALL12 + "\n\nCandidate item:\nTitle: Fargo (1996)\n\n" + LIKE),
    ("V4", 0, "like"): (None, PERSONA + CHRONO + META10 + "\n\n" + TOY + LIKE),
    ("V5", 0, "like"): (RATED_MOVIE, CHRONO + LAST10 + "\n\n" + TOY + LIKE),
    ("V7", 0, "like"): (RATED_MOVIE, CHRONO + META12 + "\n\n" + TOY + TH_LIKE),
    ("V7", 0, "dislike"): (RATED_MOVIE, CHRONO + META12 + "\n\n" + TOY + TH_DISLIKE),
    ("V1", 0, "like_para"): (None, PERSONA + CHRONO + LAST10 + "\n\n" + TOY + "Will this user's star rating for "
                             "the candidate item be 4 or 5? Answer with only Yes or No."),
    ("V1", 0, "dislike_para"): (None, PERSONA + CHRONO + LAST10 + "\n\n" + TOY + "Will this user's star rating for "
                                "the candidate item be 1 or 2? Answer with only Yes or No."),
    ("T0_probe", 0, "like"): (None, "Movie: Toy Story (1995)\nDescription: Genres: Animation, Comedy\n\n"
                                    "Is this movie widely considered good? Answer with only Yes or No."),
    ("T0_probe", 1, "quality"): (None, "Movie: Fargo (1996)\n\nIs this movie widely considered good? Answer with only "
                                       "Yes or No."),
}


@pytest.mark.parametrize("key", list(GOLDEN_RATED), ids=lambda k: "-".join(map(str, k)))
def test_golden_rated_movie_prompts(key):
    v, i, q = key
    assert render(golden_row(), i, q, v) == GOLDEN_RATED[key]


def test_golden_product_wording_and_v0_equals_build_prompt():
    row = golden_row(domain_kind="product", source="toys")
    assert render(row, 0, "like", "V5") == (RATED_PRODUCT, CHRONO + LAST10 + "\n\n" + TOY + LIKE)
    assert render(row, 0, "like", "V7")[0] == RATED_PRODUCT
    assert render(row, 1, "like", "T0_probe") == (None, "Product: Fargo (1996)\n\nIs this product widely considered "
                                                        "good? Answer with only Yes or No.")
    # without domain_kind the wording follows source (ml1m -> movie, an Amazon domain -> product)
    nodk = golden_row(source="toys")
    nodk.pop("domain_kind")
    assert render(nodk, 0, "like", "V5")[0] == RATED_PRODUCT
    nodk["source"] = "ml1m"
    assert render(nodk, 0, "like", "V5")[0] == RATED_MOVIE
    nodk.pop("source")
    with pytest.raises(ValueError, match="domain_kind"):
        render(nodk, 0, "like", "V5")
    assert render(nodk, 0, "like", "V4")[0] is None          # variants without a system message do not need it
    row = golden_row()
    for i in range(2):
        for q in pr.QUESTIONS:
            assert render(row, i, q, "V0") == (None, build_prompt(row["history"], row["candidate_titles"][i],
                                                                  row["candidate_texts"][i], q, 10))


def next_row(**kw):
    row = {"user_id": "s1", "source_event_id": "s1::7", "history": [f"Gear {k}" for k in range(1, 7)],
           "history_item_ids": [f"g{k}" for k in range(1, 7)],
           "history_meta": [f"Categories: C{k}" for k in range(1, 7)],
           "candidate_item_ids": ["b", "x"], "candidate_titles": ["Ball", "Bat"], "candidate_texts": ["A ball", ""],
           "positive_item_index": 0, "candidate_labels": [1, 0]}
    row.update(kw)
    return row


NEXT_Q = "Will this user purchase the candidate item next? Answer with only Yes or No."
BALL = "Candidate item:\nTitle: Ball\nDescription: A ball\n\n"
GEAR5 = "- Gear 2\n- Gear 3\n- Gear 4\n- Gear 5\n- Gear 6"
GEAR6 = "- Gear 1\n" + GEAR5
GMETA5 = ("- Gear 2 [Categories: C2]\n- Gear 3 [Categories: C3]\n- Gear 4 [Categories: C4]\n"
          "- Gear 5 [Categories: C5]\n- Gear 6 [Categories: C6]")
GMETA6 = "- Gear 1 [Categories: C1]\n" + GMETA5
GOLDEN_NEXT = {
    ("V0", "next"): (None, PERSONA + CHRONO + GEAR5 + "\n\n" + BALL + NEXT_Q),
    ("V1", "next"): (None, PERSONA + CHRONO + GEAR5 + "\n\n" + BALL + NEXT_Q),
    ("V1", "like"): (None, PERSONA + CHRONO + GEAR5 + "\n\n" + BALL + TH_LIKE),
    ("V2", "next"): (None, PERSONA + CHRONO + GEAR5 + "\n\n" + BALL + NEXT_Q),
    ("V3", "next"): (None, PERSONA + CHRONO + GEAR6 + "\n\n" + BALL + NEXT_Q),
    ("V4", "next"): (None, PERSONA + CHRONO + GMETA5 + "\n\n" + BALL + NEXT_Q),
    ("V5", "next"): (NEXT_PRODUCT, CHRONO + GEAR5 + "\n\n" + BALL + NEXT_Q),
    ("V7", "next"): (NEXT_PRODUCT, CHRONO + GMETA6 + "\n\n" + BALL + NEXT_Q),
    ("V7", "like"): (NEXT_PRODUCT, CHRONO + GMETA6 + "\n\n" + BALL + TH_LIKE),
}


@pytest.mark.parametrize("key", list(GOLDEN_NEXT), ids=lambda k: "-".join(k))
def test_golden_next_item_prompts(key):
    row = next_row()
    assert pr.panel_kind_of(row) == "next_item"
    v, q = key
    assert render(row, 0, q, v) == GOLDEN_NEXT[key]
    assert render(row, 0, q, v, "next_item") == GOLDEN_NEXT[key]


def test_golden_digit_readout_prompts():
    row = golden_row()
    dq = "What star rating from 1 to 5 would this user give the candidate item? Answer with only one digit from 1 to 5."
    for q in ("like", "rating"):
        assert render(row, 0, q, "V0", readout="digits") == (None, PERSONA + CHRONO + LAST10 + "\n\n" + TOY + dq)
    # amendment 2 G0: no gate variant other than the V0 control has a digit readout (battery T3 / T0 only)
    assert pr.DIGIT_VARIANTS == ("V0", "T0_probe")
    for v in set(GATE_VARIANTS) - {"V0"}:
        with pytest.raises(ValueError, match="digit readout"):
            render(row, 0, "like", v, readout="digits")
        with pytest.raises(ValueError, match="digit readout"):
            pr.question_keys(v, "digits")
    assert pr.question_keys("V0", "digits") == pr.question_keys("T0_probe", "digits") == ("like", "rating")
    assert render(row, 0, "like", "T0_probe", readout="digits") == (
        None, "Movie: Toy Story (1995)\nDescription: Genres: Animation, Comedy\n\nOn a 1-5 star scale, how good is "
              "this movie widely considered to be? Answer with only one digit from 1 to 5.")
    for v, q, ro in (("V0", "dislike", "digits"), ("V0", "rating", "yesno"), ("T0_probe", "dislike", "yesno"),
                     ("V1", "quality", "yesno")):
        with pytest.raises(ValueError, match="not defined"):
            render(row, 0, q, v, readout=ro)


def test_split_layout_edge_cases_and_meta_rendering():
    liked_only = golden_row(history=[f"M{k} (1990) (rated 5/5)" for k in range(3)], history_ratings=[5.0] * 3,
                            history_meta=["Genres: A"] * 3)
    u = render(liked_only, 1, "like", "V2")[1]
    assert "oldest to newest:\n- M0 (1990)\n- M1 (1990)\n- M2 (1990)\n\n" in u
    assert "Items this user disliked (rated 1-3 stars), oldest to newest:\n- (none)\n\nCandidate item:" in u
    nohist = render(golden_row(), 1, "like", "V2", hist_len=0)[1]
    assert nohist.count("oldest to newest:\n- (none)\n\n") == 2
    assert "User history (oldest to newest):\n- (no history available)\n\n" in render(golden_row(), 0, "like", "V4",
                                                                                     hist_len=0)[1]
    # threshold rating 3 is "disliked" (TALLRec rule: liked = rating > 3)
    three = golden_row(history=["A (rated 3/5)", "B (rated 4/5)"], history_ratings=[3.0, 4.0], history_meta=["", ""])
    assert "liked (rated 4-5 stars), oldest to newest:\n- B\n\n" in render(three, 0, "like", "V2")[1]
    assert "disliked (rated 1-3 stars), oldest to newest:\n- A\n\n" in render(three, 0, "like", "V2")[1]
    # titles are capped at 200 characters in every line format; Amazon categories at 80 characters
    long = golden_row(history=["T" * 250 + " (rated 4/5)"], history_ratings=[4.0],
                      history_meta=["Categories: " + "c" * 100], domain_kind="product")
    assert "- " + "T" * 200 + " [Categories: " + "c" * 80 + "] (rated 4/5)\n\n" in render(long, 0, "like", "V4")[1]
    split = render(long, 0, "like", "V2")[1]
    assert "- " + "T" * 200 + "\n\n" in split and "T" * 201 not in split
    assert pr.render_meta("Categories: " + "c" * 80) == "Categories: " + "c" * 80    # idempotent on builder output
    assert pr.render_meta("Categories: ") == pr.render_meta("") == pr.render_meta(None) == ""
    assert pr.render_meta("Genres: Action, Drama") == "Genres: Action, Drama"


def test_render_refuses_inconsistent_rows():
    with pytest.raises(ValueError, match="history_meta"):
        render(golden_row(history_meta=None), 0, "like", "V4")
    with pytest.raises(ValueError, match="history_meta has 3 entries"):
        render(golden_row(history_meta=["x"] * 3), 0, "like", "V7")
    with pytest.raises(ValueError, match="disagrees with history_ratings"):
        render(golden_row(history_ratings=[4.0] * 12), 0, "like", "V2")
    with pytest.raises(ValueError, match="suffix"):
        render(golden_row(history=["No suffix"] * 12), 0, "like", "V4")
    with pytest.raises(ValueError, match="V6"):
        render(golden_row(), 0, "like", "V6")
    with pytest.raises(ValueError, match="not defined"):
        render(golden_row(), 0, "next_purchase", "V1")


def test_history_window_rules():
    assert {v: (resolve_hist_len(v, "rated"), resolve_hist_len(v, "next_item")) for v in VARIANTS} == {
        "V0": (10, 5), "V1": (10, 5), "V2": (10, 5), "V3": (20, 20), "V4": (10, 5), "V5": (10, 5), "V7": (20, 20),
        "T0_probe": (0, 0)}
    assert resolve_hist_len("V0", "rated", 7) == 7 and resolve_hist_len("V3", "rated", 0) == 0
    assert resolve_hist_len("V3", "rated", 20) == 20 and resolve_hist_len("T0_probe", "rated", 10) == 0
    for v, hl in (("V1", 5), ("V3", 10), ("V7", 10), ("V4", 20)):
        with pytest.raises(ValueError, match="unregistered"):
            resolve_hist_len(v, "rated", hl)
    with pytest.raises(ValueError):
        resolve_hist_len("V0", "rated", -1)
    assert pr.SIMPLICITY_ORDER == ("V0", "V1", "V5", "V3", "V4", "V2", "V7")
    assert set(GATE_VARIANTS) == set(VARIANTS) - {"T0_probe"} and "V6" not in VARIANTS


# ---------------------------------------------------------------- registered strings = the variants JSON
def _spec_json():
    for p in SPEC_JSON:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    pytest.skip("gatefix_prompt_variants_v1.json not found")


def test_registered_strings_match_the_variants_json():
    js = _spec_json()
    assert pr.QUESTION_FAMILIES == js["question_sets"]
    assert pr.SYSTEM_MESSAGES == js["system_messages"]
    assert pr.USER_TEMPLATES == {k: v for k, v in js["user_templates"].items() if k != "T_relevant"}
    assert pr.USER_TEMPLATES["T_chrono"].replace("{hist_lines}", "{hist}").replace("{cand_title}", "{title}") \
        .replace("{cand_desc}", "{desc}") == pr.BODY
    assert js["invariants"]["answer_suffix"] == pr.YESNO_TAIL and js["invariants"]["max_model_len"] == 4096
    assert js["invariants"]["title_cap_chars"] == pr.MAX_TITLE_CHARS
    assert set(VARIANTS) == (set(js["variants"]) - {"V6"}) | {"T0_probe"}
    assert [v for v in js["simplicity_order_for_ties"] if v != "V6"] == list(pr.SIMPLICITY_ORDER)
    lf = js["history_line_formats"]
    assert {k: pr.LINE_FORMATS[k] for k in ("L_rated", "L_rated_meta", "L_plain", "L_plain_meta")} == {
        k: lf[k] for k in ("L_rated", "L_rated_meta", "L_plain", "L_plain_meta")}
    assert lf["L_split"].startswith(pr.LINE_FORMATS["L_split"] + " ") and "'- (none)'" in lf["L_split"]
    assert pr.CATEGORIES_PREFIX + "{categories string[:80]}" in lf["meta_amazon"] and pr.MAX_CATEGORY_CHARS == 80
    for v in GATE_VARIANTS:
        for kind in ("rated", "next_item"):
            j, s = js["variants"][v][kind], VARIANTS[v][kind]
            if "note" in j:          # V2 next-item: "renders identically to V0 (declared)"
                assert v == "V2" and s == VARIANTS["V0"]["next_item"]
                continue
            sysj = j["system"]
            assert s.system == (None if sysj is None else "rated" if "rated_movie" in sysj else sysj), (v, kind)
            assert (s.user, s.line, s.q) == (j["user"], j["line"], j["q"].split()[0]), (v, kind)
            assert s.hist == int(next(w for w in j["hist"].replace("(", " ").split() if w.isdigit())), (v, kind)


# ---------------------------------------------------------------- fixed strings: one sha1 for the freeze record
# Pinned on purpose: any edit to a question, template, system message, the T0 or digit wording, a cap or a stated
# operationalization changes PROMPT_STRINGS_SHA1, which must then be re-recorded with the freeze (PILOT_LOG) before
# any GPU job. Update this constant only together with that record.
PINNED_PROMPT_STRINGS_SHA1 = "cc215d598a6a6a0f6b6545df690f60d44ee6bafb"


def test_prompt_strings_cover_the_unfrozen_wording_and_are_pinned():
    s = pr.prompt_strings()
    assert s["digits"] == {"question": "What star rating from 1 to 5 would this user give the candidate item?",
                           "t0_question": "On a 1-5 star scale, how good is this {noun} widely considered to be?",
                           "tail": " Answer with only one digit from 1 to 5.", "keys": ["like", "rating"],
                           "variants": ["V0", "T0_probe"]}
    assert s["t0"]["question"] == "Is this {noun} widely considered good?"
    assert s["question_families"] == pr.QUESTION_FAMILIES and s["system_messages"] == pr.SYSTEM_MESSAGES
    assert set(s["variants"]) == set(VARIANTS) and s["variants"]["V7"]["rated"]["hist"] == 20
    ops = " ".join(s["operationalizations"])
    assert "drops the ' [...]' brackets" in s["operationalizations"][0]
    for rule in ("drops the ' [...]' brackets", "capped at 80 characters", "capped at 200 characters",
                 "both lists at hist_len 0", "digit readout", "frozen at V0"):
        assert rule in ops, rule
    s["question_families"]["like_family"]["like"] = "edited"                 # a copy: the bank is untouched
    assert pr.QUESTIONS["like"] == "Would this user like the candidate item?"
    canon = json.dumps(pr.prompt_strings(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert pr.PROMPT_STRINGS_SHA1 == hashlib.sha1(canon.encode("utf-8")).hexdigest()
    assert pr.PROMPT_STRINGS_SHA1 == PINNED_PROMPT_STRINGS_SHA1


# ---------------------------------------------------------------- chat template checks (system message, thinking off)
class _Tmpl:
    """Fake tokenizer; `keep` = roles the template renders, `think` = generation-prompt suffix, chat_template text."""
    def __init__(self, keep=("system", "user"), think="", chat_template=None):
        self.keep, self.think, self.chat_template = keep, think, chat_template

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        body = "".join(f"<{m['role']}>{m['content'].strip()}</{m['role']}>" for m in msg if m["role"] in self.keep)
        return body + "<assistant>" + (self.think if not enable_thinking else "")

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in text]}


def test_chat_ids_refuses_a_template_that_drops_a_message_or_leaves_thinking_on():
    p = pr.ChatPrompt("User history:\n- A (rated 4/5)\n\nWould this user like it? Answer with only Yes or No.",
                      pr.SYSTEM_MESSAGES["rated_movie"])
    assert pr.chat_ids(_Tmpl(), p)                                          # both messages rendered
    with pytest.raises(ValueError, match="dropped or rewrote the system message"):
        pr.chat_ids(_Tmpl(keep=("user",)), p)
    with pytest.raises(ValueError, match="dropped or rewrote the user message"):
        pr.chat_ids(_Tmpl(keep=("system",)), p)
    assert pr.chat_ids(_Tmpl(keep=("user",)), str(p))                       # V0: no system message to lose
    qwen = "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}"
    assert pr.chat_ids(_Tmpl(think="<think>\n\n</think>\n\n", chat_template=qwen), p)
    with pytest.raises(ValueError, match="thinking would be on"):
        pr.chat_ids(_Tmpl(think="", chat_template=qwen), p)                # template ignored enable_thinking=False
    assert pr.chat_ids(_Tmpl(think="", chat_template="{{ llama }}"), p)    # no thinking in the template: nothing to do


# ---------------------------------------------------------------- (d) candidate_extras
EXTRA = "(This user later rated this item 5/5.)"


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_candidate_extras_line_follows_the_description_in_every_variant(variant):
    for row in (golden_row(), next_row(domain_kind="product")):   # T0_probe needs the movie/product wording
        plain = [render(row, i, "like", variant) for i in range(2)]
        assert [render(dict(row, candidate_extras=["", ""]), i, "like", variant) for i in range(2)] == plain
        ex = dict(row, candidate_extras=[EXTRA, "Average rating by other users before this date: 3.50/5 (4 ratings)"])
        got = [render(ex, i, "like", variant) for i in range(2)]
        desc0 = "\nDescription: " + row["candidate_texts"][0]
        assert got[0] == (plain[0][0], plain[0][1].replace(desc0 + "\n\n", desc0 + "\n" + EXTRA + "\n\n", 1))
        title1 = row["candidate_titles"][1]      # no description: the extra follows the title line
        assert got[1][1] == plain[1][1].replace(
            f": {title1}\n\n", f": {title1}\nAverage rating by other users before this date: 3.50/5 (4 ratings)\n\n", 1)
        assert got[1][1] != plain[1][1] and got[1][0] == plain[1][0]
        recs = render_record(ex, ["like"], variant)
        assert [(i, q, (s, u)) for i, q, s, u in recs] == [(0, "like", got[0]), (1, "like", got[1])]


def test_extras_absent_keeps_v0_identical_and_record_requests_carry_them():
    row = golden_row()
    assert [str(p) for *_, p in ps.record_requests(row, ["like", "dislike"], 10)] == [
        build_prompt(row["history"], t, x, q, 10) for t, x in zip(row["candidate_titles"], row["candidate_texts"])
        for q in ("like", "dislike")]
    ex = dict(row, candidate_extras=[EXTRA, ""])
    ps_prompts = [p for *_, p in ps.record_requests(ex, ["like"], 10)]
    assert EXTRA in ps_prompts[0] and EXTRA not in ps_prompts[1]
    # an empty or None candidate_extras means none, in render and render_record alike (no crash, byte-identical)
    plain = render_record(row, ["like"])
    for empty in ([], None):
        assert render_record(dict(row, candidate_extras=empty), ["like"]) == plain
        assert [render(dict(row, candidate_extras=empty), i, "like") for i in range(2)] == [
            (s, u) for _, _, s, u in plain]
    # misaligned candidate lists raise ValueError naming the row and the lengths (an assert would vanish under -O)
    for bad in ({"candidate_extras": [EXTRA]}, {"candidate_texts": ["only one"]},
                {"candidate_item_ids": ["c1", "c2", "c3"]}):
        with pytest.raises(ValueError, match=r"u1::100: misaligned candidate lists"):
            ps.record_requests(dict(row, **bad), ["like"])
        with pytest.raises(ValueError, match="misaligned"):
            render(dict(row, **bad), 0, "like")
    with pytest.raises(ValueError, match="no candidate_item_ids"):
        render_record({k: v for k, v in row.items() if k != "candidate_item_ids"}, ["like"])


# ---------------------------------------------------------------- (e) digit readout
class DigitTok:
    VOCAB = ["1", " 2", "3", "4", " 4", "5", "33", "6", "0", "Yes", " no", "12"]

    def __len__(self):
        return len(self.VOCAB)

    def decode(self, ids):
        return "".join(self.VOCAB[i] for i in ids)


def test_digit_ids_by_decoded_text():
    assert digit_ids(DigitTok()) == {1: {0}, 2: {1}, 3: {2}, 4: {3, 4}, 5: {5}}


def test_read_digits_math_and_censoring():
    ids = {1: {10}, 2: {20}, 3: {30}, 4: {40, 41}, 5: {50}}
    p = {10: .1, 20: .1, 30: .2, 40: .25, 41: .05, 50: .2, 99: .05}
    top = {k: LP(math.log(v)) for k, v in p.items()}
    top[7] = LP(-math.inf)                                   # non-finite logprobs are ignored
    er, logit, mass, c = read_digits(top, ids)
    z = .1 + .1 + .2 + .3 + .2
    assert c == 0 and mass == pytest.approx(.9)
    assert er == pytest.approx((1 * .1 + 2 * .1 + 3 * .2 + 4 * .3 + 5 * .2) / z)
    assert logit == pytest.approx(math.log(.5) - math.log(.2))
    lp45, lp12, mass2, c2, er2 = read_digits_full(top, ids)
    assert (lp45, lp12, mass2, c2, er2) == pytest.approx((math.log(.5), math.log(.2), .9, 0, er))
    # digit 1 and 5 absent: each imputed with the smallest returned logprob (.05), censored 1
    top2 = {k: v for k, v in top.items() if k not in (10, 50)}
    er, logit, mass, c = read_digits(top2, ids)
    z = .05 + .1 + .2 + .3 + .05
    assert c == 1 and mass == pytest.approx(.6)
    assert er == pytest.approx((1 * .05 + 2 * .1 + 3 * .2 + 4 * .3 + 5 * .05) / z)
    assert logit == pytest.approx(math.log(.35) - math.log(.15))
    # no digit at all: censored 2, NaNs
    er, logit, mass, c = read_digits({99: LP(-1.0), 98: LP(-2.0)}, ids)
    assert c == 2 and mass == 0.0 and math.isnan(er) and math.isnan(logit)
    # E[r] is invariant to a common scale of the digit probabilities (normalised over the 5 digits)
    scaled = {k: LP(v.logprob + math.log(.5)) for k, v in top.items() if k != 7}
    scaled[99] = LP(math.log(.5))
    assert read_digits(scaled, ids)[0] == pytest.approx(read_digits(top, ids)[0])


def test_chat_prompt_key_and_system_passthrough():
    p = pr.ChatPrompt("user text", "sys")
    assert p == "user text" and p.system == "sys" and p.key == "sys\x1duser text"
    assert pr.ChatPrompt("u").key == "u" and pr.prompt_key("plain") == "plain"
    assert ps.prompts_sha1([pr.ChatPrompt("u")]) == ps.prompts_sha1(["u"])
    assert ps.prompts_sha1([p]) == ps.prompts_sha1(["sys\x1duser text"]) != ps.prompts_sha1(["user text"])

    class Tok:
        def __init__(self):
            self.msgs = None

        def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, enable_thinking=False):
            self.msgs = msg
            return "|".join(m["role"] + ":" + m["content"] for m in msg)

        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": [ord(c) for c in text]}

    t = Tok()
    assert pr.chat_ids(t, p) == pr.chat_ids(t, "user text", "sys")
    assert t.msgs == [{"role": "system", "content": "sys"}, {"role": "user", "content": "user text"}]
    pr.chat_ids(t, "user text")
    assert t.msgs == [{"role": "user", "content": "user text"}]       # no system message: the Pilot-1 path
