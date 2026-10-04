"""Cross-module contract tests: the pilot pipelines chained as scripts/sigir/run_pilot1_mirror.sh and run_pilot2_3.sh
call them (builders -> pyes_scorer.run -> analyses -> gate), on tiny synthetic inputs, with the scorer's model faked
(planted Yes/No logprobs, incl. censored rows). They guard the interfaces between the groups' modules: scorer columns
and censored codes, panel fields, pseudonymized panels, every key pilot1_gate.py reads from pilot_mirror, the
token-channel gate shared by pilots 1 and 3, and a KuaiRec directory whose name contains a space.
"""
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

from src.confrec import build_kuairec_panel as bk
from src.confrec import build_rated_panels as brp
from src.confrec import pilot_kuairec as pk
from src.confrec import pilot_mirror as pm
from src.confrec import pilot_pseudonym as pp
from src.confrec import pseudonymize as pz
from src.confrec import pyes_scorer as ps
from src.confrec import split_panel as spl

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pilot1_gate", ROOT / "scripts" / "sigir" / "pilot1_gate.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def strict(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda c: 1 / 0)


def _u(*parts) -> float:
    return int(hashlib.sha1("\x1f".join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16 ** 12


def _g(*parts) -> float:
    return math.sqrt(-2 * math.log(max(_u(*parts, "a"), 1e-12))) * math.cos(2 * math.pi * _u(*parts, "b"))


# ------------------------------------------------------------------ fake model for pyes_scorer.run
class Tok:
    """Character tokenizer with a growing vocabulary; ids 1 / 2 decode to Yes / No."""
    def __init__(self):
        self.v, self.r = {}, {1: "Yes", 2: "No"}

    def __len__(self):
        return 3 + len(self.v)

    def decode(self, ids):
        return "".join(self.r.get(i, "") for i in ids)

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, **kw):
        return "<u>" + msg[0]["content"] + "<a>"

    def __call__(self, text, add_special_tokens=False):
        for c in text:
            if c not in self.v:
                self.v[c] = len(self.v) + 3
                self.r[self.v[c]] = c
        return {"input_ids": [self.v[c] for c in text]}


def score(data, out, *flags):
    """pyes_scorer.run with a fake model: L = 1.2 (2y - 1) + 0.4 z_pop + noise (no label signal without history);
    ~2% of prompts lose the No side (censored=1), ~1% both sides (censored=2)."""
    args = ps.parse_args(["--data", str(data), "--output", str(out), "--model", "fake/model", "--dtype", "float16",
                          "--topk_logprobs", "50", "--chunk_users", "7", *flags])
    rows = [json.loads(x) for x in open(data, encoding="utf-8")]
    qs = args.questions.split(",")
    planted, prior = {}, {}
    for r in rows:
        y = ps.labels_of(r)
        for i, q, p in ps.record_requests(r, qs, args.hist_len):
            pop = r.get("candidate_popularity")
            z = (math.log1p(pop[i]) - 2) if pop else {"head": 1, "tail": -1}.get(
                (r.get("candidate_popularity_groups") or ["mid"] * 999)[i], 0)
            sign = -1 if q.startswith("dislike") else 1
            sig = 0 if args.hist_len == 0 else 1.2 * sign * (2 * y[i] - 1)
            planted[p] = sig + 0.4 * z + _g(r["user_id"], i, q, r["candidate_titles"][i])
            prior[str(r["candidate_titles"][i])[:200]] = 0.4 * z

    def load(_args):
        tok = Tok()

        class LLM:
            def generate(self, prompts, sp, use_tqdm=False, **kw):
                out = []
                for pr in prompts:
                    body = tok.decode(pr["prompt_token_ids"])[3:-3]
                    if body in planted:
                        lg = planted[body]
                    else:  # swap prompt
                        title = body.split("\nTitle: ", 1)[1].split("\n", 1)[0]
                        lg = prior.get(title, 0.0) + 0.5 * _g("swap", body)
                    py = 0.9 / (1 + math.exp(-lg))
                    top = {-k: SimpleNamespace(logprob=math.log(0.002) - k / 100) for k in range(1, 49)}
                    h = _u("cens", body)
                    if h >= 0.01:
                        top[1] = SimpleNamespace(logprob=math.log(py))
                        if h >= 0.03:
                            top[2] = SimpleNamespace(logprob=math.log(0.9 - py))
                    out.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=[top])]))
                return out
        return SimpleNamespace(llm=LLM(), tok=tok, gen_kw={}, to_prompt=lambda ids: {"prompt_token_ids": ids},
                               sp=None, version="fake")
    ps.run(args, load_model=load)
    return Path(out)


# ------------------------------------------------------------------ tiny raw inputs
def _gz(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")


STORES = ["LEGO", "Hasbro", "Mattel", "Brio", "Hape", "Crayola", "Funko", "Kids", "Magic", "Ty", "Generic", ""]


def amazon_raw(raw, n_users=45, n_items=90, seed=0):
    r = random.Random(seed)
    meta, q = [], {}
    for k in range(n_items):
        st = STORES[k % len(STORES)]
        title = f"{st} Toy {k}" if st and k % 3 else f"Toy {k}"
        meta.append({"parent_asin": f"B{k:04d}", "title": title, "categories": ["Toys & Games"],
                     "description": [r.choice(["Great for kids.", "Magic tricks for kids.", "Fun."])], "store": st})
        q[f"B{k:04d}"] = r.gauss(0, 1)
    _gz(raw / "amazon_toys" / "meta_Toys_and_Games.jsonl.gz", meta)
    revs = []
    for u in range(n_users):
        t = 1_600_000_000_000 + u * 10 ** 7
        for k in r.sample(range(n_items), 30):
            t += r.randint(1, 10 ** 5)
            rating = float(min(5, max(1, round(3 + 1.2 * q[f"B{k:04d}"] + r.gauss(0, 1.3)))))
            revs.append({"user_id": f"U{u:03d}", "parent_asin": f"B{k:04d}", "rating": rating, "timestamp": t})
        revs.append({**revs[-30], "timestamp": t + 1, "rating": 6.0 - revs[-30]["rating"]})  # re-review
    _gz(raw / "amazon_toys" / "Toys_and_Games.jsonl.gz", revs)


def ml1m_raw(d, n_users=45, n_movies=80, seed=1):
    r = random.Random(seed)
    d.mkdir(parents=True, exist_ok=True)
    (d / "movies.dat").write_text("".join(f"{m}::Movie {m} ({1990 + m % 9})::Drama|Comedy\n"
                                          for m in range(1, n_movies + 1)), encoding="latin-1")
    q = {m: r.gauss(0, 1) for m in range(1, n_movies + 1)}
    lines = []
    for u in range(1, n_users + 1):
        t = 978300000 + u * 10 ** 5
        for m in r.sample(range(1, n_movies + 1), 32):
            t += r.randint(1, 900)
            lines.append(f"{u}::{m}::{min(5, max(1, round(3 + 1.2 * q[m] + r.gauss(0, 1.3))))}::{t}\n")
    (d / "ratings.dat").write_text("".join(lines), encoding="latin-1")


def build_rated(monkeypatch, source, raw, out, n_users=30):
    argv = ["x", "--source", source, "--raw", str(raw), "--out", str(out), "--n_users", str(n_users)]
    monkeypatch.setattr(sys, "argv", argv + (["--domain", "toys"] if source == "amazon" else []))
    brp.main()
    return out


def run_main(monkeypatch, mod, argv):
    monkeypatch.setattr(sys, "argv", ["x", *map(str, argv)])
    mod.main()


def next_item_panel(path, ref_path, n_events=12, n_c=21, seed=2):
    r = random.Random(seed)
    rows, ref = [], []
    for e in range(n_events):
        uid, pos = f"AE{e:03d}", r.randrange(n_c)
        groups = [r.choice(["head", "mid", "tail"]) for _ in range(n_c)]
        ids = [f"S{e}_{c}" for c in range(n_c)]
        rows.append({"source_event_id": f"{uid}::{1000 + e}", "user_id": uid, "history": [f"Gear {e}-{h}" for h in
                                                                                          range(6)],
                     "candidate_item_ids": ids, "candidate_titles": [f"Item {i}" for i in ids],
                     "candidate_texts": [""] * n_c, "candidate_popularity_groups": groups,
                     "candidate_labels": [int(c == pos) for c in range(n_c)], "positive_item_index": pos})
        ref.append([f"{uid}::{1000 + e}", uid, r.randint(1, 15), n_c])
    path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    with gzip.open(ref_path, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_event_id", "user_id", "positive_rank", "num_candidates"])
        w.writerows(ref + [["AEXTRA::1", "AEXTRA", 3, n_c]])
    return path


# ------------------------------------------------------------------ pilot 1
def test_pilot1_chain_gate_reads_everything_pilot_mirror_writes(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    amazon_raw(raw)
    ml1m_raw(raw / "ml-1m")
    P, O = tmp_path / "panels", tmp_path / "out"
    jsons = {}
    for dom, src, rd in (("ml1m", "ml1m", raw / "ml-1m"), ("toys", "amazon", raw)):
        p = build_rated(monkeypatch, src, rd, P / f"{dom}_rated.jsonl")
        d = score(p, O / f"{dom}_rated", "--questions", "like,dislike,like_para", "--swap_k", "3", "--hist_len", "10")
        n = score(p, O / f"{dom}_rated_nohist", "--questions", "like", "--hist_len", "0")
        jsons[dom] = d / "pilot_mirror.json"
        run_main(monkeypatch, pm, ["--scores", d / "scores.csv.gz", "--panel", p, "--swap", d / "swap_prior.csv.gz",
                                   "--nohist", n / "scores.csv.gz", "--base_q", "like", "--out", jsons[dom],
                                   "--n_boot", "20"])
        res = strict(jsons[dom])
        assert res["panel_join"] == {} and set(res["censoring"]["scores"]["like"]) >= {"0", "1", "2"}
        assert {"evidence", "pmi_nohist", "mirror", "placebo"} <= set(res["arms"])
    sp = next_item_panel(P / "sports_next_1k.jsonl", tmp_path / "ccrp_v3.csv.gz")
    d = score(sp, O / "sports_next_1k", "--questions", "next,like,dislike,like_para", "--hist_len", "5")
    jsons["sports"] = d / "pilot_mirror.json"
    run_main(monkeypatch, pm, ["--scores", d / "scores.csv.gz", "--panel", sp, "--base_q", "next", "--ref_ranks",
                               tmp_path / "ccrp_v3.csv.gz", "--out", jsons["sports"], "--n_boot", "20"])
    res = strict(jsons["sports"])
    assert res["ref_ranks"]["n_joined"] == 12 and res["ref_ranks"]["ref_events_not_in_scored_panel"] == 1
    assert all(f"dNDCG@10_{a}_minus_ccrp" in res for a in ("raw", "raw_like", "mirror", "placebo"))
    code = gate.main(["--ml1m", str(jsons["ml1m"]), "--toys", str(jsons["toys"]), "--sports", str(jsons["sports"]),
                      "--out_dir", str(O)])
    dec = strict(O / "decision.json")
    assert code in (0, 3) and (O / ("GATE_PASS" if code == 0 else "GATE_FAIL")).exists()
    assert dec["gate"]["input_checks_ok"] is True and dec["missing_inputs"] == []
    for dom in ("ml1m", "toys"):
        c = dec["criteria"][dom]
        assert all(math.isfinite(c[k]) for k in ("SD_pair", "corr_a_logpop", "corr_v_logpop", "corr_pi_logpop",
                                                 "reference_SD_pair_df", "reference_corr_a_logpop_prior_pop"))
    assert math.isfinite(dec["gate"]["ccrp_same_events_NDCG@10"]) and "dNDCG@10_placebo_minus_ccrp" in \
        dec["next_item_no_loss"]
    # every label the gate can emit (amendment 2 closes the table with INDETERMINATE; INCOMPLETE = undetermined input)
    assert dec["decision"] in gate.LABELS and dec["decision"] != "INCOMPLETE" and dec["missing_inputs"] == []


# ------------------------------------------------------------------ pilot 3 (+ split_panel)
def test_pilot3_chain_pseudonym_panels_scores_and_shared_gate(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    amazon_raw(raw, n_users=50)
    P, O = tmp_path / "panels", tmp_path / "out"
    p = build_rated(monkeypatch, "amazon", raw, P / "toys_rated.jsonl", n_users=40)
    run_main(monkeypatch, pz, ["--panel", p, "--seed", "0"])
    rep = strict(P / "toys_rated_pseudonym_report.json")
    assert rep["popularity_source"] == "sidecar" and rep["n_treated"] > 0
    assert rep["stoplist"]["generic_word"]["strings"] >= 1 and all(rep["checks"][k] is True for k in (
        "placebo_keys_equal_pseudo_keys", "pseudonyms_injective"))
    pseudo = [json.loads(x) for x in open(P / "toys_rated_pseudo.jsonl", encoding="utf-8")]
    real = [json.loads(x) for x in open(p, encoding="utf-8")]
    for a, b in zip(real, pseudo):
        assert a["source_event_id"] == b["source_event_id"] and a["candidate_item_ids"] == b["candidate_item_ids"]
        assert b["history"] == [f"{t} (rated {int(x)}/5)" for t, x in zip(b["history_titles"], b["history_ratings"])]
    real_d = score(p, O / "toys_rated", "--questions", "like,dislike,like_para", "--swap_k", "3", "--hist_len", "10")
    ps_d = score(P / "toys_rated_pseudo.jsonl", O / "toys_rated_pseudo", "--questions", "like,dislike", "--swap_k",
                 "3", "--hist_len", "10")
    pl_d = score(P / "toys_rated_placebo.jsonl", O / "toys_rated_placebo", "--questions", "like", "--hist_len", "10")
    g = tmp_path / "decision.json"
    g.write_text(json.dumps({"decision": "GATE_FAIL_UNINTERPRETABLE", "gate": {"pass": False}}), encoding="utf-8")
    J = O / "toys_pilot_pseudonym.json"
    run_main(monkeypatch, pp, ["--real", real_d, "--pseudo", ps_d, "--placebo", pl_d, "--panel", p, "--out", J,
                               "--n_boot", "20", "--bi_boot", "5", "--gate", g])
    res = strict(J)
    assert res["treated"]["source"] == "pseudo_panel" and res["treated"]["n_flag_mismatch_vs_recomputed"] == 0
    assert res["rows"]["panel_join"] == {} and res["rows"]["n_analysed"] > 0
    assert res["cross_check"]["corr_a_logpop"]["all"]["real"]["n_clusters"] > 0
    assert "corr_pi_logpop" in res["cross_check"]["popularity_prior_robustness"]
    assert res["decision"]["verdict"] == "GATE_FAIL_UNINTERPRETABLE" and res["token_channel_gate"]["pass"] is False
    # LoRA panels (amendment D): exclusion of Lumen users from train + global temporal split
    ex = tmp_path / "selected_users.csv"
    ex.write_text("user_id\n" + "".join(r["user_id"] + "\n" for r in real[:5]), encoding="utf-8")
    run_main(monkeypatch, spl, ["--panel", p, "--time_split", "0.8", "--exclude_users_csv", ex])
    info = strict(P / "toys_rated_time_split.json")
    train = [json.loads(x) for x in open(P / "toys_rated_time_train.jsonl", encoding="utf-8")]
    assert info["excluded_from_train"] > 0 and not {r["user_id"] for r in train} & {r["user_id"] for r in real[:5]}
    assert all(t < info["T"] for r in train for t in r["candidate_timestamps"])


# ------------------------------------------------------------------ pilot 2
def test_pilot2_chain_kuairec_dir_with_space(tmp_path):
    r = random.Random(3)
    d = tmp_path / "KuaiRec" / "KuaiRec 2.0" / "data"
    d.mkdir(parents=True)
    cols = "user_id,video_id,play_duration,video_duration,time,date,timestamp,watch_ratio\n"
    small = [f"{u},{v},1,1,,20200701,{1.6e9 + v},{math.exp(r.gauss(0, 0.9)):.4f}\n"
             for u in range(10) for v in range(60)]
    (d / "small_matrix.csv").write_text(cols + "".join(small), encoding="utf-8")
    big = []
    for u in range(60):
        for v in r.sample(range(150), 30):
            if u < 10 and v < 60:
                continue  # KuaiRec removes the small-matrix cells from big
            big.append(f"{u},{v},1,1,,20200701,{1.5e9 + v},{math.exp(r.gauss(0.2, 0.9)):.4f}\n")
    (d / "big_matrix.csv").write_text(cols + "".join(big), encoding="utf-8")
    cap = ["video_id,manual_cover_text,caption,topic_tag,first_level_category_name\n"]
    cap += [f"{v},封面{v},{'UNKNOWN' if v % 11 == 0 else f'视频{v}'},[美食],生活\n" for v in range(150)]
    (d / "kuairec_caption_category.csv").write_text("".join(cap), encoding="utf-8")
    panel = tmp_path / "panels" / "kuairec.jsonl"
    bk.main(["--root", str(d), "--caption", str(d / "kuairec_caption_category.csv"), "--out", str(panel),
             "--n_users", "8", "--n_cands", "40"])
    meta = strict(panel.with_suffix(".meta.json"))
    assert meta["shared_pairs"] == 0 and meta["n_users"] == 8
    sd = score(panel, tmp_path / "pilot2_kuairec", "--questions", "like", "--hist_len", "10")
    out = sd / "pilot_kuairec.json"
    pk.main(["--scores", str(sd / "scores.csv.gz"), "--panel", str(panel), "--out", str(out), "--n_draws", "3",
             "--n_boot", "60"])
    res = strict(out)
    sc = res["scores"]
    assert sc["n_panel_pairs_without_score"] == sc["censored_counts"].get("2", 0) > 0
    assert "1" in sc["censored_counts_kept"] and res["n_pairs"] == 8 * 40 - sc["n_nonfinite_dropped"]
    assert res["decision"] in ("NULL", "POSITIVE", "NEGATIVE", "AMBIGUOUS")
    assert {"share", "lift", "gap", "gap_ceiling"} <= set(res["primary"])
