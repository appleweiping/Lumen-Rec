"""src/confrec/ftgrid_report.py: the Amendment-3 report (A3 section 3, P1, the knockout labels, FT-C).
Synthetic panels, raw ratings and scorer-format runs only; CPU, deterministic, no torch / transformers."""
import csv
import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.confrec import forensics as fx
from src.confrec import ftgrid_report as fr
from src.confrec import pilot_pseudonym as pps
from src.confrec.metrics import auroc, bias_index, ece, risk_coverage
from src.confrec.split_panel import filter_candidates
from src.confrec.stats import cluster_bootstrap, paired_bootstrap, platt_fit, rank_bins, sigmoid

COLS = ["source_event_id", "user_id", "item_id", "cand_idx", "label", "question", "lp_yes", "lp_no", "logit",
        "yes_no_mass", "censored", "exp_rating"]
SWAP_COLS = ["item_id", "donor_user_id", "question", "logit", "censored"]
BRANDS = ["Lumora", "Brightway", "Castlecraft", "Dinoworks", "Puzzlemill", "Rocketeer"]


# ---------------------------------------------------------------- synthetic world
def write_jsonl(path, rows):
    path.write_bytes(b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in rows))
    return path


def gz(path, header, rows):
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def ids_file(path, ids):
    path.write_bytes("\n".join(ids).encode("utf-8"))
    return path


def make_world(root: Path, domain="ml1m", seed=0, n_train=60, n_eval=520, n_bg=150, n_items=120, quantile=0.6,
               raw=True):
    """Rated panels in the layout of the spec (panels/<d>/...) plus ML-1M-format raw ratings. Label of (u, i) =
    1[b_u + mu_i + a_u.c_i + noise > 0]: an item prior mu_i and a personal term e_ui = a_u.c_i."""
    rng = np.random.default_rng(seed)
    mu, C = rng.normal(0, 1, n_items), rng.normal(0, 1, (n_items, 2))
    amazon = domain != "ml1m"
    iid = [f"B{i:05d}" if amazon else str(i + 1) for i in range(n_items)]
    brand = [BRANDS[i % len(BRANDS)] if (amazon and i % 3) else "" for i in range(n_items)]

    def title(i):
        if not amazon:
            return f"Movie {i + 1} ({1950 + i % 50})"
        return f"{brand[i]} Toy {i}" if brand[i] and i % 2 else f"Toy model {i}"

    wts = 1.0 / (np.arange(n_items) + 8.0)            # skewed item choice: heterogeneous popularity
    wts /= wts.sum()

    def gen(n_ev):
        a, b = rng.normal(0, 1, 2), rng.normal(0, 0.4)
        its = rng.choice(n_items, n_ev, replace=False, p=wts)
        ts = (rng.integers(0, 40_000) + np.cumsum(rng.integers(50, 1500, n_ev))).astype(int)
        z = b + mu[its] + C[its] @ a + rng.normal(0, 0.7, n_ev)
        y = (z > 0).astype(int)
        r = np.where(y == 1, np.where(z > 1, 5, 4), np.where(z < -1, 1, 2))
        return {"a": a, "b": b, "items": its, "ts": ts, "r": r, "y": y}

    def eligible(ev):
        yc = ev["y"][10:]
        return 3 <= yc.sum() <= len(yc) - 3

    users = {}
    for prefix, n in (("t", n_train), ("e", n_eval)):
        k = 0
        while sum(u.startswith(prefix) for u in users) < n:
            ev = gen(30)
            if eligible(ev):
                users[f"{prefix}{k}"] = ev
            k += 1
    for k in range(n_bg):
        users[f"b{k}"] = gen(25)
    pop = np.zeros(n_items, int)
    for ev in users.values():
        np.add.at(pop, ev["items"], 1)

    def row(uid):
        ev = users[uid]
        h, c = slice(0, 10), slice(10, 30)
        hi, ci = ev["items"][h], ev["items"][c]
        return {"user_id": uid, "source_event_id": f"{uid}::{int(ev['ts'][10])}",
                "history": [f"{title(i)} (rated {int(r)}/5)" for i, r in zip(hi, ev["r"][h])],
                "history_item_ids": [iid[i] for i in hi], "history_titles": [title(i) for i in hi],
                "history_ratings": [float(r) for r in ev["r"][h]], "history_brands": [brand[i] for i in hi],
                "history_meta": [""] * 10, "candidate_item_ids": [iid[i] for i in ci],
                "candidate_titles": [title(i) for i in ci],
                "candidate_texts": [f"Categories: Toys. {'word ' * (i % 7)}".strip() if amazon else "Genres: Drama"
                                    for i in ci],
                "candidate_brands": [brand[i] for i in ci], "candidate_ratings": [float(r) for r in ev["r"][c]],
                "candidate_labels": [int(y) for y in ev["y"][c]], "candidate_popularity": [int(pop[i]) for i in ci],
                "candidate_popularity_prior": [0] * 20, "candidate_timestamps": [int(t) for t in ev["ts"][c]],
                "source": domain, "domain_kind": "product" if amazon else "movie"}

    train_ids = [u for u in users if u.startswith("t")]
    eval_ids = [u for u in users if u.startswith("e")]
    train_rows, eval_rows = [row(u) for u in train_ids], [row(u) for u in eval_ids]
    T = float(np.quantile(np.concatenate([r["candidate_timestamps"] for r in train_rows + eval_rows]).astype(float),
                          quantile))
    d = root / "panels" / domain
    d.mkdir(parents=True)
    train = [filter_candidates(r, [t < T for t in r["candidate_timestamps"]]) for r in train_rows]
    train = [r for r in train if r["candidate_labels"]]
    sd_rows = []
    for r in eval_rows:
        yt = [y for y, t in zip(r["candidate_labels"], r["candidate_timestamps"]) if t >= T]
        if 0 < sum(yt) < len(yt):
            sd_rows.append(r)
    sd_test = [filter_candidates(r, [t >= T for t in r["candidate_timestamps"]]) for r in sd_rows]
    files = {"train.jsonl": write_jsonl(d / "train.jsonl", train), "eval.jsonl": write_jsonl(d / "eval.jsonl", eval_rows),
             "eval_sd_test.jsonl": write_jsonl(d / "eval_sd_test.jsonl", sd_test),
             "train_users.txt": ids_file(d / "train_users.txt", train_ids),
             "eval_users.txt": ids_file(d / "eval_users.txt", eval_ids),
             "sd_users.txt": ids_file(d / "sd_users.txt", [r["user_id"] for r in sd_rows])}
    for k in (0, 1):   # star-permuted panels: only their bytes matter here (data_sha1 identity)
        files[f"eval_sd_test_starperm{k}.jsonl"] = write_jsonl(
            d / f"eval_sd_test_starperm{k}.jsonl", [dict(r, source_event_id=r["source_event_id"] + f"::perm{k}")
                                                    for r in sd_test])
    if amazon:
        from src.confrec.pseudonymize import pseudonymize
        ps, pl, _ = pseudonymize(eval_rows, None, 0)
        write_jsonl(d / "eval_pseudo.jsonl", ps)
        write_jsonl(d / "eval_placebo.jsonl", pl)
    split = {"domain": domain, "panel_all_sha1": "0" * 40, "n_rows": n_train + n_eval, "variant": "V0",
             "quantile": quantile, "T": T,
             "train": {"users": n_train, "user_ids_sha1": fx.file_sha1(files["train_users.txt"])},
             "eval": {"users": n_eval, "user_ids_sha1": fx.file_sha1(files["eval_users.txt"])},
             "sd": {"users": len(sd_rows), "user_ids_sha1": fx.file_sha1(files["sd_users.txt"]),
                    "user_ids_path": "sd_users.txt"},
             "files": {k: fx.file_sha1(v) for k, v in sorted(files.items()) if not k.startswith("eval_sd_test_star")},
             "code_sha1": {}, "args": {"seed": 0, "n_train": n_train, "n_eval_max": n_eval, "s_max": 1000,
                                       "train_cap": 24000, "tokenizer_used": True, "tokenizer": "fake",
                                       "dev_users_sha1_checked": False, "gateft_split_checked": False},
             "gateft_T_match": None}
    (d / "ftgrid_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    if raw:
        rd = root / "raw" / "ml-1m"
        rd.mkdir(parents=True)
        with open(rd / "movies.dat", "w", encoding="latin-1") as f:
            for i in range(n_items):
                f.write(f"{iid[i]}::{title(i)}::Drama|Comedy\n")
        with open(rd / "ratings.dat", "w", encoding="latin-1") as f:
            for u, ev in users.items():
                for i, r, t in zip(ev["items"], ev["r"], ev["ts"]):
                    f.write(f"{u}::{iid[i]}::{int(r)}::{int(t)}\n")
    return SimpleNamespace(root=root, d=d, T=T, users=users, mu=mu, C=C, iid=iid, item_index={v: k for k, v in
                                                                                            enumerate(iid)},
                           eval_rows=eval_rows, sd_rows=sd_rows, sd_test=sd_test, pop=pop, domain=domain,
                           split=d / "ftgrid_split.json", raw=root / "raw" if raw else None)


def _pairs(rows):
    return [(r["source_event_id"], r["user_id"], it, c, y) for r in rows
            for c, (it, y) in enumerate(zip(r["candidate_item_ids"], r["candidate_labels"]))]


def _report(n, n2=0, mass=0.99, lora=None, panel=None, extra=None):
    rep = {"variant": "V0", "lora": lora, "model": "/models/Qwen3-8B", "backbone": "Qwen3-8B", "hist_len": 20,
           "data_sha1": fx.file_sha1(panel) if panel is not None else None, "mean_yes_no_mass": mass,
           "n_overlength": 0, "n_main_prompts": n, "censored_main": {"0": n - n2, "1": 0, "2": n2, "3": 0}}
    rep.update(extra or {})
    return rep


def score_dir(d: Path, panel: Path, pairs, logits, lora, cens2=(), mass=0.99, swap=None):
    """A pyes_scorer-format directory: scores.csv.gz (like rows), report.json, [swap_prior.csv.gz]."""
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, ((sid, u, it, c, y), lg) in enumerate(zip(pairs, logits)):
        bad = k in cens2
        rows.append([sid, u, it, c, y, "like", -1, -1, "nan" if bad else f"{lg:.6f}", mass, 2 if bad else 0, "nan"])
    gz(d / "scores.csv.gz", COLS, rows)
    extra = None
    if swap is not None:
        gz(d / "swap_prior.csv.gz", SWAP_COLS, swap)
        extra = {"swap_k": 8, "swap_prompts": len(swap), "censored_swap": {"0": len(swap), "1": 0, "2": 0, "3": 0},
                 "swap_n_overlength": 0}
    (d / "report.json").write_text(json.dumps(_report(len(rows), len(cens2), mass, lora, panel, extra)),
                                   encoding="utf-8")
    (d / "run.key").write_text(f"{fx.file_sha1(panel)} m {lora}\n", encoding="utf-8")
    return d


def write_models(w, specs: dict, arms=("like", "swap", "nohist", "starperm0", "starperm1"), seed=10, cens2=None):
    """Per model {alpha, beta, sigma}: L = alpha mu_i + beta a_u.c_i + 0.5 b_u-offset + N(0, sigma)."""
    sdir = w.root / "scores" / w.domain
    rng = np.random.default_rng(seed)
    off = {u: rng.normal(0, 1.0) for u in w.users}
    sd_ids = [r["user_id"] for r in w.sd_rows]
    for m, (name, p) in enumerate(specs.items()):
        r = np.random.default_rng(seed + 100 * (m + 1))
        lora = None if name == "zeroshot" else f"/adapters/{w.domain}/{name}"
        a, b, s = p.get("alpha", 1.0), p.get("beta", 0.5), p.get("sigma", 1.0)

        def logit(u, it, aa=None):
            i = w.item_index[it]
            au = w.users[u]["a"] if aa is None else aa
            return a * w.mu[i] + b * float(w.C[i] @ au) + off[u]
        ev_pairs = _pairs(w.eval_rows)
        base = {(u, it): logit(u, it) + r.normal(0, s) for _, u, it, _, _ in ev_pairs}
        boost = p.get("head_boost")       # familiarity: +boost on treated head items, the pseudo arm removes it
        if "like" in arms:
            score_dir(sdir / name / "like", w.d / "eval.jsonl", ev_pairs,
                      [base[(u, it)] + (boost(u, it) if boost else 0.0) for _, u, it, _, _ in ev_pairs], lora,
                      cens2=(cens2 or {}).get((name, "like"), ()))
        sd_pairs = _pairs(w.sd_test)
        if "swap" in arms:
            items = sorted({it for _, _, it, _, _ in sd_pairs})
            swap = []
            for it in items:
                for dnr in r.choice(sd_ids, 8, replace=False):
                    swap.append([it, dnr, "like", f"{logit(dnr, it) + r.normal(0, s):.6f}", 0])
            score_dir(sdir / name / "swap", w.d / "eval_sd_test.jsonl", sd_pairs,
                      [base[(u, it)] for _, u, it, _, _ in sd_pairs], lora, swap=swap)
        if "nohist" in arms:
            score_dir(sdir / name / "nohist", w.d / "eval_sd_test.jsonl", sd_pairs,
                      [a * w.mu[w.item_index[it]] + r.normal(0, 0.05) for _, u, it, _, _ in sd_pairs], lora)
        for k in (0, 1):
            arm = f"starperm{k}"
            if arm in arms:
                keep = p.get("perm_keep", 0.0)   # 1.0: the model ignores the history ratings
                vals = []
                for sid, u, it, c, y in sd_pairs:
                    i = w.item_index[it]
                    aa = r.normal(0, 1, 2)
                    personal = keep * float(w.C[i] @ w.users[u]["a"]) + (1 - keep) * float(w.C[i] @ aa)
                    vals.append(a * w.mu[i] + b * personal + off[u] + (base[(u, it)] - logit(u, it)))
                pairs_k = [(sid + f"::perm{k}", u, it, c, y) for sid, u, it, c, y in sd_pairs]
                score_dir(sdir / name / arm, w.d / f"eval_sd_test_starperm{k}.jsonl", pairs_k, vals, lora)
        for arm in ("pseudo", "placebo"):
            if arm in arms:
                vals = [base[(u, it)] + (boost(u, it) if boost and arm == "placebo" else 0.0)
                        for _, u, it, _, _ in ev_pairs]
                score_dir(sdir / name / arm, w.d / f"eval_{arm}.jsonl", ev_pairs, vals, lora,
                          cens2=(cens2 or {}).get((name, arm), ()))
    return w.root / "scores"


def run_main(w, out_name, models, n_boot=60, extra=()):
    out = w.root / "report" / f"{out_name}.json"
    args = ["--domain", w.domain, "--split", str(w.split), "--panels", str(w.d), "--scores_root", str(w.root / "scores"),
            "--models", ",".join(models), "--out", str(out), "--n_boot", str(n_boot), *extra]
    if w.raw is not None:
        args += ["--raw", str(w.raw)]
    return fr.main(args), out


ML1M_SPECS = {"zeroshot": {"beta": 0.3}, "s0": {"beta": 1.3}, "s1": {"beta": 1.3}, "s2": {"beta": 1.3},
              "p0": {"beta": 0.1}, "p1": {"beta": 0.1}}


@pytest.fixture(scope="module")
def ml1m(tmp_path_factory):
    w = make_world(tmp_path_factory.mktemp("ml1m"), "ml1m", seed=0)
    write_models(w, ML1M_SPECS)
    res, out = run_main(w, "ml1m", list(ML1M_SPECS), extra=("--refs", str(w.root / "report" / "ml1m_refs.json")))
    return SimpleNamespace(w=w, res=res, out=out)


def _strict_load(path):
    def bad(x):
        raise ValueError(f"non-strict JSON constant {x}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=bad)


# ---------------------------------------------------------------- shared statistics
def test_user_aucs_equal_forensics_uauc_with_ties():
    rng = np.random.default_rng(0)
    users = np.repeat([f"u{k}" for k in range(80)], 9)
    s = np.round(rng.normal(0, 1, len(users)), 1)            # coarse: many ties
    y = (rng.random(len(users)) < 0.4).astype(int)
    s[5] = np.nan
    mine = fr.user_aucs(s, y, users)
    ref = fx.uauc(s, y, fx._groups(users))
    assert mine.keys() == ref.keys() and all(mine[u] == pytest.approx(ref[u], abs=1e-12) for u in ref)
    mask = rng.random(len(users)) < 0.7
    assert fr.user_aucs(s, y, users, mask) == pytest.approx(fx.uauc(s, y, fx._groups(users), mask))


def test_bootstrap_draws_and_p_value_follow_the_registered_conventions():
    rng = np.random.default_rng(1)
    a = {f"u{k}": float(x) for k, x in enumerate(rng.normal(0.02, 0.1, 300))}
    D = fr.mean_draws(np.array([[a[k]] for k in sorted(a)]), 500, 0)[:, 0]
    pb = paired_bootstrap(a, {k: 0.0 for k in a}, n_boot=500, seed=0)
    r = fr.rec(np.mean(list(a.values())), D, 300, contrast=True)
    assert (r["lo"], r["hi"]) == pytest.approx((pb["lo"], pb["hi"]), abs=1e-12)
    # p = 2 min(P(d <= 0), P(d >= 0)) with +1/(B + 1) smoothing
    d = np.array([-1.0] * 30 + [1.0] * 470)
    assert fr.boot_p(d) == pytest.approx(2 * 31 / 501)
    assert fr.boot_p(np.ones(2000)) == pytest.approx(2 / 2001) and fr.boot_p(np.zeros(10)) == 1.0
    # row-level draws equal stats.cluster_bootstrap's
    cl = np.repeat(np.arange(40), rng.integers(1, 6, 40))
    x = rng.normal(0, 1, len(cl))
    mine = [float(x[i].mean()) for i in fr.Clusters(cl).draws(200, 3)]
    ref = cluster_bootstrap(lambda i: float(x[i].mean()), cl, n_boot=200, seed=3)
    assert np.quantile(mine, [0.025, 0.975]) == pytest.approx([ref["lo"], ref["hi"]], abs=1e-12)


def test_minimum_n_rule_removes_p_value_and_claims():
    d = np.r_[np.full(1990, 0.05), np.full(10, -0.01)]
    small, big = fr.rec(0.05, d, 149, contrast=True), fr.rec(0.05, d, 150, contrast=True)
    assert small["descriptive_min_n"] is True and small["p"] is None and small["ci_excludes_0"] is None
    assert small["lo"] == pytest.approx(big["lo"])                  # the interval stays as a description
    assert big["descriptive_min_n"] is False and big["p"] == pytest.approx(2 * 11 / 2001)
    assert big["ci_excludes_0"] is True
    assert "p" not in fr.rec(0.6, d, 400)                            # a level, not a contrast


def test_aurc_matches_metrics_risk_coverage():
    rng = np.random.default_rng(2)
    c = np.round(rng.random(300), 1)
    u = (rng.random(300) < 0.7).astype(float)
    assert fr.aurc(c, u) == pytest.approx(risk_coverage(c, u)[2], abs=1e-12)


# ---------------------------------------------------------------- forensics: the explicit cutoff
def test_cf_references_explicit_cutoff_never_sees_ratings_at_or_after_it(tmp_path):
    w = make_world(tmp_path, "ml1m", seed=3, n_eval=150, n_bg=80)
    P = fx.panel_pairs(w.eval_rows)
    g = fx._groups(P["user"])
    events = fx.load_raw_events(w.raw, "ml1m")
    kw = dict(mf_dim=4, mf_iters=4, n_boot=10)
    po = {}
    out, pm = fx.cf_references(P, g, events, cutoff=w.T, pairs_out=po, **kw)
    tm = out["mf_temporal"]
    assert tm["cutoff"]["T"] == w.T and tm["cutoff"]["explicit"] is True
    assert w.T != fx.mf_cutoff(P["ts"])                              # the split's T, not the panel quantile
    assert np.array_equal(po["test"], np.isfinite(P["ts"]) & (P["ts"] >= w.T))
    cand = set(zip(P["user"].tolist(), P["item"].tolist()))
    first = {}
    for u, t in zip(P["user"].tolist(), P["ts"].tolist()):
        first[u] = min(t, first.get(u, math.inf))

    def own_history(u, t):          # the user's own pre-candidate events: the fold-in (A3) may use them at any time
        return u in first and t < first[u]
    # every OTHER rating at or after T reversed (the candidates themselves excluded): item parameters never see one,
    # so the TEST pairs' MF is unchanged
    late = {u: [(t, i, 6.0 - r if t >= w.T and (u, i) not in cand and not own_history(u, t) else r)
                for t, i, r in evs] for u, evs in events.items()}
    assert late != events
    po2 = {}
    out2, _ = fx.cf_references(P, g, late, cutoff=w.T, pairs_out=po2, **kw)
    for k in ("mf_score", "mf_residual", "mf_item_bias", "mf_user_bias", "mf_warm"):
        assert np.array_equal(po[k], po2[k]), k
    assert json.dumps(fx.strict_json(out2["mf_temporal"]), sort_keys=True) == json.dumps(fx.strict_json(tm),
                                                                                         sort_keys=True)
    # the fold-in does read a user's own history after T (EVAL users whose history ends after T): documented
    hist_late = {u for u, evs in events.items() for t, i, r in evs if own_history(u, t) and t >= w.T}
    assert hist_late, "the synthetic world should hold users with post-T history"
    # a rating before T does enter
    early = {u: [(t, i, 6.0 - r if t < w.T and (u, i) not in cand else r) for t, i, r in evs]
             for u, evs in events.items()}
    po3 = {}
    fx.cf_references(P, g, early, cutoff=w.T, pairs_out=po3, **kw)
    assert not np.allclose(po["mf_score"], po3["mf_score"])
    # the default (no cutoff, no pairs_out) is untouched: the quantile T, no 'explicit' key
    base, _ = fx.cf_references(P, g, events, **kw)
    assert "explicit" not in base["mf_temporal"]["cutoff"] and base["mf_temporal"]["cutoff"]["T"] == fx.mf_cutoff(P["ts"])
    same, _ = fx.cf_references(P, g, events, cutoff=fx.mf_cutoff(P["ts"]), **kw)
    same["mf_temporal"]["cutoff"].pop("explicit")
    assert json.dumps(fx.strict_json(same), sort_keys=True) == json.dumps(fx.strict_json(base), sort_keys=True)
    # the personal residual is MF - mu - b_u - b_i (mu a constant)
    const = po["mf_score"] - po["mf_user_bias"] - po["mf_item_bias"] - po["mf_residual"]
    assert np.allclose(const, const[0])


# ---------------------------------------------------------------- E-B / seeds
def _ctx(users, y, **kw):
    uc = np.unique(np.asarray(users), return_inverse=True)[1].reshape(-1)
    return SimpleNamespace(uc=uc, y=np.asarray(y, int), **kw)


def _signal_panel(n_users=300, n_rows=12, seed=0):
    rng = np.random.default_rng(seed)
    users = np.repeat([f"u{k:04d}" for k in range(n_users)], n_rows)
    e = rng.normal(0, 1, len(users))
    y = (e + rng.normal(0, 1, len(users)) > 0).astype(int)
    return rng, users, e, y


def test_e_b_per_seed_contrasts_mean_sigma_and_failing_branches():
    rng, users, e, y = _signal_panel()
    zs = 0.3 * e + rng.normal(0, 1, len(e))
    ft = {f"s{k}": 1.2 * e + rng.normal(0, 1, len(e)) for k in range(3)}
    rows = np.ones(len(e), bool)
    out = fr.contrast_models({s: (v, zs) for s, v in ft.items()}, y, users, rows, 200, 0, n_registered=3)
    per = [out["per_seed"][s]["est"] for s in ft]
    assert all(p > 0.05 for p in per)
    assert out["mean_over_seeds"]["est"] == pytest.approx(np.mean(per))
    assert out["seeds"]["sigma_seed"] == pytest.approx(np.std(per, ddof=1))
    assert out["mean_over_seeds"]["p"] < 0.05 and out["seeds"]["sigma_seed_rule"] is True
    # identical users and rows: the seed-averaged UAUC equals the mean of the per-seed UAUCs
    ua = fr.uauc_models(ft, y, users, rows, 50, 0, n_registered=3)
    pu = [fr.user_aucs(v, y, users) for v in ft.values()]
    assert ua["mean_over_seeds"]["est"] == pytest.approx(np.mean([np.mean([d[u] for d in pu]) for u in pu[0]]))
    # one seed negative: the sign rule fails
    ft_bad = dict(ft, s2=-0.5 * e + rng.normal(0, 1, len(e)))
    bad = fr.contrast_models({s: (v, zs) for s, v in ft_bad.items()}, y, users, rows, 100, 0, n_registered=3)
    assert bad["per_seed"]["s2"]["est"] < 0 and bad["seeds"]["all_seeds_same_sign_as_mean"] is False
    assert bad["seeds"]["sigma_seed_rule"] is False
    # two seeds only: never complete
    two = fr.contrast_models({s: (ft[s], zs) for s in ("s0", "s1")}, y, users, rows, 50, 0, n_registered=3)
    assert two["seeds"]["complete"] is False and two["seeds"]["sigma_seed_rule"] is False


def test_p1_decision_rule_and_its_failing_branches():
    ok = {"est": 0.03, "p": 0.01, "descriptive_min_n": False}
    assert fr.p1_decision([0.028, 0.03, 0.032], ok)["verdict"] == "P1_HOLDS"
    neg = fr.p1_decision([0.05, 0.05, -0.01], {"est": 0.03, "p": 0.01, "descriptive_min_n": False})
    assert neg["verdict"] == "NO_EVIDENCE" and neg["conditions"]["all_seeds_positive"] is False
    noisy = fr.p1_decision([0.001, 0.06, 0.029], {"est": 0.03, "p": 0.01, "descriptive_min_n": False})
    assert noisy["conditions"]["mean_gt_2_sigma_seed"] is False and noisy["holds"] is False
    assert noisy["conditions"]["all_seeds_positive"] is True
    nop = fr.p1_decision([0.028, 0.03, 0.032], {"est": 0.03, "p": 0.08, "descriptive_min_n": False})
    assert nop["conditions"]["p_lt_0.05"] is False and nop["verdict"] == "NO_EVIDENCE"
    desc = fr.p1_decision([0.028, 0.03, 0.032], {"est": 0.03, "p": None, "descriptive_min_n": True})
    assert desc["holds"] is False and desc["conditions"]["not_descriptive_min_n"] is False
    assert fr.p1_decision([0.03, 0.03], ok)["verdict"] == "INCOMPLETE"
    assert "no evidence" in neg["wording"]


# ---------------------------------------------------------------- E-C
def test_topk_anatomy_known_answer_and_ties_decided_by_the_midpoint():
    L = np.array([3.0, 2.0, 1.0, 0.0, 5.0, 5.0, 5.0, 1.0])
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    users = np.array(["a"] * 4 + ["b"] * 4)
    m, c, cu = fr.topk_anatomy(L, y, users, np.ones(8, bool))
    assert cu[:4].tolist() == [1.5] * 4 and m[:4].tolist() == [1.5, 0.5, 0.5, 1.5]
    assert c[:4].tolist() == [1.0, 0.0, 0.0, 1.0]
    # user b: k = 1, the 1st and 2nd largest are tied at 5 -> c_u = 5, all three tied rows are decided 1
    assert cu[4] == 5.0 and c[4:].tolist() == [1.0, 0.0, 0.0, 1.0] and m[4:7].tolist() == [0.0, 0.0, 0.0]


def test_error_anatomy_recovers_a_planted_confident_error_structure():
    rng = np.random.default_rng(4)
    users = np.repeat([f"u{k:03d}" for k in range(400)], 10)
    y = np.tile([1] * 4 + [0] * 6, 400)
    L = 1.5 * (2 * y - 1) + rng.normal(0, 1, len(y))
    flip = rng.random(len(y)) < 0.05                                  # confidently wrong rows: large |L| of the wrong sign
    L[flip] = -4.0 * (2 * y[flip] - 1)
    m, c, _ = fr.topk_anatomy(L, y, users, np.ones(len(y), bool))
    st = dict(zip(fr.ANATOMY_KEYS, fr.anatomy_stats(m, c)))
    t = rank_bins(m, 3)
    w = 1 - c
    assert st["P_wrong_bottom"] == pytest.approx(w[t == 0].mean()) and st["P_wrong_top"] == pytest.approx(w[t == 2].mean())
    assert st["share_errors_top"] == pytest.approx(w[t == 2].sum() / w.sum())
    assert st["share_correct_bottom"] == pytest.approx(c[t == 0].sum() / c.sum())
    assert st["AUROC_margin_correct"] == pytest.approx(auroc(m, c.astype(int)))
    assert st["AURC"] == pytest.approx(risk_coverage(m, c)[2])
    assert st["share_errors_top"] > 1 / 3 and st["P_wrong_top"] > st["P_wrong_middle"]   # the planted S2 errors
    clean = 1.5 * (2 * y - 1) + rng.normal(0, 1, len(y))
    m2, c2, _ = fr.topk_anatomy(clean, y, users, np.ones(len(y), bool))
    st2 = dict(zip(fr.ANATOMY_KEYS, fr.anatomy_stats(m2, c2)))
    assert st2["P_wrong_bottom"] > st2["P_wrong_middle"] > st2["P_wrong_top"] and st2["AUROC_margin_correct"] > 0.6


def test_platt_map_is_fit_on_cal_rows_only_and_applied_to_test():
    rng = np.random.default_rng(5)
    n_users, n = 300, 16
    users = np.repeat([f"u{k:03d}" for k in range(n_users)], n)
    L = rng.normal(0, 2, len(users))
    cal = np.tile(np.arange(n) < 10, n_users)
    test = ~cal
    y = np.where(cal, rng.random(len(L)) < sigmoid(0.8 * L - 0.5), rng.random(len(L)) < sigmoid(0.2 * L + 1.0)).astype(int)
    cx = _ctx(users, y, cal=cal, test=test)
    out = fr.ec_block(cx, {"m": L}, 30, 0, 1)["per_model"]["m"]
    a, b = platt_fit(L[cal], y[cal])
    assert out["platt_slope"]["est"] == pytest.approx(a) and out["platt_intercept"]["est"] == pytest.approx(b)
    assert a == pytest.approx(0.8, abs=0.1) and a != pytest.approx(platt_fit(L, y)[0], abs=0.05)
    p = sigmoid(a * L[test] + b)
    assert out["ECE"]["est"] == pytest.approx(ece(p, y[test], 10, adaptive=True))
    base = y[cal].mean()
    assert out["Brier_skill"]["est"] == pytest.approx(1 - np.mean((p - y[test]) ** 2) / np.mean((base - y[test]) ** 2))
    assert sum(r["share"] * r["abs_gap"] for r in out["reliability_equal_mass"]) == pytest.approx(out["ECE"]["est"])
    assert len(out["reliability_equal_width"]) <= 10 and out["n_cal_rows"] == int(cal.sum())
    # TEST labels never move the map; CAL labels do
    y2 = y.copy()
    y2[test] = 1 - y2[test]
    o2 = fr.ec_block(_ctx(users, y2, cal=cal, test=test), {"m": L}, 0, 0, 1)["per_model"]["m"]
    assert o2["platt_slope"]["est"] == pytest.approx(a)
    y3 = y.copy()
    y3[cal] = 1 - y3[cal]
    o3 = fr.ec_block(_ctx(users, y3, cal=cal, test=test), {"m": L}, 0, 0, 1)["per_model"]["m"]
    assert o3["platt_slope"]["est"] == pytest.approx(-a, abs=1e-6)


# ---------------------------------------------------------------- E-D
def _decomposition(n_users=800, n_rows=15, n_items=800, beta=1.0, gamma=1.0, sd_eps=0.7, sd_donor=1.0, seed=6):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1, n_items)
    donors = beta * mu[:, None] + rng.normal(0, sd_donor, (n_items, 8))
    users = np.repeat([f"u{k:04d}" for k in range(n_users)], n_rows)
    items = rng.integers(0, n_items, len(users))
    off = np.repeat(rng.normal(0, 2, n_users), n_rows)
    prior, pers, eps = beta * mu[items], gamma * rng.normal(0, 1, len(users)), rng.normal(0, sd_eps, len(users))
    L = off + prior + pers + eps
    pi, pa, pb = donors.mean(1)[items], donors[:, :4].mean(1)[items], donors[:, 4:].mean(1)[items]

    def centred(x):
        g = fx._groups(users)
        out = np.empty(len(x))
        for idx in g.values():
            out[idx] = x[idx] - x[idx].mean()
        return out
    true_share = float((centred(prior) ** 2).sum() / (centred(L) ** 2).sum())
    true_r8 = beta ** 2 / (beta ** 2 + sd_donor ** 2 / 8)
    return users, L, pi, pa, pb, true_share, true_r8


def test_item_prior_share_recovers_the_planted_share_and_reliability():
    users, L, pi, pa, pb, share, r8 = _decomposition()
    out = fr.shares_block(users, np.ones(len(L), bool), {"m": (pa, pb, L, pi)}, 200, 0, 1)
    m = out["per_model"]["m"]
    assert m["item_prior_share"]["est"] == pytest.approx(share, abs=0.03)
    assert m["r8c"]["est"] == pytest.approx(r8, abs=0.03) and m["shares_reading"] == "interpretable"
    assert m["non_prior_share"]["est"] == pytest.approx(1 - m["item_prior_share"]["est"])
    assert m["item_prior_share"]["lo"] < share < m["item_prior_share"]["hi"]
    assert m["r8c"]["est"] == pytest.approx(2 * m["r_c"]["est"] / (1 + m["r_c"]["est"]))
    # noisy donors: the reliability drops below 0.7 and the shares are flagged, still unclipped
    users, L, pi, pa, pb, share, r8 = _decomposition(sd_donor=4.0)
    noisy = fr.shares_block(users, np.ones(len(L), bool), {"m": (pa, pb, L, pi)}, 50, 0, 1)["per_model"]["m"]
    assert noisy["r8c"]["est"] < 0.7 and noisy["shares_reading"] == "uninterpretable"
    assert noisy["item_prior_share"]["est"] == pytest.approx(share, abs=0.15)


def _stack_world(gamma, cf_signal, n_users=600, n_rows=10, seed=7):
    rng = np.random.default_rng(seed)
    users = np.repeat([f"u{k:04d}" for k in range(n_users)], n_rows)
    mu = rng.normal(0, 1, len(users))
    e = rng.normal(0, 1, len(users))
    y = (rng.random(len(users)) < sigmoid(1.2 * mu + 1.2 * e + np.repeat(rng.normal(0, 0.5, n_users), n_rows))
         ).astype(int)
    q = mu + rng.normal(0, 0.3, len(users))
    pi = mu + rng.normal(0, 0.3, len(users))
    L = pi + gamma * e + rng.normal(0, 0.5, len(users))
    mf = (e if cf_signal else rng.normal(0, 1, len(users))) + rng.normal(0, 0.5, len(users))
    sd = sorted(set(users.tolist()))
    from src.confrec.stats import user_halves
    fa = user_halves(sd, 0)
    cx = _ctx(users, y, fold_a=np.array([u in fa for u in users], bool))
    return cx, L, pi, q, mf


def test_information_gain_positive_with_personal_evidence_and_null_with_noise():
    cx, L, pi, q, mf = _stack_world(1.5, True)
    rows = np.ones(len(L), bool)
    g = fr.stacker_block(cx, ["m"], {"m": L}, {"m": pi}, q, mf, rows, 200, 0, 1)
    assert g["per_model"]["m"]["G"]["est"] > 0.05 and g["per_model"]["m"]["G"]["lo"] > 0
    assert g["G_CF"]["est"] > 0.05 and g["G_over_G_CF_reported"] is True
    assert g["G_over_G_CF"]["m"]["est"] == pytest.approx(g["per_model"]["m"]["G"]["est"] / g["G_CF"]["est"])
    u = g["per_model"]["m"]["UAUC"]
    assert g["per_model"]["m"]["G"]["est"] == pytest.approx(u["M2"] - u["M1"])
    assert g["rows"]["fold_a_users"] + g["rows"]["fold_b_users"] == 600
    cx, L, pi, q, mf = _stack_world(0.0, False, seed=8)
    n = fr.stacker_block(cx, ["m"], {"m": L}, {"m": pi}, q, mf, rows, 200, 0, 1)
    assert abs(n["per_model"]["m"]["G"]["est"]) < 0.01 and n["per_model"]["m"]["G"]["lo"] < 0 < n["per_model"]["m"]["G"]["hi"]
    assert abs(n["G_CF"]["est"]) < 0.01
    if not n["G_CF"]["ci_excludes_0"]:
        assert n["G_over_G_CF_reported"] is False and n["G_over_G_CF"]["m"] is None


def test_crossfit_scores_each_fold_with_the_other_folds_fit():
    rng = np.random.default_rng(9)
    x = rng.normal(0, 1, 400)
    y = (x + rng.normal(0, 1, 400) > 0).astype(int)
    fold_a = np.arange(400) < 200
    eta = fr.crossfit(x[:, None], y, fold_a, np.ones(400, bool))
    fit_b = fr.logit_fit(x[~fold_a, None], y[~fold_a])
    assert np.allclose(eta[fold_a], fr.logit_eta(fit_b, x[fold_a, None]))
    y_flip = y.copy()
    y_flip[fold_a] = 1 - y_flip[fold_a]                     # fold A's own labels never enter fold A's scores
    assert np.allclose(fr.crossfit(x[:, None], y_flip, fold_a, np.ones(400, bool))[fold_a], eta[fold_a])


def test_star_permutation_detects_rating_use_and_flags_a_model_that_ignores_ratings():
    rng, users, e, y = _signal_panel(seed=10)
    rows = np.ones(len(e), bool)
    L = e + rng.normal(0, 1, len(e))
    perm = [rng.normal(0, 1, len(e)) + (L - e) for _ in range(2)]          # the personal part destroyed
    cx = _ctx(users, y)
    uses = fr.starperm_block(cx, ["m"], {"m": L}, {"m": perm}, rows, 100, 0, 1)["per_model"]["m"]
    assert uses["dUAUC_L_minus_perm"]["est"] > 0.1 and uses["wording_flag_model_ignores_user_ratings"] is False
    same = [L + rng.normal(0, 0.01, len(e)) for _ in range(2)]
    ign = fr.starperm_block(cx, ["m"], {"m": L}, {"m": same}, rows, 100, 0, 1)["per_model"]["m"]
    assert abs(ign["dUAUC_L_minus_perm"]["est"]) < 0.005 and ign["tau_sd_within_user"] < 0.1
    assert ign["wording_flag_model_ignores_user_ratings"] is True
    tau = L - np.mean(perm, 0)
    assert uses["tau_sd_within_user"] == pytest.approx(fr.pooled_within_sd(tau, users))
    pu = fr.user_aucs(L, y, users)
    pk = [fr.user_aucs(p, y, users) for p in perm]
    assert uses["dUAUC_L_minus_perm"]["est"] == pytest.approx(np.mean([pu[u] - np.mean([d[u] for d in pk]) for u in pu]))


# ---------------------------------------------------------------- E-E
def test_partial_spearman_and_bias_index_blocks_use_the_registered_estimators():
    rng = np.random.default_rng(11)
    n_users, n = 300, 12
    users = np.repeat([f"u{k:03d}" for k in range(n_users)], n)
    pop = rng.integers(1, 2000, len(users)).astype(float)
    q = np.log1p(pop) + rng.normal(0, 1, len(users))
    Z = q[:, None]
    L_pop = 0.6 * np.log1p(pop) + rng.normal(0, 1, len(users))           # popularity beyond q-hat
    L_q = 0.6 * q + rng.normal(0, 1, len(users))                          # popularity only through q-hat
    y = (rng.random(len(users)) < 0.5).astype(int)
    b5 = rank_bins(pop, 5)
    cx = _ctx(users, y, lpop=np.log1p(pop), b5=b5, head=b5 == 4, tail=b5 == 0)
    rows = np.ones(len(users), bool)
    out = fr.partial_pair_block(cx, ["pop", "q"], {"pop": L_pop, "q": L_q}, Z, rows, 50, 0)
    assert out["per_model"]["pop"]["est"] == pytest.approx(fx.partial_spearman(L_pop, np.log1p(pop), Z))
    assert out["per_model"]["pop"]["est"] > 0.3 and abs(out["per_model"]["q"]["est"]) < 0.05
    L = rng.normal(0, 0.5, len(users))                                    # confidence independent of popularity
    yb = (rng.random(len(users)) < np.where(b5 == 4, 0.25, 0.6)).astype(int)   # head items: less accurate
    cx.y = yb
    bi = fr.bias_index_block(cx, ["m"], {"m": L}, {"m": (1.0, 0.0)}, rows, 50, 0)["per_model"]["m"]
    grp = np.where(b5 == 4, "head", np.where(b5 == 0, "tail", "mid"))
    ref = bias_index(sigmoid(L), yb, grp, 10, True)
    assert bi["head"]["est"] == pytest.approx(ref["head"]) and bi["tail"]["est"] == pytest.approx(ref["tail"])
    assert bi["head_minus_tail"]["est"] == pytest.approx(ref["head"] - ref["tail"]) and bi["head"]["est"] < 0
    assert bi["head_minus_tail"]["hi"] < 0 and bi["head_minus_tail"]["p"] is not None


# ---------------------------------------------------------------- knockout labels
def test_decide_labels_are_mapped_by_the_closed_hole_rule():
    nan = float("nan")
    ci = lambda e, lo, hi: {"est": e, "lo": lo, "hi": hi}  # noqa: E731
    pos = pps.decide(0.0, 0.0, ci(0.3, 0.1, 0.5), ci(0.01, -0.02, 0.04), ci(0.1, 0.05, 0.2))
    amb = pps.decide(0.0, 0.0, ci(0.1, -0.1, 0.3), ci(0.3, 0.1, 0.4), ci(0.1, 0.05, 0.2))
    und = pps.decide(0.0, nan, ci(0.3, 0.1, 0.5), ci(0.01, -0.02, 0.04), ci(0.1, 0.05, 0.2))
    nul = pps.decide(0.05, 0.05, ci(0.3, 0.1, 0.5), ci(0.0, 0, 0), ci(0.1, 0.05, 0.2))
    assert [d["verdict"] for d in (pos, amb, und, nul)] == ["POSITIVE", "AMBIGUOUS", "UNDETERMINED", "NULL"]
    assert [fr.knockout_label(d["verdict"]) for d in (pos, amb, und, nul)] == ["POSITIVE", "INDETERMINATE",
                                                                              "INCOMPLETE", "NULL"]
    assert fr.knockout_label("NEGATIVE") == "NEGATIVE"


def test_knockout_runs_pilot_pseudonym_on_rows_finite_in_every_arm_of_every_model(tmp_path):
    w = make_world(tmp_path, "toys", seed=12, n_eval=200, n_bg=40, raw=False)
    treated = {}
    for r in fx.read_jsonl(w.d / "eval_pseudo.jsonl"):
        for it, f in zip(r["candidate_item_ids"], r["candidate_brand_in_title"]):
            treated[(r["user_id"], it)] = f
    head = {w.iid[i] for i in np.argsort(-w.pop)[: len(w.pop) // 5]}

    def boost(u, it):
        return 0.6 if treated.get((u, it)) and it in head else 0.0
    specs = {m: {"beta": 0.5, "head_boost": boost} for m in ("zeroshot", "s0", "s1", "s2")}
    n_pairs = sum(len(r["candidate_item_ids"]) for r in w.eval_rows)
    write_models(w, specs, arms=("like", "pseudo", "placebo"), cens2={("s1", "pseudo"): (3, 17, 40)})
    res, out = run_main(w, "toys", list(specs), n_boot=20, extra=("--bi_boot", "5"))
    ko = res["knockout"]
    assert ko["available"] is True and ko["n_common_rows"] == n_pairs - 3
    assert {m: v["n_analysed"] for m, v in ko["labels"].items()} == {m: n_pairs - 3 for m in specs}
    for m, v in ko["labels"].items():
        assert v["label"] == fr.knockout_label(v["decide_verdict"])
        assert v["label"] in ("NULL", "POSITIVE", "NEGATIVE", "INDETERMINATE", "INCOMPLETE")
        assert v["values"]["hmt_pseudo"]["est"] > 0.3            # the planted head familiarity
    # no raw data and no swap / nohist / starperm arms: null blocks with reasons, never a crash
    assert res["E_D"]["ZS"]["information_gain"]["available"] is False
    assert res["E_D"]["ZS"]["star_permutation"]["available"] is False
    assert res["E_E"]["ZS"]["partial_spearman"]["available"] is False
    assert "references" in res and res["P1"]["decision"]["verdict"] == "INCOMPLETE"
    assert res["E_A"]["ZS"]["UAUC_TEST"]["references"].keys() == {"popularity"}
    rows = list(csv.DictReader(open(out.with_name("toys_tables.csv"), encoding="utf-8")))
    assert {r["model"] for r in rows if r["block"] == "FT-K"} == set(specs)


# ---------------------------------------------------------------- end to end (synthetic ML-1M, every arm)
def test_report_schema_strict_json_tables_and_meta(ml1m):
    res, out = ml1m.res, ml1m.out
    loaded = _strict_load(out)
    assert loaded == res
    for k in ("meta", "input_checks", "excluded_runs", "runs", "references", "operationalizations", "E_A", "E_B", "E_C",
              "E_D", "E_E", "P1", "knockout", "FT_C"):
        assert k in res, k
    meta = res["meta"]
    assert meta["T"] == ml1m.w.T and meta["variant"] == "V0" and meta["split_args"]["s_max"] == 1000
    assert meta["backbone"] == "Qwen3-8B" and meta["regimes"]["FT"]["models"] == ["s0", "s1", "s2"]
    assert res["input_checks"]["problems"] == [] and res["excluded_runs"] == []
    assert res["input_checks"]["sd_user_ids_sha1_match"] is True and res["input_checks"]["eval_user_ids_sha1_match"]
    assert res["input_checks"]["eval_sd_test_pairs_equal_sd_test_rows"] is True
    assert res["references"]["source"] == "computed from --raw" and res["references"]["cache_written"] is True
    assert res["knockout"]["available"] is False                          # ml1m has no pseudonym panels
    text = out.read_text(encoding="utf-8")                                 # _strict_load refused NaN / Infinity
    assert json.dumps(str(ml1m.w.root))[1:-1] not in text and ml1m.w.root.name not in text   # no path in the json
    rows = list(csv.DictReader(open(out.with_name("ml1m_tables.csv"), encoding="utf-8")))
    assert list(rows[0]) == list(fr.CSV_COLS)
    ends = {(r["block"], r["endpoint"]) for r in rows}
    for e in [("E-A", "UAUC_TEST"), ("E-A", "UAUC_TEST_q_hat"), ("E-A", "UAUC_TEST_mf_personal_residual"),
              ("E-B", "dUAUC_FT_minus_ZS"), ("E-C", "ECE"), ("E-C", "share_errors_top"), ("E-D", "item_prior_share"),
              ("E-D", "G"), ("E-D", "G_CF"), ("E-D", "starperm_dUAUC"), ("E-E", "partial_spearman_L_pair"),
              ("E-E", "bias_index_head_minus_tail"), ("P1", "G_FT_minus_G_ZS"),
              ("FT-C", "dUAUC_real_minus_permuted_TEST")]:
        assert e in ends, e


def test_report_is_deterministic_and_reuses_the_refs_cache(ml1m):
    w = ml1m.w
    res2, out2 = run_main(w, "ml1m_again", list(ML1M_SPECS), extra=("--refs", str(w.root / "report" / "ml1m_refs.json")))
    assert res2["references"]["source"] == "cache (--refs)"
    a, b = dict(ml1m.res), dict(res2)
    a.pop("references"), b.pop("references")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    t1 = ml1m.out.with_name("ml1m_tables.csv").read_bytes()
    assert out2.with_name("ml1m_again_tables.csv").read_bytes() == t1


def test_e_a_references_secondary_splits_and_min_n(ml1m):
    ea = ml1m.res["E_A"]["FT"]
    u = ea["UAUC_TEST"]
    assert set(u["per_model"]) == {"s0", "s1", "s2"} and u["seeds"]["n_seeds"] == 3
    assert u["mean_over_seeds"]["est"] == pytest.approx(np.mean([u["per_model"][m]["est"] for m in u["per_model"]]))
    assert set(u["references"]) == {"q_hat", "popularity", "mf", "mf_personal_residual"}
    assert all(r["n_users"] == u["rows"]["n_users"] for r in u["references"].values())   # identical users and rows
    assert u["references"]["q_hat"]["est"] > 0.55                     # the planted item prior
    assert u["rows"]["n_users"] >= fr.MIN_N and u["mean_over_seeds"]["descriptive_min_n"] is False
    sec = ea["secondary"]
    assert sec["UAUC_all_rows"]["rows"]["n_pairs"] > u["rows"]["n_pairs"]
    n_seen, n_unseen = sec["UAUC_TEST_seen"]["rows"]["n_users"], sec["UAUC_TEST_unseen"]["rows"]["n_users"]
    for blk in (sec["UAUC_TEST_seen"], sec["UAUC_TEST_unseen"]):
        n = blk["rows"]["n_users"]
        assert blk["mean_over_seeds"]["descriptive_min_n"] is (n < fr.MIN_N)
    assert n_seen + n_unseen > 0
    # FT beats ZS by construction (more personal evidence), seed by seed, with identical users
    eb = ml1m.res["E_B"]
    assert eb["complete"] is True and all(eb["per_seed"][s]["est"] > 0 for s in ("s0", "s1", "s2"))
    assert eb["mean_over_seeds"]["est"] == pytest.approx(np.mean([eb["per_seed"][s]["est"] for s in eb["per_seed"]]))
    assert eb["seeds"]["sigma_seed"] == pytest.approx(np.std([eb["per_seed"][s]["est"] for s in eb["per_seed"]], ddof=1))


def test_e_d_p1_and_ft_c_end_to_end(ml1m):
    res = ml1m.res
    ed = res["E_D"]["FT"]
    assert ed["models_per_sub_block"] == {"swap": ["s0", "s1", "s2"], "nohist": ["s0", "s1", "s2"],
                                          "star_permutation": ["s0", "s1", "s2"]}
    assert ed["swap_arm_like_consistency"]["s0"]["max_abs_diff"] < 1e-5
    sh = ed["shares"]["mean_over_seeds"]
    assert sh["shares_reading"] == ("uninterpretable" if sh["r8c"]["est"] < 0.7 else "interpretable")
    for m, blk in ed["shares"]["per_model"].items():
        assert blk["r8c"]["est"] == pytest.approx(2 * blk["r_c"]["est"] / (1 + blk["r_c"]["est"]))
    assert sh["item_prior_share"]["est"] == pytest.approx(np.mean([ed["shares"]["per_model"][m]["item_prior_share"]["est"]
                                                                  for m in ("s0", "s1", "s2")]))
    zs = res["E_D"]["ZS"]["information_gain"]["per_model"]["zeroshot"]["G"]["est"]
    ft = res["E_D"]["FT"]["information_gain"]["G_mean_over_seeds"]["est"]
    assert ft > zs
    assert res["E_D"]["FT"]["star_permutation"]["per_model"]["s0"]["dUAUC_L_minus_perm"]["est"] > 0
    p1 = res["P1"]
    assert p1["available"] is True and p1["role"].startswith("P1 confirmatory")
    assert p1["decision"]["verdict"] == "P1_HOLDS" and p1["mean_sign"] == 1
    assert p1["mean_over_seeds"]["est"] == pytest.approx(np.mean([p1["per_seed"][s]["est"] for s in p1["per_seed"]]))
    ftc = res["FT_C"]
    assert ftc["seed0"]["TEST"]["per_seed"]["s0"]["est"] > 0 and ftc["seed1"]["all_rows"]["per_seed"]["s1"]["est"] > 0
    assert set(ftc["seed0"]["references_TEST"]["references"]) == {"q_hat", "popularity", "mf", "mf_personal_residual"}
    ec = res["E_C"]["FT"]
    assert set(ec["per_model"]) == {"s0", "s1", "s2"} and ec["per_model"]["s0"]["ECE"]["n_boot"] == 60
    ee = res["E_E"]["FT"]
    assert ee["controls"] == ["q_hat", "release_year"] and "pi_item" in ee["partial_spearman"]


def test_failed_integrity_run_is_excluded_and_never_replaced(tmp_path):
    w = make_world(tmp_path, "ml1m", seed=13, n_eval=300, n_bg=60)
    n_pairs = sum(len(r["candidate_item_ids"]) for r in w.eval_rows)
    specs = {"zeroshot": {"beta": 0.3}, "s0": {"beta": 1.0}, "s1": {"beta": 1.0}, "s2": {"beta": 1.0}}
    write_models(w, specs, arms=("like", "swap"), cens2={("s2", "like"): tuple(range(0, n_pairs, 20))})   # 5%
    res, _ = run_main(w, "bad", list(specs), n_boot=20)
    ex = res["excluded_runs"]
    assert [(e["model"], e["arm"], e["status"]) for e in ex] == [("s2", "like", "FAILED_INTEGRITY")]
    assert res["runs"]["s2"]["like"]["integrity"]["censored2_share_le_0.005"] is False
    assert res["E_A"]["FT"]["models"] == ["s0", "s1"] and res["E_A"]["FT"]["complete"] is False
    assert res["E_B"]["complete"] is False and res["E_B"]["seeds"]["sigma_seed_rule"] is False
    assert res["P1"]["decision"]["verdict"] == "INCOMPLETE"
    assert res["E_D"]["FT"]["star_permutation"]["available"] is False      # absent arms: null with a reason
    assert res["runs"]["s0"]["nohist"]["status"] == "ABSENT"
    assert res["FT_C"]["seed0"]["available"] is False
