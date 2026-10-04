import csv
import gzip
import json

import numpy as np
import pytest

from src.confrec import gateft_data as gd
from src.confrec import gateft_eval as ge


def panel_rows(prefix, n_users, n_cands=12, seed=0, t0=0):
    """Rated-panel-like rows with candidate_timestamps spread over time (user i's candidates cluster around t0 + 100 i)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_users):
        ts = sorted(int(t0 + 100 * i + rng.integers(0, 90)) for _ in range(n_cands))
        lab = [int(x) for x in (rng.random(n_cands) < 0.5)]
        lab[0], lab[1] = 1, 0                                      # both classes present
        rows.append({"user_id": f"{prefix}{i}", "source_event_id": f"{prefix}{i}::{ts[0]}", "history": ["h (rated 4/5)"],
                     "candidate_item_ids": [f"i{j}" for j in range(n_cands)],
                     "candidate_titles": [f"t{j}" for j in range(n_cands)],
                     "candidate_texts": ["" for _ in range(n_cands)], "candidate_labels": lab,
                     "candidate_timestamps": ts, "candidate_popularity": [5] * n_cands})
    return rows


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_train_is_dev_users_before_T_and_disjoint_from_confirm(tmp_path):
    dev, conf = panel_rows("d", 30, seed=1), panel_rows("c", 20, seed=2, t0=500)
    a, b = write_jsonl(tmp_path / "dev.jsonl", dev), write_jsonl(tmp_path / "conf.jsonl", conf)
    rep = gd.main(["--dev_panel", str(a), "--confirm_panel", str(b), "--out_dir", str(tmp_path / "o")])
    all_ts = np.concatenate([r["candidate_timestamps"] for r in dev + conf])
    assert rep["T"] == pytest.approx(float(np.quantile(all_ts, 0.8)))
    train = [json.loads(line) for line in open(tmp_path / "o" / "train.jsonl", encoding="utf-8")]
    assert train and {r["user_id"][0] for r in train} == {"d"}                     # DEV users only
    assert all(t < rep["T"] for r in train for t in r["candidate_timestamps"])      # nothing at or after T
    for r in train:                                                                 # per-candidate lists stay aligned
        n = len(r["candidate_labels"])
        assert all(len(r[k]) == n for k in r if k.startswith("candidate_") and isinstance(r[k], list))
    n_dev_before = sum(t < rep["T"] for r in dev for t in r["candidate_timestamps"])
    assert rep["train"]["candidates"] == n_dev_before
    n_post = sum(t >= rep["T"] for r in conf for t in r["candidate_timestamps"])
    assert rep["eval"]["post_T_candidates"] == n_post and rep["eval"]["confirm_users"] == 20
    assert (tmp_path / "o" / "gateft_split.json").exists()


def test_shared_users_are_refused(tmp_path):
    dev, conf = panel_rows("d", 5), panel_rows("d", 5, seed=3)
    with pytest.raises(SystemExit, match="share"):
        gd.main(["--dev_panel", str(write_jsonl(tmp_path / "a.jsonl", dev)), "--confirm_panel",
                 str(write_jsonl(tmp_path / "b.jsonl", conf)), "--out_dir", str(tmp_path / "o")])


def write_run(d, panel, lora, signal, seed, variant="V0", mass=0.999, cens2=0.0, n_over=0):
    """pyes_scorer-format like scores: logit = signal * (2 y - 1) + noise."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(d / "scores.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_event_id", "user_id", "item_id", "cand_idx", "label", "question", "lp_yes", "lp_no", "logit",
                    "yes_no_mass", "censored"])
        for r in panel:
            for c, y in enumerate(r["candidate_labels"]):
                lg = signal * (2 * y - 1) + rng.normal()
                code = 2 if rng.random() < cens2 else 0
                w.writerow([r["source_event_id"], r["user_id"], r["candidate_item_ids"][c], c, y, "like", -1, -1,
                            "nan" if code else f"{lg:.6f}", mass, code])
                n += 1
    (d / "report.json").write_text(json.dumps({"variant": variant, "lora": lora, "model": "m", "hist_len": 10,
                                               "data_sha1": "x", "mean_yes_no_mass": mass, "n_overlength": n_over,
                                               "n_main_prompts": n}), encoding="utf-8")
    return d


def setup(tmp_path):
    dev, conf = panel_rows("d", 40, seed=1), panel_rows("c", 120, seed=2, t0=0)
    a, b = write_jsonl(tmp_path / "dev.jsonl", dev), write_jsonl(tmp_path / "conf.jsonl", conf)
    gd.main(["--dev_panel", str(a), "--confirm_panel", str(b), "--out_dir", str(tmp_path / "o"), "--quantile", "0.3"])
    return conf, b, tmp_path / "o" / "gateft_split.json"


def test_gate_ft_pass_and_fail_follow_the_registered_constant(tmp_path):
    conf, cp, sr = setup(tmp_path)
    strong = [write_run(tmp_path / f"s{k}", conf, f"ad{k}", 2.0, k) for k in range(3)]
    res = ge.evaluate(cp, sr, [str(d) for d in strong], n_boot=50)
    assert res["decision"] == "GATE_FT_PASS" and res["UAUC_post_T_mean_over_seeds"] >= 0.65
    assert len(res["UAUC_post_T_per_seed"]) == 3 and res["UAUC_post_T_seed_averaged_ci95"]["n_users"] > 10
    weak = [write_run(tmp_path / f"w{k}", conf, f"wd{k}", 0.1, k + 10) for k in range(3)]
    res = ge.evaluate(cp, sr, [str(d) for d in weak], n_boot=50)
    assert res["decision"] == "GATE_FT_FAIL" and res["UAUC_post_T_mean_over_seeds"] < 0.65


def test_post_T_rows_are_selected_by_the_panel_timestamps(tmp_path):
    """Informative only before T: the post-T UAUC must be near 0.5 while the all-CONFIRM UAUC is clearly above."""
    conf, cp, sr = setup(tmp_path)
    T = json.loads(sr.read_text())["T"]
    rng = np.random.default_rng(5)
    d = tmp_path / "pre"
    d.mkdir()
    with gzip.open(d / "scores.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_event_id", "user_id", "item_id", "cand_idx", "label", "question", "lp_yes", "lp_no", "logit",
                    "yes_no_mass", "censored"])
        for r in conf:
            for c, y in enumerate(r["candidate_labels"]):
                sig = 3.0 if r["candidate_timestamps"][c] < T else 0.0
                w.writerow([r["source_event_id"], r["user_id"], r["candidate_item_ids"][c], c, y, "like", -1, -1,
                            f"{sig * (2 * y - 1) + rng.normal():.6f}", 0.999, 0])
    (d / "report.json").write_text(json.dumps({"variant": "V0", "lora": "a", "model": "m", "hist_len": 10,
                                               "data_sha1": "x", "mean_yes_no_mass": 0.999, "n_overlength": 0}))
    st = ge.run_stats(d, ge.panel_timestamps(cp), T)
    # the signal exists only before T: post-T rows carry none, the whole panel clearly does
    assert abs(ge.mean(st["auc_post"]) - 0.5) < 0.08 and ge.mean(st["auc_all"]) > ge.mean(st["auc_post"]) + 0.07


def test_integrity_failures_and_wrong_seed_count_never_pass(tmp_path):
    conf, cp, sr = setup(tmp_path)
    ok = [write_run(tmp_path / f"s{k}", conf, f"ad{k}", 2.0, k) for k in range(3)]
    assert ge.evaluate(cp, sr, [str(d) for d in ok[:2]], n_boot=20)["decision"] == "GATE_FT_INCOMPLETE"
    bad = write_run(tmp_path / "bad", conf, "adx", 2.0, 7, cens2=0.05)               # 5% censored=2 rows
    res = ge.evaluate(cp, sr, [str(ok[0]), str(ok[1]), str(bad)], n_boot=20)
    assert res["decision"] == "GATE_FT_INCOMPLETE" and any("integrity" in p for p in res["problems"])
    low_mass = write_run(tmp_path / "lm", conf, "ady", 2.0, 8, mass=0.5)
    assert ge.evaluate(cp, sr, [str(ok[0]), str(ok[1]), str(low_mass)], n_boot=20)["decision"] == "GATE_FT_INCOMPLETE"
    same = write_run(tmp_path / "same", conf, "ad0", 2.0, 9)                         # the same adapter twice
    assert ge.evaluate(cp, sr, [str(ok[0]), str(ok[1]), str(same)], n_boot=20)["decision"] == "GATE_FT_INCOMPLETE"


def test_zero_shot_context_is_paired_and_never_gates(tmp_path):
    conf, cp, sr = setup(tmp_path)
    ft = [write_run(tmp_path / f"s{k}", conf, f"ad{k}", 2.0, k) for k in range(3)]
    zs = write_run(tmp_path / "zs", conf, None, 0.3, 99)
    res = ge.evaluate(cp, sr, [str(d) for d in ft], str(zs), n_boot=50)
    z = res["zero_shot_context"]
    assert z["dUAUC_finetuned_minus_zeroshot_post_T"]["est"] > 0.1 and "never gates" in z["note"]
    assert res["decision"] == "GATE_FT_PASS"
