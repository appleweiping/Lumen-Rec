import csv
import gzip
import json
import subprocess
import sys

import numpy as np


def _write(tmp, rows, panel):
    sp = tmp / "scores.csv.gz"
    with gzip.open(sp, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_event_id", "user_id", "item_id", "cand_idx", "label", "question",
                    "lp_yes", "lp_no", "logit", "yes_no_mass"])
        for r in rows:
            w.writerow(r)
    pp = tmp / "panel.jsonl"
    pp.write_text("\n".join(json.dumps(x) for x in panel))
    return sp, pp


def test_mirror_cancels_popularity_acquiescence(tmp_path):
    rng = np.random.default_rng(0)
    rows, panel = [], []
    n_items = 200
    pop = rng.integers(1, 5000, n_items)
    a_item = 0.8 * (np.log1p(pop) - np.log1p(pop).mean())  # planted popularity-linked yes-saying
    for u in range(300):
        items = rng.choice(n_items, 20, replace=False)
        y = (rng.uniform(size=20) < 0.5).astype(int)
        pref = 1.5 * (2 * y - 1) + rng.normal(0, 1, 20)
        panel.append({"candidate_item_ids": [f"i{i}" for i in items],
                      "candidate_popularity": [int(pop[i]) for i in items]})
        for c, (i, lab) in enumerate(zip(items, y)):
            ev = f"u{u}::0"
            like = pref[c] + a_item[i] + rng.normal(0, .3)
            dis = -pref[c] + a_item[i] + rng.normal(0, .3)
            para = pref[c] + a_item[i] + rng.normal(0, .3)
            for q, lg in (("like", like), ("dislike", dis), ("like_para", para)):
                rows.append([ev, f"u{u}", f"i{i}", c, int(lab), q, 0, 0, f"{lg:.6f}", 1])
    sp, pp = _write(tmp_path, rows, panel)
    out = tmp_path / "res.json"
    subprocess.run([sys.executable, "-m", "src.confrec.pilot_mirror", "--scores", str(sp),
                    "--panel", str(pp), "--out", str(out)], check=True, capture_output=True)
    res = json.loads(out.read_text())
    assert res["panel_type"] == "rated"
    assert res["acquiescence"]["spearman_a_logpop"] > 0.6           # planted effect recovered
    assert abs(res["valence_spearman_v_logpop"]) < 0.25            # valence carries no popularity
    d = res["dUAUC_mirror_minus_placebo"]
    assert d["mean"] > 0 and d["ci95"][0] > 0                       # cancelling a(i) helps ranking


def test_next_item_panel_ndcg(tmp_path):
    rows, panel = [], []
    for u in range(50):
        ids = [f"i{k}" for k in range(11)]
        panel.append({"candidate_item_ids": ids, "candidate_popularity_groups": ["head"] * 11})
        for c in range(11):
            lab = int(c == 3)
            rows.append([f"u{u}::0", f"u{u}", ids[c], c, lab, "next", 0, 0, f"{5.0 if lab else 0.0}", 1])
    sp, pp = _write(tmp_path, rows, panel)
    out = tmp_path / "res.json"
    subprocess.run([sys.executable, "-m", "src.confrec.pilot_mirror", "--scores", str(sp),
                    "--panel", str(pp), "--out", str(out)], check=True, capture_output=True)
    res = json.loads(out.read_text())
    assert res["panel_type"] == "next_item"
    assert abs(res["arms"]["raw"]["NDCG@10"] - 1.0) < 1e-9
    assert abs(res["arms"]["raw"]["top10_head_share"] - 1.0) < 1e-9
