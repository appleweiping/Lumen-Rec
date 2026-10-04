"""Training data and split report for Gate-FT (idea-stage/PREREG_AMENDMENT_2.md G9, PREREG_AMENDMENT_1.md section D).

    python -m src.confrec.gateft_data --dev_panel outputs/confrec/gatefix/panels/ml1m_dev_h20.jsonl \
        --confirm_panel outputs/confrec/gatefix/panels/ml1m_confirm_h20.jsonl --out_dir outputs/confrec/gateft [--quantile 0.8]

Registered semantics, fixed here (nothing is tunable):
  * T = the `quantile` (numpy linear, default 0.8) of ALL candidate_timestamps of the ML-1M rated panel, i.e. of the DEV
    users (the burned Pilot-1 users) together with the CONFIRM users. Computed once, recorded in gateft_split.json.
  * TRAIN = rows of the DEV users only, each row keeping its candidates with ts < T (all per-candidate lists are
    filtered together, the history is untouched; a row left without a candidate is dropped). No CONFIRM user can be in
    TRAIN (user_id disjointness is asserted), and no event at or after T is.
  * EVALUATION is on the CONFIRM users' candidates with ts >= T (the primary Gate-FT endpoint); the adapters score the
    whole CONFIRM panel and gateft_eval.py selects the post-T rows from the panel's candidate_timestamps, so the same
    scores also give the all-CONFIRM secondary endpoint and the paired zero-shot contrast (identical row keys).
The report records T, counts, the positive rate of TRAIN and the evaluation power (CONFIRM users with both classes
among their post-T candidates), plus the sha1 of every input and output file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.confrec.split_panel import filter_candidates
from src.confrec.stats import strict_json


def sha1_file(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_rows(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_threshold(dev_rows: list[dict], confirm_rows: list[dict], q: float) -> float:
    for r in dev_rows + confirm_rows:
        if "candidate_timestamps" not in r:
            raise ValueError("candidate_timestamps missing: rebuild the panels with build_rated_panels")
    ts = np.concatenate([np.asarray(r["candidate_timestamps"], float) for r in dev_rows + confirm_rows])
    return float(np.quantile(ts, q))


def build_train(dev_rows: list[dict], T: float) -> list[dict]:
    out = []
    for r in dev_rows:
        keep = [float(t) < T for t in r["candidate_timestamps"]]
        if any(keep):
            out.append(filter_candidates(r, keep))
    return out


def power(rows: list[dict], T: float) -> dict:
    """CONFIRM evaluation power at T: users with at least one post-T candidate / with both classes among them."""
    n_any = n_both = n_cand = n_pos = 0
    for r in rows:
        lab = [int(y) for t, y in zip(r["candidate_timestamps"], r["candidate_labels"]) if float(t) >= T]
        n_cand += len(lab)
        n_pos += sum(lab)
        n_any += bool(lab)
        n_both += bool(lab) and 0 < sum(lab) < len(lab)
    return {"users_with_post_T_candidates": n_any, "users_with_both_classes_post_T": n_both,
            "post_T_candidates": n_cand, "post_T_positive_rate": n_pos / n_cand if n_cand else float("nan")}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_panel", required=True)
    ap.add_argument("--confirm_panel", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--quantile", type=float, default=0.8)
    a = ap.parse_args(argv)
    if not 0.0 < a.quantile < 1.0:
        raise SystemExit("--quantile must be in (0, 1)")
    dev, conf = read_rows(a.dev_panel), read_rows(a.confirm_panel)
    du, cu = {str(r["user_id"]) for r in dev}, {str(r["user_id"]) for r in conf}
    if du & cu:
        raise SystemExit(f"DEV and CONFIRM share {len(du & cu)} users (e.g. {sorted(du & cu)[:3]}): G2 requires "
                         "disjoint user sets")
    T = split_threshold(dev, conf, a.quantile)
    train = build_train(dev, T)
    tu = {str(r["user_id"]) for r in train}
    assert tu <= du and not (tu & cu), "TRAIN must contain DEV users only"
    assert all(float(t) < T for r in train for t in r["candidate_timestamps"]), "TRAIN holds an event at or after T"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tpath = out / "train.jsonl"
    with open(tpath, "w", encoding="utf-8", newline="\n") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_c = sum(len(r["candidate_labels"]) for r in train)
    n_pos = sum(sum(int(y) for y in r["candidate_labels"]) for r in train)
    rep = {"quantile": a.quantile, "T": T, "n_candidates_for_T": sum(len(r["candidate_timestamps"]) for r in dev + conf),
           "dev_panel": str(a.dev_panel), "confirm_panel": str(a.confirm_panel),
           "inputs_sha1": {"dev_panel": sha1_file(a.dev_panel), "confirm_panel": sha1_file(a.confirm_panel)},
           "train": {"path": str(tpath), "sha1": sha1_file(tpath), "rows": len(train), "users": len(tu),
                     "candidates": n_c, "positive_rate": n_pos / n_c if n_c else float("nan"),
                     "dev_users": len(du), "dev_candidates_before_T_share": n_c / max(1, sum(
                         len(r["candidate_labels"]) for r in dev))},
           "eval": {"confirm_users": len(cu), **power(conf, T), "rule": "CONFIRM candidates with ts >= T; the "
                                                                          "adapters score the whole CONFIRM panel"},
           "disjoint_users": True}
    (out / "gateft_split.json").write_text(json.dumps(strict_json(rep), indent=2), encoding="utf-8")
    print(json.dumps(strict_json(rep)))
    if rep["eval"]["users_with_both_classes_post_T"] < 150:
        print("WARNING: fewer than 150 CONFIRM users have both classes among their post-T candidates; the Gate-FT "
              "UAUC will be noisy (SE about 0.15 / sqrt(n)). Recorded; the registered quantile is not changed.")
    return rep


if __name__ == "__main__":
    main()
