from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.baselines.internal_scores import (
    audit_score_degeneracy,
    audit_score_rows_against_candidates,
    finite_float,
    read_csv_rows,
    sha256_file,
    text,
    write_json,
    write_score_rows,
)
from src.eval.ranking_task_metrics import build_ranking_eval_frame, compute_ranking_task_metrics
from src.shadow.ccrp import apply_ccrp_scores, parse_weights
from src.utils.io import load_jsonl, save_jsonl


FULL_REPORTING_KS = (5, 10, 20)


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_weight_grid(value: str) -> list[tuple[float, float, float]]:
    text_value = str(value or "").strip()
    if not text_value:
        return [(0.5, 0.3, 0.2)]
    weights: list[tuple[float, float, float]] = []
    for chunk in text_value.split(";"):
        parts = [float(item.strip()) for item in chunk.split(",") if item.strip()]
        if len(parts) != 3:
            raise ValueError("--weight_grid entries must be triples: boundary,calibration_gap,evidence")
        weights.append((parts[0], parts[1], parts[2]))
    return weights


def _weight_label(values: tuple[float, float, float]) -> str:
    return ",".join(f"{value:g}" for value in values)


def _load_signal(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    return pd.read_json(source, lines=True)


def _score_lookup(scored_df: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    lookup: dict[tuple[str, str, str], float] = {}
    for row in scored_df.to_dict(orient="records"):
        item_id = text(row.get("candidate_item_id")) or text(row.get("item_id"))
        key = (text(row.get("source_event_id")), text(row.get("user_id")), item_id)
        if all(key):
            lookup[key] = finite_float(row.get("ccrp_risk_adjusted_score"))
    return lookup


def _candidate_score_rows(
    *,
    candidate_rows: list[dict[str, Any]],
    scored_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    lookup = _score_lookup(scored_df)
    rows: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        key = (text(candidate.get("source_event_id")), text(candidate.get("user_id")), text(candidate.get("item_id")))
        rows.append(
            {
                "source_event_id": key[0],
                "user_id": key[1],
                "item_id": key[2],
                "score": lookup.get(key, float("nan")),
            }
        )
    return rows


def _predictions_from_scores(
    ranking_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    *,
    method_name: str,
    k: int,
    tie_break_seed: int = 20260607,
) -> list[dict[str, Any]]:
    score_lookup = {
        (text(row.get("source_event_id")), text(row.get("user_id")), text(row.get("item_id"))): finite_float(row.get("score"))
        for row in score_rows
    }
    predictions: list[dict[str, Any]] = []
    for record in ranking_rows:
        source_event_id = text(record.get("source_event_id"))
        user_id = text(record.get("user_id"))
        candidate_ids = [text(item) for item in record.get("candidate_item_ids", []) if text(item)]
        # Deterministic-but-order-independent tie-break: a per-(event,item) hash
        # keyed by tie_break_seed replaces the original candidate index, so two
        # candidates with identical C-CRP scores are ordered by a stable pseudo-random
        # key rather than their position in the candidate list (removes any
        # positional artifact; reproducible via the seed recorded in provenance).
        def _tie_key(item_id: str) -> int:
            return int(
                hashlib.sha256(f"{tie_break_seed}:{source_event_id}:{user_id}:{item_id}".encode("utf-8")).hexdigest(),
                16,
            )

        scored = [
            (
                item_id,
                score_lookup.get((source_event_id, user_id, item_id), float("-inf")),
                _tie_key(item_id),
            )
            for item_id in candidate_ids
        ]
        ranked = [item_id for item_id, _, _ in sorted(scored, key=lambda item: (-item[1], item[2]))]
        prediction = dict(record)
        prediction.update(
            {
                "pred_ranked_item_ids": ranked,
                "topk_item_ids": ranked[:k],
                "parse_success": True,
                "latency": 0.0,
                "confidence": -1.0,
                "contains_out_of_candidate_item": False,
                "raw_response": method_name,
                "candidate_scores": {item_id: score for item_id, score, _ in scored},
            }
        )
        predictions.append(prediction)
    return predictions


def _evaluate_candidate_scores(
    *,
    ranking_path: str | Path,
    candidate_items_path: str | Path,
    signal_path: str | Path,
    score_mode: str,
    ablation: str,
    eta: float,
    confidence_weight: float,
    weights: tuple[float, float, float],
    k: int,
    ks: tuple[int, ...] | list[int] | None = None,
    fail_on_degeneracy: bool = True,
    max_tie_pair_rate: float = 0.7,
    max_constant_event_rate: float = 0.02,
    tie_break_seed: int = 20260607,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    ranking_rows = load_jsonl(ranking_path)
    candidate_rows = read_csv_rows(candidate_items_path)
    signal_df = _load_signal(signal_path)
    scored_df = apply_ccrp_scores(
        signal_df,
        weights=parse_weights(weights),
        eta=eta,
        ablation=ablation,
        score_mode=score_mode,
        confidence_weight=confidence_weight,
    )
    score_rows = _candidate_score_rows(candidate_rows=candidate_rows, scored_df=scored_df)
    audit = audit_score_rows_against_candidates(candidate_rows=candidate_rows, score_rows=score_rows)
    if not audit["audit_ok"]:
        raise ValueError(f"C-CRP score coverage audit failed for {signal_path}: {audit}")
    degeneracy_audit = audit_score_degeneracy(
        score_rows,
        max_tie_pair_rate=max_tie_pair_rate,
        max_constant_event_rate=max_constant_event_rate,
    )
    if fail_on_degeneracy and not degeneracy_audit["degeneracy_audit_ok"]:
        raise ValueError(f"C-CRP score degeneracy audit failed for {signal_path}: {degeneracy_audit}")
    predictions = _predictions_from_scores(
        ranking_rows,
        score_rows,
        method_name=f"ccrp:{score_mode}:{ablation}:eta={eta}:cw={confidence_weight}:w={_weight_label(weights)}",
        k=k,
        tie_break_seed=tie_break_seed,
    )
    metrics = compute_ranking_task_metrics(build_ranking_eval_frame(pd.DataFrame(predictions)), k=k, ks=ks)
    return {**audit, **degeneracy_audit, **metrics}, score_rows, scored_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select C-CRP mode on validation, export exact test scores, and optionally import them.")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--valid_ranking_path", required=True)
    parser.add_argument("--test_ranking_path", required=True)
    parser.add_argument("--valid_candidate_items_path", required=True)
    parser.add_argument("--test_candidate_items_path", required=True)
    parser.add_argument("--valid_signal_path", required=True)
    parser.add_argument("--test_signal_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--etas", default="0.5,1.0,2.0")
    parser.add_argument("--confidence_weights", default="0.5,0.7,0.9")
    parser.add_argument("--weight_grid", default="0.5,0.3,0.2")
    # Kept as compatibility shims for older launch manifests.  The current
    # protocol freezes the main selection line to ``full/full`` and evaluates
    # the other modes and ablations in the diagnostic table below, so these
    # values are intentionally accepted but do not alter the selection space.
    parser.add_argument("--score_modes", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ablations", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--selection_metric", default="NDCG@10")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--main_config_mode",
        choices=["preregistered", "valid_selected"],
        default=None,
        help=(
            "preregistered (default): main reported C-CRP uses a fixed principled config "
            "(full/full, eta=1.0, weights 0.5,0.3,0.2, confidence_weight=0.7) for fair "
            "comparison vs untuned official baselines; the valid grid is reported as a "
            "sensitivity study only. valid_selected: main config is chosen by valid "
            "NDCG@10 within ablation=full/score_mode=full (discloses a validation tuning budget)."
        ),
    )
    parser.add_argument("--prereg_eta", type=float, default=1.0)
    parser.add_argument("--prereg_confidence_weight", type=float, default=0.7)
    parser.add_argument("--prereg_weights", default="0.5,0.3,0.2")
    parser.add_argument(
        "--max_tie_pair_rate",
        type=float,
        default=0.7,
        help=(
            "Score-degeneracy gate: reject configs whose tied-candidate-pair rate exceeds this. "
            "Tightened from the 0.98 library default; set above the observed ~0.54 LLM-signal tie "
            "rate (coarse verbalized probabilities, neutralized by seeded random tie-break) while "
            "still rejecting a near-constant scorer."
        ),
    )
    parser.add_argument(
        "--max_constant_event_rate",
        type=float,
        default=0.02,
        help=(
            "Score-degeneracy gate: allow at most this fraction of fully-constant events (all "
            "candidates equal score). LLM-verbalized relevance returns all-zero for a small fraction "
            "of genuinely uninformative events; these are ranked at chance by the seeded random "
            "tie-break. Default 0.02 (2%%) sits above the observed ~0.4%% while still catching a "
            "globally degenerate scorer."
        ),
    )
    parser.add_argument(
        "--tie_break_seed",
        type=int,
        default=20260607,
        help="Fixed seed for randomized tie-breaking in ranking (removes any original-candidate-order artifact).",
    )
    parser.add_argument("--baseline_name", default=None)
    parser.add_argument("--import_exp_name", default=None)
    parser.add_argument("--import_scores", action="store_true")
    parser.add_argument("--status_label", default="same_schema_internal_method")
    parser.add_argument("--artifact_class", default="completed_result")
    args = parser.parse_args()
    if args.main_config_mode is None:
        # Older launch manifests supplied --score_modes/--ablations and
        # expected their validation sweep to select the reported row. Preserve
        # that behavior only for those manifests. New invocations retain the
        # preregistered default described in the current protocol.
        args.main_config_mode = (
            "valid_selected" if args.score_modes is not None or args.ablations is not None else "preregistered"
        )
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    etas = _parse_float_list(args.etas)
    weight_grid = _parse_weight_grid(args.weight_grid)
    prereg_weights = _parse_weight_grid(args.prereg_weights)[0]
    ranking_k = max(int(args.k), max(FULL_REPORTING_KS))

    # ------------------------------------------------------------------
    # VALIDATION SWEEP (hyperparameters ONLY, within the real method:
    # score_mode=full, ablation=full). Ablations and degenerate score-modes
    # (confidence_only / evidence_only) are NOT selection candidates — they are
    # evaluated separately below as diagnostics. This prevents a leave-one-out
    # variant or a non-method sub-model from being crowned as the main C-CRP.
    # ------------------------------------------------------------------
    valid_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    # NOTE: confidence_weight only affects score_mode=confidence_plus_evidence; under
    # the main method (score_mode=full) it is a no-op, so it is NOT swept here (sweeping
    # it would create identical duplicate configs and inflate the apparent tuning budget).
    # It is fixed to the preregistered value for the full-mode selection sweep.
    fixed_cw = float(args.prereg_confidence_weight)
    for eta in etas:
        for weights in weight_grid:
            metrics, _, _ = _evaluate_candidate_scores(
                ranking_path=args.valid_ranking_path,
                candidate_items_path=args.valid_candidate_items_path,
                signal_path=args.valid_signal_path,
                score_mode="full",
                ablation="full",
                eta=eta,
                confidence_weight=fixed_cw,
                weights=weights,
                k=ranking_k,
                ks=FULL_REPORTING_KS,
                fail_on_degeneracy=False,
                max_tie_pair_rate=args.max_tie_pair_rate,
                max_constant_event_rate=args.max_constant_event_rate,
                tie_break_seed=args.tie_break_seed,
            )
            row = {
                "domain": args.domain,
                "split": "valid",
                "validation_status": "candidate",
                "score_mode": "full",
                "ablation": "full",
                "eta": eta,
                "confidence_weight": fixed_cw,
                "weight_boundary": weights[0],
                "weight_calibration_gap": weights[1],
                "weight_evidence": weights[2],
                "weight_grid_label": _weight_label(weights),
                **metrics,
            }
            valid_rows.append(row)
            selectable = bool(row.get("audit_ok")) and bool(row.get("degeneracy_audit_ok"))
            metric_value = finite_float(row.get(args.selection_metric), default=float("nan"))
            if selectable and (best is None or metric_value > float(best[args.selection_metric])):
                best = row

    pd.DataFrame(valid_rows).to_csv(output_dir / "valid_ccrp_sweep.csv", index=False)
    if best is None:
        raise RuntimeError(
            "No selectable C-CRP validation row (full/full) passed coverage and score-degeneracy gates. "
            f"Saved validation sweep to {output_dir / 'valid_ccrp_sweep.csv'}."
        )
    write_json(best, output_dir / "selected_valid_config.json")

    # ------------------------------------------------------------------
    # MAIN CONFIG selection. Default = preregistered principled config (fair vs
    # untuned official baselines); the valid sweep above is reported as a
    # sensitivity study only. valid_selected mode discloses a tuning budget.
    # ------------------------------------------------------------------
    if args.main_config_mode == "preregistered":
        main_cfg = {
            "score_mode": "full",
            "ablation": "full",
            "eta": float(args.prereg_eta),
            "confidence_weight": float(args.prereg_confidence_weight),
            "weights": prereg_weights,
        }
    else:
        main_cfg = {
            "score_mode": "full",
            "ablation": "full",
            "eta": float(best["eta"]),
            "confidence_weight": float(best["confidence_weight"]),
            "weights": (
                float(best["weight_boundary"]),
                float(best["weight_calibration_gap"]),
                float(best["weight_evidence"]),
            ),
        }
    preregistration = {
        "main_config_mode": args.main_config_mode,
        "selection_metric": args.selection_metric,
        "selection_space": "score_mode=full, ablation=full; sweep eta x confidence_weight x weight_grid",
        "etas_grid": etas,
        "confidence_weights_grid": [float(args.prereg_confidence_weight)],
        "weight_grid": [_weight_label(w) for w in weight_grid],
        "preregistered_config": {
            "score_mode": "full",
            "ablation": "full",
            "eta": float(args.prereg_eta),
            "confidence_weight": float(args.prereg_confidence_weight),
            "weights": _weight_label(prereg_weights),
        },
        "valid_selected_config": {
            "eta": float(best["eta"]),
            "confidence_weight": float(best["confidence_weight"]),
            "weights": _weight_label(
                (float(best["weight_boundary"]), float(best["weight_calibration_gap"]), float(best["weight_evidence"]))
            ),
            f"valid_{args.selection_metric}": best[args.selection_metric],
        },
        "max_tie_pair_rate": args.max_tie_pair_rate,
        "max_constant_event_rate": args.max_constant_event_rate,
        "tie_break_seed": args.tie_break_seed,
        "note": "Ablations and degenerate score-modes are diagnostics evaluated at the frozen main config, never selection candidates.",
    }
    write_json(preregistration, output_dir / "ccrp_selection_preregistration.json")

    test_metrics, test_score_rows, test_scored_df = _evaluate_candidate_scores(
        ranking_path=args.test_ranking_path,
        candidate_items_path=args.test_candidate_items_path,
        signal_path=args.test_signal_path,
        score_mode=main_cfg["score_mode"],
        ablation=main_cfg["ablation"],
        eta=main_cfg["eta"],
        confidence_weight=main_cfg["confidence_weight"],
        weights=main_cfg["weights"],
        k=ranking_k,
        ks=FULL_REPORTING_KS,
        max_tie_pair_rate=args.max_tie_pair_rate,
        max_constant_event_rate=args.max_constant_event_rate,
        tie_break_seed=args.tie_break_seed,
    )
    score_summary = write_score_rows(test_score_rows, output_dir / "ccrp_selected_test_scores.csv")
    test_scored_df.to_csv(output_dir / "ccrp_selected_test_scored_rows.csv", index=False)

    test_row = {
        "domain": args.domain,
        "split": "test",
        "main_config_mode": args.main_config_mode,
        "selected_on": "valid" if args.main_config_mode == "valid_selected" else "preregistered",
        "selection_metric": args.selection_metric,
        f"selected_valid_{args.selection_metric}": best[args.selection_metric],
        "score_mode": main_cfg["score_mode"],
        "ablation": main_cfg["ablation"],
        "eta": main_cfg["eta"],
        "confidence_weight": main_cfg["confidence_weight"],
        "weight_boundary": main_cfg["weights"][0],
        "weight_calibration_gap": main_cfg["weights"][1],
        "weight_evidence": main_cfg["weights"][2],
        "weight_grid_label": _weight_label(main_cfg["weights"]),
        "status_label": args.status_label,
        "artifact_class": args.artifact_class,
        "valid_ranking_path": args.valid_ranking_path,
        "test_ranking_path": args.test_ranking_path,
        "valid_candidate_items_path": args.valid_candidate_items_path,
        "test_candidate_items_path": args.test_candidate_items_path,
        "valid_signal_path": args.valid_signal_path,
        "test_signal_path": args.test_signal_path,
        "valid_ranking_sha256": sha256_file(args.valid_ranking_path),
        "test_ranking_sha256": sha256_file(args.test_ranking_path),
        "valid_signal_sha256": sha256_file(args.valid_signal_path),
        "test_signal_sha256": sha256_file(args.test_signal_path),
        "valid_candidate_items_sha256": sha256_file(args.valid_candidate_items_path),
        "test_candidate_items_sha256": sha256_file(args.test_candidate_items_path),
        "etas_grid": etas,
        "confidence_weights_grid": [float(args.prereg_confidence_weight)],
        "weight_grid": [_weight_label(weights) for weights in weight_grid],
        "max_tie_pair_rate": args.max_tie_pair_rate,
        "max_constant_event_rate": args.max_constant_event_rate,
        "tie_break_seed": args.tie_break_seed,
        **score_summary,
        **test_metrics,
    }
    pd.DataFrame([test_row]).to_csv(output_dir / "selected_test_metrics.csv", index=False)
    write_json(test_row, output_dir / "ccrp_internal_provenance.json")

    # ------------------------------------------------------------------
    # COMPONENT-ABLATION DIAGNOSTICS (separate table, frozen at the main config's
    # hyperparameters). Leave-one-component-out variants and the degenerate
    # score-modes are evaluated here for transparency — never as selection
    # candidates. Honest reporting: a neutral/positive removal is recorded as-is.
    # ------------------------------------------------------------------
    diag_rows: list[dict[str, Any]] = []
    diag_specs = [
        ("full", "full"),
        ("full", "without_boundary_uncertainty"),
        ("full", "without_calibration_gap"),
        ("full", "without_evidence_support"),
        ("full", "without_counterevidence"),
        ("full", "without_risk_penalty"),
        ("confidence_only", "full"),
        ("evidence_only", "full"),
        ("confidence_plus_evidence", "full"),
    ]
    for diag_mode, diag_ablation in diag_specs:
        try:
            diag_metrics, _, _ = _evaluate_candidate_scores(
                ranking_path=args.test_ranking_path,
                candidate_items_path=args.test_candidate_items_path,
                signal_path=args.test_signal_path,
                score_mode=diag_mode,
                ablation=diag_ablation,
                eta=main_cfg["eta"],
                confidence_weight=main_cfg["confidence_weight"],
                weights=main_cfg["weights"],
                k=ranking_k,
                ks=FULL_REPORTING_KS,
                fail_on_degeneracy=False,
                max_tie_pair_rate=args.max_tie_pair_rate,
                max_constant_event_rate=args.max_constant_event_rate,
                tie_break_seed=args.tie_break_seed,
            )
        except Exception as exc:  # noqa: BLE001 — record diagnostic failure, do not abort main result
            diag_rows.append({
                "domain": args.domain, "split": "test", "row_kind": "diagnostic",
                "score_mode": diag_mode, "ablation": diag_ablation, "error": str(exc),
            })
            continue
        diag_rows.append({
            "domain": args.domain,
            "split": "test",
            "row_kind": "main" if (diag_mode == "full" and diag_ablation == "full") else "diagnostic",
            "score_mode": diag_mode,
            "ablation": diag_ablation,
            "eta": main_cfg["eta"],
            "confidence_weight": main_cfg["confidence_weight"],
            "weight_grid_label": _weight_label(main_cfg["weights"]),
            **diag_metrics,
        })
    pd.DataFrame(diag_rows).to_csv(output_dir / "ccrp_ablation_diagnostics.csv", index=False)


    if args.import_scores:
        baseline_name = args.baseline_name or f"{args.domain}_ccrp_{main_cfg['score_mode']}"
        exp_name = args.import_exp_name or f"{args.domain}_ccrp_selected_same_candidate"
        cmd = [
            sys.executable,
            "scripts/misc/main_import_same_candidate_baseline_scores.py",
            "--baseline_name",
            baseline_name,
            "--exp_name",
            exp_name,
            "--domain",
            args.domain,
            "--ranking_input_path",
            args.test_ranking_path,
            "--scores_path",
            str(output_dir / "ccrp_selected_test_scores.csv"),
            "--artifact_class",
            args.artifact_class,
            "--status_label",
            args.status_label,
            "--tie_break_seed",
            str(args.tie_break_seed),
        ]
        subprocess.run(cmd, check=True)

    print(f"Saved C-CRP valid sweep: {output_dir / 'valid_ccrp_sweep.csv'}")
    print(f"Saved selection preregistration: {output_dir / 'ccrp_selection_preregistration.json'}")
    print(f"Saved selected C-CRP scores: {output_dir / 'ccrp_selected_test_scores.csv'}")
    print(f"Saved selected test metrics: {output_dir / 'selected_test_metrics.csv'}")
    print(f"Saved ablation diagnostics: {output_dir / 'ccrp_ablation_diagnostics.csv'}")


if __name__ == "__main__":
    main()
