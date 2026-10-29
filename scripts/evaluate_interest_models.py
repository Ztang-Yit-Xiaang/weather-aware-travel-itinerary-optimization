"""Paired controlled model comparison. All outcomes here are synthetic, never user feedback."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from itinerary_system.interest_model import BayesianInterestModel, interest_features  # noqa: E402


def features(rng, count):
    return np.array(
        [
            interest_features(
                ["nature"] if rng.random() < 0.5 else ["culture"],
                nature_preference=rng.uniform(-1, 1),
                weather_suitability=None if rng.random() < 0.15 else rng.random(),
                visit_minutes=None if rng.random() < 0.15 else rng.uniform(30, 180),
                detour_minutes=rng.uniform(-5, 60),
                static_utility=rng.random(),
            )
            for _ in range(count)
        ]
    )


def main():
    rows = []
    for scenario in ("linear", "nonlinear_prior_mismatch"):
        for seed in (11, 22, 33):
            rng = np.random.default_rng(seed)
            x = features(rng, 1200)
            logits = x @ np.array([-0.2, 1.8, 0.6, -0.4, -0.6, 0.25, -0.1, 0.1])
            if scenario == "nonlinear_prior_mismatch":
                logits += 2.0 * np.sin(4 * x[:, 1]) + 1.0 * (x[:, 2] * x[:, 3]) - 0.8
            truth = 1 / (1 + np.exp(-logits))
            y = rng.binomial(1, truth)
            test_x, test_y, test_truth = x[400:], y[400:], truth[400:]
            for count in (40, 320):
                train_x, train_y = x[:count], y[:count]
                for name in ("prior_predictive", "bayesian_logistic", "logistic_point_estimate", "gaussian_process"):
                    started = time.perf_counter()
                    interval = None
                    if name in {"prior_predictive", "bayesian_logistic"}:
                        model = (
                            BayesianInterestModel.prior()
                            if name == "prior_predictive"
                            else BayesianInterestModel.fit(train_x, train_y)
                        )
                        estimate = model.predict(test_x, draws=1024, seed=seed)
                        probability, interval = estimate["mean"], estimate["interval"]
                    elif name == "logistic_point_estimate":
                        model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(train_x[:, 1:], train_y)
                        probability = model.predict_proba(test_x[:, 1:])[:, 1]
                    else:
                        model = GaussianProcessClassifier(kernel=RBF(2.0), optimizer=None, random_state=seed).fit(
                            train_x[:, 1:], train_y
                        )
                        probability = model.predict_proba(test_x[:, 1:])[:, 1]
                    elapsed = time.perf_counter() - started
                    fraction, mean = calibration_curve(test_y, probability, n_bins=8, strategy="uniform")
                    bins = np.searchsorted(np.linspace(0, 1, 9)[1:-1], probability)
                    counts = np.bincount(bins, minlength=8)
                    occupied_counts = counts[counts > 0]
                    rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "train_count": count,
                            "test_count": len(test_y),
                            "method": name,
                            "brier": brier_score_loss(test_y, probability),
                            "log_loss": log_loss(test_y, probability),
                            "auc": roc_auc_score(test_y, probability),
                            "truth_probability_rmse": float(np.sqrt(np.mean((probability - test_truth) ** 2))),
                            "parameter_interval_truth_coverage": None
                            if interval is None
                            else float(np.mean((test_truth >= interval[:, 0]) & (test_truth <= interval[:, 1]))),
                            "mean_parameter_interval_width": None
                            if interval is None
                            else float(np.mean(interval[:, 1] - interval[:, 0])),
                            "fit_and_predict_seconds": elapsed,
                            "expected_calibration_error_8_bins": float(
                                np.sum(occupied_counts * np.abs(mean - fraction)) / len(test_y)
                            ),
                            "reliability": [
                                {"sample_count": int(n), "mean_probability": float(p), "positive_fraction": float(f)}
                                for n, p, f in zip(occupied_counts, mean, fraction, strict=True)
                            ],
                        }
                    )
    summary = []
    for scenario in ("linear", "nonlinear_prior_mismatch"):
        for count in (40, 320):
            for method in sorted({row["method"] for row in rows}):
                selected = [
                    r for r in rows if r["scenario"] == scenario and r["train_count"] == count and r["method"] == method
                ]
                summary.append(
                    {
                        "scenario": scenario,
                        "train_count": count,
                        "method": method,
                        **{
                            key: float(np.mean([r[key] for r in selected]))
                            for key in ("brier", "log_loss", "auc", "fit_and_predict_seconds")
                        },
                    }
                )
    result = {
        "evidence_mode": "synthetic_controlled_comparison",
        "real_user_feedback_count": 0,
        "protocol": "Same train/test examples per method; fixed settings; 800 held-out draws, 3 seeds, 2 mechanisms; no test-set tuning.",
        "limits": [
            "No claim of real-user probability calibration or model superiority.",
            "GP settings are fixed, not exhaustively optimized.",
            "XGBoost is declared in project dependencies but unavailable in this interpreter; not executed.",
        ],
        "summary": summary,
        "rows": rows,
    }
    output = ROOT / "results" / "interest_regions_20261020" / "model_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
