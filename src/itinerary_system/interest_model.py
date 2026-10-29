"""Bayesian Bernoulli preference model, independent of optimizer utility scores.

The default is an elicited, uncalibrated prior. Fitting requires explicit
interest labels; catalog popularity, compulsory stops and silence are not labels.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = (
    "intercept",
    "category_affinity",
    "weather_fit",
    "visit_hours",
    "detour_hours",
    "source_utility",
    "weather_missing",
    "duration_missing",
)
FEATURE_SCHEMA = "interest-features-v1"
PRIOR_MEAN = np.array([-0.1, 1.6, 0.7, -0.35, -0.6, 0.4, 0.0, 0.0])
PRIOR_STD = np.array([1.0, 1.0, 0.8, 0.5, 0.5, 0.5, 0.4, 0.4])
NATURE_CATEGORIES = frozenset({"nature", "outdoor", "park", "beach", "scenic_stop", "hiking", "national_park"})
CULTURE_CATEGORIES = frozenset({"museum", "art", "culture", "cultural", "historical", "indoor", "gallery"})


def interest_features(
    categories: list[str] | tuple[str, ...],
    *,
    nature_preference: float = 0.0,
    weather_suitability: float | None = None,
    visit_minutes: float | None = None,
    detour_minutes: float | None = None,
    static_utility: float | None = None,
) -> np.ndarray:
    """Encode stated category preference and evidence-backed context, with missingness."""
    if isinstance(nature_preference, bool) or not np.isfinite(nature_preference) or not -1 <= nature_preference <= 1:
        raise ValueError("interest_preference_invalid")
    values = (weather_suitability, visit_minutes, detour_minutes, static_utility)
    if any(value is not None and (isinstance(value, bool) or not np.isfinite(value)) for value in values):
        raise ValueError("interest_feature_invalid")
    if weather_suitability is not None and not 0 <= weather_suitability <= 1:
        raise ValueError("interest_weather_invalid")
    if visit_minutes is not None and visit_minutes <= 0:
        raise ValueError("interest_duration_invalid")
    tokens = {category.casefold() for category in categories}
    category = int(bool(tokens & NATURE_CATEGORIES)) - int(bool(tokens & CULTURE_CATEGORIES))
    return np.array(
        [
            1.0,
            category * nature_preference,
            0.0 if weather_suitability is None else 2 * weather_suitability - 1,
            0.0 if visit_minutes is None else min(visit_minutes / 60, 4.0),
            0.0 if detour_minutes is None else np.clip(detour_minutes / 60, -2.0, 8.0),
            0.0 if static_utility is None else np.clip(static_utility, 0, 1) - 0.5,
            float(weather_suitability is None),
            float(visit_minutes is None),
        ],
        dtype=float,
    )


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -logits))


@dataclass(frozen=True)
class BayesianInterestModel:
    """An immutable parameter posterior; no global or cross-session mutable state."""

    mean: np.ndarray
    covariance: np.ndarray
    feedback_count: int = 0
    feedback_sha256: str | None = None

    @classmethod
    def prior(cls) -> BayesianInterestModel:
        return cls(PRIOR_MEAN.copy(), np.diag(PRIOR_STD**2))

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray) -> BayesianInterestModel:
        """Compute a Laplace posterior with a Gaussian prior and damped Newton steps."""
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=float)
        if (
            x.ndim != 2
            or x.shape[1] != len(FEATURE_NAMES)
            or not 1 <= len(x) <= 10_000
            or y.shape != (len(x),)
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or not np.isin(y, [0, 1]).all()
        ):
            raise ValueError("interest_feedback_invalid")
        precision = 1 / PRIOR_STD**2
        beta = PRIOR_MEAN.copy()

        def objective(value: np.ndarray) -> float:
            logits = x @ value
            return float(
                np.sum(np.logaddexp(0, logits) - y * logits) + 0.5 * np.sum(precision * (value - PRIOR_MEAN) ** 2)
            )

        for _ in range(100):
            probability = _sigmoid(x @ beta)
            gradient = x.T @ (probability - y) + precision * (beta - PRIOR_MEAN)
            hessian = (x.T * (probability * (1 - probability))) @ x + np.diag(precision)
            step = np.linalg.solve(hessian, gradient)
            if np.max(np.abs(gradient)) < 1e-7:
                break
            rate = 1.0
            current = objective(beta)
            while rate > 1e-10:
                candidate = beta - rate * step
                if objective(candidate) <= current - 1e-4 * rate * float(gradient @ step):
                    beta = candidate
                    break
                rate *= 0.5
            else:
                raise ValueError("interest_posterior_not_converged")
        else:
            raise ValueError("interest_posterior_not_converged")
        probability = _sigmoid(x @ beta)
        hessian = (x.T * (probability * (1 - probability))) @ x + np.diag(precision)
        digest = hashlib.sha256(
            json.dumps({"x": x.tolist(), "y": y.tolist()}, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(beta, np.linalg.solve(hessian, np.eye(len(FEATURE_NAMES))), len(x), digest)

    def predict(self, features: np.ndarray, *, draws: int = 256, seed: int = 20261020) -> dict:
        """Return probability means, 90% parameter-credible intervals and joint draws."""
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError("interest_prediction_features_invalid")
        if isinstance(draws, bool) or not isinstance(draws, int) or not 32 <= draws <= 2048 or draws % 2:
            raise ValueError("interest_draw_count_invalid")
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((draws // 2, len(FEATURE_NAMES)))
        noise = np.concatenate((noise, -noise)) @ np.linalg.cholesky(self.covariance).T
        samples = _sigmoid(x @ (self.mean + noise).T)
        return {
            "mean": samples.mean(axis=1),
            "interval": np.quantile(samples, [0.05, 0.95], axis=1).T,
            "draws": samples,
            "interval_level": 0.9,
        }

    def metadata(self) -> dict:
        payload = {
            "feature_schema_version": FEATURE_SCHEMA,
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "feedback_sha256": self.feedback_sha256,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {
            "model_id": "bayesian-logistic-laplace-v1",
            "model_sha256": digest,
            "feature_schema_version": FEATURE_SCHEMA,
            "feedback_count": self.feedback_count,
            "feedback_snapshot_sha256": self.feedback_sha256,
            "prediction_target": "explicit_interest_when_shown",
            "evidence_mode": "uncalibrated_posterior" if self.feedback_count else "prior_only",
            "calibration_status": "not_calibrated",
            "seed": 20261020,
            "interval_kind": "parameter_credible_interval",
        }
