from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

from itinerary_system.interest_model import BayesianInterestModel, interest_features


def test_prior_predictive_matches_gaussian_quadrature_and_is_reproducible():
    model = BayesianInterestModel(np.zeros(8), np.eye(8))
    x = np.zeros((2, 8))
    x[:, 0] = 1
    result = model.predict(x, draws=2048)
    # Symmetric normal logits imply E[sigmoid(Z)] = 1/2 exactly.
    np.testing.assert_allclose(result["mean"], [0.5, 0.5], atol=1e-14)
    np.testing.assert_array_equal(result["draws"], model.predict(x, draws=2048)["draws"])
    assert np.all(result["interval"][:, 0] < 0.2)
    assert np.all(result["interval"][:, 1] > 0.8)
    assert model.metadata()["evidence_mode"] == "prior_only"
    assert model.metadata()["calibration_status"] == "not_calibrated"


def test_laplace_posterior_tracks_independent_one_dimensional_integral():
    x = np.zeros((80, 8))
    x[:, 0] = 1
    labels = np.array([1] * 56 + [0] * 24)
    posterior = BayesianInterestModel.fit(x, labels)
    result = posterior.predict(x[:1], draws=2048)
    beta = np.linspace(-6, 6, 50_001)
    probability = 1 / (1 + np.exp(-beta))
    log_density = 56 * np.log(probability) + 24 * np.log1p(-probability) - 0.5 * (beta + 0.1) ** 2
    density = np.exp(log_density - log_density.max())
    expected = trapezoid(probability * density, beta) / trapezoid(density, beta)
    assert abs(result["mean"][0] - expected) < 0.012
    assert posterior.covariance[0, 0] < BayesianInterestModel.prior().covariance[0, 0]
    assert result["interval"][0, 1] - result["interval"][0, 0] < 0.2
    assert posterior.metadata()["feedback_count"] == 80
    assert posterior.metadata()["calibration_status"] == "not_calibrated"


def test_stated_category_preference_changes_probabilities_without_forced_stop_labels():
    model = BayesianInterestModel.prior()
    nature = interest_features(["outdoor"], nature_preference=1)
    culture = interest_features(["museum"], nature_preference=1)
    result = model.predict(np.array([nature, culture]))
    assert result["mean"][0] > result["mean"][1] + 0.3
    assert model.feedback_count == 0
    assert np.all((result["draws"] >= 0) & (result["draws"] <= 1))
    assert model.metadata() == BayesianInterestModel.prior().metadata()


@pytest.mark.parametrize("labels", [[0.5], [np.nan], [2], [-1]])
def test_feedback_rejects_nonbinary_or_nonfinite_labels(labels):
    with pytest.raises(ValueError, match="feedback_invalid"):
        BayesianInterestModel.fit(np.ones((1, 8)), np.array(labels))


def test_all_positive_feedback_is_regularized_and_finite():
    model = BayesianInterestModel.fit(np.ones((20, 8)), np.ones(20))
    result = model.predict(np.ones((1, 8)))
    assert 0.8 < result["mean"][0] < 1
    assert np.isfinite(model.covariance).all()


@pytest.mark.parametrize("value", [True, float("nan"), -1.1, 1.1])
def test_invalid_preference_fails(value):
    with pytest.raises(ValueError, match="preference_invalid"):
        interest_features(["outdoor"], nature_preference=value)
