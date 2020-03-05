from rulu.utils import fit_beta_distribution_params
import pytest


class TestFitBetaDistributionParams:
    def test_returns_expected_value(self):
        alpha = 1.0
        beta = 2.0
        mu = alpha / (alpha + beta)
        sigma_sq = alpha * beta / (alpha + beta) ** 2 / (alpha + beta + 1)

        assert fit_beta_distribution_params(mu, sigma_sq) == (pytest.approx(alpha, 1e-8),
                                                              pytest.approx(beta, 1e-8))
