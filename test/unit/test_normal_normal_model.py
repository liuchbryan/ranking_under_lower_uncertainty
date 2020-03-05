import pytest
from rulu.normal_normal_model import get_samples, _var_XIr_XJs, var_XIr
from scipy.stats import norm

class TestGetSamples:
    def test_have_required_list_headers(self):
        samples = get_samples(n_samples=1, N=100, M=10, mu_X=0, mu_epsilon=0,
                              sigma_sq_X=1**2, sigma_sq_1=0.5**2, sigma_sq_2=0.3**2)

        assert 'noisy_mean' in samples
        assert 'clean_mean' in samples
        assert 'improvement' in samples

    def test_return_correct_number_of_samples(self):
        samples = get_samples(n_samples=42, N=100, M=10, mu_X=0, mu_epsilon=0,
                              sigma_sq_X=1 ** 2, sigma_sq_1=0.5 ** 2, sigma_sq_2=0.3 ** 2)

        assert len(samples['noisy_mean']) == 42
        assert len(samples['clean_mean']) == 42
        assert len(samples['improvement']) == 42

    def test_return_correct_number_of_samples_large(self):

        samples = get_samples(n_samples=1726, N=100, M=10, mu_X=0, mu_epsilon=0,
                              sigma_sq_X=1 ** 2, sigma_sq_1=0.5 ** 2, sigma_sq_2=0.3 ** 2)

        assert len(samples['noisy_mean']) == 1726
        assert len(samples['clean_mean']) == 1726
        assert len(samples['improvement']) == 1726

    def test_return_more_list_header_if_r_specified(self):
        samples = get_samples(n_samples=1, N=100, M=10, mu_X=0, mu_epsilon=0,
                              sigma_sq_X=1 ** 2, sigma_sq_1=0.5 ** 2, sigma_sq_2=0.3 ** 2,
                              r=50)

        assert 'noisy_rth_observed' in samples
        assert 'noisy_rth_true' in samples
        assert 'clean_rth_observed' in samples
        assert 'clean_rth_true' in samples

    def test_return_more_list_header_if_s_specified(self):
        samples = get_samples(n_samples=1, N=100, M=10, mu_X=0, mu_epsilon=0,
                              sigma_sq_X=1 ** 2, sigma_sq_1=0.5 ** 2, sigma_sq_2=0.3 ** 2,
                              s=50)

        assert 'noisy_sth_observed' in samples
        assert 'noisy_sth_true' in samples
        assert 'clean_sth_observed' in samples
        assert 'clean_sth_true' in samples

    def test_r_has_to_be_between_1_and_N(self):
        with pytest.raises(AssertionError):
            get_samples(n_samples=1, N=100, M=10, mu_X=0, mu_epsilon=0,
                        sigma_sq_X=1 ** 2, sigma_sq_1=0.5 ** 2, sigma_sq_2=0.3 ** 2,
                        r=0)

        with pytest.raises(AssertionError):
            get_samples(n_samples=1, N=100, M=10, mu_X=0, mu_epsilon=0,
                        sigma_sq_X=1 ** 2, sigma_sq_1=0.5 ** 2, sigma_sq_2=0.3 ** 2,
                        r=101)

    def test_s_has_to_be_between_1_and_N(self):
        with pytest.raises(AssertionError):
            get_samples(n_samples=1, N=100, M=10, mu_X=0, mu_epsilon=0,
                        sigma_sq_X=1 ** 2, sigma_sq_1=0.5 ** 2, sigma_sq_2=0.3 ** 2,
                        s=0)

        with pytest.raises(AssertionError):
            get_samples(n_samples=1, N=100, M=10, mu_X=0, mu_epsilon=0,
                        sigma_sq_X=1 ** 2, sigma_sq_1=0.5 ** 2, sigma_sq_2=0.3 ** 2,
                        s=101)


class TestVarXIrXJs:
    def test_returns_expected_value(self):
        sigma_sq_X = 5
        sigma_sq_1 = 4
        sigma_sq_2 = 3
        N = 10
        s = 2
        r = 1

        expected_val_first_component = 5 * 4 * 3 / (5 * 4 + 5 * 3 + 4 * 3)
        expected_val_second_component = ((5 * 4 / (5 * 4 + 5 * 3 + 4 * 3)) ** 2 *
                                         2 * 9 / 11 ** 2 / 12 *
                                         (5 + 3) / norm.pdf(norm.ppf(2 / 11)) ** 2)
        expected_val_third_component = (5 ** 2 / (5 + 4) *
                                        1 * 10 / 11 ** 2 / 12 /
                                        norm.pdf(norm.ppf(1 / 11)) ** 2)

        assert (
            _var_XIr_XJs(r=r, s=s, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, N=N) ==
            expected_val_first_component + expected_val_second_component + expected_val_third_component)


class TestVarXIr:
    def test_returns_expected_value(self):
        sigma_sq_X = 5
        sigma_sq_eps = 4
        N = 10
        r = 1

        expected_value = (4 * 5 / (4 + 5) + 5 ** 2 / (4 + 5) * 1 * 10 / 11 ** 2 / 12 /
                          norm.pdf(norm.ppf(1 / 11)) ** 2)
        assert var_XIr(r=r, sigma_sq_X=sigma_sq_X, sigma_sq_eps=sigma_sq_eps, N=N) == expected_value
