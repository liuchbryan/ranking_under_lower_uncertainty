import pytest
from rulu.normal_normal_model import get_samples

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
