from rulu.bootstrap_CI_test import *

class TestSampleCI:
    def test_returns_zero_if_there_are_no_samples(self):
        test = VarVCITest(sigma_sq_1=0.5**2)
        assert test.sample_CI() == (0, 0)

    def test_returns_intended_percentiles_of_bootstrap_samples(self):
        test = VarVCITest(sigma_sq_1=0.5**2)
        test.bootstrap_samples = range(0, 101) # This gives you [0, 1, 2, .., 100]

        assert test.sample_CI() == (2.5, 97.5)
