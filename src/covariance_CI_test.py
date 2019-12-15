from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import norm


class CovarianceCITest(ABC):
    CI_PERCENTILE_LOW = 2.5
    CI_PERCENTILE_HIGH = 97.5

    def __init__(self):
        self.samples = []
        pass

    @abstractmethod
    def theoretical_quantity(self) -> float:
        pass

    @abstractmethod
    def add_sample(self, samples: Dict[str, List[float]]) -> None:
        pass

    @abstractmethod
    def get_test_name(self) -> str:
        pass

    def get_sample_CI(self) -> Tuple[float, float]:
        if len(self.samples) == 0:
            return 0, 0
        else:
            return (np.percentile(self.samples, self.CI_PERCENTILE_LOW),
                    np.percentile(self.samples, self.CI_PERCENTILE_HIGH))

    def theoretical_quantity_in_sample_CI(self) -> bool:
        theoretical_quantity = self.theoretical_quantity()
        sample_CI_low, sample_CI_high = self.get_sample_CI()

        return (
            (theoretical_quantity > sample_CI_low) and
            (theoretical_quantity < sample_CI_high)
        )

    def theoretical_quantity_above_sample_CI(self) -> bool:
        theoretical_quantity = self.theoretical_quantity()
        _, sample_CI_high = self.get_sample_CI()

        return theoretical_quantity > sample_CI_high

    def theoretical_quantity_below_sample_CI(self) -> bool:
        theoretical_quantity = self.theoretical_quantity()
        sample_CI_low, _ = self.get_sample_CI()

        return theoretical_quantity < sample_CI_low


class CovYrZsCITest(CovarianceCITest):
    def __init__(self, r, s, sigma_sq_X, sigma_sq_1, sigma_sq_2, N):
        super().__init__()
        self.r = r
        self.s = s
        self.sigma_sq_X = sigma_sq_X
        self.sigma_sq_1 = sigma_sq_1
        self.sigma_sq_2 = sigma_sq_2
        self.N = N

    def theoretical_quantity(self) -> float:
        return (
            self.sigma_sq_X ** 2 /
            (np.sqrt(self.sigma_sq_X + self.sigma_sq_1) *
             np.sqrt(self.sigma_sq_X + self.sigma_sq_2)) *
            self.r * (self.N - self.s + 1) /
            ((self.N + 1) ** 2 * (self.N + 2)) /
            (norm.pdf(norm.ppf(self.r / (self.N + 1))) *
             norm.pdf(norm.ppf(self.s / (self.N + 1))))
        )

    def add_sample(self, samples: Dict[str, List[float]]) -> None:
        self.samples.append(
            np.cov(samples['noisy_rth_observed'],
                   samples['clean_sth_observed'])[0][1])

    def get_test_name(self) -> str:
        return "Cov(Y_(r), Z_(s))"


class CovXIrXJsCITest(CovarianceCITest):
    def __init__(self, r, s, sigma_sq_X, sigma_sq_1, sigma_sq_2, N):
        super().__init__()
        self.r = r
        self.s = s
        self.sigma_sq_X = sigma_sq_X
        self.sigma_sq_1 = sigma_sq_1
        self.sigma_sq_2 = sigma_sq_2
        self.N = N

    def theoretical_quantity(self) -> float:
        return (
            1 / self.N *
            self.sigma_sq_X * self.sigma_sq_1 * self.sigma_sq_2 /
            (self.sigma_sq_X * self.sigma_sq_1 +
             self.sigma_sq_X * self.sigma_sq_2 +
             self.sigma_sq_1 * self.sigma_sq_2) +

            (self.N - 1) / self.N *
            self.sigma_sq_X ** 4 /
            ((self.sigma_sq_X + self.sigma_sq_1) ** 1.5 *
             (self.sigma_sq_X + self.sigma_sq_2) ** 1.5) *
            self.r * (self.N - self.s + 1) /
            ((self.N + 1) ** 2 * (self.N + 2)) /
            (norm.pdf(norm.ppf(self.r / (self.N + 1))) *
             norm.pdf(norm.ppf(self.s / (self.N + 1))))
        )

    def add_sample(self, samples: Dict[str, List[float]]) -> None:
        self.samples.append(
            np.cov(samples['noisy_rth_true'], samples['clean_sth_true'])[0][1]
        )

    def get_test_name(self) -> str:
        return "Cov(X_I(r), X_J(s))"


class CovYrYsCITest(CovarianceCITest):
    def __init__(self, r, s, N, sigma_sq_X, **kwargs):
        super().__init__()
        self.r = r
        self.s = s
        self.sigma_sq_X = sigma_sq_X
        self.sigma_sq_1 = None
        self.sigma_sq_2 = None
        self.N = N

        if not (('sigma_sq_1' in kwargs) ^ ('sigma_sq_2' in kwargs)):
            raise TypeError("Precisely one of sigma_sq_1 / sigma_sq_2 is required.")
        elif 'sigma_sq_1' in kwargs:
            self.sigma_sq_1 = kwargs['sigma_sq_1']
        elif 'sigma_sq_2' in kwargs:
            self.sigma_sq_2 = kwargs['sigma_sq_2']
        assert (self.sigma_sq_1 is None) ^ (self.sigma_sq_2 is None)

    def theoretical_quantity(self) -> float:
        if self.sigma_sq_1 is not None:
            sigma_sq_eps = self.sigma_sq_1
        else:
            sigma_sq_eps = self.sigma_sq_2

        return(
            self.r * (self.N - self.s + 1) /
            ((self.N + 1) ** 2 * (self.N + 2)) *
            (self.sigma_sq_X + sigma_sq_eps) /
            (norm.pdf(norm.ppf(self.r / (self.N + 1))) *
             norm.pdf(norm.ppf(self.s / (self.N + 1))))
        )

    def add_sample(self, samples: Dict[str, List[float]]) -> None:
        if self.sigma_sq_1 is not None:
            self.samples.append(
                np.cov(samples['noisy_rth_observed'], samples['noisy_sth_observed'])[0][1]
            )
        else:
            self.samples.append(
                np.cov(samples['clean_rth_observed'], samples['clean_sth_observed'])[0][1]
            )

    def get_test_name(self) -> str:
        return "Cov(Y_(r), Y_(s))"


class CovYrYsSecondOrderCITest(CovarianceCITest):
    def __init__(self, r, s, N, sigma_sq_X, **kwargs):
        super().__init__()
        self.r = r
        self.s = s
        self.sigma_sq_X = sigma_sq_X
        self.sigma_sq_1 = None
        self.sigma_sq_2 = None
        self.N = N

        if not (('sigma_sq_1' in kwargs) ^ ('sigma_sq_2' in kwargs)):
            raise TypeError("Precisely one of sigma_sq_1 / sigma_sq_2 is required.")
        elif 'sigma_sq_1' in kwargs:
            self.sigma_sq_1 = kwargs['sigma_sq_1']
        elif 'sigma_sq_2' in kwargs:
            self.sigma_sq_2 = kwargs['sigma_sq_2']
        assert (self.sigma_sq_1 is None) ^ (self.sigma_sq_2 is None)

    def theoretical_quantity(self) -> float:
        if self.sigma_sq_1 is not None:
            sigma_sq_eps = self.sigma_sq_1
        else:
            sigma_sq_eps = self.sigma_sq_2

        # Clear up the notations
        r = self.r
        s = self.s
        p = lambda r: r / (self.N + 1)
        q = lambda r: (self.N - r + 1) / (self.N + 1)
        Q = lambda r: norm.ppf(p(r))
        Qp = lambda r: 1 / (norm.pdf(Q(r)))
        Qpp = lambda r: Q(r) / (norm.pdf(Q(r))) ** 2
        Qppp = lambda r: (1 + 2 * Q(r) ** 2) / (norm.pdf(Q(r))) ** 3

        return(
            (self.sigma_sq_X + sigma_sq_eps) *
            (p(r) * q(s) / (self.N + 2) * Qp(r) * Qp(s) +
             p(r) * q(s) / (self.N + 2) ** 2 * (
                 (q(r) - p(r)) * Qpp(r) * Qp(s) +
                 (q(s) - p(s)) * Qp(r) * Qpp(s) +
                 1.0 / 2 * p(r) * q(r) * Qppp(r) * Qp(s) +
                 1.0 / 2 * p(s) * q(s) * Qp(r) * Qppp(s) +
                 1.0 / 2 * p(r) * q(s) * Qpp(r) * Qpp(s)
             ))
        )

    def add_sample(self, samples: Dict[str, List[float]]) -> None:
        if self.sigma_sq_1 is not None:
            self.samples.append(
                np.cov(samples['noisy_rth_observed'], samples['noisy_sth_observed'])[0][1]
            )
        else:
            self.samples.append(
                np.cov(samples['clean_rth_observed'], samples['clean_sth_observed'])[0][1]
            )

    def get_test_name(self) -> str:
        return "Cov(Y_(r), Y_(s)) - 2nd order"


# I think we can bootstrap this one instead. If not:
# TODO: include caching for the theoretical quantity
class CovV1V2CITest(CovarianceCITest):
    def __init__(self, sigma_sq_X, sigma_sq_1, sigma_sq_2, N, M):
        super().__init__()
        self.sigma_sq_X = sigma_sq_X
        self.sigma_sq_1 = sigma_sq_1
        self.sigma_sq_2 = sigma_sq_2
        self.N = N
        self.M = M

    def _cov_XIr_XJs_theoretical_quantity(self, r, s) -> float:
        return(
            1 / self.N *
            self.sigma_sq_X * self.sigma_sq_1 * self.sigma_sq_2 /
            (self.sigma_sq_X * self.sigma_sq_1 +
             self.sigma_sq_X * self.sigma_sq_2 +
             self.sigma_sq_1 * self.sigma_sq_2) +

            (self.N - 1) / self.N *
            self.sigma_sq_X ** 4 /
            ((self.sigma_sq_X + self.sigma_sq_1) ** 1.5 *
             (self.sigma_sq_X + self.sigma_sq_2) ** 1.5) *
            r * (self.N - s + 1) /
            ((self.N + 1) ** 2 * (self.N + 2)) /
            (norm.pdf(norm.ppf(r / (self.N + 1))) *
             norm.pdf(norm.ppf(s / (self.N + 1))))
        )

    def theoretical_quantity(self) -> float:
        acc = 0
        for r in range(self.N - self.M + 1, self.N + 1):
            for s in range(self.N - self.M + 1, self.N + 1):
                acc += (self._cov_XIr_XJs_theoretical_quantity(r, s))

        return acc / self.M ** 2

    def add_sample(self, samples: Dict[str, List[float]]) -> None:
        self.samples.append(
            np.cov(samples['noisy_mean'], samples['clean_mean'])[0][1])

    def get_test_name(self) -> str:
        return "Cov(V1, V2)"
