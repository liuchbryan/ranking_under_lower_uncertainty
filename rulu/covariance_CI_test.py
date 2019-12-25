from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import norm, percentileofscore


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

    def sample_CI(self) -> Tuple[float, float]:
        if len(self.samples) == 0:
            return 0, 0
        else:
            return (np.percentile(self.samples, self.CI_PERCENTILE_LOW),
                    np.percentile(self.samples, self.CI_PERCENTILE_HIGH))

    def theoretical_quantity_in_sample_CI(self) -> bool:
        theoretical_quantity = self.theoretical_quantity()
        sample_CI_low, sample_CI_high = self.sample_CI()

        return (
            (theoretical_quantity >= sample_CI_low) and
            (theoretical_quantity < sample_CI_high)
        )

    def theoretical_quantity_above_sample_CI(self) -> bool:
        theoretical_quantity = self.theoretical_quantity()
        _, sample_CI_high = self.sample_CI()

        return theoretical_quantity >= sample_CI_high

    def theoretical_quantity_below_sample_CI(self) -> bool:
        theoretical_quantity = self.theoretical_quantity()
        sample_CI_low, _ = self.sample_CI()

        return theoretical_quantity < sample_CI_low

    def theoretical_quantity_sample_percentile(self) -> float:
        if len(self.samples) == 0:
            return None
        return percentileofscore(self.samples, self.theoretical_quantity())


class CovYrZsCITest(CovarianceCITest):
    def __init__(self, r=80, s=90, sigma_sq_X=1, sigma_sq_1=0.5**2, sigma_sq_2=0.3**2, N=100):
        super().__init__()
        self.r = r
        self.s = s
        self.sigma_sq_X = sigma_sq_X
        self.sigma_sq_1 = sigma_sq_1
        self.sigma_sq_2 = sigma_sq_2
        self.N = N

    def theoretical_quantity(self) -> float:
        # The theoretical quantity formula assumes r < s
        # Since covariance is symmetrical, swap r and s if r > s
        if self.r > self.s:
            r = self.s
            s = self.r
        else:
            r = self.r
            s = self.s

        return (
            self.sigma_sq_X ** 2 /
            (np.sqrt(self.sigma_sq_X + self.sigma_sq_1) *
             np.sqrt(self.sigma_sq_X + self.sigma_sq_2)) *
            r * (self.N - s + 1) /
            ((self.N + 1) ** 2 * (self.N + 2)) /
            (norm.pdf(norm.ppf(r / (self.N + 1))) *
             norm.pdf(norm.ppf(s / (self.N + 1))))
        )

    def add_sample(self, samples: Dict[str, List[float]]) -> None:
        self.samples.append(
            np.cov(samples['noisy_rth_observed'],
                   samples['clean_sth_observed'])[0][1])

    def get_test_name(self) -> str:
        return "Cov(Y_(r), Z_(s))"


class CovYrZsSecondOrderCITest(CovarianceCITest):
    def __init__(self, r=80, s=90, sigma_sq_X=1, sigma_sq_1=0.5**2, sigma_sq_2=0.3**2, N=100):
        super().__init__()
        self.r = r
        self.s = s
        self.sigma_sq_X = sigma_sq_X
        self.sigma_sq_1 = sigma_sq_1
        self.sigma_sq_2 = sigma_sq_2
        self.N = N

    def theoretical_quantity(self) -> float:
        # The theoretical quantity formula assumes r < s
        # Since covariance is symmetrical, swap r and s if r > s
        if self.r > self.s:
            r = self.s
            s = self.r
        else:
            r = self.r
            s = self.s

        # Clear up the rest of the notations
        p = lambda r: r / (self.N + 1)
        q = lambda r: (self.N - r + 1) / (self.N + 1)
        Q = lambda r: norm.ppf(p(r))
        Qp = lambda r: 1 / (norm.pdf(Q(r)))
        Qpp = lambda r: Q(r) / (norm.pdf(Q(r))) ** 2
        Qppp = lambda r: (1 + 2 * Q(r) ** 2) / (norm.pdf(Q(r))) ** 3

        return (
            self.sigma_sq_X ** 2 /
            (np.sqrt(self.sigma_sq_X + self.sigma_sq_1) *
             np.sqrt(self.sigma_sq_X + self.sigma_sq_2)) *
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
        self.samples.append(
            np.cov(samples['noisy_rth_observed'],
                   samples['clean_sth_observed'])[0][1])

    def get_test_name(self) -> str:
        return "Cov(Y_(r), Z_(s)) - 2nd order"


class CovXIrXJsCITest(CovarianceCITest):
    def __init__(self, r=80, s=90, sigma_sq_X=1, sigma_sq_1=0.5**2, sigma_sq_2=0.3**2, N=100):
        super().__init__()
        self.r = r
        self.s = s
        self.sigma_sq_X = sigma_sq_X
        self.sigma_sq_1 = sigma_sq_1
        self.sigma_sq_2 = sigma_sq_2
        self.N = N

    def theoretical_quantity(self) -> float:
        # The theoretical quantity formula assumes r < s
        # Since covariance is symmetrical, swap r and s if r > s
        if self.r > self.s:
            r = self.s
            s = self.r
        else:
            r = self.r
            s = self.s

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
            r * (self.N - s + 1) /
            ((self.N + 1) ** 2 * (self.N + 2)) /
            (norm.pdf(norm.ppf(r / (self.N + 1))) *
             norm.pdf(norm.ppf(s / (self.N + 1))))
        )

    def add_sample(self, samples: Dict[str, List[float]]) -> None:
        self.samples.append(
            np.cov(samples['noisy_rth_true'], samples['clean_sth_true'])[0][1]
        )

    def get_test_name(self) -> str:
        return "Cov(X_I(r), X_J(s))"


class CovXIrXJsSecondOrderCITest(CovarianceCITest):
    def __init__(self, r=80, s=90, sigma_sq_X=1, sigma_sq_1=0.5**2, sigma_sq_2=0.3**2, N=100):
        super().__init__()
        self.r = r
        self.s = s
        self.sigma_sq_X = sigma_sq_X
        self.sigma_sq_1 = sigma_sq_1
        self.sigma_sq_2 = sigma_sq_2
        self.N = N

    def theoretical_quantity(self) -> float:
        # The theoretical quantity formula assumes r < s
        # Since covariance is symmetrical, swap r and s if r > s
        if self.r > self.s:
            r = self.s
            s = self.r
        else:
            r = self.r
            s = self.s

        # Clear up the rest of the notations
        p = lambda r: r / (self.N + 1)
        q = lambda r: (self.N - r + 1) / (self.N + 1)
        Q = lambda r: norm.ppf(p(r))
        Qp = lambda r: 1 / (norm.pdf(Q(r)))
        Qpp = lambda r: Q(r) / (norm.pdf(Q(r))) ** 2
        Qppp = lambda r: (1 + 2 * Q(r) ** 2) / (norm.pdf(Q(r))) ** 3

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
        self.samples.append(
            np.cov(samples['noisy_rth_true'], samples['clean_sth_true'])[0][1]
        )

    def get_test_name(self) -> str:
        return "Cov(X_I(r), X_J(s)) - 2nd order"


class CovYrYsCITest(CovarianceCITest):
    def __init__(self, r=80, s=90, N=100, sigma_sq_X=1, **kwargs):
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

        # The theoretical quantity formula assumes r < s
        # Since covariance is symmetrical, swap r and s if r > s
        if self.r > self.s:
            r = self.s
            s = self.r
        else:
            r = self.r
            s = self.s

        return(
            r * (self.N - s + 1) /
            ((self.N + 1) ** 2 * (self.N + 2)) *
            (self.sigma_sq_X + sigma_sq_eps) /
            (norm.pdf(norm.ppf(r / (self.N + 1))) *
             norm.pdf(norm.ppf(s / (self.N + 1))))
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
    def __init__(self, r=80, s=90, N=100, sigma_sq_X=1, **kwargs):
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

        # The theoretical quantity formula assumes r < s
        # Since covariance is symmetrical, swap r and s if r > s
        if self.r > self.s:
            r = self.s
            s = self.r
        else:
            r = self.r
            s = self.s

        # Clear up the rest of the notations
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


# I think we can bootstrap this one instead.
class CovV1V2CITest(CovarianceCITest):
    def __init__(self, sigma_sq_X=1, sigma_sq_1=0.5**2, sigma_sq_2=0.3**2, N=100, M=25):
        super().__init__()
        self.sigma_sq_X = sigma_sq_X
        self.sigma_sq_1 = sigma_sq_1
        self.sigma_sq_2 = sigma_sq_2
        self.N = N
        self.M = M
        # Cache the theoretical quantity as it takes a long time
        self.theoretical_quantity_cache = None

    def theoretical_quantity(self) -> float:
        if self.theoretical_quantity_cache is not None:
            return self.theoretical_quantity_cache

        acc = 0.0
        for r in range(self.N - self.M + 1, self.N + 1):
            acc += r * (self.N - r + 1) / norm.pdf(norm.ppf(r / (self.N + 1))) ** 2
            for s in range(r + 1, self.N + 1):
                acc += (2.0 * r * (self.N - s + 1) /
                        norm.pdf(norm.ppf(r / (self.N + 1))) /
                        norm.pdf(norm.ppf(s / (self.N + 1))))

        acc *= ((self.N - 1) / self.N / self.M ** 2 *
                self.sigma_sq_X ** 2 /
                (self.sigma_sq_X + self.sigma_sq_1) ** 1.5 /
                (self.sigma_sq_X + self.sigma_sq_2) ** 1.5 /
                (self.N + 1) ** 2 / (self.N + 2))

        acc += (1.0 / self.N *
                (self.sigma_sq_X * self.sigma_sq_1 * self.sigma_sq_2) /
                (self.sigma_sq_X * self.sigma_sq_1 +
                self.sigma_sq_X * self.sigma_sq_2 +
                self.sigma_sq_1 * self.sigma_sq_2))

        self.theoretical_quantity_cache = acc
        return acc

    def add_sample(self, samples: Dict[str, List[float]]) -> None:
        self.samples.append(
            np.cov(samples['noisy_mean'], samples['clean_mean'])[0][1])

    def get_test_name(self) -> str:
        return "Cov(V1, V2)"
