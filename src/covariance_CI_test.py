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
            self.sigma_sq_X ** 2 /
            ((self.sigma_sq_X + self.sigma_sq_1) *
             (self.sigma_sq_X + self.sigma_sq_2)) *
            CovYrZsCITest(self.r, self.s,
                          self.sigma_sq_X, self.sigma_sq_1, self.sigma_sq_2,
                          self.N)
            .theoretical_quantity()
        )

    def add_sample(self, samples: Dict[str, List[float]]) -> None:
        self.samples.append(
            np.cov(samples['noisy_rth_true'], samples['clean_sth_true'])[0][1]
        )


class CovV1V2CITest(CovarianceCITest):
    def __init__(self, sigma_sq_X, sigma_sq_1, sigma_sq_2, N, M):
        super().__init__()
        self.sigma_sq_X = sigma_sq_X
        self.sigma_sq_1 = sigma_sq_1
        self.sigma_sq_2 = sigma_sq_2
        self.N = N
        self.M = M

    def theoretical_quantity(self) -> float:
        acc = 0
        for r in range(self.N - self.M + 1, self.N + 1):
            for s in range(self.N - self.M + 1, self.N + 1):
                acc += (
                    CovXIrXJsCITest(
                        r, s, self.sigma_sq_X, self.sigma_sq_1,
                        self.sigma_sq_2, self.N)
                    .theoretical_quantity())

        return acc / self.M ** 2

    def add_sample(self, samples: Dict[str, List[float]]) -> None:
        self.samples.append(
            np.cov(samples['noisy_mean'], samples['clean_mean'])[0][1])
