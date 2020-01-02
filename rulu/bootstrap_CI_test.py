from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy.stats import percentileofscore, norm


def _print_nothing_in_initial_sample(test_name: str) -> None:
    print("There is nothing in the initial samples for {}.".format(test_name))


class BootstrapCITest(ABC):
    CI_PERCENTILE_LOW = 2.5
    CI_PERCENTILE_HIGH = 97.5

    def __init__(self):
        self.initial_samples = []
        self.bootstrap_samples = []

    @abstractmethod
    def theoretical_quantity(self) -> float:
        pass

    @abstractmethod
    def set_initial_samples(self, samples: Dict[str, List[float]]) -> None:
        pass

    @abstractmethod
    def generate_bootstrap_samples(self, num_bootstrap_samples: int) -> None:
        pass

    @abstractmethod
    def test_name(self) -> str:
        pass

    def sample_CI(self) -> Tuple[float, float]:
        if len(self.bootstrap_samples) == 0:
            return 0, 0
        else:
            return (np.percentile(self.bootstrap_samples, self.CI_PERCENTILE_LOW),
                    np.percentile(self.bootstrap_samples, self.CI_PERCENTILE_HIGH))
        
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

    def theoretical_quantity_sample_percentile(self) -> Optional[float]:
        if len(self.bootstrap_samples) == 0:
            return None
        return percentileofscore(self.bootstrap_samples, self.theoretical_quantity())


def _var_XIr(r, sigma_sq_X, sigma_sq_eps, N, **kwargs):
    return (
            (sigma_sq_eps * sigma_sq_X /
             (sigma_sq_X + sigma_sq_eps)) +
            sigma_sq_X ** 2 /
            (sigma_sq_X + sigma_sq_eps) *
            (r * (N - r + 1)) /
            ((N + 1) ** 2 * (N + 2)) /
            (norm.pdf(norm.ppf(r / (N + 1)))) ** 2
    )


def _cov_XIr_XIs(r, s, sigma_sq_X, sigma_sq_eps, N, **kwargs):
    if r == s:
        return _var_XIr(r, sigma_sq_X, sigma_sq_eps, N, **kwargs)

    # The formula assumes r < s, though if the input
    # r is larger than s, then we just need to swap them
    # as covariance function is symmetric.
    r_act = (r if r < s else s)
    s_act = (s if r < s else r)

    return (
            sigma_sq_X ** 2 /
            (sigma_sq_X + sigma_sq_eps) *
            (r_act * (N - s_act + 1)) /
            ((N + 1) ** 2 * (N + 2)) /
            norm.pdf(norm.ppf(r_act / (N + 1))) /
            norm.pdf(norm.ppf(s_act / (N + 1)))
    )


class VarVCITest(BootstrapCITest):
    def __init__(self, sigma_sq_X: float = 1, N: int = 100, M: int = 25, **kwargs):
        super(VarVCITest, self).__init__()
        self.sigma_sq_X = sigma_sq_X
        self.N = N
        self.M = M
        self.sigma_sq_1 = None
        self.sigma_sq_2 = None
        # Cache the theoretical quantity as it takes a long time
        self.theoretical_quantity_cache = None

        if not (('sigma_sq_1' in kwargs) ^ ('sigma_sq_2' in kwargs)):
            raise TypeError("Precisely one of sigma_sq_1 / sigma_sq_2 is required.")
        elif 'sigma_sq_1' in kwargs:
            self.sigma_sq_1 = kwargs['sigma_sq_1']
        elif 'sigma_sq_2' in kwargs:
            self.sigma_sq_2 = kwargs['sigma_sq_2']
        assert (self.sigma_sq_1 is None) ^ (self.sigma_sq_2 is None)

    def set_initial_samples(self, samples: Dict[str, List[float]]) -> None:
        if self.sigma_sq_1 is not None:
            self.initial_samples = samples['noisy_mean']
        else:
            self.initial_samples = samples['clean_mean']

    def generate_bootstrap_samples(self, num_bootstrap_samples: int) -> None:
        if self.initial_samples is None or len(self.initial_samples) == 0:
            _print_nothing_in_initial_sample(self.test_name())
            return

        for i in range(0, num_bootstrap_samples):
            self.bootstrap_samples.append(
                np.var(np.random.choice(self.initial_samples, len(self.initial_samples), replace=True))
            )

    def test_name(self) -> str:
        return "Var(V)"

    def theoretical_quantity(self) -> float:
        if self.theoretical_quantity_cache is not None:
            return self.theoretical_quantity_cache

        if self.sigma_sq_1 is not None:
            sigma_sq_eps = self.sigma_sq_1
        else:
            sigma_sq_eps = self.sigma_sq_2

        acc = 0.0
        for r in range(self.N - self.M + 1, self.N + 1):
            acc += _var_XIr(r, self.sigma_sq_X, sigma_sq_eps, self.N)
            for s in range(r + 1, self.N + 1):
                acc += 2.0 * _cov_XIr_XIs(r, s, self.sigma_sq_X, sigma_sq_eps, self.N)

        acc = acc / (self.M ** 2)

        self.theoretical_quantity_cache = acc
        return acc


class CovV1V2CITest(BootstrapCITest):
    def __init__(self, sigma_sq_X: float = 1, sigma_sq_1: float = 0.5**2, sigma_sq_2: float = 0.3**2,
                 N: int = 100, M: int = 25):
        super(CovV1V2CITest, self).__init__()
        self.sigma_sq_X = sigma_sq_X
        self.sigma_sq_1 = sigma_sq_1
        self.sigma_sq_2 = sigma_sq_2
        self.N = N
        self.M = M
        # Cache the theoretical quantity as it takes a long time
        self.theoretical_quantity_cache = None

    def test_name(self) -> str:
        return "Cov(V1, V2)"

    def set_initial_samples(self, samples: Dict[str, List[float]]) -> None:
        self.initial_samples = [[samples['noisy_mean'][i], samples['clean_mean'][i]]
                                for i in range(0, len(samples['noisy_mean']))]

    def generate_bootstrap_samples(self, num_bootstrap_samples: int) -> None:
        if self.initial_samples is None or len(self.initial_samples) == 0:
            _print_nothing_in_initial_sample(self.test_name())
            return
        for i in range(0, num_bootstrap_samples):
            indices = (
                np.random.choice(range(0, len(self.initial_samples)), len(self.initial_samples), replace=True)
                .astype(int))
            self.bootstrap_samples.append(
                np.cov(np.array(self.initial_samples)[indices][:, 0],
                       np.array(self.initial_samples)[indices][:, 1])[0][1]
            )

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


class VarDCITest(BootstrapCITest):
    def __init__(self, sigma_sq_X: float = 1, sigma_sq_1: float = 0.5**2, sigma_sq_2: float = 0.3**2,
                 N: int = 100, M: int = 25):
        super(VarDCITest, self).__init__()
        self.sigma_sq_X = sigma_sq_X
        self.sigma_sq_1 = sigma_sq_1
        self.sigma_sq_2 = sigma_sq_2
        self.N = N
        self.M = M
        # Cache the theoretical quantity as it takes a long time
        self.theoretical_quantity_cache = None

    def set_initial_samples(self, samples: Dict[str, List[float]]) -> None:
        self.initial_samples = samples['improvement']

    def generate_bootstrap_samples(self, num_bootstrap_samples: int) -> None:
        if self.initial_samples is None or len(self.initial_samples) == 0:
            _print_nothing_in_initial_sample(self.test_name())
            return

        for i in range(0, num_bootstrap_samples):
            self.bootstrap_samples.append(
                np.var(np.random.choice(self.initial_samples, len(self.initial_samples), replace=True))
            )

    def test_name(self) -> str:
        return "Var(D)"

    def theoretical_quantity(self) -> float:
        return 0