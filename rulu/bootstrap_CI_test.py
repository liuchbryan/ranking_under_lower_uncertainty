from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy.stats import percentileofscore


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
    
    
class VarVCITest(BootstrapCITest):
    def __init__(self, sigma_sq_X: float = 1, N: int = 100, M: int = 25, **kwargs):
        super(VarVCITest, self).__init__()
        self.sigma_sq_X = sigma_sq_X
        self.N = N
        self.M = M

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
        raise NotImplementedError


class CovV1V2CITest(BootstrapCITest):
    def __init__(self, sigma_sq_X: float = 1, sigma_sq_1: float = 0.5**2, sigma_sq_2: float = 0.3**2,
                 N: int = 100, M: int = 25):
        super(CovV1V2CITest, self).__init__()
        self.sigma_sq_X = sigma_sq_X
        self.sigma_sq_1 = sigma_sq_1
        self.sigma_sq_2 = sigma_sq_2
        self.N = N
        self.M = M

    def test_name(self) -> str:
        return "Cov(V1, V2)"

    def set_initial_samples(self, samples: Dict[str, List[float]]) -> None:
        raise NotImplementedError

    def generate_bootstrap_samples(self, num_bootstrap_samples: int) -> None:
        raise NotImplementedError

    def theoretical_quantity(self) -> float:
        raise NotImplementedError


class VarDCITest(BootstrapCITest):
    def __init__(self, sigma_sq_X: float = 1, sigma_sq_1: float = 0.5**2, sigma_sq_2: float = 0.3**2,
                 N: int = 100, M: int = 25):
        super(VarDCITest, self).__init__()
        self.sigma_sq_X = sigma_sq_X
        self.sigma_sq_1 = sigma_sq_1
        self.sigma_sq_2 = sigma_sq_2
        self.N = N
        self.M = M

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
        raise NotImplementedError