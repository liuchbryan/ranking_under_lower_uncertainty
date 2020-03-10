from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import pickle
import time

import numpy as np
from scipy.stats import percentileofscore, norm

from rulu.normal_normal_model import E_V, E_D, var_XIr, cov_XIr_XIs, cov_V1_V2, var_V


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


class EVCITest(BootstrapCITest):
    def __init__(self, mu_X: float = 0, sigma_sq_X: float = 1**2, N: int = 100, M: int =25, **kwargs):
        super(EVCITest, self).__init__()
        self.mu_X = mu_X
        self.sigma_sq_X = sigma_sq_X
        self.N = N
        self.M = M
        self.sigma_sq_1 = None
        self.sigma_sq_2 = None

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
                np.mean(np.random.choice(self.initial_samples, len(self.initial_samples), replace=True))
            )

    def test_name(self) -> str:
        return "E(V)"

    def theoretical_quantity(self) -> float:
        if self.sigma_sq_1 is not None:
            sigma_sq_eps = self.sigma_sq_1
        else:
            sigma_sq_eps = self.sigma_sq_2

        return E_V(mu_X=self.mu_X, sigma_sq_X=self.sigma_sq_X, sigma_sq_eps=sigma_sq_eps, N=self.N, M=self.M)



class EDCITest(BootstrapCITest):
    def __init__(self, mu_X: float = 0, sigma_sq_X: float = 1**2, sigma_sq_1: float = 0.5**2,
                 sigma_sq_2: float = 0.4 ** 2, N: int = 100, M: int =25, **kwargs):
        super(EDCITest, self).__init__()
        self.mu_X = mu_X
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
                np.mean(np.random.choice(self.initial_samples, len(self.initial_samples), replace=True))
            )

    def test_name(self) -> str:
        return "E(D)"

    def theoretical_quantity(self) -> float:
        return E_D(mu_X=self.mu_X, sigma_sq_X=self.sigma_sq_X, sigma_sq_1=self.sigma_sq_1,
                   sigma_sq_2=self.sigma_sq_2, N=self.N, M=self.M)



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

        theoretical_quantity = var_V(sigma_sq_X=self.sigma_sq_X, sigma_sq_eps=sigma_sq_eps, N=self.N, M=self.M)

        self.theoretical_quantity_cache = theoretical_quantity
        return theoretical_quantity


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

        theoretical_quantity = cov_V1_V2(sigma_sq_X=self.sigma_sq_X, sigma_sq_1=self.sigma_sq_1,
                                         sigma_sq_2=self.sigma_sq_2, N=self.N, M=self.M)
        self.theoretical_quantity_cache = theoretical_quantity
        return theoretical_quantity


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


def _print_nothing_in_initial_sample(test_name: str) -> None:
    print("There is nothing in the initial samples for {}.".format(test_name))


def print_test_collection_result(test_collection: List[BootstrapCITest]) -> None:
    if test_collection is None or len(test_collection) == 0:
        print("There is nothing in the provided test collection.")
        return

    # It usually takes a long time to calculate the theoretical quantity,
    # which will benefit from a progress bar
    within_CI = []
    for index, test in enumerate(test_collection, start=1):
        print("Calculating theoretical quantity / sample CI for test {}/{}... (N={}, M={})"
              .format(index, len(test_collection), test.N, test.M),
              end="\r")

        within_CI.append(test.theoretical_quantity_in_sample_CI())

    print(test_collection[0].test_name() +
          ": {}/{} ({}%) "
          .format(np.sum(within_CI), len(within_CI),
                  np.round(100.0 * np.sum(within_CI) / len(within_CI), 2)) +
          "of the tests have the theoretical quantity within the CI.")


def save_test_collection(test_collection: List[BootstrapCITest], in_dir: str = '../output/') -> None:
    """
    Save given `test_collection` as a pickle file in `in_dir`
    :param test_collection: List of tests
    :param in_dir: Output directory
    :return: None
    """
    if test_collection is None or len(test_collection) == 0:
        return

    file_name = (in_dir + str(test_collection[0].__class__.__name__) +
                 "_" + str(int(time.time())) + ".pickle")
    pickle_file = open(file_name, 'wb')
    pickle.dump(test_collection, pickle_file)

    print("The test collection is saved at " + file_name)
