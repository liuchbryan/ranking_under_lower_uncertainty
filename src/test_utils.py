import numpy as np
from typing import Dict

def get_test_params() -> Dict[str, float]:
    """
    Randomly generate a set of parameters from a realistic param space
    """
    params = {
        'N': int(10 ** np.random.uniform(1.0, 3.5)),
        'mu_X': np.random.uniform(-10, 10),
        'mu_epsilon': np.random.uniform(-10, 10),
        'sigma_sq_X': np.random.uniform(0.3, 10) ** 2,
        'sigma_sq_1': np.random.uniform(0.3, 10) ** 2
    }
    params['M'] = max(int(params['N'] * np.random.uniform(0.01, 0.8)), 1)
    params['sigma_sq_2'] = (
        max((np.sqrt(params['sigma_sq_1']) * np.random.uniform(0.1, 0.99)) ** 2,
            0.2 ** 2))
    params['r'] = (
        int(np.random.uniform(params['N'] - params['M'] + 1, params['N'] + 1)))
    params['s'] = (
        int(np.random.uniform(params['N'] - params['M'] + 1, params['N'] + 1)))

    return params