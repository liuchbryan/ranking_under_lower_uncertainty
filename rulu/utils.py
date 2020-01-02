import numpy as np
from typing import Dict
import os
import pickle


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
    params['M'] = int(max(params['N'] * np.random.uniform(0.01, 0.8), 1))
    params['sigma_sq_2'] = (
        max((np.sqrt(params['sigma_sq_1']) * np.random.uniform(0.1, 0.99)) ** 2,
            0.2 ** 2))
    params['r'] = (
        int(np.random.uniform(params['N'] - params['M'] + 1, params['N'] + 1)))
    params['s'] = (
        int(np.random.uniform(params['N'] - params['M'] + 1, params['N'] + 1)))

    return params


def find_all_tests_in_same_category(test, in_dir='../output'):
    """
    Retrieve all tests in `in_dir` that is of the same type as the specified `test`
    """
    def get_tests_from_pickle_file(file_path):
        filehandler = open(file_path, 'rb')
        return pickle.load(filehandler)

    tests_pickle_fps = [
        os.path.join(in_dir, file)
        for file in os.listdir(in_dir)
        if str(test.__class__.__name__) in file]

    return [test for tests in map(get_tests_from_pickle_file, tests_pickle_fps)
            for test in tests]
