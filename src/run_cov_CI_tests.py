import pickle
import time
from typing import List
import numpy as np

from src.covariance_CI_test import CovarianceCITest

def print_test_collection_result(test_collection: List[CovarianceCITest]) -> None:
    if test_collection is None or len(test_collection) == 0:
        print("There is nothing in the provided test collection.")

    within_CI = [test.theoretical_quantity_in_sample_CI()
                 for test in test_collection]

    print(test_collection[0].get_test_name() +
          ": {}/{} ({}%) "
          .format(np.sum(within_CI), len(within_CI),
                  np.round(100.0 * np.sum(within_CI) / len(within_CI), 2)) +
          "of the tests have the theoretical quantity within the CI.")


def save_test_collection(test_collection: List[CovarianceCITest]) -> None:
    if test_collection is None or len(test_collection) == 0:
        return
    file_name = ("../output/" + str(test_collection[0].__class__.__name__) +
                 "_" + str(int(time.time())) + ".pickle")
    pickle_file = open(file_name, 'wb')
    pickle.dump(test_collection, pickle_file)

    print("The test collection is saved at " + file_name)
