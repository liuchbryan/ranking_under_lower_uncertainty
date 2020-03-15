import argparse
from rulu.covariance_CI_test import *
from rulu.normal_normal_model import get_samples
from rulu.utils import get_test_params


# Argument parser work
script_description = \
    "Run multiple statistical tests to confirm the theoretical covariance between selected pairs of random variables."
parser = argparse.ArgumentParser(description=script_description)
parser.add_argument("--num_tests", default=10, type=int,
                    help="Number of statistical tests to be run")
parser.add_argument("--num_covariate_samples", default=100, type=int,
                    help="Number of covariate samples to be collected for each statistical test")
parser.add_argument("--num_runs", default=100, type=int,
                    help="Number of simulation runs to produce one sample covariance")
args = parser.parse_args()

# Prepare test collections
cov_Yr_Ys_tests = []
cov_Yr_Ys_second_order_tests = []
cov_Yr_Zs_tests = []
cov_Yr_Zs_second_order_tests = []
cov_XIr_XJs_tests = []
cov_XIr_XJs_second_order_tests = []
cov_V1_V2_tests = []

cov_test_collections = [
    cov_Yr_Ys_tests, cov_Yr_Ys_second_order_tests,
    cov_Yr_Zs_tests, cov_Yr_Zs_second_order_tests,
    cov_XIr_XJs_tests, cov_XIr_XJs_second_order_tests,
    # cov_V1_V2_tests
]

for num_test in range(0, args.num_tests):
    try:
        # Sample the parameters from a realistic parameter space
        params = get_test_params()
        N = int(params['N'])
        M = int(params['M'])
        mu_X = params['mu_X']
        mu_epsilon = params['mu_epsilon']
        sigma_sq_X = params['sigma_sq_X']
        sigma_sq_1 = params['sigma_sq_1']
        sigma_sq_2 = params['sigma_sq_2']
        r = params['r']
        s = params['s']

        # Prepare a test
        cov_Yr_Ys_test = CovYrYsCITest(r, s, N, sigma_sq_X, sigma_sq_1=sigma_sq_1)
        cov_Yr_Ys_second_order_test = CovYrYsSecondOrderCITest(r, s, N, sigma_sq_X, sigma_sq_1=sigma_sq_1)
        cov_Yr_Zs_test = CovYrZsCITest(r, s, sigma_sq_X, sigma_sq_1, sigma_sq_2, N)
        cov_Yr_Zs_second_order_test = CovYrZsSecondOrderCITest(r, s, sigma_sq_X, sigma_sq_1, sigma_sq_2, N)
        cov_XIr_XJs_test = CovXIrXJsCITest(r, s, sigma_sq_X, sigma_sq_1, sigma_sq_2, N)
        cov_XIr_XJs_second_order_test = CovXIrXJsSecondOrderCITest(r, s, sigma_sq_X, sigma_sq_1, sigma_sq_2, N)
        # cov_V1_V2_test = CovV1V2CITest(sigma_sq_X, sigma_sq_1, sigma_sq_2, N, M)

        # Arrange the covariance tests IN THE SAME ORDER as that in the collections
        cov_tests = [
            cov_Yr_Ys_test, cov_Yr_Ys_second_order_test,
            cov_Yr_Zs_test, cov_Yr_Zs_second_order_test,
            cov_XIr_XJs_test, cov_XIr_XJs_second_order_test,
            # cov_V1_V2_test
        ]

        for num_covariate_sample in range(0, args.num_covariate_samples):
            print("Test {}/{} (N={}, M={}, r={}, s={}): calculating sample {}/{}...    "
                  .format(num_test + 1, args.num_tests, N, M, r, s,
                          num_covariate_sample + 1, args.num_covariate_samples),
                  end='\r')

            # Generate samples from multiple runs to obtain one
            # covariate sample for each pair of r.v. we are intersted in
            samples = get_samples(
                args.num_runs, N, M, mu_X, mu_epsilon,
                sigma_sq_X, sigma_sq_1, sigma_sq_2,
                verbose=False, r=r, s=s)

            for test in cov_tests:
                test.add_sample(samples)

        # Once a test is completed (i.e. we collected enough covariate
        # samples), we added them to the corresponding, existing
        # collection of tests
        for (cov_test, cov_test_collection) in \
                [(cov_tests[i], cov_test_collections[i]) for i in range(0, len(cov_tests))]:
            cov_test_collection.append(cov_test)

        print("Test {}/{} (N={}, M={}, r={}, s={}): completed.                         "
              .format(num_test + 1, args.num_tests, N, M, r, s),
              end='\n')

    except KeyboardInterrupt:
        # Breaking the for loop is sufficient, as we want to
        # analyse the results accumulated so far
        break

# Print some statistics and save the test collections for future reference
# once we reached the number of experiments / process is interrupted
for cov_test_collection in cov_test_collections:
    try:
        print_test_collection_result(cov_test_collection)
    except KeyboardInterrupt:
        print("{}: Skipped.".format(cov_test_collection[0].get_test_name()))
    save_test_collection(cov_test_collection, in_dir="./output/")
