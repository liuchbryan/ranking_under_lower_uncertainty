import argparse
from rulu.bootstrap_CI_test import *
from rulu.normal_normal_model import get_samples
from rulu.utils import get_test_params

# Argument parser work
script_description = \
    "Run multiple statistical tests to confirm the theoretical variance of V and D, and theoretical " \
    "covariance between V1 and V2 using bootstrap samples."
parser = argparse.ArgumentParser(description=script_description)
parser.add_argument("--num_tests", default=10, type=int,
                    help="Number of statistical tests to be run")
parser.add_argument("--num_bootstrap_samples", default=1000, type=int,
                    help="Number of bootstrap (re-)samples to be collected for each statistical test")
parser.add_argument("--num_runs", default=1000, type=int,
                    help="Number of simulation runs to produce the initial samples")
args = parser.parse_args()
BOOTSTRAP_BATCH_SIZE = 100

# Prepare test collections
E_V_tests = []
E_D_tests = []
var_V_tests = []
cov_V1_V2_tests = []
var_D_tests = []

bootstrap_test_collections = [
    E_V_tests,
    E_D_tests,
    var_V_tests,
    cov_V1_V2_tests,
    var_D_tests
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
        E_V_test = EVCITest(mu_X=mu_X, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, N=N, M=M)
        E_D_test = EDCITest(mu_X=mu_X, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, N=N, M=M)
        var_V_test = VarVCITest(sigma_sq_X, N, M, sigma_sq_1=sigma_sq_1)
        cov_V1_V2_test = CovV1V2CITest(sigma_sq_X, sigma_sq_1, sigma_sq_2, N, M)
        var_D_test = VarDCITest(sigma_sq_X, sigma_sq_1, sigma_sq_2, N, M)

        # Arrange the covariance tests IN THE SAME ORDER as that in the collections
        bootstrap_tests = [
            E_V_test,
            E_D_test,
            var_V_test,
            cov_V1_V2_test,
            var_D_test
        ]

        # Get the initial samples
        print("Test {}/{} (N={}, M={}): calculating initial samples...    "
              .format(num_test + 1, args.num_tests, N, M),
              end='\r')

        samples = get_samples(args.num_runs, N, M, mu_X, mu_epsilon,
                              sigma_sq_X, sigma_sq_1, sigma_sq_2,
                              verbose=False, r=r, s=s)

        for test in bootstrap_tests:
            test.set_initial_samples(samples)

        #  Bootstrap from the initial samples in batches
        for i in range(0, int(args.num_bootstrap_samples / BOOTSTRAP_BATCH_SIZE)):
            print("Test {}/{} (N={}, M={}): calculating bootstrap samples {}/{}...    "
                  .format(num_test + 1, args.num_tests, N, M,
                          i * BOOTSTRAP_BATCH_SIZE, args.num_bootstrap_samples),
                  end='\r')

            for test in bootstrap_tests:
                test.generate_bootstrap_samples(BOOTSTRAP_BATCH_SIZE)

        # Generate bootstrap samples that can't fit in a batch
        for test in bootstrap_tests:
            test.generate_bootstrap_samples(args.num_bootstrap_samples % BOOTSTRAP_BATCH_SIZE)

        # Once a test is completed (i.e. we collected enough covariate
        # samples), we added them to the corresponding, existing
        # collection of tests
        for (bootstrap_test, bootstrap_test_collection) in \
                [(bootstrap_tests[i], bootstrap_test_collections[i])
                 for i in range(0, len(bootstrap_tests))]:
            bootstrap_test_collection.append(bootstrap_test)

        print("Test {}/{} (N={}, M={}): Done.                                         "
              .format(num_test + 1, args.num_tests, N, M))

    except KeyboardInterrupt:
        # Breaking the for loop is sufficient, as we want to
        # analyse the results accumulated so far
        break

# Print some statistics and save the test collections for future reference
# once we reached the number of experiments / process is interrupted
for bootstrap_test_collection in bootstrap_test_collections:
    try:
        print_test_collection_result(bootstrap_test_collection)
    except KeyboardInterrupt:
        print("{}: Skipped.".format(bootstrap_test_collection[0].test_name()))
    save_test_collection(bootstrap_test_collection, in_dir="./output/")
