def print_run_setting(count, N, M, mu_X, sigma_X, mu_eps, sigma_1, sigma_2):
    print("Cycle {}: N = {}, M = {}, mu_X = {}, sigma_X = {}, "
          "mu_epsilon = {}, sigma_1 = {}, sigma_2 = {}"
          .format(count, N, M, np.round(mu_X, 4),
                  np.round(sigma_X, 4), np.round(mu_eps, 4),
                  np.round(sigma_1, 4), np.round(sigma_2, 4)))


def calculate_norm_and_t_improvements(cycles=500):
    # Constants
    NUM_RUNS = 1000
    T_DF = 3
    CI_HIGH_PERCENTILE = 95
    CI_LOW_PERCENTILE = 5

    # Initialise empty list to hold the results
    normal_E_Ds = []
    normal_CI_lows = []
    normal_CI_highs = []
    t_E_Ds = []
    t_CI_lows = []
    t_CI_highs = []

    for cycle in range(1, cycles + 1, 1):
        params = get_test_params()
        N = int(params['N'])
        M = int(params['M'])
        mu_X = params['mu_X']
        mu_epsilon = params['mu_epsilon']
        sigma_sq_X = params['sigma_sq_X']
        sigma_sq_1 = params['sigma_sq_1']
        sigma_sq_2 = params['sigma_sq_2']

        # Reconciling the use of sigmas by numpy
        # and sigma_sqs in the theoretical calculations
        sigma_X = np.sqrt(sigma_sq_X)
        sigma_1 = np.sqrt(sigma_sq_1)
        sigma_2 = np.sqrt(sigma_sq_2)

        print_run_setting(cycle, N, M, mu_X, sigma_X,
                          mu_epsilon, sigma_1, sigma_2)

        # Normal assumptions
        normal_improvement = (
            nnm.get_samples(
                n_samples=NUM_RUNS, N=N, M=M,
                mu_X=mu_X, mu_epsilon=mu_epsilon,
                sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1,
                sigma_sq_2=sigma_sq_2, verbose=False)['improvement']
        )

        normal_E_D = np.mean(normal_improvement)
        normal_CI_low = np.percentile(normal_improvement, CI_LOW_PERCENTILE)
        normal_CI_high = np.percentile(normal_improvement, CI_HIGH_PERCENTILE)

        normal_E_Ds.append(normal_E_D)
        normal_CI_lows.append(normal_CI_low)
        normal_CI_highs.append(normal_CI_high)

        # t-assumptions
        t_improvement = (
            ttm.get_samples(
                n_samples=NUM_RUNS, N=N, M=M,
                mu_X=mu_X, mu_epsilon=mu_epsilon,
                sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1,
                sigma_sq_2=sigma_sq_2, nu=T_DF, verbose=False)['improvement']
        )

        t_E_D = np.mean(t_improvement)
        t_CI_low = np.percentile(t_improvement, 5)
        t_CI_high = np.percentile(t_improvement, 95)

        t_E_Ds.append(t_E_D)
        t_CI_lows.append(t_CI_low)
        t_CI_highs.append(t_CI_high)

        print("Cycle {}: Normal E(D) = {}; t E(D) = {}"
              .format(cycle, normal_E_D, t_E_D))

    # Prepare result in a dict
    result = {
        'normal_E_Ds': normal_E_Ds,
        'normal_CI_lows': normal_CI_lows,
        'normal_CI_highs': normal_CI_highs,
        't_E_Ds': t_E_Ds,
        't_CI_lows': t_CI_lows,
        't_CI_highs': t_CI_highs,
    }

    return result