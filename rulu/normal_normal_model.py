import numpy as np
import pandas as pd
from scipy.stats import rankdata
from typing import Dict, List

def get_samples(
    n_samples:int, N:int, M:int, mu_X:float, mu_epsilon:float,
    sigma_sq_X:float, sigma_sq_1:float, sigma_sq_2:float,
    verbose=True, **kwargs) -> Dict[str, List[float]]:
    # Numpy's normal RNG takes sigmas instead of sigma squares
    # We need to convert the sigma squares to prevent unexpected results
    sigma_X = np.sqrt(sigma_sq_X)
    sigma_1 = np.sqrt(sigma_sq_1)
    sigma_2 = np.sqrt(sigma_sq_2)

    r = None
    s = None
    if 'r' in kwargs:
        assert 1 <= kwargs['r'] <= N
        r = int(kwargs['r'])
    if 's' in kwargs:
        assert 1 <= kwargs['s'] <= N
        s = int(kwargs['s'])

    # The lists to hold the samples
    noisy_mean = []  # This is V given sigma^2_1
    clean_mean = []  # This is V given signa^2_2
    improvement = []  # This is D

    if r is not None:
        noisy_rth_observed_value = []  # This is Y_(r)
        noisy_rth_observed_value_true = []  # This is X_I(r)
        clean_rth_observed_value = []  # This is Z_(r)
        clean_rth_observed_value_true = []  # This is X_J(r)

    if s is not None:
        noisy_sth_observed_value = []  # This is Y_(s)
        noisy_sth_observed_value_true = []  # This is X_I(s)
        clean_sth_observed_value = []  # This is Z_(s)
        clean_sth_observed_value_true = []  # This is X_J(s)

    for i in range(1, n_samples + 1):
        # Step 1
        propositions = (
            pd.DataFrame(np.random.normal(mu_X, sigma_X, N),
                         columns=['true']))

        # Step 2
        propositions['observed_noisy'] = (
                propositions['true'] +
                np.random.normal(mu_epsilon, sigma_1, N))

        # Step 3
        propositions['observed_noisy_rank'] = (
            rankdata(propositions['observed_noisy']))

        # Step 4
        noisy_chosen_true = (
            propositions[propositions['observed_noisy_rank'] > (N - M)]
            ['true']
        )

        noisy_chosen_true_mean = noisy_chosen_true.mean()
        noisy_mean.append(noisy_chosen_true_mean)

        if r is not None:
            noisy_rth_observed = (
                propositions[propositions['observed_noisy_rank'] == r])
            noisy_rth_observed_value.append(
                noisy_rth_observed['observed_noisy'].values[0])
            noisy_rth_observed_value_true.append(
                noisy_rth_observed['true'].values[0])
        if s is not None:
            noisy_sth_observed = (
                propositions[propositions['observed_noisy_rank'] == s])
            noisy_sth_observed_value.append(
                noisy_sth_observed['observed_noisy'].values[0])
            noisy_sth_observed_value_true.append(
                noisy_sth_observed['true'].values[0])

        # Step 5- repeat step 2 for sigma^2_2
        propositions['observed_clean'] = (
                propositions['true'] +
                np.random.normal(mu_epsilon, sigma_2, N))

        # Step 5- repeat 3 for sigma^2_2
        propositions['observed_clean_rank'] = (
            rankdata(propositions['observed_clean']))

        # Step 5- repeat 4 for sigma^2_2
        clean_chosen_true = (
            propositions[propositions['observed_clean_rank'] > (N - M)]
            ['true']
        )

        clean_chosen_true_mean = clean_chosen_true.mean()
        clean_mean.append(clean_chosen_true_mean)

        if r is not None:
            clean_rth_observed = (
                propositions[propositions['observed_clean_rank'] == r])
            clean_rth_observed_value.append(
                clean_rth_observed['observed_clean'].values[0])
            clean_rth_observed_value_true.append(
                clean_rth_observed['true'].values[0])
        if s is not None:
            clean_sth_observed = (
                propositions[propositions['observed_clean_rank'] == s])
            clean_sth_observed_value.append(
                clean_sth_observed['observed_clean'].values[0])
            clean_sth_observed_value_true.append(
                clean_sth_observed['true'].values[0])

        # Step 6
        improvement.append(clean_chosen_true_mean - noisy_chosen_true_mean)

        # Reporting and progress tracking - print results of 20 samples
        if (verbose == True) and (i % max(int(n_samples / 20), 1) == 0):
            print("Sample {}/{}: Noisy mean: {}, Clean mean: {}. Improvement: {} ({}%)    "
                  .format(i, n_samples,
                          np.round(noisy_chosen_true_mean, 4),
                          np.round(clean_chosen_true_mean, 4),
                          np.round(clean_chosen_true_mean - noisy_chosen_true_mean, 4),
                          np.round(100.0 * ((clean_chosen_true_mean /
                                             noisy_chosen_true_mean) - 1), 3)
                          ),
                  end="\r")

    # Making sure there is no off-by-one error on ranks
    # by checking the value of the last cycle
    # TODO: Convert checks into a unit test
    assert (np.min(propositions['observed_noisy_rank']) == 1)
    assert (np.max(propositions['observed_noisy_rank']) == N)
    assert (len(noisy_chosen_true) == M)
    assert (len(clean_chosen_true) == M)

    # Prepare result to be returned
    result = {
        'noisy_mean': noisy_mean,
        'clean_mean': clean_mean,
        'improvement': improvement
    }

    if r is not None:
        result['noisy_rth_observed'] = noisy_rth_observed_value
        result['noisy_rth_true'] = noisy_rth_observed_value_true
        result['clean_rth_observed'] = clean_rth_observed_value
        result['clean_rth_true'] = clean_rth_observed_value_true
    if s is not None:
        result['noisy_sth_observed'] = noisy_sth_observed_value
        result['noisy_sth_true'] = noisy_sth_observed_value_true
        result['clean_sth_observed'] = clean_sth_observed_value
        result['clean_sth_true'] = clean_sth_observed_value_true

    return result
