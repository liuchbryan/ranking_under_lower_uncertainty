import numpy as np
import pandas as pd
from scipy.stats import rankdata


def get_samples(
        n_samples: int, N: int, M: int, mu_X: float, mu_epsilon: float,
        sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, p: float,
        verbose=True):
    # Numpy's normal RNG takes sigmas instead of sigma squares
    # We need to convert the sigma squares to prevent unexpected results
    sigma_X = np.sqrt(sigma_sq_X)
    sigma_1 = np.sqrt(sigma_sq_1)
    sigma_2 = np.sqrt(sigma_sq_2)

    # The lists to hold the samples
    noisy_mean = []
    clean_mean = []
    improvement = []

    for i in range(0, n_samples):
        # Step 1
        propositions = (
            pd.DataFrame(np.random.normal(mu_X, sigma_X, N),
                         columns=['true']))

        # Step 2
        propositions['observed_noisy'] = (
                propositions['true'] +
                np.random.normal(mu_epsilon, sigma_1, N))

        # Step 2a - ranking only
        propositions['observed_noisy_rank'] = (
            rankdata(propositions['observed_noisy']))

        # Step 2c
        noisy_chosen_true = (
            propositions[propositions['observed_noisy_rank'] > (N - M)]
            ['true']
        )

        noisy_chosen_true_mean = noisy_chosen_true.mean()
        noisy_mean.append(noisy_chosen_true_mean)

        # Step 3 - repeat step 2 for sigma^2_2
        selector = np.random.binomial(1, p, N)
        propositions['observed_clean'] = (
                propositions['true'] +
                (selector * np.random.normal(mu_epsilon, sigma_2, N) +
                 (1 - selector) * np.random.normal(mu_epsilon, sigma_1, N))
        )

        # Step 3a - repeat 2a for sigma^2_2
        propositions['observed_clean_rank'] = (
            rankdata(propositions['observed_clean']))

        # Step 3c - repeat 3c for sigma^2_2
        clean_chosen_true = (
            propositions[propositions['observed_clean_rank'] > (N - M)]
            ['true']
        )

        clean_chosen_true_mean = clean_chosen_true.mean()
        clean_mean.append(clean_chosen_true_mean)

        # Step 6
        improvement.append(clean_chosen_true_mean - noisy_chosen_true_mean)

        # Reporting and progress tracking - print results of 20 samples
        if (verbose == True) and (i % (n_samples / 20) == 0):
            print("Noisy: {}, Clean: {}. Improvement: {} ({}%)".format(
                np.round(noisy_chosen_true_mean, 4),
                np.round(clean_chosen_true_mean, 4),
                np.round(clean_chosen_true_mean - noisy_chosen_true_mean, 4),
                np.round(100.0 * ((clean_chosen_true.mean() /
                                   noisy_chosen_true.mean()) - 1), 3)
            ))

    # Making sure there is no off-by-one error on ranks
    # by checking the value of the last cycle
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

    return result
