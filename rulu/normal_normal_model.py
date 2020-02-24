import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, betabinom
from typing import Dict, List

from rulu.utils import fit_beta_distribution_params

ALPHA_BLOM = 0.4


def get_samples(
    n_samples: int, N: int, M: int, mu_X: float, mu_epsilon: float,
    sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float,
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


# Y_(r) / Z_(s)
def E_Yr():
    pass


def E_Zs():
    pass


def var_Yr(r: int, sigma_sq_X:float, sigma_sq_eps: float, N:int, **kwargs):
    pass


def var_Zs(s: int, sigma_sq_X: float, sigma_sq_2: float, N: int, **kwargs):
    return var_Yr(r=s, sigma_sq_X=sigma_sq_X, sigma_sq_eps=sigma_sq_2, N=N, **kwargs)


# X_I(r) / X_J(s)
def E_XIr(r: int, mu_X: float, sigma_sq_X: float, sigma_sq_eps: float, N: int, **kwargs):
    return (
        mu_X +
        sigma_sq_X / np.sqrt(sigma_sq_X + sigma_sq_eps) *
        norm.ppf((r - ALPHA_BLOM) / (N - 2 * ALPHA_BLOM + 1))
    )


def E_XJs(s: int, mu_X: float, sigma_sq_X: float, sigma_sq_2: float, N: int, **kwargs):
    """
    Alias for E_XIr. They have the same functional form.
    """
    return E_XIr(r=s, mu_X=mu_X, sigma_sq_X=sigma_sq_X, sigma_sq_eps=sigma_sq_2, N=N, **kwargs)


def var_XIr(r: int, sigma_sq_X: float, sigma_sq_eps: float, N: int, **kwargs):
    return (
        (sigma_sq_eps * sigma_sq_X / (sigma_sq_X + sigma_sq_eps)) +
        sigma_sq_X ** 2 / (sigma_sq_X + sigma_sq_eps) *
        (r * (N - r + 1)) /  ((N + 1) ** 2 * (N + 2)) /
        (norm.pdf(norm.ppf(r / (N + 1)))) ** 2
    )


def var_XJs(s: int, sigma_sq_X: float, sigma_sq_2: float, N: int, **kwargs):
    return var_XIr(r=s, sigma_sq_X=sigma_sq_X, sigma_sq_eps=sigma_sq_2, N=N, **kwargs)


def cov_XIr_XIs(r: int, s: int, sigma_sq_X: float, sigma_sq_eps: float, N: int, **kwargs):
    if r == s:
        return var_XIr(r, sigma_sq_X, sigma_sq_eps, N, **kwargs)

    # The formula assumes r < s, though if the input
    # r is larger than s, then we just need to swap them
    # as covariance function is symmetric.
    r_act = (r if r < s else s)
    s_act = (s if r < s else r)

    return (
            sigma_sq_X ** 2 / (sigma_sq_X + sigma_sq_eps) *
            (r_act * (N - s_act + 1)) / ((N + 1) ** 2 * (N + 2)) /
            norm.pdf(norm.ppf(r_act / (N + 1))) /
            norm.pdf(norm.ppf(s_act / (N + 1)))
    )


# D
def cov_Yr_Zs(r: int, s:int, sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, N: int, **kwargs):
    return(
        sigma_sq_X ** 2 / np.sqrt(sigma_sq_X + sigma_sq_1) / np.sqrt(sigma_sq_X + sigma_sq_2) *
        r * (N - s + 1) / (N + 1) ** 2 / (N + 2) /
        (norm.pdf(norm.ppf(r / (N + 1))) * norm.pdf(norm.ppf(s / (N + 1))))
    )


def cov_Yr_Zs_second_order(
        r: int, s:int, sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, N: int, **kwargs):
    # Clear up the rest of the notations
    p = lambda r: r / (N + 1)
    q = lambda r: (N - r + 1) / (N + 1)
    Q = lambda r: norm.ppf(p(r))
    Qp = lambda r: 1 / (norm.pdf(Q(r)))
    Qpp = lambda r: Q(r) / (norm.pdf(Q(r))) ** 2
    Qppp = lambda r: (1 + 2 * Q(r) ** 2) / (norm.pdf(Q(r))) ** 3

    return (
        sigma_sq_X ** 2 / (np.sqrt(sigma_sq_X + sigma_sq_1) * np.sqrt(sigma_sq_X + sigma_sq_2)) *
        (p(r) * q(s) / (N + 2) * Qp(r) * Qp(s) +
         p(r) * q(s) / (N + 2) ** 2 * (
             (q(r) - p(r)) * Qpp(r) * Qp(s) +
             (q(s) - p(s)) * Qp(r) * Qpp(s) +
             1.0 / 2 * p(r) * q(r) * Qppp(r) * Qp(s) +
             1.0 / 2 * p(s) * q(s) * Qp(r) * Qppp(s) +
             1.0 / 2 * p(r) * q(s) * Qpp(r) * Qpp(s)
         ))
    )


def _E_P_ZN_ZIr(r: int, sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, N: int, **kwargs):
    return(
        norm.cdf(sigma_sq_X / np.sqrt(sigma_sq_X + sigma_sq_1) / np.sqrt(sigma_sq_X + sigma_sq_2) *
                 norm.ppf((r - ALPHA_BLOM) / (N - 2 * ALPHA_BLOM + 1)))
    )


def _var_P_ZN_ZIr(r: int, sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, N: int, **kwargs):
    return(
        (var_XIr(r=r, sigma_sq_X=sigma_sq_X, sigma_sq_eps=sigma_sq_1, N=N, **kwargs) + sigma_sq_2) *
        norm.pdf(sigma_sq_X / np.sqrt(sigma_sq_X + sigma_sq_1) / np.sqrt(sigma_sq_X + sigma_sq_2) *
                 norm.ppf((r - ALPHA_BLOM) / (N - 2 * ALPHA_BLOM + 1))) ** 2 /
        (sigma_sq_X + sigma_sq_2)
    )


def cov_XIr_XJs(r: int, s: int, sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, N: int, **kwargs):
    alpha, beta = fit_beta_distribution_params(
        _E_P_ZN_ZIr(r=r, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, N=N, **kwargs),
        _var_P_ZN_ZIr(r=r,  sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, N=N, **kwargs))

    prob_Ir_eq_Js = betabinom.pmf(s-1, N-1, alpha, beta)
    return(
        prob_Ir_eq_Js *
        sigma_sq_X * sigma_sq_1 * sigma_sq_2 /
        (sigma_sq_X * sigma_sq_1 + sigma_sq_X * sigma_sq_2 + sigma_sq_1 * sigma_sq_2) +

        (1 - prob_Ir_eq_Js) *
        sigma_sq_X ** 2 / (sigma_sq_X + sigma_sq_1) / (sigma_sq_X + sigma_sq_2) *
        cov_Yr_Zs(r=r, s=s, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, N=N, **kwargs)
    )


def cov_XIr_XJs_second_order(
        r: int, s: int, sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, N: int, **kwargs):
    alpha, beta = fit_beta_distribution_params(
        _E_P_ZN_ZIr(r=r, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, N=N, **kwargs),
        _var_P_ZN_ZIr(r=r, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, N=N, **kwargs))

    prob_Ir_eq_Js = betabinom.pmf(s - 1, N - 1, alpha, beta)

    return (
        prob_Ir_eq_Js *
        sigma_sq_X * sigma_sq_1 * sigma_sq_2 /
        (sigma_sq_X * sigma_sq_1 + sigma_sq_X * sigma_sq_2 + sigma_sq_1 * sigma_sq_2) +

        (1 - prob_Ir_eq_Js) *
        sigma_sq_X ** 2 / (sigma_sq_X + sigma_sq_1) / (sigma_sq_X + sigma_sq_2) *
        cov_Yr_Zs_second_order(
            r=r, s=s, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, N=N, **kwargs)
    )

