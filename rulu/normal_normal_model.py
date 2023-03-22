import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, betabinom
from scipy.special import owens_t
from typing import Dict, List

from rulu.utils import fit_beta_distribution_params

ALPHA_BLOM = 0.4


def get_samples(
        n_samples: int,
        N: int,
        M: int,
        mu_X: float,
        mu_epsilon: float,
        sigma_sq_X: float,
        sigma_sq_1: float,
        sigma_sq_2: float,
        verbose=True,
        **kwargs
) -> Dict[str, List[float]]:
    # numpy's normal RNG takes sigmas instead of sigma squares
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
    clean_mean = []  # This is V given sigma^2_2
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

        # Step 2a - ranking
        propositions['observed_noisy_rank'] = (
            rankdata(propositions['observed_noisy']))

        # Step 2c
        noisy_chosen_true = (
            propositions[propositions['observed_noisy_rank'] > (N - M)]
            ['true']
        )

        noisy_chosen_true_mean = noisy_chosen_true.mean()
        noisy_mean.append(noisy_chosen_true_mean)

        # Step 2a - extract Y_(r) and Step 2b - extract X_I(r)
        # We also extract Y_(s) and X_I(s) in case we need them to evaluate some other quantities
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

        # Step 3 - repeat step 2 for sigma^2_2
        propositions['observed_clean'] = (
                propositions['true'] +
                np.random.normal(mu_epsilon, sigma_2, N))

        # Step 3a - repeat 2a for sigma^2_2
        propositions['observed_clean_rank'] = (
            rankdata(propositions['observed_clean']))

        # Step 3c - repeat 2c for sigma^2_2
        clean_chosen_true = (
            propositions[propositions['observed_clean_rank'] > (N - M)]
            ['true']
        )

        clean_chosen_true_mean = clean_chosen_true.mean()
        clean_mean.append(clean_chosen_true_mean)

        # Step 2a - extract Z_(s) and Step 2b - extract X_J(s)
        # We also extract Z_(r) and X_J(r) in case we need them to evaluate some other quantities
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
        if verbose and (i % max(int(n_samples / 20), 1) == 0):
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


# E(Y_(r)) / E(Z_(s))
def E_Yr():
    pass


def E_Zs():
    pass


# E(X_I(r)) / E(X_J(s))
def E_XIr(
        r: int,
        N: int,
        mu_X: float,
        sigma_sq_X: float,
        sigma_sq_eps: float = None,
        sigma_sq_1: float = None,
        **kwargs
):
    _sigma_sq_eps = sigma_sq_eps or sigma_sq_1
    assert(_sigma_sq_eps is not None), "Require non-None for either `sigma_sq_eps` or `sigma_sq_1`."
    return (
        mu_X +
        sigma_sq_X / np.sqrt(sigma_sq_X + _sigma_sq_eps) * norm.ppf((r - ALPHA_BLOM) / (N - 2 * ALPHA_BLOM + 1))
    )


def E_XJs(
        s: int,
        N: int,
        mu_X: float,
        sigma_sq_X: float,
        sigma_sq_eps: float = None,
        sigma_sq_2: float = None,
        **kwargs
):
    """
    Alias for E_XIr.
    They have the same functional form, with the rank r replaced by s and sigma_sq_1 replaced by sigma_sq_2.
    """
    _sigma_sq_eps = sigma_sq_eps or sigma_sq_2
    assert(_sigma_sq_eps is not None), "Require non-None for either `sigma_sq_eps` or `sigma_sq_2`."
    return E_XIr(r=s, N=N, mu_X=mu_X, sigma_sq_X=sigma_sq_X, sigma_sq_eps=_sigma_sq_eps, **kwargs)


# E(V)
def E_V(mu_X: float, sigma_sq_X: float, sigma_sq_eps: float, N: int, M: int, **kwargs):
    acc = 0
    for r in range(N-M+1, N+1):
        acc += E_XIr(r=r, mu_X=mu_X, sigma_sq_X=sigma_sq_X, sigma_sq_eps=sigma_sq_eps, N=N, **kwargs)
    return acc / M


# E(D)
def E_D(mu_X: float, sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, N: int, M: int, **kwargs):
    return(
        E_V(mu_X=mu_X, sigma_sq_X=sigma_sq_X, sigma_sq_eps=sigma_sq_2, N=N, M=M, **kwargs) -
        E_V(mu_X=mu_X, sigma_sq_X=sigma_sq_X, sigma_sq_eps=sigma_sq_1, N=N, M=M, **kwargs)
    )


# Var(Y_(r)) / Var(Z_(s))
def var_Yr(r: int, N: int, sigma_sq_X: float, sigma_sq_eps: float = None, sigma_sq_1: float = None, **kwargs):
    _sigma_sq_eps = sigma_sq_eps or sigma_sq_1
    assert (_sigma_sq_eps is not None), "Require non-None for either `sigma_sq_eps` or `sigma_sq_1`."
    return(
        r * (N - r + 1) / (N + 1) ** 2 / (N + 2) *
        (sigma_sq_X + _sigma_sq_eps) / (norm.pdf(norm.ppf(r / (N + 1)))) ** 2
    )


def var_Zs(s: int, sigma_sq_X: float, sigma_sq_2: float, N: int, **kwargs):
    return var_Yr(r=s, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_eps=sigma_sq_2, **kwargs)


# Var(X_I(r)) / Var(X_J(s))
def var_XIr(r: int, N: int, sigma_sq_X: float, sigma_sq_eps: float = None, sigma_sq_1: float = None, **kwargs):
    _sigma_sq_eps = sigma_sq_eps or sigma_sq_1
    assert(_sigma_sq_eps is not None), "Require non-None for either `sigma_sq_eps` or `sigma_sq_1`."
    return (
        (_sigma_sq_eps * sigma_sq_X / (sigma_sq_X + _sigma_sq_eps)) +
        sigma_sq_X ** 2 / (sigma_sq_X + _sigma_sq_eps) *
        (r * (N - r + 1)) / ((N + 1) ** 2 * (N + 2)) /
        (norm.pdf(norm.ppf(r / (N + 1)))) ** 2
    )


def var_XJs(s: int, N: int, sigma_sq_X: float, sigma_sq_eps: float = None, sigma_sq_2: float = None, **kwargs):
    _sigma_sq_eps = sigma_sq_eps or sigma_sq_2
    assert (_sigma_sq_eps is not None), "Require non-None for either `sigma_sq_eps` or `sigma_sq_2`."
    return var_XIr(r=s, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_eps=_sigma_sq_eps, **kwargs)


# Var(V1) / Var(V2)
def cov_XIr_XIs(r: int, s: int, sigma_sq_X: float, sigma_sq_eps: float, N: int, **kwargs):
    if r == s:
        return var_XIr(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_eps=sigma_sq_eps, **kwargs)

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


def var_V(sigma_sq_X: float, sigma_sq_eps: float, N: int, M: int, **kwargs):
    acc = 0.0
    for r in range(N - M + 1, N + 1):
        acc += var_XIr(r=r, sigma_sq_X=sigma_sq_X, sigma_sq_eps=sigma_sq_eps, N=N)
        for s in range(r + 1, N + 1):
            acc += 2.0 * cov_XIr_XIs(r=r, s=s, sigma_sq_X=sigma_sq_X, sigma_sq_eps=sigma_sq_eps, N=N)

    return acc / (M ** 2)


# var(D)
def cov_Yr_Zs(r: int, s: int, sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, N: int, **kwargs):
    return(
        sigma_sq_X ** 2 / np.sqrt(sigma_sq_X + sigma_sq_1) / np.sqrt(sigma_sq_X + sigma_sq_2) *
        r * (N - s + 1) / (N + 1) ** 2 / (N + 2) /
        (norm.pdf(norm.ppf(r / (N + 1))) * norm.pdf(norm.ppf(s / (N + 1))))
    )


def cov_Yr_Zs_second_order(
        r: int, s: int, sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, N: int, **kwargs):
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


def _E_ZIr(
        r: int,
        N: int,
        mu_X: float,
        mu_eps: float,
        sigma_sq_X: float,
        sigma_sq_1: float,
        **kwargs
):
    return(
        E_XIr(r=r, N=N, mu_X=mu_X, sigma_sq_X=sigma_sq_X, sigma_sq_eps=sigma_sq_1, **kwargs) + mu_eps
    )


def _var_ZIr(
        r: int,
        N: int,
        sigma_sq_X: float,
        sigma_sq_1: float,
        sigma_sq_2: float,
        **kwargs
):
    return(
        var_XIr(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_eps=sigma_sq_1, **kwargs) + sigma_sq_2
    )


def F_Zn(z: float, mu_X: float, mu_eps: float, sigma_sq_X: float, sigma_sq_2: float, **kwargs):
    """
    CDF of the r.v. Z_n, which follow N(mu_X + mu_eps, sigma_sq_X + sigma_sq_2)
    :param z: value to evaluate CDF at
    :param mu_X:
    :param mu_eps:
    :param sigma_sq_X:
    :param sigma_sq_2:
    :param kwargs:
    :return:
    """
    return norm.cdf((z - (mu_X + mu_eps)) / (sigma_sq_X + sigma_sq_2) ** 0.5)


def f_Zn(z: float, mu_X: float, mu_eps: float, sigma_sq_X: float, sigma_sq_2: float, **kwargs):
    """
    PDF of the r.v. Z_n, which follow N(mu_X + mu_eps, sigma_sq_X + sigma_sq_2)
    :param z: value to evaluate PDF at
    :param mu_X:
    :param mu_eps:
    :param sigma_sq_X:
    :param sigma_sq_2:
    :param kwargs:
    :return:
    """
    return(
        norm.pdf((z - (mu_X + mu_eps)) / (sigma_sq_X + sigma_sq_2) ** 0.5) /
        (sigma_sq_X + sigma_sq_2) ** 0.5
    )


def fp_Zn(z: float, mu_X: float, mu_eps: float, sigma_sq_X: float, sigma_sq_2: float, **kwargs):
    """
    Derivative of the PDF of the r.v. Z_n, which follow N(mu_X + mu_eps, sigma_sq_X + sigma_sq_2).
    `fp` in the function name stands for "f-prime".
    :param z: value to evaluate derivative of PDF at
    :param mu_X:
    :param mu_eps:
    :param sigma_sq_X:
    :param sigma_sq_2:
    :param kwargs:
    :return:
    """
    return(
        -(z - (mu_X + mu_eps)) / (sigma_sq_X + sigma_sq_2) ** 1.5 *
        norm.pdf((z - (mu_X + mu_eps)) / (sigma_sq_X + sigma_sq_2) ** 0.5)
    )


def fpp_Zn(z: float, mu_X: float, mu_eps: float, sigma_sq_X: float, sigma_sq_2: float, **kwargs):
    """
    Second derivative of the PDF of the r.v. Z_n, which follow N(mu_X + mu_eps, sigma_sq_X + sigma_sq_2).
    `fpp` in the function name stands for "f-prime-prime".
    :param z:
    :param mu_X:
    :param mu_eps:
    :param sigma_sq_X:
    :param sigma_sq_2:
    :param kwargs:
    :return:
    """
    mu = mu_X + mu_eps
    sigma_sq = sigma_sq_X + sigma_sq_2
    return(
        (-sigma_sq + (z - mu) ** 2) / sigma_sq ** 2 *
        f_Zn(z=z, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2)
    )


def f3_Zn(z: float, mu_X: float, mu_eps: float, sigma_sq_X: float, sigma_sq_2: float, **kwargs):
    """
    Third derivative of the PDF of the r.v. Z_n, which follow N(mu_X + mu_eps, sigma_sq_X + sigma_sq_2).
    `f3` in the function name stands for "f^(3)".
    :param z:
    :param mu_X:
    :param mu_eps:
    :param sigma_sq_X:
    :param sigma_sq_2:
    :param kwargs:
    :return:
    """
    mu = mu_X + mu_eps
    sigma_sq = sigma_sq_X + sigma_sq_2
    return(
        (3 * (z - mu) * sigma_sq - (z - mu) ** 3) / sigma_sq ** 3 *
        f_Zn(z=z, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2)
    )


def f4_Zn(z: float, mu_X: float, mu_eps: float, sigma_sq_X: float, sigma_sq_2: float, **kwargs):
    """
    Fourth derivative of the PDF of the r.v. Z_n, which follow N(mu_X + mu_eps, sigma_sq_X + sigma_sq_2).
    `f4` in the function name stands for "f^(4)".
    :param z:
    :param mu_X:
    :param mu_eps:
    :param sigma_sq_X:
    :param sigma_sq_2:
    :param kwargs:
    :return:
    """
    mu = mu_X + mu_eps
    sigma_sq = sigma_sq_X + sigma_sq_2
    return(
        (3 * sigma_sq ** 2 - 6 * (z - mu) ** 2 * sigma_sq + (z - mu) ** 4) / sigma_sq ** 4 *
        f_Zn(z=z, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2)
    )


def f5_Zn(z: float, mu_X: float, mu_eps: float, sigma_sq_X: float, sigma_sq_2: float, **kwargs):
    """
    Fifth derivative of the PDF of the r.v. Z_n, which follow N(mu_X + mu_eps, sigma_sq_X + sigma_sq_2).
    `f5` in the function name stands for "f^(5)".
    :param z:
    :param mu_X:
    :param mu_eps:
    :param sigma_sq_X:
    :param sigma_sq_2:
    :param kwargs:
    :return:
    """
    mu = mu_X + mu_eps
    sigma_sq = sigma_sq_X + sigma_sq_2
    return(
        (-15 * (z - mu) * sigma_sq ** 2 + 10 * (z - mu) ** 3 * sigma_sq - (z - mu) ** 5) / sigma_sq ** 5 *
        f_Zn(z=z, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2)
    )


def _E_F_Zn_ZIr_taylor_second_order(
        r: int,
        N: int,
        mu_X: float,
        mu_eps: float,
        sigma_sq_X: float,
        sigma_sq_1: float,
        sigma_sq_2: float,
        **kwargs
):
    val_E_ZIr = _E_ZIr(r=r, N=N, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, **kwargs)
    val_var_ZIr = _var_ZIr(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, **kwargs)
    return(
        F_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2) +
        1 / 2 * fp_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2) * val_var_ZIr
    )


def _var_F_Zn_ZIr_taylor_second_order(
        r: int,
        N: int,
        mu_X: float,
        mu_eps: float,
        sigma_sq_X: float,
        sigma_sq_1: float,
        sigma_sq_2: float,
        **kwargs
):
    val_E_ZIr = _E_ZIr(r=r, N=N, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, **kwargs)
    val_var_ZIr = _var_ZIr(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, **kwargs)

    return(
        f_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs) ** 2 *
        val_var_ZIr -
        1 / 4 *
        fp_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs) ** 2 *
        val_var_ZIr ** 2
    )


def _E_F_Zn_ZIr_taylor_fourth_order(
        r: int,
        N: int,
        mu_X: float,
        mu_eps: float,
        sigma_sq_X: float,
        sigma_sq_1: float,
        sigma_sq_2: float,
        **kwargs
):
    val_E_ZIr = _E_ZIr(r=r, N=N, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, **kwargs)
    val_var_ZIr = _var_ZIr(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, **kwargs)
    val_4th_central_moment_ZIr = 3 * val_var_ZIr ** 2  # Assuming Z_I(r) is normal enough

    val_F_Zn_E_ZIr = F_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs)
    val_fp_Zn_E_ZIr = fp_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs)
    val_f3_Zn_E_ZIr = f3_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs)
    return(
        val_F_Zn_E_ZIr +
        1 / 2 * val_fp_Zn_E_ZIr * val_var_ZIr +
        1 / 24 * val_f3_Zn_E_ZIr * val_4th_central_moment_ZIr
    )


def _var_F_Zn_ZIr_taylor_fourth_order(
        r: int,
        N: int,
        mu_X: float,
        mu_eps: float,
        sigma_sq_X: float,
        sigma_sq_1: float,
        sigma_sq_2: float,
        **kwargs
):
    val_E_ZIr = _E_ZIr(r=r, N=N, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, **kwargs)
    val_var_ZIr = _var_ZIr(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, **kwargs)
    val_4th_central_moment_ZIr = 3 * val_var_ZIr ** 2     # Assuming Z_I(r) is normal enough

    val_f_Zn_E_ZIr = f_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs)
    val_fp_Zn_E_ZIr = fp_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs)
    val_fpp_Zn_E_ZIr = (
        fpp_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs))
    val_f3_Zn_E_ZIr = f3_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs)

    return(
        val_f_Zn_E_ZIr ** 2 * val_var_ZIr -
        1 / 4 * val_fp_Zn_E_ZIr ** 2 * val_var_ZIr ** 2 +
        (1 / 4 * val_fp_Zn_E_ZIr ** 2 +
         1 / 3 * val_f_Zn_E_ZIr * val_fpp_Zn_E_ZIr -
         1 / 24 * val_fp_Zn_E_ZIr * val_f3_Zn_E_ZIr * val_var_ZIr) * val_4th_central_moment_ZIr -
        1 / 576 * val_f3_Zn_E_ZIr ** 2 * val_4th_central_moment_ZIr ** 2
    )


def _E_F_Zn_ZIr_taylor_sixth_order(
        r: int,
        N: int,
        mu_X: float,
        mu_eps: float,
        sigma_sq_X: float,
        sigma_sq_1: float,
        sigma_sq_2: float,
        **kwargs
):
    # Assuming normal, the 3rd, 5th central moments are zero
    val_E_ZIr = _E_ZIr(r=r, N=N, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, **kwargs)
    val_var_ZIr = _var_ZIr(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, **kwargs)
    val_4th_central_moment_ZIr = 3 * val_var_ZIr ** 2     # Assuming Z_I(r) is normal enough
    val_6th_central_moment_ZIr = 15 * val_var_ZIr ** 3    # Assuming Z_I(r) is normal enough

    val_F_Zn_E_ZIr = F_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs)
    val_fp_Zn_E_ZIr = fp_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs)
    val_f3_Zn_E_ZIr = f3_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs)
    val_f5_Zn_E_ZIr = f5_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs)

    return (
            val_F_Zn_E_ZIr +
            1 / 2 * val_fp_Zn_E_ZIr * val_var_ZIr +
            1 / 24 * val_f3_Zn_E_ZIr * val_4th_central_moment_ZIr +
            1 / 720 * val_f5_Zn_E_ZIr * val_6th_central_moment_ZIr
    )


def _var_F_Zn_ZIr_taylor_sixth_order(
        r: int,
        N: int,
        mu_X: float,
        mu_eps: float,
        sigma_sq_X: float,
        sigma_sq_1: float,
        sigma_sq_2: float,
        **kwargs
):
    # Assuming normal, the 3rd, 5th central moments are zero
    val_E_ZIr = _E_ZIr(r=r, N=N, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, **kwargs)
    val_var_ZIr = _var_ZIr(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, **kwargs)
    val_4th_central_moment_ZIr = 3 * val_var_ZIr ** 2     # Assuming Z_I(r) is normal enough
    val_6th_central_moment_ZIr = 15 * val_var_ZIr ** 3    # Assuming Z_I(r) is normal enough

    val_f_Zn_E_ZIr = f_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs)
    val_fp_Zn_E_ZIr = fp_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs)
    val_fpp_Zn_E_ZIr = (
        fpp_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs))
    val_f3_Zn_E_ZIr = f3_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs)
    val_f4_Zn_E_ZIr = f4_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs)
    val_f5_Zn_E_ZIr = f5_Zn(val_E_ZIr, mu_X=mu_X, mu_eps=mu_eps, sigma_sq_X=sigma_sq_X, sigma_sq_2=sigma_sq_2, **kwargs)

    return(
        val_f_Zn_E_ZIr ** 2 * val_var_ZIr -
        1 / 4 * val_fp_Zn_E_ZIr ** 2 * val_var_ZIr ** 2 +
        (1 / 4 * val_fp_Zn_E_ZIr ** 2 +
         1 / 3 * val_f_Zn_E_ZIr * val_fpp_Zn_E_ZIr -
         1 / 24 * val_fp_Zn_E_ZIr * val_f3_Zn_E_ZIr * val_var_ZIr) * val_4th_central_moment_ZIr -
        1 / 576 * val_f3_Zn_E_ZIr ** 2 * val_4th_central_moment_ZIr ** 2 +
        (1 / 36 * val_fpp_Zn_E_ZIr ** 2 +
         1 / 24 * val_fp_Zn_E_ZIr * val_f3_Zn_E_ZIr +
         1 / 60 * val_f_Zn_E_ZIr * val_f4_Zn_E_ZIr -
         1 / 720 * val_fp_Zn_E_ZIr * val_f5_Zn_E_ZIr * val_var_ZIr -
         1 / 8640 * val_f3_Zn_E_ZIr * val_f5_Zn_E_ZIr * val_4th_central_moment_ZIr) * val_6th_central_moment_ZIr -
        1 / 518400 * val_f5_Zn_E_ZIr ** 2 * val_6th_central_moment_ZIr ** 2
    )


def _E_F_Zn_ZIr_owen_integrals(
        r: int,
        N: int,
        sigma_sq_X: float,
        sigma_sq_1: float,
        sigma_sq_2: float,
        **kwargs
):
    return _E_P_ZN_ZIr(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, **kwargs)


def _var_F_Zn_ZIr_owen_integrals(
        r: int,
        N: int,
        sigma_sq_X: float,
        sigma_sq_1: float,
        sigma_sq_2: float,
        **kwargs
):
    return _var_P_ZN_ZIr(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, **kwargs)


def _E_ZStar(r: int, N: int, sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, **kwargs):
    # The (mu_X + mu_eps) terms in E(Z_I(r)) and Z^* cancels each other out
    return(
        sigma_sq_X / np.sqrt(sigma_sq_X + sigma_sq_1) / np.sqrt(sigma_sq_X + sigma_sq_2) *
        norm.ppf((r - ALPHA_BLOM) / (N - 2 * ALPHA_BLOM + 1))
    )


def _var_ZStar(r: int, N: int, sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, **kwargs):
    return(
        _var_ZIr(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, **kwargs) /
        (sigma_sq_X + sigma_sq_2)
    )


def _E_P_ZN_ZIr(r: int, N: int, sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, **kwargs):
    return(
        norm.cdf(_E_ZStar(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, **kwargs) /
                 np.sqrt(1 + _var_ZStar(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2,
                                        **kwargs)))
    )


def _var_P_ZN_ZIr(r: int, N: int, sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, **kwargs):
    E_P = _E_P_ZN_ZIr(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, **kwargs)
    E_ZStar = _E_ZStar(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, **kwargs)
    var_ZStar = _var_ZStar(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, **kwargs)
    return(
        E_P * (1 - E_P) -
        2 * owens_t(E_ZStar / np.sqrt(1 + var_ZStar), 1 / np.sqrt(1 + 2 * var_ZStar))
    )


def _var_XIr_XJs(r: int, s: int, sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, N: int, **kwargs):
    return (
        sigma_sq_X * sigma_sq_1 * sigma_sq_2 /
        (sigma_sq_X * sigma_sq_1 + sigma_sq_X * sigma_sq_2 + sigma_sq_1 * sigma_sq_2) +

        (sigma_sq_X * sigma_sq_1 /
         (sigma_sq_X * sigma_sq_1 + sigma_sq_X * sigma_sq_2 + sigma_sq_1 * sigma_sq_2)) ** 2 *
        s * (N - s + 1) / (N + 1) ** 2 / (N + 2) *
        (sigma_sq_X + sigma_sq_2) / norm.pdf(norm.ppf(s / (N+1))) ** 2 +

        sigma_sq_X ** 2 / (sigma_sq_X + sigma_sq_1) *
        r * (N - r + 1) / (N + 1) ** 2 / (N + 2) /
        norm.pdf(norm.ppf(r / (N+1))) ** 2
    )


def cov_XIr_XJs(r: int, s: int, sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, N: int, **kwargs):
    alpha, beta = fit_beta_distribution_params(
        _E_P_ZN_ZIr(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, **kwargs),
        _var_P_ZN_ZIr(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, **kwargs))

    prob_Ir_eq_Js = betabinom.pmf(s-1, N-1, alpha, beta)
    return(
        prob_Ir_eq_Js *
        _var_XIr_XJs(r=r, s=s, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, N=N, **kwargs) +

        (1 - prob_Ir_eq_Js) *
        sigma_sq_X ** 2 / (sigma_sq_X + sigma_sq_1) / (sigma_sq_X + sigma_sq_2) *
        cov_Yr_Zs(r=r, s=s, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, N=N, **kwargs)
    )


def cov_XIr_XJs_second_order(
        r: int, s: int, sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, N: int, **kwargs):
    alpha, beta = fit_beta_distribution_params(
        _E_P_ZN_ZIr(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, **kwargs),
        _var_P_ZN_ZIr(r=r, N=N, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, **kwargs))

    prob_Ir_eq_Js = betabinom.pmf(s - 1, N - 1, alpha, beta)

    return (
        prob_Ir_eq_Js *
        _var_XIr_XJs(r=r, s=s, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, N=N, **kwargs) +

        (1 - prob_Ir_eq_Js) *
        sigma_sq_X ** 2 / (sigma_sq_X + sigma_sq_1) / (sigma_sq_X + sigma_sq_2) *
        cov_Yr_Zs_second_order(
            r=r, s=s, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, N=N, **kwargs)
    )


def cov_V1_V2(sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, N: int, M: int, **kwargs):
    acc = 0
    for r in range(N - M + 1, N + 1):
        for s in range(N - M + 1, N + 1):
            acc += cov_XIr_XJs(r=r, s=s, sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, N=N)

    return acc / M ** 2


def var_D(sigma_sq_X: float, sigma_sq_1: float, sigma_sq_2: float, N: int, M: int, **kwargs):
    return (
        var_V(sigma_sq_X=sigma_sq_X, sigma_sq_eps=sigma_sq_2, N=N, M=M, **kwargs) +
        var_V(sigma_sq_X=sigma_sq_X, sigma_sq_eps=sigma_sq_1, N=N, M=M, **kwargs) -
        2.0 * cov_V1_V2(sigma_sq_X=sigma_sq_X, sigma_sq_1=sigma_sq_1, sigma_sq_2=sigma_sq_2, N=N, M=M, **kwargs))
