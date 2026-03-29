"""Evaluation utilities: log-likelihood, prediction error, pricing performance,
and K-fold cross-validation for elasticity estimation."""

import numpy as np
from optimal_pricing.optimization import solve_ppp, ProfitData, ConstraintData


def sum_log_fact(D):
    """Compute sum of log(k!) for all integer entries k in D (used for exact Poisson LL)."""
    D = np.asarray(D, dtype=int)
    max_k = D.max()
    log_vals = np.log(np.arange(max_k) + 1)
    log_fact = np.concatenate(([0.0], np.cumsum(log_vals)))
    return log_fact[D].sum()


def ll(Etilde, Dk, Pitildek):
    """Poisson log-likelihood of observed demand Dk given model parameters Etilde.

    ll = sum_i sum_t [ Dk * Y - exp(Y) - log(Dk!) ]
    where Y = Etilde @ Pitildek.
    """
    Y = Etilde @ Pitildek
    return np.sum(Dk * Y - np.exp(Y)) - sum_log_fact(Dk)


def error(Etilde, Dk, Pitildek):
    """Median absolute percentage error between predicted and observed demand."""
    Dpred = np.exp(Etilde @ Pitildek)
    return np.median(np.abs(Dk - Dpred) / Dk)


def pred_poisson_noise(Etilde, Dk, Pitildek):
    rng = np.random.default_rng(seed=9)
    Dpred = np.exp(Etilde @ Pitildek)
    Dpred_poisson = rng.poisson(lam=Dpred)
    return np.median(np.abs(Dpred_poisson - Dpred) / Dpred)


def pricing_performance(Etilde, Dk, Pitildek, pnom_sim, cost_sim, Esim):
    """Evaluate profit from optimal pricing under the estimated elasticity model.

    Solves the product-pricing problem (PPP) using the estimated Etilde, then
    evaluates the resulting profit under the *true* elasticity Esim.  This
    measures how much profit is left on the table due to estimation error.

    Returns the true profit achieved by applying the estimated-optimal prices.
    """
    n = Dk.shape[0]
    E = Etilde[:, :-1]
    dnom = np.exp(Etilde[:, -1])
    r_nom = dnom * pnom_sim.flatten()
    kappa_nom = dnom * cost_sim
    profit_data = ProfitData(r_nom=r_nom, kappa_nom=kappa_nom, elasticity=E)
    constraint_data = ConstraintData(pi_min=np.log(0.8) * np.ones(n), pi_max=np.log(1.2) * np.ones(n))
    result = solve_ppp(profit_data, constraint_data, method='NLP')
    pi_star = np.log(result.price_changes)
    # True profit at the estimated-optimal prices
    rate = np.exp(Esim @ pi_star)
    rng = np.random.default_rng(seed=0)
    poisson = rng.poisson(lam=rate.reshape(-1, 1), size=(n, 100))
    prices = np.exp(pi_star[:, np.newaxis]) * pnom_sim
    profits = np.sum(poisson * (prices - cost_sim[:, np.newaxis]), axis=0)
    return np.mean(profits)


def cross_validate(get_Etilde, get_performance, D, Pitilde, rank, lam, K=5):
    """K-fold cross-validation for elasticity estimation.

    Folds are contiguous blocks of time periods (no random shuffling).
    For each fold k, trains on all periods outside fold k and evaluates
    `get_performance` on the held-out fold.

    Parameters
    ----------
    get_Etilde      : callable(D_train, Pitilde_train, rank, lam) → (Etilde, ...)
    get_performance : callable(Etilde, D_test, Pitilde_test) → scalar
    D               : (n, N) full demand matrix
    Pitilde         : (n+1, N) full augmented price matrix
    rank            : rank passed to get_Etilde
    lam             : regularisation weight passed to get_Etilde
    K               : number of folds

    Returns
    -------
    perf : (K,) array of per-fold performance values
    """
    N = D.shape[1]
    fs = N // K  # fold size (last few periods may be dropped)
    perf = np.zeros(K)
    for k in range(K):
        # Test fold: time indices [k*fs, (k+1)*fs)
        Dk = D[:, k*fs:(k+1)*fs]
        Pitildek = Pitilde[:, k*fs:(k+1)*fs]
        # Training set: all other folds concatenated
        Dk_ = np.hstack((D[:, :k*fs], D[:, (k+1)*fs:]))
        Pitildek_ = np.hstack((Pitilde[:, :k*fs], Pitilde[:, (k+1)*fs:]))
        Etilde, _, _, _ = get_Etilde(Dk_, Pitildek_, rank=rank, lam=lam)
        perf[k] = get_performance(Etilde, Dk, Pitildek)
    return perf
