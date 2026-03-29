"""Figure 7: Log-likelihood cross-validation on real DFF retail data.

Loads preprocessed demand and price data from demand.csv / prices.csv,
grid-searches over rank r and regularization weight lambda using 5-fold CV,
and produces two plots: log-likelihood vs r and log-likelihood vs lambda.

To generate demand.csv and prices.csv from raw DFF scanner data, run dff_data.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from figure4_5 import configure_plt
from estimation import get_Etilde_ga
from evaluation import ll, error, pred_poisson_noise, cross_validate


def load_data(demand_file='demand.csv', prices_file='prices.csv'):
    """Load demand and price data from CSV files and construct the augmented price matrix.

    The rows (weeks) are truncated to the nearest multiple of 10 before transposing
    so that CV splits evenly.

    Parameters
    ----------
    demand_file : path to CSV with shape (weeks, products), integer demand counts
    prices_file : path to CSV with shape (weeks, products), positive prices

    Returns
    -------
    D       : (n, N) demand matrix
    Pitilde : (n+1, N) augmented log-price change matrix
    n       : number of products
    N       : number of time periods
    """
    D = np.loadtxt(demand_file, delimiter=',')
    P = np.loadtxt(prices_file, delimiter=',')

    # Truncate to nearest multiple of 10 and transpose to (products, weeks)
    D = D[:-(D.shape[0] % 10)].T
    P = P[:-(P.shape[0] % 10)].T

    n, N = D.shape

    pnom = np.exp(np.mean(np.log(P), axis=1, keepdims=True))
    Pi = np.log(P / pnom)
    Pitilde = np.vstack((Pi, np.ones((1, Pi.shape[1]))))

    return D, Pitilde, n, N


if __name__ == '__main__':

    D, Pitilde, n, N = load_data()

    rank_ = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18])
    lam_ = np.logspace(-3, 1, 5)

    ll_cv = np.zeros((len(rank_), len(lam_)))

    # use_scale=True normalises the convergence criterion by the initial objective
    # value, which helps when the real-data objective has a different magnitude
    get_Etilde = lambda Dk_, Pitildek_, rank, lam: get_Etilde_ga(Dk_, Pitildek_, rank=rank, lam=lam, use_scale=True)

    for i, rank in enumerate(rank_):
        for j, lam in enumerate(lam_):
            ll_cv[i, j] = np.mean(cross_validate(get_Etilde, ll, D, Pitilde, rank, lam))

    # Index of global maximum
    i_star, j_star = np.unravel_index(np.argmax(ll_cv), ll_cv.shape)

    print(f"Best rank index: {i_star}, rank = {rank_[i_star]}")
    print(f"Best lambda index: {j_star}, lambda = {lam_[j_star]}")
    print(f"Max ll_cv value: {ll_cv[i_star, j_star]}")
    
    err = np.mean(cross_validate(get_Etilde_ga, error, D, Pitilde, rank_[i_star], lam_[j_star]))
    noise = np.mean(cross_validate(get_Etilde_ga, pred_poisson_noise, D, Pitilde, rank_[i_star], lam_[j_star]))
    
    print(f"Median absolute percentage error: {err:.4f}")
    print(f"Median absolute percentage error due to Poisson noise: {noise:.4f}")

    configure_plt()

    plt.figure(figsize=(5, 3))
    x = rank_[:]
    y = ll_cv[:, j_star]
    plt.plot(x, y, marker='o', color='blue')
    plt.xticks(x, [str(r) for r in x])
    plt.xlabel(r"$r$")
    plt.ylabel("Log-likelihood")
    plt.title(r"$\lambda$" + f" = {lam_[j_star]}")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(5, 3))
    plt.semilogx(lam_, ll_cv[i_star, :], marker='o', color='blue')
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Log-likelihood")
    plt.title(r"$r$" + f" = {rank_[i_star]}")
    plt.grid(True)
    plt.show()
