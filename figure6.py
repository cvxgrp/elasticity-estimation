"""Figure 6: Pricing-performance cross-validation on synthetic data.

Grid-searches over rank r and regularisation weight lambda using CV.
The cross-validation metric is the true profit achieved by applying the
estimated-optimal prices (pricing_performance), evaluated on the held-out fold.
Produces two plots: profit vs r (at best lambda) and profit vs lambda (at best r).
"""

import numpy as np
import matplotlib.pyplot as plt
from figure4_5 import configure_plt, generate_data
from estimation import get_Etilde_ga
from evaluation import pricing_performance, cross_validate


if __name__ == '__main__':

    D, Pitilde, Esim, pnom, cost = generate_data()

    rank_ = np.array([6, 8, 10, 12, 14])
    lam_ = np.logspace(-3, 0, 4)

    profit_cv = np.zeros((len(rank_), len(lam_)))

    def get_performance(Etilde, Dk, Pitildek):
        return pricing_performance(Etilde, Dk, Pitildek, pnom_sim=pnom, cost_sim=cost, Esim=Esim)

    for i, rank in enumerate(rank_):
        for j, lam in enumerate(lam_):
            profit_cv[i, j] = np.mean(cross_validate(get_Etilde_ga, get_performance, D, Pitilde, rank, lam))

    # Index of global maximum
    i_star, j_star = np.unravel_index(np.argmax(profit_cv), profit_cv.shape)

    print(f"Best rank index: {i_star}, rank = {rank_[i_star]}")
    print(f"Best lambda index: {j_star}, lambda = {lam_[j_star]}")
    print(f"Max profit value: {profit_cv[i_star, j_star]}")

    configure_plt()

    plt.figure(figsize=(5, 3))
    x = rank_[:]
    y = profit_cv[:, j_star]
    plt.plot(x, y, marker='o', color='blue')
    plt.xticks(x, [str(r) for r in x])
    plt.xlabel(r"$r$")
    plt.ylabel("Profit")
    plt.title(r"$\lambda$" + f" = {lam_[j_star]}")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(5, 3))
    plt.semilogx(lam_, profit_cv[i_star, :], marker='o', color='blue')
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Profit")
    plt.title(r"$r$" + f" = {rank_[i_star]}")
    plt.grid(True)
    plt.show()
