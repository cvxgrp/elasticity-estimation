import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from estimation import get_Etilde_ga
from evaluation import ll, error, pred_poisson_noise, cross_validate


def configure_plt():
    """Configure matplotlib for plots with LaTeX rendering."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })


def generate_data(n=100, N=200, rank=10):
    """Generate synthetic demand data from a known low-rank elasticity model.

    Prices are drawn uniformly; the true elasticity matrix has rank `rank` plus
    a diagonal component.  Demand is Poisson with mean exp(E @ Pi) * d_nom.

    Parameters
    ----------
    n    : number of products
    N    : number of time periods
    rank : rank of the low-rank component of the true elasticity matrix

    Returns
    -------
    D         : (n, N) integer demand matrix (Poisson samples)
    Pitilde   : (n+1, N) augmented log-price-deviation matrix [Pi; 1]
    Esim      : (n, n) true elasticity matrix
    pnom      : (n,) nominal prices
    cost      : (n,) unit costs
    rnom      : (n,) nominal revenues  (pnom * d_nom)
    kappanom  : (n,) nominal costs     (cost * d_nom)
    """
    np.random.seed(0)
    P = 1 + np.random.rand(n, N)
    
    pnom = np.exp(np.mean(np.log(P), axis=1, keepdims=True))
    Pi = np.log(P / pnom)
    Pitilde = np.vstack((Pi, np.ones((1, Pi.shape[1]))))

    dnom = np.ones((n, 1))
    cost = np.random.uniform(0.8, 1.2, size=(n, 1))

    Bsim = np.sqrt(0.1) * np.random.randn(n, rank)
    Csim = np.sqrt(0.1) * np.random.randn(n, rank)
    ssim = np.random.uniform(-5.0, -1.0, size=n)
    Esim = Bsim @ Csim.T + np.diag(ssim)

    y = np.exp(Esim @ Pi) * dnom

    rng = np.random.default_rng(seed=9)
    D = rng.poisson(lam=y)
    
    return D, Pitilde, Esim, pnom, cost.flatten()


if __name__ == '__main__':
    
    D, Pitilde, Esim, _, _ = generate_data()
    
    rank_ = np.array([6, 8, 10, 12, 14])
    lam_ = np.logspace(-3, 0, 4)

    ll_cv = np.zeros((len(rank_), len(lam_)))

    for i, rank in enumerate(rank_):
        for j, lam in enumerate(lam_):
            ll_cv[i, j] = np.mean(cross_validate(get_Etilde_ga, ll, D, Pitilde, rank, lam))
            
    # index of global maximum
    i_star, j_star = np.unravel_index(np.argmax(ll_cv), ll_cv.shape)

    print(f"Best rank index: {i_star}, rank = {rank_[i_star]}")
    print(f"Best lambda index: {j_star}, lambda = {lam_[j_star]}")
    print(f"Max ll_cv value: {ll_cv[i_star, j_star]}")
    
    _, _, _, Estar, _, _ = get_Etilde_ga(D, Pitilde, rank=rank_[i_star], lam=lam_[j_star])
    
    configure_plt()
    
    plt.figure(figsize=(5, 3))
    x = rank_[:]
    y = ll_cv[:, j_star]
    plt.plot(x, y, marker='o', color='blue')
    plt.xticks(x,  [str(r) for r in x])
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
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # Define saturated colors
    colors = [
        (0.0, "#b30000"),   # hard red
        (0.5, "#ffffff"),   # pure white midpoint
        (1.0, "#0033cc")    # hard blue
    ]

    custom_rdbu = mcolors.LinearSegmentedColormap.from_list(
        "hard_RdBu",
        colors,
        N=256
    )

    vmax = np.abs(Estar).max()
    vmin = -vmax

    im0 = axes[0].imshow(Estar, cmap=custom_rdbu, interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[0].set_title(r'$E^\star$')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(Esim, cmap=custom_rdbu, interpolation='nearest', vmin=-np.abs(Esim).max(), vmax=np.abs(Esim).max())
    axes[1].set_title(r'$E^\mathrm{syn}$')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()
