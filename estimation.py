"""Elasticity matrix estimation via gradient ascent (GA), alternating maximization (AM),
and nonlinear programming (NLP). For more details, please see the paper and README.
"""

import cvxpy as cp
import numpy as np
import numpy.linalg as la


# Helper functions

def f_np(Etilde, D, Pitilde):
    """Scaled Poisson log-likelihood: sum_i sum_t [ D*Y - exp(Y) ] / N
    where Y = Etilde @ Pitilde.  (The constant -log(D!) term is omitted.)"""
    Y = Etilde @ Pitilde
    return np.sum(D * Y - np.exp(Y)) / D.shape[1]


def construct_Etilde(B, C, s, logdnom):
    """Build the augmented elasticity matrix [B@C.T + diag(s) | logdnom]."""
    return np.hstack((B @ C.T + np.diag(s), logdnom.reshape(-1, 1)))


def construct_X(B, C, s, logdnom):
    """Pack (B, C, s, logdnom) into a single parameter matrix X = [B|C|s|logdnom]."""
    return np.hstack((B, C, s.reshape(-1, 1), logdnom.reshape(-1, 1)))


def slice_X(X):
    """Unpack parameter matrix X = [B|C|s|logdnom] → (B, C, s, logdnom)."""
    r = (X.shape[1] - 2) // 2
    B = X[:, :r]
    C = X[:, r:2*r]
    s = X[:, -2]
    logdnom = X[:, -1]
    return B, C, s, logdnom


def X_to_Etilde(X):
    """Convert packed parameter matrix X directly to the augmented elasticity matrix."""
    return construct_Etilde(*slice_X(X))


# Gradient Ascent

def get_Etilde_ga(D, Pitilde, rank=10, lam=0.1, use_scale=False):
    """Estimate the augmented elasticity matrix via gradient ascent.

    Maximises the regularised Poisson log-likelihood:

        f(X) = sum_i sum_t [D*Y - exp(Y)] / N  -  (lam/2) * (||B||^2 + ||C||^2)

    where Y = Etilde @ Pitilde and Etilde is constructed from (B, C, s, logdnom).

    The step size is adapted with a simple step size rule:
      - accepted step → multiply alpha by 1.2
      - rejected step → divide alpha by 1.5

    Parameters
    ----------
    D          : (n, N) demand matrix
    Pitilde    : (n+1, N) augmented log-price-deviation matrix
    rank       : rank r of the low-rank components B and C
    lam        : L2 regularisation weight on B and C
    use_scale  : if True, normalise the gradient norm by the initial objective
                 value (helps convergence when the objective magnitude varies)

    Returns
    -------
    Etilde : (n, n+1) estimated augmented elasticity matrix
    Elow   : (n, n)   low-rank component  B @ C.T
    Ediag  : (n, n)   diagonal component  diag(s)
    Estar  : (n, n)   full elasticity matrix  Elow + Ediag
    """
    n, N = D.shape

    # Precompute D @ Pitilde.T once (used in every gradient evaluation)
    DPitildeT = D @ Pitilde.T

    def g(X):
        B, C, s, logdnom = slice_X(X)
        Etilde = construct_Etilde(B, C, s, logdnom)
        return (f_np(Etilde, D, Pitilde) - (lam / 2) * (np.sum(B ** 2) + np.sum(C ** 2)))

    def grad(X):
        B, C, s, logdnom = slice_X(X)
        Etilde = construct_Etilde(B, C, s, logdnom)
        grad_Etilde = (DPitildeT - (np.exp(Etilde @ Pitilde) @ Pitilde.T)) / N
        grad_E = grad_Etilde[:, :-1]
        grad_logdnom = grad_Etilde[:, -1]
        grad_B = grad_E @ C - lam * B
        grad_C = grad_E.T @ B - lam * C
        grad_s = np.diag(grad_E)
        return construct_X(grad_B, grad_C, grad_s, grad_logdnom)

    # Initialise B, C with small random values; s = -1 (moderate own-price elasticity)
    np.random.seed(0)
    B = np.sqrt(1 / n / rank ** 0.5) * np.random.randn(n, rank)
    C = np.sqrt(1 / n / rank ** 0.5) * np.random.randn(n, rank)
    s = -np.ones(n)
    logdnom = np.zeros(n)
    X = construct_X(B, C, s, logdnom)
    grad_X = grad(X)
    scale = g(X) * 10 if use_scale else 1.0

    # Convergence tolerances and initial step size
    alpha, eps_rel, eps_abs = 1.0, 1e-4, 1e-4

    g_values = [g(X)]

    # Gradient ascent with backtracking line search
    while (la.norm(grad_X) / scale > eps_rel * la.norm(X) + eps_abs):
        while True:
            X_hat = X + alpha * grad_X
            if g(X_hat) > g_values[-1]:  # sufficient-increase condition
                X = X_hat
                grad_X = grad(X)
                g_values.append(g(X))
                alpha *= 1.2  # increase step size after accepted step
                break
            alpha /= 1.5  # shrink step size after rejected step

    B, C, s, logdnom = slice_X(X)
    Elow  = B @ C.T        # low-rank component
    Ediag = np.diag(s)     # diagonal component
    Estar = Elow + Ediag   # full elasticity matrix
    return np.hstack((Estar, logdnom.reshape(-1, 1))), Elow, Ediag, Estar
