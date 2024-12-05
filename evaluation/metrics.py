import numpy as np


def log_likelihood(A_est, A_exact, noise_std):
    """
    Compute the log-likelihood of observed data given the model predictions.

    Parameters:
    - A_est: np.ndarray, Model-predicted signal values.
    - A_exact: np.ndarray, Observed signal values.
    - noise_std: float, Standard deviation of the noise.

    Returns:
    - float, Log-likelihood value.
    """
    var = noise_std ** 2
    SSD = np.sum((A_est - A_exact) ** 2)
    log_likelihood_value = -0.5 * np.log(2 * np.pi * var) - (SSD / (2 * var))
    return log_likelihood_value


def AIC(A_est, A_exact, deg_freedom, noise_std):
    """
    Compute the Akaike Information Criterion (AIC).

    Parameters:
    - A_est: np.ndarray, Model-predicted signal values.
    - A_exact: np.ndarray, Observed signal values.
    - deg_freedom: int, Number of model parameters.
    - noise_std: float, Standard deviation of the noise.

    Returns:
    - float, AIC value.
    """
    K = len(A_exact)
    if K / deg_freedom < 40:
        print("WARNING: K/N < 40. Adjusted AIC may be needed.")
    logL = log_likelihood(A_est, A_exact, noise_std)
    aic = 2 * deg_freedom - 2 * logL
    return aic


def BIC(A_est, A_exact, deg_freedom, noise_std):
    """
    Compute the Bayesian Information Criterion (BIC).

    Parameters:
    - A_est: np.ndarray, Model-predicted signal values.
    - A_exact: np.ndarray, Observed signal values.
    - deg_freedom: int, Number of model parameters.
    - noise_std: float, Standard deviation of the noise.

    Returns:
    - float, BIC value.
    """
    K = len(A_exact)
    logL = log_likelihood(A_est, A_exact, noise_std)
    bic = deg_freedom * np.log(K) - 2 * logL
    return bic
