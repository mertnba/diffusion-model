import numpy as np
from scipy.optimize import minimize
from transformations import transform, inverse_transform


def multiple_startx_per_voxel(max_iter, startx, avox, bvals, qhat, model_SSD):
    """
    Run multiple optimizations starting from noisy initial guesses.

    Parameters:
    - max_iter: int, Number of optimization iterations.
    - startx: np.ndarray, Initial parameter guess.
    - avox: np.ndarray, Diffusion data for a single voxel.
    - bvals: np.ndarray, b-values.
    - qhat: np.ndarray, Gradient directions.
    - model_SSD: function, Objective function for optimization.

    Returns:
    - X_optimized_params: np.ndarray, Array of optimized parameter sets.
    - X_SSD: np.ndarray, Array of Sum of Squared Differences for each run.
    """
    num_parameters = startx.size
    X_optimized_params = np.zeros((max_iter, num_parameters))
    X_SSD = np.zeros(max_iter)

    noise_std = np.abs(startx / 5)
    noise_std[3:5][noise_std[3:5] == 0] = 0.2  # Handle theta/phi if initially zero

    for i in range(max_iter):
        noise = np.random.normal(0, noise_std)
        noisy_start = startx + noise

        result = minimize(
            fun=model_SSD,
            x0=inverse_transform(noisy_start),
            args=(avox, bvals, qhat)
        )
        X_optimized_params[i, :] = result.x
        X_SSD[i] = result.fun if not np.isnan(result.fun) else np.inf

    return X_optimized_params, X_SSD


def estimate_global_minima_probability(max_iter, startx, avox, bvals, qhat, model_SSD, eps=1e-1):
    """
    Estimate the probability of finding the global minimum.

    Parameters:
    - max_iter: int, Number of optimization iterations.
    - startx: np.ndarray, Initial parameter guess.
    - avox: np.ndarray, Diffusion data for a single voxel.
    - bvals: np.ndarray, b-values.
    - qhat: np.ndarray, Gradient directions.
    - model_SSD: function, Objective function for optimization.
    - eps: float, Tolerance for considering solutions equivalent.

    Returns:
    - min_SSD: float, Minimum SSD value found.
    - best_params: np.ndarray, Parameters corresponding to the minimum SSD.
    - prob_global_min: float, Estimated probability of finding the global minimum.
    """
    X_optimized_params, X_SSD = multiple_startx_per_voxel(max_iter, startx, avox, bvals, qhat, model_SSD)

    min_SSD = np.min(X_SSD)
    min_idx = np.argmin(X_SSD)
    best_params = X_optimized_params[min_idx]
    count_min = np.isclose(X_SSD, min_SSD, atol=eps).sum()

    prob_global_min = count_min / max_iter
    return min_SSD, transform(best_params), prob_global_min


def required_iterations_for_confidence(p, confidence=0.95):
    """
    Calculate the number of iterations needed to achieve a specified confidence level.

    Parameters:
    - p: float, Probability of finding the global minimum in one run.
    - confidence: float, Desired confidence level (default: 0.95).

    Returns:
    - int, Number of iterations required.
    """
    if p <= 0 or p >= 1:
        raise ValueError("Probability p must be between 0 and 1 (exclusive).")
    return int(np.ceil(np.log(1 - confidence) / np.log(1 - p)))
