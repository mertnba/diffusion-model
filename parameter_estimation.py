import numpy as np
from scipy.optimize import minimize
from transformations import (
    ball_stick_transform, ball_stick_transform_inv,
    zeppelin_stick_transform, zeppelin_stick_transform_inv,
    zeppelin_stick_tortuosity_transform, zeppelin_stick_tortuosity_transform_inv,
    dt_transform, dt_transform_inv
)
from models import ball_stick, zeppelin_stick_model, zeppelin_stick_and_tortuosity_model, DT_model


def estimate_parameters(voxel, bvals, qhat, startx, model_SSD, transform, inverse_transform, max_iter=100):
    """
    Estimate model parameters for a single voxel using optimization.

    Parameters:
    - voxel: np.ndarray, Observed voxel data.
    - bvals: np.ndarray, b-values for diffusion MRI.
    - qhat: np.ndarray, Gradient directions.
    - startx: np.ndarray, Initial parameter guesses.
    - model_SSD: function, Objective function to minimize.
    - transform: function, Transformation function for constraints.
    - inverse_transform: function, Inverse transformation function.
    - max_iter: int, Maximum number of iterations.

    Returns:
    - optimized_params: np.ndarray, Optimized parameters.
    - SSD: float, Sum of squared differences for the optimized parameters.
    """
    results = minimize(
        fun=model_SSD,
        x0=inverse_transform(startx),
        args=(voxel, bvals, qhat),
        method='L-BFGS-B',
        options={'maxiter': max_iter}
    )

    optimized_params = transform(results['x'])
    SSD = results['fun']
    return optimized_params, SSD


def bootstrap_estimation(voxel, bvals, qhat, startx, model_SSD, transform, inverse_transform, T=100):
    """
    Perform classical bootstrap parameter estimation.

    Parameters:
    - voxel: np.ndarray, Observed voxel data.
    - bvals: np.ndarray, b-values for diffusion MRI.
    - qhat: np.ndarray, Gradient directions.
    - startx: np.ndarray, Initial parameter guesses.
    - model_SSD: function, Objective function to minimize.
    - transform: function, Transformation function for constraints.
    - inverse_transform: function, Inverse transformation function.
    - T: int, Number of bootstrap samples.

    Returns:
    - bootstrap_results: dict, Contains bootstrap estimates for each parameter.
    """
    bootstrap_params = np.zeros((T, len(startx)))

    for t in range(T):
        random_indices = np.random.randint(0, len(bvals), len(bvals))
        A_t = voxel[random_indices]
        bvals_t = bvals[random_indices]
        qhat_t = qhat[random_indices]

        params, _ = estimate_parameters(
            A_t, bvals_t, qhat_t, startx, model_SSD, transform, inverse_transform
        )
        bootstrap_params[t] = params

    bootstrap_results = {
        'mean': np.mean(bootstrap_params, axis=0),
        'std': np.std(bootstrap_params, axis=0),
        'conf_intervals': np.percentile(bootstrap_params, [2.5, 97.5], axis=0)
    }

    return bootstrap_results


def mcmc_sampling(voxel, bvals, qhat, startx, model_SSD, transform, inverse_transform,
                  burn_in=1000, interval=10, sample_length=1000, param_std=None, noise_std=0.04):
    """
    Perform MCMC sampling for parameter estimation.

    Parameters:
    - voxel: np.ndarray, Observed voxel data.
    - bvals: np.ndarray, b-values for diffusion MRI.
    - qhat: np.ndarray, Gradient directions.
    - startx: np.ndarray, Initial parameter guesses.
    - model_SSD: function, Objective function to minimize.
    - transform: function, Transformation function for constraints.
    - inverse_transform: function, Inverse transformation function.
    - burn_in: int, Burn-in period for MCMC sampling.
    - interval: int, Sampling interval.
    - sample_length: int, Number of samples to retain after burn-in.
    - param_std: np.ndarray, Standard deviation for parameter sampling.
    - noise_std: float, Noise standard deviation.

    Returns:
    - samples: np.ndarray, Parameter samples from the MCMC chain.
    - acceptance_rate: float, Acceptance rate of the MCMC chain.
    """
    num_params = len(startx)
    raw_sequence_length = burn_in + interval * sample_length
    raw_sequence = np.zeros((raw_sequence_length, num_params))
    accepted = np.zeros(raw_sequence_length)

    if param_std is None:
        param_std = startx / 5

    raw_sequence[0] = startx

    for t in range(1, raw_sequence_length):
        x_t = raw_sequence[t - 1]
        noise = np.random.normal(0, param_std, size=num_params)
        x_c = transform(inverse_transform(x_t + noise))

        SSD_t = model_SSD(x_t, voxel, bvals, qhat)
        SSD_c = model_SSD(x_c, voxel, bvals, qhat)

        alpha = np.exp((SSD_t - SSD_c) / (2 * noise_std**2))
        if alpha > np.random.rand():
            raw_sequence[t] = x_c
            accepted[t] = 1
        else:
            raw_sequence[t] = x_t

    acceptance_rate = np.sum(accepted[burn_in:]) / (raw_sequence_length - burn_in)
    samples = raw_sequence[burn_in::interval]

    return samples, acceptance_rate
