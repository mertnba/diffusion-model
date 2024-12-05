import numpy as np
from transformations import (
    ball_stick_transform, ball_stick_transform_inv,
    zeppelin_stick_transform, zeppelin_stick_transform_inv,
    zeppelin_stick_tortuosity_transform, zeppelin_stick_tortuosity_transform_inv
)


def q_sample_from_distribution(x, std, transform, inverse_transform):
    """
    Add noise to the parameters, transform them, and ensure they remain in the valid domain.

    Parameters:
    - x: np.ndarray, Current parameter values.
    - std: np.ndarray, Standard deviation for sampling noise.
    - transform: function, Transformation function for constraints.
    - inverse_transform: function, Inverse transformation function.

    Returns:
    - np.ndarray, Transformed parameter values with added noise.
    """
    noise = np.random.normal(0, std, size=x.size)
    x_noisy = x + noise
    return transform(inverse_transform(x_noisy))


def alpha_likelihood(x_c, x_t, voxel, bvals, qhat, model_SSD, noise_std):
    """
    Calculate the likelihood ratio (acceptance probability) for MCMC sampling.

    Parameters:
    - x_c: np.ndarray, Candidate parameter values.
    - x_t: np.ndarray, Current parameter values.
    - voxel: np.ndarray, Observed voxel data.
    - bvals: np.ndarray, b-values for diffusion MRI.
    - qhat: np.ndarray, Gradient directions.
    - model_SSD: function, Objective function to minimize.
    - noise_std: float, Noise standard deviation.

    Returns:
    - float, Likelihood ratio for accepting the candidate parameters.
    """
    SSD_c = model_SSD(x_c, voxel, bvals, qhat)
    SSD_t = model_SSD(x_t, voxel, bvals, qhat)
    prior_ratio = np.sin(x_t[3]) / np.sin(x_c[3])  # Prior adjustment for spherical coordinates
    return np.exp((SSD_t - SSD_c) / (2 * noise_std**2)) * prior_ratio


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
        param_std = startx / 5  # Default parameter standard deviation

    raw_sequence[0] = startx

    for t in range(1, raw_sequence_length):
        x_t = raw_sequence[t - 1]
        x_c = q_sample_from_distribution(x_t, param_std, transform, inverse_transform)

        alpha = alpha_likelihood(x_c, x_t, voxel, bvals, qhat, model_SSD, noise_std)
        if alpha > np.random.rand():
            raw_sequence[t] = x_c
            accepted[t] = 1
        else:
            raw_sequence[t] = x_t

    # Acceptance rate and final sampled sequence
    acceptance_rate = np.sum(accepted[burn_in:]) / (raw_sequence_length - burn_in)
    samples = raw_sequence[burn_in::interval]

    return samples, acceptance_rate
