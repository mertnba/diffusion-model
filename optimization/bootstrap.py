import numpy as np
from parameter_estimation import multiple_startx_per_voxel
from transformations import transform


def classical_bootstrap(vox_i, vox_j, im_slice, dwis, bvals, qhat, T=100, N=100, startx=None):
    """
    Perform classical bootstrap to estimate parameters.

    Parameters:
    - vox_i, vox_j: int, Indices of the voxel.
    - im_slice: int, Index of the image slice.
    - dwis: np.ndarray, Diffusion-weighted imaging data.
    - bvals: np.ndarray, b-values.
    - qhat: np.ndarray, Gradient directions.
    - T: int, Number of bootstrap samples.
    - N: int, Number of iterations for parameter optimization.
    - startx: np.ndarray, Initial parameter guess.

    Returns:
    - boostrap_s0: np.ndarray, Bootstrap samples for S0.
    - boostrap_diff: np.ndarray, Bootstrap samples for diffusivity.
    - boostrap_f: np.ndarray, Bootstrap samples for the fraction.
    """
    boostrap_s0 = np.zeros(T)
    boostrap_diff = np.zeros(T)
    boostrap_f = np.zeros(T)

    for t in range(T):
        random_indexes = np.random.randint(0, dwis.shape[0], size=dwis.shape[0])
        A_t = dwis[random_indexes, vox_i, vox_j, im_slice]
        bvals_t = bvals[random_indexes]
        qhat_t = qhat[random_indexes]

        x_per_voxel, x_SSD = multiple_startx_per_voxel(N, startx, A_t, bvals_t, qhat_t)
        min_idx = np.argmin(x_SSD)
        optimal_x = transform(x_per_voxel[min_idx])

        boostrap_s0[t] = optimal_x[0]
        boostrap_diff[t] = optimal_x[1]
        boostrap_f[t] = optimal_x[2]

    return boostrap_s0, boostrap_diff, boostrap_f


def parametric_bootstrap(Avox, bvals, qhat, T=100, N=100, startx=None, model_SSD=None):
    """
    Perform parametric bootstrap to estimate parameters.

    Parameters:
    - Avox: np.ndarray, Voxel diffusion data.
    - bvals: np.ndarray, b-values.
    - qhat: np.ndarray, Gradient directions.
    - T: int, Number of bootstrap samples.
    - N: int, Number of iterations for parameter optimization.
    - startx: np.ndarray, Initial parameter guess.
    - model_SSD: function, Objective function for optimization.

    Returns:
    - boostrap_s0: np.ndarray, Bootstrap samples for S0.
    - boostrap_diff: np.ndarray, Bootstrap samples for diffusivity.
    - boostrap_f: np.ndarray, Bootstrap samples for the fraction.
    """
    boostrap_s0 = np.zeros(T)
    boostrap_diff = np.zeros(T)
    boostrap_f = np.zeros(T)

    x_per_voxel, x_SSD = multiple_startx_per_voxel(N, startx, Avox, bvals, qhat)
    min_idx = np.argmin(x_SSD)
    optimal_x = transform(x_per_voxel[min_idx])

    # Generate synthetic data
    S = model_SSD(optimal_x, bvals, qhat)
    sigma = np.sqrt(np.mean((Avox - S) ** 2))

    for t in range(T):
        noise = np.random.normal(0, sigma, len(Avox))
        synthetic_data = S + noise

        x_per_voxel, x_SSD = multiple_startx_per_voxel(N, startx, synthetic_data, bvals, qhat)
        min_idx = np.argmin(x_SSD)
        optimal_x = transform(x_per_voxel[min_idx])

        boostrap_s0[t] = optimal_x[0]
        boostrap_diff[t] = optimal_x[1]
        boostrap_f[t] = optimal_x[2]

    return boostrap_s0, boostrap_diff, boostrap_f
