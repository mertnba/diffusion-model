import numpy as np
from scipy.special import expit, logit


def apply_voxel_filter(dwis, slice_idx, threshold=0):
    """
    Apply a filter to select voxels with signal values above a threshold.

    Parameters:
    - dwis: np.ndarray, Diffusion-weighted signal data.
    - slice_idx: int, Index of the slice to process.
    - threshold: float, Minimum voxel signal threshold.

    Returns:
    - valid_voxels: np.ndarray, Filtered voxel indices.
    """
    Dx, Dy = dwis.shape[1:3]
    valid_voxels = []
    for i in range(Dx):
        for j in range(Dy):
            if np.min(dwis[:, i, j, slice_idx]) > threshold:
                valid_voxels.append((i, j))
    return valid_voxels


def standardize_features(data, mean=None, std=None):
    """
    Standardize data to have zero mean and unit variance.

    Parameters:
    - data: np.ndarray, Input data to standardize.
    - mean: float or None, Precomputed mean (optional).
    - std: float or None, Precomputed standard deviation (optional).

    Returns:
    - standardized_data: np.ndarray, Standardized data.
    - mean: float, Mean used for standardization.
    - std: float, Standard deviation used for standardization.
    """
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data, mean, std


def normalize_voxels(dwis, scale_factor=1.0):
    """
    Normalize voxel signal values.

    Parameters:
    - dwis: np.ndarray, Diffusion-weighted signal data.
    - scale_factor: float, Scaling factor for normalization.

    Returns:
    - normalized_dwis: np.ndarray, Normalized diffusion data.
    """
    return dwis / scale_factor


def transform_parameters(params):
    """
    Apply physical constraints to model parameters.

    Parameters:
    - params: np.ndarray, Model parameters.

    Returns:
    - transformed_params: np.ndarray, Transformed parameters with constraints.
    """
    S0 = params[0] ** 2
    diff = params[1] ** 2
    f = expit(params[2])  # Sigmoid for range [0, 1]
    theta = params[3]
    phi = params[4]
    return np.array([S0, diff, f, theta, phi])


def inverse_transform_parameters(params):
    """
    Apply the inverse transformation to constrained parameters.

    Parameters:
    - params: np.ndarray, Constrained parameters.

    Returns:
    - original_params: np.ndarray, Original unconstrained parameters.
    """
    S0 = np.sqrt(params[0])
    diff = np.sqrt(params[1])
    f = logit(params[2])  # Inverse sigmoid
    theta = params[3]
    phi = params[4]
    return np.array([S0, diff, f, theta, phi])


def compute_diffusion_tensor(qhat, bvals):
    """
    Construct the G matrix for diffusion tensor computation.

    Parameters:
    - qhat: np.ndarray, Gradient directions.
    - bvals: np.ndarray, Normalized b-values.

    Returns:
    - G: np.ndarray, Matrix for diffusion tensor fitting.
    """
    G = np.array([
        np.ones(len(bvals)),
        -bvals * qhat[:, 0] ** 2,  # Dxx
        -2 * bvals * qhat[:, 0] * qhat[:, 1],  # Dxy
        -2 * bvals * qhat[:, 0] * qhat[:, 2],  # Dxz
        -bvals * qhat[:, 1] ** 2,  # Dyy
        -2 * bvals * qhat[:, 1] * qhat[:, 2],  # Dyz
        -bvals * qhat[:, 2] ** 2  # Dzz
    ]).T
    return G


def log_transform_signals(dwis):
    """
    Apply logarithmic transformation to diffusion-weighted signals.

    Parameters:
    - dwis: np.ndarray, Diffusion-weighted signal data.

    Returns:
    - log_dwis: np.ndarray, Log-transformed signals.
    """
    return np.log(np.clip(dwis, a_min=1e-10, a_max=None))


def clip_negative_values(data, min_value=0):
    """
    Set negative values in the data to a specified minimum.

    Parameters:
    - data: np.ndarray, Input data.
    - min_value: float, Minimum allowed value.

    Returns:
    - clipped_data: np.ndarray, Data with negative values clipped.
    """
    data[data < min_value] = min_value
    return data
