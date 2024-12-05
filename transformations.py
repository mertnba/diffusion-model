import numpy as np
from scipy.special import expit, logit


# General transformation functions
def transform_general(x):
    """
    Apply constraints to parameters using general transformations.

    Parameters:
    - x: np.ndarray, Input parameters.

    Returns:
    - constrained_params: np.ndarray, Transformed parameters.
    """
    return np.array([param**2 if i < 3 else expit(param) if i == 3 else param for i, param in enumerate(x)])


def inverse_transform_general(x):
    """
    Reverse general parameter transformations.

    Parameters:
    - x: np.ndarray, Transformed parameters.

    Returns:
    - original_params: np.ndarray, Original parameters.
    """
    return np.array([np.sqrt(param) if i < 3 else logit(param) if i == 3 else param for i, param in enumerate(x)])


# Ball and Stick model transformations
def ball_stick_transform(x):
    """
    Apply constraints for Ball-Stick model parameters.

    Parameters:
    - x: np.ndarray, Input parameters.

    Returns:
    - constrained_params: np.ndarray, Transformed parameters.
    """
    S0 = x[0]**2
    diff = x[1]**2
    f = expit(x[2])
    theta = x[3]
    phi = x[4]
    return [S0, diff, f, theta, phi]


def ball_stick_transform_inv(x):
    """
    Reverse Ball-Stick parameter transformations.

    Parameters:
    - x: np.ndarray, Transformed parameters.

    Returns:
    - original_params: np.ndarray, Original parameters.
    """
    S0 = np.sqrt(x[0])
    diff = np.sqrt(x[1])
    f = logit(x[2])
    theta = x[3]
    phi = x[4]
    return [S0, diff, f, theta, phi]


# Zeppelin-Stick model transformations
def zeppelin_stick_transform(x):
    """
    Apply constraints for Zeppelin-Stick model parameters.

    Parameters:
    - x: np.ndarray, Input parameters.

    Returns:
    - constrained_params: np.ndarray, Transformed parameters.
    """
    S0 = x[0]**2
    eig_val_2 = x[1]**2
    eig_val_diff = x[2]**2
    f = expit(x[3])
    theta = x[4]
    phi = x[5]
    return [S0, eig_val_2, eig_val_diff, f, theta, phi]


def zeppelin_stick_transform_inv(x):
    """
    Reverse Zeppelin-Stick parameter transformations.

    Parameters:
    - x: np.ndarray, Transformed parameters.

    Returns:
    - original_params: np.ndarray, Original parameters.
    """
    S0 = np.sqrt(x[0])
    eig_val_2 = np.sqrt(x[1])
    eig_val_diff = np.sqrt(x[2])
    f = logit(x[3])
    theta = x[4]
    phi = x[5]
    return [S0, eig_val_2, eig_val_diff, f, theta, phi]


# Zeppelin-Stick-Tortuosity model transformations
def zeppelin_stick_tortuosity_transform(x):
    """
    Apply constraints for Zeppelin-Stick-Tortuosity model parameters.

    Parameters:
    - x: np.ndarray, Input parameters.

    Returns:
    - constrained_params: np.ndarray, Transformed parameters.
    """
    S0 = x[0]**2
    eig_val_2 = x[1]**2
    f = expit(x[2])
    theta = x[3]
    phi = x[4]
    return [S0, eig_val_2, f, theta, phi]


def zeppelin_stick_tortuosity_transform_inv(x):
    """
    Reverse Zeppelin-Stick-Tortuosity parameter transformations.

    Parameters:
    - x: np.ndarray, Transformed parameters.

    Returns:
    - original_params: np.ndarray, Original parameters.
    """
    S0 = np.sqrt(x[0])
    eig_val_2 = np.sqrt(x[1])
    f = logit(x[2])
    theta = x[3]
    phi = x[4]
    return [S0, eig_val_2, f, theta, phi]


# Diffusion Tensor model transformations
def dt_transform(x):
    """
    Apply constraints for Diffusion Tensor model parameters.

    Parameters:
    - x: np.ndarray, Input parameters.

    Returns:
    - constrained_params: np.ndarray, Transformed parameters.
    """
    S0 = x[0]**2
    Dxx = x[1]**2
    Dxy = x[2]**2
    Dxz = x[3]**2
    Dyy = x[4]**2
    Dyz = x[5]**2
    Dzz = x[6]**2
    return [S0, Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]


def dt_transform_inv(x):
    """
    Reverse Diffusion Tensor parameter transformations.

    Parameters:
    - x: np.ndarray, Transformed parameters.

    Returns:
    - original_params: np.ndarray, Original parameters.
    """
    S0 = np.sqrt(x[0])
    Dxx = np.sqrt(x[1])
    Dxy = np.sqrt(x[2])
    Dxz = np.sqrt(x[3])
    Dyy = np.sqrt(x[4])
    Dyz = np.sqrt(x[5])
    Dzz = np.sqrt(x[6])
    return [S0, Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
