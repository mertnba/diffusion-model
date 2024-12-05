import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


def ball_stick_model(x, bvals, qhat):
    """
    Compute the Ball and Stick model signal.

    Parameters:
    - x: np.ndarray, Model parameters [S0, diff, f, theta, phi].
    - bvals: np.ndarray, B-values for the diffusion protocol.
    - qhat: np.ndarray, Gradient directions.

    Returns:
    - S: np.ndarray, Modeled signal intensities.
    """
    S0, diff, f, theta, phi = x

    # Fiber direction
    fibdir = np.array([
        np.cos(phi) * np.sin(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(theta),
    ])

    # Compute the dot product of qhat with the fiber direction
    fibdotgrad = np.sum(qhat * np.tile(fibdir, (len(bvals), 1)), axis=1)

    # Signal model
    S = S0 * (f * np.exp(-bvals * diff * (fibdotgrad**2)) + (1 - f) * np.exp(-bvals * diff))
    return S


def ball_stick_ssd(x, voxel, bvals, qhat):
    """
    Compute the sum of squared differences (SSD) between the model and observed signals.

    Parameters:
    - x: np.ndarray, Model parameters [S0, diff, f, theta, phi].
    - voxel: np.ndarray, Observed voxel signal.
    - bvals: np.ndarray, B-values for the diffusion protocol.
    - qhat: np.ndarray, Gradient directions.

    Returns:
    - ssd: float, Sum of squared differences between the model and observed data.
    """
    S = ball_stick_model(x, bvals, qhat)
    return np.sum((voxel - S) ** 2)


def fit_ball_stick(voxel, bvals, qhat, start_params):
    """
    Fit the Ball and Stick model to observed voxel signals.

    Parameters:
    - voxel: np.ndarray, Observed voxel signal.
    - bvals: np.ndarray, B-values for the diffusion protocol.
    - qhat: np.ndarray, Gradient directions.
    - start_params: np.ndarray, Initial guess for the parameters [S0, diff, f, theta, phi].

    Returns:
    - result: scipy.optimize.OptimizeResult, Result of the optimization.
    """
    return minimize(
        fun=ball_stick_ssd,
        x0=start_params,
        args=(voxel, bvals, qhat)
    )


def transform_parameters(params):
    """
    Transform parameters for physical realism.

    Parameters:
    - params: np.ndarray, Original parameters [S0, diff, f, theta, phi].

    Returns:
    - transformed_params: np.ndarray, Transformed parameters.
    """
    S0 = params[0] ** 2
    diff = params[1] ** 2
    f = expit(params[2])  # Sigmoid for range [0, 1]
    theta = params[3]
    phi = params[4]
    return np.array([S0, diff, f, theta, phi])


def inverse_transform_parameters(params):
    """
    Inverse transform parameters from constrained space.

    Parameters:
    - params: np.ndarray, Transformed parameters [S0, diff, f, theta, phi].

    Returns:
    - original_params: np.ndarray, Original parameters.
    """
    S0 = np.sqrt(params[0])
    diff = np.sqrt(params[1])
    f = logit(params[2])  # Inverse sigmoid
    theta = params[3]
    phi = params[4]
    return np.array([S0, diff, f, theta, phi])


def constrained_ssd(x, voxel, bvals, qhat):
    """
    Compute SSD with parameter constraints applied.

    Parameters:
    - x: np.ndarray, Constrained parameters.
    - voxel: np.ndarray, Observed voxel signal.
    - bvals: np.ndarray, B-values for the diffusion protocol.
    - qhat: np.ndarray, Gradient directions.

    Returns:
    - ssd: float, Sum of squared differences with constraints applied.
    """
    transformed_params = transform_parameters(x)
    return ball_stick_ssd(transformed_params, voxel, bvals, qhat)


def fit_constrained_ball_stick(voxel, bvals, qhat, start_params):
    """
    Fit the Ball and Stick model with parameter constraints.

    Parameters:
    - voxel: np.ndarray, Observed voxel signal.
    - bvals: np.ndarray, B-values for the diffusion protocol.
    - qhat: np.ndarray, Gradient directions.
    - start_params: np.ndarray, Initial guess for the parameters.

    Returns:
    - result: scipy.optimize.OptimizeResult, Result of the optimization.
    """
    return minimize(
        fun=constrained_ssd,
        x0=inverse_transform_parameters(start_params),
        args=(voxel, bvals, qhat)
    )
