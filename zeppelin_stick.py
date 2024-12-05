import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


def zeppelin_stick_model(x, bvals, qhat):
    """
    Compute the signal using the Zeppelin-Stick model.

    Parameters:
    - x: np.ndarray, Model parameters [S0, eig_val_2, eig_val_diff, f, theta, phi].
    - bvals: np.ndarray, B-values for the diffusion protocol.
    - qhat: np.ndarray, Gradient directions.

    Returns:
    - S: np.ndarray, Modeled signal intensities.
    """
    S0, eig_val_2, eig_val_diff, f, theta, phi = x
    eig_val_1 = eig_val_2 + eig_val_diff  # Largest eigenvalue

    # Fiber direction
    fibdir = np.array([
        np.cos(phi) * np.sin(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(theta),
    ])

    fibdotgrad = np.sum(qhat * np.tile(fibdir, (len(bvals), 1)), axis=1)

    # Intra-cellular signal
    S_i = np.exp(-bvals * eig_val_1 * (fibdotgrad**2))

    # Extra-cellular signal
    S_e = np.exp(-bvals * (eig_val_2 + (eig_val_1 - eig_val_2) * (fibdotgrad**2)))

    # Combined signal
    S = S0 * (f * S_i + (1 - f) * S_e)
    return S


def zeppelin_stick_tortuosity_model(x, bvals, qhat):
    """
    Compute the signal using the Zeppelin-Stick-Tortuosity model.

    Parameters:
    - x: np.ndarray, Model parameters [S0, eig_val_2, f, theta, phi].
    - bvals: np.ndarray, B-values for the diffusion protocol.
    - qhat: np.ndarray, Gradient directions.

    Returns:
    - S: np.ndarray, Modeled signal intensities.
    """
    S0, eig_val_2, f, theta, phi = x
    eig_val_1 = eig_val_2 / (1 - f)  # Tortuosity condition

    # Replace in Zeppelin-Stick model
    params = [S0, eig_val_2, eig_val_1 - eig_val_2, f, theta, phi]
    return zeppelin_stick_model(params, bvals, qhat)


def zeppelin_stick_ssd(x, voxel, bvals, qhat):
    """
    Compute SSD for Zeppelin-Stick model.

    Parameters:
    - x: np.ndarray, Model parameters [S0, eig_val_2, eig_val_diff, f, theta, phi].
    - voxel: np.ndarray, Observed voxel signal.
    - bvals: np.ndarray, B-values for the diffusion protocol.
    - qhat: np.ndarray, Gradient directions.

    Returns:
    - ssd: float, Sum of squared differences.
    """
    S = zeppelin_stick_model(x, bvals, qhat)
    return np.sum((voxel - S) ** 2)


def zeppelin_stick_tortuosity_ssd(x, voxel, bvals, qhat):
    """
    Compute SSD for Zeppelin-Stick-Tortuosity model.

    Parameters:
    - x: np.ndarray, Model parameters [S0, eig_val_2, f, theta, phi].
    - voxel: np.ndarray, Observed voxel signal.
    - bvals: np.ndarray, B-values for the diffusion protocol.
    - qhat: np.ndarray, Gradient directions.

    Returns:
    - ssd: float, Sum of squared differences.
    """
    S = zeppelin_stick_tortuosity_model(x, bvals, qhat)
    return np.sum((voxel - S) ** 2)


def zeppelin_stick_transform(x):
    """
    Apply constraints for Zeppelin-Stick model parameters.

    Parameters:
    - x: np.ndarray, Original parameters.

    Returns:
    - constrained_params: np.ndarray, Transformed parameters.
    """
    S0 = x[0]**2
    eig_val_2 = x[1]**2
    eig_val_diff = x[2]**2
    f = expit(x[3])  # Sigmoid for range [0, 1]
    theta = x[4]
    phi = x[5]
    return [S0, eig_val_2, eig_val_diff, f, theta, phi]


def zeppelin_stick_transform_inv(x):
    """
    Inverse transform parameters for Zeppelin-Stick model.

    Parameters:
    - x: np.ndarray, Transformed parameters.

    Returns:
    - original_params: np.ndarray, Original parameters.
    """
    S0 = np.sqrt(x[0])
    eig_val_2 = np.sqrt(x[1])
    eig_val_diff = np.sqrt(x[2])
    f = logit(x[3])  # Inverse sigmoid
    theta = x[4]
    phi = x[5]
    return [S0, eig_val_2, eig_val_diff, f, theta, phi]


def fit_zeppelin_stick(voxel, bvals, qhat, start_params):
    """
    Fit the Zeppelin-Stick model to observed voxel signals.

    Parameters:
    - voxel: np.ndarray, Observed voxel signal.
    - bvals: np.ndarray, B-values for the diffusion protocol.
    - qhat: np.ndarray, Gradient directions.
    - start_params: np.ndarray, Initial guess for the parameters.

    Returns:
    - result: scipy.optimize.OptimizeResult, Result of the optimization.
    """
    return minimize(
        fun=zeppelin_stick_ssd,
        x0=start_params,
        args=(voxel, bvals, qhat)
    )


def fit_zeppelin_stick_tortuosity(voxel, bvals, qhat, start_params):
    """
    Fit the Zeppelin-Stick-Tortuosity model to observed voxel signals.

    Parameters:
    - voxel: np.ndarray, Observed voxel signal.
    - bvals: np.ndarray, B-values for the diffusion protocol.
    - qhat: np.ndarray, Gradient directions.
    - start_params: np.ndarray, Initial guess for the parameters.

    Returns:
    - result: scipy.optimize.OptimizeResult, Result of the optimization.
    """
    return minimize(
        fun=zeppelin_stick_tortuosity_ssd,
        x0=start_params,
        args=(voxel, bvals, qhat)
    )
