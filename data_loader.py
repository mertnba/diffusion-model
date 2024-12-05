import numpy as np
from scipy.io import loadmat
import zipfile
import io

def load_diffusion_data(zip_file_path, data_file, bvecs_file):
    """
    Load diffusion MRI data and gradient directions from a zip file.

    Parameters:
    - zip_file_path: str, Path to the zip file.
    - data_file: str, Name of the .mat file containing diffusion data.
    - bvecs_file: str, Name of the file containing gradient directions.

    Returns:
    - dwis: np.ndarray, Diffusion-weighted signal data.
    - qhat: np.ndarray, Gradient directions.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Load the diffusion-weighted signal data from .mat file
        with zip_ref.open(data_file) as mat_file:
            dwis = loadmat(io.BytesIO(mat_file.read()))['dwis']

        # Load the gradient directions
        with zip_ref.open(bvecs_file) as bvecs_file:
            qhat = np.loadtxt(io.TextIOWrapper(bvecs_file), delimiter=" ").T

    # Convert data to double precision and adjust dimensions
    dwis = np.double(dwis).transpose((3, 0, 1, 2))
    return dwis, qhat


def normalize_bvals(qhat, bval_factor=1000):
    """
    Normalize b-values based on gradient directions.

    Parameters:
    - qhat: np.ndarray, Gradient directions.
    - bval_factor: float, Scaling factor for b-values (default is 1000).

    Returns:
    - bvals: np.ndarray, Normalized b-values.
    """
    bvals = bval_factor * np.sum(qhat * qhat, axis=1)
    return bvals


def load_isbi_data(zip_file_path, data_file, protocol_file):
    """
    Load diffusion MRI data and protocol details for the ISBI dataset.

    Parameters:
    - zip_file_path: str, Path to the zip file.
    - data_file: str, Name of the file containing diffusion signal data.
    - protocol_file: str, Name of the file containing the protocol.

    Returns:
    - D: np.ndarray, Diffusion signal data.
    - bvals: np.ndarray, Normalized b-values.
    - qhat: np.ndarray, Gradient directions.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Load the diffusion signal data
        with zip_ref.open(data_file) as fid:
            fid.readline()  # Skip header
            D = np.loadtxt(fid)

        # Load the protocol data
        with zip_ref.open(protocol_file) as fid:
            fid.readline()  # Skip header
            A = np.loadtxt(fid)

    # Extract protocol information
    grad_dirs = A[:, 0:3]
    G = A[:, 3]
    delta = A[:, 4]
    smalldel = A[:, 5]
    GAMMA = 2.675987E8  # Gyromagnetic ratio
    bvals = ((GAMMA * smalldel * G) ** 2) * (delta - smalldel / 3)

    # Convert bvals from s/m^2 to s/mm^2
    bvals = bvals / 1e6
    qhat = grad_dirs

    return D, bvals, qhat
