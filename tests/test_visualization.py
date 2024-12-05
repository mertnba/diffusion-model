import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_models_comparison
from histograms import plot_histograms
from comparison_plots import plot_bootstrap_comparisons
from transformations import transform
from bootstrap import classical_bootstrap, ParametricBootstrap
from mcmc import MCMC


def visualize_model_comparisons(voxel_data, bvals, qhat, models, model_names, optimized_params):
    """
    Visualize the comparison between models using predictions and observed data.

    Parameters:
    - voxel_data: np.ndarray, Observed data for a voxel.
    - bvals: np.ndarray, B-values.
    - qhat: np.ndarray, Gradient directions.
    - models: list of callables, Models to visualize.
    - model_names: list of str, Names of the models.
    - optimized_params: list of np.ndarray, Optimized parameters for each model.
    """
    predictions = [model(params, bvals, qhat) for model, params in zip(models, optimized_params)]

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            plot_models_comparison(
                voxel_data,
                model_names[i],
                model_names[j],
                predictions[i],
                np.sum((voxel_data - predictions[i]) ** 2),
                predictions[j],
                np.sum((voxel_data - predictions[j]) ** 2),
            )


def visualize_bootstrap_results(voxel_idx, bvals, qhat, voxel_data, startx, max_iter, T=300):
    """
    Visualize the results of bootstrap methods.

    Parameters:
    - voxel_idx: tuple, Index of the voxel to visualize.
    - bvals: np.ndarray, B-values.
    - qhat: np.ndarray, Gradient directions.
    - voxel_data: np.ndarray, Observed data for a voxel.
    - startx: np.ndarray, Initial parameters for optimization.
    - max_iter: int, Maximum iterations for optimization.
    - T: int, Number of bootstrap samples.
    """
    # Classical Bootstrap
    classical_s0, classical_diff, classical_f = classical_bootstrap(
        voxel_idx[0], voxel_idx[1], 71, T, max_iter, startx
    )
    classical_data = [classical_s0, classical_diff, classical_f]
    plot_histograms(classical_data, ["S0", "Diff", "F"], "Classical Bootstrap Results")

    # Parametric Bootstrap
    param_s0, param_diff, param_f = ParametricBootstrap(voxel_data, bvals, qhat, max_iter, T, startx)
    parametric_data = [param_s0, param_diff, param_f]
    plot_histograms(parametric_data, ["S0", "Diff", "F"], "Parametric Bootstrap Results")

    # Comparison of Bootstrap Results
    plot_bootstrap_comparisons(classical_data, parametric_data, ["S0", "Diff", "F"])


def visualize_mcmc_results(voxel_data, startx, burn_in, interval, sample_length, param_std, noise_std):
    """
    Visualize the MCMC sampling results for a given voxel.

    Parameters:
    - voxel_data: np.ndarray, Observed data for a voxel.
    - startx: np.ndarray, Initial parameters for MCMC.
    - burn_in: int, Burn-in period for MCMC.
    - interval: int, Sampling interval for MCMC.
    - sample_length: int, Number of samples to retain after burn-in.
    - param_std: np.ndarray, Standard deviations for parameter proposal.
    - noise_std: float, Noise standard deviation.
    """
    mcmc_samples, acceptance_rate = MCMC(
        voxel_data, startx, burn_in, interval, sample_length, param_std, noise_std
    )
    param_names = ["S0", "Diff", "F", "Theta", "Phi"]
    for i, param_name in enumerate(param_names[:3]):  # Visualizing only first 3 parameters
        plt.figure(figsize=(10, 6))
        plt.hist(mcmc_samples[:, i], bins=20, color="skyblue", edgecolor="black", alpha=0.7)
        plt.title(f"MCMC Sampling for {param_name}")
        plt.xlabel(param_name)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Example data (replace with actual data)
    voxel_data = np.random.rand(100)                     # Replace with actual voxel data
    bvals = np.random.rand(100)                          # Replace with actual bvals
    qhat = np.random.rand(100, 3)                        # Replace with actual qhat
    voxel_idx = (91, 64)                                 # Example voxel indices
    startx = np.array([3500, 1.0e-02, 0.45, 1.0, 1.0])   # Example start parameters
    max_iter = 100
    burn_in = 2000
    interval = 5
    sample_length = 2000
    param_std = np.array([1e1, 1e-6, 1e-3, 1e-2, 1e-2])
    noise_std = 0.04

    # Visualize Model Comparisons
    models = [ball_stick, zeppelin_stick_model, zeppelin_stick_and_tortuosity_model]
    model_names = ["Ball Stick", "Zeppelin Stick", "Zeppelin Stick Tortuosity"]
    optimized_params = [startx, startx, startx]  # Replace with actual optimized params
    visualize_model_comparisons(voxel_data, bvals, qhat, models, model_names, optimized_params)

    # Visualize Bootstrap Results
    visualize_bootstrap_results(voxel_idx, bvals, qhat, voxel_data, startx, max_iter)

    # Visualize MCMC Results
    visualize_mcmc_results(voxel_data, startx, burn_in, interval, sample_length, param_std, noise_std)
