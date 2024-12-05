import matplotlib.pyplot as plt
import numpy as np


def plot_slice(data, slice_idx, cmap="gray", title="Slice View", vmax=None):
    """
    Plot a 2D slice of 3D or 4D data.

    Parameters:
    - data: np.ndarray, The data array to slice and plot.
    - slice_idx: int, The index of the slice to visualize.
    - cmap: str, Colormap for the plot (default: 'gray').
    - title: str, Title of the plot.
    - vmax: float, Max value for color scale (optional).
    """
    plt.figure()
    plt.imshow(np.flipud(data[:, :, slice_idx].T), cmap=cmap, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    plt.show()


def plot_histogram(data, bins=20, title="Histogram", xlabel="Values", ylabel="Frequency", range=None):
    """
    Plot a histogram for the given data.

    Parameters:
    - data: np.ndarray, Data to plot the histogram.
    - bins: int, Number of bins (default: 20).
    - title: str, Title of the histogram.
    - xlabel: str, Label for the x-axis.
    - ylabel: str, Label for the y-axis.
    - range: tuple, Range of values to include in the histogram (optional).
    """
    plt.figure()
    plt.hist(data, bins=bins, range=range, color="skyblue", edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def plot_parameter_histograms(parameters, titles, bins=20, confidence_intervals=None, sigma_ranges=None):
    """
    Plot histograms for multiple parameters with optional confidence intervals and sigma ranges.

    Parameters:
    - parameters: list of np.ndarray, List of parameter data arrays.
    - titles: list of str, Titles for the histograms.
    - bins: int, Number of bins (default: 20).
    - confidence_intervals: list of tuples, 95% confidence intervals for each parameter (optional).
    - sigma_ranges: list of tuples, Sigma ranges for each parameter (optional).
    """
    num_params = len(parameters)
    fig, axs = plt.subplots(1, num_params, figsize=(6 * num_params, 6))

    for i, (param, title) in enumerate(zip(parameters, titles)):
        axs[i].hist(param, bins=bins, color="skyblue", edgecolor="black", alpha=0.7)
        axs[i].set_title(f"Histogram of {title}")
        axs[i].set_xlabel(f"{title} Values")
        axs[i].set_ylabel("Frequency")
        axs[i].grid(True)

        if confidence_intervals:
            ci_lower, ci_upper = confidence_intervals[i]
            axs[i].axvline(x=ci_lower, color="green", linestyle="--", label="95% CI")
            axs[i].axvline(x=ci_upper, color="green", linestyle="--")

        if sigma_ranges:
            sigma_lower, sigma_upper = sigma_ranges[i]
            axs[i].axvline(x=sigma_lower, color="blue", linestyle="--", label="2-Sigma Range")
            axs[i].axvline(x=sigma_upper, color="blue", linestyle="--")

        axs[i].legend()

    plt.tight_layout()
    plt.show()


def plot_comparison(A_exact, A_est_list, titles, voxel_idx):
    """
    Compare model predictions with observed data for a voxel.

    Parameters:
    - A_exact: np.ndarray, Observed data.
    - A_est_list: list of np.ndarray, List of model predictions.
    - titles: list of str, Titles for the plots (e.g., model names).
    - voxel_idx: int, Index of the voxel being analyzed.
    """
    num_models = len(A_est_list)
    fig, axs = plt.subplots(1, num_models, figsize=(6 * num_models, 6))

    for i, (A_est, title) in enumerate(zip(A_est_list, titles)):
        axs[i].plot(A_exact, "bs", label="Observed")
        axs[i].plot(A_est, "rx", label="Estimated")
        axs[i].set_title(f"{title} (Voxel {voxel_idx})")
        axs[i].set_xlabel("Signal Index")
        axs[i].set_ylabel("Signal Intensity")
        axs[i].legend()

    plt.tight_layout()
    plt.show()


def plot_quiver(n_zplane_x, n_zplane_y, title="Fibre Direction Map", figsize=(8, 8)):
    """
    Plot quiver for fiber direction using theta and phi.

    Parameters:
    - n_zplane_x: np.ndarray, x-components of directions.
    - n_zplane_y: np.ndarray, y-components of directions.
    - title: str, Title of the plot.
    - figsize: tuple, Size of the figure (default: (8, 8)).
    """
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.quiver(n_zplane_x.T, n_zplane_y.T)
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.show()


def plot_model_ranking(AICs, BICs, model_names):
    """
    Plot rankings of models based on AIC and BIC scores.

    Parameters:
    - AICs: list of float, AIC scores for the models.
    - BICs: list of float, BIC scores for the models.
    - model_names: list of str, Names of the models.
    """
    x = np.arange(len(model_names))

    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.35
    ax.bar(x - width / 2, AICs, width, label="AIC")
    ax.bar(x + width / 2, BICs, width, label="BIC")

    ax.set_xlabel("Models")
    ax.set_ylabel("Scores")
    ax.set_title("Model Rankings by AIC and BIC")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    plt.tight_layout()
    plt.show()
