import numpy as np
import matplotlib.pyplot as plt


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


def calculate_histogram_statistics(data, percentiles=(2.5, 97.5)):
    """
    Calculate histogram statistics including mean, standard deviation,
    95% confidence intervals, and 2-sigma ranges.

    Parameters:
    - data: np.ndarray, Input data array for calculations.
    - percentiles: tuple, Percentile range for confidence intervals (default: 2.5, 97.5).

    Returns:
    - dict, Statistics dictionary containing mean, sigma, confidence intervals, and sigma ranges.
    """
    mean = np.mean(data)
    sigma = np.std(data)
    confidence_interval = np.percentile(data, percentiles)
    sigma_range = [mean - 2 * sigma, mean + 2 * sigma]

    return {
        "mean": mean,
        "sigma": sigma,
        "confidence_interval": confidence_interval,
        "sigma_range": sigma_range,
    }


def summarize_bootstrap_statistics(bootstrap_data, parameter_names):
    """
    Summarize bootstrap statistics for parameters and print results.

    Parameters:
    - bootstrap_data: list of np.ndarray, Bootstrap samples for each parameter.
    - parameter_names: list of str, Names of the parameters.

    Returns:
    - dict, A dictionary containing parameter names and their statistics.
    """
    statistics = {}
    for param, name in zip(bootstrap_data, parameter_names):
        stats = calculate_histogram_statistics(param)
        statistics[name] = stats

        print(f"Parameter: {name}")
        print(f"  Mean: {stats['mean']:.5f}")
        print(f"  95% Confidence Interval: [{stats['confidence_interval'][0]:.5f}, {stats['confidence_interval'][1]:.5f}]")
        print(f"  2-Sigma Range: [{stats['sigma_range'][0]:.5f}, {stats['sigma_range'][1]:.5f}]")
        print()

    return statistics
