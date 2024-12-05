import numpy as np


def calculate_confidence_intervals(samples, confidence_level=95):
    """
    Calculate the confidence intervals for a given set of parameter samples.

    Parameters:
    - samples: np.ndarray, Array of parameter samples (e.g., from MCMC or bootstrap).
    - confidence_level: int, Desired confidence level (default: 95%).

    Returns:
    - tuple, (mean, lower_bound, upper_bound) for the confidence interval.
    """
    mean = np.mean(samples)
    lower_bound = np.percentile(samples, (100 - confidence_level) / 2)
    upper_bound = np.percentile(samples, 100 - (100 - confidence_level) / 2)
    return mean, lower_bound, upper_bound


def calculate_sigma_range(samples, sigma=2):
    """
    Calculate the range based on a multiple of standard deviations.

    Parameters:
    - samples: np.ndarray, Array of parameter samples.
    - sigma: int, Number of standard deviations for the range (default: 2).

    Returns:
    - tuple, (mean, lower_bound, upper_bound) for the sigma range.
    """
    mean = np.mean(samples)
    std_dev = np.std(samples)
    lower_bound = mean - sigma * std_dev
    upper_bound = mean + sigma * std_dev
    return mean, lower_bound, upper_bound


def summarize_parameter_statistics(samples, parameter_name):
    """
    Summarize statistics for a single parameter, including mean, confidence intervals, and sigma range.

    Parameters:
    - samples: np.ndarray, Array of parameter samples.
    - parameter_name: str, Name of the parameter being summarized.

    Returns:
    - dict, Summary of statistics.
    """
    mean, ci_lower, ci_upper = calculate_confidence_intervals(samples)
    mean_sigma, sigma_lower, sigma_upper = calculate_sigma_range(samples)

    summary = {
        "Parameter": parameter_name,
        "Mean": round(mean, 5),
        "95% Confidence Interval": [round(ci_lower, 5), round(ci_upper, 5)],
        "2-Sigma Range": [round(sigma_lower, 5), round(sigma_upper, 5)],
    }
    return summary


def generate_summary_table(parameters, param_names):
    """
    Generate a summary table for multiple parameters.

    Parameters:
    - parameters: list of np.ndarray, List of parameter samples.
    - param_names: list of str, Names of the parameters.

    Returns:
    - list of dict, Each dict contains statistics for a parameter.
    """
    summary_table = []
    for i, param_samples in enumerate(parameters):
        summary = summarize_parameter_statistics(param_samples, param_names[i])
        summary_table.append(summary)
    return summary_table
