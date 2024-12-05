import matplotlib.pyplot as plt
import numpy as np

def plot_model_comparisons(voxel_data, models, model_names, ssds, title="Model Comparison"):
    """
    Compare observed data with multiple model predictions for a given voxel.

    Parameters:
    - voxel_data: np.ndarray, Observed data for the voxel.
    - models: list of np.ndarray, Predicted data from different models.
    - model_names: list of str, Names of the models for labeling.
    - ssds: list of float, Sum of squared differences for each model.
    - title: str, Title of the plot.
    """
    if len(models) != len(model_names) or len(models) != len(ssds):
        raise ValueError("Length of models, model_names, and ssds must be the same.")
    
    num_models = len(models)
    fig, axs = plt.subplots(1, num_models, figsize=(6 * num_models, 6))
    fig.suptitle(title)

    for i, (model, name, ssd) in enumerate(zip(models, model_names, ssds)):
        axs[i].plot(voxel_data, 'bs', label='Observed')
        axs[i].plot(model, 'rx', label='Predicted')
        axs[i].set_title(f"{name}\nSSD: {ssd:.2f}")
        axs[i].legend()

    plt.tight_layout()
    plt.show()


def compare_ssd_heatmaps(ssd_maps, model_names, cmap="viridis", title="SSD Heatmap Comparison"):
    """
    Visualize and compare SSD heatmaps for different models.

    Parameters:
    - ssd_maps: list of np.ndarray, SSD heatmaps for each model.
    - model_names: list of str, Names of the models for labeling.
    - cmap: str, Colormap for the heatmaps.
    - title: str, Title of the overall comparison plot.
    """
    num_models = len(ssd_maps)
    fig, axs = plt.subplots(1, num_models, figsize=(6 * num_models, 6))
    fig.suptitle(title)

    for i, (ssd_map, name) in enumerate(zip(ssd_maps, model_names)):
        im = axs[i].imshow(ssd_map, cmap=cmap)
        axs[i].set_title(f"{name} SSD Map")
        plt.colorbar(im, ax=axs[i])

    plt.tight_layout()
    plt.show()


def compare_aic_bic(aic_values, bic_values, model_names, title="AIC and BIC Comparisons"):
    """
    Plot a comparison of AIC and BIC values for multiple models.

    Parameters:
    - aic_values: list of float, AIC values for each model.
    - bic_values: list of float, BIC values for each model.
    - model_names: list of str, Names of the models.
    - title: str, Title of the plot.
    """
    x = np.arange(len(model_names))  # Indices for the models

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.2, aic_values, 0.4, label='AIC')
    ax.bar(x + 0.2, bic_values, 0.4, label='BIC')

    ax.set_xlabel("Models")
    ax.set_ylabel("Scores")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_confidence_comparison(parameters, parameter_names, model_names, confidence_intervals):
    """
    Plot confidence intervals for different parameters across models.

    Parameters:
    - parameters: list of np.ndarray, Parameter values for each model.
    - parameter_names: list of str, Names of the parameters.
    - model_names: list of str, Names of the models.
    - confidence_intervals: list of tuples, Confidence intervals for each parameter (per model).
    """
    num_params = len(parameters)
    fig, axs = plt.subplots(1, num_params, figsize=(6 * num_params, 6))
    fig.suptitle("Confidence Interval Comparison Across Models")

    for i, param_name in enumerate(parameter_names):
        for j, model_name in enumerate(model_names):
            lower, upper = confidence_intervals[j][i]
            axs[i].errorbar(j, parameters[j][i], yerr=[[parameters[j][i] - lower], [upper - parameters[j][i]]],
                            fmt='o', label=model_name, capsize=5)
        axs[i].set_title(f"{param_name} Confidence Intervals")
        axs[i].set_xticks(range(len(model_names)))
        axs[i].set_xticklabels(model_names)
        axs[i].legend()

    plt.tight_layout()
    plt.show()
