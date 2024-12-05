import numpy as np
from metrics import AIC, BIC, log_likelihood
from ball_stick import ball_stick
from zeppelin_stick import zeppelin_stick_model, zeppelin_stick_and_tortuosity_model
from parameter_estimation import estimate_global_minima_probability
from transformations import transform, inverse_transform
from mcmc import MCMC
from bootstrap import classical_bootstrap, ParametricBootstrap


def evaluate_goodness_of_fit(models, model_names, optimized_params, voxel_data, bvals, qhat, noise_std=0.04):
    """
    Evaluate the goodness of fit for each model using AIC, BIC, and log-likelihood.

    Parameters:
    - models: list of callables, Models to evaluate.
    - model_names: list of str, Names of the models.
    - optimized_params: list of np.ndarray, Optimized parameters for each model.
    - voxel_data: np.ndarray, Observed data for a voxel.
    - bvals: np.ndarray, B-values.
    - qhat: np.ndarray, Gradient directions.
    - noise_std: float, Standard deviation of the noise.

    Returns:
    - results: dict, Contains AIC, BIC, and log-likelihood for each model.
    """
    results = {}
    for model, name, params in zip(models, model_names, optimized_params):
        predictions = model(params, bvals, qhat)
        log_likelihood_val = log_likelihood(predictions, voxel_data, noise_std)
        aic = AIC(predictions, voxel_data, len(params), noise_std)
        bic = BIC(predictions, voxel_data, len(params), noise_std)

        results[name] = {
            "log_likelihood": log_likelihood_val,
            "AIC": aic,
            "BIC": bic,
        }
        print(f"Model: {name}")
        print(f"  Log-Likelihood: {log_likelihood_val:.4f}")
        print(f"  AIC: {aic:.4f}")
        print(f"  BIC: {bic:.4f}")
    return results


if __name__ == "__main__":
    # Example voxel data and protocol
    voxel_data = np.random.rand(100)  # Replace with actual voxel data
    bvals = np.random.rand(100)  # Replace with actual bvals
    qhat = np.random.rand(100, 3)  # Replace with actual qhat

    # Define models
    models = [ball_stick, zeppelin_stick_model, zeppelin_stick_and_tortuosity_model]
    model_names = ["Ball Stick", "Zeppelin Stick", "Zeppelin Stick Tortuosity"]

    # Define starting parameters for optimization
    startx_values = [
        np.array([3500, 1.0e-02, 0.45, 1.0, 1.0]),  # Ball Stick
        np.array([1, 1e-2, 1e-3, 0.5, 1, 1]),  # Zeppelin Stick
        np.array([1, 1e-3, 0.5, 1, 1]),  # Zeppelin Stick Tortuosity
    ]
    transforms = [
        (transform, inverse_transform),  # Ball Stick
        (transform, inverse_transform),  # Zeppelin Stick
        (transform, inverse_transform),  # Zeppelin Stick Tortuosity
    ]

    # Estimate optimized parameters for each model
    optimized_params = []
    for model, startx, (transform_fn, inv_transform_fn) in zip(models, startx_values, transforms):
        max_iter = 100
        min_ssd, params, prob = estimate_global_minima_probability(
            max_iter, startx, voxel_data, bvals, qhat, model, transform_fn, inv_transform_fn
        )
        optimized_params.append(params)

    # Evaluate goodness of fit
    results = evaluate_goodness_of_fit(models, model_names, optimized_params, voxel_data, bvals, qhat, noise_std=0.04)

    # Display results
    print("\nGoodness of Fit Results:")
    for name, res in results.items():
        print(f"\nModel: {name}")
        print(f"  Log-Likelihood: {res['log_likelihood']:.4f}")
        print(f"  AIC: {res['AIC']:.4f}")
        print(f"  BIC: {res['BIC']:.4f}")
