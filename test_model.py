import numpy as np
from sklearn.metrics import mean_squared_error

from ball_stick import ball_stick
from zeppelin_stick import zeppelin_stick_model, zeppelin_stick_and_tortuosity_model
from parameter_estimation import estimate_global_minima_probability
from transformations import transform, inverse_transform

def test_model_performance(model, startx, voxel_data, bvals, qhat, transform_fn, inv_transform_fn):
    """
    Test the performance of a given model on a single voxel.

    Parameters:
    - model: callable, The model function to test.
    - startx: np.ndarray, Initial parameter estimates.
    - voxel_data: np.ndarray, Observed diffusion data for a voxel.
    - bvals: np.ndarray, B-values for the experiment.
    - qhat: np.ndarray, Gradient directions for the experiment.
    - transform_fn: callable, Transformation function for parameter constraints.
    - inv_transform_fn: callable, Inverse transformation function.

    Returns:
    - optimized_params: np.ndarray, Optimized parameters for the model.
    - ssd: float, Sum of squared differences between model predictions and observed data.
    """
    max_iter = 100
    min_ssd, optimized_params, prob = estimate_global_minima_probability(
        max_iter, startx, voxel_data, bvals, qhat, model, transform_fn, inv_transform_fn
    )
    predictions = model(optimized_params, bvals, qhat)
    ssd = mean_squared_error(voxel_data, predictions) * len(voxel_data)  # Convert MSE to SSD
    return optimized_params, ssd


def evaluate_all_models(models, model_names, startx_values, voxel_data, bvals, qhat, transforms):
    """
    Evaluate multiple models on a single voxel and compare their performances.

    Parameters:
    - models: list of callables, Model functions to evaluate.
    - model_names: list of str, Names of the models.
    - startx_values: list of np.ndarray, Initial parameter estimates for each model.
    - voxel_data: np.ndarray, Observed diffusion data for a voxel.
    - bvals: np.ndarray, B-values for the experiment.
    - qhat: np.ndarray, Gradient directions for the experiment.
    - transforms: list of tuples, Each tuple contains (transform_fn, inv_transform_fn) for a model.

    Returns:
    - results: dict, Contains optimized parameters and SSDs for each model.
    """
    results = {}
    for model, name, startx, (transform_fn, inv_transform_fn) in zip(models, model_names, startx_values, transforms):
        print(f"Evaluating model: {name}")
        optimized_params, ssd = test_model_performance(
            model, startx, voxel_data, bvals, qhat, transform_fn, inv_transform_fn
        )
        results[name] = {"parameters": optimized_params, "ssd": ssd}
        print(f"Model {name}: SSD = {ssd:.4f}")
    return results


if __name__ == "__main__":
    # Example voxel data and protocol
    voxel_data = np.random.rand(100)  # Replace with actual voxel data
    bvals = np.random.rand(100)  # Replace with actual bvals
    qhat = np.random.rand(100, 3)  # Replace with actual qhat

    # Define models and their transformations
    models = [ball_stick, zeppelin_stick_model, zeppelin_stick_and_tortuosity_model]
    model_names = ["Ball Stick", "Zeppelin Stick", "Zeppelin Stick Tortuosity"]
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

    # Evaluate models
    results = evaluate_all_models(models, model_names, startx_values, voxel_data, bvals, qhat, transforms)

    # Display results
    for name, res in results.items():
        print(f"\nModel: {name}")
        print(f"Optimized Parameters: {res['parameters']}")
        print(f"SSD: {res['ssd']:.4f}")
