import numpy as np
from data_loader import load_dwis_data, load_protocol
from preprocessing import preprocess_data
from ball_stick import ball_stick, BallStickSSD
from zeppelin_stick import zeppelin_stick_model, zeppelin_stick_and_tortuosity_model
from parameter_estimation import estimate_global_minima_probability
from transformations import transform, inverse_transform
from mcmc import MCMC
from bootstrap import classical_bootstrap, ParametricBootstrap
from metrics import AIC, BIC
from confidence_intervals import compute_confidence_intervals
from plotting import plot_signal_comparison, plot_diffusion_maps
from histograms import plot_bootstrap_histograms
from comparison_plots import plot_models_comparison
from minima_estimation import multiple_startx_per_voxel

# Define paths
zip_file_path = ""
protocol_file = ""

# Experiment settings
voxel_idx = (91, 64)  # Example voxel for analysis
slice_idx = 71  # Example slice index
startx_ball_stick = np.array([3500, 1.0e-02, 0.45, 1.0, 1.0])    # Initial guess for Ball-Stick model
startx_zeppelin_stick = np.array([1, 1e-2, 1e-3, 0.5, 1, 1])     # Initial guess for Zeppelin-Stick model
startx_tortuosity = np.array([1, 1e-3, 0.5, 1, 1])               # Initial guess for Tortuosity model

max_iter = 100
burn_in = 2000
interval = 10
sample_length = 1000
noise_std = 0.04

if __name__ == "__main__":
    # Step 1: Data Loading
    print("Loading data...")
    dwis, qhat, bvals = load_dwis_data(zip_file_path)
    print(f"Data shape: {dwis.shape}")

    # Step 2: Preprocessing
    print("Preprocessing data...")
    dwis = preprocess_data(dwis)

    # Step 3: Parameter Estimation (Ball-Stick)
    print("Fitting Ball-Stick model...")
    avox = dwis[:, voxel_idx[0], voxel_idx[1], slice_idx]
    _, params_ball_stick, _ = estimate_global_minima_probability(
        max_iter, startx_ball_stick, avox, bvals, qhat, BallStickSSD, transform, inverse_transform
    )

    # Step 4: Parameter Estimation (Zeppelin-Stick)
    print("Fitting Zeppelin-Stick model...")
    _, params_zeppelin_stick, _ = estimate_global_minima_probability(
        max_iter, startx_zeppelin_stick, avox, bvals, qhat, BallStickSSD, transform, inverse_transform
    )

    # Step 5: Parameter Estimation (Tortuosity)
    print("Fitting Zeppelin-Stick with Tortuosity model...")
    _, params_tortuosity, _ = estimate_global_minima_probability(
        max_iter, startx_tortuosity, avox, bvals, qhat, BallStickSSD, transform, inverse_transform
    )

    # Step 6: Statistical Evaluation
    print("Evaluating models...")
    models = [ball_stick, zeppelin_stick_model, zeppelin_stick_and_tortuosity_model]
    model_names = ["Ball-Stick", "Zeppelin-Stick", "Zeppelin-Stick-Tortuosity"]
    optimized_params = [params_ball_stick, params_zeppelin_stick, params_tortuosity]

    aic_bic_results = {}
    for model, name, params in zip(models, model_names, optimized_params):
        predictions = model(params, bvals, qhat)
        aic = AIC(predictions, avox, len(params), noise_std)
        bic = BIC(predictions, avox, len(params), noise_std)
        aic_bic_results[name] = {"AIC": aic, "BIC": bic}
        print(f"{name}: AIC={aic:.2f}, BIC={bic:.2f}")

    # Step 7: Confidence Intervals
    print("Computing confidence intervals...")
    ci_ball_stick = compute_confidence_intervals(params_ball_stick, avox, bvals, qhat)

    # Step 8: Bootstrap Analysis
    print("Performing bootstrap analysis...")
    classical_results = classical_bootstrap(voxel_idx[0], voxel_idx[1], slice_idx, T=300, N=max_iter, startx=startx_ball_stick)
    parametric_results = ParametricBootstrap(avox, bvals, qhat, max_iter, 300, startx=startx_ball_stick)

    # Step 9: Visualization
    print("Generating plots...")
    plot_signal_comparison(avox, ball_stick(params_ball_stick, bvals, qhat), "Ball-Stick Model Fit")
    plot_bootstrap_histograms(classical_results, ["S0", "Diffusivity", "Fractional Anisotropy"])
    plot_diffusion_maps(dwis, slice_idx)

    # Step 10: Model Comparison Plots
    print("Comparing models...")
    ball_stick_fit = ball_stick(params_ball_stick, bvals, qhat)
    zeppelin_stick_fit = zeppelin_stick_model(params_zeppelin_stick, bvals, qhat)
    plot_models_comparison(avox, "Ball-Stick Fit", "Zeppelin-Stick Fit", ball_stick_fit, None, zeppelin_stick_fit, None)

    print("Experiments completed.")
