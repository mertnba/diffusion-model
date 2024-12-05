## Overview
This project implements several parametric models to analyze diffusion-weighted MRI data. 
- Ball-Stick Model
- Zeppelin-Stick Model
- Diffusion Tensor Model

## Features
- Parameter estimation using optimization techniques.
- Bootstrapping (classical and parametric) for confidence intervals.
- Evaluation metrics like AIC and BIC for model comparison.
- Visualizations for results and parameter distributions.

## Project Structure
- `data_processing/`: Functions for loading and preprocessing diffusion MRI data.
- `models/`: Model implementations and parameter transformations.
- `optimization/`: Optimization and bootstrapping methods.
- `evaluation/`: Metrics for model evaluation.
- `visualization/`: Plotting and result visualization.
- `tests/`

## Usage
1. Load and preprocess data using `data_processing/`.
2. Train models with scripts in `optimization/`.
3. Evaluate results using metrics in `evaluation/`.
4. Visualize results with functions in `visualization/`.

## Dependencies
- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- Seaborn
- pandas
