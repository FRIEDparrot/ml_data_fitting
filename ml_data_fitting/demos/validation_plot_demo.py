from ml_data_fitting.plot import plot_model_validations

# ─── Demo ────────────────────────────────────────────────────────────────────
def main():
    # NEW STRUCTURE: results_dict[target][model] = metrics
    results_dict = {
        "Max Mises(Mpa)": {
            "CatBoost": {"MAPE_%": 4.80, "CVRMSE_%": 4.10, "MaxAPE_%": 9.2, "Pearson_r": 0.31, "R2": -0.078},
            "GradientBoost": {"MAPE_%": 4.97, "CVRMSE_%": 4.40, "MaxAPE_%": 10.1, "Pearson_r": 0.24, "R2": -0.176},
            "RandomForest": {"MAPE_%": 4.62, "CVRMSE_%": 3.95, "MaxAPE_%": 9.5, "Pearson_r": 0.35, "R2": -0.012},
            "XGBoost": {"MAPE_%": 4.83, "CVRMSE_%": 4.20, "MaxAPE_%": 9.8, "Pearson_r": 0.28, "R2": -0.140},
            "RBF": {"MAPE_%": 4.20, "CVRMSE_%": 3.70, "MaxAPE_%": 8.8, "Pearson_r": 0.42, "R2": 0.108},
            "GaussianProcess": {"MAPE_%": 4.76, "CVRMSE_%": 4.05, "MaxAPE_%": 9.1, "Pearson_r": 0.32, "R2": -0.029},
            "Ridge": {"MAPE_%": 4.46, "CVRMSE_%": 3.85, "MaxAPE_%": 9.0, "Pearson_r": 0.38, "R2": 0.080},
        },
        "Mass(KG)": {
            "CatBoost": {"MAPE_%": 2.44, "CVRMSE_%": 2.10, "MaxAPE_%": 5.1, "Pearson_r": 0.68, "R2": 0.452},
            "GradientBoost": {"MAPE_%": 2.59, "CVRMSE_%": 2.30, "MaxAPE_%": 5.8, "Pearson_r": 0.60, "R2": 0.358},
            "RandomForest": {"MAPE_%": 2.49, "CVRMSE_%": 2.15, "MaxAPE_%": 5.3, "Pearson_r": 0.67, "R2": 0.444},
            "XGBoost": {"MAPE_%": 3.33, "CVRMSE_%": 2.90, "MaxAPE_%": 7.2, "Pearson_r": 0.45, "R2": -0.016},
            "RBF": {"MAPE_%": 3.38, "CVRMSE_%": 2.95, "MaxAPE_%": 7.5, "Pearson_r": 0.40, "R2": -0.073},
            "GaussianProcess": {"MAPE_%": 3.26, "CVRMSE_%": 2.80, "MaxAPE_%": 6.9, "Pearson_r": 0.48, "R2": 0.024},
            "Ridge": {"MAPE_%": 2.51, "CVRMSE_%": 2.20, "MaxAPE_%": 5.4, "Pearson_r": 0.66, "R2": 0.436},
        },
    }

    # Test the main function
    summary = plot_model_validations(
        results_dict,
        output_dir="./test_validation_plots",
        target_cvs={"Max Mises(Mpa)": 0.15, "Mass(KG)": 0.08},  # Low CV → will use Pearson_r
        cv_threshold=0.2,
    )

    print("\nSummary:")
    for target, info in summary.items():
        print(f"\n{target}:")
        print(f"  X-axis: {info['x_axis']}")
        print(f"  Y-axis: {info['y_axis']}")
        print(f"  Pareto models: {info['pareto_models']}")
        print(f"  Top 3: {info['top3_models']}")

    print("\nTest completed! Check ./test_validation_plots/ for output images.")

if __name__ == "__main__":
    main()