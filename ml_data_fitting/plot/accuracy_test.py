import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict
from scipy.interpolate import make_interp_spline

"""
Plot function for accuracy testing and comparing
"""

def plot_tolerance_accuracy(
        y_true,
        y_pred,
        threshold_range: tuple = (0.01, 0.5),
        threshold_steps: int = 50,
        column_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        colors: Optional[List[str]] = None,
        markers: Optional[List[str]] = None,
        line_styles: Optional[List[str]] = None,
        figsize: tuple = (10, 6),
        dpi: int = 300,
        k = 3,
) -> Dict[str, np.ndarray]:
    """
    Plot accuracy prediction curve showing the relationship between error threshold and prediction accuracy
    Just a simple function,

    :param y_true: True values (n_samples, n_features) or (n_samples,)
    :param y_pred: Predicted values (n_samples, n_methods) for multiple methods or (n_samples,) for single method
    :param threshold_range: Range of error thresholds to evaluate (min, max)
    :param threshold_steps: Number of threshold points to evaluate
    :param column_names: Names of prediction methods
    :param title: Plot title (optional)
    :param save_path: Path to save the figure (optional)
    :param colors: List of colors for each method (optional)
    :param markers: List of markers for each method (optional)
    :param line_styles: List of line styles for each method (optional)
    :param figsize: Figure size (width, height)
    :param dpi: DPI for saved figure
    :param k:
    :return: Dictionary containing threshold arrays and accuracy arrays for each method
    """
    # Handle input dimensions
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)

    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)

    # Ensure y_true matches y_pred samples
    n_samples = y_pred.shape[0]
    m = y_pred.shape[1]  # Number of prediction methods

    # Set default method names
    if column_names is None:
        column_names = [f'Method {i + 1}' for i in range(m)]
    else:
        assert m == len(column_names), "Length of columns must match number of prediction methods"

    # Set default colors, markers, and line styles
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, m))
    if markers is None:
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'] * ((m // 10) + 1)
        markers = markers[:m]
    if line_styles is None:
        line_styles = ['-'] * m

    # Create threshold array
    thresholds = np.linspace(threshold_range[0], threshold_range[1], threshold_steps)

    # Calculate accuracy for each method at different thresholds
    accuracies = {}

    # Set up the plot with essay-quality styling
    plt.figure(figsize=figsize)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.2

    for i, method in enumerate(column_names):
        accuracy_list = []

        for threshold in thresholds:
            # Calculate relative error
            if y_true.shape[1] == 1:
                # If y_true is single column, use it for all methods
                relative_error = np.abs((y_true[:, 0] - y_pred[:, i]) / (y_true[:, 0] + 1e-10))
            else:
                # If y_true has multiple columns, use corresponding column
                col_idx = min(i, y_true.shape[1] - 1)
                relative_error = np.abs((y_true[:, col_idx] - y_pred[:, i]) / (y_true[:, col_idx] + 1e-10))

            # Calculate accuracy as percentage of predictions within threshold
            accuracy = np.mean(relative_error <= threshold) * 100
            accuracy_list.append(accuracy)

        accuracy_array = np.array(accuracy_list)
        accuracies[method] = accuracy_array

        # Interpolate for smooth curve
        if threshold_steps >= 10:
            # Create smooth spline interpolation
            spl = make_interp_spline(thresholds, accuracy_array, k=k)
            thresholds_smooth = np.linspace(threshold_range[0], threshold_range[1], 300)
            accuracy_smooth = spl(thresholds_smooth)

            # Plot smooth curve
            plt.plot(thresholds_smooth * 100, accuracy_smooth,
                    color=colors[i], linestyle=line_styles[i], linewidth=2.0,
                    label=method, alpha=0.85)

            # Add markers at sampled points (every few points for clarity)
            sample_indices = np.linspace(0, len(thresholds) - 1, min(10, threshold_steps), dtype=int)
            plt.scatter(thresholds[sample_indices] * 100, accuracy_array[sample_indices],
                       color=colors[i], marker=markers[i], s=60,
                       edgecolors='white', linewidth=1.2, zorder=5, alpha=0.9)
        else:
            # If too few points, just plot directly
            plt.plot(thresholds * 100, accuracy_array,
                    color=colors[i], marker=markers[i], linestyle=line_styles[i],
                    linewidth=2.0, markersize=6, label=method, alpha=0.85)

    # Styling
    plt.xlabel('Error Threshold (%)', fontsize=13, fontweight='bold')
    plt.ylabel('Prediction Accuracy (%)', fontsize=13, fontweight='bold')

    if title is None:
        title = 'Prediction Accuracy vs. Error Threshold'
    plt.title(title, fontsize=14, fontweight='bold', pad=15)

    plt.legend(loc='best', frameon=True, shadow=True, fancybox=True,
              fontsize=10, framealpha=0.95, edgecolor='gray')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    plt.xlim(threshold_range[0] * 100, threshold_range[1] * 100)
    plt.ylim(0, 105)

    # Add minor ticks for better readability
    ax = plt.gca()
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.minorticks_on()

    plt.tight_layout()

    # Save if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")

    plt.show()

    # Return results
    results = {'thresholds': thresholds}
    for method in column_names:
        results[method] = accuracies[method]

    return results


# Simple test example
def main():
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 100

    # True values
    y_true = np.random.uniform(100, 500, n_samples)

    # Simulated predictions from different methods
    # Method 1: Good predictions (low error)
    y_pred_method1 = y_true + np.random.normal(0, 10, n_samples)
    y_pred_method2 = y_true + np.random.normal(0, 20, n_samples)
    y_pred_method3 = y_true + np.random.normal(0, 35, n_samples)
    y_pred_method4 = y_true + np.random.normal(5, 50, n_samples)
    y_pred = np.column_stack([y_pred_method1, y_pred_method2, y_pred_method3, y_pred_method4])

    # Test the function with custom styling
    results = plot_tolerance_accuracy(
        y_true=y_true,
        y_pred=y_pred,
        threshold_range=(0.01, 0.30),
        threshold_steps=40,
        column_names=['Neural Network', 'Random Forest', 'Gradient Boosting', 'Linear Regression'],
        title='Comparison of ML Model Prediction Accuracy',
        save_path='../utils/accuracy_comparison.png',
        colors=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
        markers=['o', 's', '^', 'D'],
        line_styles=['-', '--', '-.', ':'],
        figsize=(12, 7),
        dpi=300
    )

    # Print some statistics
    print("\n=== Prediction Accuracy Statistics ===")
    print(f"Threshold range: {results['thresholds'][0]*100:.1f}% - {results['thresholds'][-1]*100:.1f}%")
    for method in ['Neural Network', 'Random Forest', 'Gradient Boosting', 'Linear Regression']:
        print(f"\n{method}:")
        print(f"  Accuracy at 5% threshold: {results[method][np.argmin(np.abs(results['thresholds'] - 0.05))]:.2f}%")
        print(f"  Accuracy at 10% threshold: {results[method][np.argmin(np.abs(results['thresholds'] - 0.10))]:.2f}%")
        print(f"  Accuracy at 20% threshold: {results[method][np.argmin(np.abs(results['thresholds'] - 0.20))]:.2f}%")

if __name__ == "__main__":
    main()