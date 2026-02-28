# Machine Learning Data Fitting (Regression) 

### Introduction

This code provide a **quick fitting validation** using multiple types of Machine Learning Python libraries. 

Supported Models : 

```python
SUPPORTED_METHODS = [
    'CatBoost', 'GradientBoost', 'RandomForest',
    'XGBoost', 'RBF', 'GaussianProcess', 'Ridge',
    'LightGBM', 'HistGradientBoost', 'ExtraTrees',
]
```

### Run Evaluation For each models

You can done all evaluation works in 20 lines : 

-  Note we train different model for different columns on `y` 

```python
from ml_data_fitting import plot_eval_results, run_regression_eval, normalize_inputs

x_arr, y_arr = np.array(x), np.array(y)
x_arr = normalize_inputs() # if you need to normalize inputs
x_train, x_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=0.15, random_state=42)

results = run_regression_eval(
    x_train, y_train,
    feature_names=x.columns,
    target_names=y.columns,
    methods=None,
    max_iterations=1000,
    save_path="model_selection.json"
)
```

Then plot the things 

```python
from ml_data_fitting import plot_eval_results
plot_eval_results(results)
```

Example plotting figures are shown as follows :![Max_Mises_Mpa_validation](img\Max_Mises_Mpa_validation.png)
